#!/usr/bin/env python3
"""Standalone lazy PyTorch/Triton gather and paired K/V scatter kernels.

All GPU-side operations are exact copies:

* generic gather: full source ``[N, D]`` -> preallocated compact ``[K, D]``
  using exactly the selected rows.
* paired scatter: compact ``K`` and ``V`` ``[K, D]`` -> preallocated full
  caches ``[N, D]`` using in-place indexed stores.  Inactive cache rows are
  never touched.

Variants share the same selected order, source/destination memory layouts and
preallocated output tensors.  Row/column tiling only; no per-token Python loop
is used in the benchmark path.

Importing this module does not import torch or triton, so CPU tests can import
every analysis module without CUDA.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from synthetic_traces import (
    cpu_reference_gather,
    cpu_reference_scatter_kv,
    TraceError,
)

CHUNK_G = 16
COL_TILE = 128
GENERIC_ROW_TILES = (1, 4, 16, 32)
METHODS = (
    "torch_index",
    "triton_row1",
    "triton_row4",
    "triton_row16",
    "triton_row32",
    "triton_block",
    "triton_hybrid",
)
DESCRIPTOR_METHODS = frozenset({"triton_block", "triton_hybrid"})
ROW_TILE_BY_METHOD = {
    "triton_row1": 1,
    "triton_row4": 4,
    "triton_row16": 16,
    "triton_row32": 32,
    "triton_block": 16,
    "triton_hybrid": 8,
}

_KERNEL_CACHE: dict[str, Any] | None = None


class KernelError(RuntimeError):
    """Raised for invalid tensor layouts or unsupported method names."""


def _torch():
    import torch  # lazy

    return torch


def _triton():
    import triton  # lazy

    return triton


def _kernels() -> dict[str, Any]:
    """Define (once) and return the Triton JIT kernels."""
    global _KERNEL_CACHE
    if _KERNEL_CACHE is not None:
        return _KERNEL_CACHE
    import triton
    import triton.language as tl

    @triton.jit
    def _gather_indexed(
        src_ptr,
        idx_ptr,
        out_ptr,
        k,
        D: tl.constexpr,
        BR: tl.constexpr,
        BD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = (pid * BR + tl.arange(0, BR)).to(tl.int64)
        valid = rows < k
        ids = tl.load(idx_ptr + rows, mask=valid, other=0)
        for d0 in range(0, D, BD):
            cols = d0 + tl.arange(0, BD)
            mask = valid[:, None] & (cols[None, :] < D)
            values = tl.load(
                src_ptr + ids[:, None] * D + cols[None, :], mask=mask, other=0.0
            )
            tl.store(
                out_ptr + rows[:, None] * D + cols[None, :], values, mask=mask
            )

    @triton.jit
    def _scatter_indexed(
        k_src_ptr,
        v_src_ptr,
        k_cache_ptr,
        v_cache_ptr,
        idx_ptr,
        k,
        D: tl.constexpr,
        BR: tl.constexpr,
        BD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = (pid * BR + tl.arange(0, BR)).to(tl.int64)
        valid = rows < k
        ids = tl.load(idx_ptr + rows, mask=valid, other=0)
        for d0 in range(0, D, BD):
            cols = d0 + tl.arange(0, BD)
            mask = valid[:, None] & (cols[None, :] < D)
            k_values = tl.load(
                k_src_ptr + rows[:, None] * D + cols[None, :], mask=mask, other=0.0
            )
            v_values = tl.load(
                v_src_ptr + rows[:, None] * D + cols[None, :], mask=mask, other=0.0
            )
            tl.store(
                k_cache_ptr + ids[:, None] * D + cols[None, :],
                k_values,
                mask=mask,
            )
            tl.store(
                v_cache_ptr + ids[:, None] * D + cols[None, :],
                v_values,
                mask=mask,
            )

    @triton.jit
    def _gather_block(
        src_ptr,
        idx_ptr,
        base_ptr,
        length_ptr,
        contig_ptr,
        out_ptr,
        k,
        D: tl.constexpr,
        G: tl.constexpr,
        BR: tl.constexpr,
        BD: tl.constexpr,
        TPC: tl.constexpr,
    ):
        pid = tl.program_id(0)
        chunk = pid // TPC
        tile = pid % TPC
        start = chunk * G
        length = tl.load(length_ptr + chunk)
        r = tile * BR + tl.arange(0, BR)
        valid = (r < length) & ((start + r) < k)
        base = tl.load(base_ptr + chunk)
        contiguous = tl.load(contig_ptr + chunk)
        # Only non-contiguous chunks pay for indirect index loads.  Contiguous
        # chunks use base+offset and never read the per-row index array.
        ids = tl.load(
            idx_ptr + start + r, mask=valid & (contiguous == 0), other=0
        )
        row_ids = tl.where(contiguous != 0, base + r, ids).to(tl.int64)
        out_rows = (start + r).to(tl.int64)
        for d0 in range(0, D, BD):
            cols = d0 + tl.arange(0, BD)
            mask = valid[:, None] & (cols[None, :] < D)
            values = tl.load(
                src_ptr + row_ids[:, None] * D + cols[None, :],
                mask=mask,
                other=0.0,
            )
            tl.store(
                out_ptr + out_rows[:, None] * D + cols[None, :],
                values,
                mask=mask,
            )

    @triton.jit
    def _scatter_block(
        k_src_ptr,
        v_src_ptr,
        k_cache_ptr,
        v_cache_ptr,
        idx_ptr,
        base_ptr,
        length_ptr,
        contig_ptr,
        k,
        D: tl.constexpr,
        G: tl.constexpr,
        BR: tl.constexpr,
        BD: tl.constexpr,
        TPC: tl.constexpr,
    ):
        pid = tl.program_id(0)
        chunk = pid // TPC
        tile = pid % TPC
        start = chunk * G
        length = tl.load(length_ptr + chunk)
        r = tile * BR + tl.arange(0, BR)
        valid = (r < length) & ((start + r) < k)
        base = tl.load(base_ptr + chunk)
        contiguous = tl.load(contig_ptr + chunk)
        # Only non-contiguous chunks pay for indirect index loads.
        ids = tl.load(
            idx_ptr + start + r, mask=valid & (contiguous == 0), other=0
        )
        row_ids = tl.where(contiguous != 0, base + r, ids).to(tl.int64)
        src_rows = (start + r).to(tl.int64)
        for d0 in range(0, D, BD):
            cols = d0 + tl.arange(0, BD)
            mask = valid[:, None] & (cols[None, :] < D)
            k_values = tl.load(
                k_src_ptr + src_rows[:, None] * D + cols[None, :],
                mask=mask,
                other=0.0,
            )
            v_values = tl.load(
                v_src_ptr + src_rows[:, None] * D + cols[None, :],
                mask=mask,
                other=0.0,
            )
            tl.store(
                k_cache_ptr + row_ids[:, None] * D + cols[None, :],
                k_values,
                mask=mask,
            )
            tl.store(
                v_cache_ptr + row_ids[:, None] * D + cols[None, :],
                v_values,
                mask=mask,
            )

    @triton.jit
    def _gather_hybrid(
        src_ptr,
        idx_ptr,
        base_ptr,
        contig_ptr,
        out_ptr,
        k,
        D: tl.constexpr,
        G: tl.constexpr,
        BR: tl.constexpr,
        BD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = (pid * BR + tl.arange(0, BR)).to(tl.int64)
        valid = rows < k
        chunk = rows // G
        contiguous = tl.load(contig_ptr + chunk, mask=valid, other=0)
        base = tl.load(base_ptr + chunk, mask=valid, other=0)
        # Rows in contiguous chunks use base+offset and skip index reads.
        ids = tl.load(
            idx_ptr + rows, mask=valid & (contiguous == 0), other=0
        )
        start = chunk * G
        row_ids = tl.where(contiguous != 0, base + (rows - start), ids).to(tl.int64)
        for d0 in range(0, D, BD):
            cols = d0 + tl.arange(0, BD)
            mask = valid[:, None] & (cols[None, :] < D)
            values = tl.load(
                src_ptr + row_ids[:, None] * D + cols[None, :],
                mask=mask,
                other=0.0,
            )
            tl.store(
                out_ptr + rows[:, None] * D + cols[None, :], values, mask=mask
            )

    @triton.jit
    def _scatter_hybrid(
        k_src_ptr,
        v_src_ptr,
        k_cache_ptr,
        v_cache_ptr,
        idx_ptr,
        base_ptr,
        contig_ptr,
        k,
        D: tl.constexpr,
        G: tl.constexpr,
        BR: tl.constexpr,
        BD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = (pid * BR + tl.arange(0, BR)).to(tl.int64)
        valid = rows < k
        chunk = rows // G
        contiguous = tl.load(contig_ptr + chunk, mask=valid, other=0)
        base = tl.load(base_ptr + chunk, mask=valid, other=0)
        # Rows in contiguous chunks use base+offset and skip index reads.
        ids = tl.load(
            idx_ptr + rows, mask=valid & (contiguous == 0), other=0
        )
        start = chunk * G
        row_ids = tl.where(contiguous != 0, base + (rows - start), ids).to(tl.int64)
        for d0 in range(0, D, BD):
            cols = d0 + tl.arange(0, BD)
            mask = valid[:, None] & (cols[None, :] < D)
            k_values = tl.load(
                k_src_ptr + rows[:, None] * D + cols[None, :], mask=mask, other=0.0
            )
            v_values = tl.load(
                v_src_ptr + rows[:, None] * D + cols[None, :], mask=mask, other=0.0
            )
            tl.store(
                k_cache_ptr + row_ids[:, None] * D + cols[None, :],
                k_values,
                mask=mask,
            )
            tl.store(
                v_cache_ptr + row_ids[:, None] * D + cols[None, :],
                v_values,
                mask=mask,
            )

    @triton.jit
    def _fill_descriptor(
        idx_ptr,
        chunk_start_ptr,
        base_ptr,
        length_ptr,
        contig_ptr,
        k,
        n_chunks,
        G: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        chunk = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        mask = chunk < n_chunks
        start = chunk * G
        length = tl.minimum(G, k - start)
        base = tl.load(idx_ptr + start, mask=mask, other=0)
        last_index = tl.load(
            idx_ptr + start + length - 1, mask=mask, other=0
        )
        contiguous = (last_index - base) == (length - 1)
        tl.store(chunk_start_ptr + chunk, start, mask=mask)
        tl.store(base_ptr + chunk, base, mask=mask)
        tl.store(length_ptr + chunk, length, mask=mask)
        tl.store(contig_ptr + chunk, contiguous.to(tl.uint8), mask=mask)

    _KERNEL_CACHE = {
        "gather_indexed": _gather_indexed,
        "scatter_indexed": _scatter_indexed,
        "gather_block": _gather_block,
        "scatter_block": _scatter_block,
        "gather_hybrid": _gather_hybrid,
        "scatter_hybrid": _scatter_hybrid,
        "fill_descriptor": _fill_descriptor,
    }
    return _KERNEL_CACHE


# ---------------------------------------------------------------------------
# Host-side validation and wrappers
# ---------------------------------------------------------------------------


def _check_matrix(name: str, tensor: Any) -> Any:
    torch = _torch()
    if not torch.is_tensor(tensor):
        raise KernelError(f"{name} must be a torch tensor")
    if tensor.dim() != 2:
        raise KernelError(f"{name} must be 2-D, got shape {tuple(tensor.shape)}")
    if not tensor.is_cuda:
        raise KernelError(f"{name} must be on CUDA")
    if not tensor.is_contiguous():
        raise KernelError(f"{name} must be contiguous")
    return tensor


def _check_index(torch: Any, idx: Any) -> None:
    if not torch.is_tensor(idx):
        raise KernelError("indices must be a torch tensor")
    if idx.dim() != 1:
        raise KernelError("indices must be 1-D")
    if idx.dtype != torch.int64:
        raise KernelError("indices must be int64")
    if not idx.is_cuda:
        raise KernelError("indices must be on CUDA")
    if not idx.is_contiguous():
        raise KernelError("indices must be contiguous")


def validate_trace_indices_device(idx: Any, n_rows: int | None = None) -> None:
    """One-time (non-timed) validation of device indices.

    This performs host readbacks and is therefore kept out of the timed kernel
    path.  Block-descriptor correctness relies on sorted, distinct indices.
    """
    torch = _torch()
    _check_index(torch, idx)
    k = int(idx.numel())
    if k == 0:
        return
    if int(idx.min().item()) < 0:
        raise KernelError("indices must be non-negative")
    if n_rows is not None and int(idx.max().item()) >= int(n_rows):
        raise KernelError("indices out of bounds")
    if k > 1:
        if not bool(torch.all(idx[1:] > idx[:-1]).item()):
            raise KernelError("indices must be sorted and strictly increasing")


def _check_layout(
    src: Any | None,
    idx: Any,
    out: Any | None,
    k_src: Any | None = None,
    v_src: Any | None = None,
    k_cache: Any | None = None,
    v_cache: Any | None = None,
) -> int:
    torch = _torch()
    _check_index(torch, idx)
    k = int(idx.numel())
    if src is not None:
        _check_matrix("src", src)
        if src.shape[0] <= 0:
            raise KernelError("src must have rows")
        d = int(src.shape[1])
        if out is not None:
            _check_matrix("out", out)
            if tuple(out.shape) != (k, d):
                raise KernelError(
                    f"out shape {tuple(out.shape)} != expected {(k, d)}"
                )
            if out.dtype != src.dtype:
                raise KernelError("out dtype must match src dtype")
    else:
        _check_matrix("k_src", k_src)
        _check_matrix("v_src", v_src)
        _check_matrix("k_cache", k_cache)
        _check_matrix("v_cache", v_cache)
        d = int(k_src.shape[1])
        if tuple(k_src.shape) != (k, d) or tuple(v_src.shape) != (k, d):
            raise KernelError("compact K/V shapes must both be (K, D)")
        if tuple(k_cache.shape) != tuple(v_cache.shape):
            raise KernelError("K/V cache shapes must match")
        if int(k_cache.shape[1]) != d:
            raise KernelError("cache D differs from compact D")
        if any(
            tensor.dtype != k_src.dtype
            for tensor in (v_src, k_cache, v_cache)
        ):
            raise KernelError("K/V dtypes must match")
    if d <= 0:
        raise KernelError("D must be positive")
    return d


def _check_generic_row_tile(row_tile: int) -> int:
    tile = int(row_tile)
    if tile not in GENERIC_ROW_TILES:
        raise KernelError(
            f"generic row tile must be one of {GENERIC_ROW_TILES}, got {row_tile!r}"
        )
    return tile


def gather_indexed(src: Any, idx: Any, out: Any, row_tile: int = 16) -> Any:
    """Generic indexed Triton gather into preallocated ``out`` [K, D]."""
    triton = _triton()
    row_tile = _check_generic_row_tile(row_tile)
    d = _check_layout(src, idx, out)
    k = int(idx.numel())
    if k == 0:
        return out
    kernels = _kernels()
    grid = (triton.cdiv(k, int(row_tile)),)
    kernels["gather_indexed"][grid](
        src,
        idx,
        out,
        k,
        D=d,
        BR=int(row_tile),
        BD=COL_TILE,
        num_warps=4,
    )
    return out


def scatter_kv_indexed(
    k_src: Any,
    v_src: Any,
    k_cache: Any,
    v_cache: Any,
    idx: Any,
    row_tile: int = 16,
) -> None:
    """Paired in-place K/V indexed scatter; inactive rows are preserved."""
    triton = _triton()
    row_tile = _check_generic_row_tile(row_tile)
    _check_layout(
        None, idx, None, k_src=k_src, v_src=v_src, k_cache=k_cache, v_cache=v_cache
    )
    k = int(idx.numel())
    if k == 0:
        return
    kernels = _kernels()
    d = int(k_src.shape[1])
    grid = (triton.cdiv(k, int(row_tile)),)
    kernels["scatter_indexed"][grid](
        k_src,
        v_src,
        k_cache,
        v_cache,
        idx,
        k,
        D=d,
        BR=int(row_tile),
        BD=COL_TILE,
        num_warps=4,
    )


def _descriptor_tensors(desc: Mapping[str, Any], torch: Any) -> tuple[Any, Any, Any]:
    required = ("base", "length", "contiguous")
    for key in required:
        if key not in desc:
            raise KernelError(f"descriptor missing {key}")
        if not torch.is_tensor(desc[key]) or not desc[key].is_cuda:
            raise KernelError(f"descriptor.{key} must be a CUDA tensor (build on GPU)")
    return desc["base"], desc["length"], desc["contiguous"]


def gather_block(
    src: Any,
    idx: Any,
    desc: Mapping[str, Any],
    out: Any,
    row_tile: int = 16,
) -> Any:
    """Descriptor block gather: base+offset for contiguous chunks otherwise indices."""
    triton = _triton()
    d = _check_layout(src, idx, out)
    k = int(idx.numel())
    if k == 0:
        return out
    base, length, contiguous = _descriptor_tensors(desc, _torch())
    chunk = int(desc["chunk_size"])
    row_tile = int(row_tile)
    tiles_per_chunk = triton.cdiv(chunk, row_tile)
    n_chunks = int(desc["n_chunks"])
    kernels = _kernels()
    kernels["gather_block"][(n_chunks * tiles_per_chunk,)](
        src,
        idx,
        base,
        length,
        contiguous,
        out,
        k,
        D=d,
        G=chunk,
        BR=row_tile,
        BD=COL_TILE,
        TPC=tiles_per_chunk,
        num_warps=4,
    )
    return out


def scatter_kv_block(
    k_src: Any,
    v_src: Any,
    k_cache: Any,
    v_cache: Any,
    idx: Any,
    desc: Mapping[str, Any],
    row_tile: int = 16,
) -> None:
    triton = _triton()
    _check_layout(
        None, idx, None, k_src=k_src, v_src=v_src, k_cache=k_cache, v_cache=v_cache
    )
    k = int(idx.numel())
    if k == 0:
        return
    base, length, contiguous = _descriptor_tensors(desc, _torch())
    chunk = int(desc["chunk_size"])
    row_tile = int(row_tile)
    tiles_per_chunk = triton.cdiv(chunk, row_tile)
    n_chunks = int(desc["n_chunks"])
    kernels = _kernels()
    kernels["scatter_block"][(n_chunks * tiles_per_chunk,)](
        k_src,
        v_src,
        k_cache,
        v_cache,
        idx,
        base,
        length,
        contiguous,
        k,
        D=int(k_src.shape[1]),
        G=chunk,
        BR=row_tile,
        BD=COL_TILE,
        TPC=tiles_per_chunk,
        num_warps=4,
    )


def gather_hybrid(
    src: Any,
    idx: Any,
    desc: Mapping[str, Any],
    out: Any,
    row_tile: int = 8,
) -> Any:
    """Descriptor hybrid gather: per-row contiguous base+offset / exact index."""
    triton = _triton()
    d = _check_layout(src, idx, out)
    k = int(idx.numel())
    if k == 0:
        return out
    base, _length, contiguous = _descriptor_tensors(desc, _torch())
    chunk = int(desc["chunk_size"])
    kernels = _kernels()
    kernels["gather_hybrid"][(triton.cdiv(k, int(row_tile)),)](
        src,
        idx,
        base,
        contiguous,
        out,
        k,
        D=d,
        G=chunk,
        BR=int(row_tile),
        BD=COL_TILE,
        num_warps=4,
    )
    return out


def scatter_kv_hybrid(
    k_src: Any,
    v_src: Any,
    k_cache: Any,
    v_cache: Any,
    idx: Any,
    desc: Mapping[str, Any],
    row_tile: int = 8,
) -> None:
    triton = _triton()
    _check_layout(
        None, idx, None, k_src=k_src, v_src=v_src, k_cache=k_cache, v_cache=v_cache
    )
    k = int(idx.numel())
    if k == 0:
        return
    base, _length, contiguous = _descriptor_tensors(desc, _torch())
    chunk = int(desc["chunk_size"])
    kernels = _kernels()
    kernels["scatter_hybrid"][(triton.cdiv(k, int(row_tile)),)](
        k_src,
        v_src,
        k_cache,
        v_cache,
        idx,
        base,
        contiguous,
        k,
        D=int(k_src.shape[1]),
        G=chunk,
        BR=int(row_tile),
        BD=COL_TILE,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# GPU descriptor construction (no CPU dynamic-size readback)
# ---------------------------------------------------------------------------


def allocate_descriptor(
    k: int, chunk: int = CHUNK_G, device: Any | None = None
) -> dict[str, Any]:
    """Preallocate descriptor tensors with known capacity ``ceil(K/chunk)``.

    ``K`` is known host-side; no data-dependent host readback is used.  The
    tensors are intended to be filled by :func:`fill_descriptor_gpu` and reused.
    """
    torch = _torch()
    if not isinstance(chunk, int) or chunk <= 0:
        raise KernelError("chunk must be a positive integer")
    k = int(k)
    if k < 0:
        raise KernelError("k must be non-negative")
    if device is None:
        device = torch.device("cuda", 0)
    n_chunks = (k + chunk - 1) // chunk if k else 0
    return {
        "chunk_start": torch.empty(n_chunks, dtype=torch.int64, device=device),
        "base": torch.empty(n_chunks, dtype=torch.int64, device=device),
        "length": torch.empty(n_chunks, dtype=torch.int64, device=device),
        "contiguous": torch.empty(n_chunks, dtype=torch.uint8, device=device),
        "chunk_size": chunk,
        "n_chunks": int(n_chunks),
        "k": k,
    }


def fill_descriptor_gpu(
    idx: Any, descriptor: Mapping[str, Any], block: int = 256
) -> dict[str, Any]:
    """Fused Triton fill of a preallocated descriptor (no allocations).

    Layout matches :func:`synthetic_traces.build_compact_chunks`: consecutive
    compact chunks of ``chunk_size``, ``base`` = first selected position,
    ``length`` = rows in the chunk (including a short tail), and ``contiguous``
    = ``last - base == length - 1``.
    """
    triton = _triton()
    torch = _torch()
    _check_index(torch, idx)
    k = int(idx.numel())
    if int(descriptor["k"]) != k:
        raise KernelError("descriptor capacity does not match K")
    for key in ("chunk_start", "base", "length", "contiguous"):
        if not torch.is_tensor(descriptor[key]) or not descriptor[key].is_cuda:
            raise KernelError(f"descriptor.{key} must be a CUDA tensor")
    n_chunks = int(descriptor["n_chunks"])
    if n_chunks == 0:
        return descriptor
    chunk = int(descriptor["chunk_size"])
    kernels = _kernels()
    grid = (triton.cdiv(n_chunks, int(block)),)
    kernels["fill_descriptor"][grid](
        idx,
        descriptor["chunk_start"],
        descriptor["base"],
        descriptor["length"],
        descriptor["contiguous"],
        k,
        n_chunks,
        G=chunk,
        BLOCK=int(block),
        num_warps=4,
    )
    return descriptor


def build_descriptor_gpu(idx: Any, chunk: int = CHUNK_G) -> dict[str, Any]:
    """Allocate then fused-fill a descriptor (convenience API)."""
    torch = _torch()
    _check_index(torch, idx)
    descriptor = allocate_descriptor(int(idx.numel()), chunk, idx.device)
    if descriptor["n_chunks"]:
        fill_descriptor_gpu(idx, descriptor)
    return descriptor


def descriptor_to_cpu(desc: Mapping[str, Any]) -> dict[str, Any]:
    """Copy descriptor tensors to CPU for smoke verification only."""
    import numpy as _np

    return {
        "chunk_start": desc["chunk_start"].detach().cpu().numpy(),
        "base": desc["base"].detach().cpu().numpy(),
        "length": desc["length"].detach().cpu().numpy(),
        "contiguous": desc["contiguous"].detach().cpu().numpy().astype(bool),
        "chunk_size": int(desc["chunk_size"]),
        "n_chunks": int(desc["n_chunks"]),
        "k": int(desc["k"]),
    }


# ---------------------------------------------------------------------------
# Strong Torch reference and method dispatch
# ---------------------------------------------------------------------------


def torch_gather_out(src: Any, idx: Any, out: Any) -> Any:
    """``torch.index_select(..., out=preallocated)`` reference gather."""
    torch = _torch()
    _check_layout(src, idx, out)
    return torch.index_select(src, 0, idx, out=out)


def torch_scatter_kv(
    k_src: Any, v_src: Any, k_cache: Any, v_cache: Any, idx: Any
) -> None:
    """Two in-place ``index_copy_`` writes into preallocated caches."""
    _check_layout(
        None, idx, None, k_src=k_src, v_src=v_src, k_cache=k_cache, v_cache=v_cache
    )
    k_cache.index_copy_(0, idx, k_src)
    v_cache.index_copy_(0, idx, v_src)


def gather_for_method(method: str, tensors: Mapping[str, Any]) -> Any:
    src = tensors["src"]
    idx = tensors["idx"]
    out = tensors["compact"]
    if method == "torch_index":
        return torch_gather_out(src, idx, out)
    if method.startswith("triton_row"):
        row_tile = int(method.replace("triton_row", ""))
        if row_tile not in GENERIC_ROW_TILES:
            raise KernelError(f"unknown generic row tile in {method}")
        return gather_indexed(src, idx, out, row_tile=row_tile)
    if method == "triton_block":
        return gather_block(src, idx, tensors["descriptor"], out, row_tile=16)
    if method == "triton_hybrid":
        return gather_hybrid(src, idx, tensors["descriptor"], out, row_tile=8)
    raise KernelError(f"unknown method {method!r}")


def scatter_for_method(method: str, tensors: Mapping[str, Any]) -> None:
    k_src = tensors["k_src"]
    v_src = tensors["v_src"]
    k_cache = tensors["k_cache"]
    v_cache = tensors["v_cache"]
    idx = tensors["idx"]
    if method == "torch_index":
        torch_scatter_kv(k_src, v_src, k_cache, v_cache, idx)
        return
    if method.startswith("triton_row"):
        row_tile = int(method.replace("triton_row", ""))
        if row_tile not in GENERIC_ROW_TILES:
            raise KernelError(f"unknown generic row tile in {method}")
        scatter_kv_indexed(k_src, v_src, k_cache, v_cache, idx, row_tile=row_tile)
        return
    if method == "triton_block":
        scatter_kv_block(
            k_src, v_src, k_cache, v_cache, idx, tensors["descriptor"], row_tile=16
        )
        return
    if method == "triton_hybrid":
        scatter_kv_hybrid(
            k_src, v_src, k_cache, v_cache, idx, tensors["descriptor"], row_tile=8
        )
        return
    raise KernelError(f"unknown method {method!r}")


def run_prepared(method: str, tensors: Mapping[str, Any]) -> None:
    """One gather plus one paired K/V scatter (the prepared benchmark step)."""
    if method not in METHODS:
        raise KernelError(f"unknown method {method!r}")
    gather_for_method(method, tensors)
    scatter_for_method(method, tensors)


def method_needs_descriptor(method: str) -> bool:
    return method in DESCRIPTOR_METHODS


def method_row_tile(method: str) -> int:
    return int(ROW_TILE_BY_METHOD.get(method, 16))


__all__ = [
    "CHUNK_G",
    "COL_TILE",
    "DESCRIPTOR_METHODS",
    "GENERIC_ROW_TILES",
    "METHODS",
    "KernelError",
    "allocate_descriptor",
    "build_descriptor_gpu",
    "fill_descriptor_gpu",
    "cpu_reference_gather",
    "cpu_reference_scatter_kv",
    "descriptor_to_cpu",
    "gather_block",
    "gather_for_method",
    "gather_hybrid",
    "gather_indexed",
    "method_needs_descriptor",
    "method_row_tile",
    "run_prepared",
    "scatter_for_method",
    "scatter_kv_block",
    "scatter_kv_hybrid",
    "scatter_kv_indexed",
    "torch_gather_out",
    "torch_scatter_kv",
]
