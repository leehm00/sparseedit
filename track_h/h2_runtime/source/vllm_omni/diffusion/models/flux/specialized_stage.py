# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803
"""Fixed-capacity sparse entry packing and CFG/velocity/age submission.

The selector supplies unique global indices. Packing refreshes a row map and
stable input buffers; submission reads that map to merge active predictions
with the complete previous velocity. Every returned velocity owns new storage.
The scheduler remains outside this boundary.
"""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def pack(
        L,
        IDS,
        IDX,
        OIDX,
        OL,
        OIDS,
        MAP,
        K: tl.constexpr,
        D: tl.constexpr,
        AXES: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        IDCOLS: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        columns = tl.arange(0, COLS)
        global_rows = tl.load(IDX + rows, rows < K, other=0)
        tl.store(OIDX + rows, global_rows, rows < K)
        tl.store(MAP + global_rows, rows, rows < K)
        values = tl.load(
            L + global_rows[:, None] * D + columns[None, :], (rows[:, None] < K) & (columns[None, :] < D), other=0
        )
        tl.store(OL + rows[:, None] * D + columns[None, :], values, (rows[:, None] < K) & (columns[None, :] < D))
        axes = tl.arange(0, IDCOLS)
        positions = tl.load(
            IDS + global_rows[:, None] * AXES + axes[None, :], (rows[:, None] < K) & (axes[None, :] < AXES), other=0
        )
        tl.store(OIDS + rows[:, None] * AXES + axes[None, :], positions, (rows[:, None] < K) & (axes[None, :] < AXES))

    @triton.jit
    def commit(
        PREV,
        POS,
        NEG,
        MAP,
        AGE,
        OUT,
        SCALE,
        N: tl.constexpr,
        D: tl.constexpr,
        CFG: tl.constexpr,
        POS_STRIDE: tl.constexpr,
        NEG_STRIDE: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        columns = tl.arange(0, COLS)
        active = tl.load(MAP + rows, rows < N, other=-1)
        mask = (rows[:, None] < N) & (columns[None, :] < D)
        active_mask = mask & (active[:, None] >= 0)
        previous = tl.load(PREV + rows[:, None] * D + columns[None, :], mask, other=0)
        positive = tl.load(POS + active[:, None] * POS_STRIDE + columns[None, :], active_mask, other=0).to(tl.float32)
        if CFG:
            negative = tl.load(NEG + active[:, None] * NEG_STRIDE + columns[None, :], active_mask, other=0).to(
                tl.float32
            )
            # Native PyTorch materializes each BF16/FP16 CFG intermediate.
            delta = (positive - negative).to(OUT.dtype.element_ty, fp_downcast_rounding="rtne").to(tl.float32)
            scaled = (delta * SCALE).to(OUT.dtype.element_ty, fp_downcast_rounding="rtne").to(tl.float32)
            updated = (negative + scaled).to(OUT.dtype.element_ty, fp_downcast_rounding="rtne")
        else:
            updated = positive.to(OUT.dtype.element_ty)
        velocity = tl.where(active[:, None] >= 0, updated, previous)
        tl.store(OUT + rows[:, None] * D + columns[None, :], velocity, mask)
        age = tl.load(AGE + rows, rows < N, other=0)
        tl.store(AGE + rows, tl.where(active >= 0, 0, age + 1), rows < N)

    return pack, commit


def _backend(device: torch.device, backend: str) -> str:
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be auto, torch, or triton")
    if backend == "triton" and device.type != "cuda":
        raise ValueError("the Triton stage backend requires CUDA")
    if device.type == "cuda" and backend != "torch":
        return "triton"
    return f"torch_{device.type}_reference"


def pack_sparse_inputs(
    latents, generated_ids, indices, out_indices, out_latents, out_ids, row_map, *, backend="auto"
) -> str:
    """Bind current global positions and input values into existing buffers."""
    if latents.ndim != 3 or latents.shape[0] != 1:
        raise ValueError("stage packing requires latents of shape [1,N,D]")
    _, count, channels = latents.shape
    active = indices.numel()
    axes = generated_ids.shape[1]
    if (
        generated_ids.shape[0] != count
        or out_indices.shape != (1, active)
        or out_latents.shape != (1, active, channels)
        or out_ids.shape != (active, axes)
        or row_map.shape != (count,)
    ):
        raise ValueError("stage packing buffers do not match N/K/D/position axes")
    tensors = (latents, generated_ids, indices, out_indices, out_latents, out_ids, row_map)
    if any(t.device != latents.device or not t.is_contiguous() for t in tensors):
        raise ValueError("stage packing requires contiguous buffers on one device")
    actual = _backend(latents.device, backend)
    row_map.fill_(-1)
    if actual == "triton":
        import triton

        pack, _ = _kernels()
        pack[(triton.cdiv(active, 16),)](
            latents,
            generated_ids,
            indices,
            out_indices,
            out_latents,
            out_ids,
            row_map,
            active,
            channels,
            axes,
            16,
            triton.next_power_of_2(channels),
            triton.next_power_of_2(axes),
            num_warps=4,
        )
    else:
        flat = indices.reshape(-1)
        out_indices.copy_(indices.reshape_as(out_indices))
        torch.index_select(latents, 1, flat, out=out_latents)
        torch.index_select(generated_ids, 0, flat, out=out_ids)
        row_map[flat] = torch.arange(active, device=row_map.device, dtype=row_map.dtype)
    return actual


def commit_sparse_velocity(
    previous, positive, negative, indices, row_map, age, scale, *, backend="auto"
) -> tuple[torch.Tensor, str]:
    """Combine CFG, preserve inactive velocity and update age without history aliasing."""
    if previous.ndim != 3 or previous.shape[0] != 1:
        raise ValueError("stage commit requires previous velocity of shape [1,N,D]")
    _, count, channels = previous.shape
    active = indices.numel()
    if (
        positive.shape != (1, active, channels)
        or row_map.shape != (count,)
        or age.shape != (1, count)
        or (negative is not None and negative.shape != positive.shape)
    ):
        raise ValueError("stage commit buffers do not match N/K/D")
    tensors = (previous, positive, indices, row_map, age) + (() if negative is None else (negative,))
    if any(t.device != previous.device for t in tensors):
        raise ValueError("stage commit buffers must share a device")
    if any(not t.is_contiguous() for t in (previous, row_map, age)):
        raise ValueError("stage commit state buffers must be contiguous")
    if positive.stride(-1) != 1 or (negative is not None and negative.stride(-1) != 1):
        raise ValueError("stage predictions require contiguous channels")
    if positive.dtype != previous.dtype or (negative is not None and negative.dtype != previous.dtype):
        raise ValueError("stage predictions and previous velocity must share dtype")
    actual = _backend(previous.device, backend)
    if actual == "triton":
        import triton

        _, commit = _kernels()
        result = torch.empty_like(previous)
        commit[(triton.cdiv(count, 16),)](
            previous,
            positive,
            negative if negative is not None else positive,
            row_map,
            age,
            result,
            float(scale),
            count,
            channels,
            negative is not None,
            positive.stride(1),
            negative.stride(1) if negative is not None else positive.stride(1),
            16,
            triton.next_power_of_2(channels),
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        result = previous.clone()
        prediction = positive if negative is None else negative + float(scale) * (positive - negative)
        result.index_copy_(1, indices.reshape(-1), prediction)
        age.add_(1)
        age.scatter_(1, indices.reshape(1, -1), 0)
    return result, actual
