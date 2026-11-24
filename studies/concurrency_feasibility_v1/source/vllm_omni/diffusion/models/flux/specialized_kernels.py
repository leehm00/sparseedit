# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803
"""Targeted rotation/write-back kernel for the fused FLUX specialized path.

The kernel fuses three otherwise separate operations for the active image
rows:

1. interleaved RoPE rotation of the normalized (post-RMSNorm) FLUX key,
2. the key write into the stable joint K buffer, and
3. the value write into the stable joint V buffer.

Text K/V submission is a plain copy of already-computed rows and stays in
PyTorch; it is outside this kernel's scope.  Full GEMMs, MLPs and attention are
untouched.

Arithmetic contract
-------------------
The pre-existing CUDA path rotates with FlashAttention's rotary kernel, while
the portable path uses :func:`apply_rotary_emb_torch`.  This kernel loads the
key and the (already dtype-cast) cosine/sine in ``float32``, performs the
rotation in ``float32``, and casts the result to the destination dtype.  The
value copy performs no arithmetic and is bitwise.

Consequently:

* copy-only paths (``inplace`` raw cache, text/value submission) compare
  bitwise;
* the fused key rotation compares with a declared tight floating tolerance
  (``FUSED_KERNEL_TOLERANCE``), and every downstream guided-velocity/latent/
  image comparison uses a declared tolerance rather than pretending to be
  exact.

If Triton or CUDA is unavailable, :func:`submit_rotated_kv` executes the same
semantics with the existing PyTorch RoPE and reports the actual backend.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import torch

RotaryBackend = Literal["triton", "torch_cuda_reference", "torch_cpu_reference"]

#: Declared comparison bound for a fused key rotation relative to the
#: reference RoPE.  BF16 has an 8-bit mantissa; one unit in the last place near
#: 1.0 is ~7.8e-3.  The kernel performs the same two-product/two-add rotation
#: in float32, so the expected difference is at most a couple of BF16 ULPs.
FUSED_KERNEL_TOLERANCE = {"rtol": 8e-3, "atol": 8e-3}
#: Full guided-velocity / latent / image tolerance for fused arithmetic.
FUSED_OUTPUT_TOLERANCE = {"rtol": 8e-3, "atol": 8e-3}
#: Copy-only equality.
EXACT_TOLERANCE = {"rtol": 0.0, "atol": 0.0}


def _find_triton() -> bool:
    return importlib.util.find_spec("triton") is not None


def triton_available(device: torch.device | None = None) -> bool:
    target = device if device is not None else torch.device("cpu")
    if target.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    return _find_triton()


@lru_cache(maxsize=1)
def _build_kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit
    def _rope_scatter_kv_kernel(
        K,
        V,
        OK,
        OV,
        COS,
        SIN,
        IDX,
        n_heads: tl.constexpr,
        head_dim: tl.constexpr,
        k_row_stride: tl.constexpr,
        k_head_stride: tl.constexpr,
        v_row_stride: tl.constexpr,
        v_head_stride: tl.constexpr,
        ok_row_stride: tl.constexpr,
        ok_head_stride: tl.constexpr,
        ov_row_stride: tl.constexpr,
        ov_head_stride: tl.constexpr,
        cos_row_stride: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid // n_heads
        head = pid % n_heads
        half = head_dim // 2

        d2 = tl.arange(0, BLOCK // 2)
        mask2 = d2 < half
        even = 2 * d2
        odd = even + 1
        k_base = row * k_row_stride + head * k_head_stride
        x_even = tl.load(K + k_base + even, mask=mask2, other=0.0).to(tl.float32)
        x_odd = tl.load(K + k_base + odd, mask=mask2, other=0.0).to(tl.float32)
        # Native FLUX casts position embeddings to the key dtype before RoPE.
        # Preserve that rounding even when the full position table is FP32.
        cos = tl.load(COS + row * cos_row_stride + d2, mask=mask2, other=0.0).to(K.dtype.element_ty).to(tl.float32)
        sin = tl.load(SIN + row * cos_row_stride + d2, mask=mask2, other=0.0).to(K.dtype.element_ty).to(tl.float32)
        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos

        out_row = tl.load(IDX + row)
        ok_base = out_row * ok_row_stride + head * ok_head_stride
        tl.store(OK + ok_base + even, y_even.to(OK.dtype.element_ty), mask=mask2)
        tl.store(OK + ok_base + odd, y_odd.to(OK.dtype.element_ty), mask=mask2)

        d = tl.arange(0, BLOCK)
        maskd = d < head_dim
        v_base = row * v_row_stride + head * v_head_stride
        ov_base = out_row * ov_row_stride + head * ov_head_stride
        values = tl.load(V + v_base + d, mask=maskd, other=0.0)
        tl.store(OV + ov_base + d, values, mask=maskd)

    return _rope_scatter_kv_kernel


def _contiguous_rows(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    return tensor.contiguous()


def _launch_triton(
    *,
    normalized_key: torch.Tensor,
    value: torch.Tensor,
    out_key: torch.Tensor,
    out_value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scatter_positions: torch.Tensor,
) -> None:
    head_dim = int(normalized_key.shape[-1])
    num_heads = int(normalized_key.shape[-2])
    n_rows = int(normalized_key.shape[1])
    block = 1 << (head_dim - 1).bit_length()
    kernel = _build_kernel()
    grid = (n_rows * num_heads,)
    kernel[grid](
        normalized_key,
        value,
        out_key,
        out_value,
        cos,
        sin,
        scatter_positions,
        # Keyword name must match the JIT signature (``n_heads``); Triton
        # matches constexpr arguments by name, so ``num_heads`` would raise.
        n_heads=num_heads,
        head_dim=head_dim,
        k_row_stride=int(normalized_key.stride(1)),
        k_head_stride=int(normalized_key.stride(2)),
        v_row_stride=int(value.stride(1)),
        v_head_stride=int(value.stride(2)),
        ok_row_stride=int(out_key.stride(1)),
        ok_head_stride=int(out_key.stride(2)),
        ov_row_stride=int(out_value.stride(1)),
        ov_head_stride=int(out_value.stride(2)),
        cos_row_stride=int(cos.stride(0)),
        BLOCK=block,
        num_warps=1,
    )


def _torch_rotation(
    normalized_key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope: Any | None,
) -> torch.Tensor:
    if rope is not None:
        return rope(normalized_key, cos.to(normalized_key.dtype), sin.to(normalized_key.dtype))
    from vllm_omni.diffusion.layers.rope import apply_rotary_emb_torch

    return apply_rotary_emb_torch(
        normalized_key,
        cos.to(normalized_key.dtype),
        sin.to(normalized_key.dtype),
        interleaved=True,
    )


def submit_rotated_kv(
    *,
    normalized_key: torch.Tensor,
    value: torch.Tensor,
    out_key: torch.Tensor,
    out_value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scatter_positions: torch.Tensor,
    rope: Any | None = None,
    backend: str = "auto",
) -> RotaryBackend:
    """Rotate active normalized K and submit K/V into the stable joint cache.

    ``scatter_positions`` holds absolute joint positions (text offset already
    applied).  Returns the backend actually used, never an inferred one.
    """
    if backend not in {"auto", "triton", "torch"}:
        raise ValueError(f"unsupported rotary backend: {backend!r}")
    if normalized_key.shape != value.shape:
        raise ValueError("normalized_key and value must have identical shapes")
    if normalized_key.ndim != 4:
        raise ValueError("normalized_key must be [1, rows, heads, head_dim]")
    if normalized_key.shape[1] == 0:
        raise ValueError("fused submission requires at least one row")
    if normalized_key.device != out_key.device or normalized_key.dtype != out_key.dtype:
        raise ValueError("source and destination K must share device and dtype")
    if out_key.shape != out_value.shape or out_key.ndim != 4:
        raise ValueError("joint cache buffers must be matching [1, tokens, heads, head_dim] tensors")

    key = _contiguous_rows(normalized_key, "normalized_key")
    values = _contiguous_rows(value, "value")
    cos_rows = _contiguous_rows(cos, "cos")
    sin_rows = _contiguous_rows(sin, "sin")
    if cos_rows.ndim == 3 and cos_rows.shape[0] == 1:
        cos_rows = cos_rows[0]
        sin_rows = sin_rows[0]
    if cos_rows.ndim != 2 or cos_rows.shape[0] != key.shape[1]:
        raise ValueError(
            f"cos/sin must be [rows, head_dim/2] matching {key.shape[1]} rows, got {tuple(cos_rows.shape)}"
        )
    positions = scatter_positions.reshape(-1).to(device=key.device, dtype=torch.long)
    if positions.numel() != key.shape[1]:
        raise ValueError("scatter_positions must contain one entry per active row")

    wanted_triton = backend in {"auto", "triton"}
    if wanted_triton and triton_available(key.device):
        _launch_triton(
            normalized_key=key,
            value=values,
            out_key=out_key,
            out_value=out_value,
            cos=cos_rows,
            sin=sin_rows,
            scatter_positions=positions,
        )
        return "triton"
    if backend == "triton" and not triton_available(key.device):
        raise RuntimeError("fused rotary backend requested Triton, but Triton/CUDA is unavailable")

    rotated = _torch_rotation(key, cos_rows, sin_rows, rope)
    out_key.index_copy_(1, positions, rotated)
    out_value.index_copy_(1, positions, values)
    return "torch_cuda_reference" if key.device.type == "cuda" else "torch_cpu_reference"


@dataclass(frozen=True)
class RotaryBackendInfo:
    requested: str
    actual: str
    tolerance: dict[str, float]


def describe_backend(actual: RotaryBackend) -> RotaryBackendInfo:
    tolerance = dict(EXACT_TOLERANCE) if actual == "torch_cpu_reference" else dict(FUSED_KERNEL_TOLERANCE)
    return RotaryBackendInfo(
        requested="auto",
        actual=actual,
        tolerance=tolerance,
    )


__all__ = [
    "EXACT_TOLERANCE",
    "FUSED_KERNEL_TOLERANCE",
    "FUSED_OUTPUT_TOLERANCE",
    "RotaryBackend",
    "RotaryBackendInfo",
    "describe_backend",
    "submit_rotated_kv",
    "triton_available",
]
