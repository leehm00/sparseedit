# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803
"""FLUX layer modulation and gated residual with native rounding boundaries.

Native LayerNorm statistics and all projections stay unchanged. The kernels
merge subsequent elementwise operations while retaining every original dtype
rounding; they apply to compact sparse rows and full warmup/refresh rows.
"""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _layer_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        X,
        VALUE,
        SCALE,
        SHIFT,
        OUT,
        COUNT: tl.constexpr,
        CHANNELS: tl.constexpr,
        ROWS: tl.constexpr,
        SCALE_BATCH_STRIDE: tl.constexpr,
        SHIFT_BATCH_STRIDE: tl.constexpr,
        MODE: tl.constexpr,
        CLIP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        channels = offsets % CHANNELS
        batch = offsets // (ROWS * CHANNELS)
        valid = offsets < COUNT
        x = tl.load(X + offsets, valid, other=0).to(tl.float32)
        scale = tl.load(SCALE + batch * SCALE_BATCH_STRIDE + channels, valid, other=0).to(tl.float32)
        if MODE == "modulate":
            shift = tl.load(SHIFT + batch * SHIFT_BATCH_STRIDE + channels, valid, other=0).to(tl.float32)
            multiplier = (1.0 + scale).to(OUT.dtype.element_ty, fp_downcast_rounding="rtne").to(tl.float32)
            product = (x * multiplier).to(OUT.dtype.element_ty, fp_downcast_rounding="rtne").to(tl.float32)
            result = (product + shift).to(OUT.dtype.element_ty, fp_downcast_rounding="rtne")
        else:
            value = tl.load(VALUE + offsets, valid, other=0).to(tl.float32)
            product = (scale * value).to(OUT.dtype.element_ty, fp_downcast_rounding="rtne").to(tl.float32)
            result = (x + product).to(OUT.dtype.element_ty, fp_downcast_rounding="rtne")
        if CLIP:
            result = tl.minimum(tl.maximum(result.to(tl.float32), -65504.0), 65504.0)
        tl.store(OUT + offsets, result, valid)

    return kernel


def _validate(x, scale, other):
    if x.ndim != 3 or x.shape[0] not in (1, 2) or scale.shape != (x.shape[0], x.shape[-1]):
        raise ValueError("layer fusion requires x=[B,S,C], B in {1,2}, and conditioning=[B,C]")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("layer fusion supports FP32, FP16 and BF16")
    if any(t.dtype != x.dtype or t.device != x.device for t in (scale, other)):
        raise ValueError("layer fusion tensors must share dtype and device")
    if not x.is_contiguous() or scale.stride(-1) != 1:
        raise ValueError("layer fusion requires contiguous rows and conditioning channels")


def modulate(normalized, scale, shift, *, backend="auto"):
    """Return normalized*(1+scale)+shift, preserving three dtype roundings."""
    _validate(normalized, scale, shift)
    if shift.shape != scale.shape or shift.stride(-1) != 1:
        raise ValueError("modulation shift must match conditioning shape")
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("unknown layer fusion backend")
    if backend == "triton" and normalized.device.type != "cuda":
        raise ValueError("Triton layer fusion requires CUDA")
    if normalized.device.type != "cuda" or backend == "torch":
        return normalized * (1 + scale[:, None]) + shift[:, None]
    import triton

    output = torch.empty_like(normalized)
    _layer_kernel()[(triton.cdiv(normalized.numel(), 1024),)](
        normalized,
        normalized,
        scale,
        shift,
        output,
        normalized.numel(),
        normalized.shape[-1],
        normalized.shape[1],
        scale.stride(0),
        shift.stride(0),
        "modulate",
        False,
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def gated_residual(residual, value, gate, *, clip=False, backend="auto"):
    """Return residual+round(gate*value), optionally applying native FP16 clip."""
    _validate(residual, gate, value)
    if value.shape != residual.shape or not value.is_contiguous():
        raise ValueError("residual update must share shape and contiguous layout")
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("unknown layer fusion backend")
    if backend == "triton" and residual.device.type != "cuda":
        raise ValueError("Triton layer fusion requires CUDA")
    if residual.device.type != "cuda" or backend == "torch":
        result = residual + gate[:, None] * value
        return result.clamp(-65504, 65504) if clip else result
    import triton

    output = torch.empty_like(residual)
    _layer_kernel()[(triton.cdiv(residual.numel(), 1024),)](
        residual,
        value,
        gate,
        gate,
        output,
        residual.numel(),
        residual.shape[-1],
        residual.shape[1],
        gate.stride(0),
        gate.stride(0),
        "residual",
        clip,
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def adaptive_norm(module, x, embedding, *, kind, condition=None, projected=None):
    """Keep the module's exact embedding projection and LayerNorm reduction."""
    if kind not in ("zero", "single", "continuous"):
        raise ValueError("unsupported adaptive norm kind")
    if projected is None:
        if condition is None:
            condition = module.silu(embedding)
        if kind == "continuous":
            condition = condition.to(x.dtype)
        projected = module.linear(condition)
        if isinstance(projected, tuple):
            projected = projected[0]
    count = {"zero": 6, "single": 3, "continuous": 2}[kind]
    chunks = projected.chunk(count, dim=1)
    if kind == "continuous":
        scale, shift = chunks
        return modulate(module.norm(x), scale, shift)
    shift, scale, gate = chunks[:3]
    normalized = modulate(module.norm(x), scale, shift)
    return (normalized, gate, *chunks[3:])


def prepare_layer_rotary(context, query_rotary, full_rotary, dtype):
    """Share unchanged position gathers and dtype conversions across all layers."""
    full = tuple(tensor.to(dtype) for tensor in full_rotary)
    query = tuple(tensor.to(dtype) for tensor in query_rotary)
    rows = tuple(tensor[0] if tensor.ndim == 3 else tensor for tensor in full)
    text_tokens = rows[0].shape[0] - context.generated_token_count - context.reference_token_count
    if text_tokens < 0:
        raise ValueError("full rotary rows cannot cover the image context")
    positions = context.image_indices.reshape(-1) + text_tokens
    context.layer_kv_positions = positions.unsqueeze(0)
    context.layer_image_rotary = tuple(tensor.index_select(0, positions) for tensor in rows)
    return full, query
