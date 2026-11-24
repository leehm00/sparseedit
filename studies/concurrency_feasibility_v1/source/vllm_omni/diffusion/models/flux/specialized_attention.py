# SPDX-License-Identifier: Apache-2.0
"""Attention-ready K/V resolution for the opt-in FLUX specialized runtime.

This module is the numerical bridge between the existing ``FluxAttention``
projections and the unchanged ``Attention`` op.  It is deliberately tensor-only
so it can be exercised on CPU in the focused tests.

Layouts (batch size is always one for this protocol):

* dual stream: ``key``/``value`` are the image rows; the text rows come from
  ``add_kv_proj`` (``encoder_key``/``encoder_value``).
* single stream: ``hidden_states`` is ``[text | compact image]``; the text rows
  are the prefix of ``key``/``value`` and are recomputed on every forward.

RoPE positions always stay global.  A compact query is rotated with the compact
position embeddings computed by the transformer; cached image K rows are
rotated with their original full-grid positions.  This is what makes
``Q_compact . K_full`` identical to the full-context attention score.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.models.flux.specialized_cache import SpecializedCacheError, SpecializedKVCache
from vllm_omni.diffusion.models.flux.specialized_kernels import submit_rotated_kv


def _as_rows(cos: torch.Tensor) -> torch.Tensor:
    if cos.ndim == 3 and cos.shape[0] == 1:
        return cos[0]
    return cos


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope: Any) -> torch.Tensor:
    return rope(x, cos.to(x.dtype), sin.to(x.dtype))


def split_text_image(
    key: torch.Tensor,
    value: torch.Tensor,
    encoder_key: torch.Tensor | None,
    encoder_value: torch.Tensor | None,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
    active_image_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return ``(text_k, text_v, image_k, image_v, text_tokens)`` without copying."""
    if encoder_hidden_states is None:
        text_tokens = int(hidden_states.shape[1]) - int(active_image_tokens)
        if text_tokens < 0:
            raise SpecializedCacheError("FLUX specialized single-stream image token count exceeds the sequence length")
        if encoder_key is not None or encoder_value is not None:
            raise SpecializedCacheError("single-stream specialized attention received text projections")
        return (
            key[:, :text_tokens],
            value[:, :text_tokens],
            key[:, text_tokens:],
            value[:, text_tokens:],
            text_tokens,
        )
    if encoder_key is None or encoder_value is None:
        raise SpecializedCacheError("dual-stream specialized attention is missing text K/V projections")
    if int(encoder_key.shape[1]) != int(encoder_hidden_states.shape[1]):
        raise SpecializedCacheError("dual-stream text projection length does not match the text hidden states")
    return encoder_key, encoder_value, key, value, int(encoder_key.shape[1])


def resolve_specialized_kv(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    encoder_key: torch.Tensor | None,
    encoder_value: torch.Tensor | None,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
    context: Any,
    rope: Any,
) -> tuple[torch.Tensor, torch.Tensor, bool] | None:
    """Resolve cached sparse K/V, or return ``None`` for the original path.

    Returned ``key`` is either the raw full joint K (``already_rotated=False``)
    or the attention-ready joint K (``already_rotated=True``).  The value is
    always attention-ready (values are never rotated).
    """
    cache = getattr(context, "specialized_cache", None)
    if cache is None:
        return None
    if context.kv_action not in {"store_full", "update_active"}:
        # Warmup / ratio-1 dense calls intentionally retain the original
        # full-context code path; no cache is read before its first store.
        return None
    if not isinstance(cache, SpecializedKVCache):
        raise SpecializedCacheError("specialized_cache must be a SpecializedKVCache")
    if context.full_rotary_emb is None:
        raise SpecializedCacheError("specialized sparse attention requires full-grid rotary embeddings")

    stream, layer_index = context.resolve_layer()
    text_key, text_value, image_key, image_value, text_tokens = split_text_image(
        key,
        value,
        encoder_key,
        encoder_value,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        active_image_tokens=context.image_indices.shape[1],
    )
    if cache.mode == "rotated" and text_tokens != cache.text_token_count:
        raise SpecializedCacheError(
            f"FLUX specialized text length changed within a request: got {text_tokens}, "
            f"cache owns {cache.text_token_count}"
        )

    if cache.mode == "raw":
        if context.kv_action == "store_full":
            full_image_key, full_image_value = cache.store_full_image(stream, layer_index, image_key, image_value)
        else:
            full_image_key, full_image_value = cache.update_active_image(
                stream,
                layer_index,
                image_key,
                image_value,
                context.image_indices,
            )
        cache.backends.add("torch_reference")
        return (
            torch.cat([text_key, full_image_key], dim=1),
            torch.cat([text_value, full_image_value], dim=1),
            False,
        )

    # Attention-ready joint layout [text | generated | reference].
    full_cos, full_sin = context.full_rotary_emb
    cos_rows = _as_rows(full_cos)
    sin_rows = _as_rows(full_sin)
    if cos_rows.shape[0] != cache.joint_token_count:
        raise SpecializedCacheError(
            f"full rotary embedding covers {cos_rows.shape[0]} positions, cache expects {cache.joint_token_count}"
        )

    rotated_text_key = _rotate(text_key, cos_rows[:text_tokens], sin_rows[:text_tokens], rope)

    if context.kv_action == "store_full":
        if int(image_key.shape[1]) != cache.full_image_token_count:
            raise SpecializedCacheError(
                "FLUX specialized full-cache step must receive every generated and reference image token: "
                f"got {image_key.shape[1]}, expected {cache.full_image_token_count}"
            )
        image_cos = cos_rows[text_tokens:]
        image_sin = sin_rows[text_tokens:]
        positions = (
            context.layer_kv_positions
            if context.layer_fusion and context.layer_positions and context.layer_kv_positions is not None
            else torch.arange(
                text_tokens, cache.joint_token_count, device=image_key.device, dtype=torch.long
            ).unsqueeze(0)
        )
        if cache.submission_backend() == "fused":
            storage = cache.ensure_layer(stream, layer_index)
            backend = submit_rotated_kv(
                normalized_key=image_key,
                value=image_value,
                out_key=storage.key,
                out_value=storage.value,
                cos=image_cos,
                sin=image_sin,
                scatter_positions=positions,
                rope=rope,
                backend="auto",
            )
            storage.key[:, :text_tokens].copy_(rotated_text_key.detach())
            storage.value[:, :text_tokens].copy_(text_value.detach())
            storage.writes += 1
            cache.record_submission(int(image_key.shape[1]), backend=backend)
            return storage.key, storage.value, True
        rotated_image_key = _rotate(image_key, image_cos, image_sin, rope)
        cache.backends.add("torch_reference")
        return (
            *cache.store_full_joint(
                stream,
                layer_index,
                rotated_image_key,
                image_value,
                rotated_text_key,
                text_value,
            ),
            True,
        )

    indices = context.image_indices.to(device=image_key.device, dtype=torch.long).reshape(-1)
    if indices.numel() != int(image_key.shape[1]):
        raise SpecializedCacheError("specialized active indices do not match the compact image row count")
    if context.layer_fusion and context.layer_positions and context.layer_image_rotary is not None:
        positions = context.layer_kv_positions
        image_cos, image_sin = context.layer_image_rotary
    else:
        positions = (indices + text_tokens).unsqueeze(0)
        image_cos = cos_rows.index_select(0, positions.reshape(-1))
        image_sin = sin_rows.index_select(0, positions.reshape(-1))
    if cache.submission_backend() == "fused":
        storage = cache.require_layer(stream, layer_index)
        backend = submit_rotated_kv(
            normalized_key=image_key,
            value=image_value,
            out_key=storage.key,
            out_value=storage.value,
            cos=image_cos,
            sin=image_sin,
            scatter_positions=positions,
            rope=rope,
            backend="auto",
        )
        storage.key[:, :text_tokens].copy_(rotated_text_key.detach())
        storage.value[:, :text_tokens].copy_(text_value.detach())
        storage.writes += 1
        cache.record_submission(int(image_key.shape[1]), backend=backend)
        return storage.key, storage.value, True

    rotated_image_key = _rotate(image_key, image_cos, image_sin, rope)
    cache.backends.add("torch_reference")
    return (
        *cache.update_active_joint(
            stream,
            layer_index,
            rotated_image_key,
            image_value,
            rotated_text_key,
            text_value,
            positions,
        ),
        True,
    )


__all__ = ["resolve_specialized_kv", "split_text_image"]
