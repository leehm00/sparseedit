# SPDX-License-Identifier: Apache-2.0
"""Compact queries and full cached context for native diffusers Qwen-Image-Edit.

This leaf module imports only torch and diffusers.  The original absolute image
RoPE table is indexed for compact queries; no compact image grid is generated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage import (
    apply_rotary_emb_qwen,
    compute_text_seq_len_from_mask,
)
from torch import nn
from .sparseedit_native_partition import LocalTensorCache

CacheMode = Literal["direct", "inplace", "ready"]


@dataclass(frozen=True)
class QwenSparseEditConfig:
    ratio: float = 0.2
    warmup_steps: int = 8
    refresh_steps: int = 23
    num_steps: int = 28
    local_weight: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        if (self.num_steps, self.warmup_steps, self.refresh_steps) != (28, 8, 23):
            raise ValueError("Qwen protocol fixes 28 steps, W8 and refresh 23")

    @property
    def dense_steps(self) -> frozenset[int]:
        if self.ratio == 1.0:
            return frozenset(range(self.num_steps))
        return frozenset(range(self.warmup_steps)) | {self.refresh_steps}


@dataclass
class QwenLayerKV:
    key: torch.Tensor
    value: torch.Tensor
    image_tokens: int
    generated_tokens: int
    text_tokens: int


class QwenSparseKVCache:
    """Request-owned image-only raw K/V or joint post-RoPE K/V."""

    def __init__(self, mode: CacheMode = "direct") -> None:
        if mode not in ("direct", "inplace", "ready"):
            raise ValueError(f"unsupported Qwen cache mode: {mode}")
        self.mode = mode
        self.layers: dict[tuple[str, int], QwenLayerKV] = {}
        self.layouts: dict[str, tuple] = {}
        self.full_masks: dict[str, torch.Tensor] = {}
        self.last_query_rope = None
        self.last_attention_mask = None
        self.local_aux = LocalTensorCache()

    def store(
        self,
        branch: str,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        image_tokens: int = 0,
        generated_tokens: int = 0,
        text_tokens: int = 0,
    ) -> None:
        if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
            raise ValueError(
                "Qwen K/V must be equal-shaped [1, tokens, heads, dim] tensors"
            )
        previous = self.layers.get((branch, layer))
        reusable = (
            previous is not None
            and previous.key.shape == key.shape
            and previous.key.dtype == key.dtype
            and previous.key.device == key.device
            and previous.image_tokens == image_tokens
            and previous.text_tokens == text_tokens
        )
        if self.mode == "direct" or not reusable:
            self.layers[(branch, layer)] = QwenLayerKV(
                key.detach().clone(),
                value.detach().clone(),
                image_tokens,
                generated_tokens,
                text_tokens,
            )
        else:
            previous.key.copy_(key.detach())
            previous.value.copy_(value.detach())
            previous.generated_tokens = generated_tokens

    def adopt(
        self,
        branch: str,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        image_tokens: int,
        generated_tokens: int,
    ) -> None:
        if self.mode != "direct":
            raise RuntimeError("adopt is valid only for direct raw-K control")
        text_tokens = self.require(branch, layer).text_tokens
        self.layers[(branch, layer)] = QwenLayerKV(
            key, value, image_tokens, generated_tokens, text_tokens
        )

    def require(self, branch: str, layer: int) -> QwenLayerKV:
        state = self.layers.get((branch, layer))
        if state is None:
            raise RuntimeError(
                f"Qwen sparse {branch} layer {layer} requires a dense warmup or refresh"
            )
        return state

    def reset_request(self) -> None:
        self.layers.clear()
        self.layouts.clear()
        self.full_masks.clear()
        self.last_query_rope = None
        self.last_attention_mask = None
        self.local_aux = LocalTensorCache()


def _rotate(value: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    return apply_rotary_emb_qwen(value, freqs, use_real=False)


def _scatter_rows(
    target: torch.Tensor, indices: torch.Tensor, source: torch.Tensor, offset: int = 0
) -> None:
    positions = indices.to(device=source.device, dtype=torch.long) + offset
    target.scatter_(
        1, positions.reshape(1, -1, 1, 1).expand_as(source), source.detach()
    )


class SparseQwenAttention(nn.Module):
    """Native Qwen double-stream processor with full generated/reference context."""

    def __init__(
        self, native: nn.Module, cache: QwenSparseKVCache, branch: str, layer: int
    ) -> None:
        super().__init__()
        if (
            native.added_kv_proj_dim is None
            or native.norm_q is None
            or native.norm_k is None
        ):
            raise ValueError(
                "SparseQwenAttention requires native separate-stream QK-normalized attention"
            )
        self.native = native
        self._attention_backend = native.processor._attention_backend
        self._parallel_config = native.processor._parallel_config
        self.cache = cache
        self.branch = branch
        self.layer = layer
        self.dense = True
        self.indices: torch.Tensor | None = None
        self.image_tokens = 0
        self.generated_tokens = 0
        self.full_image_freqs = None
        self.query_image_freqs = None
        self.text_freqs = None
        self.local_attention_mask = None

    def configure(
        self,
        *,
        image_tokens: int,
        indices: torch.Tensor | None,
        full_image_freqs,
        query_image_freqs,
        text_freqs,
    ) -> None:
        self.image_tokens = int(image_tokens)
        self.indices = indices
        self.full_image_freqs = full_image_freqs
        self.query_image_freqs = query_image_freqs
        self.text_freqs = text_freqs

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        image_rotary_emb=None,
        encoder_hidden_states_mask=None,
        **kwargs,
    ):
        if encoder_hidden_states is None:
            raise ValueError("native Qwen joint attention requires text")
        if self.local_attention_mask is not None:
            attention_mask = self.local_attention_mask
        attn = self.native
        heads, dim = attn.heads, attn.inner_dim // attn.heads
        text_count = encoder_hidden_states.shape[1]
        img_query = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (heads, dim)))
        img_key = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (heads, dim)))
        img_value = attn.to_v(hidden_states).unflatten(-1, (heads, dim))
        txt_query = attn.norm_added_q(
            attn.add_q_proj(encoder_hidden_states).unflatten(-1, (heads, dim))
        )
        txt_key = attn.norm_added_k(
            attn.add_k_proj(encoder_hidden_states).unflatten(-1, (heads, dim))
        )
        txt_value = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (heads, dim))
        if self.dense:
            img_freqs, txt_freqs = image_rotary_emb
            query = torch.cat(
                (_rotate(txt_query, txt_freqs), _rotate(img_query, img_freqs)), dim=1
            )
            key = torch.cat(
                (_rotate(txt_key, txt_freqs), _rotate(img_key, img_freqs)), dim=1
            )
            value = torch.cat((txt_value, img_value), dim=1)
            if self.cache.mode == "ready":
                self.cache.store(
                    self.branch,
                    self.layer,
                    key,
                    value,
                    image_tokens=self.image_tokens,
                    generated_tokens=self.generated_tokens,
                    text_tokens=text_count,
                )
            else:
                self.cache.store(
                    self.branch,
                    self.layer,
                    img_key,
                    img_value,
                    image_tokens=self.image_tokens,
                    generated_tokens=self.generated_tokens,
                    text_tokens=text_count,
                )
        else:
            if self.indices is None or hidden_states.shape[1] != self.indices.numel():
                raise ValueError("compact Qwen rows must match active indices")
            state = self.cache.require(self.branch, self.layer)
            if state.key.device != hidden_states.device or state.value.device != hidden_states.device:
                raise ValueError("partition layer KV must remain on its owning block device")
            txt_freqs = self.text_freqs
            query = torch.cat(
                (
                    _rotate(txt_query, txt_freqs),
                    _rotate(img_query, self.query_image_freqs),
                ),
                dim=1,
            )
            if self.cache.mode == "ready":
                rotated_text = _rotate(txt_key, txt_freqs)
                rotated_active = _rotate(img_key, self.query_image_freqs)
                state.key[:, :text_count].copy_(rotated_text.detach())
                state.value[:, :text_count].copy_(txt_value.detach())
                _scatter_rows(state.key, self.indices, rotated_active, text_count)
                _scatter_rows(state.value, self.indices, img_value, text_count)
                key, value = state.key, state.value
            else:
                if state.key.shape[1] != self.image_tokens:
                    raise ValueError(
                        "raw Qwen cache image length differs from original full context"
                    )
                image_key = (
                    state.key if self.cache.mode == "inplace" else state.key.clone()
                )
                image_value = (
                    state.value if self.cache.mode == "inplace" else state.value.clone()
                )
                _scatter_rows(image_key, self.indices, img_key)
                _scatter_rows(image_value, self.indices, img_value)
                if self.cache.mode == "direct":
                    self.cache.adopt(
                        self.branch,
                        self.layer,
                        image_key,
                        image_value,
                        image_tokens=self.image_tokens,
                        generated_tokens=self.generated_tokens,
                    )
                key = torch.cat(
                    (
                        _rotate(txt_key, txt_freqs),
                        _rotate(image_key, self.full_image_freqs),
                    ),
                    dim=1,
                )
                value = torch.cat((txt_value, image_value), dim=1)
        self.cache.last_attention_mask = attention_mask
        output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        output = output.flatten(2, 3).to(query.dtype)
        text_output, image_output = output.split_with_sizes(
            [text_count, output.shape[1] - text_count], dim=1
        )
        image_output = attn.to_out[0](image_output.contiguous())
        if len(attn.to_out) > 1:
            image_output = attn.to_out[1](image_output)
        return image_output, attn.to_add_out(text_output.contiguous())


class _BoundQwenAttention:
    def __init__(self, model, cache, branch):
        self.model, self.cache, self.branch = model, cache, branch
        self.saved = []
        self.wrappers = []

    def __enter__(self):
        for layer, block in enumerate(self.model.transformer_blocks):
            attention = block.attn
            self.saved.append((attention, attention.processor))
            wrapper = SparseQwenAttention(attention, self.cache, self.branch, layer)
            self.wrappers.append(wrapper)
            attention.__dict__["processor"] = wrapper
        return self.wrappers

    def __exit__(self, *exc):
        for attention, processor in reversed(self.saved):
            attention.__dict__["processor"] = processor


def _full_image_tokens(model, img_shapes) -> int:
    if not isinstance(img_shapes, list) or len(img_shapes) != 1:
        raise ValueError("Qwen SparseEdit requires batch-one full img_shapes")
    if any(
        not isinstance(shape, (list, tuple)) or len(shape) != 3
        for shape in img_shapes[0]
    ):
        raise ValueError(
            "full img_shapes must contain native (frame, height, width) entries"
        )
    return sum(math.prod(shape) for shape in img_shapes[0])


def qwen_sparse_model_call(
    model,
    *,
    cache: QwenSparseKVCache,
    branch: str,
    mode: str,
    active_indices: torch.Tensor | None = None,
    generated_tokens: int | None = None,
    full_img_shapes=None,
    native_call=None,
    prepare_only: bool = False,
    prepared: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Run native Qwen; dense uses native math and sparse uses original RoPE rows."""
    if mode not in ("dense", "sparse"):
        raise ValueError("mode must be dense or sparse")
    if (
        kwargs["hidden_states"].shape[0] != 1
        or kwargs["encoder_hidden_states"].shape[0] != 1
    ):
        raise ValueError("Qwen SparseEdit supports batch one only")
    img_shapes = kwargs.get("img_shapes")
    output_generated_tokens = generated_tokens
    image_tokens = _full_image_tokens(model, img_shapes)
    text_count, _, text_mask = compute_text_seq_len_from_mask(
        kwargs["encoder_hidden_states"], kwargs.get("encoder_hidden_states_mask")
    )
    native_grid = tuple(tuple(int(v) for v in shape) for shape in img_shapes[0])
    generated_tokens = (
        math.prod(native_grid[0]) if generated_tokens is None else generated_tokens
    )
    hidden, text = kwargs["hidden_states"], kwargs["encoder_hidden_states"]
    layout = (
        native_grid,
        generated_tokens,
        text_count,
        text.shape[2],
        hidden.dtype,
        hidden.device,
        text.dtype,
        text.device,
    )
    if mode == "dense":
        if kwargs["hidden_states"].shape[1] != image_tokens:
            raise ValueError("dense Qwen image tokens do not match full img_shapes")
        if generated_tokens != math.prod(native_grid[0]):
            raise ValueError("generated partition differs from native image layout")
        cache.layouts[branch] = layout
    else:
        if active_indices is None:
            raise ValueError("sparse call requires active indices")
        if full_img_shapes is None:
            raise ValueError("sparse call requires original full img_shapes")
        if full_img_shapes != img_shapes:
            raise ValueError("full img_shapes do not match original image context")
        if (
            active_indices.ndim != 1
            or active_indices.dtype != torch.long
            or active_indices.numel() == 0
        ):
            raise ValueError("active indices must be a nonempty int64 vector")
        if generated_tokens is None or not 0 < generated_tokens <= image_tokens:
            raise ValueError("sparse call requires the original generated token count")
        if not prepared:
            if bool((active_indices < 0).any()) or bool(
                (active_indices >= generated_tokens).any()
            ):
                raise ValueError(
                    "active indices exceed original generated image tokens"
                )
            if torch.unique(active_indices).numel() != active_indices.numel():
                raise ValueError("active indices must be unique")
        if kwargs["hidden_states"].shape[1] != active_indices.numel():
            raise ValueError("compact input must contain active generated rows only")
        if branch in cache.layouts and cache.layouts[branch] != layout:
            raise ValueError(
                "cached Qwen native grid/partition/text/dtype/device layout differs from dense context"
            )
        for layer in range(len(model.transformer_blocks)):
            state = cache.require(branch, layer)
            expected_images = image_tokens + (
                text_count if cache.mode == "ready" else 0
            )
            if (
                state.image_tokens != image_tokens
                or state.generated_tokens != generated_tokens
                or state.text_tokens != text_count
                or state.key.shape[1] != expected_images
            ):
                raise ValueError(
                    "cached Qwen context length differs from original full generated+reference tokens"
                )
    with _BoundQwenAttention(model, cache, branch) as wrappers:
        partition = getattr(model, "_sparseedit_partition", None)
        if partition is not None:
            cache.local_aux.begin()
        if mode == "sparse":
            # The native full table supplies both original absolute image rows
            # and the text rows whose starting position depends on image shapes.
            full_rope = model.pos_embed(
                img_shapes,
                max_txt_seq_len=text_count,
                device=kwargs["hidden_states"].device,
            )
            query_rope = (
                full_rope[0].index_select(0, active_indices.to(full_rope[0].device)),
                full_rope[1],
            )
            cache.last_query_rope = query_rope
        attention_kwargs = dict(kwargs.get("attention_kwargs") or {})
        if text_mask is not None:
            mask = cache.full_masks.get(branch)
            if (
                mask is None
                or mask.shape != (1, 1, 1, text_count + image_tokens)
                or mask.device != text_mask.device
            ):
                mask = torch.ones(
                    (1, 1, 1, text_count + image_tokens),
                    dtype=torch.bool,
                    device=text_mask.device,
                )
                cache.full_masks[branch] = mask
            mask[:, 0, 0, :text_count].copy_(text_mask)
            attention_kwargs["attention_mask"] = mask
        elif branch in cache.full_masks and not prepared:
            # An absent mask binds all text positions, including after a masked
            # call. Keep graph-visible storage stable across bindings.
            cache.full_masks[branch].fill_(True)
            attention_kwargs["attention_mask"] = cache.full_masks[branch]
        kwargs["attention_kwargs"] = attention_kwargs
        # Native forward would overwrite a supplied full mask with a compact
        # image mask. Text-mask normalization/RoPE length has already occurred.
        kwargs["encoder_hidden_states_mask"] = None
        if prepare_only:
            return kwargs
        for wrapper in wrappers:
            wrapper.dense = mode == "dense"
            wrapper.image_tokens = image_tokens
            wrapper.generated_tokens = generated_tokens
            device = partition.layer_device(wrapper.layer) if partition is not None else hidden.device
            if partition is not None:
                wrapper.local_attention_mask = cache.local_aux.local(
                    attention_kwargs.get("attention_mask"), device, slot=f"{branch}/mask")
            if mode == "sparse":
                wrapper.configure(
                    image_tokens=image_tokens,
                    indices=cache.local_aux.local(active_indices, device, slot=f"{branch}/indices"),
                    full_image_freqs=cache.local_aux.local(full_rope[0], device, slot=f"{branch}/full_rope"),
                    query_image_freqs=cache.local_aux.local(query_rope[0], device, slot=f"{branch}/query_rope"),
                    text_freqs=cache.local_aux.local(full_rope[1], device, slot=f"{branch}/text_rope"),
                )
        result = model(**kwargs) if native_call is None else native_call(**kwargs)
    output = result[0] if isinstance(result, tuple) else result.sample
    if output.ndim != 3:
        raise RuntimeError(f"native Qwen returned non-3D output {tuple(output.shape)}")
    return output if output_generated_tokens is None else output[:, :generated_tokens]


__all__ = [
    "QwenLayerKV",
    "QwenSparseEditConfig",
    "QwenSparseKVCache",
    "SparseQwenAttention",
    "qwen_sparse_model_call",
]
