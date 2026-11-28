# SPDX-License-Identifier: Apache-2.0
"""Single-request FLUX cache with BF16 mutable rows and online INT8 split-KV attention."""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _encode_rows(
    x,
    y,
    scales,
    n_rows,
    n_tokens: tl.constexpr,
    heads: tl.constexpr,
    xb: tl.constexpr,
    xs: tl.constexpr,
    xh: tl.constexpr,
    dim: tl.constexpr,
    rows: tl.constexpr,
):
    row = tl.program_id(0) * rows + tl.arange(0, rows)
    feature = tl.arange(0, dim)
    token = (row // heads) % n_tokens
    batch = row // (heads * n_tokens)
    head = row % heads
    x_base = batch * xb + token * xs + head * xh
    loaded = tl.load(x + x_base[:, None] + feature[None, :], row[:, None] < n_rows, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(loaded), 1) / 127.0, 2.0**-24)
    scale = scale.to(tl.float16).to(tl.float32)
    quantized = tl.clamp(tl.extra.cuda.libdevice.rint(loaded / scale[:, None]), -127.0, 127.0)
    tl.store(y + row[:, None] * dim + feature[None, :], quantized.to(tl.int8), row[:, None] < n_rows)
    tl.store(scales + row, scale, row < n_rows)


@triton.jit
def _attention_segment(
    q,
    key,
    value,
    key_scale,
    value_scale,
    running_max,
    normalizer,
    accumulator,
    batch,
    head,
    scale: tl.constexpr,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    tile: tl.constexpr,
    quantized: tl.constexpr,
    scales_at_dot: tl.constexpr,
):
    feature = tl.arange(0, dim)
    for start in range(tl.cdiv(tokens, tile)):
        row = start * tile + tl.arange(0, tile)
        base = (batch * tokens + row) * heads + head
        offset = base[:, None] * dim + feature[None, :]
        k = tl.load(key + offset, row[:, None] < tokens, 0)
        v = tl.load(value + offset, row[:, None] < tokens, 0)
        if quantized:
            k_scale = tl.load(key_scale + base, row < tokens, 1).to(tl.float32)
            v_scale = tl.load(value_scale + base, row < tokens, 1).to(tl.float32)
            if scales_at_dot:
                k = k.to(tl.bfloat16)
                v = v.to(tl.bfloat16)
            else:
                k = (k.to(tl.float32) * k_scale[:, None]).to(tl.bfloat16)
                v = (v.to(tl.float32) * v_scale[:, None]).to(tl.bfloat16)

        score = tl.dot(q, tl.trans(k)) * scale
        if quantized and scales_at_dot:
            score *= k_scale[None, :]
        score = tl.where(row[None, :] < tokens, score, float("-inf"))
        next_max = tl.maximum(running_max, tl.max(score, 1))
        correction = tl.exp2(running_max - next_max)
        probability = tl.exp2(score - next_max[:, None])
        normalizer = normalizer * correction + tl.sum(probability, 1)
        if quantized and scales_at_dot:
            probability *= v_scale[None, :]
        accumulator = tl.dot(probability.to(tl.bfloat16), v, accumulator * correction[:, None])
        running_max = next_max
    return running_max, normalizer, accumulator


def _encode_into(source: torch.Tensor, output: torch.Tensor, scales: torch.Tensor) -> None:
    if source.is_cuda:
        batch, tokens, heads, dim = source.shape
        n_rows = batch * tokens * heads
        _encode_rows[(triton.cdiv(n_rows, 8),)](
            source,
            output,
            scales,
            n_rows,
            tokens,
            heads,
            *source.stride()[:3],
            dim,
            8,
            num_warps=4,
        )
        return
    promoted = source.float()
    scale = (promoted.abs().amax(dim=-1) / 127.0).clamp_min(2.0**-24).half()
    scales.copy_(scale)
    output.copy_((promoted / scale.float().unsqueeze(-1)).round().clamp(-127, 127))


from .specialized_cache import SpecializedKVCache, SpecializedCacheError, _LayerStorage, storage_pointers


@triton.jit
def _split_attention(
    query, prefix_k, prefix_v, ref_k, ref_v, ref_ks, ref_vs, partial, lse,
    q_tokens: tl.constexpr, heads: tl.constexpr, dim: tl.constexpr,
    qb: tl.constexpr, qs: tl.constexpr, qh: tl.constexpr,
    prefix_tokens: tl.constexpr, ref_tokens: tl.constexpr, scale: tl.constexpr,
    q_tile: tl.constexpr, kv_tile: tl.constexpr, plane_rows: tl.constexpr,
):
    q_row = tl.program_id(0) * q_tile + tl.arange(0, q_tile)
    batch = tl.program_id(1) // heads
    head = tl.program_id(1) % heads
    plane = tl.program_id(2)
    channels = tl.arange(0, dim)
    query_offset = batch * qb + q_row * qs + head * qh
    q = tl.load(query + query_offset[:, None] + channels[None, :], q_row[:, None] < q_tokens, 0)
    maximum = tl.full((q_tile,), float("-inf"), tl.float32)
    norm = tl.full((q_tile,), 0.0, tl.float32)
    accum = tl.full((q_tile, dim), 0.0, tl.float32)
    if plane == 0:
        maximum, norm, accum = _attention_segment(
            q, prefix_k, prefix_v, ref_ks, ref_vs, maximum, norm, accum,
            batch, head, scale, prefix_tokens, heads, dim, kv_tile, False, False,
        )
    else:
        maximum, norm, accum = _attention_segment(
            q, ref_k, ref_v, ref_ks, ref_vs, maximum, norm, accum,
            batch, head, scale, ref_tokens, heads, dim, kv_tile, True, True,
        )
    row = plane * plane_rows + (batch * q_tokens + q_row) * heads + head
    tl.store(partial + row[:, None] * dim + channels[None, :], accum / norm[:, None],
             q_row[:, None] < q_tokens)
    tl.store(lse + row, maximum + tl.log2(norm), q_row < q_tokens)


@triton.jit
def _merge_splits(partial, lse, output, rows: tl.constexpr, dim: tl.constexpr, tile: tl.constexpr):
    row = tl.program_id(0) * tile + tl.arange(0, tile)
    channel = tl.arange(0, dim)
    base = row[:, None] * dim + channel[None, :]
    left_lse = tl.load(lse + row, row < rows, 0)
    right_lse = tl.load(lse + rows + row, row < rows, 0)
    biggest = tl.maximum(left_lse, right_lse)
    left_weight = tl.exp2(left_lse - biggest)
    right_weight = tl.exp2(right_lse - biggest)
    left = tl.load(partial + base, row[:, None] < rows, 0).to(tl.float32)
    right = tl.load(partial + rows * dim + base, row[:, None] < rows, 0).to(tl.float32)
    result = (left * left_weight[:, None] + right * right_weight[:, None]) / (
        left_weight + right_weight
    )[:, None]
    tl.store(output + base, result, row[:, None] < rows)


@dataclass
class ReferenceInt8KV:
    key: torch.Tensor
    value: torch.Tensor
    reference_key: torch.Tensor
    reference_value: torch.Tensor
    reference_key_scale: torch.Tensor
    reference_value_scale: torch.Tensor

    @classmethod
    def allocate(cls, prefix_shape, reference_shape, device):
        return cls(
            torch.empty(prefix_shape, dtype=torch.bfloat16, device=device),
            torch.empty(prefix_shape, dtype=torch.bfloat16, device=device),
            torch.empty(reference_shape, dtype=torch.int8, device=device),
            torch.empty(reference_shape, dtype=torch.int8, device=device),
            torch.empty(reference_shape[:-1], dtype=torch.float16, device=device),
            torch.empty(reference_shape[:-1], dtype=torch.float16, device=device),
        )

    @property
    def storage(self):
        return (self.key, self.value, self.reference_key, self.reference_value,
                self.reference_key_scale, self.reference_value_scale)

    def attend(self, query):
        if not query.is_cuda or query.dtype != torch.bfloat16 or query.stride(-1) != 1:
            raise ValueError("split-KV attention requires a CUDA BF16 query with contiguous features")
        batch, rows, heads, dim = query.shape
        if batch != 1 or (heads, dim) != self.key.shape[2:]:
            raise ValueError("split-KV attention expects one FLUX CFG branch with matching heads")
        q_tile, kv_tile = 64, 64
        plane_rows = batch * rows * heads
        # Scratch scales with the short query, never with the long BF16 KV sequence.
        partial = torch.empty((2, plane_rows, dim), device=query.device, dtype=query.dtype)
        lse = torch.empty((2, plane_rows), device=query.device, dtype=torch.float32)
        output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
        _split_attention[(triton.cdiv(rows, q_tile), batch * heads, 2)](
            query, self.key, self.value, self.reference_key, self.reference_value,
            self.reference_key_scale, self.reference_value_scale, partial, lse,
            rows, heads, dim, *query.stride()[:3], self.key.shape[1], self.reference_key.shape[1],
            math.log2(math.e) / math.sqrt(dim), q_tile, kv_tile, plane_rows,
            num_warps=4, num_stages=2,
        )
        _merge_splits[(triton.cdiv(plane_rows, 8),)](partial, lse, output, plane_rows, dim, 8, num_warps=4)
        return output


class SegmentedJointCache(SpecializedKVCache):
    def __post_init__(self):
        super().__post_init__()
        if (self.mode, self.submission, self.batch_size, self.dtype) != ("rotated", "torch", 1, torch.bfloat16):
            raise SpecializedCacheError("segmented KV requires batch-one rotated BF16 CFG branches")
        if self.shared_pool is not None or self.head_dim not in (16, 32, 64, 128):
            raise SpecializedCacheError("unsupported segmented KV layout")
        self.segmented_layers = {}

    def store_full_joint(self, stream, layer_index, rotated_image_key, image_value, rotated_text_key, text_value):
        self._validate_input(rotated_image_key, image_value, rows=self.full_image_token_count, stream=stream)
        self._validate_input(rotated_text_key, text_value, rows=self.text_token_count, stream=f"{stream} text")
        layer = stream, layer_index
        pack = self.segmented_layers.get(layer)
        t, g = self.text_token_count, self.generated_token_count
        if pack is None:
            pack = ReferenceInt8KV.allocate(
                (1, t + g, self.num_kv_heads, self.head_dim),
                (1, self.reference_token_count, self.num_kv_heads, self.head_dim),
                self.device,
            )
            self.segmented_layers[layer] = pack
            self.layers[layer] = _LayerStorage(pack.key, pack.value, storage_pointers(pack.key, pack.value))
        pack.key[:, :t].copy_(rotated_text_key)
        pack.value[:, :t].copy_(text_value)
        pack.key[:, t:].copy_(rotated_image_key[:, :g])
        pack.value[:, t:].copy_(image_value[:, :g])
        _encode_into(rotated_image_key[:, g:], pack.reference_key, pack.reference_key_scale)
        _encode_into(image_value[:, g:], pack.reference_value, pack.reference_value_scale)
        self.layers[layer].writes += 1
        self.record_submission(self.full_image_token_count, backend="int8_reference_storage")
        # Dense calls consume fresh BF16 K/V. No packed reference is decoded.
        return torch.cat((rotated_text_key, rotated_image_key), dim=1), torch.cat((text_value, image_value), dim=1)

    def attend_sparse(self, layer, query):
        self.backends.add("triton_split_kv_online_int8")
        return self.segmented_layers[layer].attend(query)

    def storage_tensors(self):
        for pack in self.segmented_layers.values():
            yield from pack.storage

    def clear(self):
        super().clear()
        self.segmented_layers.clear()
