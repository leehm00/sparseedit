# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from cache_dit import ForwardPattern
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_npu_available
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig
from vllm_omni.diffusion.models.flux import direct_sparse_ops
from vllm_omni.diffusion.models.flux.acceleration import (
    FluxSparseAttentionContext,
    FluxStream,
    FluxToCaState,
    gather_tokens,
    scatter_tokens_,
)
from vllm_omni.diffusion.models.flux.specialized_attention import resolve_specialized_kv
from vllm_omni.diffusion.models.flux.specialized_layer import (
    adaptive_norm,
    gated_residual,
    modulate,
    prepare_layer_rotary,
)
from vllm_omni.quantization.component_config import safe_quant_config

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers.adalayernorm import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rope_to_qk

logger = init_logger(__name__)


def _block_attention_kwargs(
    joint_attention_kwargs: dict[str, Any] | None,
    *,
    stream: FluxStream,
    layer_index: int,
    global_layer_index: int,
) -> dict[str, Any] | None:
    if joint_attention_kwargs is None:
        return None
    context = joint_attention_kwargs.get("_flux_acceleration_context")
    if isinstance(context, FluxSparseAttentionContext):
        context.set_layer(stream, layer_index)
        return joint_attention_kwargs
    if isinstance(context, FluxToCaState):
        result = dict(joint_attention_kwargs)
        result["_flux_acceleration_layer"] = layer_index
        result["_flux_acceleration_global_layer"] = global_layer_index
        return result
    return joint_attention_kwargs


def _prepared_projection(context, module):
    if isinstance(context, FluxSparseAttentionContext) and context.condition_view is not None:
        return context.condition_view.projected[id(module)]
    return None


def _forward_rotary(model, txt_ids, img_ids, context, dtype):
    if isinstance(context, FluxSparseAttentionContext) and context.request_position_view is not None:
        view = context.request_position_view
        context.full_rotary_emb = view.full
        context.layer_kv_positions = view.positions
        context.layer_image_rotary = view.image
        return view.query
    ids = torch.cat((txt_ids, img_ids), dim=0)
    if is_torch_npu_available():
        cos, sin = model.pos_embed(ids.cpu())
        query = (cos.npu(), sin.npu())
    else:
        query = model.pos_embed(ids)
    if isinstance(context, FluxSparseAttentionContext):
        full_ids = torch.cat((txt_ids, context.full_img_ids), dim=0)
        context.full_rotary_emb = model.pos_embed(full_ids)
        if context.layer_fusion and context.layer_positions:
            context.full_rotary_emb, query = prepare_layer_rotary(context, query, context.full_rotary_emb, dtype)
    return query


def _run_single_blocks(model, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs):
    context = (joint_attention_kwargs or {}).get("_flux_acceleration_context")
    joint_mode = isinstance(context, FluxSparseAttentionContext) and context.layer_fusion and context.layer_joint
    text_tokens = encoder_hidden_states.shape[1]
    joint = torch.cat((encoder_hidden_states, hidden_states), dim=1) if joint_mode else None
    for index, block in enumerate(model.single_transformer_blocks):
        layer_kwargs = _block_attention_kwargs(
            joint_attention_kwargs,
            stream="single",
            layer_index=index,
            global_layer_index=len(model.transformer_blocks) + index,
        )
        if joint_mode:
            joint = block.forward_joint(
                joint, temb, image_rotary_emb=image_rotary_emb, joint_attention_kwargs=layer_kwargs
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=layer_kwargs,
            )
    if joint_mode:
        return joint[:, :text_tokens], joint[:, text_tokens:]
    return encoder_hidden_states, hidden_states


class ColumnParallelApproxGELU(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        approximate: str,
        bias: bool = True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim_in,
            dim_out,
            bias=bias,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return F.gelu(x, approximate=self.approximate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        activation_fn: str = "gelu-approximate",
        inner_dim: int | None = None,
        bias: bool = True,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert activation_fn == "gelu-approximate", "Only gelu-approximate is supported."

        inner_dim = inner_dim or int(dim * mult)
        dim_out = dim_out or dim

        layers: list[nn.Module] = [
            ColumnParallelApproxGELU(
                dim, inner_dim, approximate="tanh", bias=bias, quant_config=quant_config, prefix=f"{prefix}.net.0"
            ),
            nn.Identity(),  # placeholder for weight loading
            RowParallelLinear(
                inner_dim,
                dim_out,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.net.2",
            ),
        ]

        self.net = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class FluxAttention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: bool | None = None,
        pre_only: bool = False,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.to_qkv",
        )

        if not self.pre_only:
            self.to_out = nn.ModuleList(
                [
                    RowParallelLinear(
                        self.inner_dim,
                        self.out_dim,
                        bias=out_bias,
                        input_is_parallel=True,
                        return_bias=False,
                        quant_config=quant_config,
                        prefix=f"{prefix}.to_out.0",
                    ),
                    nn.Dropout(dropout),
                ]
            )

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)

            self.add_kv_proj = QKVParallelLinear(
                hidden_size=self.added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                bias=added_proj_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.add_kv_proj",
            )
            self.to_add_out = RowParallelLinear(
                self.inner_dim,
                query_dim,
                bias=out_bias,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.to_add_out",
            )

        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        acceleration_context = kwargs.pop("_flux_acceleration_context", None)
        # Ensure contiguous for FP8 quantized linear layers
        hidden_states = hidden_states.contiguous()
        direct = getattr(acceleration_context, "direct_ops", "off")
        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        if direct != "off":
            query, key, value = direct_sparse_ops.project_qkv(self.to_qkv, hidden_states, (q_size, kv_size, kv_size))
        else:
            qkv, _ = self.to_qkv(hidden_states)
            query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (self.to_qkv.num_heads, -1))
        key = key.unflatten(-1, (self.to_qkv.num_kv_heads, -1))
        value = value.unflatten(-1, (self.to_qkv.num_kv_heads, -1))

        normalize = direct_sparse_ops.rms_norm if direct == "eager" else lambda module, x: module(x)
        query = normalize(self.norm_q, query)
        key = normalize(self.norm_k, key)

        encoder_query = encoder_key = encoder_value = None
        if self.added_kv_proj_dim is not None:
            encoder_hidden_states = encoder_hidden_states.contiguous()
            add_q_size = self.add_kv_proj.num_heads * self.head_dim
            add_kv_size = self.add_kv_proj.num_kv_heads * self.head_dim
            if direct != "off":
                encoder_query, encoder_key, encoder_value = direct_sparse_ops.project_qkv(
                    self.add_kv_proj, encoder_hidden_states, (add_q_size, add_kv_size, add_kv_size)
                )
            else:
                encoder_qkv, _ = self.add_kv_proj(encoder_hidden_states)
                encoder_query, encoder_key, encoder_value = encoder_qkv.split(
                    [add_q_size, add_kv_size, add_kv_size], dim=-1
                )

            encoder_query = encoder_query.unflatten(-1, (self.add_kv_proj.num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.add_kv_proj.num_kv_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.add_kv_proj.num_kv_heads, -1))

            encoder_query = normalize(self.norm_added_q, encoder_query)
            encoder_key = normalize(self.norm_added_k, encoder_key)

        specialized_result: tuple[torch.Tensor, torch.Tensor, bool] | None = None
        if isinstance(acceleration_context, FluxSparseAttentionContext):
            specialized_result = resolve_specialized_kv(
                key=key,
                value=value,
                encoder_key=encoder_key,
                encoder_value=encoder_value,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                context=acceleration_context,
                rope=self.rope,
            )

        if specialized_result is None:
            if isinstance(acceleration_context, FluxSparseAttentionContext):
                key, value = self._resolve_sparse_image_kv(
                    key,
                    value,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    context=acceleration_context,
                )

            if encoder_hidden_states is not None:
                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)
            key_already_rotated = False
        else:
            key, value, key_already_rotated = specialized_result
            if encoder_hidden_states is not None:
                # Query/text output paths remain unchanged; only K/V already
                # contain the text segment from the stable joint cache.
                query = torch.cat([encoder_query, query], dim=1)

        rotate = direct_sparse_ops.rotary if direct == "eager" else self.rope
        if isinstance(acceleration_context, FluxSparseAttentionContext) and acceleration_context.full_rotary_emb:
            if image_rotary_emb is not None:
                query_cos, query_sin = image_rotary_emb
                query = rotate(query, query_cos.to(query.dtype), query_sin.to(query.dtype))
            if not key_already_rotated:
                key_cos, key_sin = acceleration_context.full_rotary_emb
                key = rotate(key, key_cos.to(key.dtype), key_sin.to(key.dtype))
        else:
            query, key = apply_rope_to_qk(self.rope, query, key, image_rotary_emb)  # [S, D/2]

        attn_metadata = None
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_metadata = AttentionMetadata(attn_mask=attention_mask)

        hidden_states = self.attn(
            query,
            key,
            value,
            attn_metadata,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            # Contiguous for FP8 quantization in RowParallelLinear
            hidden_states = self.to_out[0](hidden_states.contiguous())
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states.contiguous())

            return hidden_states, encoder_hidden_states
        else:
            if get_tensor_model_parallel_world_size() > 1:
                hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=-1)
            return hidden_states

    @staticmethod
    def _resolve_sparse_image_kv(
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        context: FluxSparseAttentionContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stream, layer_index = context.resolve_layer()
        if encoder_hidden_states is None:
            active_image_tokens = context.image_indices.shape[1]
            text_tokens = hidden_states.shape[1] - active_image_tokens
            if text_tokens < 0:
                raise ValueError("FLUX sparse single-stream image token count exceeds the sequence length")
            text_key, image_key = key[:, :text_tokens], key[:, text_tokens:]
            text_value, image_value = value[:, :text_tokens], value[:, text_tokens:]
        else:
            text_key = text_value = None
            image_key, image_value = key, value

        if context.kv_action == "none":
            full_image_key, full_image_value = image_key, image_value
        elif context.kv_action == "store_full":
            if image_key.shape[1] != context.full_image_token_count:
                raise ValueError(
                    "FLUX sparse full-cache step must receive every generated and reference image token: "
                    f"got {image_key.shape[1]}, expected {context.full_image_token_count}"
                )
            context.kv_cache.store_full(stream, layer_index, image_key, image_value)
            full_image_key, full_image_value = image_key, image_value
        elif context.method == "spotedit":
            full_image_key, full_image_value = context.kv_cache.spotedit_fuse(
                stream,
                layer_index,
                image_key,
                image_value,
                context.image_indices,
                generated_token_count=context.generated_token_count,
                reference_token_count=context.reference_token_count,
                alpha=context.spotedit_alpha,
            )
        else:
            full_image_key, full_image_value = context.kv_cache.update_active(
                stream,
                layer_index,
                image_key,
                image_value,
                context.image_indices,
            )

        if text_key is None:
            return full_image_key, full_image_value
        return torch.cat([text_key, full_image_key], dim=1), torch.cat([text_value, full_image_value], dim=1)


class FluxTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim, quant_config=quant_config, prefix=f"{prefix}.norm1")
        self.norm1_context = AdaLayerNormZero(dim, quant_config=quant_config, prefix=f"{prefix}.norm1_context")

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config, prefix=f"{prefix}.ff")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config, prefix=f"{prefix}.ff_context")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acceleration_context = (joint_attention_kwargs or {}).get("_flux_acceleration_context")
        if isinstance(acceleration_context, FluxToCaState):
            return self._forward_toca(
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                state=acceleration_context,
            )

        fuse_layer = (
            isinstance(acceleration_context, FluxSparseAttentionContext)
            and acceleration_context.layer_fusion
            and acceleration_context.layer_elementwise
        )
        if fuse_layer:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaptive_norm(
                self.norm1,
                hidden_states,
                temb,
                kind="zero",
                condition=acceleration_context.layer_condition,
                projected=_prepared_projection(acceleration_context, self.norm1),
            )
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = adaptive_norm(
                self.norm1_context,
                encoder_hidden_states,
                temb,
                kind="zero",
                condition=acceleration_context.layer_condition,
                projected=_prepared_projection(acceleration_context, self.norm1_context),
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )
        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        if fuse_layer:
            hidden_states = gated_residual(hidden_states, attn_output, gate_msa)
        else:
            attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            modulate(norm_hidden_states, scale_mlp, shift_mlp)
            if fuse_layer
            else norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )

        ff_output = self.ff(norm_hidden_states)
        if fuse_layer:
            hidden_states = gated_residual(hidden_states, ff_output, gate_mlp)
        else:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        if fuse_layer:
            encoder_hidden_states = gated_residual(encoder_hidden_states, context_attn_output, c_gate_msa)
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            modulate(norm_encoder_hidden_states, c_scale_mlp, c_shift_mlp)
            if fuse_layer
            else norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        if fuse_layer:
            encoder_hidden_states = gated_residual(
                encoder_hidden_states,
                context_ff_output,
                c_gate_mlp,
                clip=encoder_hidden_states.dtype == torch.float16,
            )
        else:
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if not fuse_layer and encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def _forward_toca(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        *,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        joint_attention_kwargs: dict[str, Any] | None,
        state: FluxToCaState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_kwargs = joint_attention_kwargs or {}
        step_index = int(layer_kwargs["_flux_acceleration_step"])
        layer_index = int(layer_kwargs["_flux_acceleration_layer"])
        global_layer_index = int(layer_kwargs["_flux_acceleration_global_layer"])
        cache_key = ("dual", layer_index)
        full_compute = state.is_full_step(step_index) or not state.has_layer("dual", layer_index)

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        attention_kwargs = {
            key: value for key, value in layer_kwargs.items() if not key.startswith("_flux_acceleration_")
        }
        if full_compute:
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )
            state.attention_outputs[cache_key] = (
                attn_output.detach(),
                context_attn_output.detach(),
            )
        else:
            attn_output, cached_context = state.attention_outputs[cache_key]
            if cached_context is None:
                raise RuntimeError("FLUX ToCa dual-stream text attention cache is missing")
            context_attn_output = cached_context

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if full_compute:
            ff_output = self.ff(norm_hidden_states)
            state.ffn_outputs[cache_key] = ff_output.detach()
            state.reset_cache_frequency(
                global_layer_index,
                "mlp",
                batch_size=hidden_states.shape[0],
                device=hidden_states.device,
            )
            fresh_count = state.generated_token_count
            fresh_ratio = 1.0
        else:
            fresh_indices, fresh_ratio = state.select_generated_indices(
                hidden_states,
                step_index=step_index,
                global_layer_index=global_layer_index,
                module="mlp",
            )
            reference_indices = torch.arange(
                state.generated_token_count,
                state.full_image_token_count,
                device=hidden_states.device,
                dtype=torch.long,
            ).unsqueeze(0)
            recompute_indices = torch.cat([fresh_indices, reference_indices], dim=1)
            cached_ff = state.ffn_outputs[cache_key].clone()
            if recompute_indices.shape[1] > 0:
                fresh_ff = self.ff(gather_tokens(norm_hidden_states, recompute_indices))
                scatter_tokens_(cached_ff, recompute_indices, fresh_ff)
            ff_output = cached_ff
            state.ffn_outputs[cache_key] = cached_ff.detach()
            fresh_count = fresh_indices.shape[1]
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        state.diagnostics.append(
            {
                "step": step_index,
                "stream": "dual",
                "layer": layer_index,
                "execution": "full" if full_compute else "toca",
                "fresh_generated_tokens": int(fresh_count),
                "fresh_ratio": float(fresh_ratio),
                "reference_tokens_recomputed": state.reference_token_count,
            }
        )
        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        quant_config: "QuantizationConfig | None" = None,
        prefix: str = "",
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim, quant_config=safe_quant_config(quant_config), prefix=f"{prefix}.norm")
        self.proj_mlp = ReplicatedLinear(
            dim,
            self.mlp_hidden_dim,
            bias=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj_mlp",
        )
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = ReplicatedLinear(
            dim + self.mlp_hidden_dim,
            dim,
            bias=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj_out",
        )

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acceleration_context = (joint_attention_kwargs or {}).get("_flux_acceleration_context")
        if isinstance(acceleration_context, FluxToCaState):
            return self._forward_toca(
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                state=acceleration_context,
            )

        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        hidden_states = self.forward_joint(
            hidden_states,
            temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states

    def forward_joint(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Consume joint text/image rows so consecutive single blocks share layout."""
        acceleration_context = (joint_attention_kwargs or {}).get("_flux_acceleration_context")
        if isinstance(acceleration_context, FluxToCaState):
            raise ValueError("joint single-stream execution requires the full-compute or sparse runtime")

        residual = hidden_states
        fuse_layer = (
            isinstance(acceleration_context, FluxSparseAttentionContext)
            and acceleration_context.layer_fusion
            and acceleration_context.layer_elementwise
        )
        norm_hidden_states, gate = (
            adaptive_norm(
                self.norm,
                hidden_states,
                temb,
                kind="single",
                condition=acceleration_context.layer_condition,
                projected=_prepared_projection(acceleration_context, self.norm),
            )
            if fuse_layer
            else self.norm(hidden_states, emb=temb)
        )
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        projected = self.proj_out(hidden_states)
        if fuse_layer:
            hidden_states = gated_residual(residual, projected, gate, clip=projected.dtype == torch.float16)
        else:
            hidden_states = residual + gate.unsqueeze(1) * projected
        if not fuse_layer and hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

    def _forward_toca(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        *,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        joint_attention_kwargs: dict[str, Any] | None,
        state: FluxToCaState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_kwargs = joint_attention_kwargs or {}
        step_index = int(layer_kwargs["_flux_acceleration_step"])
        layer_index = int(layer_kwargs["_flux_acceleration_layer"])
        global_layer_index = int(layer_kwargs["_flux_acceleration_global_layer"])
        cache_key = ("single", layer_index)
        full_compute = state.is_full_step(step_index) or not state.has_layer("single", layer_index)
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        attention_kwargs = {
            key: value for key, value in layer_kwargs.items() if not key.startswith("_flux_acceleration_")
        }

        if full_compute:
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )
            state.attention_outputs[cache_key] = (attn_output.detach(), None)
            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
            projected = self.proj_out(torch.cat([attn_output, mlp_hidden_states], dim=2))
            state.projected_outputs[cache_key] = projected.detach()
            state.reset_cache_frequency(
                global_layer_index,
                "mlp",
                batch_size=hidden_states.shape[0],
                device=hidden_states.device,
            )
            fresh_count = state.generated_token_count
            fresh_ratio = 1.0
        else:
            attn_output, _ = state.attention_outputs[cache_key]
            fresh_indices, fresh_ratio = state.select_generated_indices(
                hidden_states[:, text_seq_len:],
                step_index=step_index,
                global_layer_index=global_layer_index,
                module="mlp",
            )
            text_indices = torch.arange(text_seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0)
            generated_indices = fresh_indices + text_seq_len
            reference_indices = torch.arange(
                text_seq_len + state.generated_token_count,
                text_seq_len + state.full_image_token_count,
                device=hidden_states.device,
                dtype=torch.long,
            ).unsqueeze(0)
            recompute_indices = torch.cat([text_indices, generated_indices, reference_indices], dim=1)
            recompute_norm = gather_tokens(norm_hidden_states, recompute_indices)
            fresh_mlp = self.act_mlp(self.proj_mlp(recompute_norm))
            fresh_attn = gather_tokens(attn_output, recompute_indices)
            fresh_projected = self.proj_out(torch.cat([fresh_attn, fresh_mlp], dim=2))
            projected = state.projected_outputs[cache_key].clone()
            scatter_tokens_(projected, recompute_indices, fresh_projected)
            state.projected_outputs[cache_key] = projected.detach()
            fresh_count = fresh_indices.shape[1]

        hidden_states = residual + gate.unsqueeze(1) * projected
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        state.diagnostics.append(
            {
                "step": step_index,
                "stream": "single",
                "layer": layer_index,
                "execution": "full" if full_compute else "toca",
                "fresh_generated_tokens": int(fresh_count),
                "fresh_ratio": float(fresh_ratio),
                "reference_tokens_recomputed": state.reference_token_count,
            }
        )
        return encoder_hidden_states, hidden_states


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            freqs_cis = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                use_real=False,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(freqs_cis.real)
            sin_out.append(freqs_cis.imag)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class FluxTransformer2DModel(nn.Module):
    """
    The Transformer model introduced in Flux.

    Args:
        od_config (`OmniDiffusionConfig`):
            The configuration for the model.
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `19`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `38`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `4096`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={
            "transformer_blocks": ForwardPattern.Pattern_1,
            "single_transformer_blocks": ForwardPattern.Pattern_1,
        }
    )

    # the small and frequently-repeated block(s) of a model
    # -- typically a transformer layer
    # used for torch compile optimizations
    _repeated_blocks = ["FluxTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return ("transformer_blocks" in name or "single_transformer_blocks" in name) and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "add_kv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig | None = None,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = True,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        theta: float = 10000.0,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        if od_config is not None:
            model_config = od_config.tf_model_config
            num_layers = model_config.num_layers
            self.parallel_config = od_config.parallel_config
        else:
            self.parallel_config = None

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.guidance_embeds = guidance_embeds

        self.pos_embed = FluxPosEmbed(theta=theta, axes_dim=axes_dims_rope)
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    quant_config=safe_quant_config(quant_config),
                    prefix=f"transformer_blocks.{i}",
                )
                for i in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    quant_config=quant_config,
                    prefix=f"single_transformer_blocks.{i}",
                )
                for i in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            quant_config=safe_quant_config(quant_config),
            prefix="norm_out",
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            img_ids: (`torch.Tensor`):
                The position ids for image tokens.
            txt_ids (`torch.Tensor`):
                The position ids for text tokens.
            guidance (`torch.Tensor`):
                Guidance embeddings for guidance-distilled variant of the model.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        hidden_states = self.x_embedder(hidden_states)
        acceleration_context = (joint_attention_kwargs or {}).get("_flux_acceleration_context")
        prepared_condition = (
            acceleration_context.condition_view
            if isinstance(acceleration_context, FluxSparseAttentionContext)
            else None
        )
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype) * 1000

        if guidance is not None:
            guidance = guidance.to(device=hidden_states.device, dtype=hidden_states.dtype) * 1000

        temb = (
            prepared_condition.temb
            if prepared_condition is not None
            else (
                self.time_text_embed(timestep, pooled_projections)
                if guidance is None
                else self.time_text_embed(timestep, guidance, pooled_projections)
            )
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        acceleration_context = (joint_attention_kwargs or {}).get("_flux_acceleration_context")
        if isinstance(acceleration_context, FluxSparseAttentionContext):
            acceleration_context.layer_condition = (
                (prepared_condition.condition if prepared_condition is not None else self.norm_out.silu(temb))
                if acceleration_context.layer_share_condition or prepared_condition is not None
                else None
            )
        image_rotary_emb = _forward_rotary(self, txt_ids, img_ids, acceleration_context, hidden_states.dtype)

        for index_block, block in enumerate(self.transformer_blocks):
            block_attention_kwargs = _block_attention_kwargs(
                joint_attention_kwargs,
                stream="dual",
                layer_index=index_block,
                global_layer_index=index_block,
            )
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=block_attention_kwargs,
            )

        encoder_hidden_states, hidden_states = _run_single_blocks(
            self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
        )

        hidden_states = (
            adaptive_norm(
                self.norm_out,
                hidden_states,
                temb,
                kind="continuous",
                condition=acceleration_context.layer_condition,
                projected=_prepared_projection(acceleration_context, self.norm_out),
            )
            if isinstance(acceleration_context, FluxSparseAttentionContext)
            and acceleration_context.layer_fusion
            and acceleration_context.layer_elementwise
            else self.norm_out(hidden_states, temb)
        )
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self-attn
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            # cross-attn
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())

        # we need to load the buffers for beta and eps (XIELU)
        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if lookup_name not in params_dict and ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")
                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)
        return loaded_params


class FluxKontextTransformer2DModel(FluxTransformer2DModel):
    def __init__(
        self,
        od_config: OmniDiffusionConfig = None,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = True,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        theta: float = 10000.0,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__(
            od_config=od_config,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
            theta=theta,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        hidden_states = self.x_embedder(hidden_states)
        acceleration_context = (joint_attention_kwargs or {}).get("_flux_acceleration_context")
        prepared_condition = (
            acceleration_context.condition_view
            if isinstance(acceleration_context, FluxSparseAttentionContext)
            else None
        )
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        if prepared_condition is not None:
            temb = prepared_condition.temb
        elif guidance is None:
            temb = self.time_text_embed(timestep, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, guidance, pooled_projections)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids is not None and txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch.Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids is not None and img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch.Tensor"
            )
            img_ids = img_ids[0]

        acceleration_context = (joint_attention_kwargs or {}).get("_flux_acceleration_context")
        if isinstance(acceleration_context, FluxSparseAttentionContext):
            acceleration_context.layer_condition = (
                (prepared_condition.condition if prepared_condition is not None else self.norm_out.silu(temb))
                if acceleration_context.layer_share_condition or prepared_condition is not None
                else None
            )
        image_rotary_emb = _forward_rotary(self, txt_ids, img_ids, acceleration_context, hidden_states.dtype)

        for index_block, block in enumerate(self.transformer_blocks):
            block_attention_kwargs = _block_attention_kwargs(
                joint_attention_kwargs,
                stream="dual",
                layer_index=index_block,
                global_layer_index=index_block,
            )
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=block_attention_kwargs,
            )

        encoder_hidden_states, hidden_states = _run_single_blocks(
            self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
        )
        hidden_states = (
            adaptive_norm(
                self.norm_out,
                hidden_states,
                temb,
                kind="continuous",
                condition=acceleration_context.layer_condition,
                projected=_prepared_projection(acceleration_context, self.norm_out),
            )
            if isinstance(acceleration_context, FluxSparseAttentionContext)
            and acceleration_context.layer_fusion
            and acceleration_context.layer_elementwise
            else self.norm_out(hidden_states, temb)
        )
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
