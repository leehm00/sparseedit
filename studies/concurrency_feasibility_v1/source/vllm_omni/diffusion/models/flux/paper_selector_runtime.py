# SPDX-License-Identifier: Apache-2.0
"""Opt-in SparseEdit runtime for the paper velocity-history selector.

The default ``sparseedit_runtime_variant='baseline'`` path is the original
compact-query/full-context runtime: it clones the full image K/V cache, scatters
the active rows, rotates the full joint K with the full-grid RoPE and
concatenates text K/V at every layer.

The specialized variants are strictly opt-in and share the same selector,
budgets, refresh schedule, CFG semantics and scheduler:

``inplace``
    Owned raw image K/V storage; sparse steps scatter in place, refresh copies
    into the same buffers.  No clone, no pointer churn.
``transformed``
    Owned attention-ready joint K/V ``[text | generated | reference]``; text is
    rewritten every forward, image K is rotated only for fresh rows and no full
    image K is re-rotated or concatenated.
``fused``
    ``transformed`` storage with a targeted Triton kernel that rotates active
    normalized K and submits K/V into the joint buffers (declared tolerance;
    PyTorch reference fallback).
``graph``
    ``transformed`` storage plus a request-scoped CUDA graph of the real
    per-branch ``predict_noise``/transformer forward.  Capture happens on the
    first sparse step; the selector/scheduler/velocity history stay outside.

Two independent CFG branches keep separate caches.  Text K/V is recomputed on
every positive and negative forward.  Positions remain global; the compact
query is rotated with the compact position embeddings.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from vllm_omni.diffusion.models.flux.acceleration import (
    FluxSparseAttentionContext,
    FluxSparseKVCache,
    gather_ids,
    gather_tokens,
    scatter_tokens_,
    with_acceleration_context,
)
from vllm_omni.diffusion.models.flux.acceleration_runtime import (
    _apply_callback,
    _positive_kwargs,
)
from vllm_omni.diffusion.models.flux.online_reuse import (
    AnchoredReuseProgram,
    ControllerStep,
    OnlineReuseTrace,
    native_euler_dt_semantics,
    prepare_native_euler_dts,
    validate_controller_replay,
)
from vllm_omni.diffusion.models.flux.paper_selector import (
    paper_local_weight,
    select_paper_mask,
    select_paper_tokens,
    select_random_tokens,
)
from vllm_omni.diffusion.models.flux.sparseedit_primitives import (
    AVDDecision,
    SparseEditAVDConfig,
    SparseEditAVDState,
)
from vllm_omni.diffusion.models.flux.specialized_cache import (
    SpecializedCacheError,
    SpecializedKVCache,
    parse_cache_mode,
)
from vllm_omni.diffusion.models.flux.specialized_graph import (
    BranchSparseGraph,
    GraphCaptureError,
    RequestGraphState,
)
from vllm_omni.diffusion.models.flux.specialized_plan import (
    FixedMaskTrace,
    SparseTraceError,
    SpecializedExecutionPlan,
)
from vllm_omni.diffusion.models.flux.specialized_template import (
    acquire_template,
    bind_template_caches,
    clear_template,
    template_pool_metadata,
)


def _sigma_value(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().to(torch.float32).cpu().item())
    return float(value)


def _index_digest(indices: torch.Tensor) -> str:
    payload = (
        indices.detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous()
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _pack_mask(mask: torch.Tensor) -> str:
    cpu_mask = mask.detach().to(device="cpu", dtype=torch.uint8).numpy()
    packed = np.packbits(cpu_mask, axis=1)
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _weight_for_step(config, *, flow_time: float, warmup_flow_time: float) -> float:
    return paper_local_weight(
        config.sparseedit_paper_mode,
        flow_time=flow_time,
        warmup_flow_time=warmup_flow_time,
        constant_weight=config.sparseedit_paper_constant_weight,
    )


def _transformer_kv_shape(pipeline: Any) -> tuple[int, int]:
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise SpecializedCacheError(
            "specialized FLUX runtime requires pipeline.transformer"
        )
    blocks = getattr(transformer, "transformer_blocks", None)
    singles = getattr(transformer, "single_transformer_blocks", None)
    if not blocks or not singles:
        raise SpecializedCacheError(
            "specialized FLUX runtime requires dual and single transformer blocks"
        )
    shapes = []
    for block in (blocks[0], singles[0]):
        attention = getattr(block, "attn", None)
        if attention is None or not hasattr(attention, "to_qkv"):
            raise SpecializedCacheError(
                "specialized FLUX runtime requires FluxAttention projections"
            )
        shapes.append((int(attention.to_qkv.num_kv_heads), int(attention.head_dim)))
    if shapes[0] != shapes[1]:
        raise SpecializedCacheError(
            f"dual/single stream K/V shapes differ: {shapes[0]} and {shapes[1]}"
        )
    return shapes[0]


def _check_tensor_parallel(pipeline: Any, variant: str) -> None:
    parallel = getattr(getattr(pipeline, "od_config", None), "parallel_config", None)
    if parallel is None:
        return
    if int(getattr(parallel, "tensor_parallel_size", 1)) > 1:
        raise ValueError(
            f"specialized sparseedit variant {variant!r} currently requires tensor_parallel_size=1"
        )


def _mask_from_indices(indices: torch.Tensor, token_count: int) -> torch.Tensor:
    mask = torch.zeros((1, token_count), dtype=torch.bool, device=indices.device)
    mask.scatter_(
        1, indices.reshape(1, -1).to(device=indices.device, dtype=torch.long), True
    )
    return mask


class _PendingStep:
    """GPU-side record materialized after the measured denoising interval."""

    __slots__ = (
        "active_mask",
        "anchor_step",
        "cache_backend",
        "decision",
        "dense",
        "flow_time",
        "graph_replayed",
        "indices",
        "kv_action",
        "local_weight",
        "output_count",
        "phase",
        "priority",
        "step",
        "token_age_before",
        "velocity_history_length",
    )

    def __init__(self, **values: Any) -> None:
        for name in self.__slots__:
            setattr(self, name, values.get(name))


def _timed_event(device: torch.device) -> torch.cuda.Event | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.Event(enable_timing=True)


def _event_elapsed(
    start: torch.cuda.Event | None, end: torch.cuda.Event | None
) -> float | None:
    if start is None or end is None:
        return None
    try:
        return float(start.elapsed_time(end)) / 1000.0
    except RuntimeError:
        return None


def _audit_step_path(audit_dir: str, step: int) -> Path:
    return Path(audit_dir) / f"step_{step:02d}.pt"


def _snapshot_audit_step(
    step: int,
    *,
    latents: torch.Tensor,
    velocity: torch.Tensor,
    indices: torch.Tensor | None,
    positive: torch.Tensor | None,
    negative: torch.Tensor | None,
    active_mask: torch.Tensor | None,
    anchor_velocity: torch.Tensor | None = None,
    anchor_step: int | None = None,
    velocity_history: Sequence[torch.Tensor] = (),
    token_age: torch.Tensor | None = None,
    controller: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Clone one step's audit tensors, staying on their device.

    This runs inside the measured denoising loop, so it must perform no host
    transfer and no file I/O: a ``.cpu()`` call would enqueue a device
    synchronization, and a ``torch.save`` would stall the stream.  Only the
    small per-step tensors are cloned; the persistent K/V caches are never
    copied here (the benchmark path keeps audit disabled entirely).
    """
    payload: dict[str, Any] = {
        "step": int(step),
        "latents": latents.detach().clone(),
        "guided_velocity": velocity.detach().clone(),
    }
    if anchor_velocity is not None:
        payload["anchor_velocity"] = anchor_velocity.detach().clone()
        payload["anchor_step"] = anchor_step
        payload["velocity_history"] = [
            value.detach().clone() for value in velocity_history
        ]
        payload["token_age"] = token_age.detach().clone()
        payload["controller"] = controller
    if indices is not None:
        payload["active_indices"] = indices.detach().clone()
    if active_mask is not None:
        payload["active_mask"] = active_mask.detach().clone()
    if positive is not None:
        payload["positive_compact"] = positive.detach().clone()
    if negative is not None:
        payload["negative_compact"] = negative.detach().clone()
    return payload


def _audit_capture_costs(
    wall_seconds: Sequence[float],
    event_pairs: Sequence[tuple[Any, Any]],
) -> tuple[float, float | None]:
    """Sum per-snapshot capture cost.

    ``wall_seconds`` holds the host-side enqueue time of each in-loop snapshot;
    ``event_pairs`` holds the CUDA event pair recorded around it.  The device
    figure is only reported when *every* pair was actually timed, so a partially
    instrumented run cannot silently under-report.  The event pairs are resolved
    here, i.e. after the denoising end event has been recorded and synchronized,
    because ``Event.elapsed_time`` synchronizes.
    """
    wall_total = float(sum(float(item) for item in wall_seconds))
    if not event_pairs:
        return wall_total, None
    resolved = [_event_elapsed(start, end) for start, end in event_pairs]
    if any(item is None for item in resolved):
        return wall_total, None
    return wall_total, float(sum(item for item in resolved if item is not None))


def _exclude_audit_capture(
    *,
    denoise_device_seconds: float | None,
    denoise_wall_seconds: float,
    audit_capture_device_seconds: float | None,
    audit_capture_wall_seconds: float,
) -> dict[str, Any]:
    """Split the raw denoising interval into reported and audit-capture parts.

    The global denoising event pair and the wall clock both span the in-loop
    audit snapshots, because those snapshots issue device copies on the measured
    stream.  The reported ``denoise_*_seconds`` values therefore subtract the
    summed capture cost, and the raw inclusive totals are kept alongside them so
    the subtraction can be recomputed independently.
    """
    wall_exclusive = max(
        0.0, float(denoise_wall_seconds) - float(audit_capture_wall_seconds)
    )
    if denoise_device_seconds is None or audit_capture_device_seconds is None:
        device_exclusive = denoise_device_seconds
    else:
        device_exclusive = max(
            0.0, float(denoise_device_seconds) - float(audit_capture_device_seconds)
        )
    return {
        "denoise_device_seconds": device_exclusive,
        "denoise_device_seconds_including_audit_capture": denoise_device_seconds,
        "denoise_wall_seconds": wall_exclusive,
        "denoise_wall_seconds_including_audit_capture": float(denoise_wall_seconds),
        "audit_capture_device_seconds": audit_capture_device_seconds,
        "audit_capture_wall_seconds": float(audit_capture_wall_seconds),
    }


def _serialize_audit_payload(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Cast one captured snapshot to CPU ``float32`` for lossless storage."""
    step = int(snapshot["step"])
    payload: dict[str, Any] = {
        "step": step,
        "latents": snapshot["latents"].to(device="cpu", dtype=torch.float32),
        "guided_velocity": snapshot["guided_velocity"].to(
            device="cpu", dtype=torch.float32
        ),
    }
    if snapshot.get("active_indices") is not None:
        payload["active_indices"] = snapshot["active_indices"].to(
            device="cpu", dtype=torch.int64
        )
    if snapshot.get("active_mask") is not None:
        payload["active_mask"] = snapshot["active_mask"].to(
            device="cpu", dtype=torch.bool
        )
    if snapshot.get("positive_compact") is not None:
        payload["positive_compact"] = snapshot["positive_compact"].to(
            device="cpu", dtype=torch.float32
        )
    if snapshot.get("negative_compact") is not None:
        payload["negative_compact"] = snapshot["negative_compact"].to(
            device="cpu", dtype=torch.float32
        )
    if snapshot.get("anchor_velocity") is not None:
        payload.update(
            anchor_velocity=snapshot["anchor_velocity"].cpu(),
            anchor_step=snapshot["anchor_step"],
            velocity_history=[value.cpu() for value in snapshot["velocity_history"]],
            token_age=snapshot["token_age"].cpu(),
            controller=snapshot["controller"],
        )
    return payload


def _flush_audit_snapshots(
    audit_dir: str, snapshots: Sequence[Mapping[str, Any]]
) -> float:
    """Write captured audit snapshots and return the elapsed seconds.

    Called strictly *after* the denoising end events are recorded and the
    device is synchronized, so neither the host copies nor the serialization
    can contaminate ``denoise_device_seconds``.
    """
    started = time.perf_counter()
    for snapshot in snapshots:
        path = _audit_step_path(audit_dir, int(snapshot["step"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_serialize_audit_payload(snapshot), path)
    return time.perf_counter() - started


def _cache_witness_tensors(cache: Any) -> dict[str, Any]:
    """Small lossless witnesses that never copy a full joint cache."""
    witness: dict[str, Any] = {}
    if isinstance(cache, SpecializedKVCache):
        witness["pointer_witness"] = cache.pointer_witness()
        items = sorted(cache.layers.items(), key=lambda item: (item[0][0], item[0][1]))
        for index, ((stream, layer), storage) in enumerate(items[:4]):
            text = cache.text_token_count
            witness[f"{stream}_{layer}_key_head"] = (
                storage.key[:, text : text + 4].detach().clone()
            )
            witness[f"{stream}_{layer}_value_head"] = (
                storage.value[:, text : text + 4].detach().clone()
            )
            if index == 0:
                witness[f"{stream}_{layer}_key_first"] = storage.key[
                    :, : min(4, storage.key.shape[1])
                ].clone()
                witness[f"{stream}_{layer}_value_first"] = storage.value[
                    :, : min(4, storage.value.shape[1])
                ].clone()
    elif isinstance(cache, FluxSparseKVCache):
        layers = sorted(cache.layers.items(), key=lambda item: (item[0][0], item[0][1]))
        witness["layer_count"] = len(cache.layers)
        for stream_layer, (key, value) in layers[:4]:
            stream, layer = stream_layer
            witness[f"{stream}_{layer}_key_first"] = (
                key[:, : min(4, key.shape[1])].detach().clone()
            )
            witness[f"{stream}_{layer}_value_first"] = (
                value[:, : min(4, value.shape[1])].detach().clone()
            )
    return witness


from .request_state import RequestState, StepSnapshot, return_value


def _iter_sparseedit_paper(
    pipeline: Any,
    *,
    config,
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
    latents,
    image_latents,
    latent_ids,
    text_ids,
    negative_text_ids,
    timesteps,
    do_true_cfg: bool,
    guidance,
    true_cfg_scale: float,
    grid_height: int,
    grid_width: int,
    callback_on_step_end=None,
    callback_inputs=None,
    request_state: RequestState | None = None,
):
    """Run the paper selector with an independent K/V cache per CFG branch."""
    context = request_state.context if request_state is not None else None

    def attention_kwargs():
        return context.base_attention_kwargs if context is not None else pipeline.joint_attention_kwargs

    count, reference_count = latents.shape[1], image_latents.shape[1]
    if grid_height * grid_width != count:
        raise ValueError("paper selector grid does not match the generated token count")
    if do_true_cfg and (
        negative_prompt_embeds is None
        or negative_pooled_prompt_embeds is None
        or negative_text_ids is None
    ):
        raise ValueError(
            "paper selector true CFG requires negative prompt embeddings and text IDs"
        )
    if not math.isfinite(float(true_cfg_scale)):
        raise ValueError("true_cfg_scale must be finite")

    variant = str(getattr(config, "sparseedit_runtime_variant", "baseline"))
    cache_layout_override = config.sparseedit_cache_layout
    stage_fusion = bool(getattr(config, "sparseedit_stage_fusion", False))
    layer_fusion = bool(getattr(config, "sparseedit_layer_fusion", False))
    layer_options = (
        bool(config.sparseedit_fuse_elementwise),
        bool(config.sparseedit_share_positions),
        bool(config.sparseedit_share_joint),
    )
    condition_mode = config.sparseedit_condition_plan_mode
    condition_prepare_graph = bool(config.sparseedit_condition_prepare_graph)
    if condition_mode != "off" and callback_on_step_end is not None:
        raise ValueError(
            "condition projection plans require fixed request conditioning; callbacks are unsupported"
        )
    condition_plan = None
    condition_preparation = None
    deferred_condition = condition_mode in {"warmup_overlap", "warmup_serial"}
    condition_setup_wall = 0.0
    condition_setup_start = condition_setup_end = None
    share_condition = bool(config.sparseedit_share_condition)
    position_reuse = bool(config.sparseedit_request_position_reuse)
    shared_position_branches = bool(
        position_reuse and do_true_cfg and torch.equal(text_ids, negative_text_ids)
    )
    position_plan = None
    position_setup_wall = 0.0
    position_setup_start = position_setup_end = None
    stage_backends: set[str] = set()
    if request_state is None and latents.device.type == "cuda" and torch.cuda.is_available():
        # Runs inside the actual model worker once per request, before cache
        # allocation. The Omni parent owns a different CUDA allocator.
        torch.accelerator.reset_peak_memory_stats(latents.device)
    specialized = variant != "baseline"
    online_mode = str(getattr(config, "sparseedit_online_reuse", "off"))
    online_enabled = online_mode != "off"
    if request_state is not None and (online_enabled or callback_on_step_end is not None):
        raise ValueError("Study executor requires frozen replay without online reuse or callbacks")
    if online_enabled and variant != "graph_reuse":
        raise ValueError(
            "online reuse requires a shared specialized graph_reuse context"
        )
    if online_enabled and not do_true_cfg:
        raise ValueError("online reuse admission requires true two-branch CFG")
    avd = None
    online_program = None
    online_trace = None
    anchor_velocity = None
    anchor_step = None
    online_previous_timestep = None
    online_force_dense = False
    if online_enabled:
        avd = SparseEditAVDState(
            SparseEditAVDConfig(
                num_inference_steps=len(timesteps),
                cache_threshold=config.sparseedit_avd_cache_threshold,
                max_consecutive_hits=config.sparseedit_avd_max_reuse,
                force_final_steps=config.sparseedit_avd_force_final_steps,
                gamma=config.sparseedit_avd_gamma,
            )
        )
    if specialized:
        _check_tensor_parallel(pipeline, variant)

    # A captured branch graph binds ``prompt_embeds`` when the closure is built
    # on the first sparse step.  ``_apply_callback`` may replace that tensor
    # after any step, and a replay would keep using the captured one, silently
    # discarding the callback update.  Refuse the combination unless the caller
    # explicitly opted into the labelled eager fallback.
    graph_disabled_reason: str | None = None
    if variant in {"graph", "graph_reuse"} and callback_on_step_end is not None:
        if online_enabled:
            raise ValueError("online reuse does not support prompt-mutating callbacks")
        if not bool(getattr(config, "sparseedit_graph_fallback", False)):
            raise GraphCaptureError(
                "the graph runtime variant cannot be combined with callback_on_step_end: "
                "a step callback may replace prompt_embeds, but a captured graph replays "
                "the embeddings captured on its first sparse step. Drop the callback, or "
                "pass sparseedit_graph_fallback=True to run this request eagerly."
            )
        graph_disabled_reason = "step callback may replace prompt_embeds; request executed eagerly because sparseedit_graph_fallback=True"

    all_generated = torch.arange(count, device=latents.device).unsqueeze(0)
    all_images = torch.arange(count + reference_count, device=latents.device).unsqueeze(
        0
    )
    generated_ids = latent_ids[:count]
    warmup = config.sparseedit_warmup_steps
    refresh_interval = config.sparseedit_refresh_interval
    ratio = config.sparseedit_ratio
    active_count = max(1, min(count, math.floor(count * ratio)))
    sigmas = getattr(pipeline.scheduler, "sigmas", None)
    if sigmas is None or len(sigmas) < len(timesteps) or warmup >= len(sigmas):
        raise ValueError(
            "paper selector requires scheduler sigmas for every denoising step"
        )
    # Flow times are frozen before the timed loop: reading scheduler sigmas
    # inside the loop would force a host sync on every step.
    flow_times = [_sigma_value(sigmas[index]) for index in range(len(timesteps))]
    warmup_flow_time = _sigma_value(sigmas[warmup])
    if not math.isfinite(warmup_flow_time) or warmup_flow_time <= 0.0:
        raise ValueError(
            "scheduler sigma at the first sparse step must be finite and positive"
        )
    for step, flow_time in enumerate(flow_times):
        if step >= warmup and (not math.isfinite(flow_time) or flow_time < 0.0):
            raise ValueError(
                f"scheduler sigma at step {step} must be finite and non-negative"
            )

    controller_times = [
        float(value) for value in timesteps.detach().float().cpu().tolist()
    ]
    online_config = None
    online_masks: dict[int, torch.Tensor] = {}
    provenance_started = time.perf_counter()
    if online_enabled:
        request_digest = hashlib.sha256()
        for value in (
            latents,
            image_latents,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            latent_ids,
            text_ids,
            negative_text_ids,
            guidance,
        ):
            if value is None:
                request_digest.update(b"none")
            else:
                request_digest.update(
                    str((tuple(value.shape), str(value.dtype))).encode()
                )
                request_digest.update(
                    value.detach()
                    .cpu()
                    .contiguous()
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
        online_config = {
            "num_inference_steps": len(timesteps),
            "generated_token_count": count,
            "reference_token_count": reference_count,
            "active_token_count": active_count,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "ratio": float(ratio),
            "warmup_steps": warmup,
            "refresh_interval": refresh_interval,
            "true_cfg": bool(do_true_cfg),
            "true_cfg_scale": float(true_cfg_scale),
            "selector_protocol": "paper",
            "selector_mode": config.sparseedit_paper_mode,
            "selector_constant_weight": float(config.sparseedit_paper_constant_weight),
            "selector_random_seed": config.sparseedit_paper_random_seed,
            "avd_cache_threshold": float(config.sparseedit_avd_cache_threshold),
            "avd_max_reuse": config.sparseedit_avd_max_reuse,
            "avd_force_final_steps": config.sparseedit_avd_force_final_steps,
            "native_euler_dt_semantics": native_euler_dt_semantics(
                pipeline.scheduler, latents
            ),
            "gamma_sha256": hashlib.sha256(
                json.dumps(
                    list(config.sparseedit_avd_gamma), separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "request_sha256": request_digest.hexdigest(),
            "schedule_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "class": type(pipeline.scheduler).__name__,
                        "timesteps": controller_times,
                        "sigmas": [_sigma_value(v) for v in sigmas],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        replay_path = str(config.sparseedit_replay_trace or "")
        if replay_path:
            online_trace = OnlineReuseTrace.load(replay_path)
            validate_controller_replay(
                online_trace,
                num_inference_steps=len(timesteps),
                generated_token_count=count,
                active_token_count=active_count,
                warmup_steps=warmup,
                refresh_interval=refresh_interval,
                gamma=config.sparseedit_avd_gamma,
            )
            if online_trace.config != online_config:
                raise ValueError(
                    "stale online trace mathematical request/config provenance"
                )
    provenance_wall_seconds = time.perf_counter() - provenance_started
    if request_state is None and latents.device.type == "cuda":
        torch.cuda.synchronize(latents.device)
    if online_enabled:
        denoise_wall_start = time.perf_counter()
        denoise_start_event = _timed_event(latents.device)
        if denoise_start_event is not None:
            denoise_start_event.record()
        online_program = AnchoredReuseProgram(latents, latents, online_mode)
        online_dts, online_dt_metadata = prepare_native_euler_dts(
            pipeline.scheduler, latents
        )
        if online_trace is not None:
            online_masks = {
                item.step: torch.tensor(
                    item.indices, dtype=torch.long, device=latents.device
                ).reshape(1, -1)
                for item in online_trace.steps
                if item.action == "sparse"
            }

    replay: FixedMaskTrace | None = None
    replay_path = str(getattr(config, "sparseedit_replay_trace", "") or "")
    if replay_path and not online_enabled:
        replay = FixedMaskTrace.load(replay_path)
        replay.validate(
            num_inference_steps=len(timesteps),
            ratio=ratio,
            warmup_steps=warmup,
            refresh_interval=refresh_interval,
            generated_token_count=count,
            reference_token_count=reference_count,
            grid_height=grid_height,
            grid_width=grid_width,
            active_token_count=active_count,
        )

    text_token_count = int(prompt_embeds.shape[1])
    from .cfg_execution import CFGExecutionPlan, choose_cfg_execution

    cfg_requested = config.sparseedit_cfg_execution
    cfg_compatible = bool(
        do_true_cfg
        and callback_on_step_end is None
        and (cfg_requested == "serial" or torch.equal(text_ids, negative_text_ids))
    )
    cfg_selected, cfg_reason = choose_cfg_execution(
        cfg_requested,
        text_tokens=text_token_count,
        active_tokens=active_count,
        min_query_rows=config.sparseedit_cfg_min_query_rows,
        compatible=cfg_compatible,
    )
    cfg_batched = cfg_selected == "batched"
    cfg_plan = None
    cfg_binding_wall = 0.0
    cfg_bind_start = cfg_bind_end = None
    if do_true_cfg and int(negative_prompt_embeds.shape[1]) != text_token_count:
        raise ValueError(
            "positive and negative text embeddings must share a sequence length"
        )
    num_kv_heads = head_dim = 0
    positive_specialized: SpecializedKVCache | None = None
    negative_specialized: SpecializedKVCache | None = None
    plan: SpecializedExecutionPlan | None = None
    graph_state: RequestGraphState | None = None
    graph_contexts: dict[str, FluxSparseAttentionContext] = {}
    template = None
    template_hit = False
    template_binding_seconds = 0.0
    request_replays_before: dict[str, int] = {}

    if variant != "graph_reuse" or graph_disabled_reason is not None:
        clear_template(pipeline)
    else:
        if attention_kwargs():
            raise ValueError(
                "graph_reuse currently requires empty joint_attention_kwargs"
            )
        binding_started = time.perf_counter()

        def layout(tensor):
            return (
                None
                if tensor is None
                else (tuple(tensor.shape), str(tensor.dtype), str(tensor.device))
            )

        signature = (
            id(pipeline.transformer),
            parse_cache_mode(variant, override=cache_layout_override),
            count,
            reference_count,
            active_count,
            grid_height,
            grid_width,
            layout(latents),
            layout(latent_ids),
            layout(prompt_embeds),
            layout(pooled_prompt_embeds),
            layout(text_ids),
            layout(negative_prompt_embeds),
            layout(negative_pooled_prompt_embeds),
            layout(negative_text_ids),
            layout(guidance),
            do_true_cfg,
            stage_fusion,
            layer_fusion,
            layer_options,
            share_condition,
            "warmup_native_serial" if deferred_condition else condition_mode,
            condition_prepare_graph,
            warmup if deferred_condition else None,
            cfg_selected,
            cfg_requested,
            config.sparseedit_cfg_min_query_rows,
            len(timesteps),
            position_reuse,
            shared_position_branches,
            config.sparseedit_condition_bank_shared,
        )
        template, template_hit = acquire_template(
            pipeline, signature, capacity=config.sparseedit_template_capacity
        )
        try:
            if template.full_img_ids is None:
                template.full_img_ids = latent_ids.detach().clone()
            else:
                template.full_img_ids.copy_(latent_ids)
            latent_ids = template.full_img_ids
            generated_ids = latent_ids[:count]
            positive_tensors = {
                "prompt": prompt_embeds,
                "pooled": pooled_prompt_embeds,
                "ids": text_ids,
            }
            if guidance is not None:
                positive_tensors["guidance"] = guidance
            positive_bound = template.bind_conditioning("positive", positive_tensors)
            prompt_embeds, pooled_prompt_embeds, text_ids = (
                positive_bound["prompt"],
                positive_bound["pooled"],
                positive_bound["ids"],
            )
            if guidance is not None:
                guidance = positive_bound["guidance"]
            if do_true_cfg:
                negative_bound = template.bind_conditioning(
                    "negative",
                    {
                        "prompt": negative_prompt_embeds,
                        "pooled": negative_pooled_prompt_embeds,
                        "ids": negative_text_ids,
                    },
                )
                (
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    negative_text_ids,
                ) = (
                    negative_bound["prompt"],
                    negative_bound["pooled"],
                    negative_bound["ids"],
                )
        except Exception:
            template.finish(success=False)
            raise
        template_binding_seconds = time.perf_counter() - binding_started

    if cfg_batched:
        cfg_bind_start, cfg_bind_end = (
            _timed_event(latents.device),
            _timed_event(latents.device),
        )
        if cfg_bind_start is not None:
            cfg_bind_start.record()
        started = time.perf_counter()
        try:
            cfg_plan = template.cfg_plan if template is not None else None
            if cfg_plan is None:
                cfg_plan = CFGExecutionPlan(
                    active_tokens=active_count,
                    latent_channels=latents.shape[-1],
                    dtype=latents.dtype,
                    device=latents.device,
                )
                if template is not None:
                    template.cfg_plan = cfg_plan
            cfg_plan.bind_request(
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                guidance,
            )
        except Exception:
            if template is not None:
                template.finish(success=False)
            raise
        cfg_binding_wall = time.perf_counter() - started
        if cfg_bind_end is not None:
            cfg_bind_end.record()

    if specialized:
        num_kv_heads, head_dim = _transformer_kv_shape(pipeline)
        cache_mode = parse_cache_mode(variant, override=cache_layout_override)
        if cache_mode == "raw" and cfg_batched:
            raise ValueError("raw cache-layout control requires serial CFG")
        submission = "fused" if variant == "fused" else "torch"

        def _new_cache(label: str, text_count: int) -> SpecializedKVCache:
            return SpecializedKVCache(
                mode=cache_mode,
                text_token_count=0 if cache_mode == "raw" else text_count,
                generated_token_count=count,
                reference_token_count=reference_count,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=latents.dtype,
                device=latents.device,
                label=label,
                submission=submission,
                batch_size=2 if label == "batched" else 1,
                shared_pool=cfg_plan.pool if cfg_plan is not None else None,
                batch_slot=(
                    0 if label == "positive" else 1 if label == "negative" else None
                )
                if cfg_plan is not None
                else None,
            )

        positive_specialized = _new_cache("positive", text_token_count)
        if do_true_cfg:
            negative_specialized = _new_cache("negative", text_token_count)
        if cfg_plan is not None and cfg_plan.joint_cache is None:
            cfg_plan.joint_cache = _new_cache("batched", text_token_count)
        plan = SpecializedExecutionPlan(
            variant=variant,
            generated_token_count=count,
            reference_token_count=reference_count,
            active_token_count=active_count,
            latent_channels=latents.shape[-1],
            dtype=latents.dtype,
            device=latents.device,
            generated_ids=generated_ids,
            fuse_stage=stage_fusion,
        )
        if variant in {"graph", "graph_reuse"}:
            graph_state = RequestGraphState(
                positive=BranchSparseGraph(
                    "cfg_batched" if cfg_batched else "positive",
                    allow_fallback=bool(config.sparseedit_graph_fallback),
                    warmup_trials=int(config.sparseedit_graph_warmup_trials),
                ),
                negative=BranchSparseGraph(
                    "negative",
                    allow_fallback=bool(config.sparseedit_graph_fallback),
                    warmup_trials=int(config.sparseedit_graph_warmup_trials),
                )
                if do_true_cfg and not cfg_batched
                else None,
                allow_fallback=bool(config.sparseedit_graph_fallback),
            )
            if graph_disabled_reason is not None:
                # Explicit opt-in fallback: this whole request runs eagerly,
                # and the reason is reported instead of a silent stale-graph
                # replay.
                graph_state.disable(graph_disabled_reason)

        if template is not None:
            if template.plan is not None:
                # Discard the as-yet-unallocated new cache objects and use the
                # owned buffers retained by the previous request's graph.
                plan.release()
                plan = template.plan
                plan.bind_count = 0
                positive_specialized = template.positive_cache
                negative_specialized = template.negative_cache
                graph_state = template.graphs
                graph_contexts = template.contexts
            else:
                if not cfg_batched:
                    # Active K changes only compact buffers, never full-context KV shape.
                    arena_signature = (
                        id(pipeline.transformer),
                        parse_cache_mode(variant, override=cache_layout_override),
                        count,
                        reference_count,
                        text_token_count,
                        num_kv_heads,
                        head_dim,
                        layout(latents),
                        do_true_cfg,
                        stage_fusion,
                        layer_fusion,
                        layer_options,
                        share_condition,
                        position_reuse,
                        shared_position_branches,
                        cfg_selected,
                    )
                    positive_specialized, negative_specialized = bind_template_caches(
                        pipeline,
                        template,
                        arena_signature,
                        positive_specialized,
                        negative_specialized,
                    )
                template.plan = plan
                template.positive_cache = positive_specialized
                template.negative_cache = negative_specialized
                template.graphs = graph_state
                template.contexts = graph_contexts
            request_replays_before = {
                name: branch.replay_count
                for name, branch in [
                    ("positive", graph_state.positive),
                    ("negative", graph_state.negative),
                ]
                if branch is not None
            }

    if position_reuse:
        from .request_positions import RequestPositionPlan

        position_setup_start = _timed_event(latents.device)
        position_setup_end = _timed_event(latents.device)
        if position_setup_start is not None:
            position_setup_start.record()
        setup_start = time.perf_counter()
        try:
            position_plan = template.position_plan if template is not None else None
            if position_plan is None:
                position_plan = RequestPositionPlan(
                    text_tokens=text_token_count,
                    image_tokens=count + reference_count,
                    active_tokens=active_count,
                    dtype=latents.dtype,
                    device=latents.device,
                    share_branches=shared_position_branches,
                )
                if template is not None:
                    template.position_plan = position_plan
            branch_ids = {"positive": text_ids}
            if do_true_cfg:
                branch_ids["negative"] = negative_text_ids
            position_plan.bind(pipeline.transformer.pos_embed, latent_ids, branch_ids)
        except Exception:
            if template is not None:
                template.finish(success=False)
            raise
        position_setup_wall = time.perf_counter() - setup_start
        if position_setup_end is not None:
            position_setup_end.record()

    if condition_mode != "off":
        from .condition_projection_plan import RequestConditionPlan

        condition_setup_start, condition_setup_end = (
            _timed_event(latents.device),
            _timed_event(latents.device),
        )
        if condition_setup_start is not None:
            condition_setup_start.record()
        started = time.perf_counter()
        try:
            condition_plan = template.condition_plan if template is not None else None
            if condition_plan is None:
                condition_bank = None
                if (
                    config.sparseedit_condition_bank_shared
                    and condition_mode == "reuse"
                ):
                    from collections import OrderedDict

                    condition_bank = getattr(
                        pipeline, "_sparseedit_condition_bank", None
                    )
                    if condition_bank is None:
                        condition_bank = OrderedDict()
                        pipeline._sparseedit_condition_bank = condition_bank
                condition_plan = RequestConditionPlan(
                    mode="serial" if deferred_condition else condition_mode,
                    cache=condition_bank,
                    prepare_graph=condition_prepare_graph,
                )
                if template is not None:
                    template.condition_plan = condition_plan
            if condition_plan.prepare_graph != condition_prepare_graph:
                raise ValueError(
                    "condition preparation backend changed within a template"
                )
            pooled_by_branch = {"positive": pooled_prompt_embeds}
            if do_true_cfg:
                pooled_by_branch["negative"] = negative_pooled_prompt_embeds

            def bind_conditions():
                condition_plan.bind(
                    pipeline.transformer,
                    timesteps[warmup:] if deferred_condition else timesteps,
                    pooled_by_branch,
                    guidance=guidance,
                    dtype=latents.dtype,
                )

            if deferred_condition:
                from .condition_preparation import WarmupConditionPreparation

                condition_preparation = WarmupConditionPreparation(
                    condition_plan,
                    bind_conditions,
                    device=latents.device,
                    overlap=condition_mode == "warmup_overlap",
                )
            else:
                bind_conditions()
        except Exception:
            if template is not None:
                template.finish(success=False)
            raise
        condition_setup_wall = time.perf_counter() - started
        if condition_setup_end is not None:
            condition_setup_end.record()

    startup_metadata = None
    if (
        config.sparseedit_prewarm_ratios
        and config.sparseedit_prewarm_boundary == "startup"
    ):
        from .budget_prewarm import startup_budget_menu

        try:
            startup_metadata = startup_budget_menu(
                pipeline,
                template,
                latents=latents,
                timesteps=timesteps,
                config=config,
            )
        except Exception:
            template.finish(success=False)
            raise

    positive_cache = FluxSparseKVCache()
    negative_cache = FluxSparseKVCache() if do_true_cfg else None
    unused_legacy = FluxSparseKVCache()

    token_age = torch.zeros((1, count), device=latents.device, dtype=torch.int64)
    velocity_history: deque[torch.Tensor] = deque(maxlen=2)
    since_dense = 0
    pending: list[_PendingStep] = []
    computed = dense_count = sparse_count = warmup_count = refresh_count = 0
    audit_directory = str(getattr(config, "sparseedit_audit_dir", "") or "")
    collect_audit = bool(getattr(config, "sparseedit_collect_audit", False)) and bool(
        audit_directory
    )
    collect_trace = bool(getattr(config, "sparseedit_collect_trace", False))
    defer_diagnostics = bool(getattr(config, "sparseedit_defer_diagnostics", False))
    audit_seconds = 0.0
    # Audit mode keeps device-resident clones of each step's audit tensors and
    # serializes them only after the denoising end events are recorded.  The
    # benchmark path leaves ``collect_audit`` false, so these stay empty and
    # the measured loop allocates no extra host memory.
    audit_snapshots: list[dict[str, Any]] = []
    audit_capture_wall: list[float] = []
    audit_capture_events: list[tuple[Any, Any]] = []
    if not online_enabled:
        denoise_wall_start = time.perf_counter()
        denoise_start_event = _timed_event(latents.device)
        if denoise_start_event is not None:
            denoise_start_event.record()
    reuse_events: list[tuple[Any, Any]] = []
    reuse_wall_seconds: list[float] = []
    decision_wall_seconds: list[float] = []
    step_forward_events: list[
        tuple[torch.cuda.Event | None, torch.cuda.Event | None]
    ] = []
    selection_events: list[tuple[torch.cuda.Event | None, torch.cuda.Event | None]] = []

    pipeline.scheduler.set_begin_index(0)
    request_succeeded = False
    prewarm_done = startup_metadata is not None
    prewarm_metadata = startup_metadata
    try:
        if request_state is not None:
            request_state.config = config
            request_state.latents = latents
            request_state.image_latents = image_latents
            request_state.timesteps = timesteps
            request_state.conditioning = {
                "positive": prompt_embeds,
                "negative": negative_prompt_embeds,
                "positive_pooled": pooled_prompt_embeds,
                "negative_pooled": negative_pooled_prompt_embeds,
                "text_ids": text_ids,
                "negative_text_ids": negative_text_ids,
                "guidance": guidance,
            }
            request_state.caches = tuple(
                cache for cache in (positive_specialized or positive_cache, negative_specialized or negative_cache)
                if cache is not None
            )
            request_state.template = template
            request_state.token_age = token_age
            request_state.velocity_history = velocity_history
            yield None
        with pipeline.progress_bar(total=len(timesteps)) as progress:
            for step, t in enumerate(timesteps):
                if context is None:
                    pipeline._current_timestep = t
                else:
                    context.step_index = step
                    context.timestep = t
                refresh = bool(
                    step >= warmup
                    and refresh_interval
                    and since_dense > 0
                    and since_dense % refresh_interval == 0
                )
                dense = ratio == 1 or step < warmup or refresh
                decision = None
                if online_enabled:
                    decision_started = time.perf_counter()
                    force_reason = (
                        "full_ratio"
                        if ratio == 1
                        else "warmup"
                        if step < warmup
                        else "refresh_full"
                        if refresh
                        else "first_sparse_after_dense_reset"
                    )
                    if online_trace is None:
                        decision = avd.evaluate(
                            step_index=step,
                            timestep=controller_times[step],
                            previous_timestep=online_previous_timestep,
                            force_compute=dense or online_force_dense,
                            force_reason=force_reason,
                            cache_available=anchor_velocity is not None,
                        )
                    else:
                        frozen = online_trace.steps[step]
                        decision = AVDDecision(
                            frozen.action != "reuse",
                            frozen.ratio,
                            frozen.accumulate,
                            1.0 - frozen.accumulate
                            if frozen.action == "reuse"
                            else 0.0,
                            frozen.force_reason,
                        )
                        dense = frozen.action == "dense"
                        avd.accumulate = frozen.accumulate
                        avd.hit_steps += int(frozen.action == "reuse")
                        avd.computed_steps += int(frozen.action != "reuse")
                    online_force_dense = False
                    decision_wall_seconds.append(time.perf_counter() - decision_started)
                if ratio == 1:
                    phase = "full"
                elif step < warmup:
                    phase = "warmup"
                elif refresh:
                    phase = "refresh_full"
                elif online_enabled and dense:
                    phase = "reset_full"
                else:
                    phase = "sparse"
                flow_time = flow_times[step]
                if (
                    online_enabled
                    and decision is not None
                    and not decision.compute_transformer
                ):
                    assert online_program is not None and anchor_velocity is not None
                    reuse_started = time.perf_counter()
                    reuse_start_event = _timed_event(latents.device)
                    if reuse_start_event is not None:
                        reuse_start_event.record()
                    age_before = (
                        token_age.clone()
                        if collect_audit or not defer_diagnostics
                        else None
                    )
                    latents, velocity, token_age = online_program.run(
                        latents,
                        anchor_velocity,
                        float(decision.ratio),
                        online_dts[step],
                        age=token_age,
                    )
                    pipeline.scheduler._step_index += 1
                    since_dense += 1
                    velocity_history.append(velocity)
                    reuse_end_event = _timed_event(latents.device)
                    if reuse_end_event is not None:
                        reuse_end_event.record()
                    reuse_events.append((reuse_start_event, reuse_end_event))
                    reuse_wall_seconds.append(time.perf_counter() - reuse_started)
                    pending.append(
                        _PendingStep(
                            step=step,
                            phase="avd_hit",
                            flow_time=flow_time,
                            local_weight=None,
                            kv_action="none",
                            output_count=0,
                            indices=None,
                            active_mask=None,
                            priority=None,
                            token_age_before=age_before,
                            velocity_history_length=len(velocity_history),
                            cache_backend="online_anchor",
                            graph_replayed=False,
                            dense=False,
                            decision=decision,
                            anchor_step=anchor_step,
                        )
                    )
                    if collect_audit:
                        capture_start_event = _timed_event(latents.device)
                        if capture_start_event is not None:
                            capture_start_event.record()
                        capture_wall_start = time.perf_counter()
                        snapshot = _snapshot_audit_step(
                            step,
                            latents=latents,
                            velocity=velocity,
                            indices=None,
                            positive=None,
                            negative=None,
                            active_mask=None,
                            anchor_velocity=anchor_velocity,
                            anchor_step=anchor_step,
                            velocity_history=velocity_history,
                            token_age=token_age,
                            controller={
                                "action": "reuse",
                                "ratio": decision.ratio,
                                "accumulate": decision.accumulate,
                                "reason": decision.reason,
                            },
                        )
                        audit_capture_wall.append(
                            time.perf_counter() - capture_wall_start
                        )
                        capture_end_event = _timed_event(latents.device)
                        if capture_end_event is not None:
                            capture_end_event.record()
                        audit_snapshots.append(snapshot)
                        audit_capture_events.append(
                            (capture_start_event, capture_end_event)
                        )
                    online_previous_timestep = controller_times[step]
                    progress.update()
                    continue
                local_weight = (
                    None
                    if step < warmup
                    else _weight_for_step(
                        config,
                        flow_time=flow_time,
                        warmup_flow_time=warmup_flow_time,
                    )
                )

                selection_start = _timed_event(latents.device)
                if selection_start is not None:
                    selection_start.record()
                active_mask = None
                priority = None
                token_age_before = None
                trace_step = replay.step(step) if replay is not None else None
                if dense:
                    indices = all_generated
                    if specialized:
                        model_input = torch.cat([latents, image_latents], 1)
                        compact_ids = latent_ids
                        image_indices = all_images
                    else:
                        model_input = torch.cat([latents, image_latents], 1)
                        compact_ids = latent_ids
                        image_indices = all_images
                    kv_action = (
                        "store_full"
                        if ratio < 1 and (step == warmup - 1 or refresh)
                        else "none"
                    )
                    dense_count += 1
                    if step < warmup:
                        warmup_count += 1
                    elif refresh:
                        refresh_count += 1
                    if trace_step is not None and trace_step.phase != phase:
                        raise SparseTraceError(
                            f"stale mask replay at step {step}: trace phase {trace_step.phase!r} != {phase!r}"
                        )
                else:
                    if len(velocity_history) < 2:
                        raise RuntimeError(
                            "paper selector requires two preceding complete scheduler velocities"
                        )
                    expected_action = "update_active"
                    if trace_step is not None:
                        if (
                            trace_step.phase != phase
                            or trace_step.kv_action != expected_action
                        ):
                            raise SparseTraceError(
                                f"stale mask replay at step {step}: trace "
                                f"({trace_step.phase!r}, {trace_step.kv_action!r}) != ({phase!r}, {expected_action!r})"
                            )
                        indices = replay.indices_for(step, device=latents.device)
                        active_mask = _mask_from_indices(indices, count)
                    elif online_trace is not None:
                        indices = online_masks[step]
                        active_mask = _mask_from_indices(indices, count)
                    elif config.sparseedit_paper_mode == "random":
                        active_mask, priority, indices = select_random_tokens(
                            batch=1,
                            count=count,
                            active_count=active_count,
                            device=latents.device,
                            seed=config.sparseedit_paper_random_seed,
                            step=step,
                        )
                    elif specialized:
                        local_weight = float(local_weight)
                        active_mask, priority, indices = select_paper_tokens(
                            velocity_history[-1],
                            velocity_history[-2],
                            token_age,
                            active_count,
                            grid_height,
                            grid_width,
                            local_weight=local_weight,
                        )
                        indices = indices.reshape(1, -1)
                    else:
                        local_weight = float(local_weight)
                        active_mask, priority = select_paper_mask(
                            velocity_history[-1],
                            velocity_history[-2],
                            token_age,
                            active_count,
                            grid_height,
                            grid_width,
                            local_weight=local_weight,
                        )
                        indices = active_mask.nonzero(as_tuple=False)[:, 1].reshape(
                            1, active_count
                        )
                    token_age_before = (
                        token_age.detach().clone()
                        if collect_audit or not defer_diagnostics
                        else None
                    )
                    if specialized:
                        plan_indices, compact_latents, compact_ids = plan.bind_sparse(
                            latents, indices
                        )
                        if stage_fusion:
                            stage_backends.add(plan.stage_backend)
                        indices = plan_indices
                        model_input = compact_latents
                        image_indices = plan_indices
                    else:
                        model_input = gather_tokens(latents, indices)
                        compact_ids = gather_ids(generated_ids, indices)
                        image_indices = indices
                    kv_action = "update_active"
                    sparse_count += 1
                selection_end = _timed_event(latents.device)
                if selection_end is not None:
                    selection_end.record()
                selection_events.append((selection_start, selection_end))

                output_count = count if dense else active_count
                forward_start = _timed_event(latents.device)
                if forward_start is not None:
                    forward_start.record()
                if cfg_plan is not None and not dense:
                    cfg_plan.bind_step(model_input)
                prepared_this_step = condition_plan is not None and (
                    not deferred_condition or step >= warmup
                )
                if prepared_this_step:
                    if condition_preparation is not None:
                        condition_preparation.wait()
                    condition_plan.prepare(
                        step - warmup if deferred_condition else step
                    )
                if position_plan is not None:
                    position_plan.prepare(image_indices, dense=dense)
                positive_output = negative_output = None
                graph_replayed = False
                use_graph = (
                    graph_state is not None
                    and kv_action == "update_active"
                    and not dense
                )
                prewarm_refresh = bool(
                    config.sparseedit_prewarm_ratios
                    and config.sparseedit_prewarm_boundary == "warmup_refresh"
                    and step == warmup - 2
                )
                prewarm_seed = False
                if prewarm_refresh:
                    from .budget_prewarm import prewarm_menu_needed

                    prewarm_seed = prewarm_menu_needed(
                        pipeline, template, config.sparseedit_prewarm_ratios
                    )

                if specialized:
                    if use_graph:
                        # The compact buffers were already bound once for this
                        # step in the selection branch above; re-binding would
                        # re-run the gather and double-count ``bind_count``.
                        # Only the dynamic timestep is refreshed here.
                        plan.bind_timestep(
                            t.expand(1).to(device=latents.device, dtype=latents.dtype)
                            / 1000
                        )
                    timestep = (
                        plan.timestep
                        if use_graph
                        else t.expand(1).to(device=latents.device, dtype=latents.dtype)
                        / 1000
                    )
                    positive_attention_context = FluxSparseAttentionContext(
                        method="sparseedit",
                        kv_cache=unused_legacy,
                        kv_action="store_full" if prewarm_seed else kv_action,
                        image_indices=image_indices,
                        full_img_ids=latent_ids,
                        generated_token_count=count,
                        reference_token_count=reference_count,
                        specialized_cache=positive_specialized,
                        layer_fusion=layer_fusion,
                        condition_view=condition_plan.view("positive")
                        if prepared_this_step
                        else None,
                        layer_share_condition=share_condition,
                        layer_elementwise=layer_options[0],
                        layer_positions=layer_options[1],
                        layer_joint=layer_options[2],
                        request_position_view=position_plan.view(
                            "positive", dense=dense
                        )
                        if position_plan is not None
                        else None,
                    )
                    negative_attention_context = (
                        FluxSparseAttentionContext(
                            method="sparseedit",
                            kv_cache=unused_legacy,
                            kv_action="store_full" if prewarm_seed else kv_action,
                            image_indices=image_indices,
                            full_img_ids=latent_ids,
                            generated_token_count=count,
                            reference_token_count=reference_count,
                            specialized_cache=negative_specialized,
                            layer_fusion=layer_fusion,
                            condition_view=condition_plan.view("negative")
                            if prepared_this_step
                            else None,
                            layer_share_condition=share_condition,
                            layer_elementwise=layer_options[0],
                            layer_positions=layer_options[1],
                            layer_joint=layer_options[2],
                            request_position_view=position_plan.view(
                                "negative", dense=dense
                            )
                            if position_plan is not None
                            else None,
                        )
                        if do_true_cfg
                        else None
                    )
                else:
                    positive_attention_context = FluxSparseAttentionContext(
                        method="sparseedit",
                        direct_ops=config.sparseedit_direct_ops,
                        kv_cache=positive_cache,
                        kv_action=kv_action,
                        image_indices=image_indices,
                        full_img_ids=latent_ids,
                        generated_token_count=count,
                        reference_token_count=reference_count,
                    )
                    negative_attention_context = (
                        FluxSparseAttentionContext(
                            method="sparseedit",
                            direct_ops=config.sparseedit_direct_ops,
                            kv_cache=negative_cache,
                            kv_action=kv_action,
                            image_indices=image_indices,
                            full_img_ids=latent_ids,
                            generated_token_count=count,
                            reference_token_count=reference_count,
                        )
                        if do_true_cfg
                        else None
                    )
                    timestep = (
                        t.expand(1).to(device=latents.device, dtype=latents.dtype)
                        / 1000
                    )

                def _branch_kwargs(
                    branch: str,
                    attention_context: FluxSparseAttentionContext,
                    *,
                    timestep=timestep,
                    compact_ids=compact_ids,
                    step=step,
                    prompt_embeds=prompt_embeds,
                    model_input=model_input,
                ) -> dict[str, Any]:
                    if branch == "batched":
                        return _positive_kwargs(
                            hidden_states=cfg_plan.input,
                            timestep=timestep.expand(2),
                            guidance=cfg_plan.guidance,
                            pooled_prompt_embeds=cfg_plan.pooled,
                            prompt_embeds=cfg_plan.prompt,
                            text_ids=text_ids,
                            image_ids=compact_ids,
                            attention_kwargs=with_acceleration_context(
                                attention_kwargs(),
                                attention_context,
                                step_index=step,
                            ),
                        )
                    if branch == "positive":
                        pooled = pooled_prompt_embeds
                        prompt = prompt_embeds
                        txt = text_ids
                    else:
                        assert negative_pooled_prompt_embeds is not None
                        assert negative_prompt_embeds is not None
                        assert negative_text_ids is not None
                        pooled = negative_pooled_prompt_embeds
                        prompt = negative_prompt_embeds
                        txt = negative_text_ids
                    return _positive_kwargs(
                        hidden_states=model_input,
                        timestep=timestep,
                        guidance=guidance,
                        pooled_prompt_embeds=pooled,
                        prompt_embeds=prompt,
                        text_ids=txt,
                        image_ids=compact_ids,
                        attention_kwargs=with_acceleration_context(
                            attention_kwargs(),
                            attention_context,
                            step_index=step,
                        ),
                    )

                def run_branch(
                    branch: str,
                    attention_context: FluxSparseAttentionContext,
                    runner: BranchSparseGraph | None,
                    *,
                    step=step,
                    latents=latents,
                    use_graph=use_graph,
                    _branch_kwargs=_branch_kwargs,
                ) -> tuple[torch.Tensor, bool]:
                    if (
                        config.sparseedit_profile_directory
                        and step == config.sparseedit_profile_step
                        and branch == "positive"
                    ):
                        from .scope_profile import profile_forward

                        return profile_forward(
                            pipeline.transformer,
                            lambda: pipeline.predict_noise(
                                **_branch_kwargs(branch, attention_context)
                            ),
                            Path(config.sparseedit_profile_directory)
                            / f"step_{step:02d}",
                            device=latents.device,
                        ), False
                    if runner is None or not use_graph:
                        return pipeline.predict_noise(
                            **_branch_kwargs(branch, attention_context)
                        ), False
                    if not runner.graph_active and runner.fallback_reason is None:
                        context_holder = attention_context
                        graph_contexts[branch] = context_holder
                        kwargs = _branch_kwargs(branch, context_holder)

                        def closure() -> torch.Tensor:
                            return pipeline.predict_noise(**kwargs)

                        captured = runner.capture(closure, device=latents.device)
                        if not captured:
                            return pipeline.predict_noise(**kwargs), False
                        return runner.replay(), True
                    if runner.graph_active:
                        return runner.replay(), True
                    return pipeline.predict_noise(
                        **_branch_kwargs(branch, attention_context)
                    ), False

                if (
                    config.sparseedit_prewarm_ratios
                    and config.sparseedit_prewarm_boundary == "sparse"
                    and use_graph
                    and not prewarm_done
                ):
                    from .budget_prewarm import prewarm_budget_menu

                    contexts = {"positive": positive_attention_context}
                    if do_true_cfg:
                        contexts["negative"] = negative_attention_context
                    prewarm_metadata = prewarm_budget_menu(
                        pipeline,
                        template,
                        latents=latents,
                        timesteps=timesteps,
                        step=step,
                        contexts=contexts,
                        config=config,
                    )
                    prewarm_done = True

                if cfg_batched and not dense:
                    batched_context = FluxSparseAttentionContext(
                        method="sparseedit",
                        kv_cache=unused_legacy,
                        kv_action=kv_action,
                        image_indices=image_indices,
                        full_img_ids=latent_ids,
                        generated_token_count=count,
                        reference_token_count=reference_count,
                        specialized_cache=cfg_plan.joint_cache,
                        layer_fusion=layer_fusion,
                        layer_elementwise=layer_options[0],
                        layer_positions=layer_options[1],
                        layer_joint=layer_options[2],
                        layer_share_condition=share_condition,
                        request_position_view=position_plan.view(
                            "positive", dense=False
                        ),
                        condition_view=condition_plan.view("batched")
                        if prepared_this_step
                        else None,
                    )
                    batched_output, graph_replayed = run_branch(
                        "batched",
                        batched_context,
                        graph_state.positive if graph_state is not None else None,
                    )
                    positive, negative = batched_output[0:1], batched_output[1:2]
                    positive_output, negative_output = positive, negative
                    noise = (
                        None
                        if stage_fusion
                        else negative[:, :output_count]
                        + float(true_cfg_scale)
                        * (positive[:, :output_count] - negative[:, :output_count])
                    )
                else:
                    positive, positive_graph = run_branch(
                        "positive",
                        positive_attention_context,
                        graph_state.positive if graph_state is not None else None,
                    )
                    positive_output = positive
                    if condition_preparation is not None:
                        condition_preparation.after_forward_enqueue(
                            producer_start=forward_start
                        )
                    graph_replayed = graph_replayed or positive_graph
                    if do_true_cfg:
                        assert negative_attention_context is not None
                        negative, negative_graph = run_branch(
                            "negative",
                            negative_attention_context,
                            graph_state.negative if graph_state is not None else None,
                        )
                        negative_output = negative
                        if condition_preparation is not None:
                            condition_preparation.after_forward_enqueue(
                                producer_start=forward_start
                            )
                        graph_replayed = graph_replayed or negative_graph
                        positive_values = positive[:, :output_count]
                        negative_values = negative[:, :output_count]
                        noise = (
                            None
                            if stage_fusion and not dense
                            else negative_values
                            + float(true_cfg_scale)
                            * (positive_values - negative_values)
                        )
                    else:
                        noise = positive[:, :output_count]

                if prewarm_refresh:
                    from .budget_prewarm import prewarm_budget_menu

                    contexts = {"positive": positive_attention_context}
                    if do_true_cfg:
                        contexts["negative"] = negative_attention_context
                    prewarm_metadata = prewarm_budget_menu(
                        pipeline,
                        template,
                        latents=latents,
                        timesteps=timesteps,
                        step=step,
                        contexts=contexts,
                        config=config,
                        refresh_boundary=True,
                    )
                    prewarm_metadata["extra_cache_seed_step"] = (
                        step if prewarm_seed else None
                    )
                    prewarm_done = True

                if dense:
                    velocity = noise
                    token_age.zero_()
                    since_dense = 0
                else:
                    if len(velocity_history) < 1:
                        raise RuntimeError(
                            "paper selector sparse update requires a complete preceding velocity"
                        )
                    if stage_fusion:
                        from vllm_omni.diffusion.models.flux.specialized_stage import (
                            commit_sparse_velocity,
                        )

                        velocity, stage_backend = commit_sparse_velocity(
                            velocity_history[-1],
                            positive[:, :output_count],
                            negative[:, :output_count] if do_true_cfg else None,
                            indices,
                            plan.row_map,
                            token_age,
                            true_cfg_scale,
                        )
                        stage_backends.add(stage_backend)
                    else:
                        velocity = velocity_history[-1].clone()
                        scatter_tokens_(velocity, indices, noise)
                        token_age.add_(1)
                        token_age.scatter_(1, indices, 0)
                    since_dense += 1
                computed += 1
                if online_enabled:
                    online_force_dense = dense
                    anchor_velocity = velocity.detach().clone()
                    anchor_step = step
                    assert decision is not None
                velocity_history.append(
                    velocity.detach()
                    if stage_fusion and not dense
                    else velocity.detach().clone()
                )
                forward_end = _timed_event(latents.device)
                if forward_end is not None:
                    forward_end.record()
                step_forward_events.append((forward_start, forward_end))

                pending.append(
                    _PendingStep(
                        step=step,
                        phase=phase,
                        flow_time=flow_time,
                        local_weight=local_weight,
                        kv_action=kv_action,
                        output_count=output_count,
                        # The specialized path reuses one plan-owned index
                        # buffer across steps, so a plain ``detach()`` view
                        # would be overwritten by the next sparse step.  Snapshot
                        # the exact indices before they are reused.
                        # The specialized path reuses one plan-owned index
                        # buffer across steps, so a plain ``detach()`` view
                        # would be overwritten by the next sparse step.  Snapshot
                        # the exact indices before they are reused.
                        indices=(
                            None
                            if dense
                            else (
                                indices.detach().clone()
                                if specialized
                                else indices.detach()
                            )
                        ),
                        active_mask=active_mask,
                        priority=priority,
                        token_age_before=token_age_before,
                        velocity_history_length=len(velocity_history),
                        cache_backend=(
                            positive_specialized.submission_backend()
                            if positive_specialized is not None
                            else "baseline_clone"
                        ),
                        graph_replayed=graph_replayed,
                        dense=dense,
                        decision=decision,
                        anchor_step=anchor_step,
                    )
                )

                latents = pipeline.scheduler.step(
                    velocity, t, latents, return_dict=False
                )[0]
                if online_enabled:
                    online_previous_timestep = controller_times[step]
                if collect_audit:
                    # Clone only; no host transfer and no file write inside the
                    # measured interval.  Serialization happens after the
                    # denoising end events are recorded.  The clone issues real
                    # device copies on the measured stream, so each snapshot is
                    # bracketed by its own event pair and host timer; those costs
                    # are summed and subtracted from the reported denoising
                    # interval after the denoising end event is recorded.
                    capture_start_event = _timed_event(latents.device)
                    if capture_start_event is not None:
                        capture_start_event.record()
                    capture_wall_start = time.perf_counter()
                    snapshot = _snapshot_audit_step(
                        step,
                        latents=latents,
                        velocity=velocity,
                        indices=None if dense else indices,
                        positive=positive_output[:, :output_count]
                        if positive_output is not None
                        else None,
                        negative=negative_output[:, :output_count]
                        if negative_output is not None
                        else None,
                        active_mask=active_mask,
                        anchor_velocity=anchor_velocity if online_enabled else None,
                        anchor_step=anchor_step,
                        velocity_history=velocity_history,
                        token_age=token_age,
                        controller={
                            "action": "dense" if dense else "sparse",
                            "ratio": None,
                            "accumulate": decision.accumulate,
                            "reason": decision.reason,
                        }
                        if online_enabled
                        else None,
                    )
                    audit_capture_wall.append(time.perf_counter() - capture_wall_start)
                    capture_end_event = _timed_event(latents.device)
                    if capture_end_event is not None:
                        capture_end_event.record()
                    audit_snapshots.append(snapshot)
                    audit_capture_events.append(
                        (capture_start_event, capture_end_event)
                    )
                latents, prompt_embeds = _apply_callback(
                    pipeline,
                    callback_on_step_end,
                    callback_inputs or ["latents"],
                    step_index=step,
                    timestep=t,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                )
                progress.update()
                if request_state is not None:
                    request_state.latents = latents
                    request_state.token_age = token_age
                    yield StepSnapshot(step, latents.detach().clone(), velocity.detach().clone()) if request_state.capture_steps else None
        request_succeeded = True
    finally:
        execution_error = sys.exception()
        cleanup_error = None

        def release_resource(action):
            nonlocal cleanup_error, request_succeeded
            try:
                action()
            except BaseException as error:  # noqa: BLE001 - release remaining owners before propagating cancellation
                request_succeeded = False
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    cleanup_error.add_note(
                        f"Additional resource cleanup failed: {type(error).__name__}: {error}"
                    )

        try:
            # Auxiliary projections may still be pending on an exception before
            # the first sparse consumer. Join before graphs/owned buffers close.
            if condition_preparation is not None:
                condition_preparation.join()
            denoise_end_event = _timed_event(latents.device)
            if denoise_end_event is not None:
                denoise_end_event.record()
            if latents.device.type == "cuda" and torch.cuda.is_available():
                if request_state is None:
                    torch.accelerator.synchronize(latents.device)
                else:
                    denoise_end_event.synchronize()
            denoise_wall_seconds = time.perf_counter() - denoise_wall_start
            denoise_device_seconds = _event_elapsed(
                denoise_start_event, denoise_end_event
            )
            step_forward_device = [
                elapsed
                for elapsed in (
                    _event_elapsed(start, end) for start, end in step_forward_events
                )
                if elapsed is not None
            ]
            selection_device = [
                elapsed
                for elapsed in (
                    _event_elapsed(start, end) for start, end in selection_events
                )
                if elapsed is not None
            ]
            # Resolve the per-snapshot capture events now that the denoising end
            # event is recorded and the device synchronized (``elapsed_time`` syncs).
            audit_capture_wall_seconds, audit_capture_device_seconds = (
                _audit_capture_costs(audit_capture_wall, audit_capture_events)
            )
            interval = _exclude_audit_capture(
                denoise_device_seconds=denoise_device_seconds,
                denoise_wall_seconds=denoise_wall_seconds,
                audit_capture_device_seconds=audit_capture_device_seconds,
                audit_capture_wall_seconds=audit_capture_wall_seconds,
            )
            # Audit serialization is strictly post-timing: the denoising end event
            # has already been recorded and the wall clock already stopped, so the
            # CPU copies and file writes cannot inflate ``denoise_device_seconds``.
            if collect_audit and audit_snapshots:
                audit_seconds = _flush_audit_snapshots(audit_directory, audit_snapshots)
                audit_snapshots = []
            audit_capture_events = []
            audit_capture_wall = []
            cfg_witness = cfg_plan.metadata() if cfg_plan is not None else {}
            graph_metadata = graph_state.metadata() if graph_state is not None else None
            if graph_metadata is not None:
                graph_metadata["cfg_execution"] = cfg_selected
                graph_metadata["cfg_branches_per_replay"] = 2 if cfg_batched else 1
            if template is not None and graph_metadata is not None:
                for name, branch in graph_metadata["branches"].items():
                    branch["replay_count"] -= request_replays_before.get(name, 0)
                    if template_hit:
                        for field in (
                            "preparation_seconds",
                            "warmup_seconds",
                            "capture_seconds",
                        ):
                            branch[field] = 0.0
                        branch["warmup_trials"] = branch["capture_trials"] = 0
                graph_metadata["replay_counts"] = {
                    name: branch["replay_count"]
                    for name, branch in graph_metadata["branches"].items()
                }
                graph_metadata["total_replays"] = sum(
                    graph_metadata["replay_counts"].values()
                )
                graph_metadata["template_reuse"] = {
                    "cache_hit": template_hit,
                    "request_number": template.requests + 1,
                    "binding_host_seconds": template_binding_seconds,
                    "persistent_between_requests": True,
                    "prewarm": prewarm_metadata,
                    "pool": template_pool_metadata(pipeline),
                }
            plan_witness = plan.witness() if plan is not None else None
            cache_witness_payload: dict[str, Any] = {}
            if collect_audit:
                cache_witness_payload["positive"] = _cache_witness_tensors(
                    positive_specialized or positive_cache
                )
                if negative_specialized is not None or negative_cache is not None:
                    cache_witness_payload["negative"] = _cache_witness_tensors(
                        negative_specialized or negative_cache
                    )
                witness_path = Path(audit_directory) / "cache_witness.pt"
                witness_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(cache_witness_payload, witness_path)
            cache_summary = {
                "positive_cache_layers": len(positive_specialized.layers)
                if positive_specialized is not None
                else len(positive_cache.layers),
                "negative_cache_layers": (
                    len(negative_specialized.layers)
                    if negative_specialized is not None
                    else (
                        len(negative_cache.layers) if negative_cache is not None else 0
                    )
                ),
                "positive_cache_uid": id(positive_specialized or positive_cache),
                "negative_cache_uid": (
                    id(negative_specialized or negative_cache)
                    if (negative_specialized or negative_cache) is not None
                    else None
                ),
                "positive_cache_witness": (
                    positive_specialized.pointer_witness()
                    if positive_specialized is not None
                    else positive_cache_witness(positive_cache)
                ),
                "negative_cache_witness": (
                    negative_specialized.pointer_witness()
                    if negative_specialized is not None
                    else (
                        positive_cache_witness(negative_cache)
                        if negative_cache is not None
                        else None
                    )
                ),
            }
        except BaseException as error:  # noqa: BLE001 - always release owners and preserve the primary failure
            cleanup_error = error
            request_succeeded = False
        finally:
            if graph_state is not None and template is None:
                release_resource(graph_state.close)
            if plan is not None and template is None:
                release_resource(plan.release)
            if positive_specialized is not None and template is None:
                release_resource(positive_specialized.clear)
            if negative_specialized is not None and template is None:
                release_resource(negative_specialized.clear)
            release_resource(positive_cache.clear)
            if negative_cache is not None:
                release_resource(negative_cache.clear)
            if cfg_plan is not None and template is None:
                release_resource(cfg_plan.release)
            if template is not None:
                release_resource(
                    lambda: template.finish(
                        success=request_succeeded
                        and cleanup_error is None
                        and execution_error is None
                    )
                )
            if context is None:
                pipeline._current_timestep = None
            else:
                context.timestep = None
        if cleanup_error is not None:
            if execution_error is None:
                raise cleanup_error
            execution_error.add_note(
                f"Preparation/cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}"
            )

    records: list[dict[str, Any]] = []
    for item in pending:
        record: dict[str, Any] = {
            "step": item.step,
            "phase": item.phase,
            "flow_time": item.flow_time,
            **(
                {"controller_timestep": controller_times[item.step]}
                if online_enabled
                else {}
            ),
            "local_weight": item.local_weight,
            "transformer_computed": item.phase != "avd_hit",
            "is_sparse_step": item.phase == "sparse",
            "active_token_count": item.output_count,
            "active_ratio": item.output_count / count,
            "kv_action": item.kv_action,
            "cfg_branch_count": 2 if do_true_cfg else 1,
            "reference_policy": "cached" if not item.dense else "full",
            "execution_backend": item.cache_backend,
        }
        if online_enabled:
            assert item.decision is not None
            record.update(
                controller_action="reuse"
                if item.phase == "avd_hit"
                else ("dense" if item.dense else "sparse"),
                avd_decision=item.decision.reason,
                avd_ratio=item.decision.ratio if item.phase == "avd_hit" else None,
                avd_accumulate=item.decision.accumulate,
                avd_error=item.decision.error,
                velocity_anchor_step=item.anchor_step,
                dense_refresh_reset=item.dense,
            )
        if item.indices is not None:
            record["selected_index_digest"] = _index_digest(item.indices)
        if item.active_mask is not None:
            record["mask_token_count"] = count
            record["active_mask_encoding"] = "base64_packbits_row_major"
            record["active_mask_b64"] = _pack_mask(item.active_mask)
            record["indices"] = (
                item.indices.detach()
                .to(device="cpu", dtype=torch.int64)
                .reshape(-1)
                .tolist()
            )
        if item.priority is not None and not defer_diagnostics and not online_enabled:
            record.update(
                priority_mean=float(item.priority.mean().item()),
                priority_min=float(item.priority.min().item()),
                priority_max=float(item.priority.max().item()),
            )
        if item.token_age_before is not None and not defer_diagnostics:
            record.update(
                token_age_before_mean=float(
                    item.token_age_before.float().mean().item()
                ),
                token_age_before_max=int(item.token_age_before.max().item()),
            )
        if item.phase == "sparse" or online_enabled:
            record["velocity_history_length"] = item.velocity_history_length
        if item.graph_replayed:
            record["graph_replayed"] = True
        records.append(record)

    trace_payload = None
    trace_sha = None
    if collect_trace and not online_enabled:
        trace_config = {
            "variant": variant,
            "ratio": float(ratio),
            "warmup_steps": int(warmup),
            "refresh_interval": int(refresh_interval),
            "num_inference_steps": len(timesteps),
            "generated_token_count": count,
            "reference_token_count": reference_count,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "active_token_count": active_count,
            "true_cfg": bool(do_true_cfg),
            "true_cfg_scale": float(true_cfg_scale),
            "selector_protocol": "paper",
            "selector_mode": config.sparseedit_paper_mode,
            "selector_constant_weight": float(config.sparseedit_paper_constant_weight),
            "selector_random_seed": config.sparseedit_paper_random_seed,
        }
        trace = FixedMaskTrace.from_step_records(records, config=trace_config)
        trace_payload = trace.to_dict()
        trace_sha = trace.sha256

    runtime_timing = {
        # The reported denoising figures exclude the in-loop audit snapshots'
        # own device copies and host enqueue time; the raw inclusive totals are
        # kept alongside so the subtraction can be checked independently.
        "denoise_wall_seconds": interval["denoise_wall_seconds"],
        "denoise_wall_seconds_including_audit_capture": interval[
            "denoise_wall_seconds_including_audit_capture"
        ],
        "denoise_device_seconds": interval["denoise_device_seconds"],
        "denoise_device_seconds_including_audit_capture": interval[
            "denoise_device_seconds_including_audit_capture"
        ],
        "audit_capture_device_seconds": interval["audit_capture_device_seconds"],
        "audit_capture_wall_seconds": interval["audit_capture_wall_seconds"],
        "audit_write_seconds": float(audit_seconds),
        "audit_write_scope": "after_denoise_timing"
        if collect_audit
        else "not_collected",
        "audit_accounting_scope": "audit_capture_excluded_from_denoise"
        if collect_audit
        else "not_collected",
        "step_forward_device_seconds": step_forward_device,
        "reuse_device_seconds": [
            _event_elapsed(start, end) for start, end in reuse_events
        ],
        "reuse_wall_enqueue_seconds": reuse_wall_seconds,
        "decision_wall_seconds": decision_wall_seconds,
        "provenance_preparation_wall_seconds": provenance_wall_seconds,
        "denoise_scope": "synchronized preparation, binding, decisions, transformer and reuse; provenance in request wall"
        if online_enabled
        else "synchronized denoising loop",
        "reuse_timing_scope": "binding, anchored Euler, scheduler index, ages and history",
        "native_euler_dt": online_dt_metadata if online_enabled else None,
        "selection_device_seconds": selection_device,
        "collection": "deferred_after_denoising"
        if defer_diagnostics
        else "per_step_diagnostics",
        "flow_time_source": "pipeline.scheduler.sigmas[step] precomputed before denoising",
        "unavoidable_selection_sync": "none in the timed sparse loop"
        if defer_diagnostics
        else "per-step diagnostic scalar reads (debug mode)",
    }
    memory_summary: dict[str, Any] = {}
    if latents.device.type == "cuda" and torch.cuda.is_available():
        memory_summary = {
            "measurement_pid": os.getpid(),
            "scope": "model_worker_denoising_request_peak",
            "max_memory_allocated": int(
                torch.accelerator.max_memory_allocated(latents.device)
            ),
            "max_memory_reserved": int(
                torch.accelerator.max_memory_reserved(latents.device)
            ),
            "memory_allocated": int(torch.accelerator.memory_allocated(latents.device)),
            "memory_reserved": int(torch.accelerator.memory_reserved(latents.device)),
            "device": str(latents.device),
        }

    metadata = {
        "schema_version": 1,
        "method": "sparseedit",
        "selector_protocol": "paper",
        "runtime_variant": variant,
        "run_config": {
            **config.to_metadata(),
            "warmup_steps": warmup,
            "refresh_interval": refresh_interval,
            "ratio": ratio,
            "ratio_rounding": "floor",
            "generated_token_count": count,
            "reference_token_count": reference_count,
            "active_token_count": active_count,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "selector": "uniform_exact_k"
            if config.sparseedit_paper_mode == "random"
            else "paper_velocity_history",
            "selector_mode": config.sparseedit_paper_mode,
            "selector_constant_weight": config.sparseedit_paper_constant_weight,
            "selector_random_seed": config.sparseedit_paper_random_seed,
            "base_score": "owned_seed_step_permutation_rank"
            if config.sparseedit_paper_mode == "random"
            else "normalize(||v[i-1]-v[i-2]||_2 * (1 + 0.5 * age))",
            "local_score": "not_used"
            if config.sparseedit_paper_mode == "random"
            else "centered_in_grid_5x5_mean_of_normalized_base",
            "priority": "random_permutation_rank"
            if config.sparseedit_paper_mode == "random"
            else "(1 - local_weight) * base + local_weight * local",
            "flow_time_source": "pipeline.scheduler.sigmas[step]",
            "tie_break": "stable_lower_grid_index",
            "velocity_history": "two_preceding_complete_scheduler_velocities",
            **({"avd_final_steps_force": "transformer_only"} if online_enabled else {}),
            "true_cfg": do_true_cfg,
            "true_cfg_scale": float(true_cfg_scale),
            "cfg_combination": "negative + true_cfg_scale * (positive - negative)",
            "text_recompute": "every_positive_and_negative_forward",
            "reference_policy": "full_cached_context",
            "num_inference_steps": len(timesteps),
            "scheduler_class": type(pipeline.scheduler).__name__,
            "latent_dtype": str(latents.dtype),
            "scheduler_begin_index": 0,
            "execution_plan": plan_witness,
            "cfg_execution_plan": {
                "requested": cfg_requested,
                "selected": cfg_selected,
                "reason": cfg_reason,
                "query_rows": text_token_count + active_count,
                "min_query_rows": config.sparseedit_cfg_min_query_rows,
                "policy_source": "explicit_query_shape_threshold_from_projection_probe",
                **cfg_witness,
                "binding_wall_seconds": cfg_binding_wall,
                "binding_device_seconds": _event_elapsed(cfg_bind_start, cfg_bind_end),
                "binding_scope": "included_in_denoise_preparation"
                if online_enabled
                else "before_denoise_included_in_request_wall",
                "step_pack_scope": "included_in_step_forward",
            },
            "condition_plan": {
                "enabled": condition_mode != "off",
                **(condition_plan.metadata() if condition_plan is not None else {}),
                "mode": condition_mode,
                "prepare_graph_requested": condition_prepare_graph,
                "template_identity_mode": "warmup_native_serial"
                if deferred_condition
                else condition_mode,
                "preparation": condition_preparation.metadata(
                    denoise_start=denoise_start_event
                )
                if condition_preparation is not None
                else None,
                "projection_scope": "during_denoise"
                if deferred_condition or online_enabled
                else "before_denoise",
                "prepared_consumers_from_step": warmup if deferred_condition else 0,
                "setup_wall_enqueue_seconds": condition_setup_wall,
                "setup_device_seconds": _event_elapsed(
                    condition_setup_start, condition_setup_end
                ),
                "setup_scope": "included_in_denoise_preparation"
                if online_enabled
                else "before_denoise_included_in_request_wall",
                "step_binding_scope": "included_in_step_forward",
            },
            "request_positions": {
                "enabled": position_reuse,
                **(position_plan.metadata() if position_plan is not None else {}),
                "setup_wall_enqueue_seconds": position_setup_wall,
                "setup_device_seconds": _event_elapsed(
                    position_setup_start, position_setup_end
                ),
                "setup_scope": "included_in_denoise_preparation"
                if online_enabled
                else "before_denoise_included_in_request_wall",
                "step_preparation_scope": "included_in_step_forward",
            },
            "graph": graph_metadata,
            "replay": {
                "enabled": replay is not None or online_trace is not None,
                "kind": "online_controller"
                if online_trace is not None
                else "fixed_mask"
                if replay is not None
                else None,
                "path": replay_path or None,
                "sha256": online_trace.sha256
                if online_trace is not None
                else replay.sha256
                if replay is not None
                else None,
            },
        },
        "step_records": records,
        "timing": runtime_timing,
        "memory": memory_summary,
        "summary": {
            "transformer_compute_steps": computed,
            "avd_hit_steps": avd.hit_steps if avd is not None else 0,
            "transformer_skipped_steps": avd.hit_steps if avd is not None else 0,
            "dense_steps": dense_count,
            "sparse_steps": sparse_count,
            "reuse_calls": online_program.calls if online_program is not None else 0,
            "reuse_fused_kernel_launches": online_program.fused_kernel_launches
            if online_program is not None
            else 0,
            "reuse_mode": online_mode,
            "graph_reuse_captured": False,
            "warmup_steps": warmup_count,
            "refresh_steps": refresh_count,
            "num_inference_steps": len(timesteps),
            "true_cfg": do_true_cfg,
            "true_cfg_scale": float(true_cfg_scale),
            "active_token_count": active_count,
            "shared_active_mask": True,
            "stage_backends": sorted(stage_backends),
            "caches_cleared": template is None or not request_succeeded,
            "persistent_template_retained": template is not None and request_succeeded,
            "runtime_variant": variant,
            "specialized_cache_mode": (
                parse_cache_mode(variant, override=cache_layout_override)
                if specialized
                else None
            ),
            "submission_backend": (
                positive_specialized.submission_backend()
                if positive_specialized is not None
                else "baseline_clone"
            ),
            "actual_backends": sorted(positive_specialized.backends)
            if positive_specialized is not None
            else [],
            "operated_rows": int(positive_specialized.operated_rows)
            if positive_specialized is not None
            else None,
            "cache_counter_scope": "python_forward_calls_including_graph_preparation_excluding_replay",
            "graph_active": bool(graph_metadata["graph_active"])
            if graph_metadata is not None
            else False,
            "graph_fallback": bool(graph_metadata["fallback"])
            if graph_metadata is not None
            else None,
            "replay_applied": replay is not None or online_trace is not None,
            **cache_summary,
        },
    }
    if online_enabled:
        online_steps = [
            ControllerStep(
                step=item.step,
                action="reuse"
                if item.phase == "avd_hit"
                else "dense"
                if item.dense
                else "sparse",
                ratio=float(item.decision.ratio) if item.phase == "avd_hit" else None,
                accumulate=float(item.decision.accumulate),
                force_reason=item.decision.reason,
                indices=None
                if item.indices is None
                else tuple(item.indices.detach().cpu().reshape(-1).tolist()),
                anchor_step=item.anchor_step,
            )
            for item in pending
        ]
        produced_trace = OnlineReuseTrace.from_steps(
            config=online_config, steps=online_steps
        )
        validate_controller_replay(produced_trace, num_inference_steps=len(timesteps))
        if online_trace is not None and produced_trace.sha256 != online_trace.sha256:
            raise ValueError("online replay produced a different controller trace")
        metadata["online_reuse_trace"] = produced_trace.to_dict()
        metadata["online_reuse_trace_sha256"] = produced_trace.sha256
    if trace_payload is not None:
        metadata["fixed_mask_trace"] = trace_payload
        metadata["fixed_mask_trace_sha256"] = trace_sha
    return latents, metadata


def positive_cache_witness(cache: FluxSparseKVCache) -> dict[str, Any]:
    return {
        "layer_count": len(cache.layers),
        "layers": sorted(
            [f"{stream}:{layer}" for (stream, layer) in cache.layers],
        ),
    }


__all__ = [
    "_audit_capture_costs",
    "_exclude_audit_capture",
    "_flush_audit_snapshots",
    "_serialize_audit_payload",
    "_snapshot_audit_step",
    "diffuse_sparseedit_paper",
]


def _diffuse_sparseedit_paper_original(pipeline, **kwargs):
    return return_value(_iter_sparseedit_paper(pipeline, **kwargs))


def iter_sparseedit_paper(pipeline, *, request_state, **kwargs):
    config = kwargs["config"]
    if getattr(config, "sparseedit_standard_backend", "off") != "off":
        raise ValueError("Standard backend session mutates the shared transformer")
    return _iter_sparseedit_paper(pipeline, request_state=request_state, **kwargs)


def diffuse_sparseedit_paper(pipeline, **kwargs):
    config = kwargs['config']
    if getattr(config, 'sparseedit_standard_backend', 'off') == 'off':
        return _diffuse_sparseedit_paper_original(pipeline, **kwargs)
    from .standard_backend import BackendSession
    session = BackendSession(pipeline, config)
    try:
        session.__enter__()
        latents, metadata = _diffuse_sparseedit_paper_original(pipeline, **kwargs)
    finally:
        session.__exit__(*sys.exc_info())
    metadata['standard_backend'] = session.final
    if session.candidate == 'standard_sparse_graph':
        for row in metadata['step_records']:
            row['graph_replayed'] = row['kv_action'] == 'update_active'
        metadata['summary']['graph_active'] = True
    return latents, metadata
