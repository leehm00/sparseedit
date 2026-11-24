# SPDX-License-Identifier: Apache-2.0
"""Prepare finite budget templates at a serial warmup boundary.

Trial forwards may change text and selected image KV. Protect only this known
prefix and restore it before any real sparse consumer resumes. No denoising,
selector, scheduler or full KV allocation is performed for sibling templates.
"""

from __future__ import annotations

import time
from dataclasses import replace

import torch

from .acceleration import (
    FluxSparseAttentionContext,
    FluxSparseKVCache,
    with_acceleration_context,
)
from .acceleration_runtime import _positive_kwargs
from .condition_projection_plan import RequestConditionPlan
from .request_positions import RequestPositionPlan
from .specialized_graph import BranchSparseGraph, RequestGraphState
from .specialized_plan import SpecializedExecutionPlan
from .specialized_template import bind_template_caches, reserve_template


class PartialKVSnapshot:
    def __init__(self, caches, *, active_rows):
        self.entries = []
        self.bytes = 0
        for cache in caches:
            if cache is None:
                continue
            if (
                cache.mode != "rotated"
                or not 0 < active_rows <= cache.generated_token_count
            ):
                raise ValueError(
                    "prewarm snapshots require rotated cache and valid active rows"
                )
            if not cache.layers:
                raise RuntimeError(
                    "prewarm requires completed dense cache initialization"
                )
            rows = cache.text_token_count + active_rows
            for storage in cache.layers.values():
                key, value = (
                    storage.key[:, :rows].clone(),
                    storage.value[:, :rows].clone(),
                )
                self.entries.append((storage.key, storage.value, key, value, rows))
                self.bytes += (key.numel() + value.numel()) * key.element_size()

    def restore(self):
        for target_key, target_value, key, value, rows in self.entries:
            target_key[:, :rows].copy_(key)
            target_value[:, :rows].copy_(value)

    def release(self):
        self.entries.clear()
        self.bytes = 0


def _prepare_sibling(
    pipeline,
    sibling,
    owner,
    *,
    active_count,
    latents,
    timesteps,
    step,
    contexts,
    config,
):
    original = owner.plan
    if sibling is not owner:
        sibling.full_img_ids = owner.full_img_ids.detach().clone()
        for branch, tensors in owner.conditioning.items():
            sibling.bind_conditioning(branch, tensors)
        sibling.positive_cache, sibling.negative_cache = bind_template_caches(
            pipeline,
            sibling,
            getattr(owner, "arena_signature", None)
            or owner.signature[:4] + owner.signature[5:],
            owner.positive_cache,
            owner.negative_cache,
        )
        sibling.plan = SpecializedExecutionPlan(
            variant="graph_reuse",
            generated_token_count=original.generated_token_count,
            reference_token_count=original.reference_token_count,
            active_token_count=active_count,
            latent_channels=original.latent_channels,
            dtype=original.dtype,
            device=original.device,
            generated_ids=sibling.full_img_ids[: original.generated_token_count],
            fuse_stage=original.fuse_stage,
        )
        sibling.position_plan = RequestPositionPlan(
            text_tokens=owner.position_plan.text_tokens,
            image_tokens=owner.position_plan.image_tokens,
            active_tokens=active_count,
            dtype=original.dtype,
            device=original.device,
            share_branches=owner.position_plan.share_branches,
        )
        sibling.position_plan.bind(
            pipeline.transformer.pos_embed,
            sibling.full_img_ids,
            {
                branch: tensors["ids"]
                for branch, tensors in sibling.conditioning.items()
            },
        )
        sibling.condition_plan = RequestConditionPlan(
            mode="reuse", cache=pipeline._sparseedit_condition_bank
        )
        sibling.condition_plan.bind(
            pipeline.transformer,
            timesteps,
            {
                branch: tensors["pooled"]
                for branch, tensors in sibling.conditioning.items()
            },
            guidance=sibling.conditioning["positive"].get("guidance"),
            dtype=original.dtype,
        )
        sibling.graphs = RequestGraphState(
            positive=BranchSparseGraph(
                "positive", warmup_trials=config.sparseedit_graph_warmup_trials
            ),
            negative=BranchSparseGraph(
                "negative", warmup_trials=config.sparseedit_graph_warmup_trials
            )
            if "negative" in contexts
            else None,
        )
    # The active owner keeps exactly the buffers captured by the enclosing loop.
    indices = torch.arange(
        active_count, dtype=torch.long, device=original.device
    ).unsqueeze(0)
    sibling.plan.bind_sparse(latents, indices)
    sibling.plan.bind_timestep(
        timesteps[step].expand(1).to(device=original.device, dtype=original.dtype)
        / 1000
    )
    sibling.position_plan.prepare(sibling.plan.active_indices, dense=False)
    sibling.condition_plan.prepare(step)
    guidance = sibling.conditioning["positive"].get("guidance")
    for branch, source in contexts.items():
        cache = (
            sibling.positive_cache if branch == "positive" else sibling.negative_cache
        )
        context = replace(
            source,
            kv_action="update_active",
            image_indices=sibling.plan.active_indices,
            full_img_ids=sibling.full_img_ids,
            specialized_cache=cache,
            condition_view=sibling.condition_plan.view(branch),
            request_position_view=sibling.position_plan.view(branch, dense=False),
        )
        sibling.contexts[branch] = context
        tensors = sibling.conditioning[branch]
        kwargs = _positive_kwargs(
            hidden_states=sibling.plan.compact_latents,
            timestep=sibling.plan.timestep,
            guidance=guidance,
            pooled_prompt_embeds=tensors["pooled"],
            prompt_embeds=tensors["prompt"],
            text_ids=tensors["ids"],
            image_ids=sibling.plan.compact_ids,
            attention_kwargs=with_acceleration_context(
                pipeline.joint_attention_kwargs,
                context,
                step_index=config.sparseedit_warmup_steps
                if config.sparseedit_prewarm_boundary != "sparse"
                else step,
            ),
        )
        runner = (
            sibling.graphs.positive if branch == "positive" else sibling.graphs.negative
        )
        runner.capture(
            lambda kwargs=kwargs: pipeline.predict_noise(**kwargs),
            device=original.device,
        )


def prewarm_menu_needed(pipeline, owner, ratios):
    pool = pipeline._sparseedit_template_pool
    for ratio in ratios:
        count = max(1, int(owner.plan.generated_token_count * ratio))
        signature = owner.signature[:4] + (count,) + owner.signature[5:]
        template = pool.templates.get(signature)
        if (
            template is None
            or template.closed
            or template.graphs is None
            or not template.graphs.positive.graph_active
        ):
            return True
    return False


def prewarm_budget_menu(
    pipeline,
    owner,
    *,
    latents,
    timesteps,
    step,
    contexts,
    config,
    refresh_boundary=False,
):
    pool = pipeline._sparseedit_template_pool
    counts = sorted(
        {
            max(1, int(owner.plan.generated_token_count * r))
            for r in config.sparseedit_prewarm_ratios
        }
    )
    missing = []
    hits = []
    for count in counts:
        signature = owner.signature[:4] + (count,) + owner.signature[5:]
        candidate = pool.templates.get(signature)
        if candidate is owner and not refresh_boundary:
            continue
        if (
            candidate is not None
            and not candidate.closed
            and candidate.graphs is not None
            and candidate.graphs.positive.graph_active
        ):
            hits.append(count)
        else:
            missing.append((count, signature))
    result = {
        "counts": counts,
        "current_count": owner.plan.active_token_count,
        "already_ready_counts": hits,
        "prepared_counts": [],
        "snapshot_bytes": 0,
        "host_seconds": 0.0,
        "device_seconds": 0.0,
        "boundary": "before_final_dense_refresh"
        if refresh_boundary
        else "first_sparse_step_before_actual_forward",
        "protected_by_next_dense_refresh": refresh_boundary,
        "scope": (
            "menu binding and capture; extra seed belongs to dense forward"
            if refresh_boundary
            else "includes row snapshot, sibling binding, warmup, capture and restoration"
        ),
    }
    if not missing:
        return result
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    began = time.perf_counter()
    snapshot = (
        None
        if refresh_boundary
        else PartialKVSnapshot(
            (owner.positive_cache, owner.negative_cache),
            active_rows=max(c for c, _ in missing),
        )
    )
    result["snapshot_bytes"] = snapshot.bytes if snapshot is not None else 0
    prepared = []
    dirty = False
    try:
        for count, signature in missing:
            sibling, _ = reserve_template(pipeline, owner, signature)
            prepared.append(sibling)
            dirty = True
            _prepare_sibling(
                pipeline,
                sibling,
                owner,
                active_count=count,
                latents=latents,
                timesteps=timesteps,
                step=step,
                contexts=contexts,
                config=config,
            )
            # Each trial sees exactly the same original numerical context.
            if snapshot is not None:
                snapshot.restore()
                dirty = False
            result["prepared_counts"].append(count)
    except Exception:
        for sibling in prepared:
            sibling.close()
        raise
    finally:
        if snapshot is not None:
            if dirty:
                snapshot.restore()
            snapshot.release()
        end.record()
        end.synchronize()
        result["host_seconds"] = time.perf_counter() - began
        result["device_seconds"] = start.elapsed_time(end) / 1000
    result["includes_owner_capture"] = any(sibling is owner for sibling in prepared)
    result["capture_seconds"] = sum(
        runner.capture_seconds
        for sibling in prepared
        for runner in (sibling.graphs.positive, sibling.graphs.negative)
        if runner is not None
    )
    return result


def startup_budget_menu(pipeline, owner, *, latents, timesteps, config):
    """Warm templates on provisional KV before warmup publishes real numerical KV."""
    checked = time.perf_counter()
    needed = prewarm_menu_needed(pipeline, owner, config.sparseedit_prewarm_ratios)
    if not needed:
        counts = sorted(
            {
                max(1, int(owner.plan.generated_token_count * r))
                for r in config.sparseedit_prewarm_ratios
            }
        )
        return {
            "counts": counts,
            "current_count": owner.plan.active_token_count,
            "already_ready_counts": counts,
            "prepared_counts": [],
            "snapshot_bytes": 0,
            "initialized_bytes": 0,
            "host_seconds": time.perf_counter() - checked,
            "device_seconds": 0.0,
            "includes_owner_capture": False,
            "boundary": "before_denoising",
            "protected_by_full_store_step": config.sparseedit_warmup_steps - 1,
            "scope": "ready CPU lookup only; no CUDA event, initialization or synchronization",
        }
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    began = checked
    initialized = 0
    if needed:
        for cache in (owner.positive_cache, owner.negative_cache):
            if cache is None:
                continue
            for stream, blocks in (
                ("dual", pipeline.transformer.transformer_blocks),
                ("single", pipeline.transformer.single_transformer_blocks),
            ):
                for layer in range(len(blocks)):
                    storage = cache.ensure_layer(stream, layer)
                    storage.key.zero_()
                    storage.value.zero_()
                    initialized += (
                        storage.key.numel() + storage.value.numel()
                    ) * storage.key.element_size()
    contexts = {}
    for branch, cache in (
        ("positive", owner.positive_cache),
        ("negative", owner.negative_cache),
    ):
        if cache is None:
            continue
        contexts[branch] = FluxSparseAttentionContext(
            method="sparseedit",
            kv_cache=FluxSparseKVCache(),
            kv_action="update_active",
            image_indices=owner.plan.active_indices,
            full_img_ids=owner.full_img_ids,
            generated_token_count=owner.plan.generated_token_count,
            reference_token_count=owner.plan.reference_token_count,
            specialized_cache=cache,
            layer_fusion=config.sparseedit_layer_fusion,
            layer_share_condition=config.sparseedit_share_condition,
            layer_elementwise=config.sparseedit_fuse_elementwise,
            layer_positions=config.sparseedit_share_positions,
            layer_joint=config.sparseedit_share_joint,
        )
    result = prewarm_budget_menu(
        pipeline,
        owner,
        latents=latents,
        timesteps=timesteps,
        step=0,
        contexts=contexts,
        config=config,
        refresh_boundary=True,
    )
    end.record()
    end.synchronize()
    result.update(
        boundary="before_denoising",
        initialized_bytes=initialized,
        host_seconds=time.perf_counter() - began,
        device_seconds=start.elapsed_time(end) / 1000,
        protected_by_full_store_step=config.sparseedit_warmup_steps - 1,
        scope="includes provisional KV zero initialization, bindings and all menu captures; request wall includes it",
    )
    return result
