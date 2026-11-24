# SPDX-License-Identifier: Apache-2.0
"""Request-scoped SparseEdit spatial updates and AVD for native FLUX Kontext."""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.models.flux.ablation_diagnostics import collect_sparseedit_mask_diagnostics
from vllm_omni.diffusion.models.flux.acceleration import (
    FluxSparseAttentionContext,
    FluxSparseKVCache,
    gather_ids,
    gather_tokens,
    scatter_tokens_,
    with_acceleration_context,
)
from vllm_omni.diffusion.models.flux.acceleration_runtime import _apply_callback, _positive_kwargs
from vllm_omni.diffusion.models.flux.sparseedit_primitives import (
    SparseEditAVDConfig,
    SparseEditAVDGammaStats,
    SparseEditAVDState,
    score_sparseedit_spatial_error_tokens,
    select_sparseedit_linear_mask,
    select_sparseedit_spatial_error_mask,
)
from vllm_omni.diffusion.models.flux.warmup_diagnostics import collect_warmup_step_diagnostics


def diffuse_sparseedit(
    pipeline: Any,
    *,
    config,
    prompt_embeds,
    pooled_prompt_embeds,
    latents,
    image_latents,
    latent_ids,
    text_ids,
    timesteps,
    guidance,
    grid_height,
    grid_width,
    callback_on_step_end=None,
    callback_inputs=None,
):
    count, reference_count = latents.shape[1], image_latents.shape[1]
    all_generated = torch.arange(count, device=latents.device).unsqueeze(0)
    all_images = torch.arange(count + reference_count, device=latents.device).unsqueeze(0)
    generated_ids = latent_ids[:count]
    cache = FluxSparseKVCache()
    warmup = config.sparseedit_warmup_steps
    refresh_interval = config.sparseedit_refresh_interval
    active_count = max(1, min(count, int(count * config.sparseedit_ratio)))
    avd = None
    if config.sparseedit_avd_enabled:
        avd = SparseEditAVDState(
            SparseEditAVDConfig(
                num_inference_steps=len(timesteps),
                cache_threshold=config.sparseedit_avd_cache_threshold,
                max_consecutive_hits=config.sparseedit_avd_max_reuse,
                force_final_steps=config.sparseedit_avd_force_final_steps,
                gamma=config.sparseedit_avd_gamma,
            )
        )
    gamma_stats = SparseEditAVDGammaStats(len(timesteps)) if config.sparseedit_collect_gamma else None
    staleness = torch.zeros((1, count), device=latents.device, dtype=torch.int16)
    previous_latents = previous_previous_latents = None
    previous_velocity = last_computed_velocity = None
    since_dense = 0
    records = []
    computed = hits = dense_count = sparse_count = 0
    pipeline.scheduler.set_begin_index(0)
    try:
        with pipeline.progress_bar(total=len(timesteps)) as progress:
            for step, t in enumerate(timesteps):
                pipeline._current_timestep = t
                mask_diagnostics = None
                refresh = bool(
                    step >= warmup and refresh_interval and since_dense > 0 and since_dense % refresh_interval == 0
                )
                dense = config.sparseedit_ratio == 1 or step < warmup or refresh
                reason = (
                    "full_ratio"
                    if config.sparseedit_ratio == 1
                    else "warmup"
                    if step < warmup
                    else "refresh_full"
                    if refresh
                    else None
                )
                decision = None
                if avd is not None:
                    force = dense or since_dense == 0
                    decision = avd.evaluate(
                        step_index=step,
                        timestep=t,
                        previous_timestep=timesteps[step - 1] if step else None,
                        force_compute=force,
                        force_reason=reason or "first_sparse_after_dense_reset",
                        cache_available=last_computed_velocity is not None,
                    )
                compute = decision is None or decision.compute_transformer
                if not compute:
                    # Preserve the established SparseEdit convention: each hit scales the last
                    # computed full velocity; the accumulated factor controls the reuse gate.
                    noise = last_computed_velocity * decision.ratio
                    indices = all_generated[:, :0]
                    action = "none"
                    phase = "avd_hit"
                    hits += 1
                    staleness.add_(1)
                    since_dense += 1
                else:
                    if dense:
                        indices = all_generated
                        model_input = torch.cat([latents, image_latents], 1)
                        compact_ids = latent_ids
                        image_indices = all_images
                        action = (
                            "store_full" if config.sparseedit_ratio < 1 and (step == warmup - 1 or refresh) else "none"
                        )
                        phase = "warmup" if step < warmup else "refresh_full" if refresh else "full"
                        dense_count += 1
                    else:
                        scores = score_sparseedit_spatial_error_tokens(
                            latents=latents,
                            previous_latents=previous_latents,
                            num_tokens=count,
                            grid_height=grid_height,
                            grid_width=grid_width,
                            staleness_counter=staleness,
                        )
                        linear_locality = config.sparseedit_locality_mode == "linear"
                        if linear_locality and len(timesteps) - warmup <= 1:
                            # A single sparse step is the endpoint of the interpolation.
                            sparse_progress = 1.0
                        else:
                            sparse_progress = (step - warmup) / max(
                                1, len(timesteps) - warmup - int(linear_locality)
                            )
                        selection_progress = 1.0 if config.sparseedit_locality_mode == "off" else sparse_progress
                        if linear_locality:
                            mask, _ = select_sparseedit_linear_mask(
                                scores=scores,
                                grid_height=grid_height,
                                grid_width=grid_width,
                                active_count=active_count,
                                progress=selection_progress,
                            )
                        else:
                            mask, _ = select_sparseedit_spatial_error_mask(
                                update_strategy="spatial_error_smooth_decay",
                                scores=scores,
                                grid_height=grid_height,
                                grid_width=grid_width,
                                active_count=active_count,
                                progress=selection_progress,
                            )
                        if config.sparseedit_collect_masks:
                            mask_diagnostics = collect_sparseedit_mask_diagnostics(
                                active_mask=mask,
                                scores=scores,
                                grid_height=grid_height,
                                grid_width=grid_width,
                                active_count=active_count,
                                progress=sparse_progress,
                                selection_progress=selection_progress,
                            )
                        indices = mask.nonzero(as_tuple=False)[:, 1].reshape(1, active_count)
                        model_input = gather_tokens(latents, indices)
                        compact_ids = gather_ids(generated_ids, indices)
                        image_indices = indices
                        action = "update_active"
                        phase = "sparse"
                        sparse_count += 1
                    context = None
                    if action != "none":
                        context = FluxSparseAttentionContext(
                            method="sparseedit",
                            kv_cache=cache,
                            kv_action=action,
                            image_indices=image_indices,
                            full_img_ids=latent_ids,
                            generated_token_count=count,
                            reference_token_count=reference_count,
                        )
                    attention = pipeline.joint_attention_kwargs
                    if context is not None:
                        attention = with_acceleration_context(attention, context, step_index=step)
                    timestep = t.expand(1).to(device=latents.device, dtype=latents.dtype) / 1000
                    output = pipeline.predict_noise(
                        **_positive_kwargs(
                            hidden_states=model_input,
                            timestep=timestep,
                            guidance=guidance,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            prompt_embeds=prompt_embeds,
                            text_ids=text_ids,
                            image_ids=compact_ids,
                            attention_kwargs=attention,
                        )
                    )
                    if dense:
                        noise = output[:, :count]
                        staleness.zero_()
                        since_dense = 0
                    else:
                        if previous_velocity is None:
                            raise RuntimeError("SparseEdit requires a dense velocity before sparse updates")
                        noise = previous_velocity.clone()
                        scatter_tokens_(noise, indices, output[:, :active_count])
                        staleness.add_(1)
                        staleness.scatter_(1, indices, 0)
                        since_dense += 1
                    last_computed_velocity = noise.detach().clone()
                    computed += 1
                record = {
                    "step": step,
                    "phase": phase,
                    "transformer_computed": compute,
                    "is_sparse_step": phase == "sparse",
                    "active_token_count": int(indices.shape[1]),
                    "active_ratio": indices.shape[1] / count,
                    "kv_action": action,
                    "sparse_avd_hit": not compute,
                    "reference_policy": "cached" if not dense else "full",
                }
                if mask_diagnostics is not None:
                    record["mask_diagnostics"] = mask_diagnostics
                if decision is not None:
                    record.update(
                        sparse_avd_reason=decision.reason,
                        sparse_avd_ratio=decision.ratio,
                        sparse_avd_error=decision.error,
                        sparse_avd_accumulate=decision.accumulate,
                    )
                if config.sparseedit_collect_warmup and step < warmup:
                    sigma = pipeline.scheduler.sigmas[step].to(latents.device, dtype=latents.dtype)
                    record.update(
                        collect_warmup_step_diagnostics(
                            latents=latents,
                            previous_latents=previous_latents,
                            previous_previous_latents=previous_previous_latents,
                            pred_x0=latents - sigma * noise,
                            reference_latents=image_latents,
                            grid_height=grid_height,
                            grid_width=grid_width,
                        )
                    )
                if gamma_stats is not None:
                    gamma_stats.observe(step_index=step, timestep=t, guided_velocity=noise)
                records.append(record)
                previous_previous_latents = previous_latents
                previous_latents = latents.detach().clone()
                previous_velocity = noise.detach().clone()
                latents = pipeline.scheduler.step(noise, t, latents, return_dict=False)[0]
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
    finally:
        cache.clear()
        pipeline._current_timestep = None
    metadata = {
        "schema_version": 1,
        "method": "sparseedit",
        "run_config": {
            **config.to_metadata(),
            "warmup_steps": warmup,
            "refresh_interval": refresh_interval,
            "ratio": config.sparseedit_ratio,
            "ratio_rounding": "floor",
            "strategy": "spatial_error_smooth_decay",
            "reference_policy": "frozen_after_warmup",
            "avd_final_steps_force": "transformer_only",
        },
        "step_records": records,
        "summary": {
            "transformer_compute_steps": computed,
            "avd_hit_steps": hits,
            "dense_steps": dense_count,
            "sparse_steps": sparse_count,
        },
    }
    if gamma_stats is not None:
        metadata["sparse_avd_gamma_stats"] = gamma_stats.to_dict()
    return latents, metadata
