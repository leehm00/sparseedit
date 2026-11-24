# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Denoising loops for FLUX.1 Kontext-dev acceleration baselines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from vllm_omni.diffusion.models.flux.acceleration import (
    FluxKontextAccelerationConfig,
    FluxRASState,
    FluxRegionEState,
    FluxSparseAttentionContext,
    FluxSparseKVCache,
    FluxSpotEditConfig,
    FluxSpotEditState,
    FluxToCaState,
    boundary_aware_smoothing,
    gather_ids,
    gather_tokens,
    scatter_tokens_,
    spotedit_alpha,
    with_acceleration_context,
)


def diffuse_with_flux_kontext_acceleration(
    pipeline: Any,
    *,
    config: FluxKontextAccelerationConfig,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    negative_pooled_prompt_embeds: torch.Tensor | None,
    latents: torch.Tensor,
    image_latents: torch.Tensor | None,
    latent_ids: torch.Tensor,
    text_ids: torch.Tensor,
    negative_text_ids: torch.Tensor | None,
    timesteps: torch.Tensor,
    do_true_cfg: bool,
    guidance: torch.Tensor | None,
    true_cfg_scale: float,
    height: int,
    width: int,
    callback_on_step_end: Callable[[int, int, dict], None] | None = None,
    callback_on_step_end_tensor_inputs: list[str] | None = None,
    request_state=None,
):
    """Run ToCa, RAS, RegionE, or SpotEdit without storing state on the model."""

    grid_metadata = _validate_runtime(pipeline, config, latents, image_latents, latent_ids, do_true_cfg)
    if config.method == "regione" and callback_on_step_end is not None:
        raise ValueError("FLUX RegionE does not expose callbacks while the latent sequence is compact")
    if image_latents is None:
        raise ValueError(f"FLUX {config.method} requires one reference image")
    generated_grid = grid_metadata["generated_grid"]
    grid_height = generated_grid["height"]
    grid_width = generated_grid["width"]
    callback_inputs = callback_on_step_end_tensor_inputs or ["latents"]
    method_metadata = {}
    paper_sparseedit = config.method == "sparseedit" and config.sparseedit_selector_protocol == "paper"

    def publish(metadata, diagnostics):
        combined = {
            "config": config.to_metadata(),
            "grid_height": grid_height,
            "grid_width": grid_width,
            **grid_metadata,
            "resolved_height": height,
            "resolved_width": width,
            "diagnostics": diagnostics,
            **metadata,
        }
        pipeline._last_flux_acceleration_metadata = combined
        if request_state is not None:
            request_state.metadata = combined

    if request_state is not None and not paper_sparseedit:
        raise ValueError("Study step executor requires the SparseEdit paper selector")

    if config.method == "toca":
        result, diagnostics = _diffuse_toca(
            pipeline,
            config=config,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            latents=latents,
            image_latents=image_latents,
            latent_ids=latent_ids,
            text_ids=text_ids,
            negative_text_ids=negative_text_ids,
            timesteps=timesteps,
            do_true_cfg=do_true_cfg,
            guidance=guidance,
            true_cfg_scale=true_cfg_scale,
            grid_height=grid_height,
            grid_width=grid_width,
            callback_on_step_end=callback_on_step_end,
            callback_inputs=callback_inputs,
        )
    elif config.method == "ras":
        result, diagnostics = _diffuse_ras(
            pipeline,
            config=config,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            latents=latents,
            image_latents=image_latents,
            latent_ids=latent_ids,
            text_ids=text_ids,
            timesteps=timesteps,
            guidance=guidance,
            callback_on_step_end=callback_on_step_end,
            callback_inputs=callback_inputs,
        )
    elif config.method == "regione":
        result, diagnostics = _diffuse_regione(
            pipeline,
            config=config,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            latents=latents,
            image_latents=image_latents,
            latent_ids=latent_ids,
            text_ids=text_ids,
            timesteps=timesteps,
            guidance=guidance,
            grid_height=grid_height,
            grid_width=grid_width,
            callback_on_step_end=callback_on_step_end,
            callback_inputs=callback_inputs,
        )
    elif config.method == "spotedit":
        result, diagnostics = _diffuse_spotedit(
            pipeline,
            config=config,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            latents=latents,
            image_latents=image_latents,
            latent_ids=latent_ids,
            text_ids=text_ids,
            timesteps=timesteps,
            guidance=guidance,
            grid_height=grid_height,
            grid_width=grid_width,
            height=height,
            width=width,
            callback_on_step_end=callback_on_step_end,
            callback_inputs=callback_inputs,
        )
    elif config.method == "sparseedit":
        if paper_sparseedit:
            from vllm_omni.diffusion.models.flux.paper_selector_runtime import (
                diffuse_sparseedit_paper, iter_sparseedit_paper,
            )

            paper_kwargs = dict(
                config=config,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                latents=latents,
                image_latents=image_latents,
                latent_ids=latent_ids,
                text_ids=text_ids,
                negative_text_ids=negative_text_ids,
                timesteps=timesteps,
                do_true_cfg=do_true_cfg,
                guidance=guidance,
                true_cfg_scale=true_cfg_scale,
                grid_height=grid_height,
                grid_width=grid_width,
                callback_on_step_end=callback_on_step_end,
                callback_inputs=callback_inputs,
            )
            if request_state is not None:
                def steps():
                    result, metadata = yield from iter_sparseedit_paper(
                        pipeline, request_state=request_state, **paper_kwargs
                    )
                    publish(metadata, metadata["step_records"])
                    return result

                return steps()
            result, method_metadata = diffuse_sparseedit_paper(pipeline, **paper_kwargs)
        else:
            from vllm_omni.diffusion.models.flux.sparseedit_runtime import diffuse_sparseedit

            result, method_metadata = diffuse_sparseedit(
                pipeline,
                config=config,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                latents=latents,
                image_latents=image_latents,
                latent_ids=latent_ids,
                text_ids=text_ids,
                timesteps=timesteps,
                guidance=guidance,
                grid_height=grid_height,
                grid_width=grid_width,
                callback_on_step_end=callback_on_step_end,
                callback_inputs=callback_inputs,
            )
        diagnostics = method_metadata["step_records"]
    else:
        raise ValueError(f"method {config.method} does not use the custom FLUX denoising loop")

    publish(method_metadata, diagnostics)
    return result


def _validate_runtime(
    pipeline: Any,
    config: FluxKontextAccelerationConfig,
    latents: torch.Tensor,
    image_latents: torch.Tensor | None,
    latent_ids: torch.Tensor,
    do_true_cfg: bool,
) -> dict[str, dict[str, int]]:
    if latents.shape[0] != 1:
        raise ValueError(f"FLUX {config.method} currently supports batch size 1")
    if image_latents is None or image_latents.shape[0] != 1 or image_latents.shape[1] == 0:
        raise ValueError(f"FLUX {config.method} requires one original reference image")
    cache_backend = getattr(pipeline.od_config, "cache_backend", "none")
    if cache_backend not in (None, "", "none"):
        raise ValueError(f"FLUX {config.method} cannot be combined with cache_backend={cache_backend}")
    parallel = getattr(pipeline.od_config, "parallel_config", None)
    if parallel is not None:
        if getattr(parallel, "sequence_parallel_size", 1) > 1:
            raise ValueError(f"FLUX {config.method} currently requires sequence parallel size 1")
        if getattr(parallel, "cfg_parallel_size", 1) > 1:
            raise ValueError(f"FLUX {config.method} currently requires CFG parallel size 1")
    paper_sparseedit = config.method == "sparseedit" and config.sparseedit_selector_protocol == "paper"
    legacy_cfg_rejected = config.method in {"ras", "regione", "spotedit"} or (
        config.method == "sparseedit" and not paper_sparseedit
    )
    if do_true_cfg and legacy_cfg_rejected:
        raise ValueError(f"FLUX {config.method} currently requires true_cfg_scale=1.0")
    grid_metadata = _image_grid_metadata(latents, image_latents, latent_ids)
    if config.method in {"regione", "spotedit", "sparseedit"}:
        generated_token_count = latents.shape[1]
        reference_token_count = image_latents.shape[1]
        if generated_token_count != reference_token_count:
            raise ValueError(
                f"FLUX {config.method} requires token-aligned generated and reference images: "
                f"generated={generated_token_count}, reference={reference_token_count}"
            )
        generated_ids = latent_ids[:generated_token_count]
        reference_ids = latent_ids[generated_token_count:]
        generated_grid = grid_metadata["generated_grid"]
        reference_grid = grid_metadata["reference_grid"]
        if generated_grid != reference_grid:
            raise ValueError(
                f"FLUX {config.method} requires matching generated and reference spatial grids: "
                f"generated={generated_grid['height']}x{generated_grid['width']}, "
                f"reference={reference_grid['height']}x{reference_grid['width']}"
            )
        if not torch.equal(generated_ids[:, 1:], reference_ids[:, 1:]):
            raise ValueError(
                f"FLUX {config.method} requires matching generated and reference spatial token coordinates"
            )
    return grid_metadata


def _image_grid_metadata(
    latents: torch.Tensor,
    image_latents: torch.Tensor,
    latent_ids: torch.Tensor,
) -> dict[str, dict[str, int]]:
    generated_token_count = latents.shape[1]
    reference_token_count = image_latents.shape[1]
    expected_id_count = generated_token_count + reference_token_count
    if latent_ids.ndim != 2 or latent_ids.shape[0] != expected_id_count:
        raise ValueError(
            f"FLUX Kontext requires {expected_id_count} generated/reference image IDs, "
            f"got shape {tuple(latent_ids.shape)}"
        )
    generated_height, generated_width = _infer_grid(latent_ids[:generated_token_count])
    reference_height, reference_width = _infer_grid(latent_ids[generated_token_count:])
    return {
        "generated_grid": {
            "height": generated_height,
            "width": generated_width,
            "token_count": generated_token_count,
        },
        "reference_grid": {
            "height": reference_height,
            "width": reference_width,
            "token_count": reference_token_count,
        },
    }


def _infer_grid(ids: torch.Tensor) -> tuple[int, int]:
    if ids.ndim != 2 or ids.shape[1] < 3:
        raise ValueError("FLUX latent ids must have shape [tokens, at least three axes]")
    height = int(torch.unique(ids[:, 1]).numel())
    width = int(torch.unique(ids[:, 2]).numel())
    if height * width != ids.shape[0]:
        raise ValueError(
            f"FLUX image token ids do not cover a complete rectangular grid: {height}x{width} != {ids.shape[0]}"
        )
    spatial_ids = ids[:, 1:3]
    y_ids = spatial_ids[:, 0]
    x_ids = spatial_ids[:, 1]
    valid_coordinates = (
        torch.isfinite(spatial_ids).all()
        and torch.equal(y_ids, torch.round(y_ids))
        and torch.equal(x_ids, torch.round(x_ids))
        and bool(torch.all((y_ids >= 0) & (y_ids < height)))
        and bool(torch.all((x_ids >= 0) & (x_ids < width)))
    )
    if valid_coordinates:
        linear_ids = y_ids.to(dtype=torch.long) * width + x_ids.to(dtype=torch.long)
        valid_coordinates = torch.unique(linear_ids).numel() == ids.shape[0]
    if not valid_coordinates:
        raise ValueError("FLUX image token ids do not cover a complete rectangular grid")

    expected_y = torch.arange(height, device=ids.device, dtype=ids.dtype).repeat_interleave(width)
    expected_x = torch.arange(width, device=ids.device, dtype=ids.dtype).repeat(height)
    expected_spatial_ids = torch.stack([expected_y, expected_x], dim=1)
    if not torch.equal(spatial_ids, expected_spatial_ids):
        raise ValueError("FLUX image token ids do not use row-major spatial token order")
    return height, width


def _positive_kwargs(
    *,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    guidance: torch.Tensor | None,
    pooled_prompt_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    image_ids: torch.Tensor,
    attention_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "guidance": guidance,
        "pooled_projections": pooled_prompt_embeds,
        "encoder_hidden_states": prompt_embeds,
        "txt_ids": text_ids,
        "img_ids": image_ids,
        "joint_attention_kwargs": attention_kwargs,
        "return_dict": False,
    }


def _apply_callback(
    pipeline: Any,
    callback: Callable[[int, int, dict], None] | None,
    callback_inputs: list[str],
    *,
    step_index: int,
    timestep: torch.Tensor,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if callback is None:
        return latents, prompt_embeds
    local_values = {"latents": latents, "prompt_embeds": prompt_embeds}
    outputs = callback(pipeline, step_index, timestep, {name: local_values[name] for name in callback_inputs})
    return outputs.pop("latents", latents), outputs.pop("prompt_embeds", prompt_embeds)


def _diffuse_toca(
    pipeline: Any,
    *,
    config: FluxKontextAccelerationConfig,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    negative_pooled_prompt_embeds: torch.Tensor | None,
    latents: torch.Tensor,
    image_latents: torch.Tensor,
    latent_ids: torch.Tensor,
    text_ids: torch.Tensor,
    negative_text_ids: torch.Tensor | None,
    timesteps: torch.Tensor,
    do_true_cfg: bool,
    guidance: torch.Tensor | None,
    true_cfg_scale: float,
    grid_height: int,
    grid_width: int,
    callback_on_step_end: Callable[[int, int, dict], None] | None,
    callback_inputs: list[str],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    total_layers = len(pipeline.transformer.transformer_blocks) + len(pipeline.transformer.single_transformer_blocks)
    positive_state = FluxToCaState(
        config,
        total_layers,
        latents.shape[1],
        image_latents.shape[1],
        grid_height,
        grid_width,
    )
    negative_state = (
        FluxToCaState(
            config,
            total_layers,
            latents.shape[1],
            image_latents.shape[1],
            grid_height,
            grid_width,
        )
        if do_true_cfg
        else None
    )
    pipeline.scheduler.set_begin_index(0)
    with pipeline.progress_bar(total=len(timesteps)) as progress_bar:
        for step_index, timestep_value in enumerate(timesteps):
            pipeline._current_timestep = timestep_value
            model_input = torch.cat([latents, image_latents], dim=1)
            timestep = timestep_value.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype) / 1000
            positive = _positive_kwargs(
                hidden_states=model_input,
                timestep=timestep,
                guidance=guidance,
                pooled_prompt_embeds=pooled_prompt_embeds,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                image_ids=latent_ids,
                attention_kwargs=with_acceleration_context(
                    pipeline.joint_attention_kwargs,
                    positive_state,
                    step_index=step_index,
                ),
            )
            negative = None
            if do_true_cfg:
                if negative_prompt_embeds is None or negative_pooled_prompt_embeds is None or negative_text_ids is None:
                    raise ValueError("FLUX ToCa true CFG requires negative prompt embeddings and ids")
                negative = _positive_kwargs(
                    hidden_states=model_input,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    prompt_embeds=negative_prompt_embeds,
                    text_ids=negative_text_ids,
                    image_ids=latent_ids,
                    attention_kwargs=with_acceleration_context(
                        pipeline.joint_attention_kwargs,
                        negative_state,
                        step_index=step_index,
                    ),
                )
            noise = pipeline.predict_noise_maybe_with_cfg(
                do_true_cfg,
                true_cfg_scale,
                positive,
                negative,
                cfg_normalize=False,
                output_slice=latents.shape[1],
            )
            latents = pipeline.scheduler_step_maybe_with_cfg(noise, timestep_value, latents, do_true_cfg)
            latents, prompt_embeds = _apply_callback(
                pipeline,
                callback_on_step_end,
                callback_inputs,
                step_index=step_index,
                timestep=timestep_value,
                latents=latents,
                prompt_embeds=prompt_embeds,
            )
            progress_bar.update()
    pipeline._current_timestep = None
    diagnostics = list(positive_state.diagnostics)
    if negative_state is not None:
        diagnostics.extend({**record, "cfg_branch": "negative"} for record in negative_state.diagnostics)
    return latents, diagnostics


def _diffuse_ras(
    pipeline: Any,
    *,
    config: FluxKontextAccelerationConfig,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    latents: torch.Tensor,
    image_latents: torch.Tensor,
    latent_ids: torch.Tensor,
    text_ids: torch.Tensor,
    timesteps: torch.Tensor,
    guidance: torch.Tensor | None,
    callback_on_step_end: Callable[[int, int, dict], None] | None,
    callback_inputs: list[str],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    state = FluxRASState(config, latents.shape[1])
    kv_cache = FluxSparseKVCache()
    generated_count = latents.shape[1]
    reference_count = image_latents.shape[1]
    generated_ids = latent_ids[:generated_count]
    reference_ids = latent_ids[generated_count:]
    pipeline.scheduler.set_begin_index(0)
    with pipeline.progress_bar(total=len(timesteps)) as progress_bar:
        for step_index, timestep_value in enumerate(timesteps):
            pipeline._current_timestep = timestep_value
            decision = state.decision(step_index, device=latents.device)
            generated_input = latents if decision.dense else gather_tokens(latents, decision.active_indices)
            model_input = torch.cat([generated_input, image_latents], dim=1)
            compact_ids = torch.cat([gather_ids(generated_ids, decision.active_indices), reference_ids], dim=0)
            reference_indices = torch.arange(
                generated_count,
                generated_count + reference_count,
                device=latents.device,
                dtype=torch.long,
            ).unsqueeze(0)
            image_indices = torch.cat([decision.active_indices, reference_indices], dim=1)
            context = FluxSparseAttentionContext(
                method="ras",
                kv_cache=kv_cache,
                kv_action="store_full" if decision.dense else "update_active",
                image_indices=image_indices,
                full_img_ids=latent_ids,
                generated_token_count=generated_count,
                reference_token_count=reference_count,
            )
            timestep = timestep_value.expand(1).to(device=latents.device, dtype=latents.dtype) / 1000
            output = pipeline.predict_noise(
                **_positive_kwargs(
                    hidden_states=model_input,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    image_ids=compact_ids,
                    attention_kwargs=with_acceleration_context(
                        pipeline.joint_attention_kwargs,
                        context,
                        step_index=step_index,
                    ),
                )
            )
            full_noise = state.reconstruct(decision, output[:, : decision.active_count])
            latents = pipeline.scheduler.step(full_noise, timestep_value, latents, return_dict=False)[0]
            latents, prompt_embeds = _apply_callback(
                pipeline,
                callback_on_step_end,
                callback_inputs,
                step_index=step_index,
                timestep=timestep_value,
                latents=latents,
                prompt_embeds=prompt_embeds,
            )
            state.diagnostics.append(
                {
                    "step": step_index,
                    "execution": "dense" if decision.dense else "compact",
                    "dense_reason": decision.dense_reason,
                    "requested_ratio": decision.requested_ratio,
                    "active_generated_tokens": decision.active_count,
                    "reference_tokens": reference_count,
                    "kv_action": context.kv_action,
                }
            )
            progress_bar.update()
    pipeline._current_timestep = None
    kv_cache.clear()
    return latents, state.diagnostics


def _diffuse_regione(
    pipeline: Any,
    *,
    config: FluxKontextAccelerationConfig,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    latents: torch.Tensor,
    image_latents: torch.Tensor,
    latent_ids: torch.Tensor,
    text_ids: torch.Tensor,
    timesteps: torch.Tensor,
    guidance: torch.Tensor | None,
    grid_height: int,
    grid_width: int,
    callback_on_step_end: Callable[[int, int, dict], None] | None,
    callback_inputs: list[str],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    generated_count = latents.shape[1]
    reference_count = image_latents.shape[1]
    state = FluxRegionEState(config, generated_count, grid_height, grid_width)
    kv_cache = FluxSparseKVCache()
    generated_ids = latent_ids[:generated_count]
    previous_timestep = None
    pipeline.scheduler.set_begin_index(0)
    sigmas = pipeline.scheduler.sigmas
    with pipeline.progress_bar(total=len(timesteps)) as progress_bar:
        for step_index, timestep_value in enumerate(timesteps):
            pipeline._current_timestep = timestep_value
            if step_index in state.merge_step_indices and latents.shape[1] != generated_count:
                latents = state.merge(latents)

            compute_transformer, avd_ratio, avd_reason = state.avd_decision(
                step_index,
                timestep_value,
                previous_timestep,
            )
            local_step = state.is_local_step(step_index)
            if local_step:
                if state.edited_indices is None:
                    raise RuntimeError("RegionE local step reached before adaptive region partition")
                active_indices = state.edited_indices
                current_generated = latents
                compact_ids = gather_ids(generated_ids, active_indices)
                model_input = current_generated
                image_indices = active_indices
                kv_action = "update_active"
            else:
                active_indices = torch.arange(generated_count, device=latents.device).unsqueeze(0)
                current_generated = latents
                compact_ids = latent_ids
                model_input = torch.cat([current_generated, image_latents], dim=1)
                image_indices = torch.arange(generated_count + reference_count, device=latents.device).unsqueeze(0)
                kv_action = "store_full" if state.is_full_cache_step(step_index) else "none"

            if compute_transformer:
                context = FluxSparseAttentionContext(
                    method="regione",
                    kv_cache=kv_cache,
                    kv_action=kv_action,
                    image_indices=image_indices,
                    full_img_ids=latent_ids,
                    generated_token_count=generated_count,
                    reference_token_count=reference_count,
                )
                timestep = timestep_value.expand(1).to(device=latents.device, dtype=latents.dtype) / 1000
                output = pipeline.predict_noise(
                    **_positive_kwargs(
                        hidden_states=model_input,
                        timestep=timestep,
                        guidance=guidance,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        prompt_embeds=prompt_embeds,
                        text_ids=text_ids,
                        image_ids=compact_ids,
                        attention_kwargs=with_acceleration_context(
                            pipeline.joint_attention_kwargs,
                            context,
                            step_index=step_index,
                        ),
                    )
                )
                velocity = output[:, : current_generated.shape[1]]
                state.update_velocity(velocity)
            else:
                if avd_ratio is None:
                    raise RuntimeError("RegionE AVD cache hit did not provide a velocity ratio")
                velocity = state.cached_velocity(avd_ratio)

            current_sigma = sigmas[step_index].to(device=latents.device, dtype=latents.dtype)
            normal_next = pipeline.scheduler.step(velocity, timestep_value, current_generated, return_dict=False)[0]
            if state.is_full_cache_step(step_index):
                if step_index == state.partition_step_index:
                    final_sigma = sigmas[-1].to(device=latents.device, dtype=latents.dtype)
                    estimated_x0 = current_generated + (final_sigma - current_sigma) * velocity
                    state.set_partition(estimated_x0, image_latents)
                if state.edited_indices is None or state.unedited_indices is None:
                    raise RuntimeError("RegionE partition indices are unavailable")
                next_merge = next(target for target in state.merge_step_indices if target > step_index)
                target_sigma = sigmas[next_merge].to(device=latents.device, dtype=latents.dtype)
                edited_next = gather_tokens(normal_next, state.edited_indices)
                unedited_current = gather_tokens(current_generated, state.unedited_indices)
                unedited_velocity = gather_tokens(velocity, state.unedited_indices)
                state.unedited_latents = unedited_current + (target_sigma - current_sigma) * unedited_velocity
                latents = edited_next
            else:
                latents = normal_next

            latents, prompt_embeds = _apply_callback(
                pipeline,
                callback_on_step_end,
                callback_inputs,
                step_index=step_index,
                timestep=timestep_value,
                latents=latents,
                prompt_embeds=prompt_embeds,
            )
            state.diagnostics.append(
                {
                    "step": step_index,
                    "execution": (
                        "full" if not local_step else ("local_compute" if compute_transformer else "avd_reuse")
                    ),
                    "active_generated_tokens": int(active_indices.shape[1]),
                    "reference_tokens": reference_count if not local_step else 0,
                    "kv_action": kv_action,
                    "avd_reason": avd_reason,
                    "avd_ratio": avd_ratio,
                }
            )
            previous_timestep = timestep_value
            progress_bar.update()
    pipeline._current_timestep = None
    if latents.shape[1] != generated_count:
        latents = state.merge(latents)
    kv_cache.clear()
    return latents, state.diagnostics


def _diffuse_spotedit(
    pipeline: Any,
    *,
    config: FluxKontextAccelerationConfig,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    latents: torch.Tensor,
    image_latents: torch.Tensor,
    latent_ids: torch.Tensor,
    text_ids: torch.Tensor,
    timesteps: torch.Tensor,
    guidance: torch.Tensor | None,
    grid_height: int,
    grid_width: int,
    height: int,
    width: int,
    callback_on_step_end: Callable[[int, int, dict], None] | None,
    callback_inputs: list[str],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    generated_count = latents.shape[1]
    reference_count = image_latents.shape[1]
    state = FluxSpotEditState(
        FluxSpotEditConfig.from_acceleration_config(config),
        pipeline.vae,
        generated_count,
        grid_height,
        grid_width,
    )
    kv_cache = FluxSparseKVCache()
    generated_ids = latent_ids[:generated_count]
    pipeline.scheduler.set_begin_index(0)
    with pipeline.progress_bar(total=len(timesteps)) as progress_bar:
        for step_index, timestep_value in enumerate(timesteps):
            pipeline._current_timestep = timestep_value
            scheduled_full = state.config.is_full_step(step_index) or state.previous_x0 is None
            if scheduled_full:
                reuse_mask = torch.zeros(generated_count, device=latents.device, dtype=torch.bool)
            elif state.config.select_every_step or state.reuse_mask is None:
                reuse_mask = state.select(
                    image_latents,
                    image_size=(height, width),
                    vae_scale_factor=pipeline.vae_scale_factor,
                ).to(device=latents.device)
            else:
                reuse_mask = state.reuse_mask.to(device=latents.device)

            sparse = bool(reuse_mask.any())
            if sparse:
                active_indices = (~reuse_mask).nonzero(as_tuple=False).squeeze(-1).unsqueeze(0)
                model_input = gather_tokens(latents, active_indices)
                compact_ids = gather_ids(generated_ids, active_indices)
                image_indices = active_indices
                kv_action = "update_active"
            else:
                active_indices = torch.arange(generated_count, device=latents.device).unsqueeze(0)
                model_input = torch.cat([latents, image_latents], dim=1)
                compact_ids = latent_ids
                image_indices = torch.arange(generated_count + reference_count, device=latents.device).unsqueeze(0)
                kv_action = "store_full"

            sigma = float(timestep_value.detach().float().item()) / 1000.0
            context = FluxSparseAttentionContext(
                method="spotedit",
                kv_cache=kv_cache,
                kv_action=kv_action,
                image_indices=image_indices,
                full_img_ids=latent_ids,
                generated_token_count=generated_count,
                reference_token_count=reference_count,
                spotedit_alpha=spotedit_alpha(sigma, profile=state.config.profile),
            )
            timestep = timestep_value.expand(1).to(device=latents.device, dtype=latents.dtype) / 1000
            output = pipeline.predict_noise(
                **_positive_kwargs(
                    hidden_states=model_input,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    image_ids=compact_ids,
                    attention_kwargs=with_acceleration_context(
                        pipeline.joint_attention_kwargs,
                        context,
                        step_index=step_index,
                    ),
                )
            )
            compact_noise = output[:, : active_indices.shape[1]]
            if sparse:
                if state.previous_full_noise is None:
                    raise RuntimeError("SpotEdit sparse step reached before a full velocity was cached")
                full_noise = state.previous_full_noise.clone()
                scatter_tokens_(full_noise, active_indices, compact_noise)
                if state.config.reuse_mode == "velocity":
                    denominator = max(sigma, torch.finfo(latents.dtype).eps)
                    straight_velocity = (latents - image_latents) / denominator
                    full_noise[:, reuse_mask] = straight_velocity[:, reuse_mask]
            else:
                full_noise = compact_noise
            state.previous_full_noise = full_noise.detach().clone()
            state.previous_x0 = (latents - sigma * full_noise).detach()
            latents = pipeline.scheduler.step(full_noise, timestep_value, latents, return_dict=False)[0]
            latents, prompt_embeds = _apply_callback(
                pipeline,
                callback_on_step_end,
                callback_inputs,
                step_index=step_index,
                timestep=timestep_value,
                latents=latents,
                prompt_embeds=prompt_embeds,
            )
            state.diagnostics.append(
                {
                    "step": step_index,
                    "execution": "compact" if sparse else "full",
                    "active_generated_tokens": int(active_indices.shape[1]),
                    "reused_generated_tokens": int(reuse_mask.sum().item()),
                    "reference_tokens": 0 if sparse else reference_count,
                    "kv_action": kv_action,
                    "spotedit_alpha": context.spotedit_alpha,
                }
            )
            progress_bar.update()
    pipeline._current_timestep = None
    if state.final_reuse_mask is not None:
        final_mask = state.final_reuse_mask.to(device=latents.device).unsqueeze(0)
        if state.config.reuse_mode == "overwrite":
            latents = torch.where(final_mask.unsqueeze(-1), image_latents, latents)
        latents = boundary_aware_smoothing(
            latents,
            image_latents,
            final_mask,
            grid_height=grid_height,
            grid_width=grid_width,
            weight=0.8,
        )
    kv_cache.clear()
    return latents, state.diagnostics


__all__ = ["diffuse_with_flux_kontext_acceleration"]
