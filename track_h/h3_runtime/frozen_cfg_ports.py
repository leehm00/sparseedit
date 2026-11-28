# SPDX-License-Identifier: Apache-2.0
"""Explicit serial CFG adaptations of local FLUX RAS and RegionE.

RAS composes independent native selectors. RegionE shares geometry and guided
AVD while isolating all branch K/V. Admission observation is opt-in, internal,
and copies tensors to CPU; timed benchmark requests must leave it disabled.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import math
from typing import Any

import torch

from vllm_omni.diffusion.models.flux.acceleration import (
    FluxRASState, FluxRegionEState, FluxSparseAttentionContext, FluxSparseKVCache,
    gather_ids, gather_tokens, scatter_tokens_, with_acceleration_context,
)

METHODS = {'ras_cfg_compose': 'ras', 'regione_cfg_isolated': 'regione'}


def _copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy(v) for v in value)
    return value


def _cache_witness(cache, generated_count):
    """Sample boundary rows in every actual layer/stream, including references."""
    result = {}
    for (stream, layer), (key, value) in sorted(cache.layers.items()):
        rows = sorted({0, generated_count - 1, generated_count, key.shape[1] - 1})
        rows = [i for i in rows if 0 <= i < key.shape[1]]
        result[f'{stream}:{layer}'] = {
            'shape': list(key.shape), 'dtype': str(key.dtype), 'rows': rows,
            'key': key[:, rows], 'value': value[:, rows],
            'key_storage': key.untyped_storage().data_ptr(),
            'value_storage': value.untyped_storage().data_ptr(),
        }
    return result


def _forward(pipeline, kw, *, branch, sample, ids, context, step, timestep):
    from vllm_omni.diffusion.models.flux.acceleration_runtime import _positive_kwargs

    negative = branch == 'negative'
    return pipeline.predict_noise(**_positive_kwargs(
        hidden_states=sample,
        timestep=timestep.expand(1).to(device=sample.device, dtype=sample.dtype) / 1000,
        guidance=kw['guidance'],
        pooled_prompt_embeds=kw['negative_pooled_prompt_embeds'] if negative else kw['pooled_prompt_embeds'],
        prompt_embeds=kw['negative_prompt_embeds'] if negative else kw['prompt_embeds'],
        text_ids=kw['negative_text_ids'] if negative else kw['text_ids'], image_ids=ids,
        attention_kwargs=with_acceleration_context(pipeline.joint_attention_kwargs, context, step_index=step),
    ))


def _observe(pipeline, record, caches, generated_count):
    observer = getattr(pipeline, '_cfg_ports_admission_observer', None)
    if observer is None:
        return
    for branch in ('positive', 'negative'):
        record['branches'][branch]['cache_witness'] = _cache_witness(caches[branch], generated_count)
    # Observer receives no live tensor or mutable cache handle.
    observer(_copy(record))


def _branch_record(indices, reference_count, kv_action, velocity, *, computed, velocity_kind):
    return {'active_indices': indices, 'reference_tokens': reference_count, 'kv_action': kv_action,
            'computed': computed, 'full_velocity': velocity, 'velocity_kind': velocity_kind}


class DiskObserver:
    """Fresh, per-request admission capture; completion is emitted after all 28 steps."""

    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.config = config.to_metadata()
        self.inventory = {}
        self.sources = {}
        for name in ('cfg_ports_runtime.py', 'acceleration.py', 'acceleration_runtime.py', 'pipeline_flux_kontext.py'):
            path = Path(__file__).parent / name
            self.sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def __call__(self, record):
        step = len(self.inventory)
        if record['step'] != step or step >= 28:
            raise ValueError('admission capture steps must advance exactly 0..27')
        name = f'step_{step:02d}.pt'
        path = self.directory / name
        with path.open('xb') as handle:
            torch.save(record, handle)
        self.inventory[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def finish(self):
        if len(self.inventory) != 28:
            raise ValueError('incomplete 28-step admission capture')
        with (self.directory / 'capture.json').open('x') as handle:
            json.dump({'schema_version': 1, 'status': 'CAPTURED', 'steps': 28, 'config': self.config,
                       'sources': self.sources, 'inventory': self.inventory}, handle, indent=2)
            handle.write('\n')


def diffuse_cfg_port(pipeline, **kw):
    directory = kw['config'].cfg_ports_admission_directory
    if not directory:
        return _diffuse_cfg_port(pipeline, **kw)
    if getattr(pipeline, '_cfg_ports_admission_observer', None) is not None:
        raise ValueError('one admission observer per request')
    observer = DiskObserver(directory, kw['config'])
    prior = pipeline.__dict__.get('_cfg_ports_admission_observer', None)
    pipeline._cfg_ports_admission_observer = observer
    try:
        result = _diffuse_cfg_port(pipeline, **kw)
        observer.finish()
        return result
    finally:
        pipeline._cfg_ports_admission_observer = prior


def admission_diffuse(pipeline, **kw):
    """Delegate the native call, optionally observing the actual dense branch path.

    Port captures are handled inside the port. Dense observation wraps native
    prediction/scheduler methods for one call and restores their exact binding.
    """
    config = kw['acceleration_config']
    directory = config.cfg_ports_admission_directory
    if not directory or config.method in METHODS:
        return pipeline.diffuse(**kw)
    if config.method != 'dense' or not kw['do_true_cfg']:
        raise ValueError('native raw admission capture supports serial true-CFG dense only')
    if getattr(pipeline, '_cfg_ports_admission_observer', None) is not None:
        raise ValueError('one admission observer per request')
    # Validate backend/layout/parallel before recording the native dense control.
    from vllm_omni.diffusion.models.flux.acceleration_runtime import _validate_runtime
    _validate_runtime(pipeline, replace(config, method='ras'), kw['latents'], kw['image_latents'], kw['latent_ids'], False)
    for name in ('negative_prompt_embeds', 'negative_pooled_prompt_embeds', 'negative_text_ids'):
        if not isinstance(kw[name], torch.Tensor) or not kw[name].numel():
            raise ValueError('native dense capture requires negative conditions')
    observer = DiskObserver(directory, config)
    n, m = kw['latents'].shape[1], kw['image_latents'].shape[1]
    predictions = []
    original_predict = pipeline.predict_noise
    original_scheduler = pipeline.scheduler_step_maybe_with_cfg
    absent = object()
    previous = {name: pipeline.__dict__.get(name, absent)
                for name in ('predict_noise', 'scheduler_step_maybe_with_cfg')}

    def predict(**inputs):
        value = original_predict(**inputs)
        predictions.append(value[:, :n].detach().clone())
        return value

    def scheduler(velocity, t, sample, do_true_cfg, **options):
        if not do_true_cfg or len(predictions) != 2:
            raise ValueError('native dense capture requires two serial branch predictions per step')
        pre = sample.detach().clone()
        post = original_scheduler(velocity, t, sample, do_true_cfg, **options)
        indices = torch.arange(n, device=sample.device).unsqueeze(0)
        branches = {b: _branch_record(indices, m, 'none', predictions[i], computed=True,
                                      velocity_kind='full_computed')
                    for i, b in enumerate(('positive', 'negative'))}
        for branch in branches.values(): branch['cache_witness'] = {}
        observer(_copy({'step': len(observer.inventory), 'execution': 'dense', 'pre_latents': pre,
            'post_latents': post, 'scheduler_pre_latents': pre, 'scheduler_post_latents': post,
            'guided_velocity': velocity, 'guided_velocity_layout': 'full', 'branches': branches,
            'unedited_indices': torch.empty(1, 0, dtype=torch.long), 'unedited_projection_target_step': None}))
        predictions.clear()
        return post

    pipeline.predict_noise = predict
    pipeline.scheduler_step_maybe_with_cfg = scheduler
    try:
        result = pipeline.diffuse(**kw)
        observer.finish()
        return result
    finally:
        for name, value in previous.items():
            if value is absent: delattr(pipeline, name)
            else: setattr(pipeline, name, value)
        pipeline._current_timestep = None


def _diffuse_cfg_port(pipeline, **kw):
    from vllm_omni.diffusion.models.flux.acceleration_runtime import (
        _validate_runtime, _diffuse_ras, _diffuse_regione,
    )
    config = kw['config']
    if config.method not in METHODS or config.cfg_ports_version != 'serial_v1':
        raise ValueError('explicit serial_v1 CFG port method required')
    legacy = replace(config, method=METHODS[config.method])
    # Reuse every native layout/backend/parallel guard; original CFG guard remains intact.
    grid = _validate_runtime(pipeline, legacy, kw['latents'], kw['image_latents'], kw['latent_ids'], False)
    if config.method == 'regione_cfg_isolated' and kw['callback_on_step_end'] is not None:
        raise ValueError('FLUX RegionE does not expose callbacks while the latent sequence is compact')
    if kw['do_true_cfg']:
        for name in ('negative_prompt_embeds', 'negative_pooled_prompt_embeds', 'negative_text_ids'):
            value = kw[name]
            if not isinstance(value, torch.Tensor) or value.numel() == 0 or not torch.isfinite(value).all():
                raise ValueError('CFG port requires complete nonempty negative conditions: ' + name)
        if (kw['negative_prompt_embeds'].ndim != 3 or kw['negative_prompt_embeds'].shape[0] != 1
                or kw['negative_pooled_prompt_embeds'].ndim != 2
                or kw['negative_pooled_prompt_embeds'].shape[0] != 1
                or kw['negative_text_ids'].ndim != 2
                or kw['negative_text_ids'].shape[0] != kw['negative_prompt_embeds'].shape[1]):
            raise ValueError('CFG port negative conditions have incompatible batch/text layouts')
        if not math.isfinite(kw['true_cfg_scale']) or kw['true_cfg_scale'] <= 1:
            raise ValueError('true CFG requires a finite scale greater than one')
    if len(kw['timesteps']) != 28:
        raise ValueError('CFG port requires all 28 native timesteps')
    gh, gw = grid['generated_grid']['height'], grid['generated_grid']['width']
    if not kw['do_true_cfg'] and not config.cfg_ports_full_execution:
        if getattr(pipeline, '_cfg_ports_admission_observer', None) is not None:
            raise ValueError('CFG1 compatibility delegates the unchanged loop; use unobserved comparison')
        delegated = {key: kw[key] for key in (
            'prompt_embeds', 'pooled_prompt_embeds', 'latents', 'image_latents', 'latent_ids',
            'text_ids', 'timesteps', 'guidance', 'callback_on_step_end')}
        delegated.update(config=legacy, callback_inputs=kw['callback_on_step_end_tensor_inputs'] or ['latents'])
        if legacy.method == 'regione': delegated.update(grid_height=gh, grid_width=gw)
        try:
            result, diagnostics = (_diffuse_ras if legacy.method == 'ras' else _diffuse_regione)(pipeline, **delegated)
        finally:
            pipeline._current_timestep = None
    else:
        if not kw['do_true_cfg']:
            raise ValueError('full-execution admission control requires true CFG')
        result, diagnostics = (_ras if legacy.method == 'ras' else _regione)(pipeline, kw, gh, gw)
    pipeline._last_flux_acceleration_metadata = {
        'config': config.to_metadata(), **grid, 'grid_height': gh, 'grid_width': gw,
        'resolved_height': kw['height'], 'resolved_width': kw['width'], 'diagnostics': diagnostics,
        'cfg_port_identity': ('local_ras_independent_branch_composition' if legacy.method == 'ras'
                              else 'regione_branch_isolated_shared_guided_geometry'),
        'observation_enabled': getattr(pipeline, '_cfg_ports_admission_observer', None) is not None,
    }
    return result


def _ras(pipeline, kw, gh, gw):
    from vllm_omni.diffusion.models.flux.acceleration_runtime import _apply_callback

    del gh, gw
    config = kw['config']; latents = kw['latents']; reference = kw['image_latents']
    n, m = latents.shape[1], reference.shape[1]
    states = {b: FluxRASState(config, n) for b in ('positive', 'negative')}
    caches = {b: FluxSparseKVCache() for b in states}
    generated_ids, reference_ids = kw['latent_ids'][:n], kw['latent_ids'][n:]
    refs = torch.arange(n, n + m, device=latents.device).unsqueeze(0)
    diagnostics = []
    try:
        pipeline.scheduler.set_begin_index(0)
        with pipeline.progress_bar(total=28) as bar:
            for step, t in enumerate(kw['timesteps']):
                if getattr(pipeline, 'interrupt', False):
                    raise RuntimeError('CFG port request interrupted before complete admission')
                pipeline._current_timestep = t
                pre = latents
                branches = {}; velocities = {}; work = {}
                for branch, state in states.items():
                    decision = state.decision(step, device=latents.device)
                    active = decision.active_indices
                    sample = torch.cat([latents if decision.dense else gather_tokens(latents, active), reference], 1)
                    ids = torch.cat([gather_ids(generated_ids, active), reference_ids])
                    action = 'store_full' if decision.dense else 'update_active'
                    context = FluxSparseAttentionContext(
                        method='ras', kv_cache=caches[branch], kv_action=action,
                        image_indices=torch.cat([active, refs], 1), full_img_ids=kw['latent_ids'],
                        generated_token_count=n, reference_token_count=m)
                    raw = _forward(pipeline, kw, branch=branch, sample=sample, ids=ids, context=context, step=step, timestep=t)
                    velocities[branch] = state.reconstruct(decision, raw[:, :decision.active_count])
                    branches[branch] = _branch_record(active, m, action, velocities[branch],
                                                      computed=True, velocity_kind='reconstructed_branch')
                    work[branch] = {'active_generated_tokens': decision.active_count, 'reference_tokens': m,
                                    'kv_action': action, 'dense_reason': decision.dense_reason,
                                    'requested_ratio': decision.requested_ratio,
                                    'active_indices': active.detach().cpu().tolist()}
                guided = pipeline.combine_cfg_noise(velocities['positive'], velocities['negative'],
                                                     kw['true_cfg_scale'], cfg_normalize=False)
                latents = pipeline.scheduler_step_maybe_with_cfg(guided, t, latents, True)
                latents, kw['prompt_embeds'] = _apply_callback(
                    pipeline, kw['callback_on_step_end'], kw['callback_on_step_end_tensor_inputs'] or ['latents'],
                    step_index=step, timestep=t, latents=latents, prompt_embeds=kw['prompt_embeds'])
                execution = 'dense' if all(x['kv_action'] == 'store_full' for x in branches.values()) else 'compact'
                diagnostics.append({'step': step, 'execution': execution, 'branches': work,
                                    'scheduler_updates': 1, 'branch_forward_count': 2})
                _observe(pipeline, {'step': step, 'execution': execution, 'pre_latents': pre,
                    'post_latents': latents, 'scheduler_pre_latents': pre, 'scheduler_post_latents': latents,
                    'guided_velocity': guided, 'guided_velocity_layout': 'full', 'branches': branches,
                    'unedited_indices': torch.empty(1, 0, device=latents.device, dtype=torch.long),
                    'unedited_projection_target_step': None}, caches, n)
                bar.update()
    finally:
        for cache in caches.values(): cache.clear()
        pipeline._current_timestep = None
    return latents, diagnostics


def _regione(pipeline, kw, gh, gw):
    config = kw['config']; latents = kw['latents']; reference = kw['image_latents']
    n, m = latents.shape[1], reference.shape[1]
    state = FluxRegionEState(config, n, gh, gw)
    caches = {b: FluxSparseKVCache() for b in ('positive', 'negative')}
    previous_full = {}; guided_full = None; previous_t = None; projection_target = None
    full_execution = config.cfg_ports_full_execution
    all_gen = torch.arange(n, device=latents.device).unsqueeze(0)
    all_image = torch.arange(n + m, device=latents.device).unsqueeze(0)
    sigmas = pipeline.scheduler.sigmas
    diagnostics = []

    def full_snapshot(sample):
        return state.merge(sample) if sample.shape[1] != n else sample

    try:
        pipeline.scheduler.set_begin_index(0)
        with pipeline.progress_bar(total=28) as bar:
            for step, t in enumerate(kw['timesteps']):
                if getattr(pipeline, 'interrupt', False):
                    raise RuntimeError('CFG port request interrupted before complete admission')
                pipeline._current_timestep = t
                if not full_execution and step in state.merge_step_indices:
                    if latents.shape[1] != n:
                        latents = state.merge(latents)
                    projection_target = None
                pre = full_snapshot(latents)
                if full_execution:
                    compute, ratio, reason = True, None, 'full_execution_control'
                else:
                    compute, ratio, reason = state.avd_decision(step, t, previous_t)
                local = not full_execution and state.is_local_step(step)
                cache_step = not full_execution and state.is_full_cache_step(step)
                active = state.edited_indices if local else all_gen
                if active is None: raise RuntimeError('RegionE local step before partition')
                current = latents
                ids = gather_ids(kw['latent_ids'][:n], active) if local else kw['latent_ids']
                sample = current if local else torch.cat([current, reference], 1)
                image_indices = active if local else all_image
                action = 'update_active' if local else ('store_full' if cache_step else 'none')
                branches = {}; velocities = {}
                if compute:
                    for branch, cache in caches.items():
                        context = FluxSparseAttentionContext(
                            method='regione', kv_cache=cache, kv_action=action, image_indices=image_indices,
                            full_img_ids=kw['latent_ids'], generated_token_count=n, reference_token_count=m)
                        output = _forward(pipeline, kw, branch=branch, sample=sample, ids=ids,
                                          context=context, step=step, timestep=t)
                        velocities[branch] = output[:, :current.shape[1]]
                        if local:
                            full = previous_full[branch].clone()
                            scatter_tokens_(full, active, velocities[branch])
                        else: full = velocities[branch]
                        previous_full[branch] = full.detach().clone()
                    velocity = pipeline.combine_cfg_noise(velocities['positive'], velocities['negative'],
                                                          kw['true_cfg_scale'], cfg_normalize=False)
                    state.update_velocity(velocity)
                else:
                    if ratio is None: raise RuntimeError('RegionE AVD hit without ratio')
                    velocity = state.cached_velocity(ratio)
                if local:
                    audit_guided = guided_full.clone()
                    scatter_tokens_(audit_guided, active, velocity)
                else: audit_guided = velocity
                guided_full = audit_guided.detach().clone()
                for branch in caches:
                    branches[branch] = _branch_record(active, 0 if local else m,
                        action if compute else 'skipped', previous_full[branch], computed=compute,
                        velocity_kind=('cached_last_computed_branch' if not compute else
                                       ('local_reconstructed_last_computed' if local else 'full_computed')))
                    branches[branch]['computed_velocity'] = velocities.get(branch)
                normal_next = pipeline.scheduler_step_maybe_with_cfg(velocity, t, current, True)
                if cache_step:
                    sigma = sigmas[step].to(device=latents.device, dtype=latents.dtype)
                    if step == state.partition_step_index:
                        final = sigmas[-1].to(device=latents.device, dtype=latents.dtype)
                        state.set_partition(current + (final - sigma) * velocity, reference)
                    projection_target = next(target for target in state.merge_step_indices if target > step)
                    target = sigmas[projection_target].to(device=latents.device, dtype=latents.dtype)
                    state.unedited_latents = (gather_tokens(current, state.unedited_indices)
                        + (target - sigma) * gather_tokens(velocity, state.unedited_indices))
                    latents = gather_tokens(normal_next, state.edited_indices)
                else: latents = normal_next
                execution = 'full' if not local else ('local_compute' if compute else 'avd_reuse')
                diagnostics.append({'step': step, 'execution': execution, 'active_generated_tokens': active.shape[1],
                    'reference_tokens': 0 if local else m, 'kv_action': action if compute else 'skipped',
                    'avd_ratio': ratio, 'avd_reason': reason, 'scheduler_updates': 1,
                    'branch_forward_count': 2 if compute else 0,
                    'active_indices': active.detach().cpu().tolist(),
                    'unedited_projection_target_step': projection_target})
                _observe(pipeline, {'step': step, 'execution': execution, 'pre_latents': pre,
                    'post_latents': full_snapshot(latents), 'scheduler_pre_latents': current,
                    'scheduler_post_latents': normal_next, 'guided_velocity': audit_guided,
                    'scheduler_velocity': velocity, 'guided_velocity_layout': 'full_reconstructed',
                    'branches': branches, 'edited_indices': state.edited_indices,
                    'unedited_indices': state.unedited_indices,
                    'unedited_projection_target_step': projection_target,
                    'unedited_projection_sigma': None if projection_target is None else sigmas[projection_target],
                    'unedited_rows_semantics': 'direct_projection_to_future_merge' if projection_target else 'ordinary',
                    'avd_ratio': ratio, 'avd_reason': reason}, caches, n)
                previous_t = t
                bar.update()
        if latents.shape[1] != n: latents = state.merge(latents)
    finally:
        for cache in caches.values(): cache.clear()
        pipeline._current_timestep = None
    return latents, diagnostics


def enrich_metadata(pipeline, metadata, *, true_cfg_scale, guidance_scale, negative_prompt,
                    do_true_cfg, generated_tokens, reference_tokens):
    """Child-native realized protocol and whole-request allocator peaks."""
    import os
    parameter = next(pipeline.transformer.parameters())
    metadata['native_observation'] = {
        'pid': os.getpid(), 'dtype': str(parameter.dtype), 'device': str(parameter.device),
        'true_cfg_scale': float(true_cfg_scale), 'guidance_scale': float(guidance_scale),
        'negative_prompt': negative_prompt, 'do_true_cfg': bool(do_true_cfg),
        'generated_tokens': generated_tokens, 'reference_tokens': reference_tokens,
        'cache_layers': len(pipeline.transformer.transformer_blocks) + len(pipeline.transformer.single_transformer_blocks),
        'cfg_parallel_size': getattr(pipeline.od_config.parallel_config, 'cfg_parallel_size', 1),
        'state_scope': 'native request',
    }
    if torch.cuda.is_available():
        metadata['memory'] = {'peak_allocated': int(torch.cuda.max_memory_allocated()),
                              'peak_reserved': int(torch.cuda.max_memory_reserved()),
                              'provenance': 'pipeline child allocator reset at forward entry'}
    return metadata
