# SPDX-License-Identifier: Apache-2.0
"""Opt-in diagnostic profiling. Never include this invocation in timing samples."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch


def _category(module):
    name = type(module).__name__
    if "LayerNorm" in name:
        return "layernorm"
    if "RMSNorm" in name:
        return "qk_norm"
    if name == "Attention":
        return "attention"
    if "RotaryEmbedding" in name:
        return "rotary"
    if name == "FluxPosEmbed":
        return "position_generation"
    if "Linear" in name:
        return "projection"
    if name in {"SiLU", "GELU", "GELUActivation"}:
        return "activation"
    return None


def profile_forward(transformer, forward, directory, *, device):
    """Profile one real eager forward and restore hooks/wrappers even on error.

    Scope inclusive CUDA time is useful per category, but nested scopes (e.g.
    a linear inside a fused activation module) must not be summed as a partition.
    Raw device events and trace are retained for non-overlapping attribution.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("events.json", "trace.json")):
        raise FileExistsError("refusing to overwrite an existing forward profile")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.device(device).type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    hooks = []
    open_scopes = {}

    def begin(module, _args):
        context = torch.profiler.record_function("scope." + _category(module))
        context.__enter__()
        open_scopes.setdefault(id(module), []).append(context)

    def end(module, _args, _result):
        stack = open_scopes.get(id(module), [])
        if stack:
            stack.pop().__exit__(None, None, None)

    def wrapped(fn, name):
        def invoke(*args, **kwargs):
            with torch.profiler.record_function("scope." + name):
                return fn(*args, **kwargs)

        return invoke

    try:
        for module in transformer.modules():
            if _category(module) is not None:
                hooks.extend(
                    (module.register_forward_pre_hook(begin), module.register_forward_hook(end, always_call=True))
                )
        with ExitStack() as patches:
            from . import flux_transformer, specialized_layer

            for name in ("modulate", "gated_residual", "prepare_layer_rotary"):
                for owner in (specialized_layer, flux_transformer):
                    if hasattr(owner, name):
                        patches.enter_context(patch.object(owner, name, wrapped(getattr(owner, name), name)))
            with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
                result = forward()
        profiler.export_chrome_trace(str(directory / "trace.json"))
        events = []
        for event in profiler.key_averages(group_by_input_shape=True):
            events.append(
                {
                    "name": event.key,
                    "count": event.count,
                    "self_cpu_us": event.self_cpu_time_total,
                    "cpu_us": event.cpu_time_total,
                    "self_device_us": event.self_device_time_total,
                    "device_us": event.device_time_total,
                    "shapes": event.input_shapes,
                }
            )
        (directory / "events.json").write_text(
            json.dumps(
                {
                    "diagnostic_only": True,
                    "category_times_may_overlap": True,
                    "events": events,
                },
                indent=2,
            )
            + "\n"
        )
        return result
    finally:
        for handle in hooks:
            handle.remove()
