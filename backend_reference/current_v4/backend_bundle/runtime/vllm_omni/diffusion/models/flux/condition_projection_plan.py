# SPDX-License-Identifier: Apache-2.0
"""Request-owned conditioning projections with stable current-step consumers."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class ConditionView:
    temb: torch.Tensor
    condition: torch.Tensor
    projected: dict[int, torch.Tensor]


def condition_norms(transformer):
    items = []
    for block in transformer.transformer_blocks:
        items.extend((block.norm1, block.norm1_context))
    items.extend(block.norm for block in transformer.single_transformer_blocks)
    items.append(transformer.norm_out)
    return items


class RequestConditionPlan:
    """Batch only layer-specific projections; preserve native time embeddings.

    serial provides the exact reference. cfg/schedule use different GEMM row
    counts, so floating-point equality is evaluated separately from semantics.
    No latent or numerical KV state is stored in this plan.
    """

    def __init__(self, *, mode, cache=None, prepare_graph=False):
        if mode not in ("serial", "cfg", "schedule", "reuse"):
            raise ValueError("unsupported condition projection mode")
        if prepare_graph and mode != "serial":
            raise ValueError(
                "condition preparation graph requires native serial projections"
            )
        self.mode = mode
        self.prepare_graph = bool(prepare_graph)
        self.preparation_graph = None
        self.closed = False
        self.signature = None
        self.views = {}
        self.current_temb = self.current_condition = self.current_projected = None
        self.temb_values = self.condition_values = self.projected_values = None
        self.steps = self.branches = self.projections = self.condition_evaluations = (
            self.bind_count
        ) = 0
        self.offsets = []
        self.cache = OrderedDict() if cache is None else cache
        self.cache_hits = []
        self.cache_key_digests = {}
        self.cache_capacity = 4

    def _cache_key(self, transformer, modules, timesteps, pooled, guidance, dtype):
        digest = hashlib.sha256()
        identities = []
        for owner in [
            transformer.time_text_embed,
            *(module.linear for module in modules),
        ]:
            for parameter in owner.parameters():
                try:
                    version = parameter._version
                except RuntimeError:
                    version = None  # Inference weights remain immutable for the template lifetime.
                identities.append(
                    (
                        id(parameter),
                        version,
                        str(parameter.dtype),
                        str(parameter.device),
                        tuple(parameter.shape),
                    )
                )
        digest.update(
            repr((id(transformer), identities, str(dtype), str(pooled.device))).encode()
        )
        for value in (timesteps, pooled, guidance):
            if value is None:
                digest.update(b"none")
            else:
                digest.update(repr((tuple(value.shape), str(value.dtype))).encode())
                digest.update(
                    value.detach()
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                    .tobytes()
                )
        return digest.hexdigest()

    @torch.no_grad()
    def bind(self, transformer, timesteps, pooled_by_branch, *, guidance, dtype):
        if self.closed:
            raise RuntimeError("condition plan is closed")
        if self.prepare_graph:
            return self._bind_graph(
                transformer, timesteps, pooled_by_branch, guidance=guidance, dtype=dtype
            )
        modules = condition_norms(transformer)
        names = tuple(pooled_by_branch)
        device = pooled_by_branch[names[0]].device
        signature = (
            id(transformer),
            tuple(id(n) for n in modules),
            len(timesteps),
            names,
            dtype,
            device,
        )
        if self.signature is not None and self.signature != signature:
            raise ValueError(
                "condition plan shape/model changed; new template required"
            )
        self.steps = len(timesteps)
        self.branches = len(names)
        self.projections = self.condition_evaluations = self.bind_count = 0
        cached = {}
        keys = {}
        self.cache_hits = []
        if self.mode == "reuse":
            for branch, (name, pooled) in enumerate(pooled_by_branch.items()):
                key = self._cache_key(
                    transformer, modules, timesteps, pooled, guidance, dtype
                )
                keys[branch] = key
                if key in self.cache:
                    cached[branch] = self.cache[key]
                    self.cache.move_to_end(key)
                    self.cache_hits.append(name)
        self.cache_key_digests = {names[branch]: key for branch, key in keys.items()}
        values = []
        for step, t in enumerate(timesteps):
            for branch, pooled in enumerate(pooled_by_branch.values()):
                if branch in cached:
                    values.append(cached[branch][0][step : step + 1])
                    continue
                # Same dtype roundings as runtime -> transformer, including /1000.
                normalized = t.expand(1).to(device=device, dtype=dtype) / 1000
                native_timestep = normalized.to(device=device, dtype=dtype) * 1000
                if guidance is None:
                    temb = transformer.time_text_embed(native_timestep, pooled)
                else:
                    temb = transformer.time_text_embed(
                        native_timestep, guidance.to(dtype) * 1000, pooled
                    )
                if temb.dtype != dtype or temb.shape[0] != 1:
                    raise ValueError(
                        "condition plan expects native single-request embedding dtype"
                    )
                values.append(temb)
                self.condition_evaluations += 1
        tembs = torch.stack(values).reshape(self.steps, self.branches, -1)
        conditions = torch.stack(
            [
                cached[row % self.branches][1][
                    row // self.branches : row // self.branches + 1
                ]
                if row % self.branches in cached
                else modules[0].silu(value)
                for row, value in enumerate(values)
            ]
        ).reshape_as(tembs)
        channels = tembs.shape[-1]
        offsets = []
        total = 0
        for module in modules:
            method = getattr(module.linear, "quant_method", None)
            if (
                method is not None
                and type(method).__name__ != "UnquantizedLinearMethod"
            ):
                raise ValueError(
                    "condition plan requires unquantized native linear modules"
                )
            width = module.linear.weight.shape[0]
            offsets.append((id(module), total, total + width))
            total += width
        if self.signature is None:
            self.current_temb = torch.empty(
                (self.branches, 1, channels), dtype=dtype, device=device
            )
            self.current_condition = torch.empty_like(self.current_temb)
            self.current_projected = torch.empty(
                (self.branches, 1, total), dtype=dtype, device=device
            )
            self.temb_values = torch.empty_like(tembs)
            self.condition_values = torch.empty_like(conditions)
            self.projected_values = torch.empty(
                (self.steps, self.branches, total), dtype=dtype, device=device
            )
            for branch, name in enumerate(names):
                self.views[name] = ConditionView(
                    self.current_temb[branch],
                    self.current_condition[branch],
                    {
                        key: self.current_projected[branch, :, start:end]
                        for key, start, end in offsets
                    },
                )
            if self.branches == 2:
                self.views["batched"] = ConditionView(
                    self.current_temb[:, 0],
                    self.current_condition[:, 0],
                    {
                        key: self.current_projected[:, 0, start:end]
                        for key, start, end in offsets
                    },
                )
            self.offsets = offsets
            self.signature = signature
        self.temb_values.copy_(tembs)
        self.condition_values.copy_(conditions)
        for branch, entry in cached.items():
            self.projected_values[:, branch].copy_(entry[2])
        for module, (_, start, end) in zip(modules, self.offsets):
            source = conditions.reshape(-1, channels)
            group = {
                "serial": 1,
                "reuse": 1,
                "cfg": self.branches,
                "schedule": self.steps * self.branches,
            }[self.mode]
            for row in range(0, source.shape[0], group):
                if self.mode == "reuse" and row % self.branches in cached:
                    continue
                projected = module.linear(source[row : row + group])
                if isinstance(projected, tuple):
                    projected = projected[0]
                self.projected_values.reshape(-1, total)[
                    row : row + group, start:end
                ].copy_(projected)
                self.projections += 1

        if self.mode == "reuse":
            for branch, key in keys.items():
                if branch not in cached:
                    self.cache[key] = (
                        self.temb_values[:, branch].clone(),
                        self.condition_values[:, branch].clone(),
                        self.projected_values[:, branch].clone(),
                    )
                    self.cache.move_to_end(key)
                    while len(self.cache) > self.cache_capacity:
                        self.cache.popitem(last=False)

    def _bind_graph(self, transformer, timesteps, pooled_by_branch, *, guidance, dtype):
        from .condition_graph import ConditionPreparationGraph

        if self.preparation_graph is None:
            self.preparation_graph = ConditionPreparationGraph()
        reference = self.preparation_graph.bind(
            transformer, timesteps, pooled_by_branch, guidance=guidance, dtype=dtype
        )
        names = tuple(pooled_by_branch)
        if self.signature is None:
            self.current_temb = torch.empty_like(reference.current_temb)
            self.current_condition = torch.empty_like(reference.current_condition)
            self.current_projected = torch.empty_like(reference.current_projected)
            self.offsets = reference.offsets
            for branch, name in enumerate(names):
                self.views[name] = ConditionView(
                    self.current_temb[branch],
                    self.current_condition[branch],
                    {
                        key: self.current_projected[branch, :, start:end]
                        for key, start, end in self.offsets
                    },
                )
            if len(names) == 2:
                self.views["batched"] = ConditionView(
                    self.current_temb[:, 0],
                    self.current_condition[:, 0],
                    {
                        key: self.current_projected[:, 0, start:end]
                        for key, start, end in self.offsets
                    },
                )
            self.signature = reference.signature
        self.temb_values, self.condition_values, self.projected_values = (
            reference.temb_values,
            reference.condition_values,
            reference.projected_values,
        )
        self.steps, self.branches = reference.steps, reference.branches
        self.condition_evaluations, self.projections = (
            reference.condition_evaluations,
            reference.projections,
        )
        self.bind_count = 0
        self.cache_hits = []

    def close(self):
        if self.closed:
            return
        if self.preparation_graph is not None:
            self.preparation_graph.close()
        self.views.clear()
        self.signature = None
        self.current_temb = self.current_condition = self.current_projected = None
        self.temb_values = self.condition_values = self.projected_values = None
        self.closed = True

    @torch.no_grad()
    def prepare(self, step):
        if self.signature is None or not 0 <= step < self.steps:
            raise ValueError("condition plan step is outside the bound request")
        self.current_temb[:, 0].copy_(self.temb_values[step])
        self.current_condition[:, 0].copy_(self.condition_values[step])
        self.current_projected[:, 0].copy_(self.projected_values[step])
        self.bind_count += 1

    def view(self, branch):
        return self.views[branch]

    def metadata(self):
        arrays = (
            self.current_temb,
            self.current_condition,
            self.current_projected,
            self.temb_values,
            self.condition_values,
            self.projected_values,
        )
        return {
            "mode": self.mode,
            "plan_identity": id(self),
            "cache_identity": id(self.cache),
            "cache_key_digests": self.cache_key_digests,
            "current_consumer_pointers": [
                t.data_ptr()
                for t in (
                    self.current_temb,
                    self.current_condition,
                    self.current_projected,
                )
                if t is not None
            ],
            "preparation_graph": self.preparation_graph.metadata()
            if self.preparation_graph is not None
            else {
                "requested": self.prepare_graph,
                "backend": "not_started" if self.prepare_graph else "not_requested",
                "capture_count": 0,
                "replay_count": 0,
                "closed": self.closed,
            },
            "native_condition_evaluations": self.condition_evaluations,
            "projection_evaluations": self.projections,
            "step_binds": self.bind_count,
            "owned_bytes": sum(
                t.numel() * t.element_size() for t in arrays if t is not None
            )
            + (
                self.preparation_graph.additional_owned_bytes()
                if self.preparation_graph is not None
                else 0
            ),
            "bitwise_reference_mode": self.mode in {"serial", "reuse"},
            "cache_hits": self.cache_hits,
            "cache_entries": len(self.cache),
            "cache_bytes": sum(
                t.numel() * t.element_size()
                for entry in self.cache.values()
                for t in entry
            ),
        }
