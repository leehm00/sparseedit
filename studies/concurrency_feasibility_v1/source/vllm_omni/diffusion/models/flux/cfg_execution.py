# SPDX-License-Identifier: Apache-2.0
"""Stable sparse CFG batch layout and shared owned K/V allocation."""

from __future__ import annotations

import torch


def choose_cfg_execution(mode, *, text_tokens, active_tokens, min_query_rows, compatible):
    if mode not in ("serial", "batched", "auto"):
        raise ValueError("unsupported CFG execution mode")
    if mode == "serial":
        return "serial", "explicit_serial"
    if not compatible:
        if mode == "batched":
            raise ValueError("batched CFG requires two branches and equal positional coordinates")
        return "serial", "incompatible_branch_positions_or_count"
    if mode == "batched":
        return "batched", "explicit_batched"
    if text_tokens + active_tokens >= min_query_rows:
        return "batched", "query_rows_at_or_above_threshold"
    return "serial", "query_rows_below_threshold"


class CFGCachePool:
    """One allocation per layer; serial branches and joint calls use views."""

    def __init__(self):
        self.layers = {}
        self.ready = {}

    def allocate(self, key, shape, *, dtype, device):
        full_shape = (2, *shape[1:])
        if key not in self.layers:
            self.layers[key] = (
                torch.empty(full_shape, dtype=dtype, device=device),
                torch.empty(full_shape, dtype=dtype, device=device),
            )
            self.ready[key] = set()
        k, v = self.layers[key]
        if k.shape != full_shape or k.dtype != dtype or k.device != device:
            raise ValueError("CFG cache pool shape/dtype/device changed")
        return k, v

    def mark_full(self, key, slot):
        self.ready[key].update((0, 1) if slot is None else (slot,))

    def require_full(self, key, slot):
        expected = {0, 1} if slot is None else {slot}
        if not expected <= self.ready.get(key, set()):
            raise RuntimeError(
                "both CFG branches must store full cache before joint reads"
                if slot is None
                else "CFG branch used before its full cache was stored"
            )

    def owned_bytes(self):
        return sum(t.numel() * t.element_size() for pair in self.layers.values() for t in pair)

    def clear(self):
        self.layers.clear()
        self.ready.clear()


class CFGExecutionPlan:
    """One step packing; branch dimension persists through all transformer layers."""

    def __init__(self, *, active_tokens, latent_channels, dtype, device):
        self.pool = CFGCachePool()
        self.input = torch.empty((2, active_tokens, latent_channels), dtype=dtype, device=device)
        self.prompt = self.pooled = self.guidance = None
        self.joint_cache = None
        self.step_binds = 0
        self.requests = 0

    @torch.no_grad()
    def bind_request(self, positive_prompt, negative_prompt, positive_pooled, negative_pooled, guidance):
        values = {
            "prompt": torch.cat((positive_prompt, negative_prompt), dim=0),
            "pooled": torch.cat((positive_pooled, negative_pooled), dim=0),
        }
        if guidance is not None:
            values["guidance"] = guidance.expand(2)
        for name, value in values.items():
            owned = getattr(self, name)
            if owned is None:
                setattr(self, name, value.detach().clone())
            else:
                if owned.shape != value.shape or owned.dtype != value.dtype:
                    raise ValueError("CFG conditioning layout changed")
                owned.copy_(value)
        self.requests += 1
        self.step_binds = 0

    @torch.no_grad()
    def bind_step(self, model_input):
        if model_input.shape != (1, self.input.shape[1], self.input.shape[2]):
            raise ValueError("CFG sparse input shape changed")
        self.input.copy_(model_input.expand(2, -1, -1))
        self.step_binds += 1

    def metadata(self):
        arrays = (self.input, self.prompt, self.pooled, self.guidance)
        return {
            "step_binds": self.step_binds,
            "lifetime_requests": self.requests,
            "input_and_conditioning_bytes": sum(t.numel() * t.element_size() for t in arrays if t is not None),
            "unique_kv_bytes": self.pool.owned_bytes(),
            "kv_allocation": "shared_B2_base_with_disjoint_serial_views",
            "per_layer_kv_pack_copies": 0,
            "dense_execution": "serial",
            "sparse_execution": "batched",
        }

    def release(self):
        if self.joint_cache is not None:
            self.joint_cache.clear()
        self.pool.clear()
        self.input = self.prompt = self.pooled = self.guidance = None
