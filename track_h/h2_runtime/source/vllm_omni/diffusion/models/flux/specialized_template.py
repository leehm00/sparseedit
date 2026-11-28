# SPDX-License-Identifier: Apache-2.0
"""Serial graph templates with bounded budget LRU and shared full-context ownership."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from .condition_projection_plan import ConditionView
from .request_positions import PositionView
from .specialized_graph import BranchSparseGraph


class DenseForwardTemplate:
    """Batch-one full forwards with stable slots and a shared dense graph pool.

    Within an action the positive graph is captured and replayed first.
    CFG consumes both outputs before replay at the next step.
    """

    def __init__(self):
        self.hidden_states = None
        self.timestep = None
        self.slots = {}
        self.graphs = {}
        self.pool = None
        self.closed = False

    @staticmethod
    def _bind(slot, tensor):
        if slot is None:
            return tensor.detach().clone()
        if (slot.shape, slot.dtype, slot.device) != (tensor.shape, tensor.dtype, tensor.device):
            raise ValueError("dense graph slot layout changed")
        slot.copy_(tensor)
        return slot

    def bind_step(self, hidden_states, timestep):
        if self.closed or hidden_states.shape[0] != 1:
            raise ValueError("dense graph requires a live batch-one lease")
        self.hidden_states = self._bind(self.hidden_states, hidden_states)
        self.timestep = self._bind(self.timestep, timestep)

    def bind_tensors(self, key, values):
        if self.closed:
            raise ValueError("dense graph template has closed")
        slots = self.slots.setdefault(key, {})
        if slots and slots.keys() != values.keys():
            raise ValueError("dense graph slot set changed")
        for name, value in values.items():
            slots[name] = self._bind(slots.get(name), value)
        return slots

    def forward(self, branch, context, kwargs, condition_plan, predict):
        key = (context.kv_action, branch)
        if key[0] not in ("none", "store_full") or branch not in ("positive", "negative"):
            raise ValueError("unexpected dense graph phase or branch")
        position, condition = context.request_position_view, context.condition_view
        if condition is None or position is None:
            raise ValueError("dense graph requires prepared condition and position views")
        branch_index = 0 if branch == "positive" else 1
        args = self.bind_tensors(branch + "_args", {
            name: tensor for name, tensor in kwargs.items()
            if isinstance(tensor, torch.Tensor) and name not in ("hidden_states", "timestep")
        })
        bound = self.bind_tensors(branch + "_views", {
            "temb": condition.temb,
            "condition": condition.condition,
            "projected": condition_plan.current_projected[branch_index],
            "cos": position.full[0],
            "sin": position.full[1],
            "positions": position.positions,
            "indices": context.image_indices,
        })
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        runner = self.graphs.get(key)
        if runner is None:
            runner = BranchSparseGraph(
                branch, borrow_output=True, shared_pool=self.pool,
                captured_region=f"full FLUX transformer, kv_action={key[0]}",
            )
            full = (bound["cos"], bound["sin"])
            text_tokens = kwargs["txt_ids"].shape[0]
            static_context = replace(
                context,
                image_indices=bound["indices"],
                full_img_ids=args["img_ids"],
                request_position_view=PositionView(
                    full, full, tuple(t[text_tokens:] for t in full), bound["positions"],
                ),
                condition_view=ConditionView(
                    bound["temb"], bound["condition"],
                    {identifier: bound["projected"][:, start:end]
                     for identifier, start, end in condition_plan.offsets},
                ),
            )
            captured = {**kwargs, **args, "hidden_states": self.hidden_states, "timestep": self.timestep}
            captured["joint_attention_kwargs"] = {
                **kwargs["joint_attention_kwargs"], "_flux_acceleration_context": static_context,
            }
            runner.capture(lambda: predict(**captured), device=self.hidden_states.device)
            self.graphs[key] = runner
            if branch == "negative":
                positive = self.graphs[(key[0], "positive")]
                if positive.output_storage_ptr() == runner.output_storage_ptr():
                    raise RuntimeError("dense positive and negative output slots alias")
        return runner.replay()

    def metadata(self):
        graphs = {f"{action}_{branch}": g.metadata() for (action, branch), g in self.graphs.items()}
        pool_bytes = [g.capture_pool_bytes for g in self.graphs.values() if g.capture_pool_bytes is not None]
        return {"graphs": graphs, "shared_pool_bytes": max(pool_bytes, default=None)}

    def close(self):
        for graph in self.graphs.values():
            graph.close()
        self.graphs.clear()
        self.slots.clear()
        self.hidden_states = self.timestep = self.pool = None
        self.closed = True


@dataclass
class SparseForwardTemplate:
    signature: tuple
    conditioning: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    plan: Any = None
    position_plan: Any = None
    condition_plan: Any = None
    cfg_plan: Any = None
    positive_cache: Any = None
    negative_cache: Any = None
    graphs: Any = None
    full_img_ids: torch.Tensor | None = None
    contexts: dict = field(default_factory=dict)
    in_use: bool = False
    closed: bool = False
    requests: int = 0
    arena: Any = None
    arena_signature: tuple | None = None

    def bind_conditioning(
        self, branch: str, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.closed:
            raise RuntimeError("template is closed")
        if branch not in self.conditioning:
            self.conditioning[branch] = {
                key: tensor.detach().clone() for key, tensor in tensors.items()
            }
        else:
            bound = self.conditioning[branch]
            if bound.keys() != tensors.keys():
                raise ValueError("conditioning shape/key mismatch")
            for key, tensor in tensors.items():
                if bound[key].shape != tensor.shape or bound[key].dtype != tensor.dtype:
                    raise ValueError("conditioning shape/dtype mismatch")
                bound[key].copy_(tensor)
        return self.conditioning[branch]

    def finish(self, *, success: bool) -> None:
        self.in_use = False
        if success:
            self.requests += 1
        else:
            if self.arena is not None:
                self.arena.clear_dense_template()
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        failure = None

        def release(action):
            nonlocal failure
            try:
                action()
            except BaseException as error:  # noqa: BLE001 - release all owners before propagating cancellation
                if failure is None:
                    failure = error
                else:
                    failure.add_note(
                        f"Additional template cleanup failed: {type(error).__name__}: {error}"
                    )

        if self.graphs is not None:
            release(self.graphs.close)
        if self.plan is not None:
            release(self.plan.release)
        if self.arena is not None:
            release(self.arena.release)
        else:
            for cache in (self.positive_cache, self.negative_cache):
                if cache is not None:
                    release(cache.clear)
        if self.condition_plan is not None:
            close = getattr(self.condition_plan, "close", None)
            if close is not None:
                release(close)
        if self.cfg_plan is not None:
            release(self.cfg_plan.release)
        self.arena = self.positive_cache = self.negative_cache = None
        self.graphs = self.plan = self.condition_plan = self.cfg_plan = None
        self.position_plan = None
        self.conditioning.clear()
        self.contexts.clear()
        self.full_img_ids = None
        self.in_use = False
        self.closed = True
        if failure is not None:
            raise failure


@dataclass
class _CacheArena:
    """Full numerical KV is shared by compatible templates for serial requests.

    In joint CFG, this arena owns the B2 base and both B1 views. Graphs own
    compact per-budget buffers and are destroyed before releasing the final
    arena reference. Every request executes dense store_full before replay.
    """

    positive: Any
    negative: Any
    joint_cache: Any = None
    references: int = 0
    dense_template: DenseForwardTemplate | None = None

    def get_dense_template(self):
        if self.dense_template is None:
            self.dense_template = DenseForwardTemplate()
        return self.dense_template

    def clear_dense_template(self):
        if self.dense_template is not None:
            self.dense_template.close()
            self.dense_template = None

    def release(self):
        self.references -= 1
        if self.references == 0:
            self.clear_dense_template()
            for cache in (self.positive, self.negative):
                if cache is not None:
                    cache.clear()
            if self.joint_cache is not None:
                self.joint_cache.clear()
                self.joint_cache.shared_pool.clear()


class _TemplatePool:
    def __init__(self, capacity):
        self.capacity = capacity
        self.templates = OrderedDict()
        self.arenas = {}
        self.hits = self.misses = self.evictions = 0

    def close(self):
        for template in self.templates.values():
            template.close()
        self.templates.clear()
        self.arenas.clear()

    def metadata(self):
        arenas = [a for a in self.arenas.values() if a.references]
        unique = {}
        for arena in arenas:
            for cache in (arena.positive, arena.negative):
                if cache is not None:
                    for tensor in cache.storage_tensors():
                        unique[tensor.data_ptr()] = tensor.nbytes
        return {
            "capacity": self.capacity,
            "entries": len(self.templates),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "arenas": len(arenas),
            "arena_references": [a.references for a in arenas],
            "unique_kv_bytes": sum(unique.values()),
        }


def acquire_template(
    pipeline: Any, signature: tuple, *, capacity: int = 1
) -> tuple[SparseForwardTemplate, bool]:
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or not 1 <= capacity <= 3
    ):
        raise ValueError("template capacity must be an integer in [1, 3]")
    existing = getattr(pipeline, "_sparseedit_forward_template", None)
    pool = getattr(pipeline, "_sparseedit_template_pool", None)
    if (existing is not None and existing.in_use) or (
        pool is not None and any(t.in_use for t in pool.templates.values())
    ):
        raise RuntimeError(
            "sparse forward template is in use; serial requests are required"
        )
    if pool is None or pool.capacity != capacity:
        if pool is not None:
            pool.close()
        elif existing is not None:
            existing.close()
        pool = _TemplatePool(capacity)
        pipeline._sparseedit_template_pool = pool
    for key, template in list(pool.templates.items()):
        if template.closed:
            del pool.templates[key]
    for key, arena in list(pool.arenas.items()):
        if not arena.references:
            del pool.arenas[key]
    hit = signature in pool.templates
    if hit:
        pool.hits += 1
        existing = pool.templates[signature]
        pool.templates.move_to_end(signature)
    else:
        pool.misses += 1
        if len(pool.templates) >= capacity:
            _, evicted = pool.templates.popitem(last=False)
            evicted.close()
            pool.evictions += 1
        existing = SparseForwardTemplate(signature=signature)
        pool.templates[signature] = existing
    pipeline._sparseedit_forward_template = existing
    existing.in_use = True
    return existing, hit


def reserve_template(pipeline, owner, signature):
    """Reserve a sibling under an active serial owner without releasing its lease."""
    pool = getattr(pipeline, "_sparseedit_template_pool", None)
    if (
        pool is None
        or not owner.in_use
        or owner.closed
        or pipeline._sparseedit_forward_template is not owner
        or not any(t is owner for t in pool.templates.values())
    ):
        raise RuntimeError("prewarm reservation requires the active owner lease")
    existing = pool.templates.get(signature)
    if existing is not None and not existing.closed:
        pool.templates.move_to_end(signature)
        return existing, True
    if existing is not None:
        del pool.templates[signature]
    if len(pool.templates) >= pool.capacity:
        evict = next(
            (
                key
                for key, value in pool.templates.items()
                if value is not owner and not value.in_use
            ),
            None,
        )
        if evict is None:
            raise RuntimeError("prewarm capacity cannot evict its active owner")
        pool.templates.pop(evict).close()
        pool.evictions += 1
    sibling = SparseForwardTemplate(signature=signature)
    pool.templates[signature] = sibling
    return sibling, False


def bind_template_caches(pipeline, template, arena_signature, positive, negative, *, joint_cache=None):
    """Attach a new template to compatible full KV; keep buffers graph-stable."""
    if template.arena is not None:
        return template.arena.positive, template.arena.negative
    pool = pipeline._sparseedit_template_pool
    arena = pool.arenas.get(arena_signature)
    if arena is None or not arena.references:
        arena = _CacheArena(positive, negative, joint_cache=joint_cache)
        pool.arenas[arena_signature] = arena
    elif (arena.joint_cache is None) != (joint_cache is None):
        raise RuntimeError("full-context arena CFG layout differs")
    arena.references += 1
    template.arena = arena
    template.arena_signature = arena_signature
    return arena.positive, arena.negative


def template_pool_metadata(pipeline):
    pool = getattr(pipeline, "_sparseedit_template_pool", None)
    result = pool.metadata() if pool is not None else {}
    bank = getattr(pipeline, "_sparseedit_condition_bank", None)
    result["condition_bank_entries"] = len(bank) if bank is not None else 0
    result["condition_bank_bytes"] = (
        sum(t.numel() * t.element_size() for entry in bank.values() for t in entry)
        if bank is not None
        else 0
    )
    return result


def clear_template(pipeline: Any) -> None:
    template = getattr(pipeline, "_sparseedit_forward_template", None)
    pool = getattr(pipeline, "_sparseedit_template_pool", None)
    if (template is not None and template.in_use) or (
        pool is not None and any(t.in_use for t in pool.templates.values())
    ):
        raise RuntimeError("sparse forward template is in use")
    if pool is not None:
        pool.close()
    elif template is not None:
        template.close()
    bank = getattr(pipeline, "_sparseedit_condition_bank", None)
    if bank is not None:
        bank.clear()
    pipeline._sparseedit_template_pool = None
    pipeline._sparseedit_forward_template = None
