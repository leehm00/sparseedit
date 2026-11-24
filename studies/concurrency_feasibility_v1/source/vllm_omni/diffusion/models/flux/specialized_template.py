# SPDX-License-Identifier: Apache-2.0
"""Serial graph templates with bounded budget LRU and shared full-context ownership."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch


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
    """Full numerical KV is shared only by serial templates with identical layout.

    Graphs own compact per-budget buffers. Destroy graphs before releasing the
    final arena reference. Every request executes dense store_full before replay.
    """

    positive: Any
    negative: Any
    references: int = 0

    def release(self):
        self.references -= 1
        if self.references == 0:
            for cache in (self.positive, self.negative):
                if cache is not None:
                    cache.clear()


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
                    for storage in cache.layers.values():
                        for tensor in (storage.key, storage.value):
                            unique[tensor.data_ptr()] = (
                                tensor.numel() * tensor.element_size()
                            )
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


def bind_template_caches(pipeline, template, arena_signature, positive, negative):
    """Attach a new template to compatible full KV; keep buffers graph-stable."""
    if template.arena is not None:
        return template.arena.positive, template.arena.negative
    pool = pipeline._sparseedit_template_pool
    arena = pool.arenas.get(arena_signature)
    if arena is None or not arena.references:
        arena = _CacheArena(positive, negative)
        pool.arenas[arena_signature] = arena
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
