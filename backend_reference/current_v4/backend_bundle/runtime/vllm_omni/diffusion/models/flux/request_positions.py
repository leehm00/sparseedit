# SPDX-License-Identifier: Apache-2.0
"""Owned request-invariant rotary data and stable step-specific graph views."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PositionView:
    full: tuple[torch.Tensor, torch.Tensor]
    query: tuple[torch.Tensor, torch.Tensor]
    image: tuple[torch.Tensor, torch.Tensor]
    positions: torch.Tensor


class RequestPositionPlan:
    """Build full-grid frequencies at bind; gather exact current rows per step.

    Only positional metadata is reusable. Numerical hidden state/KV is owned by
    the existing per-request cache lifecycle. Full, compact and dense buffers
    keep addresses on rebind; equal CFG coordinate branches share one view.
    """

    def __init__(self, *, text_tokens, image_tokens, active_tokens, dtype, device, share_branches=False):
        self.text_tokens = text_tokens
        self.image_tokens = image_tokens
        self.active_tokens = active_tokens
        self.dtype = dtype
        self.device = torch.device(device)
        self.share_branches = share_branches
        self.views = {}
        self.binds = self.prepares = self.evaluations = 0
        self.bound = False

    def _allocate(self, full):
        text, image, active = self.text_tokens, self.image_tokens, self.active_tokens
        dense_positions = torch.arange(text, text + image, device=self.device).unsqueeze(0)
        compact_positions = torch.empty((1, active), dtype=torch.long, device=self.device)
        compact_image = tuple(torch.empty((active, t.shape[-1]), dtype=self.dtype, device=self.device) for t in full)
        compact_query = tuple(
            torch.empty((text + active, t.shape[-1]), dtype=self.dtype, device=self.device) for t in full
        )
        dense = PositionView(full, full, tuple(t[text:] for t in full), dense_positions)
        sparse = PositionView(full, compact_query, compact_image, compact_positions)
        return {True: dense, False: sparse}

    def bind(self, embed, image_ids, branch_text_ids):
        if image_ids.shape != (self.image_tokens, 3):
            raise ValueError("request image coordinates do not match position plan")
        for text in branch_text_ids.values():
            if text.shape != (self.text_tokens, 3):
                raise ValueError("request text coordinates do not match position plan")
        if self.share_branches and not all(
            torch.equal(branch_text_ids["positive"], text) for text in branch_text_ids.values()
        ):
            raise ValueError("shared CFG positions require equal current text coordinates")
        self.evaluations = self.prepares = 0
        for branch, text in branch_text_ids.items():
            if self.share_branches and branch != "positive":
                self.views[branch] = self.views["positive"]
                continue
            # Match original frequency arithmetic, then its layer-level dtype cast.
            values = tuple(t.to(self.dtype) for t in embed(torch.cat((text, image_ids), dim=0)))
            self.evaluations += 1
            if branch not in self.views:
                self.views[branch] = self._allocate(tuple(t.detach().clone() for t in values))
            else:
                for destination, value in zip(self.views[branch][True].full, values):
                    destination.copy_(value)
            sparse = self.views[branch][False]
            for compact, full in zip(sparse.query, sparse.full):
                compact[: self.text_tokens].copy_(full[: self.text_tokens])
        self.binds += 1
        self.bound = True

    def prepare(self, indices, *, dense):
        if not self.bound:
            raise RuntimeError("position plan must be rebound before preparation")
        if dense:
            if indices.numel() != self.image_tokens:
                raise ValueError("dense position plan expects every image token")
        else:
            if indices.shape != (1, self.active_tokens) or indices.dtype != torch.long:
                raise ValueError("sparse indices must match the fixed active token count")
            seen = set()
            for branch_views in self.views.values():
                if id(branch_views) in seen:
                    continue
                seen.add(id(branch_views))
                view = branch_views[False]
                torch.add(indices, self.text_tokens, out=view.positions)
                for full, image, query in zip(view.full, view.image, view.query):
                    torch.index_select(full, 0, view.positions.reshape(-1), out=image)
                    query[self.text_tokens :].copy_(image)
        self.prepares += 1

    def view(self, branch, *, dense):
        return self.views[branch][dense]

    def metadata(self):
        tensors = {}
        for views in self.views.values():
            for view in views.values():
                for t in (*view.full, *view.query, *view.image, view.positions):
                    tensors[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return {
            "full_position_evaluations": self.evaluations,
            "step_preparations": self.prepares,
            "shared_cfg_positions": self.share_branches,
            "owned_bytes": sum(tensors.values()),
            "lifetime_binds": self.binds,
            "scope": "positions_only_current_request_rebound",
        }
