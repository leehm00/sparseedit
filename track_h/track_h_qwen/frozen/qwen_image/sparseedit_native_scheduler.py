# SPDX-License-Identifier: Apache-2.0
"""Qwen native CFG normalization and complete-velocity trajectory history."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import torch

from .sparseedit_native_runtime import (
    QwenSparseEditConfig,
)

# Load the tensor-only selector without importing vllm_omni.diffusion.models.flux,
# whose optional vLLM runtime is deliberately absent from leaf CPU tests.
_SELECTOR_PATH = Path(__file__).resolve().parents[1] / "flux" / "paper_selector.py"
_SELECTOR_SPEC = importlib.util.spec_from_file_location(
    "_qwen_sparseedit_paper_selector", _SELECTOR_PATH
)
if _SELECTOR_SPEC is None or _SELECTOR_SPEC.loader is None:
    raise ImportError(f"cannot load SparseEdit paper selector: {_SELECTOR_PATH}")
_SELECTOR = importlib.util.module_from_spec(_SELECTOR_SPEC)
_SELECTOR_SPEC.loader.exec_module(_SELECTOR)
paper_local_weight = _SELECTOR.paper_local_weight
select_paper_tokens = _SELECTOR.select_paper_tokens


@dataclass(frozen=True)
class QwenSelection:
    indices: torch.Tensor
    mask: torch.Tensor
    priority: torch.Tensor


def native_cfg_rescale(
    positive: torch.Tensor, negative: torch.Tensor, *, true_cfg_scale: float
) -> torch.Tensor:
    """Exactly reproduce QwenImageEditPipeline's post-two-branch CFG equation."""
    combined = negative + true_cfg_scale * (positive - negative)
    cond_norm = torch.norm(positive, dim=-1, keepdim=True)
    noise_norm = torch.norm(combined, dim=-1, keepdim=True)
    return combined * (cond_norm / noise_norm)


class QwenVelocityHistory:
    """Retain two preceding complete, CFG-normalized scheduler velocities."""

    def __init__(self, config: QwenSparseEditConfig) -> None:
        self.config = config
        self.velocities: list[torch.Tensor] = []
        self.flow_times: list[float] = []
        self.ages: torch.Tensor | None = None

    @staticmethod
    def flow_time(timestep: torch.Tensor | float) -> float:
        value = (
            float(timestep.reshape(-1)[0].item())
            if isinstance(timestep, torch.Tensor)
            else float(timestep)
        )
        if not 0 <= value <= 1000:
            raise ValueError("native Qwen scheduler timestep must be in [0, 1000]")
        return value / 1000.0

    def record(self, complete_velocity: torch.Tensor, *, flow_time: float) -> None:
        if complete_velocity.ndim != 3 or complete_velocity.shape[0] != 1:
            raise ValueError(
                "Qwen velocity must have shape [1, generated_tokens, channels]"
            )
        if self.velocities and self.velocities[-1].shape != complete_velocity.shape:
            raise ValueError("velocity shape changed within request")
        self.velocities.append(complete_velocity.detach().clone())
        self.velocities = self.velocities[-2:]
        self.flow_times.append(float(flow_time))
        self.flow_times = self.flow_times[-2:]
        if self.ages is None:
            self.ages = torch.zeros(
                complete_velocity.shape[:2],
                device=complete_velocity.device,
                dtype=torch.int64,
            )

    def select(
        self,
        *,
        flow_time: float,
        warmup_flow_time: float,
        grid_height: int,
        grid_width: int,
    ) -> QwenSelection | None:
        if len(self.velocities) < 2 or self.ages is None:
            return None
        if not 0 <= flow_time <= 1:
            raise ValueError("flow_time must be scheduler timestep/1000")
        count = max(1, int(self.config.ratio * grid_height * grid_width))
        weight = paper_local_weight(
            "forward", flow_time=flow_time, warmup_flow_time=warmup_flow_time
        )
        mask, priority, indices = select_paper_tokens(
            self.velocities[-1],
            self.velocities[-2],
            self.ages,
            count,
            grid_height,
            grid_width,
            local_weight=weight,
            validate_values=False,
        )
        return QwenSelection(indices=indices, mask=mask, priority=priority)

    def commit_sparse(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        indices: torch.Tensor,
        *,
        true_cfg_scale: float,
    ) -> torch.Tensor:
        if not self.velocities or self.ages is None:
            raise RuntimeError("sparse commit requires a complete preceding velocity")
        if positive.shape != negative.shape or positive.shape[1] != indices.numel():
            raise ValueError(
                "active CFG predictions and indices have incompatible shapes"
            )
        active = native_cfg_rescale(positive, negative, true_cfg_scale=true_cfg_scale)
        complete = self.velocities[-1].clone()
        complete.scatter_(1, indices.reshape(1, -1, 1).expand_as(active), active)
        self.ages.add_(1)
        self.ages.scatter_(1, indices.reshape(1, -1), 0)
        return complete

    def commit_dense(
        self, positive: torch.Tensor, negative: torch.Tensor, *, true_cfg_scale: float
    ) -> torch.Tensor:
        if positive.shape != negative.shape:
            raise ValueError("dense CFG branches must have matching shapes")
        if self.ages is not None:
            self.ages.zero_()
        return native_cfg_rescale(positive, negative, true_cfg_scale=true_cfg_scale)


__all__ = ["QwenSelection", "QwenVelocityHistory", "native_cfg_rescale"]
