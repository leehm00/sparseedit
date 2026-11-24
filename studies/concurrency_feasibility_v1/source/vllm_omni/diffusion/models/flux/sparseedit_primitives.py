# SPDX-License-Identifier: Apache-2.0
"""Tensor-only SparseEdit and AVD primitives for FLUX image editing.

This module deliberately has no pipeline or acceleration-runtime imports.  FLUX
AVD coefficients are model and scheduler specific, so callers must supply a
calibrated gamma sequence for their exact inference protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

SPARSEEDIT_AVD_UPDATE_STRATEGIES = frozenset({"spatial_error_smooth_decay", "spatial_error_cond_rerank"})


@dataclass(frozen=True)
class AVDDecision:
    """Whether an AVD step computes the transformer or reuses velocity."""

    compute_transformer: bool
    ratio: float | None
    accumulate: float
    error: float
    reason: str


@dataclass(frozen=True)
class SparseEditConditionRerankConfig:
    """Configuration for condition-residual reranking of sparse candidates."""

    candidate_multiplier: float = 2.0
    condition_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.candidate_multiplier < 1.0:
            raise ValueError("candidate_multiplier must be >= 1")
        if self.condition_weight < 0.0:
            raise ValueError("condition_weight must be >= 0")


def _normalize_token_scores(scores: torch.Tensor) -> torch.Tensor:
    minimum = scores.min(dim=1, keepdim=True).values
    span = scores.max(dim=1, keepdim=True).values - minimum
    normalized = (scores - minimum) / span.clamp(min=1e-8)
    return torch.where(span > 1e-8, normalized, torch.zeros_like(scores))


def select_condition_rerank_mask(
    *,
    dynamic_scores: torch.Tensor,
    condition_scores: torch.Tensor | None,
    active_count: int,
    config: SparseEditConditionRerankConfig | None = None,
) -> torch.Tensor:
    """Rerank a dynamic top-k pool with an optional condition score map."""
    batch_size, num_tokens = dynamic_scores.shape
    active_count = max(1, min(num_tokens, int(active_count)))
    rerank_config = config or SparseEditConditionRerankConfig()
    if (
        condition_scores is None
        or condition_scores.ndim != 2
        or condition_scores.shape[0] != batch_size
        or condition_scores.shape[1] < num_tokens
    ):
        indices = torch.topk(dynamic_scores, active_count, dim=1).indices
        mask = torch.zeros_like(dynamic_scores, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        return mask

    candidate_count = min(
        num_tokens,
        max(active_count, int(math.ceil(active_count * rerank_config.candidate_multiplier))),
    )
    candidate_indices = torch.topk(dynamic_scores, candidate_count, dim=1).indices
    dynamic = torch.gather(dynamic_scores, 1, candidate_indices)
    condition = condition_scores[:, :num_tokens].to(device=dynamic_scores.device, dtype=dynamic_scores.dtype)
    condition = torch.gather(condition, 1, candidate_indices)
    combined = _normalize_token_scores(dynamic) + rerank_config.condition_weight * _normalize_token_scores(condition)
    selected_in_pool = torch.topk(combined, active_count, dim=1).indices
    selected_indices = torch.gather(candidate_indices, 1, selected_in_pool)
    mask = torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=dynamic_scores.device)
    mask.scatter_(1, selected_indices, True)
    return mask


def score_sparseedit_spatial_error_tokens(
    *,
    latents: torch.Tensor,
    previous_latents: torch.Tensor | None,
    num_tokens: int,
    grid_height: int,
    grid_width: int,
    staleness_counter: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score latent change, token age, and stale-neighborhood boundaries."""
    if grid_height * grid_width != num_tokens:
        raise ValueError("SparseEdit spatial scoring requires grid dimensions matching num_tokens")
    current = latents[:, :num_tokens, :]
    if previous_latents is None:
        scores = torch.norm(current, p=2, dim=-1)
    else:
        scores = torch.norm(current - previous_latents[:, :num_tokens, :], p=2, dim=-1)
    if staleness_counter is None:
        return scores

    staleness = staleness_counter[:, :num_tokens].float().to(scores.device)
    scores = scores * (1.0 + 0.5 * staleness)
    staleness_2d = staleness.reshape(latents.shape[0], 1, grid_height, grid_width)
    kernel = torch.ones(1, 1, 3, 3, device=scores.device, dtype=staleness_2d.dtype) / 9.0
    neighborhood = torch.nn.functional.conv2d(staleness_2d, kernel, padding=1)
    boundary = torch.abs(staleness_2d - neighborhood).reshape(latents.shape[0], num_tokens)
    return scores * (1.0 + 0.3 * boundary)


def select_sparseedit_spatial_error_mask(
    *,
    update_strategy: str,
    scores: torch.Tensor,
    grid_height: int,
    grid_width: int,
    active_count: int,
    progress: float,
    cfg_residual_scores: torch.Tensor | None = None,
    condition_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a decayed local-average spatial prior and optional condition rerank."""
    if update_strategy not in SPARSEEDIT_AVD_UPDATE_STRATEGIES:
        raise ValueError(f"unsupported SparseEdit spatial strategy: {update_strategy}")
    batch_size, num_tokens = scores.shape
    if grid_height * grid_width != num_tokens:
        raise ValueError("SparseEdit spatial selection requires grid dimensions matching score tokens")
    active_count = max(1, min(num_tokens, int(active_count)))
    progress = max(0.0, min(1.0, float(progress)))
    radius, beta = (2, 0.5) if progress < 0.5 else (1, 0.25) if progress < 0.75 else (0, 0.0)
    adjusted = scores
    if radius:
        normalized = _normalize_token_scores(scores).reshape(batch_size, 1, grid_height, grid_width)
        prior = torch.nn.functional.avg_pool2d(
            normalized, kernel_size=2 * radius + 1, stride=1, padding=radius, count_include_pad=False
        ).reshape(batch_size, num_tokens)
        adjusted = scores * (1.0 + beta * _normalize_token_scores(prior))
    if update_strategy == "spatial_error_cond_rerank":
        mask = select_condition_rerank_mask(
            dynamic_scores=adjusted,
            condition_scores=cfg_residual_scores,
            active_count=active_count,
            config=SparseEditConditionRerankConfig(condition_weight=condition_weight),
        )
    else:
        indices = torch.topk(adjusted, active_count, dim=1).indices
        mask = torch.zeros_like(adjusted, dtype=torch.bool)
        mask.scatter_(1, indices, True)
    return mask, adjusted


def select_sparseedit_linear_mask(
    *,
    scores: torch.Tensor,
    grid_height: int,
    grid_width: int,
    active_count: int,
    progress: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate normalized base scores with their centered 5x5 mean."""
    batch, count = scores.shape
    if grid_height * grid_width != count:
        raise ValueError("linear locality requires grid dimensions matching score tokens")
    if not math.isfinite(progress) or not 0 <= progress <= 1:
        raise ValueError("linear locality progress must be finite and in [0, 1]")
    active_count = max(1, min(count, int(active_count)))
    normalized = _normalize_token_scores(scores)
    local = torch.nn.functional.avg_pool2d(
        normalized.reshape(batch, 1, grid_height, grid_width),
        kernel_size=5,
        stride=1,
        padding=2,
        count_include_pad=False,
    ).reshape(batch, count)
    adjusted = progress * normalized + (1 - progress) * local
    # Preserve the exact raw-score tie/order convention at the endpoint.
    ranking = scores if progress == 1 else adjusted
    indices = torch.topk(ranking, active_count, dim=1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask, adjusted


@dataclass(frozen=True)
class SparseEditAVDConfig:
    """FLUX AVD settings with mandatory model-specific transition gamma."""

    num_inference_steps: int = 28
    cache_threshold: float = 0.03
    max_consecutive_hits: int = 1
    force_final_steps: int = 1
    gamma: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.num_inference_steps <= 1:
            raise ValueError("num_inference_steps must be greater than one")
        if self.gamma is None:
            raise ValueError("FLUX SparseEdit AVD requires explicit model-specific gamma values")
        if len(self.gamma) != self.num_inference_steps - 1:
            raise ValueError("sparse AVD gamma must contain num_inference_steps - 1 values")
        if not all(math.isfinite(float(value)) for value in self.gamma):
            raise ValueError("sparse AVD gamma values must be finite")
        if self.cache_threshold < 0.0:
            raise ValueError("sparse_avd_cache_threshold must be >= 0")
        if self.max_consecutive_hits < 0:
            raise ValueError("sparse_avd_max_reuse must be >= 0")
        if not 0 <= self.force_final_steps <= self.num_inference_steps:
            raise ValueError("sparse_avd_force_final_steps must be in [0, num_inference_steps]")

    @property
    def resolved_gamma(self) -> tuple[float, ...]:
        assert self.gamma is not None
        return tuple(float(value) for value in self.gamma)


@dataclass
class SparseEditAVDState:
    """Request-scoped AVD accumulation and decisions."""

    config: SparseEditAVDConfig
    accumulate: float = 1.0
    consecutive_hits: int = 0
    hit_steps: int = 0
    computed_steps: int = 0
    max_observed_consecutive_hits: int = 0
    decisions: list[AVDDecision] = field(default_factory=list)

    @staticmethod
    def _as_float(value: torch.Tensor | float) -> float:
        return float(value.detach().float().item()) if isinstance(value, torch.Tensor) else float(value)

    def _compute_decision(self, *, reason: str, ratio: float | None = None) -> AVDDecision:
        self.accumulate = 1.0
        self.consecutive_hits = 0
        self.computed_steps += 1
        decision = AVDDecision(True, ratio, 1.0, 0.0, reason)
        self.decisions.append(decision)
        return decision

    def evaluate(
        self,
        *,
        step_index: int,
        timestep: torch.Tensor | float,
        previous_timestep: torch.Tensor | float | None,
        force_compute: bool = False,
        force_reason: str = "forced_compute",
        cache_available: bool = True,
    ) -> AVDDecision:
        """Choose a transformer compute or reuse of the previous velocity."""
        if not 0 <= step_index < self.config.num_inference_steps:
            raise IndexError(f"SparseEdit AVD step index out of range: {step_index}")
        if force_compute:
            return self._compute_decision(reason=force_reason)
        if not cache_available:
            return self._compute_decision(reason="cache_unavailable")
        if previous_timestep is None or step_index == 0:
            return self._compute_decision(reason="missing_previous_timestep")
        if step_index >= self.config.num_inference_steps - self.config.force_final_steps:
            return self._compute_decision(reason="final_step")
        if self.consecutive_hits >= self.config.max_consecutive_hits:
            return self._compute_decision(reason="max_consecutive_hits")
        ratio = self.config.resolved_gamma[step_index - 1] * (
            1.0 + (self._as_float(timestep) - self._as_float(previous_timestep)) / 1000.0
        )
        if ratio >= 1.0:
            return self._compute_decision(reason="ratio_ge_one", ratio=ratio)
        candidate_accumulate = self.accumulate * ratio
        error = 1.0 - candidate_accumulate
        if error > self.config.cache_threshold:
            return self._compute_decision(reason="error_threshold", ratio=ratio)
        self.accumulate = candidate_accumulate
        self.consecutive_hits += 1
        self.hit_steps += 1
        self.max_observed_consecutive_hits = max(self.max_observed_consecutive_hits, self.consecutive_hits)
        decision = AVDDecision(False, ratio, candidate_accumulate, error, "cache_hit")
        self.decisions.append(decision)
        return decision


@dataclass
class SparseEditAVDGammaStats:
    """Accumulate adjacent velocity products without retaining graph tensors."""

    num_inference_steps: int
    records: list[dict[str, float | int]] = field(default_factory=list)
    _previous_velocity: torch.Tensor | None = field(default=None, init=False, repr=False)
    _previous_timestep: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_inference_steps <= 1:
            raise ValueError("num_inference_steps must be greater than one for AVD gamma fitting")

    def observe(self, *, step_index: int, timestep: torch.Tensor | float, guided_velocity: torch.Tensor) -> None:
        current = guided_velocity.detach().float()
        current_timestep = SparseEditAVDState._as_float(timestep)
        if self._previous_velocity is not None and self._previous_timestep is not None:
            previous = self._previous_velocity.to(current.device)
            if previous.shape != current.shape:
                raise ValueError("adjacent guided velocities must have identical shapes")
            self.records.append(
                {
                    "step_index": int(step_index),
                    "current_timestep": current_timestep,
                    "previous_timestep": self._previous_timestep,
                    "dot_current_previous": float(torch.sum(current * previous, dtype=torch.float64).item()),
                    "previous_norm_sq": float(torch.sum(previous * previous, dtype=torch.float64).item()),
                    "current_norm_sq": float(torch.sum(current * current, dtype=torch.float64).item()),
                }
            )
        self._previous_velocity = current.cpu()
        self._previous_timestep = current_timestep

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_inference_steps": self.num_inference_steps,
            "record_count": len(self.records),
            "records": list(self.records),
        }


def fit_sparseedit_avd_gamma(probe_records: list[dict[str, Any]], *, num_inference_steps: int) -> dict[str, Any]:
    """Fit one transition gamma from aggregated adjacent velocity statistics."""
    if num_inference_steps <= 1:
        raise ValueError("num_inference_steps must be greater than one")
    grouped = {
        step: {"dot": 0.0, "previous_norm_sq": 0.0, "current_norm_sq": 0.0, "count": 0}
        for step in range(1, num_inference_steps)
    }
    timesteps: dict[int, tuple[float, float]] = {}
    for record in probe_records:
        step = int(record["step_index"])
        if step not in grouped:
            raise ValueError(f"gamma probe step_index out of range: {step}")
        group = grouped[step]
        for key, source in (
            ("dot", "dot_current_previous"),
            ("previous_norm_sq", "previous_norm_sq"),
            ("current_norm_sq", "current_norm_sq"),
        ):
            group[key] += float(record[source])
        group["count"] += 1
        pair = (float(record["current_timestep"]), float(record["previous_timestep"]))
        existing = timesteps.setdefault(step, pair)
        if any(abs(left - right) > 1e-6 for left, right in zip(existing, pair, strict=True)):
            raise ValueError(f"inconsistent scheduler timesteps for step {step}")
    missing = [step for step, group in grouped.items() if not group["count"]]
    if missing:
        raise ValueError(f"missing gamma probe records for steps: {missing}")

    gamma: list[float] = []
    gamma_fp16: list[float] = []
    step_stats: list[dict[str, Any]] = []
    for step in range(1, num_inference_steps):
        group = grouped[step]
        if group["previous_norm_sq"] <= 0.0:
            raise ValueError(f"previous velocity norm is zero at step {step}")
        current, previous = timesteps[step]
        factor = 1.0 + (current - previous) / 1000.0
        if abs(factor) <= 1e-12:
            raise ValueError(f"invalid timestep factor at step {step}")
        ratio = group["dot"] / group["previous_norm_sq"]
        value = ratio / factor
        fp16 = float(torch.tensor(value, dtype=torch.float16).item())
        residual = max(
            0.0,
            group["current_norm_sq"] - 2.0 * ratio * group["dot"] + ratio * ratio * group["previous_norm_sq"],
        )
        relative_error = math.sqrt(residual / max(group["current_norm_sq"], 1e-20))
        gamma.append(value)
        gamma_fp16.append(fp16)
        step_stats.append(
            {
                "step_index": step,
                "sample_count": group["count"],
                "current_timestep": current,
                "previous_timestep": previous,
                "fitted_ratio": ratio,
                "gamma": value,
                "gamma_fp16": fp16,
                "relative_approximation_error": relative_error,
            }
        )
    errors = sorted(item["relative_approximation_error"] for item in step_stats)
    p90_index = max(0, math.ceil(0.9 * len(errors)) - 1)
    return {
        "num_inference_steps": num_inference_steps,
        "gamma": gamma,
        "gamma_fp16": gamma_fp16,
        "step_stats": step_stats,
        "mean_relative_approximation_error": sum(errors) / len(errors),
        "p90_relative_approximation_error": errors[p90_index],
    }


__all__ = [
    "AVDDecision",
    "SPARSEEDIT_AVD_UPDATE_STRATEGIES",
    "SparseEditAVDConfig",
    "SparseEditAVDGammaStats",
    "SparseEditAVDState",
    "SparseEditConditionRerankConfig",
    "fit_sparseedit_avd_gamma",
    "score_sparseedit_spatial_error_tokens",
    "select_condition_rerank_mask",
    "select_sparseedit_spatial_error_mask",
]
