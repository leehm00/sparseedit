# SPDX-License-Identifier: Apache-2.0
"""Tensor-only stage-aware selector for the SparseEdit rebuttal study.

This module implements the current manuscript score exactly:

``base = normalize(||v[i-1] - v[i-2]||_2 * (1 + 0.5 * age))``

where normalization is a per-sample min-max normalization whose span is set to
zero when it is at most ``1e-8``.  The local score is the centered in-grid 5x5
mean of the normalized base score and is *not* renormalized.  The final
priority is ``(1 - local_weight) * base + local_weight * local`` and stable
grid-index order breaks ties.

It deliberately contains no pipeline, cache, latent, or CFG code.  Scoring is
driven only by the two preceding complete scheduler velocities and the token
age counter, never by latent change or model activations.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral

import torch
import torch.nn.functional as F

PAPER_SELECTOR_MODES = frozenset({"forward", "reverse", "constant", "random"})
_NORMALIZATION_EPS = 1e-8
_LOCAL_KERNEL_SIZE = 5
_AGE_WEIGHT = 0.5


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def paper_local_weight(
    mode: str,
    *,
    flow_time: float,
    warmup_flow_time: float,
    constant_weight: float | None = None,
) -> float:
    """Return the neighborhood weight for one sparse step.

    ``forward`` uses ``flow_time / warmup_flow_time`` and ``reverse`` uses its
    complement, where ``warmup_flow_time`` is the flow time at the first sparse
    step (``W``).  Flow times must come from the scheduler and must not be
    replaced by step indices.
    """
    if mode not in PAPER_SELECTOR_MODES:
        raise ValueError(f"unsupported paper selector mode: {mode!r}")
    flow = _finite_float(flow_time, "flow_time")
    warmup = _finite_float(warmup_flow_time, "warmup_flow_time")
    if flow < 0.0:
        raise ValueError("flow_time must be non-negative")
    if warmup <= 0.0:
        raise ValueError("warmup_flow_time must be positive")
    if flow > warmup:
        raise ValueError("flow_time must not exceed warmup_flow_time for a sparse step")
    if mode == "random":
        return 0.0
    if mode == "constant":
        if constant_weight is None:
            raise ValueError("constant mode requires constant_weight")
        weight = _finite_float(constant_weight, "constant_weight")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("constant_weight must be in [0, 1]")
        return weight
    ratio = flow / warmup
    return float(ratio if mode == "forward" else 1.0 - ratio)


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    minimum = scores.min(dim=1, keepdim=True).values
    span = scores.max(dim=1, keepdim=True).values - minimum
    safe_span = torch.where(span > _NORMALIZATION_EPS, span, torch.ones_like(span))
    normalized = (scores - minimum) / safe_span
    return torch.where(span > _NORMALIZATION_EPS, normalized, torch.zeros_like(scores))


def _validate_velocity(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, tokens, channels]")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_age(value: object, *, batch: int, count: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError("token_age must be a torch.Tensor")
    expected_shape = (batch, count)
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"token_age must have shape {expected_shape}, got {tuple(value.shape)}"
        )
    if torch.is_complex(value):
        raise ValueError("token_age must be real-valued")
    if value.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("token_age must use an integer dtype")
    if bool((value < 0).any()):
        raise ValueError("token_age must be non-negative")
    return value


def _paper_selection(
    previous_velocity: torch.Tensor,
    previous_previous_velocity: torch.Tensor,
    token_age: torch.Tensor,
    active_count: int,
    grid_height: int,
    grid_width: int,
    *,
    local_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared score/mask computation for both public selector entry points.

    Returns ``(mask, priority, selected)`` where ``selected`` holds the
    ``[batch, active_count]`` winning grid indices in selection order.  The
    caller decides whether to sort them; the mask-only API deliberately does
    not, so the baseline runtime pays no extra order-statistics work.
    """
    previous = _validate_velocity(previous_velocity, "previous_velocity")
    previous_previous = _validate_velocity(
        previous_previous_velocity, "previous_previous_velocity"
    )
    if previous.shape != previous_previous.shape:
        raise ValueError(
            "previous and previous-previous velocities must have identical shapes"
        )
    batch, count, _channels = previous.shape
    height = _positive_int(grid_height, "grid_height")
    width = _positive_int(grid_width, "grid_width")
    if height * width != count:
        raise ValueError(
            f"grid {height}x{width} does not match {count} velocity tokens"
        )
    if isinstance(active_count, bool) or not isinstance(active_count, Integral):
        raise ValueError("active_count must be an integer")
    selected_count = int(active_count)
    if not 1 <= selected_count <= count:
        raise ValueError("active_count must be in [1, token_count]")
    weight = _finite_float(local_weight, "local_weight")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("local_weight must be in [0, 1]")
    age = _validate_age(token_age, batch=batch, count=count)

    # Cast both complete velocities before subtraction.  Subtracting two BF16
    # tensors first would round a small difference at the BF16 grid.
    difference = previous.to(torch.float32) - previous_previous.to(torch.float32)
    raw = torch.linalg.vector_norm(difference, dim=-1)
    aged = raw * (1.0 + _AGE_WEIGHT * age.to(device=raw.device, dtype=torch.float32))
    normalized = _normalize_scores(aged)
    local = F.avg_pool2d(
        normalized.reshape(batch, 1, height, width),
        kernel_size=_LOCAL_KERNEL_SIZE,
        stride=1,
        padding=_LOCAL_KERNEL_SIZE // 2,
        count_include_pad=False,
    ).reshape(batch, count)
    priority = (1.0 - weight) * normalized + weight * local

    # An ascending stable sort of the negated priority preserves ascending
    # grid-index order among exactly equal scores on both CPU and CUDA.
    order = torch.argsort(-priority, dim=1, stable=True)
    selected = order[:, :selected_count]
    mask = torch.zeros((batch, count), dtype=torch.bool, device=priority.device)
    mask.scatter_(1, selected, True)
    return mask, priority, selected


def random_generator_seed(seed, step):
    for name, value in (("seed", seed), ("step", step)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= value < 2**63
        ):
            raise ValueError(f"{name} must be a nonnegative 63-bit integer")
    payload = f"sparseedit-random-v1:{int(seed)}:{int(step)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def select_random_tokens(*, batch, count, active_count, device, seed, step):
    """Uniform exact-K mask with a request/step-owned generator and sorted rows."""
    batch = _positive_int(batch, "batch")
    count = _positive_int(count, "count")
    active_count = _positive_int(active_count, "active_count")
    if active_count > count:
        raise ValueError("active_count exceeds generated token count")
    for name, value in (("seed", seed), ("step", step)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= value < 2**63
        ):
            raise ValueError(f"{name} must be a nonnegative 63-bit integer")
    generator = torch.Generator(device=device)
    generator.manual_seed(random_generator_seed(seed, step))
    mask = torch.zeros((batch, count), dtype=torch.bool, device=device)
    ranks = torch.empty((batch, count), dtype=torch.float32, device=device)
    selections = []
    for row in range(batch):
        permutation = torch.randperm(count, generator=generator, device=device)
        ranks[row].scatter_(
            0,
            permutation,
            torch.arange(count, 0, -1, dtype=torch.float32, device=device),
        )
        indices = permutation[:active_count].sort().values
        mask[row].scatter_(0, indices, True)
        selections.append(indices)
    return mask, ranks, torch.stack(selections)


def select_paper_tokens(
    previous_velocity: torch.Tensor,
    previous_previous_velocity: torch.Tensor,
    token_age: torch.Tensor,
    active_count: int,
    grid_height: int,
    grid_width: int,
    *,
    local_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select tokens and return ``(mask, priority, sorted_indices)``.

    ``sorted_indices`` has shape ``[batch, active_count]`` and is sorted in
    ascending grid order, exactly matching ``mask.nonzero()[:, 1]``.  Returning
    it directly lets the fixed-K runtime avoid a host wait for the
    ``nonzero`` result shape while preserving the selector's exact stable tie
    rule.  This extra sort is paid only by callers that request indices.
    """
    mask, priority, selected = _paper_selection(
        previous_velocity,
        previous_previous_velocity,
        token_age,
        active_count,
        grid_height,
        grid_width,
        local_weight=local_weight,
    )
    sorted_indices = torch.sort(selected, dim=1, stable=True).values
    return mask, priority, sorted_indices


def select_paper_mask(
    previous_velocity: torch.Tensor,
    previous_previous_velocity: torch.Tensor,
    token_age: torch.Tensor,
    active_count: int,
    grid_height: int,
    grid_width: int,
    *,
    local_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select generated tokens with the manuscript's velocity-history score.

    Returns ``(mask, priority)`` and performs no index sorting of its own, so
    the baseline runtime cost is unchanged from the pre-specialization
    implementation.  See :func:`select_paper_tokens` for the sorted fixed-K
    indices used by the specialized runtime.
    """
    mask, priority, _selected = _paper_selection(
        previous_velocity,
        previous_previous_velocity,
        token_age,
        active_count,
        grid_height,
        grid_width,
        local_weight=local_weight,
    )
    return mask, priority


__all__ = [
    "PAPER_SELECTOR_MODES",
    "paper_local_weight",
    "select_paper_mask",
    "select_paper_tokens",
    "select_random_tokens",
]
