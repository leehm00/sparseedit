# SPDX-License-Identifier: Apache-2.0
"""Predict allocator peaks for private SparseEdit requests on one device."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


def _validate_bytes(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class RequestFootprint:
    """Disjoint allocated costs and the optional total reserved graph pool size."""

    kv_bytes: int
    activation_bytes: int
    workspace_bytes: int
    graph_bytes: int
    graph_reserved_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("kv_bytes", "activation_bytes", "workspace_bytes", "graph_bytes"):
            _validate_bytes(name, getattr(self, name))
        if self.graph_reserved_bytes is not None:
            _validate_bytes("graph_reserved_bytes", self.graph_reserved_bytes)
            if self.graph_reserved_bytes < self.graph_bytes:
                raise ValueError("graph_reserved_bytes must cover graph_bytes")

    @property
    def allocated_bytes(self) -> int:
        return self.kv_bytes + self.activation_bytes + self.workspace_bytes + self.graph_bytes

    @property
    def graph_slack_bytes(self) -> int:
        if self.graph_reserved_bytes is None:
            return 0
        return self.graph_reserved_bytes - self.graph_bytes


class _HasFootprint(Protocol):
    @property
    def footprint(self) -> RequestFootprint: ...


def _footprint(state: RequestFootprint | _HasFootprint) -> RequestFootprint:
    footprint = state if isinstance(state, RequestFootprint) else getattr(state, "footprint", None)
    if not isinstance(footprint, RequestFootprint):
        raise TypeError("request must carry a measured or calibrated RequestFootprint")
    return footprint


class CapacityManager:
    """Count the shared worker allocation once and admit using reserved bytes."""

    def __init__(
        self,
        *,
        shared_allocated_bytes: int,
        budget_bytes: int,
        shared_reserved_slack_bytes: int = 0,
        per_request_reserved_slack_bytes: int = 0,
    ) -> None:
        for name, value in (
            ("shared_allocated_bytes", shared_allocated_bytes),
            ("budget_bytes", budget_bytes),
            ("shared_reserved_slack_bytes", shared_reserved_slack_bytes),
            ("per_request_reserved_slack_bytes", per_request_reserved_slack_bytes),
        ):
            _validate_bytes(name, value)
        self.shared_allocated_bytes = shared_allocated_bytes
        self.budget_bytes = budget_bytes
        self.shared_reserved_slack_bytes = shared_reserved_slack_bytes
        self.per_request_reserved_slack_bytes = per_request_reserved_slack_bytes

    def predict_allocated_bytes(self, states: Sequence[RequestFootprint | _HasFootprint]) -> int:
        return self.shared_allocated_bytes + sum(_footprint(state).allocated_bytes for state in states)

    def _private_reserved_bytes(self, footprint: RequestFootprint) -> int:
        return footprint.allocated_bytes + footprint.graph_slack_bytes + self.per_request_reserved_slack_bytes

    def predict_reserved_bytes(self, states: Sequence[RequestFootprint | _HasFootprint]) -> int:
        return (
            self.shared_allocated_bytes
            + self.shared_reserved_slack_bytes
            + sum(self._private_reserved_bytes(_footprint(state)) for state in states)
        )

    def predict_peak_bytes(self, states: Sequence[RequestFootprint | _HasFootprint]) -> int:
        """Return the reserved peak used for admission."""
        return self.predict_reserved_bytes(states)

    def can_admit(
        self, admitted: Sequence[RequestFootprint | _HasFootprint], candidate: RequestFootprint | _HasFootprint
    ) -> bool:
        return self.predict_peak_bytes([*admitted, candidate]) <= self.budget_bytes

    def predict_max_concurrent(self, template: RequestFootprint | _HasFootprint, budget_bytes: int) -> int:
        _validate_bytes("budget_bytes", budget_bytes)
        reserved_per_request = self._private_reserved_bytes(_footprint(template))
        if reserved_per_request <= 0:
            raise ValueError("template must have a positive request reservation")
        return max(0, (budget_bytes - self.predict_peak_bytes([])) // reserved_per_request)
