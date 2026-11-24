# SPDX-License-Identifier: Apache-2.0
"""State and results for a single resumable FLUX request."""

from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .request_context import RequestContext


@dataclass
class StepSnapshot:
    step: int
    latents: torch.Tensor
    velocity: torch.Tensor


@dataclass
class RequestState:
    context: RequestContext
    pipeline: Any
    stream: torch.cuda.Stream | None = None
    config: Any = None
    continuation: Generator | None = None
    next_step: int = 0
    timesteps: Sequence = ()
    latents: torch.Tensor | None = None
    image_latents: torch.Tensor | None = None
    conditioning: dict[str, Any] = field(default_factory=dict)
    caches: tuple[Any, ...] = ()
    template: Any = None
    token_age: torch.Tensor | None = None
    velocity_history: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    capture_steps: bool = False
    steps: list[StepSnapshot] = field(default_factory=list)
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None
    started_at: float = 0

    @property
    def request_id(self) -> str:
        return self.context.request_id


@dataclass
class RequestResult:
    latents: torch.Tensor
    output: Any
    metadata: dict[str, Any]
    steps: list[StepSnapshot]
    timing: dict[str, Any]
    memory: dict[str, Any]
    output_png_sha256: str | None = None


def return_value(generator: Generator):
    """Read the return value of a continuation that must have no remaining yields."""
    try:
        next(generator)
    except StopIteration as completed:
        return completed.value
    else:
        generator.close()
        raise RuntimeError("Unexpected denoising suspension")
