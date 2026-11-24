# SPDX-License-Identifier: Apache-2.0
"""Explicit request-local inputs for the FLUX paper selector."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RequestContext:
    request_id: str
    step_index: int
    timestep: torch.Tensor | None
    base_attention_kwargs: dict[str, Any]
    stream: torch.cuda.Stream | None = None


def context_kwargs(base: dict[str, Any], context: RequestContext) -> dict[str, Any]:
    return {**base, "_flux_request_context": context}
