# SPDX-License-Identifier: Apache-2.0
"""Request-scoped CUDA Graph capture for the FLUX specialized sparse forward.

The captured region is the real per-branch ``predict_noise`` call, i.e. the
whole ``FluxKontextTransformer2DModel.forward`` for one CFG branch at the fixed
sparse shape.  The sparse selector, scheduler step, velocity history and token
ages stay *outside* the graph; capture trials only re-run the stateless
transformer on buffers that already hold the current step's data, so trial
writes are idempotent.

The class never reports ``graph_active=True`` for an eager run.  If CUDA Graph
capture raises and the caller opted into ``sparseedit_graph_fallback``, the
metadata records the failure reason and the caller must use an eager path.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from .track_a_timing import copy_ledger

GRAPH_CAPTURED_REGION = (
    "pipeline.predict_noise -> FluxKontextTransformer2DModel.forward "
    "(all dual/single blocks, CFG branch fixed, sparse fixed-K shape)"
)


class GraphCaptureError(RuntimeError):
    """Raised when a requested CUDA graph cannot be captured safely."""


@dataclass
class BranchSparseGraph:
    """One CFG branch's captured sparse transformer forward."""

    branch: str
    allow_fallback: bool = False
    warmup_trials: int = 2
    captured_region: str = GRAPH_CAPTURED_REGION
    borrow_output: bool = False
    shared_pool: Any = field(default=None, repr=False)
    graph_active: bool = False
    fallback_reason: str | None = None
    warmup_calls: int = 0
    capture_calls: int = 0
    replay_count: int = 0
    capture_seconds: float = 0.0
    warmup_seconds: float = 0.0
    preparation_seconds: float = 0.0
    capture_pool_bytes: int | None = None
    capture_reserved_delta_bytes: int = 0
    capture_stream: str | None = None
    _graph: Any = field(default=None, repr=False)
    _graph_output: torch.Tensor | None = field(default=None, repr=False)
    _owned_output: torch.Tensor | None = field(default=None, repr=False)

    def disable(self, reason: str) -> None:
        """Mark this branch as an explicitly-requested eager fallback.

        Used when the request is known up front to be incompatible with
        capture (for example a step callback that may replace the prompt
        embeddings a captured closure would have bound).  The branch then
        reports ``graph_active=False`` with the given reason and
        :meth:`capture` refuses, so no eager run can be labelled graph.
        """
        self.fallback_reason = reason
        self.graph_active = False
        self._graph = None
        self._graph_output = None
        self._owned_output = None

    def capture(self, fn: Callable[[], torch.Tensor], *, device: torch.device | str) -> bool:
        target = torch.device(device)
        if self.fallback_reason is not None:
            raise GraphCaptureError(
                f"branch {self.branch!r} was explicitly disabled for this request: {self.fallback_reason}"
            )
        if target.type != "cuda" or not torch.cuda.is_available():
            return self._fallback("cuda_unavailable")
        if self.graph_active:
            raise GraphCaptureError("this branch graph has already been captured")
        try:
            return self._capture_cuda(fn, target)
        except GraphCaptureError:
            raise
        except Exception as error:  # noqa: BLE001 - capture failures are heterogeneous
            return self._fallback(f"{type(error).__name__}: {error}")

    def _fallback(self, reason: str) -> bool:
        if not self.allow_fallback:
            raise GraphCaptureError(
                f"CUDA graph capture failed for branch {self.branch!r}: {reason}. "
                "Pass sparseedit_graph_fallback=True to use an explicitly-labelled eager fallback."
            )
        self.fallback_reason = reason
        self.graph_active = False
        self._graph = None
        self._graph_output = None
        self._owned_output = None
        return False

    def _capture_cuda(self, fn: Callable[[], torch.Tensor], device: torch.device) -> bool:
        preparation_started = time.perf_counter()
        torch.accelerator.synchronize(device)
        warmup_started = time.perf_counter()
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream):
            with torch.no_grad():
                for _ in range(self.warmup_trials):
                    fn()
                    self.warmup_calls += 1
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.accelerator.synchronize(device)
        self.warmup_seconds = time.perf_counter() - warmup_started
        graph = torch.cuda.CUDAGraph()
        pool = self.shared_pool if self.shared_pool is not None else torch.cuda.graph_pool_handle()
        reserved_before = torch.accelerator.memory_reserved(device)
        started = time.perf_counter()
        with torch.cuda.graph(graph, pool=pool):
            with torch.no_grad():
                output = fn()
                self.capture_calls += 1
        torch.accelerator.synchronize(device)
        self.capture_seconds = time.perf_counter() - started
        self.capture_reserved_delta_bytes = int(torch.accelerator.memory_reserved(device) - reserved_before)
        # Global reserved bytes can fall when capture releases unrelated cached
        # blocks. Query the graph's own pool for its actual reserved capacity.
        try:
            segments = torch.cuda.memory_snapshot(mempool_id=pool, include_traces=False)
            self.capture_pool_bytes = sum(int(segment["total_size"]) for segment in segments)
        except (TypeError, RuntimeError):
            self.capture_pool_bytes = None
        if not isinstance(output, torch.Tensor):
            raise GraphCaptureError("captured forward did not return a tensor")
        self._graph = graph
        self._graph_output = output
        # Owned outside the graph pool so a later replay of another branch that
        # shares allocator state cannot overwrite an output still needed by the
        # CFG combination.
        if not self.borrow_output:
            self._owned_output = torch.empty_like(output)
        self.preparation_seconds = time.perf_counter() - preparation_started
        self.graph_active = True
        self.fallback_reason = None
        return True

    def replay(self, *, snapshot: bool = False) -> torch.Tensor:
        if not self.graph_active or self._graph is None or self._graph_output is None:
            raise GraphCaptureError(
                f"branch {self.branch!r} graph is not active; refusing to label eager work as graph"
            )
        self._graph.replay()
        self.replay_count += 1
        if snapshot:
            return self._graph_output.clone()
        if self.borrow_output:
            return self._graph_output
        if self._owned_output is None:
            raise GraphCaptureError("graph output owner was released")
        with copy_ledger():
            self._owned_output.copy_(self._graph_output)
        return self._owned_output

    def output_storage_ptr(self) -> int:
        if self._graph_output is None:
            raise GraphCaptureError("graph output is not captured")
        return self._graph_output.untyped_storage().data_ptr()

    def metadata(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "requested": True,
            "graph_active": bool(self.graph_active),
            "captured_region": self.captured_region if self.graph_active else None,
            "warmup_trials": int(self.warmup_calls),
            "capture_trials": int(self.capture_calls),
            "capture_seconds": float(self.capture_seconds),
            "warmup_seconds": float(self.warmup_seconds),
            "preparation_seconds": float(self.preparation_seconds),
            "capture_pool_bytes": self.capture_pool_bytes,
            "capture_pool_bytes_source": (
                "allocator_pool_snapshot" if self.capture_pool_bytes is not None else "unavailable"
            ),
            "capture_reserved_delta_bytes": self.capture_reserved_delta_bytes,
            "replay_count": int(self.replay_count),
            "borrowed_output": self.borrow_output,
            "fallback": self.fallback_reason,
            "eager_fallback": bool(self.fallback_reason is not None and not self.graph_active),
        }

    def close(self) -> None:
        self._graph = None
        self._graph_output = None
        self._owned_output = None
        self.graph_active = False


@dataclass
class RequestGraphState:
    """Per-request graph registry with the exact cleanup contract."""

    positive: BranchSparseGraph
    negative: BranchSparseGraph | None = None
    allow_fallback: bool = False

    def metadata(self) -> dict[str, Any]:
        branches = {"positive": self.positive.metadata()}
        if self.negative is not None:
            branches["negative"] = self.negative.metadata()
        active = all(item["graph_active"] for item in branches.values())
        replays = {name: item["replay_count"] for name, item in branches.items()}
        return {
            "requested": True,
            "graph_active": active,
            "branches": branches,
            "replay_counts": replays,
            "total_replays": sum(replays.values()),
            "fallback": any(item["fallback"] for item in branches.values()),
        }

    def disable(self, reason: str) -> None:
        """Disable every branch up front and record why capture was skipped."""
        self.positive.disable(reason)
        if self.negative is not None:
            self.negative.disable(reason)

    def close(self) -> None:
        self.positive.close()
        if self.negative is not None:
            self.negative.close()


__all__ = [
    "GRAPH_CAPTURED_REGION",
    "BranchSparseGraph",
    "GraphCaptureError",
    "RequestGraphState",
]
