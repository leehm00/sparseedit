# SPDX-License-Identifier: Apache-2.0
"""Persistent native-row condition preparation with isolated graph producers."""

from __future__ import annotations

import hashlib
import time

import torch


def _layout(value):
    return (
        None
        if value is None
        else (
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            str(value.device),
        )
    )


def _signature(transformer, timesteps, pooled, guidance, dtype):
    from .condition_projection_plan import condition_norms

    modules = condition_norms(transformer)
    weights = []
    for owner in (transformer.time_text_embed, *(module.linear for module in modules)):
        for parameter in owner.parameters():
            try:
                version = parameter._version
            except RuntimeError:
                version = None
            weights.append(
                (id(parameter), parameter.data_ptr(), version, _layout(parameter))
            )
    return (
        id(transformer),
        id(transformer.time_text_embed),
        tuple(id(module) for module in modules),
        tuple(weights),
        tuple((name, _layout(value)) for name, value in pooled.items()),
        _layout(timesteps),
        _layout(guidance),
        str(dtype),
    )


class ConditionPreparationGraph:
    """Graph-owned inputs/producer outputs; live current views stay outside capture."""

    def __init__(self):
        from .condition_projection_plan import RequestConditionPlan

        self.reference = RequestConditionPlan(mode="serial")
        self.signature = None
        self.inputs = None
        self.graph = None
        self.capture_stream = None
        self.capture_stream_id = None
        self.capture_count = self.capture_attempts = self.replay_count = (
            self.cpu_reference_calls
        ) = 0
        self.warmup_count = 0
        self.closed = self.failed = False
        self.backend = "not_bound"
        self.request = {}
        self.cleanup_wall_seconds = 0.0

    def _run_native(self, transformer, dtype):
        self.reference.bind(
            transformer,
            self.inputs["timesteps"],
            self.inputs["pooled"],
            guidance=self.inputs["guidance"],
            dtype=dtype,
        )

    def _capture_native(self, transformer, dtype, device):
        producer = torch.cuda.current_stream(device)
        if self.capture_stream is None:
            self.capture_stream = torch.cuda.Stream(device=device, priority=0)
            self.capture_stream_id = self.capture_stream.cuda_stream
        self.capture_stream.wait_stream(producer)
        # Prime native M=1 operations on the same owned nondefault stream that
        # will capture them. Replay is issued only after this context restores
        # the caller's selected serial/auxiliary producer stream.
        with torch.cuda.stream(self.capture_stream):
            started = time.perf_counter()
            for _ in range(2):
                self._run_native(transformer, dtype)
                self.warmup_count += 1
                self.request["request_warmup_count"] = (
                    self.request.get("request_warmup_count", 0) + 1
                )
            self.request["cold_warmup_wall_seconds"] = time.perf_counter() - started
            graph = torch.cuda.CUDAGraph()
            self.graph = graph
            self.capture_attempts += 1
            started = time.perf_counter()
            self.request["cold_capture_synchronizes"] = True
            try:
                with torch.cuda.graph(graph, stream=self.capture_stream):
                    self._run_native(transformer, dtype)
            finally:
                self.request["cold_capture_wall_seconds"] = (
                    time.perf_counter() - started
                )
            self.capture_count += 1
            self.request["request_capture_count"] = 1

    @torch.no_grad()
    def bind(self, transformer, timesteps, pooled, *, guidance, dtype):
        if self.closed or self.failed:
            raise RuntimeError("condition preparation graph is closed or failed")
        signature = _signature(transformer, timesteps, pooled, guidance, dtype)
        if self.signature is not None and signature != self.signature:
            raise ValueError(
                "condition preparation graph model/weight/input layout changed; new graph plan required"
            )
        device = next(iter(pooled.values())).device
        self.backend = "cuda_graph" if device.type == "cuda" else "cpu_graph_reference"
        cache_hit = self.signature is not None
        self.request = {
            "cache_hit": cache_hit,
            "request_bind_count": 1,
            "request_capture_count": 0,
            "request_replay_count": 0,
            "request_warmup_count": 0,
            "bind_wall_seconds": 0.0,
            "cold_warmup_wall_seconds": 0.0,
            "cold_capture_wall_seconds": 0.0,
            "replay_enqueue_wall_seconds": 0.0,
            "cold_capture_synchronizes": False,
        }
        started = time.perf_counter()
        if self.inputs is None:
            self.inputs = {
                "timesteps": timesteps.detach().to(device).clone(),
                "pooled": {
                    name: value.detach().clone() for name, value in pooled.items()
                },
                "guidance": None
                if guidance is None
                else guidance.detach().to(device).clone(),
            }
            self.signature = signature
        else:
            self.inputs["timesteps"].copy_(timesteps)
            for name, value in pooled.items():
                self.inputs["pooled"][name].copy_(value)
            if guidance is not None:
                self.inputs["guidance"].copy_(guidance)
        self.request["bind_wall_seconds"] = time.perf_counter() - started
        try:
            if device.type != "cuda":
                self._run_native(transformer, dtype)
                self.cpu_reference_calls += 1
            else:
                if self.graph is None:
                    self._capture_native(transformer, dtype, device)
                self.request["replay_stream_id"] = torch.cuda.current_stream(
                    device
                ).cuda_stream
                started = time.perf_counter()
                self.graph.replay()
                self.replay_count += 1
                self.request["request_replay_count"] = 1
                self.request["replay_enqueue_wall_seconds"] = (
                    time.perf_counter() - started
                )
        except BaseException:
            self.failed = True
            raise
        return self.reference

    def additional_owned_bytes(self):
        if self.inputs is None or self.reference is None:
            return 0
        tensors = [
            self.inputs["timesteps"],
            *self.inputs["pooled"].values(),
            self.inputs["guidance"],
            self.reference.current_temb,
            self.reference.current_condition,
            self.reference.current_projected,
        ]
        return sum(
            value.numel() * value.element_size()
            for value in tensors
            if value is not None
        )

    def metadata(self):
        return {
            "requested": True,
            "backend": self.backend,
            "capture_count": self.capture_count,
            "capture_attempts": self.capture_attempts,
            "replay_count": self.replay_count,
            "cpu_reference_calls": self.cpu_reference_calls,
            "warmup_count": self.warmup_count,
            "graph_identity": id(self),
            "capture_stream_id": self.capture_stream_id,
            "signature_sha256": None
            if self.signature is None
            else hashlib.sha256(repr(self.signature).encode()).hexdigest(),
            "additional_owned_tensor_bytes": self.additional_owned_bytes(),
            "graph_private_pool_bytes": None,
            "weight_guard": "identity_storage_layout_and_available_parameter_version; versionless inference weights remain immutable",
            "closed": self.closed,
            "failed": self.failed,
            "cleanup_wall_seconds": self.cleanup_wall_seconds,
            **self.request,
        }

    def close(self):
        if self.closed:
            return
        started = time.perf_counter()
        try:
            if self.graph is not None:
                self.graph.reset()
        finally:
            self.graph = self.inputs = self.reference = self.capture_stream = None
            self.closed = True
            self.cleanup_wall_seconds = time.perf_counter() - started
