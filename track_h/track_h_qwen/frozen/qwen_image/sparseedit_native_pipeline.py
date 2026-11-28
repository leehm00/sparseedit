# SPDX-License-Identifier: Apache-2.0
"""Request-scoped interception of native Qwen's two-CFG edit denoising loop.

The installed pipeline retains text encoding, seeded noise, reference latents,
CFG normalization/rescaling, VAE decoding and scheduler stepping. Only model and
scheduler calls are intercepted, and all wrappers are restored in ``finally``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from .sparseedit_native_partition import synchronize_devices
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from .sparseedit_native_buffers import (
    QwenStableBuffers,
)
from .sparseedit_native_runtime import (
    QwenSparseEditConfig,
    QwenSparseKVCache,
    qwen_sparse_model_call,
)
from .sparseedit_native_scheduler import (
    QwenVelocityHistory,
    native_cfg_rescale,
)
from .sparseedit_native_trace import (
    QwenFixedMaskTrace,
    QwenTraceStep,
)


@dataclass
class QwenDeviceEventTimer:
    enabled: bool = False
    events: list[tuple[str, Any, Any]] = field(default_factory=list)
    device: Any = None

    def mark(self, name: str) -> None:
        if self.enabled:
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            with torch.cuda.device(self.device):
                start.record()
            self.events.append((name, start, end))

    def end(self) -> None:
        if self.enabled and self.events:
            with torch.cuda.device(self.device):
                self.events[-1][2].record()

    def milliseconds(self) -> list[dict[str, float | str]]:
        if not self.events:
            return []
        torch.cuda.synchronize(self.device)
        return [
            {"name": name, "milliseconds": start.elapsed_time(end), "device": str(self.device or "cuda:current"), "scope": "device0_events" if self.device is not None else "single_device_events"}
            for name, start, end in self.events
        ]


@dataclass
class QwenDenoisingResult:
    output: Any
    trace: QwenFixedMaskTrace
    metadata: dict[str, Any]


@dataclass
class _QwenBranch:
    cache: QwenSparseKVCache
    pending: torch.Tensor | None = None


class QwenSparseDenoiser:
    def __init__(
        self,
        pipeline,
        config: QwenSparseEditConfig,
        *,
        mode: str = "direct",
        graph: bool = False,
        graph_fallback: bool = False,
    ) -> None:
        if mode not in ("direct", "inplace", "ready"):
            raise ValueError("unsupported Qwen execution mode")
        if graph and getattr(getattr(pipeline, "transformer", None), "_sparseedit_partition", None) is not None:
            raise ValueError("partition execution rejects whole-transformer graph mode; per-stage graph admission required")
        if graph and mode != "ready":
            raise ValueError(
                "CUDA graphs require stable ready-mode post-RoPE KV storage"
            )
        self.pipeline = pipeline
        self.config = config
        self.mode = mode
        self.graph = graph
        self.graph_fallback = graph_fallback
        self.history = QwenVelocityHistory(config)
        self.branches = {
            name: _QwenBranch(QwenSparseKVCache(mode))
            for name in ("positive", "negative")
        }
        self.graph_buffers: dict[str, QwenStableBuffers] = {}
        self.branch_layouts = {}
        self.timer = QwenDeviceEventTimer()
        self.step = -1
        self.action = "full"
        self.indices = None
        self.full_img_shapes = None
        self.generated_tokens = 0
        self.reference_tokens = 0
        self.trace_device_rows = []
        self.branch_order = []
        self.true_cfg_calls = 0
        self.graph_replays = {"positive": 0, "negative": 0}
        self.graph_fallbacks = []
        self.fixed_trace = None
        self.admission = None
        self._flow_times = None
        self._warmup_flow_time = None
        self._fixed_indices = None
        self._grid_shape = None
        self.denoising_started = None
        self.denoising_ended = None

    def _timesteps(self):
        timesteps = getattr(self.pipeline.scheduler, "timesteps", None)
        if timesteps is None or len(timesteps) != self.config.num_steps:
            raise RuntimeError(
                "native Qwen pipeline did not prepare exactly 28 timesteps"
            )
        return timesteps

    def _start_step(self):
        if self.step >= self.config.num_steps - 1:
            raise RuntimeError("native Qwen made too many denoising steps")
        self.step += 1
        self.branch_order = []
        flow_time = self._flow_times[self.step]
        self.action = "full" if self.step in self.config.dense_steps else "sparse"
        if self.fixed_trace is not None:
            row = self.fixed_trace.steps[self.step]
            self.action = row.action
            self.indices = self._fixed_indices[self.step]
        elif self.action == "full":
            self.indices = None
        else:
            height, width = self._grid_shape
            selection = self.history.select(
                flow_time=flow_time,
                warmup_flow_time=self._warmup_flow_time,
                grid_height=height,
                grid_width=width,
            )
            if selection is None:
                raise RuntimeError(
                    f"sparse step {self.step} lacks two complete CFG velocities"
                )
            self.indices = selection.indices[0]
        self.trace_device_rows.append(
            (
                self.step,
                self.action,
                flow_time,
                None if self.indices is None else self.indices.detach().clone(),
            )
        )

    def _prepare_branch(self, branch, call, hidden_states):
        if self.action == "full":
            return call, hidden_states, self.indices
        hidden_states = hidden_states.index_select(
            1, self.indices.to(hidden_states.device)
        )
        call = dict(call)
        if self.graph:
            if branch not in self.graph_buffers:
                self.graph_buffers[branch] = QwenStableBuffers(
                    self.indices.numel(),
                    hidden_states.shape[2],
                    hidden_states.dtype,
                    hidden_states.device,
                    text_tokens=call["encoder_hidden_states"].shape[1],
                    text_channels=call["encoder_hidden_states"].shape[2],
                    timestep_dtype=call["timestep"].dtype,
                )
            buffers = self.graph_buffers[branch]
            buffers.bind(
                latents=hidden_states,
                indices=self.indices,
                timestep=call["timestep"],
                text=call["encoder_hidden_states"],
                text_mask=call.get("encoder_hidden_states_mask"),
            )
            call.update(
                hidden_states=buffers.latents,
                timestep=buffers.timestep,
                encoder_hidden_states=buffers.text,
                encoder_hidden_states_mask=buffers.text_mask,
            )
            return call, hidden_states, buffers.indices
        call["hidden_states"] = hidden_states
        return call, hidden_states, self.indices

    def _forward(self, branch, **kwargs):
        state = self.branches[branch]
        call = dict(kwargs)
        text = call["encoder_hidden_states"]
        layout = (text.shape, text.dtype, text.device)
        if branch in self.branch_layouts and self.branch_layouts[branch] != layout:
            raise ValueError(f"Qwen branch {branch} text layout changed within request")
        self.branch_layouts[branch] = layout
        hidden_states = call["hidden_states"]
        call, _, indices = self._prepare_branch(branch, call, hidden_states)
        # CPU trace validation or fixed-count stable-sort selection precedes graph binding.
        if self.graph and self.action == "sparse":
            call = qwen_sparse_model_call(
                self.pipeline.transformer,
                cache=state.cache,
                branch=branch,
                mode="sparse",
                active_indices=indices,
                generated_tokens=self.generated_tokens,
                full_img_shapes=self.full_img_shapes,
                native_call=self._original_transformer_forward,
                prepare_only=True,
                prepared=True,
                **call,
            )

        def forward():
            return qwen_sparse_model_call(
                self.pipeline.transformer,
                cache=state.cache,
                branch=branch,
                mode="dense" if self.action == "full" else "sparse",
                active_indices=indices,
                generated_tokens=self.generated_tokens,
                full_img_shapes=self.full_img_shapes,
                native_call=self._original_transformer_forward,
                prepared=True,
                **call,
            )

        if self.graph and self.action == "sparse":
            buffers = self.graph_buffers[branch]
            if buffers.graph is None:
                try:
                    buffers.capture(forward)
                except Exception as error:
                    if not self.graph_fallback:
                        raise
                    self.graph_fallbacks.append(
                        f"{branch}:{type(error).__name__}:{error}"
                    )
            if buffers.graph is not None:
                prediction = buffers.replay()
                self.graph_replays[branch] += 1
            else:
                prediction = forward()
                if self.graph_fallback:
                    self.graph_fallbacks.append(
                        f"{branch}:step{self.step}:eager_not_optimized"
                    )
        else:
            prediction = forward()
        state.pending = prediction
        self.true_cfg_calls += 1
        return prediction

    def _transformer_call(self, *args, **kwargs):
        if args:
            raise RuntimeError("native Qwen wrapper expects keyword arguments")
        if self.full_img_shapes is None:
            self.full_img_shapes = kwargs["img_shapes"]
            shapes = self.full_img_shapes[0]
            tokens = [shape[0] * shape[1] * shape[2] for shape in shapes]
            if len(tokens) != 2 or kwargs["hidden_states"].shape[1] != sum(tokens):
                raise ValueError(
                    "Qwen SparseEdit requires original generated+reference image shapes"
                )
            self.generated_tokens, self.reference_tokens = tokens
            self._grid_shape = shapes[0][1:]
            self._flow_times = (
                self._timesteps().detach().to("cpu", torch.float32).div(1000).tolist()
            )
            self._warmup_flow_time = self._flow_times[self.config.warmup_steps]
            if self.fixed_trace is not None:
                if (
                    self.fixed_trace.generated_tokens != self.generated_tokens
                    or self.fixed_trace.active_count
                    != max(1, int(self.config.ratio * self.generated_tokens))
                ):
                    raise ValueError(
                        "fixed Qwen trace generated count or active ratio mismatch"
                    )
                for row in self.fixed_trace.steps:
                    expected = (
                        "full" if row.step in self.config.dense_steps else "sparse"
                    )
                    if row.action != expected:
                        raise ValueError(
                            "fixed trace action differs from current Qwen protocol"
                        )
                self._fixed_indices = [
                    None
                    if row.action == "full"
                    else torch.tensor(
                        row.indices,
                        dtype=torch.long,
                        device=kwargs["hidden_states"].device,
                    )
                    for row in self.fixed_trace.steps
                ]
        elif kwargs["img_shapes"] != self.full_img_shapes:
            raise ValueError("Qwen native image layout changed within request")
        if not self.branch_order:
            if self.denoising_started is None:
                if kwargs["hidden_states"].device.type == "cuda":
                    synchronize_devices(self.pipeline.transformer)
                    self.timer.enabled = True
                    self.timer.device = kwargs["hidden_states"].device
                self.denoising_started = time.perf_counter()
            self._start_step()
            self.timer.mark(f"step{self.step}:bind+2forwards+cfg+velocity_commit")
        if len(self.branch_order) >= 2:
            raise RuntimeError("native Qwen made more than two CFG calls")
        branch = ("positive", "negative")[len(self.branch_order)]
        self.branch_order.append(branch)
        prediction = self._forward(branch, **kwargs)
        return (
            (prediction,)
            if kwargs.get("return_dict", True) is False
            else Transformer2DModelOutput(sample=prediction)
        )

    def _scheduler_step(self, noise_pred, timestep, latents, *args, **kwargs):
        if self.step < 0 or self.branch_order != ["positive", "negative"]:
            raise RuntimeError(
                "Qwen scheduler commit requires real positive and negative CFG calls"
            )
        if self.action == "sparse":
            complete = self.history.commit_sparse(
                self.branches["positive"].pending,
                self.branches["negative"].pending,
                self.indices,
                true_cfg_scale=self._true_cfg_scale,
            )
        else:
            complete = native_cfg_rescale(
                self.branches["positive"].pending,
                self.branches["negative"].pending,
                true_cfg_scale=self._true_cfg_scale,
            )
            if self.history.ages is not None:
                self.history.ages.zero_()
        self.timer.end()
        if self.admission is not None:
            self.admission.append(
                {
                    "step": self.step,
                    "action": self.action,
                    "positive": self.branches["positive"].pending.detach().clone(),
                    "negative": self.branches["negative"].pending.detach().clone(),
                    "velocity": complete.detach().clone(),
                    "latents": latents.detach().clone(),
                }
            )
        self.timer.mark(f"euler_history{self.step}")
        result = self._original_scheduler_step(
            complete, timestep, latents, *args, **kwargs
        )
        self.history.record(complete, flow_time=self._flow_times[self.step])
        self.timer.end()
        if self.admission is not None:
            self.admission[-1]["latents_after"] = result[0].detach().clone()
        self.branch_order = []
        if self.step == self.config.num_steps - 1:
            if self.timer.enabled:
                synchronize_devices(self.pipeline.transformer)
            self.denoising_ended = time.perf_counter()
        return result

    def run(
        self,
        *pipeline_args,
        fixed_trace=None,
        admission_tensors=False,
        identity=None,
        source_hashes=None,
        **pipeline_kwargs,
    ):
        if self.step != -1:
            raise RuntimeError("denoiser is request-scoped")
        if (
            pipeline_kwargs.get("true_cfg_scale", 1.0) <= 1
            or pipeline_kwargs.get("negative_prompt") is None
            and pipeline_kwargs.get("negative_prompt_embeds") is None
        ):
            raise ValueError(
                "Qwen SparseEdit requires two CFG branches with a real negative condition"
            )
        if pipeline_kwargs.get("num_inference_steps") != self.config.num_steps:
            raise ValueError("Qwen protocol requires 28 steps")
        if fixed_trace is not None:
            fixed_trace.validate()
        self.fixed_trace = fixed_trace
        self.admission = [] if admission_tensors else None
        self._true_cfg_scale = float(pipeline_kwargs["true_cfg_scale"])
        self._original_scheduler_step = self.pipeline.scheduler.step
        self._original_transformer_forward = self.pipeline.transformer.forward
        self.pipeline.transformer.forward = self._transformer_call
        self.pipeline.scheduler.step = self._scheduler_step
        started = time.perf_counter()
        try:
            output = self.pipeline(*pipeline_args, **pipeline_kwargs)
        finally:
            self.pipeline.transformer.forward = self._original_transformer_forward
            self.pipeline.scheduler.step = self._original_scheduler_step
        synchronize_devices(self.pipeline.transformer)
        native_request_wall_seconds = time.perf_counter() - started
        if self.step != self.config.num_steps - 1 or self.branch_order:
            raise RuntimeError("incomplete native Qwen denoising")
        denoising_wall_seconds = (
            self.denoising_ended - self.denoising_started
            if self.denoising_started is not None and self.denoising_ended is not None
            else None
        )
        rows = [
            QwenTraceStep(
                step,
                action,
                list(range(self.generated_tokens))
                if action == "full"
                else indices.cpu().tolist(),
                flow_time,
            )
            for step, action, flow_time, indices in self.trace_device_rows
        ]
        trace = QwenFixedMaskTrace.create(
            generated_tokens=self.generated_tokens,
            active_count=len(rows[self.config.warmup_steps].indices),
            steps=rows,
            identity=identity,
            source_hashes=source_hashes,
        )
        metadata = {
            "mode": self.mode,
            "partition": getattr(self.pipeline.transformer, "_sparseedit_partition", None).payload
            if getattr(self.pipeline.transformer, "_sparseedit_partition", None) is not None else None,
            "sparse_auxiliary_transfers": {name: list(state.cache.local_aux.transfers) for name, state in self.branches.items()},
            "native_block_transfers": list(getattr(getattr(self.pipeline.transformer, "_sparseedit_transfer_cache", None), "transfers", [])),
            "transfer_counter_scope": "explicit_block_arguments_and_sparse_wrapper_auxiliary_copies; accelerate_return_and_component_staging_transfers_included_in_synchronized_wall",
            "arithmetic": "native_common",
            "true_cfg_calls": self.true_cfg_calls,
            "cfg_normalization": "native_cond_norm_over_combined_norm",
            "graph_requested": self.graph,
            "graph_replays": self.graph_replays,
            "graph_captures": {
                name: buffer.capture_count
                for name, buffer in self.graph_buffers.items()
            },
            "graph_fallbacks": self.graph_fallbacks,
            "device_events": self.timer.milliseconds(),
            "denoising_wall_seconds": denoising_wall_seconds,
            "native_request_wall_seconds": native_request_wall_seconds,
            "whole_pipeline_wall_seconds": time.perf_counter() - started,
            "net_timing_scope": "indices_ready_through_binding_two_cfg_forwards_and_complete_velocity_commit",
            "timing_scope": "net_events_exclude_selection_euler_history_and_admission_snapshots; wall_includes_all_denoising_work",
            "index_validation": "CPU_trace_validated_before_loop_or_unique_bounded_stable_sort_producer; no_device_to_host_index_validation",
            "admission_copy_scope": "device_snapshots_after_net_interval_host_transfer_after_request"
            if self.admission is not None
            else None,
            "generated_tokens": self.generated_tokens,
            "reference_tokens": self.reference_tokens,
            "reference_tokens_stale_until_refresh": True,
            "velocity_reuse": "inactive_rows_only_no_step_skip",
        }
        if self.admission is not None:
            metadata["admission_tensors"] = [
                {
                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in row.items()
                }
                for row in self.admission
            ]
        return QwenDenoisingResult(output, trace, metadata)


__all__ = ["QwenDenoisingResult", "QwenDeviceEventTimer", "QwenSparseDenoiser"]
