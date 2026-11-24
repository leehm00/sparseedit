# SPDX-License-Identifier: Apache-2.0
"""Resume the same FLUX request on a private scheduler and CUDA stream."""

import copy
from contextlib import nullcontext
from time import perf_counter

import torch

from .request_context import RequestContext
from .request_state import RequestResult, RequestState, return_value
from .specialized_template import clear_template


def private_pipeline(pipeline):
    # Weight modules are shared. Top-level request attributes, scheduler and
    # template/condition owners are private to the execution.
    request_pipeline = copy.copy(pipeline)
    request_pipeline.scheduler = copy.deepcopy(pipeline.scheduler)
    request_pipeline._sparseedit_template_pool = None
    request_pipeline._sparseedit_forward_template = None
    request_pipeline._sparseedit_condition_bank = None
    return request_pipeline


class RequestExecutor:
    def __init__(self, pipeline, *, post_process, capture_steps=False) -> None:
        self.pipeline = pipeline
        self.post_process = post_process
        self.capture_steps = capture_steps

    @staticmethod
    def stream_context(state):
        return torch.cuda.stream(state.stream) if state.stream is not None else nullcontext()

    @torch.inference_mode()
    def init_state(self, case, config: RequestContext, *, stream=None) -> RequestState:
        start_wall = perf_counter()
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(stream.device))
        context = RequestContext(config.request_id, -1, None, dict(config.base_attention_kwargs), stream)
        pipeline = private_pipeline(self.pipeline)
        state = RequestState(context, pipeline, stream=stream, capture_steps=self.capture_steps, started_at=start_wall)
        with self.stream_context(state):
            if stream is not None:
                state.start_event = torch.cuda.Event(enable_timing=True)
                state.start_event.record()
            state.continuation = pipeline.iter_request(case, request_state=state)
            try:
                if next(state.continuation) is not None:
                    raise RuntimeError("Request did not suspend after its initialization")
            except BaseException:
                self.abort(state)
                raise
        return state

    @torch.inference_mode()
    def run_step(self, state: RequestState, step: int) -> None:
        if step != state.next_step or step >= len(state.timesteps) or state.continuation is None:
            raise ValueError(f"Expected step {state.next_step}, got {step}")
        with self.stream_context(state):
            record = next(state.continuation)
            if record is not None:
                if record.step != step:
                    raise RuntimeError(f"Request yielded step {record.step}, expected {step}")
                state.steps.append(record)
        state.next_step += 1

    @torch.inference_mode()
    def finish(self, state: RequestState):
        if state.next_step != len(state.timesteps) or state.continuation is None:
            raise ValueError("Finalize requires every denoising step")
        with self.stream_context(state):
            diffusion = return_value(state.continuation)
            state.continuation = None
        return state, diffusion.output

    @torch.inference_mode()
    def complete_batch(self, futures):
        results = []
        for state, output in futures:
            with self.stream_context(state):
                processed = self.post_process(output)
                if state.stream is not None:
                    state.end_event = torch.cuda.Event(enable_timing=True)
                    state.end_event.record()
                    state.end_event.synchronize()
                clear_template(state.pipeline)
            results.append(RequestResult(
                latents=state.latents,
                output=processed,
                metadata=state.metadata,
                steps=state.steps,
                timing={
                    "cuda_event_seconds": (
                        state.start_event.elapsed_time(state.end_event) / 1000.0
                        if state.start_event is not None else None
                    ),
                    "request_seconds": perf_counter() - state.started_at,
                },
                memory=state.metadata.get("memory", {}),
            ))
        return results

    def finalize(self, state: RequestState) -> RequestResult:
        return self.complete_batch([self.finish(state)])[0]

    def abort(self, state: RequestState) -> None:
        with self.stream_context(state):
            if state.stream is not None:
                state.stream.synchronize()
            if state.continuation is not None:
                state.continuation.close()
                state.continuation = None
            clear_template(state.pipeline)
