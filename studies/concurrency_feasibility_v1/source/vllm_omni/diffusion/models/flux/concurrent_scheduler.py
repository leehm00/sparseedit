# SPDX-License-Identifier: Apache-2.0
"""Step-interleaved scheduling of independent FLUX requests."""

from .request_executor import RequestExecutor
from .request_state import RequestResult, RequestState


class ConcurrentScheduler:
    def __init__(self, executor: RequestExecutor) -> None:
        self.executor = executor
        self.enqueue_order: list[dict] = []

    def run(self, states: list[RequestState]) -> list[RequestResult]:
        if not states:
            return []
        steps = len(states[0].timesteps)
        if any(len(state.timesteps) != steps for state in states):
            raise ValueError("Interleaved requests must have equal step counts")
        stream_ids = [state.stream.cuda_stream for state in states if state.stream is not None]
        if stream_ids and len(set(stream_ids)) != len(states):
            raise ValueError("Every CUDA request requires a distinct stream")
        self.enqueue_order.clear()
        try:
            for step in range(steps):
                for state in states:
                    self.executor.run_step(state, step)
                    self.enqueue_order.append({
                        "request_id": state.context.request_id,
                        "step": step,
                        "stream_id": state.stream.cuda_stream if state.stream is not None else None,
                        "enqueue_order": len(self.enqueue_order),
                    })
            return self.executor.complete_batch([self.executor.finish(state) for state in states])
        except BaseException as error:
            for state in states:
                try:
                    self.executor.abort(state)
                except Exception as cleanup_error:
                    error.add_note(f"Request cleanup failed: {cleanup_error}")
            raise
