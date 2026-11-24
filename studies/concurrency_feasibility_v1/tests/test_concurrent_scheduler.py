from contextlib import nullcontext
from types import SimpleNamespace

from vllm_omni.diffusion.models.flux.concurrent_scheduler import ConcurrentScheduler


def test_interleaving_preserves_per_request_order():
    enqueue = []
    states = [
        SimpleNamespace(context=SimpleNamespace(request_id=request_id), timesteps=range(28), stream=None)
        for request_id in ("A", "B")
    ]

    class Executor:
        def run_step(self, state, step):
            enqueue.append((state.context.request_id, step))

        def finish(self, state):
            return state.context.request_id

        def complete_batch(self, futures):
            return futures

        def abort(self, state):
            raise AssertionError("unexpected abort")

    assert ConcurrentScheduler(Executor()).run(states) == ["A", "B"]
    assert enqueue == [(request_id, step) for step in range(28) for request_id in ("A", "B")]
