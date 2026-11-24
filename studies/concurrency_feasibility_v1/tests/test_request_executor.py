from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.flux.request_context import RequestContext
from vllm_omni.diffusion.models.flux.request_executor import RequestExecutor
from vllm_omni.diffusion.models.flux.request_state import StepSnapshot


class Pipeline:
    scheduler = SimpleNamespace(progress=0)
    _sparseedit_template_pool = {"full": object()}
    _sparseedit_condition_bank = {"pos": object()}

    def iter_request(self, case, *, request_state):
        assert self.scheduler is not Pipeline.scheduler
        assert self._sparseedit_template_pool is None
        assert self._sparseedit_condition_bank is None
        request_state.timesteps = range(3)
        request_state.latents = torch.full((1,), case)
        yield None
        for index in range(3):
            self.scheduler.progress += 1
            request_state.latents = request_state.latents + 1
            yield StepSnapshot(index, request_state.latents, torch.ones(1))
        return SimpleNamespace(output={"payload": {"image": request_state.latents}, "metadata": {}})


def test_executor_advances_one_step_and_owns_progress():
    pipeline = Pipeline()
    executor = RequestExecutor(pipeline, post_process=lambda output: output, capture_steps=True)
    state = executor.init_state(5, RequestContext("x", -1, None, {}))
    assert state.pipeline.scheduler.progress == 0
    with pytest.raises(ValueError, match="Expected step"):
        executor.run_step(state, 1)
    for step in range(3):
        executor.run_step(state, step)
        assert state.pipeline.scheduler.progress == step + 1
    result = executor.finalize(state)
    assert pipeline.scheduler.progress == 0
    assert torch.equal(result.latents, torch.tensor([8]))
    assert [snap.latents.item() for snap in result.steps] == [6, 7, 8]


def test_step_digest_requires_index_witness_only_on_sparse_steps(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_isolation import step_indices_match

    dense = {"is_sparse_step": False}
    sparse = {"is_sparse_step": True}
    assert step_indices_match(dense, dense)
    assert not step_indices_match(dense, sparse)
    assert not step_indices_match(sparse, sparse)
    assert step_indices_match(dict(sparse, selected_index_digest="d"), dict(sparse, selected_index_digest="d"))


def test_cuda_end_event_follows_postprocessing(monkeypatch):
    from contextlib import nullcontext
    from vllm_omni.diffusion.models.flux.request_state import RequestState

    timeline = []

    class Event:
        def record(self):
            timeline.append("end")

        def synchronize(self):
            timeline.append("synchronized")

        def elapsed_time(self, event):
            return 120

    class Stream:
        def synchronize(self):
            timeline.append("synchronized")

    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: Event())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    pipeline = Pipeline()
    pipeline._sparseedit_template_pool = None
    pipeline._sparseedit_condition_bank = None
    state = RequestState(RequestContext("x", 0, None, {}), pipeline, stream=Stream(), start_event=Event())
    state.end_event = Event()

    def steps():
        yield None
        return SimpleNamespace(output="decoded")

    state.continuation = steps()
    next(state.continuation)

    def post_process(decoded):
        assert decoded == "decoded"
        timeline.append("postprocess")
        return "image"

    result = RequestExecutor(pipeline, post_process=post_process).finalize(state)
    assert result.timing["cuda_event_seconds"] == .12
    assert timeline.index("postprocess") < timeline.index("end") < timeline.index("synchronized")
