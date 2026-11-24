from types import SimpleNamespace

import torch

from vllm_omni.diffusion.models.flux.request_context import RequestContext, context_kwargs


def test_context_kwargs_does_not_mutate_base():
    base = {"keep": 1}
    ctx = RequestContext("r0", 3, None, base)
    out = context_kwargs(base, ctx)
    assert base == {"keep": 1}
    assert out == {"keep": 1, "_flux_request_context": ctx}


def test_full_entry_uses_explicit_context_without_pipeline_attention_state(monkeypatch):
    from vllm_omni.diffusion.models.flux import acceleration_runtime, paper_selector_runtime

    class Pipeline:
        @property
        def joint_attention_kwargs(self):
            raise AssertionError("pipeline attention kwargs accessed")

        @property
        def _current_timestep(self):
            raise AssertionError("pipeline timestep accessed")

    pipeline = Pipeline()
    from vllm_omni.diffusion.models.flux.request_state import RequestState

    ctx = RequestContext("r0", -1, None, {})
    state = RequestState(ctx, pipeline)
    seen = []

    def steps(actual_pipeline, **kwargs):
        assert actual_pipeline is pipeline
        seen.append(kwargs["request_state"])
        assert kwargs["request_state"].context.base_attention_kwargs == {}
        yield None
        return torch.zeros(1, 1, 1), {"step_records": []}

    monkeypatch.setattr(paper_selector_runtime, "iter_sparseedit_paper", steps)
    monkeypatch.setattr(acceleration_runtime, "_validate_runtime", lambda *args: {
        "generated_grid": {"height": 1, "width": 1}
    })
    config = SimpleNamespace(method="sparseedit", sparseedit_selector_protocol="paper", to_metadata=lambda: {})
    tensor = torch.zeros(1, 1, 1)
    task = acceleration_runtime.diffuse_with_flux_kontext_acceleration(
        pipeline, config=config, prompt_embeds=tensor, pooled_prompt_embeds=tensor,
        negative_prompt_embeds=tensor, negative_pooled_prompt_embeds=tensor,
        latents=tensor, image_latents=tensor, latent_ids=tensor, text_ids=tensor,
        negative_text_ids=tensor, timesteps=tensor, do_true_cfg=True, guidance=tensor,
        true_cfg_scale=4.0, height=768, width=768, request_state=state,
    )
    assert next(task) is None
    try:
        next(task)
    except StopIteration as completed:
        assert torch.equal(completed.value, tensor)
    else:
        raise AssertionError("missing result")
    assert seen == [state]
    assert pipeline._last_flux_acceleration_metadata["step_records"] == []
