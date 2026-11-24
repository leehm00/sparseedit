from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.flux.request_executor import private_pipeline
from vllm_omni.diffusion.models.flux.specialized_template import acquire_template, clear_template


def test_concurrent_pipelines_own_different_template_leases():
    pipeline = SimpleNamespace(scheduler=object(), transformer=object())
    serial, _ = acquire_template(pipeline, (1,))
    first = private_pipeline(pipeline)
    second = private_pipeline(pipeline)
    a, _ = acquire_template(first, (1,))
    b, _ = acquire_template(second, (1,))
    assert len({id(a), id(b), id(serial)}) == 3
    assert first.transformer is second.transformer is pipeline.transformer
    assert len({id(first.scheduler), id(second.scheduler), id(pipeline.scheduler)}) == 3
    inputs = [torch.ones(1), torch.zeros(1)]
    bound = [
        template.bind_conditioning("positive", {"prompt": tensor})["prompt"]
        for template, tensor in zip((a, b), inputs, strict=True)
    ]
    assert not torch.equal(*bound)
    assert bound[0].untyped_storage().data_ptr() != bound[1].untyped_storage().data_ptr()
    with pytest.raises(RuntimeError, match="in use"):
        acquire_template(first, (2,))
    with pytest.raises(RuntimeError, match="in use"):
        clear_template(first)
    for owner, template in ((pipeline, serial), (first, a), (second, b)):
        template.finish(success=True)
        clear_template(owner)
        assert template.closed
