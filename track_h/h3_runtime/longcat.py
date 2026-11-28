# SPDX-License-Identifier: Apache-2.0
"""Native LongCat dense controls from the frozen A5 1024px protocol."""

from contextlib import contextmanager
import gc
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter

from .contracts import artifact, file_sha, read

COHORT_SHA256 = "d69ed0abb3a2216e5c17373932529ad56db8a9a3210db11c354c718db2dc9207"


def frozen_cases(assets, mode):
    cohort = read(artifact(assets, "cohort.json", COHORT_SHA256))
    cases = cohort["cases"][mode]
    return cohort, [{**case, "input_path": artifact(assets, case["input"], case["input_sha256"])} for case in cases]


def verify_model(model, cohort):
    if not model.is_absolute() or not model.is_dir():
        raise ValueError("models.longcat must point to the frozen local checkpoint")
    identity = cohort["model"]
    for name, digest in identity["files"].items():
        if file_sha(model / name) != digest:
            raise ValueError(f"LongCat checkpoint SHA-256 differs for {name}")
    if read(model / "model_index.json")["_class_name"] != identity["family"]:
        raise ValueError("LongCat model pipeline class differs")
    return identity


def verify_environment(cohort):
    expected = cohort["environment"]
    package = Path(find_spec("diffusers").origin).parent
    actual = {
        "versions": {name: version(name).split("+", 1)[0] for name in expected["versions"]},
        "source_hashes": {name: file_sha(package / name) for name in expected["source_hashes"]},
    }
    if actual != expected:
        raise ValueError(f"native LongCat environment differs from A5: {actual}")
    return actual


@contextmanager
def replace_attr(obj, name, replacement):
    original = getattr(obj, name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        setattr(obj, name, original)


def layout(tensor):
    return tensor.shape, tensor.stride(), tensor.dtype, tensor.device


class DenseForwardGraph:
    """A5 request-scoped capture of the stock transformer forward."""

    dynamic = ("hidden_states", "timestep", "img_ids", "txt_ids", "encoder_hidden_states")

    def __init__(self, torch, forward):
        self.torch = torch
        self.forward = forward
        self.graphs = {}
        self.replays = 0

    def capture(self, values):
        torch = self.torch
        inputs = {**values, **{key: values[key].clone(memory_format=torch.preserve_format) for key in self.dynamic}}
        stream = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            self.forward(**inputs)
            self.forward(**inputs)
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.forward(**inputs)[0]
        return inputs, graph, output

    def __call__(self, **kwargs):
        kwargs.setdefault("guidance", None)
        sample = kwargs["hidden_states"]
        if kwargs["guidance"] is not None or kwargs["return_dict"] is not False:
            raise ValueError("native LongCat dense forward changed")
        if sample.shape[0] != 1 or sample.shape[1] % 2:
            raise ValueError("native LongCat dense graph requires one edit with equal token grids")
        signature = tuple(layout(kwargs[name]) for name in self.dynamic)
        if signature not in self.graphs:
            self.graphs[signature] = self.capture(kwargs)
        inputs, graph, output = self.graphs[signature]
        if inputs.keys() != kwargs.keys():
            raise ValueError("native LongCat dense graph argument schema changed")
        for name, bound in inputs.items():
            current = kwargs[name]
            if name in self.dynamic:
                if layout(bound) != layout(current):
                    raise ValueError(f"native LongCat dense graph layout changed for {name}")
                bound.copy_(current)
            elif isinstance(bound, self.torch.Tensor) and bound is not current or (
                not isinstance(bound, self.torch.Tensor) and bound != current
            ):
                raise ValueError(f"native LongCat dense graph constant changed for {name}")
        graph.replay()
        self.replays += 1
        return (output[:, :sample.shape[1] // 2].clone(),)


@contextmanager
def open_model(torch, model, cohort):
    from diffusers import LongCatImageEditPipeline

    torch.cuda.synchronize()
    start = perf_counter()
    pipeline = LongCatImageEditPipeline.from_pretrained(
        model, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    placement = {
        name: sorted({str(param.device) for param in component.parameters()})
        for name, component in pipeline.components.items() if isinstance(component, torch.nn.Module)
    }
    if any(devices not in ([], ["cuda:0"]) for devices in placement.values()):
        raise ValueError("LongCat parameter placement is not fully resident")
    if getattr(pipeline.transformer, "_cache_config", None) is not None:
        raise ValueError("native LongCat requires the uncached pipeline")
    try:
        yield pipeline, {"model_load_seconds": perf_counter() - start,
                         "placement": placement, "protocol": cohort["protocol"]}
    finally:
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


def request(torch, pipeline, case, arm):
    from contextlib import ExitStack
    from PIL import Image

    actions = []
    scheduler_step = pipeline.scheduler.step

    def observed_scheduler(*args, **kwargs):
        value = scheduler_step(*args, **kwargs)
        actions.append({"step": len(actions), "action": "full"})
        return value

    graph = DenseForwardGraph(torch, pipeline.transformer.forward) if arm == "dense_graph" else None
    with Image.open(case["input_path"]) as raw, raw.convert("RGB") as image, ExitStack() as stack:
        if image.size != (1024, 1024):
            raise ValueError("LongCat requires a frozen 1024px input")
        stack.enter_context(torch.inference_mode())
        stack.enter_context(replace_attr(pipeline.scheduler, "step", observed_scheduler))
        if graph is not None:
            stack.enter_context(replace_attr(pipeline.transformer, "forward", graph))
        torch.cuda.synchronize()
        started = perf_counter()
        result = pipeline(
            image=image, prompt=case["instruction"], negative_prompt="", num_inference_steps=28,
            guidance_scale=3.5, num_images_per_prompt=1,
            generator=torch.Generator(device="cuda").manual_seed(110),
        )
        torch.cuda.synchronize()
        wall = perf_counter() - started
    if len(actions) != 28 or graph is not None and graph.replays != 56:
        raise ValueError("native LongCat scheduler step count or dense graph replay count differs")
    metadata = {"timing_scope": "synchronized native LongCat inference through PIL output"}
    if graph is not None:
        metadata.update(dense_graph_templates=len(graph.graphs), dense_graph_replays=graph.replays)
    return result.images[0], wall, actions, metadata
