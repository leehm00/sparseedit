# SPDX-License-Identifier: Apache-2.0
"""Single-GPU native Qwen execution, stage timing and L3 vision/VAE reuse."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
from importlib import metadata
import inspect
from io import BytesIO
import time

from .contracts import PROTOCOL, digest_json, require, sha_file, timing_summary, validate_graph

DEVICE = "cuda:0"


def resident_placement(pipeline, dtype):
    require(not getattr(pipeline, "_all_hooks", None) and not getattr(pipeline, "hf_device_map", None),
            "pipeline has offload hooks or a dispatch device map")
    placement = {}
    for name in ("transformer", "text_encoder", "vae"):
        model = getattr(pipeline, name)
        require(not getattr(model, "hf_device_map", None), f"{name} has an offload device map")
        for path, mod in model.named_modules():
            require(not getattr(mod, "_hf_hook", None), f"{name}.{path} has an offload hook")
        parameter_bytes = 0
        for path, tensor in model.named_parameters():
            require(str(tensor.device) == DEVICE, f"{name}.{path} is not resident on {DEVICE}")
            require(tensor.dtype == dtype, f"{name}.{path} has unexpected parameter dtype {tensor.dtype}")
            parameter_bytes += tensor.numel() * tensor.element_size()
        require(parameter_bytes > 0, f"{name} has no resident parameters")
        for path, tensor in model.named_buffers():
            require(str(tensor.device) == DEVICE, f"{name}.{path} buffer is not resident on {DEVICE}")
        placement[name] = {"device": DEVICE, "dtype": str(dtype), "parameter_bytes": parameter_bytes}
    return placement


def select_gpu(torch, expected_uuid):
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "exactly one visible CUDA device is required")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    actual_uuid = str(props.uuid)
    if not actual_uuid.startswith("GPU-"):
        actual_uuid = f"GPU-{actual_uuid}"
    require(expected_uuid.lower() == actual_uuid.lower(), "visible CUDA device UUID differs from configuration")
    family = props.name.upper().split("-")
    require(props.major == 9 and any("H20" == word or "H200" == word for part in family for word in part.split()),
            f"Track H1 requires Hopper H20 or H200, found {props.name}")
    return {"uuid": actual_uuid, "name": props.name, "memory_total_bytes": props.total_memory,
            "capability": [props.major, props.minor], "torch": torch.__version__, "torch_cuda": torch.version.cuda}


def load_pipeline(torch, directory):
    # These releases contain the native call sites used by the frozen SparseEdit adapter.
    required = {"diffusers": "0.38.0", "transformers": "5.5.3"}
    versions = {package: metadata.version(package) for package in required}
    require(versions == required, f"Qwen native ABI requires {required}, found {versions}")

    from diffusers import QwenImageEditPipeline

    start = time.perf_counter()
    pipeline = QwenImageEditPipeline.from_pretrained(
        directory, torch_dtype=torch.bfloat16, local_files_only=True, use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    pipeline.to(device=DEVICE, dtype=torch.bfloat16)
    pipeline.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    placement = resident_placement(pipeline, torch.bfloat16)
    classes = (type(pipeline), type(pipeline.transformer), type(pipeline.text_encoder), type(pipeline.vae))
    paths = {cls.__module__: inspect.getfile(cls) for cls in classes}
    return pipeline, {
        "parameter_placement": placement,
        "load_wall_seconds": time.perf_counter() - start,
        "libraries": versions,
        "installed_model_code_sha256": {module: sha_file(path) for module, path in paths.items()},
    }


@contextmanager
def patch(obj, name, build_wrapper):
    original = getattr(obj, name)
    setattr(obj, name, build_wrapper(original))
    try:
        yield
    finally:
        setattr(obj, name, original)


class DeviceLedger:
    """Bracket native work on the request stream without synchronizing each segment."""

    def __init__(self, torch):
        self.cuda = torch.cuda
        self.segments = []
        self.step = 0
        self.branches = 0
        self.open_step = None
        self.origin = self.event()
        self.denoise_started = None
        self.denoise_wall_seconds = None

    def event(self):
        event = self.cuda.Event(enable_timing=True)
        event.record()
        return event

    def start(self, segment, **identity):
        return (segment, identity, self.event(), time.perf_counter_ns(), self.cuda.memory_allocated())

    def end(self, started):
        name, identity, cuda_start, wall_start, allocated_start = started
        cuda_end = self.event()
        self.segments.append((cuda_start, cuda_end, {
            "segment": name, **identity,
            "host_enqueue_seconds": (time.perf_counter_ns() - wall_start) * 1e-9,
            "allocated_bytes_at_start": allocated_start,
            "allocated_bytes_at_end": self.cuda.memory_allocated(),
            "request_peak_allocated_bytes_so_far": self.cuda.max_memory_allocated(),
            "request_peak_reserved_bytes_so_far": self.cuda.max_memory_reserved(),
        }))

    @contextmanager
    def span(self, segment, **identity):
        start = self.start(segment, **identity)
        yield
        self.end(start)

    def transformer(self, call, *, action):
        hidden, text = call["hidden_states"], call["encoder_hidden_states"]
        require(hidden.shape[0] == text.shape[0] == 1, "Qwen transformer requires a single request")
        require(str(hidden.device) == str(text.device) == DEVICE, "transformer inputs must remain on CUDA")
        if self.branches == 0:
            if self.step == 0:
                self.cuda.synchronize()
                self.denoise_started = time.perf_counter()
            self.open_step = self.start("denoise_step", step=self.step, action=action)
        self.branches += 1
        require(self.branches <= 2, "each Qwen step must execute exactly two CFG branches")

    def commit(self):
        require(self.branches == 2 and self.open_step is not None, "missing Qwen branches before Euler commit")
        self.end(self.open_step)
        self.open_step = None
        self.branches = 0
        self.step += 1
        if self.step == PROTOCOL["steps"]:
            self.cuda.synchronize()
            self.denoise_wall_seconds = time.perf_counter() - self.denoise_started

    def read(self):
        require(self.step == 28 and self.branches == 0 and self.open_step is None,
                "incomplete Qwen denoising ledger")
        rows = [
            {**row, "cuda_start_seconds": self.origin.elapsed_time(begin) * 1e-3,
             "cuda_seconds": begin.elapsed_time(end) * 1e-3}
            for begin, end, row in self.segments
        ]
        return sorted(rows, key=lambda row: row["cuda_start_seconds"])


def tensor_layout(tensor):
    return (tuple(tensor.shape), str(tensor.dtype), str(tensor.device))


def tensor_sha(torch, tensor):
    value = tensor.detach().contiguous()
    if value.ndim == 0:
        value = value.reshape(1)
    digest = hashlib.sha256(repr((tuple(tensor.shape), str(tensor.dtype))).encode())
    digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def request_kwargs(torch, case, image, prompt):
    return {
        "image": image, "prompt": prompt if prompt is not None else case["prompt"],
        "negative_prompt": PROTOCOL["negative_prompt"], "height": PROTOCOL["height"], "width": PROTOCOL["width"],
        "num_inference_steps": PROTOCOL["steps"], "num_images_per_prompt": 1,
        "guidance_scale": PROTOCOL["guidance_scale"], "true_cfg_scale": PROTOCOL["true_cfg_scale"],
        "generator": torch.Generator(device=DEVICE).manual_seed(PROTOCOL["seed"]),
    }


def native_request(torch, pipeline, case, *, mode, budget=None, graph=False, trace=None,
                   cache=None, session=False, prompt=None, admission=False, identity=None, source_hashes=None):
    from PIL import Image

    raw = case["input_path"].read_bytes()
    require(hashlib.sha256(raw).hexdigest() == case["input_sha256"], "Qwen request input bytes changed")
    with Image.open(BytesIO(raw)) as opened:
        image = opened.convert("RGB")
    kwargs = request_kwargs(torch, case, image, prompt)
    observations = {"vision": [], "vae": []}
    conditioning = {}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ledger = DeviceLedger(torch)

    def prompt_wrap(original):
        def encode(*args, **call):
            branch = "positive" if "positive_prompt" not in conditioning else "negative"
            with ledger.span("vl_encode", branch=branch):
                value = original(*args, **call)
            conditioning[branch + "_prompt"] = value
            return value
        return encode

    def vision_wrap(original):
        def encode(pixel_values, image_grid_thw, **call):
            layout = (tensor_layout(pixel_values), tensor_layout(image_grid_thw))
            with ledger.span("vision_tower"):
                if cache is None:
                    value, hit = original(pixel_values, image_grid_thw, **call), False
                else:
                    value, hit = cache.get(
                        "vision", layout, lambda: original(pixel_values, image_grid_thw, **call),
                    )
            require(all(str(feature.device) == DEVICE for feature in value.pooler_output),
                    "L3 vision features left the request CUDA device")
            observations["vision"].append(hit)
            return value
        return encode

    def vae_wrap(original):
        def encode(image, generator):
            require(str(image.device) == DEVICE, "VAE reference input left the request CUDA device")
            with ledger.span("reference_vae_encode"):
                if cache is None:
                    value, hit = original(image, generator), False
                else:
                    value, hit = cache.get("vae", tensor_layout(image), lambda: original(image, generator))
            require(str(value.device) == DEVICE, "VAE latents left the request CUDA device")
            observations["vae"].append(hit)
            conditioning["reference_latents"] = (value,)
            return value
        return encode

    def decode_wrap(original):
        def decode(*args, **call):
            with ledger.span("vae_decode"):
                return original(*args, **call)
        return decode

    def forward_wrap(original):
        def forward(**call):
            ledger.transformer(call, action="full")
            return original(**call)
        return forward

    def step_wrap(original):
        def step(*args, **call):
            value = original(*args, **call)
            ledger.commit()
            return value
        return step

    if mode != "dense":
        from .frozen.qwen_image.sparseedit_native_pipeline import QwenSparseDenoiser
        from .frozen.qwen_image.sparseedit_native_runtime import QwenSparseEditConfig

        class ObservedDenoiser(QwenSparseDenoiser):
            def _transformer_call(self, *args, **call):
                step = self.step + (not self.branch_order)
                action = "full" if step in self.config.dense_steps else "sparse"
                ledger.transformer(call, action=action)
                return super()._transformer_call(*args, **call)

            def _scheduler_step(self, *args, **call):
                value = super()._scheduler_step(*args, **call)
                ledger.commit()
                return value

        runner = ObservedDenoiser(pipeline, QwenSparseEditConfig(ratio=budget / 100), mode=mode, graph=graph)

    executed_trace = None
    states = None
    with ExitStack() as stack:
        stack.enter_context(torch.no_grad())
        stack.enter_context(patch(pipeline, "encode_prompt", prompt_wrap))
        stack.enter_context(patch(pipeline, "_encode_vae_image", vae_wrap))
        stack.enter_context(patch(pipeline.vae, "decode", decode_wrap))
        if session:
            stack.enter_context(patch(pipeline.text_encoder.model, "get_image_features", vision_wrap))
        if mode == "dense":
            stack.enter_context(patch(pipeline.transformer, "forward", forward_wrap))
            stack.enter_context(patch(pipeline.scheduler, "step", step_wrap))
        started = time.perf_counter()
        if mode == "dense":
            images = pipeline(**kwargs).images
            runtime = {}
            torch.cuda.synchronize()
            request_wall_seconds = time.perf_counter() - started
        else:
            result = runner.run(**kwargs, fixed_trace=trace, admission_tensors=admission,
                                identity=identity, source_hashes=source_hashes)
            # The frozen wrapper materializes trace indices and optional admission tensors
            # after the native Pipeline call. Time actual completion at that call boundary.
            request_wall_seconds = result.metadata["native_request_wall_seconds"]
            images = result.output.images
            runtime = result.metadata
            executed_trace = result.trace
            states = runtime.pop("admission_tensors", None)
            validate_graph(runtime, requested=graph)
            require(not any(runtime["sparse_auxiliary_transfers"].values()) and not runtime["native_block_transfers"],
                    "unexpected sparse staging transfers")
            for branch_state in runner.branches.values():
                for item in branch_state.cache.layers.values():
                    require(str(item.key.device) == str(item.value.device) == DEVICE, "KV left selected CUDA device")
            require(runtime["true_cfg_calls"] == 56, "Qwen sparse CFG branch execution count differs")
            if trace is not None:
                require(digest_json(executed_trace.payload()["steps"]) == digest_json(trace.payload()["steps"]),
                        "executed masks differ from the fixed trajectory")

    segments = ledger.read()
    require(len(images) == 1 and images[0].size == (1024, 1024), "Qwen did not return one 1024px output")
    record = {
        "request_wall_seconds": request_wall_seconds,
        "timing_scope": "synchronized native preprocessing, conditioning, denoising, decode and image postprocess",
        "denoising_wall_seconds": ledger.denoise_wall_seconds,
        "segments": segments,
        **timing_summary(segments, dense=mode == "dense"),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "runtime": runtime,
        "cache_events": observations,
        "conditioning": {
            name: [None if t is None else tensor_sha(torch, t) for t in values]
            for name, values in conditioning.items()
        } if session else {},
    }
    return images[0], record, executed_trace, states


def admission_comparison(torch, left, right):
    require(len(left) == len(right) == 28, "numerical admission requires 28 matching steps")
    fields = ("positive", "negative", "velocity", "latents", "latents_after")
    checks = []
    for step, (expected, actual) in enumerate(zip(left, right, strict=True)):
        require(expected["step"] == actual["step"] == step and expected["action"] == actual["action"],
                "numerical admission schedule differs")
        for field in fields:
            a, b = expected[field], actual[field]
            require(a.shape == b.shape and a.dtype == b.dtype, f"numerical admission shape or dtype differs at {step}")
            if a.dtype in (torch.bfloat16, torch.float16):
                atol, rtol = 0.02, 0.02
            else:
                atol, rtol = 2e-5, 2e-4
            a, b = a.float(), b.float()
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            checks.append({"step": step, "field": field, "passed": finite and torch.allclose(a, b, atol=atol, rtol=rtol),
                           "max_abs": float((a - b).abs().max()) if finite else None, "atol": atol, "rtol": rtol})
    return {"passed": all(row["passed"] for row in checks), "checks": checks}
