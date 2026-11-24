#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect synchronized F2 timing on the frozen FLUX Kontext calibration cases."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

STUDY = Path(__file__).resolve().parents[1]
V4 = STUDY.parents[1] / "backend_reference" / "current_v4"
BUNDLE_ROOT = V4 / "backend_bundle"
MODEL = STUDY.parents[2] / "flux_a40_kontext_28step_20260912/models/FLUX.1-Kontext-dev"
PYTHON = MODEL.parents[1] / "envs/flux/bin/python"
MODEL_CLASS = "FluxKontextPipeline"
NEGATIVE_PROMPT = " "
ARMS = ("full", "direct_sparse", "standard_sparse", "dense")
MODES = ("serial", "concurrent", "true_batch")
TIMING_BOUNDARY = (
    "Synchronized model worker wall including normal request encoding, denoising, VAE decode, "
    "and runtime image postprocessing. Model load, input resize, independent warmup, "
    "PNG serialization and scoring recorded separately."
)
FROZEN_PROTOCOL = {
    "model": "FLUX.1-Kontext-dev",
    "height": 768,
    "width": 768,
    "dtype": "bfloat16",
    "steps": 28,
    "warmup": 8,
    "refresh": 23,
    "refresh_interval": 15,
    "true_cfg_scale": 4.0,
    "guidance_scale": 1.0,
    "N": 2304,
    "reference_tokens": 2304,
    "K": 460,
    "sparse_ratio": 0.2,
    "seed": 110,
    "batch_size": 1,
    "velocity_reuse": False,
    "routing": False,
}
BoundCase = tuple[dict[str, Any], Path, Path, dict[str, Any]]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(4 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def load_protocol_cases() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from run_baseline import load_cases

    protocol = read_json(STUDY / "protocol.json")
    if any(protocol.get(field) != value for field, value in FROZEN_PROTOCOL.items()):
        raise ValueError("Fixed F2 protocol differs")
    bundle = read_json(BUNDLE_ROOT / "bundle.json")
    identities = {
        case["case_tag"]: {
            "case_tag": case["case_tag"],
            "input_sha256": case["prepared_sha256"],
            "trace_sha256": case["trace_sha256"],
            "source_pixel_sha256": case["source_pixel_sha256"],
            "instruction": case["instruction"],
        }
        for case, _, _, _ in load_cases(BUNDLE_ROOT, bundle)
    }
    if len(identities) != 3:
        raise ValueError("Three distinct frozen F2 cases required")
    return protocol, identities


def resize_case(bound_case: BoundCase, protocol: dict[str, Any]) -> Any:
    """Match the frozen native worker's separately timed preparation."""
    from PIL import Image

    _, image_path, _, _ = bound_case
    with Image.open(image_path) as source:
        return source.convert("RGB").resize((protocol["width"], protocol["height"]))


def make_request(
    bound_case: BoundCase,
    arm: str,
    protocol: dict[str, Any],
    *,
    image: Any = None,
    lane: str = "private_graph",
) -> Any:
    """Use the native prompt builder, option decoder and per-request seed."""
    import torch
    from run_baseline import options
    from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.model_extras import build_image_to_image_prompt, get_extra_body_params

    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    case, _, trace_path, _ = bound_case
    if arm == "dense":
        extras = {"flux_acceleration_method": "dense", "sparseedit_return_metadata": True}
    else:
        candidate = {
            "full": "full",
            "direct_sparse": "direct_eager",
            "standard_sparse": "standard_fastops_eager",
        }[arm]
        extras = options(candidate, trace_path, audit=False)
        if arm == "full" and lane == "private_eager":
            extras["sparseedit_runtime_variant"] = "transformed"
    extras["negative_prompt"] = NEGATIVE_PROMPT
    sampling = OmniDiffusionSamplingParams(
        generator=torch.Generator(device="cuda").manual_seed(protocol["seed"]),
        true_cfg_scale=protocol["true_cfg_scale"],
        guidance_scale=protocol["guidance_scale"],
        num_inference_steps=protocol["steps"],
        num_outputs_per_prompt=1,
        resolution=max(protocol["height"], protocol["width"]),
        height=protocol["height"],
        width=protocol["width"],
    )
    declared = get_extra_body_params(MODEL_CLASS)
    if declared:
        apply_declared_extra_args(sampling, declared, extras)
    else:
        sampling.extra_args.update(extras)
    prompt = build_image_to_image_prompt(
        model_class_name=MODEL_CLASS,
        prompt=case["instruction"],
        negative_prompt=NEGATIVE_PROMPT,
        input_image=resize_case(bound_case, protocol) if image is None else image,
        height=protocol["height"],
        width=protocol["width"],
    )
    return DiffusionRequestBatch(
        [OmniDiffusionRequest(prompt=prompt, sampling_params=sampling, request_id=case["case_tag"])]
    )


@contextmanager
def load_pipeline(model: Path) -> Iterator[tuple[Any, Any]]:
    """Load the same FLUX pipeline via the native worker in this process."""
    import torch
    from vllm.config import set_current_vllm_config
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.registry import get_diffusion_post_process_func

    config = OmniDiffusionConfig.from_kwargs(
        model=str(model),
        model_class_name=MODEL_CLASS,
        dtype=torch.bfloat16,
        vae_use_tiling=True,
        cache_backend=None,
        enable_cpu_offload=False,
        enforce_eager=True,
        parallel_config=DiffusionParallelConfig(
            ulysses_degree=1, ring_degree=1, cfg_parallel_size=1, tensor_parallel_size=1
        ),
    )
    config.enrich_config()
    if config.model_class_name != MODEL_CLASS:
        raise ValueError(f"Unexpected FLUX pipeline class {config.model_class_name}")
    worker = DiffusionWorker(local_rank=0, rank=0, od_config=config, skip_load_model=True)
    try:
        worker.load_model(load_format=config.diffusion_load_format)
        post_process = get_diffusion_post_process_func(config)
        if post_process is None:
            raise RuntimeError("Native FLUX postprocessor not registered")
        with (
            set_forward_context(vllm_config=worker.vllm_config, omni_diffusion_config=config),
            set_current_vllm_config(worker.vllm_config),
        ):
            yield worker.model_runner.pipeline, post_process
    finally:
        worker.shutdown()


def calibration_pairs(tags: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    if len(tags) != 3 or len(set(tags)) != 3:
        raise ValueError("F2 requires three unique calibration case tags")
    a, b, c = tags
    return (a, b), (b, c), (c, a)


@dataclass(frozen=True)
class Measurement:
    arm: str
    mode: str
    pair: int
    repetition: int
    cases: tuple[str, ...]

    @property
    def directory(self) -> str:
        suffix = f"_pair{self.pair}" if self.pair else ""
        return f"{self.arm}_{self.mode}_n{len(self.cases)}_r{self.repetition}{suffix}"


def measurement_schedule(
    args: argparse.Namespace,
    tags: tuple[str, ...],
) -> Iterator[Measurement]:
    pairs = calibration_pairs(tags)
    for repetition in range(args.start_repetition, args.start_repetition + args.repetitions):
        if args.matrix:
            order = ("serial", "concurrent") if repetition % 2 == 0 else ("concurrent", "serial")
            for pair, tag in enumerate(tags):
                for mode in order:
                    yield Measurement("full", mode, pair, repetition, (tag,))
            for pair, case_tags in enumerate(pairs):
                for mode in order:
                    yield Measurement("full", mode, pair, repetition, case_tags)
                yield Measurement("standard_sparse", "concurrent", pair, repetition, case_tags)
                yield Measurement("dense", "true_batch", pair, repetition, case_tags)
        else:
            case_sets = pairs if args.concurrency == 2 else tuple((tag,) for tag in tags)
            for pair, case_tags in enumerate(case_sets):
                yield Measurement(args.arm, args.mode, pair, repetition, case_tags)


def _positive(number: Any) -> bool:
    return type(number) in (int, float) and math.isfinite(number) and number > 0


def validate_measurement(
    data: dict[str, Any],
    protocol: dict[str, Any],
    identities: dict[str, dict[str, Any]],
) -> None:
    if data.get("status") != "PASS" or data.get("protocol") != protocol:
        raise ValueError("F2 measurement has an unexpected status or protocol")
    if data.get("arm") not in ARMS or data.get("mode") not in MODES:
        raise ValueError("Unknown F2 arm or mode")
    count = data.get("concurrency")
    if count not in (1, 2) or type(count) is not int:
        raise ValueError("Unexpected F2 concurrency")
    cases = data.get("cases", [])
    if len(cases) != count or len({case.get("case_tag") for case in cases}) != count:
        raise ValueError("F2 case count or identity differs")
    if not _hash(data.get("source_sha")) or not data.get("device_uuid"):
        raise ValueError("F2 runtime or device identity missing")
    environment = data.get("environment", {})
    if any(not environment.get(field) for field in ("python", "torch", "triton", "vllm")):
        raise ValueError("F2 runtime environment incomplete")
    if not data.get("timing_boundary") or not data.get("runtime_lane"):
        raise ValueError("F2 timing boundary or runtime lane missing")
    for field in ("pair", "repetition", "measurement_order"):
        if type(data.get(field)) is not int or data[field] < 0:
            raise ValueError(f"F2 {field} missing or invalid")

    wall = data.get("batch_wall_seconds")
    if data.get("synchronized_before_return") is not True or not _positive(wall):
        raise ValueError("Synchronized batch wall timing required")
    if data.get("correctness", {}).get("all_match") is not True:
        raise ValueError("F2 output correctness failed")
    for case in cases:
        identity = identities.get(case["case_tag"])
        if identity is None or any(case.get(field) != value for field, value in identity.items()):
            raise ValueError(f"F2 case identity differs for {case['case_tag']}")
        if not _positive(case.get("cuda_event_seconds")) or not _positive(case.get("request_seconds")):
            raise ValueError("Per-request synchronized wall and CUDA event timing required")
        if case["request_seconds"] > wall + 0.001 or not _hash(case.get("output_png_sha256")):
            raise ValueError("F2 request wall or PNG hash invalid")
    if data["mode"] == "serial":
        summed = sum(case["request_seconds"] for case in cases)
        if not math.isclose(summed, wall, rel_tol=0, abs_tol=0.001):
            raise ValueError("Serial request_seconds sum differs from synchronized batch wall")
    if data["mode"] == "true_batch":
        if (
            data["arm"] != "dense"
            or count != 2
            or data.get("transformer_batch_sizes") != [count] * (2 * protocol["steps"])
        ):
            raise ValueError("True-batch arm lacks FLUX transformer batch-dimension witness")
    streams = data.get("streams", [])
    stream_count = count if data["mode"] == "concurrent" else 1
    if len(streams) != stream_count or len(set(streams)) != stream_count:
        raise ValueError("F2 stream IDs are missing or aliased")
    allocated = data.get("peak_allocated_bytes")
    reserved = data.get("peak_reserved_bytes")
    if not _positive(allocated) or not _positive(reserved) or reserved < allocated:
        raise ValueError("F2 peak memory evidence incomplete")
    throughput = data.get("throughput_requests_per_second")
    if not _positive(throughput) or not math.isclose(throughput, count / wall, rel_tol=1e-9):
        raise ValueError("F2 throughput must use synchronized batch wall")


def source_tree_sha() -> str:
    root = STUDY / "source"
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix != ".pyc" and not {"__pycache__", ".pytest_cache"}.intersection(path.parts)
    }
    return sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_isolation_gate(
    root: Path,
    pairs: Sequence[tuple[str, str]],
    protocol: dict[str, Any],
    identities: dict[str, dict[str, Any]],
    source_sha: str,
    device_uuid: str,
    *,
    lane: str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Bind F2 to passing F1 checks on the specific device, source and cases."""
    lanes = (lane,) if lane is not None else ("private_graph", "private_eager")
    receipts = {runtime: {} for runtime in lanes}
    paths = set(root.glob("*/completion.json")) | set(root.rglob("completion.json"))
    for tags in pairs:
        pair_name = "__".join(tags)
        options = sorted(path for path in paths if path.parent.name.startswith(pair_name))
        for path in options:
            row = read_json(path)
            runtime = row.get("runtime_lane")
            if (
                row.get("status") != "PASS"
                or row.get("protocol") != protocol
                or row.get("source_sha") != source_sha
                or row.get("device_uuid") != device_uuid
                or row.get("correctness", {}).get("all_match") is not True
                or runtime not in receipts
            ):
                continue
            cases = row.get("cases", [])
            if tuple(case.get("case_tag") for case in cases) != tags:
                continue
            if not all(
                _hash(case.get("output_png_sha256"))
                and all(case.get(key) == identities[tag][key] for key in ("input_sha256", "trace_sha256"))
                for case, tag in zip(cases, tags, strict=True)
            ):
                continue
            receipts[runtime].setdefault(
                tags, {**row, "receipt": str(path.resolve()), "receipt_sha256": sha256_file(path)}
            )
    for gate in receipts.values():
        if set(gate) == set(pairs):
            return gate
    missing = {runtime: [list(tags) for tags in pairs if tags not in gate] for runtime, gate in receipts.items()}
    raise ValueError(f"F1 matching PASS evidence missing on {device_uuid}: {missing}")


def _unwrap_images(value: Any, count: int) -> list[Any]:
    if isinstance(value, dict):
        value = value["payload"]["image"]
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("Native postprocessing did not return one image per planned case")
    if any(getattr(image, "mode", None) != "RGB" for image in value):
        raise ValueError("FLUX postprocessing must return a list of RGB images")
    return value


def _serial_sparse(
    executor: Any,
    requests: list[Any],
    protocol: dict[str, Any],
    started: float,
) -> tuple[list[Any], list[int], float]:
    import time

    import torch
    from vllm_omni.diffusion.models.flux.request_context import RequestContext

    stream = torch.cuda.current_stream()
    results = []
    for req in requests:
        device_start = torch.cuda.Event(enable_timing=True)
        device_end = torch.cuda.Event(enable_timing=True)
        device_start.record(stream)
        ctx = RequestContext(req.request_id, 0, None, {})
        state = executor.init_state(req, ctx)
        try:
            for step in range(protocol["steps"]):
                executor.run_step(state, step)
            result = executor.finalize(state)
        except BaseException:
            executor.abort(state)
            raise
        del state
        device_end.record(stream)
        torch.cuda.synchronize()
        stopped = time.perf_counter()
        result.timing["request_seconds"] = stopped - started
        result.timing["cuda_event_seconds"] = device_start.elapsed_time(device_end) * 1e-3
        started = stopped
        results.append(result)
    return results, [int(stream.cuda_stream)], stopped


def _parallel_sparse(
    executor: Any,
    requests: list[Any],
    started: float,
) -> tuple[list[Any], list[int], list[dict[str, Any]]]:
    import torch
    from vllm_omni.diffusion.models.flux.concurrent_scheduler import ConcurrentScheduler
    from vllm_omni.diffusion.models.flux.request_context import RequestContext

    streams = [torch.cuda.Stream() for _ in requests]
    states = []
    try:
        for req, stream in zip(requests, streams, strict=True):
            context = RequestContext(req.request_id, 0, None, {})
            states.append(executor.init_state(req, context, stream=stream))
    except BaseException:
        for state in states:
            executor.abort(state)
        raise
    scheduler = ConcurrentScheduler(executor)
    results = scheduler.run(states)
    for state, result in zip(states, results, strict=True):
        service_seconds = result.timing["request_seconds"]
        result.timing.update(
            service_seconds=service_seconds,
            request_seconds=service_seconds + state.started_at - started,
        )
    return results, [int(stream.cuda_stream) for stream in streams], scheduler.enqueue_order


def prepare_dense_reference_batch(processor, images):
    """Use the same RGB conversion as the single-image VAE preprocessing path."""
    return processor.numpy_to_pt(processor.pil_to_numpy(images))


def _native_forward(
    pipeline: Any,
    post_process: Any,
    requests: list[Any],
    *,
    true_batch: bool,
    trace: dict[str, Any] | None = None,
) -> tuple[list[Any], list[dict[str, Any]], list[int]]:
    """Run native forward, witnessing the dense transformer batch when applicable."""
    import time
    from types import SimpleNamespace

    import torch

    count = len(requests)
    if not true_batch and count != 1:
        raise ValueError("Dense serial helper requires one request")
    batch_sizes = []

    def record_batch(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        states = kwargs.get("hidden_states", args[0] if args else None)
        batch_sizes.append(int(states.shape[0]))

    stream = torch.cuda.current_stream()
    device_start, device_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    with torch.inference_mode():
        hook = pipeline.transformer.register_forward_pre_hook(record_batch, with_kwargs=True)
        try:
            started = time.perf_counter()
            device_start.record(stream)
            if true_batch:
                sampling = requests[0].sampling_params
                sampling.generator = [request.sampling_params.generator for request in requests]
                # The pipeline resizes a tensor batch; its resize helper takes
                # one PIL image at a time. Keep the conversion inside the timer.
                batch = SimpleNamespace(prompts=[], sampling_params=sampling)
                images = [request.prompts[0]["multi_modal_data"]["image"] for request in requests]
                image_batch = prepare_dense_reference_batch(pipeline.image_processor, images)
                raw = pipeline.forward(
                    batch,
                    prompt=[request.prompts[0]["prompt"] for request in requests],
                    negative_prompt=[NEGATIVE_PROMPT] * count,
                    image=image_batch,
                ).output
            else:
                raw = pipeline.forward(requests[0]).output
            images = _unwrap_images(post_process(raw), count)
            device_end.record(stream)
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
        finally:
            hook.remove()
    metadata = raw.get("metadata", {}).get("flux_sparseedit", {}) if isinstance(raw, dict) else {}
    sparse_timing = _prune_metadata(metadata, trace) if trace is not None else {}
    rows = [
        {
            "request_seconds": wall,
            "cuda_event_seconds": device_start.elapsed_time(device_end) * 1e-3,
            "stage_timing": metadata.get("pipeline_timing"),
            **sparse_timing,
        }
        for _ in requests
    ]
    return images, rows, batch_sizes


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _png_records(
    images: list[Any],
    tags: tuple[str, ...],
    identities: dict[str, dict[str, Any]],
    root: Path,
) -> list[dict[str, Any]]:
    import time

    rows = []
    for image, tag in zip(images, tags, strict=True):
        if image.size != (FROZEN_PROTOCOL["width"], FROZEN_PROTOCOL["height"]):
            raise ValueError(f"Output resolution changed for {tag}")
        path = root / f"{tag}.png"
        started = time.perf_counter()
        image.save(path)
        rows.append(
            {
                **identities[tag],
                "output_png": path.name,
                "output_png_sha256": sha256_file(path),
                "png_write_seconds": time.perf_counter() - started,
            }
        )
    return rows


def _prune_metadata(data: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    from run_baseline import validate_trace_application

    validate_trace_application(data, trace)
    indices = [row["step"] for row in trace["steps"] if row["phase"] == "sparse"]
    timing = data["timing"]
    forward = timing["step_forward_device_seconds"]
    return {
        "denoise_device_seconds": timing["denoise_device_seconds"],
        "denoise_wall_seconds": timing["denoise_wall_seconds"],
        "first_sparse_step_forward_seconds": forward[indices[0]],
        "steady_sparse_step_forward_seconds": _median([forward[i] for i in indices[1:]]),
        "runtime_variant": data.get("runtime_variant"),
        "step_forward_device_seconds": forward,
        "graph_replayed_steps": sum(row.get("graph_replayed") is True for row in data["step_records"]),
        "step_timing": timing,
    }


def _median(values: list[float]) -> float:
    from statistics import median

    return float(median(values))


def _gpu_query(device_uuid: str, category: str, fields: str) -> list[list[str]]:
    import csv
    import subprocess

    result = subprocess.run(
        ["nvidia-smi", "--id", device_uuid, f"--query-{category}={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


@contextmanager
def _monitor_utilization(device_uuid: str) -> Iterator[list[dict[str, Any]]]:
    import subprocess
    import threading
    import time

    samples = []
    try:
        process = subprocess.Popen(
            [
                "nvidia-smi",
                "--id",
                device_uuid,
                "--query-gpu=utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
                "-lms",
                "500",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1,
        )
    except OSError:
        yield samples
        return

    def read_samples() -> None:
        for line in process.stdout:
            try:
                gpu, memory = [int(value) for value in line.split(",")]
            except ValueError:
                continue
            samples.append({"monotonic_seconds": time.perf_counter(), "gpu_percent": gpu, "memory_percent": memory})

    reader = threading.Thread(target=read_samples, daemon=True)
    reader.start()
    try:
        yield samples
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        reader.join(timeout=5)


def summarize_gpu_overlap(trace: dict[str, Any]) -> dict[str, Any]:
    """Count concurrent CUDA kernel activity using device timestamps."""
    from collections import Counter

    starts = []
    kernel_counts = Counter()
    for event in trace.get("traceEvents", []):
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        args = event.get("args", {})
        stream, device = args.get("stream"), args.get("device")
        if type(stream) is not int or type(device) is not int or not _positive(event.get("dur")):
            continue
        kernel_counts[stream] += 1
        starts.append((event["ts"], device, stream, 1))
        starts.append((event["ts"] + event["dur"], device, stream, -1))
    active_streams = Counter()
    device_counts = Counter()
    previous = 0.0
    union = overlap = 0.0
    for moment, device, stream, delta in sorted(starts):
        active = max(device_counts.values(), default=0)
        if active:
            union += moment - previous
        if active > 1:
            overlap += moment - previous
        key = device, stream
        was_active = active_streams[key] > 0
        active_streams[key] += delta
        device_counts[device] += int(active_streams[key] > 0) - int(was_active)
        previous = moment
    return {
        "per_stream_kernel_count": dict(kernel_counts),
        "gpu_kernel_active_microseconds": union,
        "gpu_multi_stream_overlap_microseconds": overlap,
        "scope": "all CUDA kernel streams in the independent profiled run, including auxiliary streams",
    }


def _profile_cell(root: Path, cell: Measurement, run: Any) -> dict[str, Any]:
    import torch

    root.mkdir(parents=True, exist_ok=False)
    trace_path = root / "trace.json"
    data = {"status": "UNAVAILABLE", "arm": cell.arm, "case_tags": cell.cases, "profiled_separately": True}
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            raw = run(cell)
        for image in raw["images"]:
            image.close()
        profile.export_chrome_trace(str(trace_path))
        device_timeline = summarize_gpu_overlap(read_json(trace_path))
        data.update(
            status="PASS" if device_timeline["per_stream_kernel_count"] else "UNAVAILABLE",
            cuda_trace=str(trace_path),
            cuda_trace_sha256=sha256_file(trace_path),
            enqueue_order=raw["enqueue_order"],
            **device_timeline,
        )
    except Exception as error:
        data["failure"] = str(error)
    _write_json(root / "completion.json", data)
    path = root / "completion.json"
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "status": data["status"]}


def _png_expectations(gate: dict[tuple[str, str], dict[str, Any]]) -> dict[str, str]:
    expected = {}
    for row in gate.values():
        for case in row["cases"]:
            tag = case["case_tag"]
            digest = case["output_png_sha256"]
            if tag in expected and expected[tag] != digest:
                raise ValueError(f"Inconsistent F1 reference PNG across calibration pairs for {tag}")
            expected[tag] = digest
    return expected


def _run_cell(
    pipeline: Any,
    post_process: Any,
    measurement: Measurement,
    protocol: dict[str, Any],
    lane: str,
    bound_cases: dict[str, BoundCase],
) -> dict[str, Any]:
    import time

    import torch

    tags, arm, mode = measurement.cases, measurement.arm, measurement.mode
    prepared = []
    resize_seconds = []
    for tag in tags:
        started = time.perf_counter()
        prepared.append(resize_case(bound_cases[tag], protocol))
        resize_seconds.append(time.perf_counter() - started)
    results = None
    try:
        requests = [
            make_request(bound_cases[tag], arm, protocol, image=image, lane=lane)
            for tag, image in zip(tags, prepared, strict=True)
        ]
        transformer_batch_sizes = []
        enqueue_order = []
        if arm == "standard_sparse" and mode == "concurrent":
            raise NotImplementedError("The v4 standard backend session mutates the shared FLUX transformer")
        if arm not in ("dense", "standard_sparse"):
            from vllm_omni.diffusion.models.flux.request_executor import RequestExecutor

            executor = RequestExecutor(pipeline, post_process=post_process, capture_steps=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = {
            "allocated_before_batch_bytes": int(torch.cuda.memory_allocated()),
            "reserved_before_batch_bytes": int(torch.cuda.memory_reserved()),
        }
        started = time.perf_counter()
        stopped = None
        if arm in ("dense", "standard_sparse"):
            if mode == "true_batch" and arm != "dense":
                raise NotImplementedError("SparseEdit inputs stay request-local batch=1")
            if mode == "concurrent":
                raise NotImplementedError("Dense multistream requires private dense step state")
            streams = [int(torch.cuda.current_stream().cuda_stream)]
            if mode == "true_batch":
                images, timings, transformer_batch_sizes = _native_forward(
                    pipeline,
                    post_process,
                    requests,
                    true_batch=True,
                )
            else:
                images, timings = [], []
                boundary = started
                for req, tag in zip(requests, tags, strict=True):
                    trace = bound_cases[tag][3] if arm == "standard_sparse" else None
                    output, rows, sizes = _native_forward(
                        pipeline,
                        post_process,
                        [req],
                        true_batch=False,
                        trace=trace,
                    )
                    stopped = time.perf_counter()
                    rows[0]["request_seconds"] = stopped - boundary
                    boundary = stopped
                    images.append(output[0])
                    timings.extend(rows)
                    transformer_batch_sizes.extend(sizes)
        else:
            if mode == "true_batch":
                raise NotImplementedError("SparseEdit inputs stay request-local batch=1")
            with torch.inference_mode():
                if mode == "serial":
                    results, streams, stopped = _serial_sparse(executor, requests, protocol, started)
                else:
                    results, streams, enqueue_order = _parallel_sparse(executor, requests, started)
        if stopped is None:
            torch.cuda.synchronize()
            stopped = time.perf_counter()
        wall = stopped - started
        peaks = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        if results is not None:
            images = [_unwrap_images(result.output, 1)[0] for result in results]
            timings = [
                {
                    **result.timing,
                    **_prune_metadata(result.metadata, bound_cases[tag][3]),
                }
                for tag, result in zip(tags, results, strict=True)
            ]
        if arm == "full":
            expected = ("graph_reuse", 19) if lane == "private_graph" else ("transformed", 0)
            if any((row["runtime_variant"], row["graph_replayed_steps"]) != expected for row in timings):
                raise ValueError("Full F2 backend lane differs from its F1 admission")
        return {
            "batch_wall_seconds": wall,
            "timings": timings,
            "resize_seconds": resize_seconds,
            "streams": streams,
            "images": images,
            "transformer_batch_sizes": transformer_batch_sizes,
            "enqueue_order": enqueue_order,
            "timed_monotonic_range_seconds": [started, stopped],
            **before,
            **peaks,
        }
    finally:
        for image in prepared:
            image.close()


def _environment() -> dict[str, str]:
    from importlib.metadata import version

    return {
        "python": sys.version.split()[0],
        "torch": version("torch"),
        "triton": version("triton"),
        "vllm": version("vllm"),
        "python_path": sys.executable,
    }


def _check_measurement_device(device_uuid: str) -> dict[str, Any]:
    _, uuid, name, total, free, driver = _gpu_query(
        device_uuid,
        "gpu",
        "index,uuid,name,memory.total,memory.free,driver_version",
    )[0]
    if uuid != device_uuid:
        raise ValueError("Selected UUID differs from requested experiment device")
    apps = _gpu_query(device_uuid, "compute-apps", "gpu_uuid,pid,used_memory")
    if apps:
        raise ValueError("Selected device has existing compute processes")
    return {
        "device_uuid": uuid,
        "device_name": name,
        "total_bytes": int(total) * 2**20,
        "free_bytes_before_model_load": int(free) * 2**20,
        "driver_version": driver,
        "compute_processes_before_model_load": apps,
    }


def _measurement_receipt(
    root: Path,
    scheduled: Measurement,
    order: int,
    base: dict[str, Any],
    identities: dict[str, dict[str, Any]],
    expected_pngs: dict[str, str],
    warmup: dict[str, Any],
    gpu_samples: list[dict[str, Any]],
    run: Any,
) -> bool:
    import torch

    root.mkdir(parents=True, exist_ok=False)
    count = len(scheduled.cases)
    data = {
        **base,
        "phase": "F2",
        "status": "FAILED",
        "arm": scheduled.arm,
        "mode": scheduled.mode,
        "concurrency": count,
        "measurement_order": order,
        "pair": scheduled.pair,
        "repetition": scheduled.repetition,
        "cases": [identities[tag] for tag in scheduled.cases],
        "runtime_lane": {
            "full": base["runtime_lane"],
            "direct_sparse": "direct_eager",
            "standard_sparse": "standard_fastops_eager",
            "dense": "native_dense",
        }[scheduled.arm],
        "independent_warmup": warmup,
    }
    try:
        if warmup["status"] == "UNSUPPORTED":
            raise NotImplementedError(warmup["failure"])
        if warmup["status"] == "CAPACITY_BLOCKED":
            raise torch.cuda.OutOfMemoryError(warmup["failure"])
        if warmup["status"] != "PASS":
            raise RuntimeError(f"Independent warmup failed: {warmup['failure']}")
        raw = run(scheduled)
        images = raw.pop("images")
        timings = raw.pop("timings")
        resize_seconds = raw.pop("resize_seconds")
        cases = _png_records(images, scheduled.cases, identities, root)
        for image in images:
            image.close()
        for case, timing, resize in zip(cases, timings, resize_seconds, strict=True):
            case.update(timing, input_resize_seconds=resize)
        start, stop = raw["timed_monotonic_range_seconds"]
        samples = [sample for sample in gpu_samples if start <= sample["monotonic_seconds"] <= stop]
        exact = [case["output_png_sha256"] == expected_pngs[case["case_tag"]] for case in cases]
        if scheduled.arm == "dense":
            correctness = {
                "all_match": (
                    len({case["output_png_sha256"] for case in cases}) == count
                    and raw["transformer_batch_sizes"]
                    == [count if scheduled.mode == "true_batch" else 1]
                    * (2 * base["protocol"]["steps"] * (count if scheduled.mode == "serial" else 1))
                ),
                "method": "distinct correctly shaped PNG outputs and expected CUDA CFG forward batch sizes",
            }
        else:
            correctness = {
                "all_match": all(exact),
                "method": "original F1 serial reference PNG SHA-256 with current frozen trace execution",
                "per_request_exact": exact,
            }
        data.update(
            raw,
            synchronized_before_return=True,
            cases=cases,
            status="PASS",
            correctness=correctness,
            throughput_requests_per_second=count / raw["batch_wall_seconds"],
            latency_p50_seconds=_median([case["request_seconds"] for case in cases]),
            gpu_utilization={
                "status": "MEASURED" if samples else "UNAVAILABLE",
                "sampling_interval_ms": 500,
                "samples": samples,
                "mean_gpu_percent": (
                    sum(sample["gpu_percent"] for sample in samples) / len(samples) if samples else None
                ),
                "observer": "nvidia-smi device utilization samples within synchronized batch wall",
            },
            scoring="separate analysis of saved outputs",
        )
        validate_measurement(data, base["protocol"], identities)
    except torch.cuda.OutOfMemoryError as error:
        data.update(
            status="CAPACITY_BLOCKED",
            failure=str(error),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
    except NotImplementedError as error:
        data.update(status="UNSUPPORTED", failure=str(error))
    except Exception as error:
        import traceback

        data.update(status="FAILED", failure=str(error), traceback=traceback.format_exc())
    finally:
        _write_json(root / "completion.json", data)
    print(f"{data['status']} {root / 'completion.json'}", flush=True)
    return data["status"] == "PASS"


def _warm_up(
    root: Path,
    schedule: list[Measurement],
    run: Any,
    base: dict[str, Any],
    identities: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Prime each execution arm and scheduling mode with a separate request."""
    from dataclasses import replace

    import torch

    observations = {}
    for cell in schedule:
        arm_mode = cell.arm, cell.mode
        if arm_mode in observations:
            continue
        if cell.mode != "true_batch":
            cell = replace(cell, cases=cell.cases[:1])
        folder = root / f"warmup_{cell.arm}_{cell.mode}_r{cell.repetition}"
        folder.mkdir(parents=True, exist_ok=False)
        row = {
            **base,
            "phase": "F2_warmup",
            "arm": cell.arm,
            "mode": cell.mode,
            "repetition": cell.repetition,
            "concurrency": len(cell.cases),
            "cases": [identities[tag] for tag in cell.cases],
        }
        try:
            raw = run(cell)
            for image in raw.pop("images"):
                image.close()
            row.update(raw, status="PASS")
        except torch.cuda.OutOfMemoryError as error:
            row.update(
                status="CAPACITY_BLOCKED",
                failure=str(error),
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            )
        except NotImplementedError as error:
            row.update(status="UNSUPPORTED", failure=str(error))
        except Exception as error:
            import traceback

            row.update(status="FAILED", failure=str(error), traceback=traceback.format_exc())
        _write_json(folder / "completion.json", row)
        observations[arm_mode] = {
            "status": row["status"],
            "wall_seconds": row.get("batch_wall_seconds"),
            "failure": row.get("failure"),
            "peak_allocated_bytes": row.get("peak_allocated_bytes"),
            "peak_reserved_bytes": row.get("peak_reserved_bytes"),
            "path": str((folder / "completion.json").resolve()),
            "sha256": sha256_file(folder / "completion.json"),
        }
        if row["status"] != "PASS":
            print(f"{row['status']} {folder / 'completion.json'}", flush=True)
    return observations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="full")
    parser.add_argument("--mode", choices=MODES, default="concurrent")
    parser.add_argument("--concurrency", type=int, choices=(1, 2), default=2)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--start-repetition", type=int, default=0)
    parser.add_argument("--matrix", action="store_true", help="Run paired full N=1/2 and matched strong controls.")
    parser.add_argument("--output-root", type=Path, default=STUDY / "artifacts/concurrency_v1")
    parser.add_argument("--isolation-root", type=Path, default=STUDY / "artifacts/isolation_v1")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--seed", type=int, choices=(110,), default=110)
    parser.add_argument(
        "--model",
        type=Path,
        help="Verify every file against the frozen v4 model hashes at this explicit path.",
    )
    args = parser.parse_args()
    if args.repetitions < 1 or args.start_repetition < 0:
        parser.error("Choose a positive repetition count and a nonnegative starting repetition.")
    if not args.output_root.resolve().is_relative_to(STUDY / "artifacts"):
        parser.error("Select an output directory under the study artifacts.")
    return args


def main() -> int:
    import fcntl
    import os
    import socket
    import time
    import traceback
    from datetime import UTC, datetime
    from functools import partial

    args = _parse_args()
    sys.path.insert(0, str(STUDY / "controls"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    attempt = args.output_root / f"attempt_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S_%fZ')}"
    attempt.mkdir(exist_ok=False)
    data = {
        "phase": "F2_attempt",
        "status": "BLOCKED",
        "device_uuid": args.gpu_uuid,
        "node": socket.gethostname(),
        "timing_boundary": TIMING_BOUNDARY,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "source_sha": source_tree_sha(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "command": sys.argv,
        "start_repetition": args.start_repetition,
        "repetitions": args.repetitions,
        "environment": {"python": sys.version.split()[0], "python_path": sys.executable},
        "planned_measurements": [],
        "passed_cells": 0,
    }
    try:
        if Path(sys.prefix).resolve() != PYTHON.parents[1].resolve():
            raise RuntimeError(f"F2 requires the fixed Python at {PYTHON}")
        data["environment"] = _environment()
        protocol, identities = load_protocol_cases()
        data.update(protocol=protocol, calibration_cases=list(identities.values()))
        schedule = list(measurement_schedule(args, tuple(identities)))
        data["planned_measurements"] = [row.directory for row in schedule]
        for row in schedule:
            if (args.output_root / row.directory).exists():
                raise ValueError(f"Measurement directory already exists: {row.directory}")
        gate = load_isolation_gate(
            args.isolation_root,
            calibration_pairs(tuple(identities)),
            protocol,
            identities,
            data["source_sha"],
            args.gpu_uuid,
        )
        data["isolation_receipts"] = [
            {
                "path": row["receipt"],
                "relative_path": os.path.relpath(row["receipt"], args.isolation_root.resolve()),
                "sha256": row["receipt_sha256"],
            }
            for row in gate.values()
        ]
        data["runtime_lane"] = next(iter(gate.values()))["runtime_lane"]
        expected_pngs = _png_expectations(gate)
        lock_path = Path(f"/tmp/sparseedit-{args.gpu_uuid}.lock")
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            data.update(_check_measurement_device(args.gpu_uuid))
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
            os.environ.pop("PYTORCH_ALLOC_CONF", None)
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
            sys.path.insert(0, str(STUDY / "source"))
            from run_baseline import load_cases, verify_hashes, verify_model_identity

            bundle = read_json(BUNDLE_ROOT / "bundle.json")
            data["bundle_sha256"] = sha256_file(BUNDLE_ROOT / "bundle.json")
            model = MODEL if args.model is None else args.model.resolve()
            if args.model is None:
                data["model_identity"] = verify_model_identity(bundle["model_hashes"])
            else:
                verify_hashes(model, bundle["model_hashes"])
                data["model_identity"] = {
                    "model_path": str(model),
                    "files": len(bundle["model_hashes"]),
                    "method": "current full-file hashes matched the frozen v4 bundle",
                }
            bound_cases = {row[0]["case_tag"]: row for row in load_cases(BUNDLE_ROOT, bundle)}
            model_started = time.perf_counter()
            with load_pipeline(model) as (pipeline, post_process):
                data["model_load_seconds"] = time.perf_counter() - model_started
                run = partial(
                    _run_cell,
                    pipeline,
                    post_process,
                    protocol=protocol,
                    lane=data["runtime_lane"],
                    bound_cases=bound_cases,
                )
                warmups = _warm_up(attempt, schedule, run, data, identities)
                with _monitor_utilization(args.gpu_uuid) as samples:
                    for order, cell in enumerate(schedule):
                        warmup = warmups[cell.arm, cell.mode]
                        cell_ok = _measurement_receipt(
                            args.output_root / cell.directory,
                            cell,
                            order,
                            data,
                            identities,
                            expected_pngs,
                            warmup,
                            samples,
                            run,
                        )
                        data["passed_cells"] += int(cell_ok)
                if args.matrix:
                    profile_cell = next(
                        row
                        for row in schedule
                        if row.arm == "full" and row.mode == "concurrent" and len(row.cases) == 2
                    )
                    if read_json(args.output_root / profile_cell.directory / "completion.json")["status"] == "PASS":
                        data["gpu_timeline"] = _profile_cell(attempt / "gpu_timeline", profile_cell, run)
                optional_standard = sum(
                    row.arm == "standard_sparse"
                    and row.mode == "concurrent"
                    and warmups[row.arm, row.mode]["status"] == "UNSUPPORTED"
                    for row in schedule
                )
                required = len(schedule) - optional_standard if args.matrix else len(schedule)
                data["status"] = "PASS" if data["passed_cells"] == required else "PARTIAL"
    except Exception as error:
        data.update(failure=str(error), traceback=traceback.format_exc())
    finally:
        data["completed_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(attempt / "completion.json", data)
    print(f"{data['status']} {attempt / 'completion.json'}", flush=True)
    return 0 if data["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
