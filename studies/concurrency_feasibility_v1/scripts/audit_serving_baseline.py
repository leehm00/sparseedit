#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit native vLLM-Omni async execution on frozen FLUX calibration requests."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path


STUDY = Path(__file__).resolve().parents[1]
REPORT = STUDY / "artifacts/serving_baseline_v1"
REQUIRED = ("model", "workload", "algorithm", "backend", "concurrency", "timing", "quality", "hardware", "status")
TIMING = "AsyncOmni.generate submission to completed image output, including queue wait. Worker CUDA events are diagnostic."


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def local_throughput(row):
    evidence = row.get("measured_local")
    if not evidence:
        raise ValueError("A local receipt is required for every quantitative comparison.")
    path = Path(evidence["receipt"])
    if sha256(path) != evidence["sha256"]:
        raise ValueError(f"Local receipt hash differs: {path}")
    data = load_json(path)
    seconds = data.get("batch_wall_seconds", 0)
    if data.get("status") != "PASS" or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"Missing successful batch measurement: {path}")
    requests = row["workload"]["requests"]
    if data.get("concurrency") != requests:
        raise ValueError(f"Local receipt request count differs: {path}")
    return requests / seconds


def validate_baseline_report(report):
    rows = report["baselines"]
    for name, row in rows.items():
        missing = set(REQUIRED) - row.keys()
        if missing or row.get("status") not in {"RUNNABLE", "ADAPTED", "INCOMPATIBLE"}:
            raise ValueError(f"Invalid baseline {name}: missing {sorted(missing)} or unknown status.")
        if "speedup" in row:
            raise ValueError(f"Unpaired speedup in baseline {name}.")
    for comparison in report["speedups"]:
        reference = rows[comparison["reference"]]
        candidate = rows[comparison["candidate"]]
        for field in ("model", "workload", "timing", "hardware"):
            if reference[field] != candidate[field]:
                raise ValueError(f"Cannot compare speedup with different {field}.")
        expected = local_throughput(candidate) / local_throughput(reference)
        if not math.isclose(expected, comparison["speedup"], rel_tol=1e-9):
            raise ValueError("Speedup differs from the ratio recomputed from local receipts.")


def classify_worker_timeline(client, worker):
    if len(client) != 2 or len(worker) != 2 or len({row["request_id"] for row in client}) != 2:
        return "UNRESOLVED"
    for row in client:
        prefix = row["request_id"] + "-"
        if sum(item["request_id"].startswith(prefix) for item in worker) != 1:
            return "UNRESOLVED"
        if row["submitted_ns"] >= row["completed_ns"]:
            return "UNRESOLVED"
    if max(row["submitted_ns"] for row in client) >= min(row["completed_ns"] for row in client):
        return "UNRESOLVED"
    first, second = sorted(worker, key=lambda row: row["cuda_begin_ms"])
    if first["pid"] != second["pid"] or any(row["cuda_begin_ms"] >= row["cuda_end_ms"] for row in worker):
        return "UNRESOLVED"
    if first["stream_id"] == second["stream_id"]:
        if first["cuda_end_ms"] <= second["cuda_begin_ms"]:
            return "SERIAL_BASELINE"
        return "UNRESOLVED"
    if first["cuda_end_ms"] > second["cuda_begin_ms"]:
        return "OVERLAPPING_WORKER_WINDOWS"
    return "UNRESOLVED"


class WorkerTimelineExtension:
    """Add CUDA events to native request dispatch through the worker extension API."""

    def execute_model(self, req, od_config, kv_prefetch_job=None):
        if not req.request_id.startswith("f5-cal"):
            return super().execute_model(req, od_config, kv_prefetch_job=kv_prefetch_job)
        import torch

        stream = torch.cuda.current_stream(self.device)
        if not hasattr(self, "_f5_events"):
            self._f5_events = []
            self._f5_anchor = torch.cuda.Event(enable_timing=True)
            self._f5_anchor.record(stream)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started_ns = time.monotonic_ns()
        begin.record(stream)
        output = super().execute_model(req, od_config, kv_prefetch_job=kv_prefetch_job)
        end.record(stream)
        self._f5_events.append(({
            "request_id": req.request_id,
            "pid": os.getpid(),
            "rank": self.rank,
            "stream_id": int(stream.cuda_stream),
            "worker_entered_ns": started_ns,
            "worker_returned_ns": time.monotonic_ns(),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }, begin, end))
        return output

    def f5_worker_timeline(self):
        rows = []
        for row, begin, end in getattr(self, "_f5_events", []):
            end.synchronize()
            rows.append({
                **row,
                "cuda_begin_ms": self._f5_anchor.elapsed_time(begin),
                "cuda_end_ms": self._f5_anchor.elapsed_time(end),
                "cuda_event_seconds": begin.elapsed_time(end) / 1000,
            })
        return rows


def timeline_rows(payload):
    if isinstance(payload, dict):
        if "cuda_begin_ms" in payload:
            yield payload
    elif isinstance(payload, (list, tuple)):
        for element in payload:
            yield from timeline_rows(element)


async def run_async(omni, sampling, cases, parameters, output, receipt):
    from PIL import Image

    from admission import validate_trace_application
    from protocol import options
    from run_flux_specialized_runtime import NEGATIVE_PROMPT, _extract_flux_metadata

    async def generate(index, case, *, warmup=False):
        bound, image_file, trace_file, trace = case
        request_id = "f5-warmup" if warmup else f"f5-cal{index}"
        resize_started = time.perf_counter()
        with Image.open(image_file) as source:
            image = source.convert("RGB").resize((parameters["width"], parameters["height"]))
        input_resize_seconds = time.perf_counter() - resize_started
        with image:
            prompt = sampling.build_image_to_image_prompt(
                model_class_name=sampling.model_class_name,
                prompt=bound["instruction"],
                negative_prompt=NEGATIVE_PROMPT,
                input_image=image,
                height=parameters["height"],
                width=parameters["width"],
            )
            params = sampling._sampling_params(
                method="sparseedit",
                extra_args=options("full", trace_file, audit=False),
                seed=parameters["seed"],
                height=parameters["height"],
                width=parameters["width"],
            )
            submitted_ns = time.monotonic_ns()
            outputs = [
                value async for value in omni.generate(prompt, request_id=request_id, sampling_params_list=params)
            ]
            completed_ns = time.monotonic_ns()
        metadata = _extract_flux_metadata(outputs)
        if not metadata or metadata["runtime_variant"] != "graph_reuse":
            raise ValueError(f"{request_id} did not use the frozen full runtime.")
        validate_trace_application(metadata, trace)
        sparse = [step["step"] for step in trace["steps"] if step["phase"] == "sparse"]
        if not all(metadata["step_records"][step].get("graph_replayed") for step in sparse):
            raise ValueError(f"{request_id} did not replay every sparse graph step.")
        images = sampling.extract_images_from_outputs(outputs)
        if len(images) != 1 or images[0].size != (parameters["width"], parameters["height"]):
            raise ValueError(f"{request_id} returned an unexpected image batch.")
        return {
            "request_id": request_id,
            "case_tag": bound["case_tag"],
            "input_sha256": bound["prepared_sha256"],
            "trace_sha256": bound["trace_sha256"],
            "submitted_ns": submitted_ns,
            "completed_ns": completed_ns,
            "input_resize_seconds": input_resize_seconds,
            "executed_steps": len(metadata["step_records"]),
            "graph_replay_steps": sparse,
            "memory": metadata["memory"],
        }, images[0]

    print("Running separate native warmup", flush=True)
    warmup, _ = await generate(0, cases[0], warmup=True)
    receipt["warmup"] = warmup
    print("Submitting two independent AsyncOmni.generate requests", flush=True)
    outcomes = await asyncio.gather(*(generate(i, case) for i, case in enumerate(cases)), return_exceptions=True)
    receipt["request_errors"] = [
        {"index": i, "error": repr(result)}
        for i, result in enumerate(outcomes)
        if isinstance(result, BaseException)
    ]
    for result in outcomes:
        if isinstance(result, BaseException):
            continue
        row, image = result
        path = output / f"{row['request_id']}.png"
        image.save(path)
        row["output_png_sha256"] = sha256(path)
        row["output_png"] = path.name
        receipt["cases"].append(row)
    if receipt["request_errors"]:
        raise RuntimeError(f"Async requests failed: {receipt['request_errors']}")
    payload = await omni.collective_rpc("f5_worker_timeline", timeout=60.0)
    receipt["worker_timeline"] = list(timeline_rows(payload))
    receipt["worker_timeline_rpc"] = payload
    receipt["execution_status"] = classify_worker_timeline(receipt["cases"], receipt["worker_timeline"])
    if receipt["execution_status"] != "SERIAL_BASELINE":
        raise ValueError(f"Native async execution classification: {receipt['execution_status']}")


def run_native(args, output, receipt):
    from run_baseline import ALLOCATOR, MODEL, V4, frozen, load_cases, verify_hashes, verify_model_identity

    bundle_root = V4 / "backend_bundle"
    bundle_file = bundle_root / "bundle.json"
    bundle = load_json(bundle_file)
    verify_hashes(bundle_root / "runtime", bundle["runtime_hashes"])
    verify_hashes(STUDY / "controls", {
        name: bundle["control_hashes"][name]
        for name in ("run_flux_specialized_runtime.py", "admission.py", "protocol.py")
    })
    receipt["bundle_sha256"] = sha256(bundle_file)
    receipt["source_sha256"] = frozen.sha256_payload(bundle["runtime_hashes"])
    receipt["script_sha256"] = sha256(__file__)
    if args.verify_full_model_hashes:
        verify_hashes(MODEL, bundle["model_hashes"])
        receipt["model_identity"] = {"model_path": str(MODEL), "files": len(bundle["model_hashes"]),
                                     "method": "frozen v4 file checksums"}
    else:
        receipt["model_identity"] = verify_model_identity(bundle["model_hashes"])
    cases = load_cases(bundle_root, bundle)[:2]
    receipt["scheduled_cases"] = [
        {
            "case_tag": case["case_tag"],
            "input_sha256": case["prepared_sha256"],
            "trace_sha256": case["trace_sha256"],
        }
        for case, *_ in cases
    ]
    preflight = subprocess.run(
        [sys.executable, str(STUDY / "scripts/check_env.py"), "--gpu-uuid", args.gpu_uuid,
         "--require-gpu-name", args.require_gpu_name],
        capture_output=True, text=True, check=False,
    )
    receipt["environment"] = json.loads(preflight.stdout)
    if preflight.returncode:
        receipt["status"] = "BLOCKED"
        raise RuntimeError(f"Device preflight failed: {receipt['environment']['reasons']}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ALLOCATOR
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

    import torch
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
    from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.entrypoints.openai.stage_params import clone_sampling_params
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras import (
        build_image_to_image_prompt,
        get_extra_body_params,
        get_model_class_name,
        should_init_extra_args_for_non_diffusion_stages,
    )
    from vllm_omni.platforms import current_omni_platform

    if not Path(sys.modules["vllm_omni"].__file__).resolve().is_relative_to(bundle_root / "runtime"):
        raise ValueError("F5 requires importing the frozen v4 runtime from PYTHONPATH.")
    receipt["entrypoint"] = "AsyncOmni.generate"
    receipt["requested_engine_config"] = {
        "step_execution": False,
        "max_num_seqs": 1,
        "parallel_world_size": 1,
        "startup_timeout_s": args.startup_timeout_s,
    }
    print("Loading frozen v4 AsyncOmni model", flush=True)
    started = time.perf_counter()
    omni = AsyncOmni(
        model=str(MODEL),
        vae_use_tiling=True,
        cache_backend=None,
        parallel_config=DiffusionParallelConfig(
            ulysses_degree=1, ring_degree=1, cfg_parallel_size=1, tensor_parallel_size=1,
        ),
        enable_cpu_offload=False,
        enforce_eager=True,
        max_num_seqs=1,
        step_execution=False,
        init_timeout=args.startup_timeout_s,
        stage_init_timeout=args.startup_timeout_s,
        worker_extension_cls="audit_serving_baseline.WorkerTimelineExtension",
    )
    try:
        receipt["model_load_seconds"] = time.perf_counter() - started
        resolved_config = omni.get_diffusion_od_config()
        get_config = resolved_config.get if isinstance(resolved_config, dict) else (
            lambda name: getattr(resolved_config, name, None)
        )
        receipt["resolved_engine_config"] = {
            name: get_config(name) for name in ("step_execution", "max_num_seqs", "model_class_name")
        }
        sampling = frozen.NativeWorker.__new__(frozen.NativeWorker)
        sampling.omni = omni
        sampling.torch = torch
        sampling.device_type = current_omni_platform.device_type
        sampling.OmniDiffusionSamplingParams = OmniDiffusionSamplingParams
        sampling.extract_images_from_outputs = extract_images_from_outputs
        sampling.apply_declared_extra_args = apply_declared_extra_args
        sampling.clone_sampling_params = clone_sampling_params
        sampling.build_image_to_image_prompt = build_image_to_image_prompt
        sampling.should_init_extra_args_for_non_diffusion_stages = should_init_extra_args_for_non_diffusion_stages
        sampling.model_class_name = get_model_class_name(omni)
        sampling.declared_extra_body_params = get_extra_body_params(sampling.model_class_name)
        asyncio.run(run_async(omni, sampling, cases, receipt["protocol"], output, receipt))
    finally:
        omni.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("report", type=Path)
    run = commands.add_parser("run-native")
    run.add_argument("--gpu-uuid", required=True)
    run.add_argument("--output-root", required=True, type=Path)
    run.add_argument("--startup-timeout-s", type=int, default=1800)
    run.add_argument("--require-gpu-name", default="NVIDIA A40")
    run.add_argument("--verify-full-model-hashes", action="store_true")
    args = parser.parse_args()
    if args.command == "validate":
        validate_baseline_report(load_json(args.report))
        return 0
    if not args.gpu_uuid.startswith("GPU-") or not args.gpu_uuid[4:].replace("-", "").isalnum():
        parser.error("Use a physical GPU UUID.")
    if args.startup_timeout_s <= 0:
        parser.error("--startup-timeout-s must be positive.")
    output = args.output_root.resolve()
    artifacts = STUDY / "artifacts"
    if not output.is_relative_to(artifacts) or output in (artifacts, REPORT):
        parser.error("Choose a fresh run directory under study artifacts.")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "phase": "F5_native_async_audit",
        "status": "FAILED",
        "host": socket.gethostname(),
        "device_uuid": args.gpu_uuid,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "repetition": 0,
        "protocol": load_json(STUDY / "protocol.json"),
        "timing_boundary": TIMING,
        "cases": [],
        "requested_requests": 2,
        "command": [sys.executable, *sys.argv],
    }
    lock_path = Path(f"/tmp/sparseedit-{args.gpu_uuid}.lock")
    receipt["uuid_lock"] = str(lock_path)
    try:
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                receipt["status"] = "BLOCKED"
                raise
            run_native(args, output, receipt)
        receipt["status"] = "PASS"
    except Exception as error:
        receipt["failure"] = repr(error)
        (output / "failure.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        receipt["completed_requests"] = len(receipt["cases"])
        receipt["finished_at_utc"] = datetime.now(UTC).isoformat()
        write_json(output / "completion.json", receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
