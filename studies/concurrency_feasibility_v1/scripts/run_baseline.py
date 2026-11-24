#!/usr/bin/env python3
"""Replay v4 full runtime on the three 768px calibration cases."""

import argparse
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
V4 = STUDY.parents[1] / "backend_reference/current_v4"
MODEL = STUDY.parents[2] / "flux_a40_kontext_28step_20260912/models/FLUX.1-Kontext-dev"
MODEL_IDENTITY = STUDY.parents[1] / "outputs/baseline_identity_fixed_v2/identity.json"
MODEL_IDENTITY_SHA = "6c07238c6d62c111b2c54cbc81bf49f39e9a781bd78d05d1f51159b6d8e2e99a"
ALLOCATOR = "expandable_segments:True"
sys.path.insert(0, str(STUDY / "controls"))

import run_flux_specialized_runtime as frozen
from admission import validate_trace_application
from protocol import PROTOCOL, options, validate_trace


def load_json(path):
    return json.loads(path.read_text())


def verify_hashes(root, expected):
    for name, digest in expected.items():
        if frozen.sha256_file(root / name) != digest:
            raise ValueError(f"Frozen file hash differs: {root / name}")


def verify_model_identity(model_hashes):
    verify_hashes(MODEL_IDENTITY.parent, {MODEL_IDENTITY.name: MODEL_IDENTITY_SHA})
    identity = load_json(MODEL_IDENTITY)
    bindings = identity["bindings"]
    witness = bindings["model_hash_interval"]
    if (
        identity["status"] != "PASS"
        or identity["model_bytes_verified"] is not True
        or bindings["model_hashes"] != model_hashes
        or Path(bindings["model"]).resolve() != MODEL.resolve()
        or witness["status"] != "VERIFIED_STABLE"
        or witness["before"] != witness["after"]
        or witness["after"] != bindings["model_stats"]
    ):
        raise ValueError("Reviewed v4 model identity differs.")
    actual = {}
    for file in MODEL.rglob("*"):
        relative = file.relative_to(MODEL)
        if file.is_file() and not any(part.startswith(".") for part in relative.parts):
            stat = file.stat()
            actual[str(relative)] = dict(
                size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                inode=stat.st_ino, device=stat.st_dev,
            )
    if actual != bindings["model_stats"]:
        raise ValueError("Model files changed since the reviewed complete hash interval.")
    return {
        "identity": str(MODEL_IDENTITY),
        "identity_sha256": MODEL_IDENTITY_SHA,
        "method": "reviewed stable full-hash interval and fresh filesystem stat comparison",
        "files": len(actual),
        "model_path": str(MODEL.resolve()),
    }


def load_cases(bundle_root, bundle):
    cases = bundle["cases"]["calibration"]
    if len(cases) != 3 or len({case["source_pixel_sha256"] for case in cases}) != 3:
        raise ValueError("The v4 bundle must bind three distinct calibration sources.")
    bound = []
    for case in cases:
        image = bundle_root / "inputs" / case["input_filename"]
        trace_file = bundle_root / "traces" / f'{case["case_tag"]}.json'
        verify_hashes(image.parent, {image.name: case["prepared_sha256"]})
        verify_hashes(trace_file.parent, {trace_file.name: case["trace_sha256"]})
        bound.append((case, image, trace_file, validate_trace(load_json(trace_file))))
    return bound


def readback_by_case(v4_root, native_readback):
    records = {}
    for phase in ("cal3", "calibration_timing"):
        if native_readback is not None:
            root = native_readback / phase
            receipt = load_json(root / "completion.json")
        else:
            root = v4_root / "formal_export" / phase
            receipt = load_json(v4_root / "readback" / f"{phase}_completion.json")
        if receipt["status"] != "PASS":
            continue
        for relative_path, digest in receipt["records"].items():
            path = root / relative_path
            if not path.is_file():
                continue
            verify_hashes(root, {relative_path: digest})
            record = load_json(path)
            if record["status"] == "PASS" and not record["warmup"]:
                records.setdefault(record["case_tag"], []).append((record, path))
    return records


def run_case(worker, bound_case, parameters, output):
    case, image_file, trace_file, trace = bound_case
    started = time.perf_counter()
    with worker.pil_image.open(image_file) as source:
        image = source.convert("RGB").resize((parameters["width"], parameters["height"]))
    resize_seconds = time.perf_counter() - started
    with image:
        result = worker.run(
            image, case["instruction"], method="sparseedit",
            extra_args=options("full", trace_file, audit=False), seed=parameters["seed"],
            height=parameters["height"], width=parameters["width"], output_path=output,
        )
    metadata = result["runtime_metadata"]
    validate_trace_application(metadata, trace)
    if metadata["runtime_variant"] != "graph_reuse" or (
        result["resolved_width"], result["resolved_height"]
    ) != (parameters["width"], parameters["height"]):
        raise ValueError("Actual resolution or full runtime differs from the F0 protocol.")
    sparse = [step["step"] for step in trace["steps"] if step["phase"] == "sparse"]
    if not all(metadata["step_records"][i].get("graph_replayed") for i in sparse):
        raise ValueError("Full runtime did not replay every sparse step.")
    timing = metadata["timing"]
    forward = timing["step_forward_device_seconds"]
    request_seconds = result["request_wall_seconds"]
    if not math.isfinite(request_seconds) or request_seconds <= 0:
        raise ValueError("Invalid synchronized full request timing.")
    return {
        "case_tag": case["case_tag"],
        "source_pixel_sha256": case["source_pixel_sha256"],
        "instruction": case["instruction"],
        "input_sha256": case["prepared_sha256"],
        "trace_sha256": case["trace_sha256"],
        "request_seconds": request_seconds,
        "denoise_device_seconds": timing["denoise_device_seconds"],
        "denoise_wall_seconds": timing["denoise_wall_seconds"],
        "first_sparse_step_forward_seconds": forward[sparse[0]],
        "steady_sparse_step_forward_seconds": statistics.median(forward[i] for i in sparse[1:]),
        "input_resize_seconds": resize_seconds,
        "output_png_sha256": frozen.sha256_file(output) if output else None,
        "peak_allocated_bytes": result["memory"]["max_memory_allocated"],
        "peak_reserved_bytes": result["memory"]["max_memory_reserved"],
        "request": result,
    }


def compare_readback(case, readback):
    matches = [
        (row, path) for row, path in readback
        if row["input_sha256"] == case["input_sha256"]
        and row["trace_sha256"] == case["trace_sha256"]
    ]
    full = [
        (row, path) for row, path in matches
        if row["arm"] == "full" and row["audit"] is False and row["native_execution"] is True
    ]
    if full:
        reference = statistics.mean(row["complete_request_seconds"] for row, _ in full)
        deviation = case["request_seconds"] / reference - 1.0
    else:
        reference = deviation = None
    return {
        "full_reference_seconds": reference,
        "full_relative_deviation": deviation,
        "full_latency_within_2_percent": abs(deviation) <= 0.02 if deviation is not None else None,
        "full_manifests": [str(path) for _, path in full],
        "png_readback": [
            {
                "arm": row["arm"],
                "png_equal": case["output_png_sha256"] == row["output_sha256"],
                "manifest": str(path),
            }
            for row, path in matches
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--v4-root", type=Path, default=V4)
    parser.add_argument("--native-readback", type=Path)
    parser.add_argument("--model", type=Path, help="Explicit model directory to verify against every v4 file hash.")
    parser.add_argument("--output-root", type=Path, default=STUDY / "artifacts/baseline_v1")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if not output.is_relative_to(STUDY / "artifacts"):
        parser.error("Choose a fresh output directory under this study's artifacts.")
    output.mkdir(parents=True, exist_ok=False)

    parameters = load_json(STUDY / "protocol.json")
    receipt = {
        "status": "FAILED", "phase": "baseline", "protocol": parameters, "variant": "full",
        "host": socket.gethostname(),
        "device_uuid": args.gpu_uuid, "repetition": 0, "scheduled_cases": 3, "cases": [],
        "timing_boundary": PROTOCOL["timer"],
        "step_timing_boundary": "v4 step_forward_device_seconds at sparse steps",
        "started_at_utc": frozen.utc_now(),
    }
    worker = None
    try:
        bundle_root = args.v4_root / "backend_bundle"
        bundle = load_json(bundle_root / "bundle.json")
        receipt["bundle_sha256"] = frozen.sha256_file(bundle_root / "bundle.json")
        verify_hashes(STUDY / "source", bundle["runtime_hashes"])
        receipt["source_sha"] = frozen.sha256_payload(bundle["runtime_hashes"])
        controls = {
            name: bundle["control_hashes"][name]
            for name in ("admission.py", "protocol.py", "run_flux_specialized_runtime.py")
        }
        verify_hashes(STUDY / "controls", controls)
        receipt["runner_sha256"] = controls["run_flux_specialized_runtime.py"]
        cases = load_cases(bundle_root, bundle)
        readback = readback_by_case(args.v4_root, args.native_readback)
        model = MODEL if args.model is None else args.model.resolve()
        if args.model is None:
            receipt["model_identity"] = verify_model_identity(bundle["model_hashes"])
        else:
            print(f"Verifying frozen model files at {model}", flush=True)
            verify_hashes(model, bundle["model_hashes"])
            receipt["model_identity"] = {
                "method": "current full file hashes verified against frozen v4 bundle",
                "files": len(bundle["model_hashes"]),
                "model_path": str(model),
            }
        environment = subprocess.run(
            [sys.executable, str(STUDY / "scripts/check_env.py"), "--gpu-uuid", args.gpu_uuid],
            capture_output=True, text=True,
        )
        admission = json.loads(environment.stdout)
        receipt.update(admission)
        receipt.update(status="FAILED", phase="baseline")
        if environment.returncode:
            receipt.update(status="BLOCKED", failure=admission["reasons"])
            return 1
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ALLOCATOR
        receipt["allocator"] = ALLOCATOR
        print("Loading the frozen v4 FLUX runtime", flush=True)
        worker = frozen.NativeWorker(argparse.Namespace(model=model))
        receipt["model_load_seconds"] = worker.model_load_seconds
        print("Running the separate unmeasured warmup", flush=True)
        warmup = run_case(worker, cases[0], parameters, None)
        receipt["warmup_request_seconds"] = warmup["request_seconds"]
        for ordinal, bound in enumerate(cases):
            case_tag = bound[0]["case_tag"]
            print(f"Measuring case {ordinal + 1}/3: {case_tag}", flush=True)
            path = output / case_tag
            case_record = run_case(worker, bound, parameters, path / "output.png")
            case_record["readback"] = compare_readback(case_record, readback.get(case_tag, []))
            frozen.write_json(path / "manifest.json", case_record)
            receipt["cases"].append({k: v for k, v in case_record.items() if k != "request"})
        reported = [case["readback"] for case in receipt["cases"]]
        receipt["mean_request_seconds"] = statistics.mean(
            case["request_seconds"] for case in receipt["cases"]
        )
        if any(row["full_latency_within_2_percent"] is None for row in reported):
            receipt.update(
                status="BLOCKED",
                failure="Same-case, no-audit v4 full runtime readback is missing.",
            )
        elif all(row["full_latency_within_2_percent"] for row in reported):
            receipt["status"] = "PASS"
        else:
            receipt["failure"] = "At least one full request exceeds the 2% reproduction bound."
    except Exception as error:
        receipt.update(failure=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        print(receipt["traceback"], file=sys.stderr)
    finally:
        try:
            if worker is not None:
                worker.close()
        finally:
            receipt["finished_at_utc"] = frozen.utc_now()
            receipt["completed_cases"] = len(receipt["cases"])
            frozen.write_json(output / "completion.json", receipt)
            print(f"{receipt['status']} {output / 'completion.json'}", flush=True)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
