#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fit F3 on one N=1 allocator receipt and validate independent peak and OOM receipts."""

import argparse
import hashlib
import json
import math
import sys
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "source"))

IDENTITY_FIELDS = ("device_uuid", "environment", "protocol", "source_sha", "memory_arm", "budget_bytes")
ERROR_LIMIT = 0.05


def read_receipt(path):
    data = path.read_bytes()
    return json.loads(data), {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}


def summarize_f0_capacity(anchor_path):
    anchor, reference = read_receipt(anchor_path)
    if anchor["status"] != "PASS" or anchor["reference_kind"] != "fresh_same_case_native_v4_full":
        raise ValueError("F0 context requires the verified fresh v4 full anchor")
    case_by_tag = {case["case_tag"]: case for case in anchor["cases"]}
    rows = []
    for relative, expected_hash in anchor["evidence_sha256"].items():
        path = STUDY / relative
        if path.name != "manifest.json":
            continue
        record, witness = read_receipt(path)
        case_tag = record["case_tag"]
        case = case_by_tag[case_tag]
        if (
            witness["sha256"] != expected_hash
            or record["input_sha256"] != case["input_sha256"]
            or record["trace_sha256"] != case["trace_sha256"]
        ):
            raise ValueError(f"F0 source binding differs for {path}")
        runtime = record["request"]["runtime_metadata"]
        graph = runtime["run_config"]["graph"]
        condition = runtime["run_config"]["condition_plan"]
        memory = runtime["memory"]
        rows.append(
            {
                "case_tag": case_tag,
                "input_sha256": case["input_sha256"],
                "trace_sha256": case["trace_sha256"],
                "receipt": witness,
                "peak_allocated_bytes": memory["max_memory_allocated"],
                "peak_reserved_bytes": memory["max_memory_reserved"],
                "kv_bytes": graph["template_reuse"]["pool"]["unique_kv_bytes"],
                "graph_reserved_bytes": sum(branch["capture_pool_bytes"] for branch in graph["branches"].values()),
                "condition_cache_bytes": condition["cache_bytes"],
                "condition_owned_bytes": condition["owned_bytes"],
                "position_owned_bytes": runtime["run_config"]["request_positions"]["owned_bytes"],
            }
        )
    if len(rows) != anchor["requests"]:
        raise ValueError("F0 anchor does not bind every measured request manifest")
    return {
        "receipt": reference,
        "scope": "native v4 serial worker memory and retained template, observed before request cleanup",
        "device_uuid": anchor["device_uuid"],
        "source_sha256": anchor["source_sha256"],
        "environment": anchor["environment"],
        "protocol": anchor["protocol"],
        "timing_boundary": anchor["timing_boundary"],
        "repetitions": anchor["repetitions"],
        "shared_allocated_bytes": None,
        "counts": {"requests": len(rows)},
        "rows": rows,
    }


def read_peaks(receipt):
    allocated = receipt["peak_allocated_bytes"]
    reserved = receipt["peak_reserved_bytes"]
    if type(allocated) is not int or type(reserved) is not int or not 0 < allocated <= reserved:
        raise ValueError("a successful run requires positive allocated and reserved allocator peaks")
    if receipt["synchronized_before_return"] is not True:
        raise ValueError("successful allocator peaks require a synchronized worker")
    return allocated, reserved


def fit_capacity(calibration):
    from vllm_omni.diffusion.models.flux.capacity_manager import CapacityManager, RequestFootprint

    if calibration["status"] != "PASS" or calibration["concurrency"] != 1:
        raise ValueError("calibration requires a completed N=1 execution")
    allocated, reserved = read_peaks(calibration)
    shared = calibration["shared_allocated_bytes"]
    shared_reserved = calibration["shared_reserved_bytes"]
    if type(shared) is not int or type(shared_reserved) is not int or not 0 < shared <= shared_reserved <= reserved:
        raise ValueError("invalid shared allocator costs")
    footprint_data, = calibration["footprints"]
    footprint_data = dict(footprint_data)
    workspace_kind = "measured"
    if "workspace_bytes" not in footprint_data:
        footprint_data["workspace_bytes"] = allocated - shared - sum(
            footprint_data[key] for key in ("kv_bytes", "activation_bytes", "graph_bytes")
        )
        workspace_kind = "N=1 peak residual after shared, KV, activation, and graph allocated bytes"
    footprint = RequestFootprint(**footprint_data)
    if footprint.graph_bytes and footprint.graph_reserved_bytes is None:
        raise ValueError("active graph requires allocator-pool reservation evidence")
    if allocated < shared + footprint.allocated_bytes:
        raise ValueError("request allocated costs exceed observed N=1 peak")
    allocator_slack = reserved - allocated - (shared_reserved - shared) - footprint.graph_slack_bytes
    manager = CapacityManager(
        shared_allocated_bytes=shared,
        shared_reserved_slack_bytes=shared_reserved - shared,
        per_request_reserved_slack_bytes=max(0, allocator_slack),
        budget_bytes=calibration["budget_bytes"],
    )
    return manager, footprint, workspace_kind


def validate_point(receipt, identity, case_bindings):
    for field, expected in identity.items():
        if receipt[field] != expected:
            raise ValueError(f"measurement {field} differs from the calibration")
    n = receipt["concurrency"]
    if type(n) is not int or n < 1 or len(receipt["cases"]) != n:
        raise ValueError("capacity measurements require one input and trace identity per requested slot")
    for case in receipt["cases"]:
        binding = (case["input_sha256"], case["trace_sha256"])
        tag = case["case_tag"]
        if case_bindings.setdefault(tag, binding) != binding:
            raise ValueError(f"case {tag} has inconsistent input or trace hashes")


def compare_point(manager, footprint, measurement, reference):
    n = measurement["concurrency"]
    actual_allocated, actual_reserved = read_peaks(measurement)
    predicted_allocated = manager.predict_allocated_bytes([footprint] * n)
    predicted_reserved = manager.predict_reserved_bytes([footprint] * n)
    return {
        "concurrency": n,
        "repetition": measurement["repetition"],
        "receipt": reference,
        "predicted_allocated_bytes": predicted_allocated,
        "predicted_reserved_bytes": predicted_reserved,
        "observed_allocated_bytes": actual_allocated,
        "observed_reserved_bytes": actual_reserved,
        "allocated_relative_error": abs(predicted_allocated - actual_allocated) / actual_allocated,
        "reserved_relative_error": abs(predicted_reserved - actual_reserved) / actual_reserved,
    }


def has_cuda_oom(receipt):
    failure = receipt.get("failure")
    return (
        receipt["status"] in ("OOM", "CAPACITY_BLOCKED")
        and isinstance(failure, dict)
        and failure.get("type") in ("OutOfMemoryError", "CUDAOutOfMemoryError")
        and "cuda out of memory" in failure.get("message", "").lower()
    )


def analyze_capacity(calibration_path, measurement_paths, f0_anchor=None):
    calibration, calibration_reference = read_receipt(calibration_path)
    identity = {field: calibration[field] for field in IDENTITY_FIELDS}
    case_bindings = {}
    validate_point(calibration, identity, case_bindings)
    manager, footprint, workspace_kind = fit_capacity(calibration)
    points, outcomes, oom_counts = [], [], []
    successful_counts = set()
    seen_receipts = {calibration_reference["sha256"]}
    for path in measurement_paths:
        measurement, reference = read_receipt(path)
        if reference["sha256"] in seen_receipts:
            raise ValueError("validation requires independent measurement receipts")
        seen_receipts.add(reference["sha256"])
        validate_point(measurement, identity, case_bindings)
        n = measurement["concurrency"]
        outcomes.append(
            {
                "receipt": reference,
                "concurrency": n,
                "repetition": measurement["repetition"],
                "status": measurement["status"],
                "failure": measurement.get("failure"),
                "peak_allocated_bytes": measurement.get("peak_allocated_bytes"),
                "peak_reserved_bytes": measurement.get("peak_reserved_bytes"),
                "cases": measurement["cases"],
            }
        )
        if measurement["status"] == "PASS":
            if n == 1 and measurement["repetition"] == calibration["repetition"]:
                raise ValueError("N=1 validation requires an independent repetition")
            points.append(compare_point(manager, footprint, measurement, reference))
            successful_counts.add(n)
        elif has_cuda_oom(measurement):
            oom_counts.append(n)

    missing = sorted({1, 2} - successful_counts)
    failed_error = any(
        max(row["allocated_relative_error"], row["reserved_relative_error"]) > ERROR_LIMIT for row in points
    )
    fixed_status = "FAIL" if failed_error else "BLOCKED" if missing else "PASS"
    predicted_n = manager.predict_max_concurrent(footprint, manager.budget_bytes)
    last_success = max(successful_counts, default=None)
    first_oom = min(oom_counts, default=None)
    if last_success is None:
        boundary_status = "BLOCKED"
    elif first_oom is None:
        boundary_status = "CONSERVATIVE_ONLY"
    elif last_success <= predicted_n < first_oom:
        boundary_status = "PASS"
    else:
        boundary_status = "FAIL"

    failure_types = {
        row["failure"].get("type") for row in outcomes
        if isinstance(row["failure"], dict)
    }
    if first_oom is not None:
        stop_reason = "CUDA_OOM"
    elif "AdmissionBudgetExceeded" in failure_types:
        stop_reason = "PREDICTED_BUDGET_CEILING"
    elif any(row["status"] != "PASS" for row in outcomes):
        stop_reason = "MEASUREMENT_FAILURE"
    else:
        stop_reason = "SCAN_LIMIT"

    statuses = (fixed_status, boundary_status)
    status = "FAIL" if "FAIL" in statuses else "PASS" if statuses == ("PASS", "PASS") else "BLOCKED"
    return {
        "status": status,
        "phase": "F3",
        "f0_context": summarize_f0_capacity(f0_anchor) if f0_anchor else None,
        **identity,
        "timing_boundary": calibration["timing_boundary"],
        "cases": calibration["cases"],
        "calibration": {
            "receipt": calibration_reference,
            "repetition": calibration["repetition"],
            "shared_allocated_bytes": manager.shared_allocated_bytes,
            "shared_reserved_slack_bytes": manager.shared_reserved_slack_bytes,
            "per_request_reserved_slack_bytes": manager.per_request_reserved_slack_bytes,
            "request_footprint": asdict(footprint),
            "workspace_kind": workspace_kind,
        },
        "fixed_n": {
            "status": fixed_status,
            "error_limit": ERROR_LIMIT,
            "missing_n": missing,
            "points": sorted(points, key=lambda item: (item["concurrency"], item["repetition"])),
        },
        "maximum_n": {
            "status": boundary_status,
            "last_successful_n": last_success,
            "first_oom_n": first_oom,
            "predicted_n": predicted_n,
            "stop_reason": stop_reason,
            "conservative_admission_n": max((n for n in successful_counts if n <= predicted_n), default=None),
        },
        "outcomes": outcomes,
        "counts": {"calibration": 1, "observations": len(outcomes), "success": len(points), "oom": len(oom_counts)},
    }


def plan_admitted_batches(manager, requests):
    batches, pending, admitted = [], [], []
    for request, footprint in requests:
        if not manager.can_admit(admitted, footprint):
            if not manager.can_admit([], footprint):
                raise ValueError("request cannot fit alone within the admission budget")
            batches.append(pending)
            pending, admitted = [], []
        pending.append(request)
        admitted.append(footprint)
    if pending:
        batches.append(pending)
    return batches


def sweep_inputs(observations, n, repetition):
    return [
        (case, {**identity, "request_id": f"f3-n{n}-r{repetition}-{position}-{identity['case_tag']}"})
        for position in range(n)
        for case, identity in (observations[position % len(observations)],)
    ]


def owned_cuda_bytes(tensors):
    """Count each underlying CUDA storage once even when CFG branches use views."""
    allocations = {}
    for tensor in tensors:
        if tensor is not None and tensor.is_cuda:
            storage = tensor.untyped_storage()
            allocations[storage.data_ptr()] = storage.nbytes()
    return sum(allocations.values())


def read_request_footprint(state):
    import torch
    from vllm_omni.diffusion.models.flux.specialized_cache import cache_allocation_bytes

    kv_storages = [
        tensor
        for cache in state.caches
        for layer in cache.layers.values()
        for tensor in (layer.key, layer.value)
    ]
    kv_bytes = owned_cuda_bytes(kv_storages)
    expected_kv_bytes = sum(
        cache.batch_size * cache_allocation_bytes(
            mode=cache.mode,
            text_token_count=cache.text_token_count,
            generated_token_count=cache.generated_token_count,
            reference_token_count=cache.reference_token_count,
            num_kv_heads=cache.num_kv_heads,
            head_dim=cache.head_dim,
            dtype=cache.dtype,
            num_layers=len(cache.layers),
        )
        for cache in state.caches
    )
    if kv_bytes <= 0 or expected_kv_bytes != kv_bytes:
        raise ValueError("observed private KV storage differs from the configured branch capacity")
    activation_bytes = owned_cuda_bytes(
        [state.latents, state.image_latents, state.timesteps, state.token_age, *state.conditioning.values(),
         *state.velocity_history]
    )
    graph_bytes, graph_reserved = 0, 0
    template = state.template
    if template is not None and template.graphs is not None:
        branches = (template.graphs.positive, template.graphs.negative)
        pools = {branch._graph.pool() for branch in branches if branch is not None and branch.graph_active}
        for pool in pools:
            segments = torch.cuda.memory_snapshot(mempool_id=pool, include_traces=False)
            graph_bytes += sum(segment["allocated_size"] for segment in segments)
            graph_reserved += sum(segment["total_size"] for segment in segments)
    return {
        "kv_bytes": kv_bytes,
        "activation_bytes": activation_bytes,
        "graph_bytes": graph_bytes,
        "graph_reserved_bytes": graph_reserved,
    }


def classify_execution_error(error):
    import traceback
    import torch

    causes = set()
    current = error
    while current is not None and id(current) not in causes:
        causes.add(id(current))
        if isinstance(current, torch.cuda.OutOfMemoryError) and "cuda out of memory" in str(current).lower():
            return "OOM", {
                "type": type(current).__name__,
                "message": str(current),
                "traceback": "".join(traceback.format_exception(error)),
            }
        current = current.__cause__ or current.__context__
    return "FAILED", {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }


def collect_group(bench, pipeline, post_process, inputs, identity, repetition, shared, measured):
    """Collect whole-request peaks while the native request executes on private streams."""
    import torch
    from admission import validate_trace_application
    from vllm_omni.diffusion.models.flux.concurrent_scheduler import ConcurrentScheduler
    from vllm_omni.diffusion.models.flux.request_context import RequestContext
    from vllm_omni.diffusion.models.flux.request_executor import RequestExecutor

    footprints = []
    class CapacityExecutor(RequestExecutor):
        def finish(self, state):
            if measured:
                footprints.append(read_request_footprint(state))
            return super().finish(state)

    executor = CapacityExecutor(pipeline, post_process=post_process, capture_steps=False)
    states = []
    receipt = {
        **identity,
        **shared,
        "status": "FAILED",
        "synchronized_before_return": False,
        "concurrency": len(inputs),
        "repetition": repetition,
        "cases": [case for _, case in inputs],
        "failure": None,
    }
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        with ExitStack() as source_images:
            for case, case_identity in inputs:
                image = source_images.enter_context(bench.resize_case(case, identity["protocol"]))
                request = bench.make_request(
                    case, "full", identity["protocol"], image=image, lane=identity["memory_arm"],
                )
                request_id = case_identity.get("request_id", case_identity["case_tag"])
                request.requests[0].request_id = request_id
                stream = torch.cuda.Stream()
                context = RequestContext(request_id, 0, None, {})
                states.append(executor.init_state(request, context, stream=stream))
            results = ConcurrentScheduler(executor).run(states)
            torch.cuda.synchronize()
        for (bound_case, _), result in zip(inputs, results, strict=True):
            metadata = result.metadata
            validate_trace_application(metadata, bound_case[3])
            graph_active = metadata["summary"]["graph_active"]
            if graph_active != (identity["memory_arm"] == "private_graph"):
                raise ValueError("achieved graph/eager arm differs from requested capacity arm")
        receipt["status"] = "PASS"
        receipt["synchronized_before_return"] = True
        if measured:
            for footprint, result in zip(footprints, results, strict=True):
                config = result.metadata["run_config"]
                footprint["activation_bytes"] += (
                    config["condition_plan"].get("owned_bytes", 0) + config["request_positions"].get("owned_bytes", 0)
                )
            receipt["footprints"] = footprints
    except Exception as error:
        receipt["status"], receipt["failure"] = classify_execution_error(error)
        for state in states:
            if state.continuation is not None:
                try:
                    executor.abort(state)
                except Exception as cleanup_error:
                    receipt["failure"].setdefault("cleanup_errors", []).append(str(cleanup_error))
    receipt["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    receipt["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    return receipt


def run_capacity_sweep(bench, pipeline, post_process, observations, identity, output, args):
    import gc
    import torch
    from vllm_omni.diffusion.models.flux.capacity_manager import CapacityManager

    def preserve(name, receipt):
        path = output / name / "completion.json"
        path.parent.mkdir()
        write_receipt(path, receipt)
        print(f"F3 {name} {receipt['status']}", flush=True)
        return path

    def collect(inputs, repetition, name, shared, measured=False):
        gc.collect()
        torch.cuda.empty_cache()
        receipt = collect_group(bench, pipeline, post_process, inputs, identity, repetition, shared, measured)
        return preserve(name, receipt), receipt

    def allocator_baseline():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return {
            "shared_allocated_bytes": int(torch.cuda.memory_allocated()),
            "shared_reserved_bytes": int(torch.cuda.memory_reserved()),
            "free_bytes": int(torch.cuda.mem_get_info()[0]),
        }

    # One independent request primes kernels. Its private cache and templates
    # are released by RequestExecutor before recording the shared baseline.
    cold = allocator_baseline()
    _, warmup = collect(observations[:1], 0, "warmup", cold)
    if warmup["status"] != "PASS":
        return {
            **identity, "status": "BLOCKED", "failure": "Independent capacity warmup failed.",
            "warmup": str(output / "warmup/completion.json"), "fixed_n": {"status": "BLOCKED"},
            "maximum_n": {"status": "BLOCKED"},
        }
    shared = allocator_baseline()
    physical_budget = shared["free_bytes"] + shared["shared_reserved_bytes"]
    logical_budget = args.sweep_budget_gib
    shared["physical_budget_bytes"] = physical_budget
    shared["budget_bytes"] = min(physical_budget, int(logical_budget * 2**30)) if logical_budget else physical_budget
    calibration_path, calibration = collect(sweep_inputs(observations, 1, 0), 0, "fit_n1", shared, measured=True)
    if calibration["status"] != "PASS":
        return {
            **identity, **shared, "status": "BLOCKED", "failure": "Capacity calibration request failed.",
            "calibration": str(calibration_path), "fixed_n": {"status": "BLOCKED"},
            "maximum_n": {"status": "BLOCKED"},
        }
    manager, footprint, _ = fit_capacity(calibration)

    # Use a logical budget equal to the fitted single-request reservation to
    # exercise admission and FIFO fallback on the same case objects and device.
    fallback_manager = CapacityManager(
        shared_allocated_bytes=manager.shared_allocated_bytes,
        shared_reserved_slack_bytes=manager.shared_reserved_slack_bytes,
        per_request_reserved_slack_bytes=manager.per_request_reserved_slack_bytes,
        budget_bytes=manager.predict_peak_bytes([footprint]),
    )
    queue = plan_admitted_batches(fallback_manager, [(item, footprint) for item in observations[:2]])
    held_out_paths = []
    held_out_pass = True
    for index, inputs in enumerate(queue, start=1):
        path, receipt = collect(inputs, index, f"holdout_n1_r{index}", shared)
        held_out_paths.append(path)
        held_out_pass = held_out_pass and receipt["status"] == "PASS"
    admission = {
        "status": "PASS" if len(queue) == 2 and held_out_pass else "FAIL",
        "planning_budget_bytes": fallback_manager.budget_bytes,
        "enqueued_case_tags": [item[1]["case_tag"] for item in observations[:2]],
        "admitted": [[item[1]["case_tag"] for item in batch] for batch in queue],
        "receipts": [read_receipt(path)[1] for path in held_out_paths],
    }
    # The physical sweep bypasses admission on purpose to observe the first
    # actual CUDA capacity failure. Keep each point including the failure.
    for n in range(2, args.max_concurrency + 1):
        inputs = sweep_inputs(observations, n, 0)
        name = f"sweep_n{n}"
        if logical_budget is not None and manager.predict_peak_bytes([footprint] * n) > manager.budget_bytes:
            held_out_paths.append(preserve(name, {
                **identity,
                **shared,
                "status": "CAPACITY_BLOCKED",
                "concurrency": n,
                "repetition": 0,
                "cases": [case for _, case in inputs],
                "failure": {
                    "type": "AdmissionBudgetExceeded",
                    "message": "Predicted reservation exceeds the configured sweep budget.",
                },
            }))
            break
        path, receipt = collect(inputs, 0, name, shared)
        held_out_paths.append(path)
        if receipt["status"] != "PASS":
            break
    return calibration_path, held_out_paths, admission


def write_receipt(path, receipt):
    with path.open("x") as destination:
        destination.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def collect_capacity(args, output):
    import fcntl
    import os
    import socket
    import bench_concurrency as bench
    from run_baseline import load_cases, verify_hashes, verify_model_identity

    protocol, identities = bench.load_protocol_cases()
    identity = {
        "phase": "F3",
        "device_uuid": args.gpu_uuid,
        "host": socket.gethostname(),
        "protocol": protocol,
        "source_sha": bench.source_tree_sha(),
        "environment": bench._environment(),
        "memory_arm": args.memory_arm,
        "timing_boundary": "worker allocator peak before request preparation through synchronized postprocessing",
        "validation_scope": "capacity-only diagnostic, independent of the F1 numerical and F2 timing gates",
    }

    if Path(sys.executable) != bench.PYTHON:
        raise RuntimeError(f"F3 requires the fixed Python at {bench.PYTHON}")
    # Use the same file as F1/F2/F5. The compute-process preflight runs under
    # the UUID lock before setting CUDA visibility and loading the worker.
    with Path(f"/tmp/sparseedit-{args.gpu_uuid}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity.update(bench._check_measurement_device(args.gpu_uuid))
        if args.require_gpu_name and args.require_gpu_name.lower() not in identity["device_name"].lower():
            raise ValueError(f"Selected device is {identity['device_name']}, requires {args.require_gpu_name}")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        bundle = bench.read_json(bench.BUNDLE_ROOT / "bundle.json")
        if args.verify_full_model_hashes:
            verify_hashes(bench.MODEL, bundle["model_hashes"])
            identity["model_identity"] = {
                "method": "current full SHA-256 matches all frozen v4 model files",
                "model_path": str(bench.MODEL.resolve()),
                "files": len(bundle["model_hashes"]),
            }
        else:
            identity["model_identity"] = verify_model_identity(bundle["model_hashes"])
        bound_cases = [(case, identities[case[0]["case_tag"]]) for case in load_cases(bench.BUNDLE_ROOT, bundle)]
        with bench.load_pipeline(bench.MODEL) as (pipeline, post_process):
            collection = run_capacity_sweep(bench, pipeline, post_process, bound_cases, identity, output, args)
    if isinstance(collection, dict):
        return collection
    calibration_path, measurement_paths, admission = collection
    report = analyze_capacity(calibration_path, measurement_paths, args.f0_anchor)
    report["admission"] = admission
    report["validation_scope"] = identity["validation_scope"]
    report["model_identity"] = identity["model_identity"]
    report["physical_budget_bytes"] = read_receipt(calibration_path)[0]["physical_budget_bytes"]
    if admission["status"] != "PASS":
        report["status"] = "FAIL"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0-anchor", type=Path, default=STUDY / "artifacts/f0_anchor_v1/completion.json")
    parser.add_argument("--gpu-uuid", help="collect the memory-only F3 sweep on this UUID")
    parser.add_argument("--memory-arm", choices=("private_graph", "private_eager"), default="private_graph")
    parser.add_argument("--require-gpu-name", help="require this device model substring before worker start")
    parser.add_argument("--verify-full-model-hashes", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=3, metavar="2..16")
    parser.add_argument("--sweep-budget-gib", type=float, help="stop before exceeding this conservative allocator budget")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--measurement", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=STUDY / "artifacts/capacity_v1")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(STUDY / "artifacts"):
        parser.error("choose an output directory under this study's artifacts")
    if args.gpu_uuid and (args.calibration or args.measurement):
        parser.error("collect a fresh sweep or analyze existing measurements")
    if args.measurement and not args.calibration:
        parser.error("measurements require an N=1 calibration receipt")
    if not 2 <= args.max_concurrency <= 16:
        parser.error("--max-concurrency must be at least 2 and at most 16")
    if args.sweep_budget_gib is not None and (not math.isfinite(args.sweep_budget_gib) or args.sweep_budget_gib <= 0):
        parser.error("--sweep-budget-gib must be a positive finite budget")
    output_root.mkdir(parents=True, exist_ok=not bool(args.gpu_uuid))
    if args.gpu_uuid:
        try:
            receipt = collect_capacity(args, output_root)
        except Exception as error:
            import traceback

            receipt = {
                "status": "BLOCKED" if isinstance(error, BlockingIOError) else "FAILED",
                "phase": "F3",
                "device_uuid": args.gpu_uuid,
                "memory_arm": args.memory_arm,
                "failure": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
            }
            traceback.print_exc()
    elif args.calibration:
        receipt = analyze_capacity(args.calibration, args.measurement, args.f0_anchor)
    else:
        context = summarize_f0_capacity(args.f0_anchor)
        receipt = {
            "status": "BLOCKED",
            "phase": "F3",
            "f0_context": context,
            "failure": "Needs same-arm shared allocator costs and isolated request peaks under the frozen protocol.",
            "fixed_n": {"status": "BLOCKED", "missing_n": [1, 2], "points": []},
            "maximum_n": {"status": "BLOCKED", "last_successful_n": None, "first_oom_n": None, "predicted_n": None},
        }
    receipt["finished_at_utc"] = datetime.now(UTC).isoformat()
    write_receipt(output_root / "completion.json", receipt)
    print(f"F3 {receipt['status']}: {output_root / 'completion.json'}")
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
