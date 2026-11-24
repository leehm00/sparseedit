#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe request-private graph and eager templates after F1 passes."""

import argparse
from datetime import UTC, datetime
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import statistics
import sys
from time import perf_counter
import traceback
from types import SimpleNamespace

from bench_concurrency import (
    BUNDLE_ROOT, MODEL, STUDY, calibration_pairs, load_pipeline, load_protocol_cases, make_request, resize_case,
    source_tree_sha,
)
from check_env import query
from run_baseline import load_cases, load_json, verify_hashes, verify_model_identity
from run_isolation import exception_status, serial, validate_lane, write_output

TIMING = "Synchronized encoding through image postprocessing, excluding PNG writing and independent warmup."


def f1_admission(root, expected, protocol, device):
    lanes = {}
    for path in root.rglob("completion.json"):
        row = load_json(path)
        if row.get("phase") != "F1" or row.get("status") != "PASS" or row.get("device_uuid") != device:
            continue
        if (row.get("source_sha") != source_tree_sha() or row.get("protocol") != protocol
                or row.get("correctness", {}).get("all_match") is not True or len(row.get("cases", [])) != 2):
            continue
        tags = frozenset(case["case_tag"] for case in row["cases"])
        if tags in expected:
            lanes.setdefault(row["runtime_lane"], {})[tags] = sha256(path.read_bytes()).hexdigest()
    for lane, pairs in lanes.items():
        if set(pairs) == expected:
            return {"lane": lane, "pair_receipt_sha256": list(pairs.values())}
    raise ValueError("F4 requires all three F1 PASS pairs for this source, protocol and GPU")


def separate_k_signature(k):
    from vllm_omni.diffusion.models.flux.specialized_template import acquire_template, clear_template

    owner = SimpleNamespace()
    first, first_hit = acquire_template(owner, ("K", k), capacity=2)
    first.finish(success=True)
    second, second_hit = acquire_template(owner, ("K", k + 1), capacity=2)
    second.finish(success=True)
    repeated, repeated_hit = acquire_template(owner, ("K", k), capacity=2)
    if first is second or first_hit or second_hit or repeated is not first or not repeated_hit:
        raise RuntimeError("Template K signatures do not have distinct cache entries")
    repeated.finish(success=True)
    clear_template(owner)
    return {
        "status": "PASS", "signatures": [k, k + 1], "hit_sequence": [False, False, True],
        "scope": "host-side template keys; all GPU requests use the frozen K",
    }


def latency_detail(runtime):
    records = runtime["step_records"]
    sparse = [row["step"] for row in records if row["is_sparse_step"]]
    values = runtime["timing"]["step_forward_device_seconds"]
    graph = runtime["run_config"].get("graph") or {}
    return {
        "first_sparse_step_seconds": values[sparse[0]],
        "steady_sparse_step_seconds": statistics.median(values[index] for index in sparse[1:]),
        "capture_count": sum(branch["capture_trials"] for branch in graph.get("branches", {}).values()),
        "template_hit": (graph.get("template_reuse") or {}).get("cache_hit"),
        "template_pool": (graph.get("template_reuse") or {}).get("pool"),
        "fallback": graph.get("fallback"),
    }


def run_lane(lane, root, protocol, cases, pipeline, post_process, native_reuse):
    import torch
    from vllm_omni.diffusion.models.flux.concurrent_scheduler import ConcurrentScheduler
    from vllm_omni.diffusion.models.flux.request_context import RequestContext
    from vllm_omni.diffusion.models.flux.request_executor import RequestExecutor
    from vllm_omni.diffusion.models.flux.specialized_template import clear_template

    executor = RequestExecutor(pipeline, post_process=post_process)
    images = [resize_case(case, protocol) for case in cases]
    tags = [case[0]["case_tag"] for case in cases]
    matches = {}
    serial_requests = []
    groups = ((0, 0), (0, 1))
    warm = serial(executor, make_request(cases[0], "full", protocol, image=images[0], lane=lane), tags[0])
    warmup_seconds = warm.timing["request_seconds"]
    del warm
    for index, (case, image, tag) in enumerate(zip(cases, images, tags, strict=True)):
        candidate = serial(executor, make_request(case, "full", protocol, image=image, lane=lane), tag)
        validate_lane(candidate.metadata, lane, case[3])
        png = write_output(candidate.output, root / f"serial_{index}.png")
        matches[index] = (candidate.latents, png)
        serial_requests.append({"case_tag": tag, **candidate.timing, **latency_detail(candidate.metadata)})
    batches = []
    for group in groups:
        states = []
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        try:
            for ordinal, index in enumerate(group):
                request = make_request(cases[index], "full", protocol, image=images[index], lane=lane)
                context = RequestContext(f"{tags[index]}_{ordinal}", -1, None, {})
                states.append(executor.init_state(request, context, stream=torch.cuda.Stream()))
            scheduler = ConcurrentScheduler(executor)
            output = scheduler.run(states)
            torch.cuda.synchronize()
            wall = perf_counter() - started
        finally:
            for state in states:
                executor.abort(state)
        rows = []
        for ordinal, (index, result) in enumerate(zip(group, output, strict=True)):
            validate_lane(result.metadata, lane, cases[index][3])
            png = write_output(result.output, root / f"pair{len(batches)}_{ordinal}.png")
            expected_latent, expected_png = matches[index]
            rows.append({
                "case_tag": tags[index], "exact": png == expected_png and torch.equal(expected_latent, result.latents),
                "output_png_sha256": png, **result.timing, **latency_detail(result.metadata),
            })
        batches.append({
            "case_tags": [tags[i] for i in group], "batch_wall_seconds": wall,
            "synchronized_before_return": True, "requests": rows,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
    if lane == "private_graph":
        try:
            for ordinal, index in enumerate((0, 0, 1)):
                request = make_request(cases[index], "full", protocol, image=images[index], lane=lane)
                torch.cuda.synchronize()
                started = perf_counter()
                output = post_process(pipeline.forward(request).output)
                torch.cuda.synchronize()
                wall = perf_counter() - started
                metadata = pipeline._last_flux_acceleration_metadata
                validate_lane(metadata, lane, cases[index][3])
                png = write_output(output, root / f"persistent_serial_{ordinal}.png")
                native_reuse.append({
                    "case_tag": tags[index], "request_seconds": wall, "png_equal_to_private_serial": png == matches[index][1],
                    **latency_detail(metadata),
                })
        finally:
            clear_template(pipeline)
    return {
        "status": "PASS" if all(row["exact"] for batch in batches for row in batch["requests"])
        and all(row["png_equal_to_private_serial"] for row in native_reuse) else "FAILED",
        "independent_warmup_seconds": warmup_seconds,
        "serial_requests": serial_requests, "batches": batches, "persistent_serial_reference": native_reuse,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--isolation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-full-model-hashes", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not root.is_relative_to(STUDY / "artifacts"):
        parser.error("Select a fresh path inside study artifacts")
    root.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "BLOCKED", "phase": "F4", "device_uuid": args.gpu_uuid, "host": socket.gethostname(),
        "source_sha": source_tree_sha(), "repetition": 0, "timing_boundary": TIMING, "arms": {},
    }
    try:
        versions = {"python": sys.version.split()[0]}
        from importlib.metadata import version

        versions.update({name: version(name) for name in ("torch", "triton", "vllm")})
        protocol, identities = load_protocol_cases()
        receipt.update(environment=versions, protocol=protocol, cases=list(identities.values()))
        receipt["f1_gate"] = f1_admission(args.isolation_root, {
            frozenset(pair) for pair in calibration_pairs(tuple(identities))
        }, protocol, args.gpu_uuid)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        with Path(f"/tmp/sparseedit-{args.gpu_uuid}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if query(args.gpu_uuid, "compute-apps", "gpu_uuid,pid,used_memory"):
                raise RuntimeError("GPU has an external compute job")
            bundle = load_json(BUNDLE_ROOT / "bundle.json")
            if args.verify_full_model_hashes:
                verify_hashes(MODEL, bundle["model_hashes"])
            else:
                verify_model_identity(bundle["model_hashes"])
            cases = load_cases(BUNDLE_ROOT, bundle)[:2]
            receipt["k_signature_probe"] = separate_k_signature(protocol["K"])
            receipt["structure_only"] = {
                "status": "UNSUPPORTED",
                "reason": "Captured branch graphs bind concrete conditioning and KV addresses. Each request owns its graph pool.",
            }
            import torch

            with load_pipeline(MODEL) as (pipeline, post_process), torch.inference_mode():
                for lane in ("private_eager", "private_graph"):
                    lane_root = root / lane
                    lane_root.mkdir()
                    try:
                        receipt["arms"][lane] = run_lane(lane, lane_root, protocol, cases, pipeline, post_process, [])
                    except Exception as error:
                        status, failure = exception_status(error)
                        receipt["arms"][lane] = {"status": status, "failure": failure or str(error)}
                        (lane_root / "traceback.log").write_text(traceback.format_exc())
                        torch.cuda.empty_cache()
            serial_graph = receipt["arms"]["private_graph"].get("persistent_serial_reference", [])
            if len(serial_graph) >= 2:
                cold, repeated = serial_graph[:2]
                receipt["serial_pool_cold_to_warm_speedup"] = cold["request_seconds"] / repeated["request_seconds"]
                receipt["serial_pool_repeated_case_hit"] = repeated["template_hit"]
            receipt["pool_conclusion"] = (
                "Current graph pools amortize capture within a serial owner. "
                "Concurrent requests require independent graph and KV owners with this capture layout."
            )
            states = {row["status"] for row in receipt["arms"].values()}
            receipt["status"] = "PASS" if states == {"PASS"} else "PARTIAL"
    except Exception as error:
        receipt["failure"] = str(error)
        (root / "traceback.log").write_text(traceback.format_exc())
    finally:
        receipt["finished_at_utc"] = datetime.now(UTC).isoformat()
        (root / "completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
