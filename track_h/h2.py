#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Track H2 single-request FLUX resolution and reference KV experiment."""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import traceback

from h2_runtime import flux_native as flux
from h4 import capture_features

ARMS = flux.ARMS
SIDES = (768, 1024, 1536)
CAL3_RATIOS = (.1, .2, .35)
SEED = flux.SEED
WALL_BOUNDARY = "synchronized native Omni.generate, including encoding, preparation, denoising and VAE decode"


@dataclass(frozen=True)
class Point:
    side: int
    ratio: float
    partition: str
    cases: tuple
    repetitions: int

    @property
    def directory(self):
        return f"{self.partition}/{self.side}/k{round(self.ratio * 100):02d}"


def experiment_points(bundle, run_mode, representative_sources):
    if run_mode not in ("cal3", "formal"):
        raise ValueError("run_mode must be cal3 or formal")
    if type(representative_sources) is not int or representative_sources not in (24, 48):
        raise ValueError("representative_sources must be 24 or 48")
    cohort = bundle["cases"]
    points = [
        Point(side, ratio, "calibration", tuple(cohort["calibration"]), 1)
        for side in SIDES for ratio in CAL3_RATIOS
    ]
    if run_mode == "formal":
        cases = tuple(cohort["test"][:representative_sources])
        repetitions = 2 if representative_sources == 48 else 1
        points.extend(Point(side, .2, "formal", cases, repetitions) for side in SIDES)
    return points


def clone_traffic_bytes(side):
    """Model the full K and V clone read and write across CFG4's two branches."""
    context = 512 + 2 * flux.token_count(side)
    return context * 57 * 24 * 128 * 2 * 2 * 2 * 2


def projected_int8_saved_bytes(side):
    """Reference K and V 128-wide BF16 rows become INT8 rows plus FP16 scales."""
    return flux.token_count(side) * 57 * 24 * 2 * 2 * (128 - 2)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(seconds):
    return type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0


def validate_execution(md, point, arm, trace):
    summary = md["summary"]
    config = md["run_config"]
    records = md.get("step_records", [])
    pipeline = md["pipeline_timing"]
    if arm == "fresh_dense":
        _require(md["method"] == "dense" and summary.get("dense_steps") == 28, "expected Fresh Dense")
        _require(not md.get("dense_graph"), "Fresh Dense unexpectedly used a dense graph")
        return {
            "dense_steps": 28, "sparse_steps": 0, "reuse_steps": 0,
            "denoise_wall_s": pipeline.get("denoise_wall_seconds"),
            "encoding_device_s": pipeline.get("encoding_s"), "decode_device_s": pipeline.get("decode_s"),
        }
    _require(md["method"] == "sparseedit" and len(records) == 28, "expected 28 SparseEdit forward records")
    expected_summary = {"dense_steps": 9, "sparse_steps": 19, "transformer_skipped_steps": 0}
    _require(all(summary.get(k) == v for k, v in expected_summary.items()), "unexpected fixed denoising schedule")
    n = flux.token_count(point.side)
    k = int(n * point.ratio)
    expected_config = {
        "generated_token_count": n, "reference_token_count": n, "active_token_count": k,
        "latent_dtype": "torch.bfloat16", "true_cfg": True, "true_cfg_scale": 4.0,
    }
    _require(all(config.get(key) == value for key, value in expected_config.items()), "actual FLUX shape/CFG differs")
    cfg_execution = config["cfg_execution_plan"]["selected"]
    _require(cfg_execution == ("batched" if arm == "joint_cfg" else "serial"), "actual CFG execution differs")
    replay = config.get("replay") or {}
    if trace is None:
        _require(not replay.get("enabled"), "trace witness unexpectedly used replay")
        trace = flux.validate_trace(md["fixed_mask_trace"], point.side, point.ratio)
        _require(md["fixed_mask_trace_sha256"] == trace["sha256"], "reported recorded trace digest differs")
    else:
        _require(replay.get("enabled") and replay["sha256"] == trace["sha256"], "reported replay digest differs")
    for actual, expected in zip(records, trace["steps"], strict=True):
        _require(all(actual.get(key) == expected.get(key) for key in ("step", "phase", "kv_action", "indices")),
                 "the executed token or KV trace differs")
        _require(actual.get("cfg_branch_count") == 2 and actual.get("transformer_computed") is True,
                 "the executed CFG branches or compute step differs")
        if expected["phase"] == "sparse":
            _require(actual["active_token_count"] == k, "the executed sparse step has a different token budget")
        else:
            _require(actual["reference_policy"] == "full", "the executed dense step changed the refresh policy")
    graph = config.get("graph")
    persistent_kv_bytes = summary.get("persistent_kv_bytes")
    if arm in ("full", "joint_cfg", "segmented_int8"):
        branches = 2 if arm == "joint_cfg" else 1
        _require(
            isinstance(graph, dict) and graph["graph_active"] and graph["fallback"] is False
            and graph["cfg_branches_per_replay"] == branches and graph["total_replays"] == 38 // branches,
            "expected actual sparse graph replays with the requested CFG branching",
        )
        if persistent_kv_bytes is None and arm == "full":
            persistent_kv_bytes = (graph.get("template_reuse") or {}).get("pool", {}).get("unique_kv_bytes")
        _require(type(persistent_kv_bytes) is int and persistent_kv_bytes > 0,
                 "native graph run has no measured persistent KV bytes")
    elif arm == "direct_eager":
        _require(config["sparseedit_direct_ops"] == "eager" and summary["submission_backend"] == "baseline_clone",
                 "expected direct eager with clone/scatter")
    elif arm == "standard_fastops_eager":
        backend = md["standard_backend"]
        _require(
            backend.get("status") == "PASS" and backend.get("candidate") == arm and backend.get("fallback") is False
            and len(backend.get("calls", [])) == 56 and len(backend.get("scheduler_checks", [])) == 28,
            "expected all 56 standard fastops forward calls and 28 scheduler checks",
        )
    if arm == "segmented_int8":
        backends = set(summary["actual_backends"])
        _require({"int8_reference_storage", "triton_split_kv_online_int8"} <= backends,
                 "expected both packed INT8 reference storage and Triton split-KV attention")
    elif arm in ("full", "joint_cfg"):
        _require("int8_reference_storage" not in summary["actual_backends"], "expected BF16 reference storage")
    timing = md["timing"]
    intervals = timing["step_forward_device_seconds"]
    _require(len(intervals) == 28 and all(_positive(s) for s in intervals), "missing synchronized forward intervals")
    full = [intervals[i] for i in flux.DENSE_STEPS]
    sparse = [intervals[i] for i in flux.SPARSE_STEPS]
    return {
        "dense_steps": len(full), "sparse_steps": len(sparse), "reuse_steps": 0,
        "cfg_execution": cfg_execution, "cfg_execution_plan": config["cfg_execution_plan"],
        "persistent_kv_bytes": persistent_kv_bytes,
        "graph": graph, "actual_backends": summary.get("actual_backends", []),
        "replay_trace_payload_sha256": trace["sha256"],
        "full_device_s": statistics.mean(full),
        "steady_sparse_device_s": statistics.mean(sparse[1:]),
        "full_device_total_s": sum(full), "sparse_device_total_s": sum(sparse),
        "first_sparse_device_s": sparse[0],
        "step_forward_device_seconds": intervals,
        "forward_device_scope": "native CUDA events around each denoiser forward step",
        "denoise_wall_s": timing["denoise_wall_seconds"],
        "denoise_device_s": timing["denoise_device_seconds"],
        "encoding_device_s": pipeline.get("encoding_s"), "decode_device_s": pipeline.get("decode_s"),
    }


def _row_identity(point, case, arm, repetition, warmup):
    return {
        "partition": point.partition, "height": point.side, "width": point.side,
        "active_ratio": point.ratio, "active_tokens": int(flux.token_count(point.side) * point.ratio),
        "case_tag": case["case_tag"], "source_id": case["source_pixel_sha256"],
        "input_filename": case["input_filename"], "input_sha256": case["prepared_sha256"],
        "instruction": case["instruction"], "seed": SEED, "batch_size": 1,
        "arm": arm, "repetition": repetition, "warmup": warmup,
    }


def _reason(error):
    message = str(error)
    traceback_message = " ".join(traceback.format_exception(error))
    oom = "out of memory" in traceback_message.lower() or "OutOfMemoryError" in traceback_message
    return {
        "failure_kind": "out_of_memory" if oom else "native_worker_or_execution_error",
        "error_type": type(error).__name__, "error": message, "traceback": traceback_message,
    }


def _record(path, row):
    flux.write_json(path / "manifest.json", row)
    print(row["partition"], row["height"], row["active_ratio"], row["arm"], row["case_tag"], row["repetition"],
          "warmup" if row["warmup"] else "request", row["status"], row.get("request_wall_s", row.get("error", "")),
          flush=True)
    return row


def _run_cell(
    worker, flux_assets, path, point, case, arm, repetition, *, warmup=False, trace_path=None,
    trace_origin=None, trace_reference=None, witness_arm=None, witness_output_digest=None,
):
    path.mkdir(parents=True)
    run_arm = witness_arm or arm
    row = _row_identity(point, case, arm, repetition, warmup)
    if trace_path is not None:
        row.update(
            trace_file_sha256=flux.sha256(trace_path), trace_origin=trace_origin,
            trace_reference=trace_reference,
        )
    try:
        trace = flux.read_json(trace_path) if trace_path is not None else None
        method, extra = flux.request_options(
            run_arm, point.ratio, trace_path, side=point.side, collect_trace=witness_arm is not None
        )
        png = None if warmup else path / "output.png"
        result = flux.run_native(
            worker, flux_assets / "inputs" / case["input_filename"], case["instruction"],
            point.side, method, extra, output=png,
        )
        md = result["runtime_metadata"]
        _require(isinstance(md, dict), "missing native FLUX request metadata")
        _require(_positive(result["request_wall_seconds"]), "invalid synchronized native request latency")
        memory = md["complete_request_peak_memory"]
        _require(
            all(type(memory.get(key)) is int and memory[key] > 0
                for key in ("max_memory_allocated", "max_memory_reserved")),
            "invalid worker allocator peak memory",
        )
        steps_path = path / "step_records.json"
        if md.get("step_records"):
            flux.write_json(steps_path, md["step_records"])
            row["step_records_sha256"] = flux.sha256(steps_path)
        evidence = validate_execution(md, point, run_arm, trace)
        row.update(
            status="PASS", request_wall_s=result["request_wall_seconds"], memory=memory,
            worker_denoise_memory=md.get("memory"), pipeline_timing=md["pipeline_timing"],
            input_resize_s=result["input_resize_s"], model_load_s=result["model_load_seconds"],
            prepared_input_pixel_sha256=result["prepared_input_pixel_sha256"],
            gpu_before=result["gpu_before"], gpu_after=result["gpu_after"],
            output_png=png.name if png else None, output_sha256=flux.sha256(png) if png else None,
            **evidence,
        )
        if witness_output_digest:
            row["trace_witness_output_sha256"] = witness_output_digest
            row["trace_png_match"] = row["output_sha256"] == witness_output_digest
        if witness_arm is not None:
            row["trace_origin"] = f"native_serial_selector_{witness_arm}"
        if witness_arm is not None and not warmup:
            flux.write_json(path / "recorded_trace.json", md["fixed_mask_trace"])
            row["trace_file_sha256"] = flux.sha256(path / "recorded_trace.json")
            row["trace_reference"] = "recorded_trace.json"
        if arm == "segmented_int8_capacity_probe":
            row.update(
                scope="unpaired 1536px reference KV capacity; serial Full-v4 trace unavailable",
                projected_bf16_persistent_kv_bytes=clone_traffic_bytes(point.side) // 2,
                projected_persistent_kv_saved_bytes=projected_int8_saved_bytes(point.side),
            )
    except flux.GPUIsolationError:
        raise
    except Exception as error:
        row.update(status="FAIL", **_reason(error))
    return _record(path, row)


def _unavailable(root, point, case, arm, rep, *, reason):
    path = root / arm / f"rep{rep}" / case["case_tag"]
    path.mkdir(parents=True)
    row = _row_identity(point, case, arm, rep, False)
    row.update(status="UNAVAILABLE", **reason)
    encoded_path = str(path.relative_to(root) / "manifest.json")
    return dict(_record(path, row), point_manifest=encoded_path)


def _prepare_traces(flux_assets, point, root, model, gpu_uuid):
    traces = {}
    if point.side == 768 and point.ratio == .2:
        for case in point.cases:
            tag = case["case_tag"]
            path = root / "frozen_traces" / f"{tag}.json"
            path.parent.mkdir(exist_ok=True)
            shutil.copyfile(flux_assets / "traces" / path.name, path)
            traces[tag] = (path, None, "frozen_track_a_source")
        return traces
    witness_root = root / "serial_full_trace_witness"
    witness_root.mkdir()
    try:
        with flux.native_worker(model, gpu_uuid=gpu_uuid, profile=True) as worker:
            first = point.cases[0]
            warmup = _run_cell(
                worker, flux_assets, witness_root / "warmup", point, first, "trace_witness", 0,
                warmup=True, witness_arm="full",
            )
            if warmup["status"] != "PASS":
                raise RuntimeError(f"serial Full-v4 trace warmup failed: {warmup['error']}")
            for case in point.cases:
                tag = case["case_tag"]
                path = witness_root / tag
                row = _run_cell(worker, flux_assets, path, point, case, "trace_witness", 0, witness_arm="full")
                if row["status"] == "PASS":
                    traces[tag] = (path / "recorded_trace.json", row["output_sha256"], "native_serial_full_v4")
    except flux.GPUIsolationError:
        raise
    except Exception as error:
        flux.write_json(witness_root / "worker_failure.json", _reason(error))
    return traces


def _standalone_capacity(flux_assets, point, root, traces, model, gpu_uuid):
    if point.side != 1536:
        return []
    missing = [case for case in point.cases if case["case_tag"] not in traces]
    if not missing:
        return []
    arm = "segmented_int8_capacity_probe"
    probe = root / arm
    probe.mkdir()
    rows = []
    try:
        with flux.native_worker(model, gpu_uuid=gpu_uuid, profile=point.partition == "calibration") as worker:
            warmup = _run_cell(
                worker, flux_assets, probe / "warmup", point, missing[0], arm, 0,
                warmup=True, witness_arm="segmented_int8",
            )
            if warmup["status"] != "PASS":
                raise RuntimeError(f"standalone INT8 capacity warmup failed: {warmup['error']}")
            for case in missing:
                dest = probe / case["case_tag"]
                row = _run_cell(worker, flux_assets, dest, point, case, arm, 0, witness_arm="segmented_int8")
                rows.append(dict(row, manifest=f"{point.directory}/{dest.relative_to(root)}/manifest.json"))
    except flux.GPUIsolationError:
        raise
    except Exception as error:
        flux.write_json(probe / "worker_failure.json", _reason(error))
    return rows


def _run_arm(flux_assets, point, root, arm, traces, model, gpu_uuid):
    rows = []
    done = set()
    terminal_error = None
    eligible = [case for case in point.cases if arm == "fresh_dense" or case["case_tag"] in traces]
    try:
        if eligible:
            with flux.native_worker(model, gpu_uuid=gpu_uuid, profile=point.partition == "calibration") as worker:
                for rep in range(point.repetitions):
                    order = eligible if rep == 0 else list(reversed(eligible))
                    first = order[0]
                    warm = _run_cell(
                        worker, flux_assets, root / arm / f"rep{rep}" / "warmup", point, first, arm, rep,
                        warmup=True, trace_path=traces[first["case_tag"]][0] if arm != "fresh_dense" else None,
                    )
                    if warm["status"] != "PASS":
                        terminal_error = {key: warm[key] for key in ("failure_kind", "error_type", "error")}
                        break
                    for case in order:
                        tag = case["case_tag"]
                        trace_path, trace_png, origin = traces[tag] if arm != "fresh_dense" else (None, None, None)
                        dest = root / arm / f"rep{rep}" / tag
                        row = _run_cell(
                            worker, flux_assets, dest, point, case, arm, rep, trace_path=trace_path,
                            trace_origin=origin, trace_reference=str(trace_path.relative_to(root)) if trace_path else None,
                            witness_output_digest=trace_png,
                        )
                        evidence = dict(row, point_manifest=str(dest.relative_to(root) / "manifest.json"))
                        if point.partition == "calibration" and rep == 0 and case is first:
                            warm_path = root / arm / "rep0/warmup/manifest.json"
                            evidence["calibration_capture"] = {
                                **capture_features(warm.get("graph") or {}),
                                "manifest_sha256": flux.sha256(warm_path),
                            }
                        rows.append(evidence)
                        done.add((tag, rep))
    except flux.GPUIsolationError:
        raise
    except Exception as error:
        terminal_error = _reason(error)
        log_path = root / arm
        log_path.mkdir(exist_ok=True)
        flux.write_json(log_path / "worker_failure.json", terminal_error)
    for rep in range(point.repetitions):
        for case in point.cases:
            tag = case["case_tag"]
            if (tag, rep) in done:
                continue
            if arm != "fresh_dense" and tag not in traces:
                reason = {"failure_kind": "matched_trace_unavailable", "error": "serial Full-v4 trace was unavailable"}
            else:
                reason = terminal_error or {"failure_kind": "native_worker_incomplete", "error": "request did not run"}
            rows.append(_unavailable(root, point, case, arm, rep, reason=reason))
    return rows


def paired_summary(rows):
    groups = {}
    for row in rows:
        if row["status"] != "PASS" or row["active_ratio"] != .2:
            continue
        key = (row["partition"], row["height"], row["case_tag"], row.get("repetition", 0))
        groups.setdefault(key, {})[row["arm"]] = row
    comparisons, capacity = {}, {}
    reported_partition = "formal" if any(key[0] == "formal" for key in groups) else "calibration"
    for (partition, side, _, _), per_source in groups.items():
        if partition != reported_partition or "full" not in per_source:
            continue
        full = per_source["full"]
        by_side = comparisons.setdefault(str(side), {})
        for arm, row in per_source.items():
            if arm in ("fresh_dense", "full"):
                continue
            result = by_side.setdefault(arm, {
                "pairs": 0, "matching_pngs": 0, "paired_wall_ratios": [], "paired_steady_sparse_device_ratios": [],
            })
            result["pairs"] += 1
            result["matching_pngs"] += (row["output_sha256"] == full["output_sha256"])
            result["paired_wall_ratios"].append(row["request_wall_s"] / full["request_wall_s"])
            if row.get("steady_sparse_device_s") and full.get("steady_sparse_device_s"):
                result["paired_steady_sparse_device_ratios"].append(
                    row["steady_sparse_device_s"] / full["steady_sparse_device_s"]
                )
        int8 = per_source.get("segmented_int8")
        if int8 and all(type(r.get("persistent_kv_bytes")) is int for r in (full, int8)):
            result = capacity.setdefault(str(side), {
                "persistent_kv_saved_bytes": [], "peak_allocated_saved_bytes": [], "peak_reserved_saved_bytes": [],
                "expected_persistent_kv_saved_bytes": projected_int8_saved_bytes(side),
            })
            result["persistent_kv_saved_bytes"].append(full["persistent_kv_bytes"] - int8["persistent_kv_bytes"])
            if "memory" in int8 and "memory" in full:
                for metric in ("allocated", "reserved"):
                    field = f"max_memory_{metric}"
                    result[f"peak_{metric}_saved_bytes"].append(full["memory"][field] - int8["memory"][field])
    speedups = {}
    for arm in ("direct_eager", "standard_fastops_eager"):
        speedups[arm] = {
            side: statistics.mean(group[arm]["paired_wall_ratios"])
            for side, group in comparisons.items() if arm in group
        }
    full_speedup_monotonic = {
        arm: all(ratios[str(a)] < ratios[str(b)] for a, b in zip(SIDES, SIDES[1:]))
        if all(str(side) in ratios for side in SIDES) else None
        for arm, ratios in speedups.items()
    }
    return {
        "partition": reported_partition, "fixed_trace_comparisons": comparisons, "capacity": capacity,
        "pair_ratio_numerator": "candidate", "pair_ratio_denominator": "serial Full-v4",
        "paired_clone_arm_wall_over_full_v4": speedups, "full_v4_speedup_strictly_increases": full_speedup_monotonic,
        "modeled_clone_kv_read_write_bytes_per_sparse_step": {str(side): clone_traffic_bytes(side) for side in SIDES},
    }


def _observation(row):
    if row["status"] != "PASS" or "full_device_s" not in row:
        return None
    graph = row.get("graph") or {}
    observation = {
        "model": "flux", "resolution": row["height"], "runtime": row["arm"],
        "split": "calibration" if row["partition"] == "calibration" else "heldout",
        "source_id": row["source_id"], "ratio": row["active_ratio"], "request_wall_s": row["request_wall_s"],
        "denoise_wall_s": row["denoise_wall_s"], "full_steps": row["dense_steps"],
        "sparse_steps": row["sparse_steps"], "reuse_steps": row["reuse_steps"],
        "full_device_s": row["full_device_total_s"], "steady_sparse_device_s": row["steady_sparse_device_s"],
        **capture_features(graph),
        "repetition": row["repetition"], "manifest": row["manifest"],
        "workload": "fixed_replay",
    }
    if "calibration_capture" in row:
        observation["calibration_capture"] = row["calibration_capture"]
    return observation


def _tree_hashes(root):
    return {
        str(path.relative_to(root)): flux.sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")
    }


def finalize(output, result):
    files = _tree_hashes(output)
    receipt = {"phase": "H2", "schema_version": 1, **result, "files": files}
    completion = output / "completion.json"
    flux.write_json(completion, receipt)
    with (output / "sha256sums.txt").open("x", encoding="utf-8") as stream:
        for name, digest in files.items():
            stream.write(f"{digest}  {name}\n")
        stream.write(f"{flux.sha256(completion)}  completion.json\n")
    return receipt


def _run(config_path, output):
    config = flux.read_json(config_path)
    gpu_uuid = config["gpu_uuid"]
    gpu = flux.verify_gpu(gpu_uuid)
    assets = Path(config["assets"])
    _require(assets.is_absolute(), "assets must be an absolute unpacked asset directory")
    flux_assets = assets / "flux"
    bundle = flux.load_bundle(flux_assets)
    representative_sources = config.get("representative_sources", 24)
    points = experiment_points(bundle, config["run_mode"], representative_sources)
    model = Path(config["models"]["flux"])
    print("Verifying FLUX checkpoint components", flush=True)
    model_hashes = flux.verify_model(model, bundle)
    dependency_versions = {}
    for package in ("torch", "triton", "vllm", "diffusers", "transformers", "Pillow"):
        try:
            dependency_versions[package] = version(package)
        except PackageNotFoundError:
            dependency_versions[package] = None
    protocol = {
        "run_mode": config["run_mode"], "representative_sources": representative_sources,
        "gpu": gpu, "native_dependencies": dependency_versions, "python": sys.version.split()[0],
        "config_sha256": flux.sha256(config_path), "model_components_sha256": model_hashes,
        "flux_bundle_sha256": flux.BUNDLE_SHA256, "flux_runtime_sha256": _tree_hashes(flux.RUNTIME_SOURCE.parent),
        "wall_boundary": WALL_BOUNDARY,
        "separate_costs": ["model loading", "input resize", "warmup", "trace collection", "PNG writing", "CPU metrics"],
        "dtype": "bfloat16", "batch_size": 1, "steps": 28, "warmup_steps": 8, "refresh_step_index": 23,
        "int8_role": "reference KV capacity variant, serial CFG",
        "standalone_int8_capacity": "1536px cases missing a serial Full-v4 matched trace use INT8's own selector",
        "joint_cfg": "Track A budget policy at K <= 921, explicit B2 request at larger K",
        "quality": "offline against paired PNGs",
        "points": [
            {"partition": p.partition, "side": p.side, "ratio": p.ratio, "repetitions": p.repetitions,
             "cases": [case["case_tag"] for case in p.cases], "arms": list(ARMS)}
            for p in points
        ],
    }
    flux.write_json(output / "protocol.json", protocol)
    rows = []
    standalone = []
    observations = output / "latency_observations.jsonl"
    with observations.open("x", encoding="utf-8") as stream:
        for point in points:
            root = output / point.directory
            root.mkdir(parents=True)
            traces = _prepare_traces(flux_assets, point, root, model, gpu_uuid)
            for arm in ARMS:
                for row in _run_arm(flux_assets, point, root, arm, traces, model, gpu_uuid):
                    row["manifest"] = f"{point.directory}/{row.pop('point_manifest')}"
                    rows.append(row)
                    obs = _observation(row)
                    if obs:
                        stream.write(json.dumps(obs, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
            standalone.extend(_standalone_capacity(flux_assets, point, root, traces, model, gpu_uuid))
    if standalone:
        flux.write_json(output / "standalone_capacity_probes.json", standalone)
    successes = sum(row["status"] == "PASS" for row in rows)
    expected = sum(len(point.cases) * point.repetitions * len(ARMS) for point in points)
    flux.write_json(output / "native_records.json", rows)
    summary = paired_summary(rows)
    flux.write_json(output / "paired_summary.json", summary)
    trace_matches = [row["trace_png_match"] for row in rows
                     if row["arm"] == "full" and row.get("trace_png_match") is not None]
    gaps = Counter(
        (row["partition"], row["height"], row["active_ratio"], row["arm"], row["failure_kind"])
        for row in rows if row["status"] != "PASS"
    )
    requests = [
        {
            "model": "flux", "resolution": row["height"], "arm": row["arm"], "request_wall_s": row["request_wall_s"],
            "source_id": row["source_id"], "repetition": row["repetition"], "manifest": row["manifest"],
        }
        for row in rows
        if row["status"] == "PASS" and row["active_ratio"] == .2 and row["partition"] == summary["partition"]
    ]
    return {
        "status": "PASS" if successes == expected else "PARTIAL" if successes else "FAIL",
        "native_execution": bool(successes or any(r["status"] == "PASS" for r in standalone)), "gpu_uuid": gpu_uuid,
        "expected_requests": expected, "completed_requests": successes,
        "unavailable_or_failed_requests": expected - successes, "requests": requests,
        "coverage_gaps": [
            dict(partition=partition, resolution=side, ratio=ratio, arm=arm, reason=reason, requests=count)
            for (partition, side, ratio, arm, reason), count in sorted(gaps.items())
        ],
        "standalone_capacity_probes": "standalone_capacity_probes.json" if standalone else None,
        "standalone_capacity_successes": sum(row["status"] == "PASS" for row in standalone),
        "native_trace_full_replay_png_pairs": len(trace_matches), "native_trace_full_replay_png_matches": sum(trace_matches),
        "quality_status": "PNG ready for offline scoring", "paired_summary": "paired_summary.json",
        "latency_observations": observations.name,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"choose an empty output directory: {output}")
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = _run(args.config, output)
    except Exception as error:
        result = {"status": "FAIL", "native_execution": False, **_reason(error)}
        print(result["traceback"], file=sys.stderr, flush=True)
    finalize(output, {
        **result, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    return 0 if result["status"] in ("PASS", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
