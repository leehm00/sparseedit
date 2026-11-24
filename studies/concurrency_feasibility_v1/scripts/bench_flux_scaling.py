#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure exploratory FLUX single-device scaling on the frozen 768px trace."""

from __future__ import annotations

import argparse
import fcntl
import gc
import math
import os
import socket
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import bench_concurrency as bench

ARTIFACTS = bench.STUDY / "artifacts"
SCOPE = "exploratory_frozen_768_scaling"


def scaling_schedule(tags: tuple[str, ...], counts: list[int], repetitions: int) -> Iterator[bench.Measurement]:
    """Rotate the calibration cases and alternate the paired measurement order."""
    for repetition in range(repetitions):
        modes = ("serial", "concurrent") if repetition % 2 == 0 else ("concurrent", "serial")
        for count in counts:
            batch = tuple(tags[(repetition + slot) % len(tags)] for slot in range(count))
            for mode in modes:
                yield bench.Measurement("full", mode, 0, repetition, batch)


def bind_cell(cell, bound_cases, identities, namespace=None):
    """Give repeated sources unique worker IDs while keeping their frozen trace."""
    namespace = namespace or cell.directory
    request_ids = tuple(f"{namespace}_i{index}_{tag}" for index, tag in enumerate(cell.cases))
    cases = {}
    records = {}
    for request_id, tag in zip(request_ids, cell.cases, strict=True):
        case, image, trace_path, trace = bound_cases[tag]
        cases[request_id] = ({**case, "case_tag": request_id}, image, trace_path, trace)
        records[request_id] = {**identities[tag], "request_id": request_id}
    return request_ids, cases, records


def classify_outputs(cases, references):
    matches = []
    for case in cases:
        expected = references.get(case["case_tag"])
        matches.append(case["output_png_sha256"] == expected if expected is not None else None)
    if False in matches:
        return "OUTPUT_MISMATCH", matches
    if None in matches:
        return "DIAGNOSTIC", matches
    return "PASS", matches


def paired_speedups(rows):
    rows_by_pair = {}
    for row in rows:
        if row["status"] == "PASS":
            pair = row["concurrency"], row["repetition"]
            rows_by_pair.setdefault(pair, {})[row["mode"]] = row
    result = []
    for (count, repetition), arms in sorted(rows_by_pair.items()):
        if arms.keys() != {"serial", "concurrent"}:
            continue
        serial, concurrent = arms["serial"], arms["concurrent"]
        if [case["case_tag"] for case in serial["cases"]] != [case["case_tag"] for case in concurrent["cases"]]:
            continue
        serial_wall = serial["batch_wall_seconds"]
        concurrent_wall = concurrent["batch_wall_seconds"]
        result.append({
            "concurrency": count,
            "repetition": repetition,
            "serial_seconds": serial_wall,
            "concurrent_seconds": concurrent_wall,
            "throughput_speedup": serial_wall / concurrent_wall,
        })
    return result


def validate_execution(raw, mode, request_ids, steps):
    from itertools import accumulate

    count = len(request_ids)
    wall = raw["batch_wall_seconds"]
    timing = raw["timings"]
    if not bench._positive(wall) or len(timing) != count:
        raise ValueError("Missing synchronized batch wall or request timings")
    seconds = [item["request_seconds"] for item in timing]
    if any(
        not bench._positive(item["cuda_event_seconds"]) or not bench._positive(elapsed) or elapsed > wall + 0.001
        for item, elapsed in zip(timing, seconds, strict=True)
    ):
        raise ValueError("Invalid request completion or CUDA timing")
    streams = raw["streams"]
    if mode == "serial":
        if len(streams) != 1 or not math.isclose(sum(seconds), wall, rel_tol=0, abs_tol=0.001):
            raise ValueError("Serial execution differs from the synchronized batch boundary")
        return list(accumulate(seconds))

    if len(streams) != count or len(set(streams)) != count:
        raise ValueError("Concurrent requests do not have distinct CUDA streams")
    observed = [(row["step"], row["request_id"], row["stream_id"]) for row in raw["enqueue_order"]]
    expected = [
        (step, request_id, stream)
        for step in range(steps)
        for request_id, stream in zip(request_ids, streams, strict=True)
    ]
    if observed != expected:
        raise ValueError("Concurrent execution did not preserve step interleaving")
    return seconds


def collect_cell(folder, cell, pipeline, post_process, protocol, lane, bound_cases, identities, references=None):
    """Write each cell independently so a later timeout cannot erase it."""
    import torch
    from analyze_capacity import classify_execution_error

    folder.mkdir(parents=True, exist_ok=False)
    request_ids, cases, records = bind_cell(cell, bound_cases, identities, namespace=folder.name)
    measurement = bench.Measurement(cell.arm, cell.mode, cell.pair, cell.repetition, request_ids)
    receipt = {
        "scope": SCOPE,
        "status": "RUNNING",
        "arm": cell.arm,
        "mode": cell.mode,
        "runtime_lane": lane,
        "concurrency": len(request_ids),
        "repetition": cell.repetition,
        "cases": list(records.values()),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "synchronized_before_return": False,
    }
    bench._write_json(folder / "attempt.json", receipt)
    print(f"FLUX {lane} {len(request_ids)} {cell.mode}: {folder}", flush=True)
    try:
        raw = bench._run_cell(pipeline, post_process, measurement, protocol, lane, cases)
        images = raw.pop("images")
        with ExitStack() as outputs:
            for image in images:
                outputs.callback(image.close)
            completion = validate_execution(raw, cell.mode, request_ids, protocol["steps"])
            rows = bench._png_records(images, request_ids, records, folder)
        timings = raw.pop("timings")
        resize_seconds = raw.pop("resize_seconds")
        for row, timing, resize, elapsed in zip(rows, timings, resize_seconds, completion, strict=True):
            row.update(timing, input_resize_seconds=resize, completion_seconds_from_batch_start=elapsed)
        status, matches = ("PASS", []) if references is None else classify_outputs(rows, references)
        if references is not None:
            for row, match in zip(rows, matches, strict=True):
                row.update(serial_reference_png_sha256=references.get(row["case_tag"]), output_matches_serial=match)
        receipt.update(
            raw,
            status=status,
            synchronized_before_return=True,
            cases=rows,
            output_check="same_worker_serial_png" if references is not None else "reference_capture",
            measured_images_per_minute=60 * len(rows) / raw["batch_wall_seconds"],
        )
    except Exception as error:
        status, failure = classify_execution_error(error)
        receipt.update(
            status="CAPACITY_BLOCKED" if status == "OOM" else "FAILED",
            failure=failure,
        )
        if torch.cuda.is_initialized():
            receipt.update(
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            )
    receipt["completed_at_utc"] = datetime.now(UTC).isoformat()
    bench._write_json(folder / "completion.json", receipt)
    print(f"{receipt['status']} {folder / 'completion.json'}", flush=True)
    if receipt["status"] == "CAPACITY_BLOCKED":
        gc.collect()
        torch.cuda.empty_cache()
    return receipt


def run_sweep(args, root, receipt):
    if Path(sys.prefix).resolve() != bench.PYTHON.parents[1].resolve():
        raise RuntimeError(f"FLUX scaling requires the frozen environment at {bench.PYTHON}")
    sys.path.insert(0, str(bench.STUDY / "source"))
    from analyze_concurrency import summarize_arms
    from run_baseline import load_cases, verify_hashes

    protocol, identities = bench.load_protocol_cases()
    bundle = bench.read_json(bench.BUNDLE_ROOT / "bundle.json")
    bound = {case[0]["case_tag"]: case for case in load_cases(bench.BUNDLE_ROOT, bundle)}
    receipt.update(
        environment=bench._environment(),
        protocol=protocol,
        bundle_sha256=bench.sha256_file(bench.BUNDLE_ROOT / "bundle.json"),
        calibration_cases=list(identities.values()),
    )
    with Path(f"/tmp/sparseedit-{args.gpu_uuid}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt["device"] = bench._check_measurement_device(args.gpu_uuid)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
        verify_hashes(args.model, bundle["model_hashes"])
        receipt["model_identity"] = {
            "method": "every file SHA-256 matches the frozen v4 bundle",
            "path": str(args.model),
            "file_count": len(bundle["model_hashes"]),
        }
        bench._write_json(root / "preflight.json", {**receipt, "status": "PASS"})
        started = time.perf_counter()
        with bench.load_pipeline(args.model) as (pipeline, post_process):
            receipt["model_load_seconds"] = time.perf_counter() - started
            tags = tuple(identities)
            for mode in ("serial", "concurrent"):
                collect_cell(
                    root / "warmup" / mode, bench.Measurement("full", mode, 0, 0, tags[:1]),
                    pipeline, post_process, protocol, args.lane, bound, identities,
                )
            references = {}
            reference_rows = []
            for tag in tags:
                reference_path = root / "serial_reference" / tag
                row = collect_cell(
                    reference_path, bench.Measurement("full", "serial", 0, 0, (tag,)),
                    pipeline, post_process, protocol, args.lane, bound, identities,
                )
                reference_rows.append({
                    "case_tag": tag, "status": row["status"],
                    "receipt": str((reference_path / "completion.json").relative_to(root)),
                })
                if row["status"] == "PASS":
                    references[tag] = row["cases"][0]["output_png_sha256"]
            receipt["reference_receipts"] = reference_rows

            rows = []
            for cell in scaling_schedule(tags, args.counts, args.repetitions):
                folder = root / cell.directory
                row = collect_cell(
                    folder, cell, pipeline, post_process, protocol,
                    args.lane, bound, identities, references,
                )
                rows.append(row)
                receipt["cells"].append({
                    "path": str((folder / "completion.json").relative_to(root)),
                    "status": row["status"],
                    "mode": row["mode"],
                    "concurrency": row["concurrency"],
                    "repetition": row["repetition"],
                })
    passing = [row for row in rows if row["status"] == "PASS"]
    receipt["summaries"] = summarize_arms(passing)
    receipt["paired_speedups"] = paired_speedups(passing)
    if len(passing) == len(rows):
        receipt["status"] = "PASS"
    elif any(row.get("synchronized_before_return") for row in rows):
        receipt["status"] = "PARTIAL"
    elif all(row["status"] == "CAPACITY_BLOCKED" for row in rows):
        receipt["status"] = "CAPACITY_BLOCKED"
    else:
        receipt["status"] = "FAILED"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lane", choices=("private_eager", "private_graph"), default="private_eager")
    parser.add_argument("--counts", type=int, nargs="+", choices=(1, 2, 4, 8), default=[1, 2, 4, 8])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    args = parser.parse_args(argv)
    if args.repetitions < 1 or len(args.counts) != len(set(args.counts)):
        parser.error("Choose positive repetitions and distinct concurrency counts")
    root = args.output_root.resolve()
    if not root.is_relative_to(ARTIFACTS) or root == ARTIFACTS:
        parser.error("Select a new output directory below the study artifacts")
    root.mkdir(parents=True, exist_ok=False)
    args.model = args.model.resolve()
    receipt = {
        "phase": "flux_scaling",
        "scope": SCOPE,
        "status": "BLOCKED",
        "runtime_lane": args.lane,
        "counts": args.counts,
        "repetitions": args.repetitions,
        "requested_resolution": {"height": args.height, "width": args.width},
        "timing_boundary": bench.TIMING_BOUNDARY,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "device_uuid": args.gpu_uuid,
        "source_sha": bench.source_tree_sha(),
        "runner_sha256": bench.sha256_file(Path(__file__)),
        "cells": [],
    }
    bench._write_json(root / "attempt.json", {**receipt, "status": "RUNNING"})
    try:
        if (args.height, args.width) == (bench.FROZEN_PROTOCOL["height"], bench.FROZEN_PROTOCOL["width"]):
            run_sweep(args, root, receipt)
        else:
            receipt.update(
                status="UNSUPPORTED",
                failure="Frozen SparseEdit traces are shape-bound to 768×768; this resolution requires new traces.",
            )
    except Exception as error:
        receipt.update(failure=str(error), traceback=traceback.format_exc())
        if receipt["cells"]:
            receipt["status"] = "PARTIAL"
    finally:
        receipt["completed_at_utc"] = datetime.now(UTC).isoformat()
        bench._write_json(root / "completion.json", receipt)
    print(f"{receipt['status']} {root / 'completion.json'}", flush=True)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
