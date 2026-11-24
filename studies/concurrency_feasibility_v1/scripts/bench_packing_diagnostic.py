#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure synthetic FLUX projection packing on a single H200."""

import argparse
from datetime import UTC, datetime
import fcntl
import os
from pathlib import Path
from statistics import median

from bench_concurrency import STUDY, _write_json, sha256_file
from run_h200_window import preflight

ROWS, HIDDEN, MLP, SEED = 460, 3072, 12288, 110


def project(mode, rows, weight):
    import torch
    from torch.nn.functional import linear

    if mode == "packed":
        return linear(torch.cat(rows), weight).split(len(rows[0]))
    return [linear(request, weight) for request in rows]


def summarize_trials(projection, count, trials):
    return {
        "projection": projection, "requests": count,
        "scope": "synthetic_projection_including_pack_and_split",
        "median_loop_ms": median(row["loop_ms"] for row in trials),
        "median_packed_ms": median(row["packed_ms"] for row in trials),
        "median_paired_speedup": median(row["loop_ms"] / row["packed_ms"] for row in trials),
        "trials": trials,
    }


def measure(receipt, repetitions):
    import torch

    rng = torch.Generator(device="cuda").manual_seed(SEED)

    def rand(shape):
        return torch.randn(shape, generator=rng, dtype=torch.bfloat16, device="cuda")

    def timed(mode, rows, weight):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = project(mode, rows, weight)
        end.record()
        end.synchronize()
        return start.elapsed_time(end), result

    with torch.inference_mode():
        for name, in_features, out_features in (("expand", HIDDEN, MLP), ("contract", MLP, HIDDEN)):
            weight = rand((out_features, in_features))
            for count in (1, 2, 4):
                rows = [rand((ROWS, in_features)) for _ in range(count)]
                for _ in range(5):
                    project("loop", rows, weight)
                    project("packed", rows, weight)
                torch.cuda.synchronize()
                trials = []
                projections = {}
                for repetition in range(repetitions):
                    order = ("loop", "packed") if repetition % 2 == 0 else ("packed", "loop")
                    trial = {"order": order}
                    for mode in order:
                        trial[f"{mode}_ms"], projections[mode] = timed(mode, rows, weight)
                    trials.append(trial)
                pairs = list(zip(projections["loop"], projections["packed"], strict=True))
                maximum = max((a.float() - b.float()).abs().max().item() for a, b in pairs)
                bitwise = all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in pairs)
                receipt["projections"].append({
                    **summarize_trials(name, count, trials),
                    "max_absolute_output_difference": maximum, "bitwise_equal": bitwise,
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if args.repetitions < 2 or args.repetitions % 2:
        parser.error("Choose a positive even repetition count of at least two.")
    output = args.output_root.resolve()
    if not output.is_relative_to(STUDY / "artifacts") or output == STUDY / "artifacts":
        parser.error("Choose a fresh directory under study artifacts.")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "phase": "F6", "status": "FAILED", "device_uuid": args.gpu_uuid,
        "protocol": {"dtype": "bfloat16", "rows_per_request": ROWS, "hidden": HIDDEN, "mlp": MLP, "seed": SEED},
        "runner_sha256": sha256_file(Path(__file__)), "projections": [],
        "timing_boundary": "CUDA events for synthetic F.linear, including CUDA concatenation in packed mode",
    }
    try:
        with Path(f"/tmp/sparseedit-{args.gpu_uuid}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt["device"] = preflight(args.gpu_uuid)
            if receipt["device"]["status"] != "PASS":
                receipt["status"] = "BLOCKED"
                raise ValueError("Physical H200 preflight failed")
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
            measure(receipt, args.repetitions)
            receipt["status"] = "PASS"
    except (OSError, ValueError, RuntimeError) as error:
        receipt["failure"] = str(error)
    finally:
        receipt["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(output / "completion.json", receipt)
    return int(receipt["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
