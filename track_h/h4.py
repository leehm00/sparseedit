#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fit the SparseEdit affine step law on Hopper cal3 and predict online holdouts."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean
import traceback

from hopper_io import hashes, sha, write


def finite(value, *, positive=False):
    if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("invalid time or work-count observation")
    return value


def capture_features(graph):
    branches = list(graph.get("branches", {}).values())
    return {
        "capture_trials": sum(b.get("warmup_trials", 0) + b.get("capture_trials", 0) for b in branches),
        "cfg_forwards_per_step": len(branches) or 2,
        "capture_wall_s": sum(b.get("preparation_seconds", 0.0) for b in branches),
    }


def fit(calibration):
    grouped = defaultdict(dict)
    for row in calibration:
        ratio = finite(row["ratio"], positive=True)
        if ratio >= 1 or ratio in grouped[row["source_id"]]:
            raise ValueError("calibration repeats the same source/budget cell")
        grouped[row["source_id"]][ratio] = row
    budgets = {tuple(sorted(rows)) for rows in grouped.values()}
    if len(grouped) != 3 or len(budgets) != 1 or len(next(iter(budgets))) < 2:
        raise ValueError("require three distinct calibration sources with balanced distinct fixed ratios")
    samples = [(ratio, mean(finite(rows[ratio]["steady_sparse_device_s"], positive=True)
                             for rows in grouped.values())) for ratio in sorted(next(iter(budgets)))]
    center_ratio, center_s = map(mean, zip(*samples))
    variance = sum((r - center_ratio)**2 for r, _ in samples)
    beta = sum((r - center_ratio) * (t - center_s) for r, t in samples) / variance
    alpha = center_s - center_ratio * beta
    out = mean(finite(row["request_wall_s"] - row["denoise_wall_s"]) for row in calibration)
    dense = mean(finite(r["full_device_s"], positive=True) / finite(r["full_steps"], positive=True)
                 for r in calibration)
    cold = []
    for row in calibration:
        capture = row.get("calibration_capture", row)
        trials = finite(capture.get("capture_trials", 0))
        if trials and capture.get("capture_wall_s") is not None:
            equivalent_steps = trials / finite(capture.get("cfg_forwards_per_step", 2), positive=True)
            cold.append(max(0.0, finite(capture["capture_wall_s"]) - equivalent_steps * (alpha + beta * row["ratio"])))
    return dict(alpha_s=alpha, beta_s=beta, T_out_s=out, t_dense_s=dense, t_reuse_s=0.0,
                capture_dispatch_s=mean(cold) if cold else None,
                budget_ratios=[r for r, _ in samples], calibration_sources=sorted(grouped))


def predict(row, fit_values):
    ratio = finite(row["ratio"], positive=True)
    sparse = finite(row["sparse_steps"])
    full = finite(row["full_steps"], positive=True)
    reused = finite(row["reuse_steps"])
    if ratio >= 1 or full + sparse + reused != 28:
        raise ValueError("online workload trace must cover the fixed 28 steps")
    step = finite(fit_values["alpha_s"] + fit_values["beta_s"] * ratio, positive=True)
    capture = finite(row.get("capture_trials", 0))
    extra = 0.0
    if capture:
        if fit_values["capture_dispatch_s"] is None:
            raise ValueError("cold holdouts need cal3 capture timing")
        extra = capture / finite(row.get("cfg_forwards_per_step", 2), positive=True) * step
        extra += fit_values["capture_dispatch_s"]
    return fit_values["T_out_s"] + full * fit_values["t_dense_s"] + sparse * step + extra


def evaluate(rows):
    groups = defaultdict(lambda: {"calibration": [], "heldout": []})
    for row in rows:
        if row["split"] == "heldout" and row.get("workload", "online") != "online":
            continue
        groups[row["model"], row["resolution"], row["runtime"]][row["split"]].append(row)
    comparisons = []
    for key, group in sorted(groups.items()):
        cal, test = group["calibration"], group["heldout"]
        sources = {row["source_id"] for row in test}
        entry = dict(model=key[0], resolution=key[1], runtime=key[2], heldout_sources=len(sources))
        try:
            fit_values = fit(cal)
            entry["fit"] = fit_values
            if set(fit_values["calibration_sources"]) & sources:
                raise ValueError("calibration and heldout source hashes overlap")
            if len(sources) != 48:
                raise ValueError("require all 48 held-out online sources")
            errors = []
            for row in test:
                actual = finite(row["request_wall_s"], positive=True)
                predicted = predict(row, fit_values)
                errors.append(dict(source_id=row["source_id"], repetition=row.get("repetition"),
                                   predicted_s=predicted, measured_s=actual,
                                   relative_error_percent=abs(predicted - actual) / actual * 100))
            mape = mean(row["relative_error_percent"] for row in errors)
            entry.update(status="PASS" if mape < 5 else "FAIL", mape_percent=mape,
                         under_5_percent=mape < 5, predictions=errors)
        except (ValueError, KeyError, TypeError) as exc:
            entry.update(status="BLOCKED", reason=str(exc))
        comparisons.append(entry)
    tested = [row for row in comparisons if row["status"] != "BLOCKED"]
    status = "BLOCKED" if not tested else (
        "FAIL" if any(row["status"] == "FAIL" for row in tested)
        else "PASS" if len(tested) == len(comparisons) else "PARTIAL"
    )
    return dict(
        status=status, comparisons=comparisons,
        assumption="reuse update cost fixed to zero per Track A Amdahl model; cold capture calibrated separately",
        track_h_scope="predictions only where Hopper cal3 and 48 source-disjoint online holdouts are both available",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir()
    config = json.loads(args.config.read_text())
    root = Path(config["prior_results"])
    paths = sorted(root.glob("H[123]/**/latency_observations.jsonl"))
    receipt = dict(phase="H4", gpu_uuid=config["gpu_uuid"], source_sha256={"h4.py": sha(__file__)},
                   input_sha256={p.relative_to(root).as_posix(): sha(p) for p in paths})
    try:
        rows = [json.loads(line) for p in paths for line in p.read_text().splitlines() if line.strip()]
        receipt.update(evaluate(rows))
        if not rows:
            receipt["reason"] = "No earlier stage published usable Hopper latency observations"
    except Exception:
        receipt.update(status="FAIL", traceback=traceback.format_exc())
    receipt["output_sha256"] = hashes(args.output)
    write(args.output / "completion.json", receipt)
    return 0 if receipt["status"] in ("PASS", "PARTIAL", "BLOCKED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
