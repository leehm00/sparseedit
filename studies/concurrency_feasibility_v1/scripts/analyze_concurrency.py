#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate the F2 timing evidence and apply the predeclared decision rule."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from bench_concurrency import (
    PYTHON,
    STUDY,
    calibration_pairs,
    load_protocol_cases,
    sha256_file,
    validate_measurement,
)


def _percentile(values: list[float], fraction: float) -> float:
    index = (len(values) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    weight = index - lower
    return values[lower] + weight * (values[upper] - values[lower])


def analyze_paired(
    receipts: list[dict[str, Any]],
    protocol: dict[str, Any],
    identities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Average within calibration pair, then bootstrap the three pair means."""
    rows = [row for row in receipts if row["arm"] == "full" and row["concurrency"] == 2]
    if not rows:
        raise ValueError("No measured F2 full N=2 batches")
    indexed: dict[tuple[int, int, str], dict[str, Any]] = {}
    pairs = calibration_pairs(tuple(identities))
    for row in rows:
        validate_measurement(row, protocol, identities)
        key = row["pair"], row["repetition"], row["mode"]
        if key in indexed:
            raise ValueError(f"Duplicate paired comparison cell {key}")
        indexed[key] = row
    for field in ("device_uuid", "source_sha", "environment", "timing_boundary", "runtime_lane"):
        if any(row[field] != rows[0][field] for row in rows):
            raise ValueError(f"F2 comparison mixes different {field} values")

    repetitions = sorted({row["repetition"] for row in rows})
    if repetitions not in (list(range(2)), list(range(4))):
        raise ValueError("Expected two paired repetitions or two plus their two planned repeats")
    expected_cells = {
        (pair, repetition, mode)
        for pair in range(len(pairs))
        for repetition in repetitions
        for mode in ("serial", "concurrent")
    }
    if set(indexed) != expected_cells:
        raise ValueError("Missing or unexpected frozen calibration pair measurement")

    comparisons = []
    pair_means = []
    for pair, tags in enumerate(pairs):
        ratios = []
        observed_orders = []
        for repetition in repetitions:
            serial = indexed[pair, repetition, "serial"]
            concurrent = indexed[pair, repetition, "concurrent"]
            for row in (serial, concurrent):
                if tuple(case["case_tag"] for case in row["cases"]) != tags:
                    raise ValueError(f"Frozen pair identity or input order changed for pair {pair}")
            first = serial["measurement_order"]
            second = concurrent["measurement_order"]
            if first == second:
                raise ValueError(f"Serial/concurrent measurement order is ambiguous for pair {pair}")
            observed_orders.append(first < second)
            expected_pngs = [case["output_png_sha256"] for case in serial["cases"]]
            actual_pngs = [case["output_png_sha256"] for case in concurrent["cases"]]
            if actual_pngs != expected_pngs:
                raise ValueError(f"Full serial and concurrent PNGs diverged for pair {pair}")
            serial_wall = serial["batch_wall_seconds"]
            concurrent_wall = concurrent["batch_wall_seconds"]
            ratio = serial_wall / concurrent_wall
            ratios.append(ratio)
            comparisons.append(
                {
                    "pair": pair,
                    "cases": list(tags),
                    "repetition": repetition,
                    "first_mode": "serial" if first < second else "concurrent",
                    "serial_batch_wall_seconds": serial_wall,
                    "concurrent_batch_wall_seconds": concurrent_wall,
                    "throughput_ratio": ratio,
                }
            )
        if sum(observed_orders) * 2 != len(repetitions):
            raise ValueError(f"F2 order is unbalanced for pair {pair}")
        pair_means.append(mean(ratios))

    rng = random.Random(110)
    draws = sorted(mean(rng.choices(pair_means, k=len(pair_means))) for _ in range(5000))
    low = _percentile(draws, 0.025)
    high = _percentile(draws, 0.975)
    point = mean(pair_means)
    if point < 1.05:
        decision = "NO_GO"
    elif point >= 1.15 and low > 1.05:
        decision = "GO"
    else:
        decision = "WEAK"

    crosses_threshold = low <= 1.05 <= high
    needs_more = len(repetitions) == 2 and (crosses_threshold or decision == "WEAK")
    return {
        "status": "REPEAT_REQUIRED" if needs_more else "PASS",
        "decision": None if needs_more else decision,
        "point_estimate": point,
        "confidence_interval_95": [low, high],
        "confidence_interval_crosses_1_05": crosses_threshold,
        "bootstrap_draws": 5000,
        "bootstrap_seed": 110,
        "bootstrap_unit": "calibration pair mean across repetitions",
        "pair_means": pair_means,
        "paired_repetitions": len(repetitions),
        "additional_repetitions": 2 if needs_more else 0,
        "comparisons": comparisons,
    }


def collect_receipts(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, json.loads(path.read_text())) for path in sorted(root.glob("*/completion.json"))]


def _optional_mean(values) -> float | None:
    measured = [value for value in values if value is not None]
    return mean(measured) if measured else None


def summarize_arms(receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in receipts:
        groups[row["arm"], row["mode"], row["concurrency"]].append(row)
    summaries = []
    for (arm, mode, concurrency), rows in sorted(groups.items()):
        batch_wall = sum(row["batch_wall_seconds"] for row in rows)
        request_seconds = sorted(case["request_seconds"] for row in rows for case in row["cases"])
        summaries.append(
            {
                "arm": arm,
                "mode": mode,
                "concurrency": concurrency,
                "batches": len(rows),
                "requests": len(request_seconds),
                "mean_batch_wall_seconds": batch_wall / len(rows),
                "images_per_second": len(request_seconds) / batch_wall,
                "latency_p50_seconds": _percentile(request_seconds, 0.5),
                "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
                "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in rows),
                "first_sparse_step_forward_seconds": _optional_mean(
                    case.get("first_sparse_step_forward_seconds") for row in rows for case in row["cases"]
                ),
                "steady_sparse_step_forward_seconds": _optional_mean(
                    case.get("steady_sparse_step_forward_seconds") for row in rows for case in row["cases"]
                ),
                "mean_gpu_utilization_percent": _optional_mean(
                    row.get("gpu_utilization", {}).get("mean_gpu_percent") for row in rows
                ),
            }
        )
    return summaries


def _quality_scores(measurements: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    """Compare paired saved full and dense images without affecting run timing."""
    from time import perf_counter

    import numpy as np
    from PIL import Image

    started = perf_counter()
    indexes = {
        (row["pair"], row["repetition"], row["arm"], row["mode"]): (path, row)
        for path, row in measurements
        if row["concurrency"] == 2
    }
    observations = []
    for (pair, repetition, arm, mode), (path, row) in indexes.items():
        if arm != "dense" or mode not in ("true_batch", "concurrent"):
            continue
        other = indexes.get((pair, repetition, "full", "concurrent"))
        if other is None:
            continue
        sparse_path, sparse = other
        sparse_cases = {case["case_tag"]: case for case in sparse["cases"]}
        for dense_case in row["cases"]:
            tag = dense_case["case_tag"]
            full_case = sparse_cases[tag]
            with (
                Image.open(path.parent / dense_case["output_png"]) as dense_image,
                Image.open(sparse_path.parent / full_case["output_png"]) as full_image,
            ):
                dense_rgb = np.asarray(dense_image, dtype=np.float32) / 255
                full_rgb = np.asarray(full_image, dtype=np.float32) / 255
                difference = full_rgb - dense_rgb
                mse = float(np.mean(difference**2))
            observations.append(
                {
                    "case_tag": tag,
                    "pair": pair,
                    "repetition": repetition,
                    "full_vs_dense_rgb_mse": mse,
                    "full_vs_dense_psnr_db": -10 * math.log10(mse) if mse else None,
                    "png_equal": dense_case["output_png_sha256"] == full_case["output_png_sha256"],
                    "dense_receipt": str(path.resolve()),
                    "full_receipt": str(sparse_path.resolve()),
                }
            )
    return {
        "status": "MEASURED" if observations else "UNAVAILABLE",
        "scoring_seconds": perf_counter() - started,
        "observations": observations,
        "method": "source-paired RGB pixel MSE and PSNR on saved PNGs",
    }


def _verify_observations(
    receipts: list[tuple[Path, dict[str, Any]]],
    protocol: dict[str, Any],
    identities: dict[str, dict[str, Any]],
    isolation_root: Path,
) -> None:
    """Verify measured outputs and their F1 anchors before accepting a row."""
    candidates = {sha256_file(path): path for path in isolation_root.rglob("completion.json")}
    audited = {}
    for path, row in receipts:
        validate_measurement(row, protocol, identities)
        for case in row["cases"]:
            png = (path.parent / case["output_png"]).resolve()
            if not png.is_relative_to(path.parent.resolve()) or sha256_file(png) != case["output_png_sha256"]:
                raise ValueError(f"F2 output SHA-256 differs for {png}")
        anchors = row.get("isolation_receipts", [])
        if len(anchors) != 3 or len({entry["sha256"] for entry in anchors}) != 3:
            raise ValueError(f"F2 receipt lacks all 3 F1 pair anchors: {path}")
        observed_pairs = set()
        f1_pngs = {}
        for entry in anchors:
            digest = entry["sha256"]
            if digest not in audited:
                anchor = candidates.get(digest)
                original = Path(entry["path"])
                if anchor is None and original.is_file() and sha256_file(original) == digest:
                    anchor = original
                if anchor is None:
                    raise ValueError(f"F1 anchor SHA-256 unavailable for {original}")
                witness = json.loads(anchor.read_text())
                for case in witness["cases"]:
                    tag = case["case_tag"]
                    if tag not in identities:
                        raise ValueError(f"Unknown F1 case tag {tag!r}")
                    png = anchor.parent / f"{tag}.png"
                    serial = anchor.parent.parent / "serial_reference" / png.name
                    if (
                        sha256_file(png) != case["output_png_sha256"]
                        or sha256_file(serial) != case["serial_png_sha256"]
                        or case["output_png_sha256"] != case["serial_png_sha256"]
                    ):
                        raise ValueError(f"F1 output SHA-256 differs for {png} or {serial}")
                audited[digest] = witness
            witness = audited[digest]
            if (
                witness.get("status") != "PASS"
                or witness.get("phase") != "F1"
                or any(witness.get(field) != row[field] for field in ("device_uuid", "source_sha", "protocol"))
                or (row["arm"] == "full" and witness.get("runtime_lane") != row["runtime_lane"])
                or witness.get("correctness", {}).get("all_match") is not True
            ):
                raise ValueError(f"F1 anchor differs from F2 run for {entry['path']}")
            observed_pairs.add(tuple(case["case_tag"] for case in witness["cases"]))
            for case in witness["cases"]:
                tag = case["case_tag"]
                digest = case["output_png_sha256"]
                if tag in f1_pngs and f1_pngs[tag] != digest:
                    raise ValueError(f"F1 output SHA-256 differs across pair anchors for {tag}")
                f1_pngs[tag] = digest
        if observed_pairs != set(calibration_pairs(tuple(identities))):
            raise ValueError(f"F1 pair coverage differs for {path}")
        if row["arm"] == "full" and any(
            f1_pngs[case["case_tag"]] != case["output_png_sha256"] for case in row["cases"]
        ):
            raise ValueError(f"F2 full output differs from its F1 receipt for {path}")


def _read_isolation(root: Path) -> list[dict[str, Any]]:
    observations = []
    for path in sorted(root.rglob("completion.json")):
        row = json.loads(path.read_text())
        if row.get("phase") != "F1":
            continue
        observations.append(
            {
                "status": row["status"],
                "runtime_lane": row.get("runtime_lane"),
                "device_uuid": row.get("device_uuid"),
                "failure": row.get("failure"),
                "peak_allocated_bytes": row.get("peak_allocated_bytes"),
                "peak_reserved_bytes": row.get("peak_reserved_bytes"),
                "cases": [case["case_tag"] for case in row.get("cases", [])],
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    return observations


def _blocked_reason(isolation: list[dict[str, Any]], receipts: list[dict[str, Any]]) -> str:
    if receipts:
        return "F2 已有部分收据，仍缺少通过验收的同批次计时或强对照。"
    valid = defaultdict(set)
    for row in isolation:
        if row["status"] == "PASS" and len(row["cases"]) == 2:
            valid[row["device_uuid"], row["runtime_lane"]].add(tuple(row["cases"]))
    if any(len(pairs) == 3 for pairs in valid.values()):
        return "F1 已有三个批次的通过收据，F2 配对计时等待执行。"
    if any(row["status"] == "CAPACITY_BLOCKED" for row in isolation):
        return "F1 在已试设备上遇到原协议显存不足。须在更大显存单卡上完成 F1 隔离验收后采集吞吐。"
    if any(row["status"] == "FAILED" for row in isolation):
        return "F1 在已试执行路径上未通过。F2 吞吐采集等待 F1 验收。"
    return "F2 吞吐采集等待同设备、同执行路径的三个校准批次 F1 隔离收据。"


def render_report(result: dict[str, Any]) -> str:
    lines = ["# SparseEdit Phase A 可行性实验", "", "## F2 同步端到端吞吐", ""]
    decision = result.get("decision")
    if decision:
        low, high = result["confidence_interval_95"]
        lines.extend(
            [
                f"**{decision}**。并发 N=2 相对同批次逐请求串行的配对吞吐比为 "
                f"{result['point_estimate']:.3f}×，95% percentile bootstrap 区间为 "
                f"[{low:.3f}, {high:.3f}]。每个批次有 {result['paired_repetitions']} 次配对重复。",
                "",
                "完整请求计时包含编码、去噪、VAE 解码与运行时后处理，"
                "另记加载、输入 resize、独立预热、PNG 写盘和评分。三个批次由同一组三个"
                "校准来源两两构成，区间用于可行性决策。",
                "",
            ]
        )
    elif result["status"] == "REPEAT_REQUIRED":
        lines.extend(["配对结果触发了顺序平衡复测。再完成 2 次重复后确定 F2 结论。", ""])
    else:
        lines.extend([f"状态为 **{result['status']}**。{result.get('reason', '')}", ""])
    if result.get("arms"):
        lines.extend(
            [
                "| Arm | Mode | N | 批次 | 平均完整批次（s） | images/s | p50（s） | 峰值 reserved（GiB） |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["arms"]:
            lines.append(
                f"| {row['arm']} | {row['mode']} | {row['concurrency']} | {row['batches']} | "
                f"{row['mean_batch_wall_seconds']:.3f} | {row['images_per_second']:.4f} | "
                f"{row['latency_p50_seconds']:.3f} | {row['peak_reserved_bytes'] / 2**30:.2f} |"
            )
        lines.append("")
    if result.get("single_request_p50_ratio") is not None:
        lines.extend(
            [
                f"N=1 的多流调度路径 / 逐请求串行 p50 比为 {result['single_request_p50_ratio']:.3f}×。",
                "",
            ]
        )
    if result.get("quality", {}).get("status") == "MEASURED":
        quality = result["quality"]["observations"]
        scores = [row["full_vs_dense_psnr_db"] for row in quality if row["full_vs_dense_psnr_db"] is not None]
        if scores:
            lines.extend(
                [
                    f"同来源 full 与 dense PNG 的 PSNR 为 {mean(scores):.2f} dB（"
                    f"{len(scores)} 个非逐位匹配输出）。逐请求 RGB MSE、PSNR 和评分时间见汇总。",
                    "",
                ]
            )
        else:
            lines.extend([f"同来源 full 与 dense 的 {len(quality)} 个 PNG 配对逐字节相同。", ""])
    if result.get("f1_receipts"):
        lines.extend(["## F1 执行证据", ""])
        for row in result["f1_receipts"]:
            lane = row["runtime_lane"] or "该次运行"
            memory = ""
            if row["peak_reserved_bytes"]:
                memory = f"，峰值 reserved {row['peak_reserved_bytes'] / 2**30:.2f} GiB"
            lines.append(f"- {lane} 为 {row['status']}{memory}。[隔离收据]({row['path']})")
        lines.append("")
    if result.get("nonpassing_receipts"):
        lines.extend(["## 运行状态", ""])
        for row in result["nonpassing_receipts"]:
            lines.append(f"- {row['status']} [{Path(row['path']).parent.name}]({row['path']})")
        lines.append("")
    if result["status"] != "PASS":
        repeat_options = (
            "--start-repetition 2 --repetitions 2" if result["status"] == "REPEAT_REQUIRED" else ("--repetitions 2")
        )
        lines.extend(
            [
                "## 运行命令",
                "",
                "F1 验收通过后，用同一个设备 UUID 运行配对矩阵。",
                "内网主机上使用复制的模型时，添加 `--model` 并给出模型目录。",
                "",
                "```sh",
                f"'{PYTHON}' '{STUDY / 'scripts/bench_concurrency.py'}'"
                f' --matrix --gpu-uuid "$gpu_uuid" {repeat_options}',
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 收据",
            "",
            f"- [实验汇总]({result['summary_path']})",
            f"- [原始批次收据目录]({result['receipt_root']})",
            "",
        ]
    )
    return "\n".join(lines)


def _has_required_controls(
    receipts: list[dict[str, Any]],
    repetitions: list[int],
    tags: tuple[str, ...],
) -> bool:
    first, last = repetitions[0], repetitions[-1]
    selected = [row for row in receipts if first <= row["repetition"] <= last]
    expected_n1 = {(tag, rep) for tag in tags for rep in repetitions}
    for mode in ("serial", "concurrent"):
        observed = {
            (row["cases"][0]["case_tag"], row["repetition"])
            for row in selected
            if row["arm"] == "full" and row["mode"] == mode and row["concurrency"] == 1
        }
        if observed != expected_n1:
            return False
    expected_n2 = {(pair, rep) for pair in range(len(tags)) for rep in repetitions}
    dense = {
        (row["pair"], row["repetition"])
        for row in selected
        if row["arm"] == "dense" and row["mode"] in ("true_batch", "concurrent") and row["concurrency"] == 2
    }
    return dense == expected_n2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=STUDY / "artifacts/concurrency_v1")
    parser.add_argument("--results-root", type=Path, default=STUDY / "results")
    parser.add_argument("--isolation-root", type=Path, default=STUDY / "artifacts/isolation_v1")
    parser.add_argument("--skip-quality-scoring", action="store_true",
                        help="Verify output hashes and calculate throughput without dense-relative image metrics")
    args = parser.parse_args()
    protocol, identities = load_protocol_cases()
    all_receipts = collect_receipts(args.input_root)
    measured = [(path, row) for path, row in all_receipts if row.get("phase") == "F2" and row["status"] == "PASS"]
    valid_receipts = [row for _, row in measured]
    f1 = _read_isolation(args.isolation_root)
    result = {
        "schema_version": 1,
        "phase": "F2",
        "status": "BLOCKED",
        "decision": None,
        "f1_receipts": f1,
        "reason": _blocked_reason(f1, valid_receipts),
    }
    if args.skip_quality_scoring:
        result["quality"] = {"status": "SKIPPED"}
    try:
        if not valid_receipts:
            raise ValueError(result["reason"])
        _verify_observations(measured, protocol, identities, args.isolation_root)
        result.update(analyze_paired(valid_receipts, protocol, identities))
        result.pop("reason")
        if not args.skip_quality_scoring:
            result["quality"] = _quality_scores(measured)
        full = next(row for row in valid_receipts if row["arm"] == "full" and row["concurrency"] == 2)
        for row in valid_receipts:
            for field in ("device_uuid", "source_sha", "environment", "timing_boundary"):
                if row[field] != full[field]:
                    raise ValueError(f"Comparison arm mixes different {field} values")
        repetitions = list(range(result["paired_repetitions"]))
        result["arms"] = summarize_arms(valid_receipts)
        n1_p50 = {
            row["mode"]: row["latency_p50_seconds"]
            for row in result["arms"]
            if row["arm"] == "full" and row["concurrency"] == 1
        }
        if n1_p50.keys() >= {"serial", "concurrent"}:
            result["single_request_p50_ratio"] = n1_p50["concurrent"] / n1_p50["serial"]
        if not _has_required_controls(valid_receipts, repetitions, tuple(identities)):
            result.update(status="INCOMPLETE", decision=None, reason="缺少完整的 N=1 或 dense N=2 对照。")
    except (KeyError, OSError, ValueError) as error:
        result.update(status="BLOCKED", decision=None, reason=str(error))
    args.results_root.mkdir(parents=True, exist_ok=True)
    summary = args.results_root / "feasibility_summary.json"
    result.update(
        protocol=protocol,
        receipt_root=str(args.input_root.resolve()),
        summary_path=str(summary.resolve()),
        receipts=[
            {"path": str(path.resolve()), "status": row["status"], "sha256": sha256_file(path)}
            for path, row in all_receipts
        ],
        nonpassing_receipts=[
            {"path": str(path.resolve()), "status": row["status"], "failure": row.get("failure")}
            for path, row in all_receipts
            if row["status"] != "PASS"
        ],
        gpu_timelines=[row["gpu_timeline"] for _, row in all_receipts if "gpu_timeline" in row],
    )
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (args.results_root / "feasibility_report.md").write_text(render_report(result))
    print(f"{result['status']} {summary}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
