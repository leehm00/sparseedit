#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Index and summarize the SparseEdit Phase A evidence on an offline machine."""

import argparse
import hashlib
import json
import os
from pathlib import Path

PHASES = ("F0", "Task2", "F1", "F2", "F3", "F4", "F5")
STATUS_LABELS = {
    "PASS": "通过", "FAIL": "未通过", "FAILED": "失败", "BLOCKED": "受阻",
    "TIMEOUT": "超时", "CAPACITY_BLOCKED": "显存受阻", "PARTIAL": "部分完成", "UNSUPPORTED": "不支持",
    "UNRUN": "待执行", "INCOMPLETE": "证据不完整", "REPEAT_REQUIRED": "需追加重复",
}


def read_json(path):
    return json.loads(path.read_text())


def evidence_entry(path, root):
    return {
        "receipt": os.path.relpath(path.resolve(), root.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def summarize(paths, root):
    rows = {}
    data = {}
    for phase in PHASES:
        path = paths.get(phase)
        if path is None or not path.is_file():
            rows[phase] = {"status": "UNRUN"}
            continue
        receipt = read_json(path)
        data[phase] = receipt
        rows[phase] = {
            **evidence_entry(path, root),
            "status": receipt["status"],
            **{key: receipt[key] for key in ("device_uuid", "host", "failure", "reason") if key in receipt},
        }

    f0 = data.get("F0", {})
    rows["F0"].update({key: f0[key] for key in ("requests", "mean_request_seconds", "max_absolute_relative_delta",
                                              "historical_reproduction_status") if key in f0})
    task2 = data.get("Task2", {})
    rows["Task2"].update({key: task2[key] for key in ("requests", "latency_offset_requires_profile") if key in task2})
    f1 = data.get("F1", {})
    if f1.get("pairs") or f1.get("status") == "PASS":
        pairs = []
        root_path = paths["F1"].parent
        for pair in f1.get("pairs", []):
            original = Path(pair["path"])
            selected = root_path / f1.get("pair_receipt_directory", "")
            relocated = (selected / original.parent.name / original.name) if original.is_absolute() else selected / original
            path = relocated if relocated.is_file() else original
            pair_data = read_json(path) if path.is_file() else None
            pairs.append({
                "status": pair["status"], "verified": pair_data is not None and pair_data["status"] == pair["status"],
                **(evidence_entry(path, root) if path.is_file() else {"receipt": str(original)}),
            })
        rows["F1"]["pairs"] = pairs
        if f1["status"] == "PASS" and (len(pairs) != 3 or not all(row["verified"] and row["status"] == "PASS" for row in pairs)):
            rows["F1"].update(status="INCOMPLETE", reason="缺少三对能够复核的 F1 PASS 收据。")

    decision = None
    if rows["F1"]["status"] == "PASS":
        f2 = data.get("F2", {})
        if f2.get("status") == "PASS" and f2.get("decision") in {"GO", "WEAK", "NO_GO"}:
            decision = f2["decision"]
        for key in ("point_estimate", "confidence_interval_95", "paired_repetitions", "decision",
                    "bootstrap_draws", "bootstrap_seed"):
            if key in f2:
                rows["F2"][key] = f2[key]
    else:
        rows["F2"].update(status="BLOCKED", reason="F2 采集要求相同协议的 F1 隔离通过。")
        rows["F4"].update(status="BLOCKED", reason="模板实验要求 F1 隔离通过。")
    f3 = data.get("F3", {})
    rows["F3"].update({key: f3[key] for key in ("fixed_n", "maximum_n", "admission") if key in f3})
    f4 = data.get("F4", {})
    if "structure_only" in f4:
        rows["F4"]["structure_only"] = f4["structure_only"]
    f5 = data.get("F5", {})
    if "baselines" in f5:
        rows["F5"]["baselines"] = [
            {"name": name, **{key: row[key] for key in ("status", "execution_status") if key in row}}
            for name, row in f5["baselines"].items()
        ]
    return {
        "schema_version": 1, "decision": decision,
        "status": "COMPLETE" if decision is not None and all(
            row["status"] in {"PASS", "UNSUPPORTED"} for row in rows.values()
        ) else "INCOMPLETE",
        "phases": rows,
    }


def render(summary):
    lines = [
        "# SparseEdit Phase A 实验状态", "",
        f"F2 判定 **{summary['decision'] or '待定'}**。", "",
        "| 阶段 | 状态 | 收据 |", "|---|---|---|",
    ]
    for phase, item in summary["phases"].items():
        status = item["status"]
        path = item.get("receipt")
        link = f"[completion]({path})" if path else "—"
        lines.append(f"| {phase} | {STATUS_LABELS.get(status, status)}（`{status}`） | {link} |")
    lines.append("")
    f0 = summary["phases"]["F0"]
    if f0.get("mean_request_seconds"):
        means = " 和 ".join(f"{value:.3f} s" for value in f0["mean_request_seconds"])
        lines.append(f"- **F0** 两轮均值为 {means}，逐例最大相对差 {f0['max_absolute_relative_delta']:.3%}。")
    f2 = summary["phases"]["F2"]
    if f2["status"] == "PASS" and "point_estimate" in f2 and "confidence_interval_95" in f2:
        low, high = f2["confidence_interval_95"]
        lines.append(
            f"- **F2** 配对吞吐比 {f2['point_estimate']:.3f}×，95% 区间 [{low:.3f}×, {high:.3f}×]，"
            f"{f2['paired_repetitions']} 次配对重复。"
        )
    capacity = summary["phases"]["F3"].get("maximum_n", {})
    if "last_successful_n" in capacity:
        lines.append(
            f"- **F3** 最后成功 N={capacity['last_successful_n']}，"
            f"首次 OOM N={capacity.get('first_oom_n')}，预测 N={capacity.get('predicted_n')}。"
        )
    f5 = summary["phases"]["F5"]
    if f5.get("baselines"):
        labels = [
            f"{row['name']}={row.get('execution_status') or row.get('status')}"
            for row in f5["baselines"]
        ]
        lines.append(f"- **F5** {'，'.join(labels)}。")
    for phase, row in summary["phases"].items():
        reason = row.get("reason", row.get("failure"))
        if reason:
            if isinstance(reason, dict):
                reason = reason.get("message", str(reason))
            lines.append(f"- **{phase}** {reason}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--receipt", action="append", default=[], metavar="PHASE=PATH")
    args = parser.parse_args()
    study = args.study.resolve()
    run = args.run_root or study / "artifacts"
    paths = {
        "F0": study / "artifacts/f0_anchor_v1/completion.json",
        "Task2": run / "executor_equivalence_v1/completion.json",
        "F1": run / "isolation_v1/completion.json",
        "F2": (run / "results" if args.run_root else study / "results") / "feasibility_summary.json",
        "F3": run / "capacity_v1/completion.json",
        "F4": run / "template_probe_v1/completion.json",
        "F5": study / "artifacts/serving_baseline_v1/completion.json",
    }
    for item in args.receipt:
        phase, sep, path = item.partition("=")
        if not sep or phase not in PHASES or not path:
            parser.error(f"Expected PHASE=PATH for {item!r}")
        paths[phase] = Path(path)
    args.results_root.mkdir(parents=True, exist_ok=True)
    summary = summarize(paths, args.results_root)
    (args.results_root / "phase_a_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.results_root / "phase_a_report.md").write_text(render(summary))
    print(f"{summary['status']} {args.results_root / 'phase_a_report.md'}")


if __name__ == "__main__":
    main()
