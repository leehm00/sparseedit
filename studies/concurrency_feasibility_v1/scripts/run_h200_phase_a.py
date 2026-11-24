#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the frozen Phase A study on one H200 and preserve every stage receipt."""

import argparse
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

from bench_concurrency import MODEL, PYTHON, STUDY, load_protocol_cases, source_tree_sha
from check_env import query
from summarize_phase_a import read_json

EXPECTED = {"python": "3.12.14", "torch": "2.11.0", "triton": "3.6.0", "vllm": "0.26.0"}
SCRIPT_ROOT = STUDY / "scripts"
ARTIFACTS = STUDY / "artifacts"


def classify_h200(row, processes, expected_uuid):
    index, uuid, name, total, free, driver = row
    reasons = []
    if uuid != expected_uuid or not re.match(r"^NVIDIA H200(?:\s|$)", name) or int(total) < 120 * 1024:
        reasons.append("Selected UUID is not a full H200.")
    if processes:
        reasons.append("Selected GPU has a pre-existing compute process.")
    return {
        "status": "BLOCKED" if reasons else "PASS", "device_uuid": uuid, "device_name": name,
        "device_index": int(index), "driver_version": driver, "total_bytes": int(total) * 2**20,
        "free_bytes": int(free) * 2**20, "external_processes": processes, "reasons": reasons,
    }


def phase_a_stages(graph_status, eager_status, *, throughput_only=False):
    f1_pass = "PASS" in (graph_status, eager_status)
    if throughput_only:
        return ["F0", "Task2", "F1", *(["F2"] if f1_pass else [])]
    gated = ["F4", "F2"] if f1_pass else []
    return ["F0", "Task2", "F1", *gated, "F3", "F5"]


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)
        output.write("\n")


def frozen_preflight(gpu, root):
    receipt = {
        "status": "BLOCKED", "phase": "resource_inventory", "host": socket.gethostname(),
        "device_uuid": gpu, "checked_at_utc": datetime.now(UTC).isoformat(),
        "python_path": sys.executable, "repetition": 0,
        "timing_boundary": "Read-only device and file inventory before CUDA initialization.",
        "source_sha": source_tree_sha(),
    }
    try:
        if Path(sys.executable).absolute() != PYTHON:
            raise ValueError(f"Run from the plan's Python path {PYTHON}")
        environment = {"python": sys.version.split()[0], **{name: version(name) for name in ("torch", "triton", "vllm")}}
        receipt["environment"] = environment
        if environment != EXPECTED:
            raise ValueError(f"Frozen environment mismatch: {environment!r}")
        protocol, identities = load_protocol_cases()
        receipt.update(protocol=protocol, cases=list(identities.values()))
        if not MODEL.is_dir():
            raise FileNotFoundError(f"Frozen model is required at {MODEL}")
        (row,) = query(gpu, "gpu", "index,uuid,name,memory.total,memory.free,driver_version")
        receipt.update(classify_h200(row, query(gpu, "compute-apps", "gpu_uuid,pid,used_memory"), gpu))
    except (OSError, ValueError, PackageNotFoundError, subprocess.SubprocessError) as error:
        receipt["failure"] = str(error)
    write(root / "resource_inventory/completion.json", receipt)
    return receipt


class PhaseA:
    def __init__(self, root, gpu, capacity_max, *, f1_timeout_s=1800, throughput_only=False):
        self.root = root
        self.gpu = gpu
        self.capacity_max = capacity_max
        self.f1_timeout_s = f1_timeout_s
        self.throughput_only = throughput_only
        self.env = {
            **os.environ, "PYTHONPATH": str(STUDY / "source"), "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        }
        self.commands = []

    def invoke(self, label, script, *arguments, timeout=None):
        stdout = self.root / "logs" / f"{label}.log"
        stdout.parent.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-u", str(SCRIPT_ROOT / script), *(str(arg) for arg in arguments)]
        print(f"{label} → {stdout}", flush=True)
        with stdout.open("xb") as output:
            try:
                result = subprocess.run(argv, cwd=STUDY, env=self.env, stdout=output,
                                        stderr=subprocess.STDOUT, timeout=timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                output.write(f"\nStage timed out after {timeout} seconds. Kept all partial receipts.\n".encode())
                code = 124
        self.commands.append({
            "label": label, "command": argv, "returncode": code,
            "log": str(stdout.relative_to(self.root)), "timeout_seconds": timeout,
        })
        return code

    def gpu_stage(self, label, script, output, *options, timeout=None):
        code = self.invoke(label, script, "--gpu-uuid", self.gpu, "--output-root", output, *options, timeout=timeout)
        path = output / "completion.json"
        if not path.is_file():
            status = "TIMEOUT" if code == 124 else "FAILED"
            reason = f"Stage exceeded {timeout} seconds." if code == 124 else "Runner did not write a receipt."
            report = read_json(self.root / "resource_inventory/completion.json") if (self.root / "resource_inventory/completion.json").is_file() else {}
            phase = "F1" if label.startswith("F1_") else label
            receipt = {
                "status": status, "phase": phase, "device_uuid": self.gpu, "failure": reason,
                **{key: report[key] for key in ("source_sha", "environment", "protocol", "cases") if key in report},
            }
            if phase == "F1":
                receipt["runtime_lane"] = label.removeprefix("F1_")
            write(path, receipt)
        return read_json(path)

    def run(self):
        f0 = self.root / "f0_anchor_v1"
        self.invoke("F0", "finalize_f0.py", "--comparison", ARTIFACTS / "node3_f0/comparison.json", "--output-root", f0)
        anchor = read_json(f0 / "completion.json")
        common = ["--verify-full-model-hashes"]
        equivalent = self.root / "executor_equivalence_v1"
        isolation = self.root / "isolation_v1"
        f1_report = None
        lane = "private_graph"
        if anchor["status"] == "PASS":
            task2 = self.gpu_stage("Task2", "run_isolation.py", equivalent,
                                   "--lane", lane, "--attempt", "1", "--equivalence", *common,
                                   timeout=self.f1_timeout_s)
            if task2["status"] == "PASS":
                for index, candidate in enumerate(("private_graph", "private_eager"), start=1):
                    lane = candidate
                    lane_output = isolation / lane
                    f1_report = self.gpu_stage(f"F1_{lane}", "run_isolation.py", lane_output,
                                               "--lane", lane, "--attempt", index, *common,
                                               timeout=self.f1_timeout_s)
                    if f1_report["status"] == "PASS":
                        break
        isolation.mkdir(parents=True, exist_ok=True)
        if f1_report is None:
            f1_report = {"phase": "F1", "status": "BLOCKED", "device_uuid": self.gpu,
                         "reason": "F1 requires F0 and the Task 2 three-case graph equivalence."}
        else:
            f1_report = {**f1_report, "pairs": [
                {**pair, "path": str(Path(pair["path"]).relative_to(isolation / lane))}
                for pair in f1_report.get("pairs", [])
            ]}
            f1_report["pair_receipt_directory"] = lane
        write(isolation / "completion.json", f1_report)

        timings = self.root / "concurrency_v1"
        results = self.root / "results"
        stages = phase_a_stages(f1_report["status"], None, throughput_only=self.throughput_only)
        f1_pass = "F2" in stages
        if "F4" in stages:
            self.gpu_stage("F4", "probe_template_concurrency.py", self.root / "template_probe_v1",
                           "--isolation-root", isolation / lane, *common)
        if f1_pass:
            bench_args = ("--matrix", "--gpu-uuid", self.gpu, "--repetitions", "2", "--model", MODEL,
                          "--isolation-root", isolation / lane, "--output-root", timings)
            self.invoke("F2_initial", "bench_concurrency.py", *bench_args)

        analysis_args = ("--input-root", timings, "--isolation-root", isolation / lane, "--results-root", results)
        if self.throughput_only:
            analysis_args += ("--skip-quality-scoring",)
        self.invoke("F2_analysis_initial", "analyze_concurrency.py", *analysis_args)
        f2_summary = results / "feasibility_summary.json"
        if f1_pass and f2_summary.is_file() and read_json(f2_summary)["status"] == "REPEAT_REQUIRED":
            self.invoke("F2_repeat", "bench_concurrency.py", *bench_args, "--start-repetition", 2)
            self.invoke("F2_analysis_repeat", "analyze_concurrency.py", *analysis_args)

        if anchor["status"] == "PASS" and "F3" in stages:
            self.gpu_stage("F3", "analyze_capacity.py", self.root / "capacity_v1",
                           "--require-gpu-name", "H200", "--memory-arm", lane, "--f0-anchor", f0 / "completion.json",
                           "--max-concurrency", self.capacity_max, *common)
        summary_args = ("--throughput-only",) if self.throughput_only else ()
        self.invoke("phase_a_summary", "summarize_phase_a.py",
                    "--run-root", self.root, "--results-root", results, "--receipt", f"F0={f0 / 'completion.json'}",
                    *summary_args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capacity-max", type=int, choices=range(2, 17), default=16)
    parser.add_argument("--f1-timeout-s", type=int, default=1800, help="Per-lane watchdog, including model hash and load")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--throughput-only", action="store_true",
                        help="Run numerical admission and F2, omitting image scoring, F3 and F4")
    args = parser.parse_args()
    if args.f1_timeout_s <= 0:
        parser.error("--f1-timeout-s must be positive")
    root = args.output_root.resolve()
    if not root.is_relative_to(ARTIFACTS) or root == ARTIFACTS:
        parser.error("Select a fresh directory below study artifacts")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", args.gpu_uuid):
        parser.error("Supply the physical GPU UUID from nvidia-smi -L")
    root.mkdir(parents=True, exist_ok=False)
    inventory = frozen_preflight(args.gpu_uuid, root)
    if inventory["status"] != "PASS" or args.preflight_only:
        print(f"{inventory['status']} {root / 'resource_inventory/completion.json'}")
        return 0 if inventory["status"] == "PASS" else 1
    workflow = PhaseA(root, args.gpu_uuid, args.capacity_max, f1_timeout_s=args.f1_timeout_s,
                      throughput_only=args.throughput_only)
    try:
        workflow.run()
    finally:
        report = root / "results/phase_a_summary.json"
        summary = read_json(report) if report.is_file() else {}
        write(root / "completion.json", {
            "status": summary.get("status", "INCOMPLETE"), "phase": "phase_a_h200",
            "scope": "throughput" if args.throughput_only else "phase_a",
            "decision": summary.get("decision"), "device_uuid": args.gpu_uuid,
            "source_sha": inventory["source_sha"], "environment": inventory["environment"],
            "protocol": inventory["protocol"], "cases": inventory["cases"],
            "commands": workflow.commands, "finished_at_utc": datetime.now(UTC).isoformat(),
        })
    return 0 if summary.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
