#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect independent SparseEdit experiments during one H200 allocation."""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from bench_concurrency import BUNDLE_ROOT, MODEL, PYTHON, STUDY, sha256_file, source_tree_sha
from check_env import query
from run_h200_phase_a import EXPECTED, classify_h200

ARTIFACTS = STUDY / "artifacts"
MODEL_ROOT = STUDY.parents[2].parent / "models"
PASSING_STATUSES = {"PASS", "COMPLETE"}
PERFORMANCE_FIELDS = {"arm", "mode", "concurrency", "images_per_minute",
                      "latency_mean_seconds", "latency_p50_seconds"}


@dataclass(frozen=True)
class Stage:
    name: str
    script: str
    args: tuple
    model: Path | None = None
    required_files: tuple[Path, ...] = ()
    frozen_runtime: bool = False
    timeout_seconds: int = 10800


def build_stages(root, gpu, longcat, qwen, *, formal_assets=None, repetitions=2, scaling_repetitions=2, capacity_max=16):
    common = ("--gpu-uuid", gpu)

    def dense(model_kind, path, *, size=None):
        name = f"{model_kind}_dense" if size is None else f"{model_kind}_dense_{size}"
        args = common + ("--model-kind", model_kind, "--model", path, "--repetitions", repetitions)
        if size is not None:
            args += ("--size", size)
        return Stage(name, "bench_native_models.py", args, path, timeout_seconds=2700)

    def scaling(lane):
        return Stage(
            f"flux_scaling_{lane}", "bench_flux_scaling.py",
            common + ("--model", MODEL, "--lane", f"private_{lane}",
                      "--counts", 1, 2, 4, 8, "--repetitions", scaling_repetitions),
            MODEL, timeout_seconds=5400,
        )

    plan = [
        dense("flux", MODEL),
        dense("longcat", longcat),
        dense("qwen", qwen),
        Stage("flux_f2", "run_h200_phase_a.py", common + ("--throughput-only",), MODEL),
        scaling("eager"),
        dense("flux", MODEL, size=1024),
        scaling("graph"),
        Stage("flux_f6_packing", "bench_packing_diagnostic.py", common, timeout_seconds=900),
        Stage(
            "flux_native_serving", "audit_serving_baseline.py",
            ("run-native", *common, "--verify-full-model-hashes", "--require-gpu-name", "NVIDIA H200"),
            MODEL, frozen_runtime=True, timeout_seconds=3600,
        ),
        Stage(
            "flux_templates", "probe_template_concurrency.py",
            common + ("--isolation-root", root / "flux_f2/isolation_v1",
                      "--verify-full-model-hashes", "--diagnostic-without-f1"),
            MODEL, timeout_seconds=2700,
        ),
        Stage(
            "flux_capacity", "analyze_capacity.py",
            common + ("--require-gpu-name", "H200", "--memory-arm", "private_eager",
                      "--max-concurrency", capacity_max, "--verify-full-model-hashes"),
            MODEL,
        ),
    ]
    formal_assets = formal_assets or STUDY.parents[1] / "offline_formal/sparseedit_h200_flux_formal_48_v1"
    plan.append(Stage(
        "flux_formal_48", "bench_flux_formal.py",
        common + ("--model", MODEL, "--formal-assets-root", formal_assets), MODEL,
        tuple(formal_assets / path for path in (
            "backend_bundle/bundle.json", "code/backend_reference.py", "code/full_method.py", "code/analyze_backend.py",
        )),
        timeout_seconds=21600,
    ))
    return plan


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    pending.replace(path)


def missing_input(stage):
    if stage.model is not None and not stage.model.is_dir():
        return f"Local model directory is unavailable: {stage.model}"
    for path in stage.required_files:
        if not path.is_file():
            return f"Required experiment asset is unavailable: {path}"
    return None


def run_stages(root, stages, execute, *, timeout_seconds, budget_seconds=None):
    started = time.monotonic()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results = []
    for stage in stages:
        stage_started = time.monotonic()
        remaining = None if budget_seconds is None else budget_seconds - (stage_started - started)
        output = root / stage.name
        log = logs / f"{stage.name}.log"
        receipt = output / "completion.json"
        returncode = None
        status = "FAILED"
        reason = None
        result = {}
        if remaining is not None and remaining <= 0:
            status, reason = "NOT_RUN_BUDGET", "The operator's H200 window ended."
        elif reason := missing_input(stage):
            status = "UNAVAILABLE"
        else:
            time_limit = stage.timeout_seconds if timeout_seconds is None else timeout_seconds
            if remaining is not None:
                time_limit = min(time_limit, remaining)
            print(f"Starting {stage.name}: {log}", flush=True)
            try:
                returncode = execute(stage, output, log, time_limit)
                if receipt.is_file():
                    result = json.loads(receipt.read_text())
                    status = result["status"]
                    reason = result.get("reason") or result.get("failure")
                elif returncode == 124:
                    status, reason = "TIMEOUT", "Stage watchdog expired. Partial receipts and logs are preserved."
                else:
                    reason = "Stage exited before writing its completion receipt."
            except (KeyError, OSError, TypeError, ValueError) as error:
                reason = str(error)
        if not receipt.is_file():
            write_json(receipt, {"status": status, "phase": stage.name, "failure": reason})
        stats = result.get("arms") or result.get("summaries") or []
        if stage.name == "flux_f2":
            f2 = output / "results/feasibility_summary.json"
            if f2.is_file():
                try:
                    stats = json.loads(f2.read_text()).get("arms") or []
                except (OSError, ValueError):
                    pass
        performance = [
            row for row in stats if isinstance(row, dict) and PERFORMANCE_FIELDS <= row.keys()
        ] if isinstance(stats, list) else []
        results.append({
            "name": stage.name, "status": status, "decision": result.get("decision"),
            "reason": reason, "receipt": receipt.relative_to(root).as_posix(),
            "log": log.relative_to(root).as_posix() if log.is_file() else None,
            "returncode": returncode, "wall_seconds": time.monotonic() - stage_started,
            "performance": performance, "evidence_scope": result.get("evidence_scope"),
        })
        write_json(root / "session_progress.json", {"stages": results})
        print(f"{stage.name}: {status}", flush=True)
    return results


def run_child(stage, output, log, timeout):
    source = BUNDLE_ROOT / "runtime" if stage.frozen_runtime else STUDY / "source"
    path = [source, STUDY / "scripts"]
    env = {
        **os.environ, "PYTHONPATH": os.pathsep.join(map(str, path)), "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    command = [sys.executable, "-u", str(STUDY / "scripts" / stage.script),
               *(str(arg) for arg in stage.args), "--output-root", str(output)]
    with log.open("xb") as stream:
        stream.write((json.dumps(command) + "\n").encode())
        stream.flush()
        with subprocess.Popen(command, cwd=STUDY, env=env, stdout=stream, stderr=subprocess.STDOUT,
                              start_new_session=True) as child:
            try:
                code = child.wait(timeout=timeout)
                if code:
                    terminate_stage(child)
                return code
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                terminate_stage(child)
                if isinstance(error, KeyboardInterrupt):
                    raise
                stream.write(f"\nStage exceeded {timeout:.1f}s.\n".encode())
                return 124


def terminate_stage(child):
    """Let worker-owning launchers release their leases before advancing."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    deadline = time.monotonic() + 35
    while time.monotonic() < deadline:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            child.wait()
            return
        time.sleep(0.1)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def preflight(gpu):
    environment = {"python": sys.version.split()[0], **{
        name: version(name) for name in ("torch", "triton", "vllm")
    }}
    if Path(sys.executable).absolute() != PYTHON or environment != EXPECTED:
        raise ValueError(f"Use the frozen environment at {PYTHON}")
    (row,) = query(gpu, "gpu", "index,uuid,name,memory.total,memory.free,driver_version")
    result = classify_h200(row, query(gpu, "compute-apps", "gpu_uuid,pid,used_memory"), gpu)
    return {**result, "environment": environment}


def render(results):
    lines = ["# H200 实验运行汇总", "", "| 实验 | 结果 | F2 判定 | 收据 | 日志 |",
             "|---|---|---|---|---|"]
    for row in results:
        log = f"[log]({row['log']})" if row["log"] else "—"
        status = row["status"] + ("（诊断）" if row.get("evidence_scope") == "diagnostic" else "")
        lines.append(f"| {row['name']} | {status} | {row['decision'] or '—'} | "
                     f"[completion]({row['receipt']}) | {log} |")
    if any(row.get("performance") for row in results):
        lines.extend(["", "## 吞吐与请求延迟", "",
                      "| 实验 | Arm | Mode | N | images/min | 平均 / p50 (s) |",
                      "|---|---|---|---:|---:|---:|"])
        for stage in results:
            for row in stage.get("performance", []):
                lines.append(
                    f"| {stage['name']} | {row['arm']} | {row['mode']} | {row['concurrency']} | "
                    f"{row['images_per_minute']:.3f} | {row['latency_mean_seconds']:.3f} / "
                    f"{row['latency_p50_seconds']:.3f} |"
                )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--longcat-model", type=Path, default=MODEL_ROOT / "meituan-longcat/LongCat-Image-Edit")
    parser.add_argument("--qwen-model", type=Path, default=MODEL_ROOT / "Qwen-Image-Edit")
    parser.add_argument("--formal-flux-assets", type=Path)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--scaling-repetitions", type=int, default=2)
    parser.add_argument("--capacity-max", type=int, choices=range(2, 17), default=16)
    parser.add_argument("--stage-timeout-minutes", type=int,
                        help="Override each stage watchdog; defaults depend on the expected workload")
    parser.add_argument("--budget-minutes", type=int, help="Total time window supplied by the operator")
    parser.add_argument("--stage", action="append", help="Select only named experiments; repeat to select more")
    parser.add_argument("--list", action="store_true", help="Print the run order and file readiness without opening CUDA")
    args = parser.parse_args()
    if min(args.repetitions, args.scaling_repetitions) < 1:
        parser.error("Repetitions must be positive")
    if args.stage_timeout_minutes is not None and args.stage_timeout_minutes < 1:
        parser.error("Stage timeout must be positive")
    if args.budget_minutes is not None and args.budget_minutes < 1:
        parser.error("Budget must be positive")
    root = args.output_root.resolve()
    if not root.is_relative_to(ARTIFACTS) or root == ARTIFACTS:
        parser.error("Select a fresh output directory under study artifacts")
    stages = build_stages(
        root, args.gpu_uuid, args.longcat_model.resolve(), args.qwen_model.resolve(),
        formal_assets=args.formal_flux_assets.resolve() if args.formal_flux_assets else None,
        repetitions=args.repetitions, scaling_repetitions=args.scaling_repetitions, capacity_max=args.capacity_max,
    )
    if args.stage:
        unknown = set(args.stage) - {stage.name for stage in stages}
        if unknown:
            parser.error(f"Unknown stages: {', '.join(sorted(unknown))}")
        stages = [stage for stage in stages if stage.name in args.stage]
    if args.list:
        for stage in stages:
            print(f"{stage.name}: {missing_input(stage) or 'READY'}")
        return 0
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", args.gpu_uuid):
        parser.error("Supply a physical GPU UUID from nvidia-smi -L")
    root.mkdir(parents=True, exist_ok=False)
    try:
        inventory = preflight(args.gpu_uuid)
    except (OSError, ValueError, PackageNotFoundError, subprocess.SubprocessError) as error:
        inventory = {"status": "BLOCKED", "reason": str(error)}
    write_json(root / "resource_inventory.json", inventory)
    receipt = {
        "phase": "h200_window", "device_uuid": args.gpu_uuid, "host": socket.gethostname(),
        "source_sha": source_tree_sha(), "runner_sha256": sha256_file(Path(__file__)),
        "planned_stages": [stage.name for stage in stages],
        "started_at_utc": datetime.now(UTC).isoformat(), "status": "BLOCKED",
    }
    if inventory["status"] != "PASS":
        write_json(root / "completion.json", receipt)
        return 1
    with Path(f"/tmp/sparseedit-window-{args.gpu_uuid}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            write_json(root / "completion.json", {**receipt, "failure": "A session already holds this UUID."})
            return 1
        try:
            results = run_stages(
                root, stages, run_child,
                timeout_seconds=None if args.stage_timeout_minutes is None else args.stage_timeout_minutes * 60,
                budget_seconds=None if args.budget_minutes is None else args.budget_minutes * 60,
            )
        except KeyboardInterrupt:
            write_json(root / "completion.json", {**receipt, "status": "INTERRUPTED",
                                                   "progress": "session_progress.json"})
            return 130
    has_evidence = any(row["status"] in PASSING_STATUSES or row["status"] == "PARTIAL" for row in results)
    status = "COMPLETE" if all(row["status"] in PASSING_STATUSES for row in results) else "PARTIAL"
    write_json(root / "completion.json", {
        **receipt, "status": status, "finished_at_utc": datetime.now(UTC).isoformat(), "stages": results,
    })
    (root / "report.md").write_text(render(results))
    return 0 if has_evidence else 1


if __name__ == "__main__":
    raise SystemExit(main())
