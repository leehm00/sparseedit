#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Single-command serial Hopper evaluation and result archive."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import socket
from statistics import mean
import subprocess
import sys
import traceback

from hopper_io import hashes, seal_result, sha, verified_assets, write

ROOT = Path(__file__).resolve().parent
STAGES = ("H1", "H2", "H3", "H4")
DEFAULT_MODELS = {
    "qwen": "/mnt/sda1/models/Qwen-Image-Edit",
    "flux": "/mnt/sda1/experiments/flux_a40_kontext_28step_20260912/models/FLUX.1-Kontext-dev",
    "longcat": "/mnt/sda1/models/meituan-longcat/LongCat-Image-Edit",
}


def selected_stages(stages):
    if len(stages) != len(set(stages)):
        raise ValueError("duplicate stage selection")
    return tuple(stage for stage in STAGES if stage in stages)


def parse_device(uuid, listing, info, apps):
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid):
        raise ValueError("a full physical GPU UUID is required")
    matches = [row for row in csv.reader(info.splitlines(), skipinitialspace=True) if row and row[0] == uuid]
    if len(matches) != 1:
        raise ValueError("the requested GPU UUID is not uniquely present on this machine")
    _, idx, model, memory, free, driver = matches[0]
    if not re.search(r"\bH(?:20|200)\b", model):
        raise ValueError(f"Track H accepts an H20/H200 GPU, got {model}")
    expected = f"GPU {idx}: {model} (UUID: {uuid})"
    if expected not in {line.strip() for line in listing.splitlines() if not line.startswith(" ")}:
        raise ValueError("nvidia-smi -L does not match physical index, model and UUID")
    if re.search(rf"^\s+MIG .*\(UUID:", listing.split(expected, 1)[1].split("\nGPU ", 1)[0], re.MULTILINE):
        raise ValueError("Track H requires a complete physical GPU without MIG slices")
    compute = []
    for record in csv.reader(apps.splitlines(), skipinitialspace=True):
        if not record or record[0].strip() == "gpu_uuid":
            continue
        if len(record) != 2 or not record[0].startswith("GPU-") or not record[1].strip().isdigit():
            raise ValueError(f"unrecognized compute-process record: {record!r}")
        compute.append({"gpu_uuid": record[0].strip(), "pid": int(record[1])})
    active = [p for p in compute if p["gpu_uuid"] == uuid]
    if active:
        raise ValueError(f"selected GPU has compute processes {active}")
    return dict(gpu_uuid=uuid, physical_index=int(idx), model=model, total_mib=int(memory), free_mib=int(free),
                driver_version=driver, device_list=listing, compute_processes=compute, host=socket.gethostname())


def check_device(uuid):
    def smi(*argv):
        return subprocess.check_output(["nvidia-smi", *argv], text=True)
    return parse_device(
        uuid, smi("-L"),
        smi("--query-gpu=uuid,index,name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"),
        smi("--query-compute-apps=gpu_uuid,pid", "--format=csv"),
    )


def stage_result(name, dest, returncode):
    path = dest / "completion.json"
    if not path.is_file():
        return dict(status="FAIL", returncode=returncode, reason="Stage produced no completion.json")
    data = json.loads(path.read_text())
    if data.get("phase") != name or not data.get("status"):
        raise ValueError(f"invalid stage receipt: {path}")
    status = "FAIL" if returncode and data["status"] == "PASS" else data["status"]
    return dict(
        status=status, returncode=returncode, completion_sha256=sha(path),
        coverage_gaps=data.get("coverage_gaps", data.get("gaps", data.get("unsupported"))),
    )


def summaries(data):
    requests = data.get("requests", [])
    if isinstance(requests, dict):
        requests = list(requests.values())
    groups = {}
    for row in requests:
        if (not isinstance(row, dict) or row.get("warmup") or row.get("status") not in (None, "PASS")
                or row.get("timing_role") not in (None, "measurement")):
            continue
        value = row.get("request_wall_s", row.get("request_wall_seconds"))
        if not isinstance(value, (int, float)) or value <= 0:
            continue
        grouping = (row.get("model", "—"), str(row.get("resolution", row.get("side", "—"))),
                    str(row.get("arm", row.get("method", "unknown"))))
        groups.setdefault(grouping, []).append(value)
    return [(group, len(values), mean(values)) for group, values in sorted(groups.items())]


def report(output, receipt):
    lines = [
        "# SparseEdit Track H 实验结果", "",
        f"总状态 **{receipt['status']}**", "",
        "| 阶段 | 实测状态 | 退出码 | 收据 |",
        "|---|---|---:|---|",
    ]
    for name, row in receipt["stages"].items():
        rel = output / name / "completion.json"
        link = f"[completion.json]({name}/completion.json)" if rel.is_file() else "—"
        lines.append(f"| {name} | {row['status']} | {row.get('returncode', '—')} | {link} |")
    for name, row in receipt["stages"].items():
        if row.get("coverage_gaps"):
            gaps = json.dumps(row["coverage_gaps"], ensure_ascii=False)
            lines += ["", f"{name} 缺项见该阶段收据。", "", f"`{gaps}`"]
        path = output / name / "completion.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        table = summaries(data)
        if table:
            lines += ["", f"## {name} 同步请求墙钟", "",
                      "| 模型 | 分辨率 | 配置 | 请求数 | 均值（s） |", "|---|---|---|---:|---:|"]
            for group, n, value in table:
                lines.append(f"| {' | '.join(group)} | {n} | {value:.3f} |")
        if name == "H4" and data.get("comparisons"):
            lines += ["", "## H4 样本外预测", "",
                      "| 模型 | 分辨率 | 配置 | 状态 | MAPE |", "|---|---|---|---|---:|"]
            for item in data["comparisons"]:
                error = f"{item['mape_percent']:.3f}%" if "mape_percent" in item else "—"
                lines.append(f"| {item['model']} | {item['resolution']} | {item['runtime']} | {item['status']} | {error} |")
    if receipt.get("error"):
        lines += ["", "## 启动或执行错误", "", "```text", receipt["error"].rstrip(), "```"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def progress(output, receipt):
    temp = output / ".session_progress.json.tmp"
    temp.write_text(json.dumps(receipt["stages"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(output / "session_progress.json")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-uuid")
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--run-mode", choices=("cal3", "formal"), default="formal")
    p.add_argument("--stages", choices=STAGES, nargs="+", default=list(STAGES))
    p.add_argument("--representative-sources", type=int, choices=(24, 48), default=24)
    p.add_argument("--dry-run", action="store_true")
    for name, path in DEFAULT_MODELS.items():
        p.add_argument(f"--{name}-model", type=Path, default=Path(path))
    p.add_argument("--qwen-python", type=Path)
    args = p.parse_args(argv)
    stages = selected_stages(args.stages)
    models = {name: str(getattr(args, name + "_model").resolve()) for name in DEFAULT_MODELS}
    if args.dry_run:
        with verified_assets(args.assets.resolve(), None) as (_, identity):
            print(json.dumps({"stages": {s: (ROOT / (s.lower() + ".py")).is_file() for s in stages},
                              "asset_files": len(identity["files"]), "model_paths_present": {
                                  name: Path(path).is_dir() for name, path in models.items()
                              }, "run_mode": args.run_mode}, indent=2))
        return 0
    if not args.gpu_uuid or args.output is None:
        p.error("real execution requires --gpu-uuid and --output")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    exclude = {"payload_data", "__pycache__", ".pytest_cache", ".git"}
    code = {
        path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(ROOT.rglob("*"))
        if path.is_file() and not path.is_symlink() and not (exclude & set(path.relative_to(ROOT).parts))
    }
    version = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    receipt = dict(
        phase="Track_H", status="FAIL", gpu_uuid=args.gpu_uuid, run_mode=args.run_mode, batch_size=1,
        started_utc=datetime.now(timezone.utc).isoformat(), stages={s: {"status": "PENDING"} for s in stages},
        source_sha256=code, git_commit=version.stdout.strip() if version.returncode == 0 else None,
    )
    try:
        receipt["before_lock"] = check_device(args.gpu_uuid)
        with Path(f"/tmp/gpu-{args.gpu_uuid}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt["under_lock"] = check_device(args.gpu_uuid)
            with verified_assets(args.assets.resolve(), output.parent) as (assets, identity):
                receipt["inputs"] = identity
                config = dict(models=models, gpu_uuid=args.gpu_uuid, assets=str(assets),
                              prior_results=str(output), run_mode=args.run_mode,
                              representative_sources=args.representative_sources)
                write(output / "run_config.json", config)
                progress(output, receipt)
                for stage in stages:
                    dest = output / stage
                    runner = ROOT / (stage.lower() + ".py")
                    if not runner.is_file():
                        receipt["stages"][stage] = {"status": "BLOCKED", "reason": "stage runner missing"}
                        progress(output, receipt)
                        continue
                    witness = check_device(args.gpu_uuid)
                    write(output / f"{stage}_device.json", witness)
                    receipt["stages"][stage] = {"status": "RUNNING"}
                    progress(output, receipt)
                    print(f"START {stage} {output / (stage + '.log')}", flush=True)
                    python = (args.qwen_python or sys.executable) if stage == "H1" else sys.executable
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu_uuid if stage != "H4" else "",
                           "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                           "TRANSFORMERS_OFFLINE": "1", "SPARSEEDIT_SINGLE_REQUEST": "1"}
                    with (output / f"{stage}.log").open("x") as log:
                        child = subprocess.run([str(python), "-u", str(runner), "--config",
                                                str(output / "run_config.json"), "--output", str(dest)],
                                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    receipt["stages"][stage] = stage_result(stage, dest, child.returncode)
                    progress(output, receipt)
                    print(f"END {stage} {receipt['stages'][stage]['status']}", flush=True)
                receipt["after_stages"] = check_device(args.gpu_uuid)
        outcomes = [item["status"] for item in receipt["stages"].values()]
        receipt["status"] = "PASS" if all(s == "PASS" for s in outcomes) else "PARTIAL"
    except KeyboardInterrupt:
        receipt.update(status="INTERRUPTED", error="Interrupted by operator")
    except Exception:
        receipt["error"] = traceback.format_exc()
        print(receipt["error"], file=sys.stderr, flush=True)
    finally:
        for state in receipt["stages"].values():
            if state["status"] == "RUNNING":
                state["status"] = "INTERRUPTED" if receipt["status"] == "INTERRUPTED" else "FAIL"
            elif state["status"] == "PENDING":
                state["status"] = "BLOCKED"
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        progress(output, receipt)
        report(output, receipt)
        archive, checksum = seal_result(output, receipt)
        print(f"{receipt['status']}\n{output / 'report.md'}\n{archive}\n{checksum}", flush=True)
    return {"PASS": 0, "PARTIAL": 2, "INTERRUPTED": 130}.get(receipt["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
