#!/usr/bin/env python3
"""Inventory available experiment nodes without opening a CUDA context."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys

STUDY = Path(__file__).resolve().parents[1]
ENV_PROBE = ('import sys,importlib.metadata as m; print(sys.executable,sys.version.split()[0],'
             '*[m.version(x) for x in ("torch","triton","vllm")],sep=",")')
GPU_FIELDS = "index,name,uuid,memory.total,memory.free,driver_version"
APP_FIELDS = "gpu_uuid,pid,used_memory"


def report_node(node, python_path):
    local = node == socket.gethostname()

    def run(*argv):
        command = argv if local else ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", node, shlex.join(argv))
        result = subprocess.run(command, text=True, capture_output=True, timeout=15)
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    def rows(category, fields):
        code, stdout, stderr = run("nvidia-smi", f"--query-{category}={fields}", "--format=csv,noheader,nounits")
        if code:
            raise RuntimeError(stderr)
        return [dict(zip(fields.split(","), row, strict=True))
                for row in csv.reader(stdout.splitlines(), skipinitialspace=True)]

    report = {"node": node, "devices": []}
    try:
        report["devices"] = rows("gpu", GPU_FIELDS)
        apps = rows("compute-apps", APP_FIELDS)
        for device in report["devices"]:
            device["total_bytes"] = int(device.pop("memory.total")) * 2**20
            device["free_bytes"] = int(device.pop("memory.free")) * 2**20
            device["compute_processes"] = [
                app for app in apps if app["gpu_uuid"] == device["uuid"]
            ]
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        report["inventory_error"] = str(error)
        return report
    code, out, err = run(python_path, "-c", ENV_PROBE)
    if code:
        report["protocol_environment"] = {"status": "UNAVAILABLE", "python_path": python_path, "detail": err}
        # Historical F0 ran with a node-local absolute path. Inventory it read-only.
        alternate = python_path.replace("/mnt/sda1/experiments", "/mnt/sdb/homie")
        alt_code, out, _ = run(alternate, "-c", ENV_PROBE)
        if alt_code == 0:
            report["historical_node_local_environment"] = dict(zip(
                ("python_path", "python", "torch", "triton", "vllm"), out.split(","), strict=True
            ))
    else:
        report["protocol_environment"] = {"status": "PASS", **dict(zip(
            ("python_path", "python", "torch", "triton", "vllm"), out.split(","), strict=True
        ))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", nargs="+", default=[socket.gethostname(), "gpu-node2", "gpu-node3"])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not root.is_relative_to(STUDY / "artifacts"):
        parser.error("Select a fresh study artifact directory")
    root.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=len(args.nodes)) as workers:
        reports = list(workers.map(lambda node: report_node(node, sys.executable), args.nodes))
    matched = [device for report in reports for device in report["devices"]
               if any(name in device["name"] for name in ("H20", "H200"))]
    receipt = {
        "status": "PASS" if matched else "PARTIAL",
        "phase": "resource_inventory",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "protocol": json.loads((STUDY / "protocol.json").read_text()),
        "nodes": reports,
        "large_memory_candidates": matched,
        "large_memory_mapping_status": "VERIFIED" if matched else "UNLOCATED",
    }
    (root / "completion.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
