#!/usr/bin/env python3
"""Check the selected device without opening a CUDA context."""

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version


def query(uuid, name, fields):
    command = [
        "nvidia-smi", f"--id={uuid}", f"--query-{name}={fields}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--require-gpu-name", default="NVIDIA A40")
    args = parser.parse_args()

    (index, uuid, name, total_mib, free_mib), = query(
        args.gpu_uuid, "gpu", "index,uuid,name,memory.total,memory.free"
    )
    processes = query(args.gpu_uuid, "compute-apps", "gpu_uuid,pid,used_memory")
    compute_apps = [
        {"gpu_uuid": gpu_uuid, "pid": int(pid), "used_memory_mib": memory_mib}
        for gpu_uuid, pid, memory_mib in processes
    ]
    free_bytes = int(free_mib) * 2**20
    reasons = []
    if uuid != args.gpu_uuid or not (name == args.require_gpu_name or name.startswith(args.require_gpu_name + " ")):
        reasons.append(f"Selected device differs from the requested {args.require_gpu_name}.")
    if free_bytes <= 40 * 2**30:
        reasons.append("F0 requires more than 40 GiB of free GPU memory.")
    if compute_apps:
        reasons.append("The selected GPU has existing compute processes.")

    result = {
        "phase": "baseline_preflight",
        "status": "BLOCKED" if reasons else "PASS",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "python_path": sys.executable,
        "torch": version("torch"),
        "triton": version("triton"),
        "vllm": version("vllm"),
        "device_uuid": uuid,
        "device_index": int(index),
        "device_name": name,
        "total_bytes": int(total_mib) * 2**20,
        "free_bytes": free_bytes,
        "compute_processes": compute_apps,
        "reasons": reasons,
    }
    print(json.dumps(result, indent=2))
    return 1 if reasons else 0


if __name__ == "__main__":
    sys.exit(main())
