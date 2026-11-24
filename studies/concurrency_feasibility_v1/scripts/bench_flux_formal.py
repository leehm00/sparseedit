#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the frozen v4 FLUX serial three-arm study on an available H200."""

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from bench_concurrency import _environment, read_json, sha256_file, source_tree_sha
from check_env import query
from run_h200_phase_a import classify_h200

PINS = {
    "backend_bundle/bundle.json": "e559de093f959c3985faf4ed911d49efb4d10b89da5a4bcf53f1a3fd4563776d",
    "code/backend_reference.py": "427ef2669922d0f2c912c51dfa9a1ccbd598c829138cd1811e0ccd4765d85c0a",
    "code/full_method.py": "9c01addcf471aef2f302aae18d9aa293902ae19e4675035f2b3d4d142a3c453e",
    "code/analyze_backend.py": "331cab32ed0049c24b04e6355fff5bd0322a17dbc0af3687035f2d6fc6aee150",
}
PHASES = ("cal3", "calibration_timing", "formal288")


def verify_assets(root):
    for relative, expected in PINS.items():
        path = root / relative
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen FLUX formal asset differs: {relative}")
    return root / "backend_bundle"


def device(gpu):
    (row,) = query(gpu, "gpu", "index,uuid,name,memory.total,memory.free,driver_version")
    processes = query(gpu, "compute-apps", "gpu_uuid,pid,used_memory")
    result = classify_h200(row, processes, gpu)
    if result["status"] != "PASS":
        raise RuntimeError("; ".join(result["reasons"]))
    return result


def attestation(path, output_root):
    return {"path": str(path.relative_to(output_root)), "sha256": sha256_file(path)}


def run(args):
    root = args.formal_assets_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    native = output / "native"
    summary = output / "formal_summary.json"
    receipt = {
        "phase": "E1_flux_formal_768",
        "status": "FAILED",
        "coverage": "single_request_direct_standard_full",
        "through": args.through,
        "device_uuid": args.gpu_uuid,
        "frozen_bundle_sha256": PINS["backend_bundle/bundle.json"],
        "source_sha": source_tree_sha(),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "commands": [],
    }
    try:
        bundle = verify_assets(root)
        if not args.model.is_dir():
            raise FileNotFoundError(f"Local FLUX model directory is missing: {args.model}")
        receipt.update(device_inventory=device(args.gpu_uuid), environment=_environment())
        env = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NCCL_CUMEM_ENABLE": "0",
        }
        env.pop("PYTORCH_ALLOC_CONF", None)
        with TemporaryDirectory(prefix="sparseedit-flux-formal-locks-") as locks:
            # Both the archived launcher and the study jobs use the same UUID lease.
            lock = Path(locks) / f"{args.gpu_uuid}.lock"
            lock.symlink_to(Path("/tmp") / f"sparseedit-{args.gpu_uuid}.lock")
            command = [
                sys.executable, "-u", str(root / "code/backend_reference.py"),
                "--bundle", str(bundle), "--model", str(args.model.resolve()),
                "--gpu", args.gpu_uuid, "--locks", locks, "--output", str(native),
                "--through", args.through, "--timeout", str(args.timeout_s),
            ]
            receipt["commands"].append(command)
            result = subprocess.run(command, env=env, check=False)
            if result.returncode:
                resource_status = native / "resource_status.json"
                if resource_status.exists():
                    receipt["native_resource_status"] = attestation(resource_status, output)
                raise RuntimeError(f"Frozen FLUX native runner exited with code {result.returncode}")
        completed = native / args.through / "completion.json"
        raw = read_json(completed)
        if raw.get("status") != "PASS" or raw.get("native_execution") is not True:
            raise ValueError("Missing passing native formal phase receipt")
        receipt["native_receipt"] = attestation(completed, output)
        if args.through == "formal288":
            command = [
                sys.executable, "-u", str(root / "code/analyze_backend.py"), str(native), str(summary),
            ]
            receipt["commands"].append(command)
            subprocess.run(command, env=env, check=True)
            report = read_json(summary)
            if report.get("status") != "PASS" or report.get("sources") != 48:
                raise ValueError("Formal analysis does not contain 48 held-out sources")
            receipt.update(summary=attestation(summary, output), sources=report["sources"])
        receipt["status"] = "PASS"
    except FileNotFoundError as error:
        receipt.update(status="UNAVAILABLE", failure={"type": type(error).__name__, "message": str(error)})
    except Exception as error:
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        receipt["finished_at_utc"] = datetime.now(UTC).isoformat()
        with (output / "completion.json").open("x") as stream:
            json.dump(receipt, stream, allow_nan=False, indent=2)
            stream.write("\n")
    print(f"{receipt['status']} {output / 'completion.json'}", flush=True)
    return int(receipt["status"] != "PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-assets-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--through", choices=PHASES, default="formal288")
    parser.add_argument("--timeout-s", type=int, default=21600, help="Per-phase timeout for the archived runner")
    args = parser.parse_args()
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", args.gpu_uuid):
        parser.error("Supply the physical GPU UUID from nvidia-smi -L")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
