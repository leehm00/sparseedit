#!/usr/bin/env python3
"""Verify the two archived fresh v4 runs and write a new F0 anchor."""

import argparse
from datetime import UTC, datetime
from hashlib import file_digest, sha256
import json
from pathlib import Path
from math import isfinite
from os.path import relpath
from statistics import mean

STUDY = Path(__file__).resolve().parents[1]
BUNDLE = STUDY.parents[1] / "backend_reference/current_v4/backend_bundle"
VERSIONS = ("python", "torch", "triton", "vllm")


def file_hash(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(comparison, receipt):
    evidence_hashes = receipt["evidence_sha256"]

    def read(path, expected=None):
        digest = file_hash(path)
        require(expected is None or digest == expected, f"Hash mismatch for {path}")
        evidence_hashes[relpath(path, STUDY)] = digest
        return json.loads(path.read_text()) if path.suffix == ".json" else digest

    pair_info = read(comparison)
    root = comparison.parent
    first = read(root / "baseline_v1/completion.json", pair_info["first_repetition_receipt_sha256"])
    second_env = read(root / "baseline_v1_native_repetition_01/environment.json")
    first_env = read(root / "f0_environment.json")
    bundle = read(BUNDLE / "bundle.json", pair_info["bundle_sha256"])
    actual_protocol = json.loads((STUDY / "protocol.json").read_text())
    protocol = pair_info["protocol"]
    require(protocol == first["protocol"], "F0 evidence uses inconsistent protocols")
    require(all(actual_protocol.get(key) == value for key, value in protocol.items()), "Current frozen protocol differs")
    frozen_tree_hash = sha256(json.dumps(bundle["runtime_hashes"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(pair_info["source_sha"] == first["source_sha"] == frozen_tree_hash, "Frozen source identity differs")
    require(first["host"] == pair_info["host"] == "gpu-node3", "Host identity differs")
    for key in (*VERSIONS, "device_uuid", "device_name", "total_bytes"):
        require(first[key] == first_env[key] == second_env[key], f"F0 environment differs for {key}")
    require(first["device_uuid"] == pair_info["device_uuid"], "Comparison GPU differs")
    require(first["timing_boundary"] == pair_info["timing_boundary"], "Timing boundaries differ")
    require(first["completed_cases"] == len(first["cases"]) == len(pair_info["cases"]) == 3, "Expected three complete pairs")

    receipt.update(
        host=first["host"],
        device_uuid=first["device_uuid"],
        device_name=first["device_name"],
        environment={key: first[key] for key in VERSIONS},
        protocol=protocol,
        source_sha256=frozen_tree_hash,
        bundle_sha256=pair_info["bundle_sha256"],
        repetitions=[0, 1],
        timing_boundary=first["timing_boundary"],
    )
    cases = {row["case_tag"]: row for row in bundle["cases"]["calibration"]}
    for pair, first_summary in zip(pair_info["cases"], first["cases"], strict=True):
        tag = pair["case_tag"]
        frozen_case = cases.pop(tag)
        local_path = root / "baseline_v1" / tag / "manifest.json"
        native_path = root / pair["native_run_manifest"]
        local = read(local_path)
        native = read(native_path, pair["native_run_manifest_sha256"])
        require(local["case_tag"] == native["case_tag"] == first_summary["case_tag"] == tag, "Case tag mismatch")
        require(native["status"] == "PASS" and native["native_execution"] is True and native["warmup"] is False
                and native["audit"] is False and native["arm"] == "full" and native["repetition"] == 1, "Invalid native full receipt")
        for key, frozen_key in (("input_sha256", "prepared_sha256"), ("trace_sha256", "trace_sha256")):
            require(local[key] == native[key] == first_summary[key] == frozen_case[frozen_key], f"{tag}: {key} mismatch")
        for manifest in (local, native):
            request = manifest["request"]
            meta = request["runtime_metadata"]
            cfg = meta["run_config"]
            records = meta["step_records"]
            require(
                meta["runtime_variant"] == "graph_reuse"
                and cfg["active_token_count"] == protocol["K"]
                and cfg["generated_token_count"] == protocol["N"]
                and cfg["reference_token_count"] == protocol["reference_tokens"]
                and cfg["warmup_steps"] == protocol["warmup"]
                and cfg["refresh_interval"] == protocol["refresh_interval"]
                and cfg["true_cfg_scale"] == protocol["true_cfg_scale"]
                and cfg["latent_dtype"] == "torch.bfloat16"
                and cfg["sparseedit_collect_audit"] is False
                and cfg["sparseedit_avd_enabled"] is False
                and cfg["sparseedit_online_reuse"] == "off"
                and request["resolved_height"] == request["resolved_width"] == protocol["height"]
                and len(records) == protocol["steps"]
                and all(row.get("graph_replayed") for row in records if row["is_sparse_step"]),
                f"{tag}: v4 full graph protocol mismatch",
            )
        for key in ("fixed_mask_trace_sha256",):
            require(local["request"]["runtime_metadata"][key] == native["request"]["runtime_metadata"][key],
                    f"{tag}: executed mask trace differs")
        first_seconds = local["request"]["request_wall_seconds"]
        native_seconds = native["request"]["request_wall_seconds"]
        require(first_seconds == local["request_seconds"] == first_summary["request_seconds"], "Study timing mismatch")
        require(native_seconds == native["complete_request_seconds"], "Native timing mismatch")
        require(all(isfinite(value) and value > 0 for value in (first_seconds, native_seconds)), "Invalid timing")
        delta = abs(first_seconds / native_seconds - 1)
        require(delta <= .02, f"{tag}: fresh timing difference exceeds 2%")
        png_hash = read(local_path.with_name("output.png"))
        require(png_hash == read(native_path.with_name("output.png"))
                == local["output_png_sha256"] == native["output_sha256"], f"{tag}: PNG mismatch")
        receipt["cases"].append({
            "case_tag": tag, "input_sha256": local["input_sha256"], "trace_sha256": local["trace_sha256"],
            "request_seconds": [first_seconds, native_seconds], "absolute_relative_delta": delta,
            "output_png_sha256": png_hash, "peak_reserved_bytes": [
                manifest["request"]["memory"]["max_memory_reserved"] for manifest in (local, native)
            ],
        })

    require(not cases, "Missing calibration case")
    receipt.update(
        status="PASS", fresh_stability_status="PASS", requests=6, all_fresh_pngs_equal=True,
        max_absolute_relative_delta=max(row["absolute_relative_delta"] for row in receipt["cases"]),
        mean_request_seconds=[mean(row["request_seconds"][rep] for row in receipt["cases"]) for rep in range(2)],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if not output.is_relative_to(STUDY / "artifacts"):
        parser.error("Select a fresh directory below the study artifacts")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "FAILED", "phase": "F0", "fresh_stability_status": "FAILED",
        "historical_reproduction_status": "UNAVAILABLE",
        "historical_reproduction_reason": "Archived historical same-case full timings are unavailable.",
        "reference_kind": "fresh_same_case_native_v4_full",
        "evidence_sha256": {}, "cases": [],
    }
    try:
        verify(args.comparison.resolve(), receipt)
    except (ValueError, OSError, KeyError, TypeError) as error:
        receipt["failure"] = str(error)
        print(str(error))
    finally:
        receipt["finished_at_utc"] = datetime.now(UTC).isoformat()
        (output / "completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
