import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_serving_baseline import classify_worker_timeline, validate_baseline_report


@pytest.fixture
def comparison(tmp_path):
    records = {}
    for name, seconds in (("serial", 2.0), ("concurrent", 1.5)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"status": "PASS", "batch_wall_seconds": seconds, "concurrency": 2}))
        records[name] = {
            "model": "FLUX.1-Kontext-dev",
            "workload": {"cases": ["first", "second"], "requests": 2, "steps": 28},
            "algorithm": "SparseEdit full",
            "backend": "frozen v4",
            "concurrency": name,
            "timing": "synchronized complete request wall clock",
            "quality": "bitwise",
            "hardware": "GPU-test",
            "status": "RUNNABLE",
            "measured_local": {"receipt": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        }
    return {
        "baselines": records,
        "speedups": [{"reference": "serial", "candidate": "concurrent", "speedup": 4 / 3}],
    }


def test_admits_a_hash_bound_local_speedup(comparison):
    validate_baseline_report(comparison)


@pytest.mark.parametrize("field", ["model", "workload", "timing", "hardware"])
def test_rejects_incomparable_speedup(comparison, field):
    comparison["baselines"]["concurrent"][field] = "incompatible"
    with pytest.raises(ValueError, match=field):
        validate_baseline_report(comparison)


def test_rejects_external_speedup(comparison):
    row = comparison["baselines"]["concurrent"]
    row["reported_external"] = {"source": "10.1145/3767295.3769379"}
    row.pop("measured_local")
    with pytest.raises(ValueError, match="local receipt"):
        validate_baseline_report(comparison)


def test_rejects_receipt_modified_after_comparison(comparison):
    Path(comparison["baselines"]["concurrent"]["measured_local"]["receipt"]).write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        validate_baseline_report(comparison)


def test_rejects_unsubstantiated_speedup(comparison):
    comparison["speedups"][0]["speedup"] = 2
    with pytest.raises(ValueError, match="recomputed"):
        validate_baseline_report(comparison)


def test_async_submission_and_one_worker_stream_classify_as_serial():
    client = [
        {"request_id": "r1", "submitted_ns": 1, "completed_ns": 17},
        {"request_id": "r2", "submitted_ns": 2, "completed_ns": 30},
    ]
    worker = [
        {"request_id": "r1-worker", "pid": 300, "stream_id": 0, "cuda_begin_ms": 2, "cuda_end_ms": 10},
        {"request_id": "r2-worker", "pid": 300, "stream_id": 0, "cuda_begin_ms": 11, "cuda_end_ms": 24},
    ]
    assert classify_worker_timeline(client, worker) == "SERIAL_BASELINE"
    worker[1]["cuda_begin_ms"] = 5
    assert classify_worker_timeline(client, worker) == "UNRESOLVED"
    worker[1]["stream_id"] = 1
    assert classify_worker_timeline(client, worker) == "OVERLAPPING_WORKER_WINDOWS"


def test_native_probe_accepts_a_separate_checkpoint_startup_timeout():
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_serving_baseline.py"
    proc = subprocess.run(
        [
            sys.executable, str(script), "run-native", "--gpu-uuid", "invalid",
            "--output-root", "unused", "--startup-timeout-s", "1800",
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 2
    assert "Use a physical GPU UUID" in proc.stderr


def test_native_probe_can_write_a_h200_session_receipt(tmp_path, monkeypatch):
    import audit_serving_baseline as audit

    monkeypatch.setattr(audit, "STUDY", tmp_path)
    (tmp_path / "protocol.json").write_text("{}")
    output = tmp_path / "artifacts/h200_window_01/flux_native_serving"
    def run_native(args, root, receipt):
        assert root == output
        assert args.verify_full_model_hashes and args.require_gpu_name == "NVIDIA H200"
        receipt.update(execution_status="SERIAL_BASELINE")
    monkeypatch.setattr(audit, "run_native", run_native)
    monkeypatch.setattr(sys, "argv", [
        "audit_serving_baseline.py", "run-native", "--gpu-uuid", "GPU-test",
        "--output-root", str(output), "--verify-full-model-hashes", "--require-gpu-name", "NVIDIA H200",
    ])
    assert audit.main() == 0
    assert json.loads((output / "completion.json").read_text())["execution_status"] == "SERIAL_BASELINE"
