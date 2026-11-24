import json
from pathlib import Path
import sys

import pytest

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "scripts"))

from package_h200 import transfer_member
from run_h200_phase_a import classify_h200, phase_a_stages
from summarize_phase_a import summarize


def test_h200_preflight_rejects_a40_and_occupied_h200():
    name, uuid = "NVIDIA H200", "GPU-e085e0e1-a283-82f2-325b-01981b51f984"
    row = ["0", uuid, name, "143771", "143030", "580.95.05"]
    assert classify_h200(row, [], uuid)["status"] == "PASS"
    assert classify_h200(row, [[uuid, "123", "2"]], uuid)["status"] == "BLOCKED"
    a40 = ["0", uuid, "NVIDIA A40", "46068", "45487", "580.95.05"]
    assert classify_h200(a40, [], uuid)["status"] == "BLOCKED"


def test_f1_gate_enables_f2_f4_only_after_pass():
    stages = phase_a_stages("CAPACITY_BLOCKED", "FAILED")
    assert "F2" not in stages and "F4" not in stages
    assert "F3" in stages and "F5" in stages
    stages = phase_a_stages("PASS", "PASS")
    assert stages.index("F4") < stages.index("F2") < stages.index("F3")


@pytest.mark.parametrize("path", ["/etc/passwd", "../source/a.py", "path/../../a", "."])
def test_transfer_path_rejects_outside_members(path):
    with pytest.raises(ValueError):
        transfer_member(path)


def test_phase_a_report_has_no_decision_without_f1(tmp_path):
    evidence = tmp_path / "run"
    evidence.mkdir()
    stage = evidence / "F1.json"
    stage.write_text(json.dumps({"phase": "F1", "status": "CAPACITY_BLOCKED", "device_uuid": "GPU-example"}))
    f2 = evidence / "f2.json"
    f2.write_text(json.dumps({"status": "PASS", "decision": "GO"}))
    summary = summarize({"F1": stage, "F2": f2}, evidence)
    assert summary["decision"] is None and summary["phases"]["F1"]["status"] == "CAPACITY_BLOCKED"
    assert summary["phases"]["F2"]["status"] == "BLOCKED"
    assert len(summary["phases"]["F1"]["sha256"]) == 64


def test_batched_reference_images_preserve_individual_preprocessing():
    import torch
    from diffusers.image_processor import VaeImageProcessor
    from PIL import Image

    from bench_concurrency import prepare_dense_reference_batch

    processor = VaeImageProcessor(vae_scale_factor=16)
    images = [Image.new("RGB", (32, 32), color) for color in ((8, 123, 185), (241, 121, 16))]
    batch = prepare_dense_reference_batch(processor, images)
    actual = processor.preprocess(processor.resize(batch, 32, 32), 32, 32)
    want = torch.cat([processor.preprocess(processor.resize(image, 32, 32), 32, 32) for image in images])
    assert torch.equal(actual, want)


def test_phase_a_report_preserves_f5_arms(tmp_path):
    path = tmp_path / "f5.json"
    path.write_text(json.dumps({
        "status": "PASS",
        "baselines": {
            "native_vllm_omni_async": {"status": "RUNNABLE", "execution_status": "SERIAL_BASELINE"},
            "flashps": {"status": "INCOMPATIBLE"},
        },
    }))
    summary = summarize({"F5": path}, tmp_path)
    assert summary["phases"]["F5"]["baselines"] == [
        {"name": "native_vllm_omni_async", "status": "RUNNABLE", "execution_status": "SERIAL_BASELINE"},
        {"name": "flashps", "status": "INCOMPATIBLE"},
    ]


def test_f1_watchdog_records_timeout_and_releases_child(tmp_path, monkeypatch):
    import subprocess
    from run_h200_phase_a import PhaseA

    def expired(args, **kwargs):
        assert kwargs["timeout"] == 10
        raise subprocess.TimeoutExpired(args, 10)

    monkeypatch.setattr(subprocess, "run", expired)
    runner = PhaseA(tmp_path, "GPU-example", 3, f1_timeout_s=10)
    receipt = runner.gpu_stage("F1_private_graph", "run_isolation.py", tmp_path / "graph", timeout=10)
    assert receipt["status"] == "TIMEOUT"
    assert runner.commands[0]["returncode"] == 124
    assert "10" in (tmp_path / "logs/F1_private_graph.log").read_text()
