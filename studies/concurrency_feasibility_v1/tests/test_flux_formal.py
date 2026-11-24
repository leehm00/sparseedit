import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "scripts"))

import bench_flux_formal as formal


def test_checks_pins_before_executing_any_archive_script(tmp_path, monkeypatch):
    files = {
        "backend_bundle/bundle.json": "{}",
        "code/backend_reference.py": "print('runner')",
        "code/full_method.py": "# helper",
    }
    for relative, content in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    monkeypatch.setattr(formal, "PINS", {
        relative: formal.sha256_file(tmp_path / relative) for relative in files
    })
    assert formal.verify_assets(tmp_path) == tmp_path / "backend_bundle"
    (tmp_path / "code/backend_reference.py").write_text("changed")
    with pytest.raises(ValueError, match="backend_reference.py"):
        formal.verify_assets(tmp_path)


def test_runs_independently_and_keeps_raw_and_analysis_receipts(tmp_path, monkeypatch):
    root = tmp_path / "formal"
    out = tmp_path / "run"
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(formal, "verify_assets", lambda path: path / "backend_bundle")
    monkeypatch.setattr(formal, "device", lambda gpu: {"device_uuid": gpu})
    monkeypatch.setattr(formal, "_environment", lambda: {"python": "test"})
    monkeypatch.setattr(formal, "source_tree_sha", lambda: "source")
    commands = []

    def child(command, *, env, check):
        commands.append(command)
        assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
        if "--bundle" in command:
            assert command[command.index("--through") + 1] == "formal288"
            assert Path(command[command.index("--locks") + 1], "GPU-example.lock").is_symlink()
            receipt = out / "native/formal288/completion.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps({"status": "PASS", "native_execution": True}))
        else:
            (out / "formal_summary.json").write_text(json.dumps({"status": "PASS", "sources": 48}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(formal.subprocess, "run", child)
    args = argparse.Namespace(
        formal_assets_root=root, output_root=out, gpu_uuid="GPU-example",
        model=model, through="formal288", timeout_s=900,
    )
    assert formal.run(args) == 0
    receipt = json.loads((out / "completion.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["native_receipt"]["path"] == "native/formal288/completion.json"
    assert receipt["summary"]["path"] == "formal_summary.json"
    assert len(commands) == 2


def test_missing_assets_write_completion_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(formal, "_environment", lambda: {"python": "test"})
    monkeypatch.setattr(formal, "source_tree_sha", lambda: "source")
    model = tmp_path / "model"
    model.mkdir()
    args = argparse.Namespace(
        formal_assets_root=tmp_path / "missing", output_root=tmp_path / "run",
        gpu_uuid="GPU-example", model=model, through="formal288", timeout_s=300,
    )
    assert formal.run(args) == 1
    receipt = json.loads((args.output_root / "completion.json").read_text())
    assert receipt["status"] == "UNAVAILABLE"
    assert "bundle.json" in receipt["failure"]["message"]
