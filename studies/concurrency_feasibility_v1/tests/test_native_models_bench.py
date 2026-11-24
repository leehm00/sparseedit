import json
import sys
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "scripts"))

from bench_native_models import identification, main, measurement_cells, model_protocol


@pytest.mark.parametrize(
    ("kind", "model_class", "side", "guidance", "cfg"),
    [
        ("flux", "FluxKontextPipeline", 768, 1.0, 4.0),
        ("longcat", "LongCatImageEditPipeline", 1024, 3.5, 3.5),
        ("qwen", "QwenImageEditPipeline", 1024, 1.0, 4.0),
    ],
)
def test_native_dense_protocol_is_model_specific(kind, model_class, side, guidance, cfg):
    protocol = model_protocol(kind)
    assert protocol["model_class"] == model_class
    assert protocol["height"] == protocol["width"] == side
    assert protocol["guidance_scale"] == guidance
    assert protocol["true_cfg_scale"] == cfg
    assert protocol["steps"] == 28
    assert protocol["seed"] == 110
    assert protocol["cpu_offload"] is False


def test_only_flux_schedules_real_multi_input_native_batches():
    for kind in ("longcat", "qwen"):
        rows = list(measurement_cells(kind, ("a", "b", "c"), 2))
        assert len(rows) == 6
        assert all(len(row.cases) == 1 and row.mode == "serial" for row in rows)
    rows = list(measurement_cells("flux", ("a", "b", "c"), 2))
    assert len(rows) == 12
    assert {row.cases for row in rows[6:] if row.repetition == 0} == {("a", "b"), ("b", "c"), ("c", "a")}
    assert all(row.mode == "true_batch" for row in rows[6:])


def test_model_identity_digests_weights_and_rejects_the_wrong_pipeline(tmp_path):
    (tmp_path / "model_index.json").write_text('{"_class_name":"QwenImageEditPipeline"}')
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    shard = transformer / "diffusion_pytorch_model.safetensors"
    shard.write_bytes(b"first checkpoint")
    first = identification("qwen", tmp_path)
    assert "transformer/diffusion_pytorch_model.safetensors" in first["files"]
    shard.write_bytes(b"second checkpoint")
    second = identification("qwen", tmp_path)
    assert first["model_sha256"] != second["model_sha256"]
    with pytest.raises(ValueError, match="LongCatImageEditPipeline"):
        identification("longcat", tmp_path)


def test_missing_local_model_writes_a_standalone_failure_receipt(tmp_path):
    root = tmp_path / "output"
    code = main([
        "--model-kind", "qwen", "--model", str(tmp_path / "absent"),
        "--gpu-uuid", "GPU-example", "--output-root", str(root),
    ])
    assert code == 1
    receipt = json.loads((root / "completion.json").read_text())
    assert receipt["status"] == "BLOCKED"
    assert receipt["model_kind"] == "qwen"
    assert receipt["arms"] == []
    assert "local model" in receipt["failure"]


def test_flux_batch_rejects_duplicate_outputs_even_with_valid_shape_witness(tmp_path, monkeypatch):
    from PIL import Image
    import torch

    import bench_native_models as runner
    from bench_concurrency import Measurement

    protocol = model_protocol("flux")
    cases = {
        tag: ({"case_tag": tag, "prepared_sha256": tag * 64, "source_pixel_sha256": tag * 64}, None, None, {})
        for tag in ("a", "b")
    }
    monkeypatch.setattr(runner, "resize_case", lambda *args: Image.new("RGB", (768, 768)))
    monkeypatch.setattr(runner, "make_request", lambda *args, **kwargs: object())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    for fn in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, fn, lambda: 4096)
    timings = [{"request_seconds": 1.0, "cuda_event_seconds": 0.9}] * 2
    monkeypatch.setattr(
        runner, "_native_forward",
        lambda *args, **kwargs: ([Image.new("RGB", (768, 768)) for _ in range(2)], timings, [2] * 56),
    )
    cell = Measurement("dense", "true_batch", 0, 0, ("a", "b"))
    result = runner._run_cell("flux", cell, cases, protocol, (None, None, None), tmp_path / "batch")
    assert result["status"] == "FAILED"
    assert "identical" in result["failure"]
    assert json.loads((tmp_path / "batch/completion.json").read_text())["status"] == "FAILED"


def test_flux_accepts_an_explicit_1024_dense_probe():
    protocol = model_protocol("flux", 1024)
    assert protocol["height"] == protocol["width"] == 1024
    assert protocol["steps"] == 28
    assert protocol["true_cfg_scale"] == 4.0
    assert model_protocol("flux")["height"] == 768
