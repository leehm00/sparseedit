# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UUID = "GPU-12345678-1234-1234-1234-123456789abc"


def test_cuda_hidden_cli_writes_failed_receipt(tmp_path):
    config = tmp_path / "config.json"
    output = tmp_path / "receipt"
    config.write_text(json.dumps({"models": {"qwen": "/unused"}, "gpu_uuid": UUID, "run_mode": "cal3"}))
    result = subprocess.run(
        [sys.executable, str(ROOT / "h1.py"), "--config", str(config), "--output", str(output)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    receipt = json.loads((output / "completion.json").read_text())
    assert result.returncode == 1
    assert receipt["phase"] == "H1"
    assert receipt["status"] == "FAILED"
    assert receipt["requests"] == []
    assert "visible" in receipt["error"]["message"]


def test_case_inventory_keeps_holdouts_out_of_calibration():
    from track_h_qwen.contracts import inventory

    if not (ROOT / "payload_data" / "h1" / "cohort_bindings.json").is_file():
        pytest.skip("private H1 bundle assets are unavailable")
    cal3 = inventory("cal3")
    formal = inventory("formal")
    assert len(cal3["primary"]) == 3
    assert len(formal["primary"]) == 48
    assert len(formal["session"]) == 5
    assert {r["source_pixel_sha256"] for r in cal3["primary"]}.isdisjoint(
        r["source_pixel_sha256"] for r in formal["primary"]
    )
    assert len({r["source_pixel_sha256"] for r in formal["primary"]}) == 48


def test_l3_same_source_uses_vision_and_vae_separately_then_evicts():
    from track_h_qwen.contracts import SessionCache

    cache = SessionCache()
    assert cache.begin(("first", 1024, 1024, "bfloat16")) is False
    # A cold request evaluates both CFG vision calls symmetrically.
    assert cache.get("vision", (1, "bf16"), lambda: "vision") == ("vision", False)
    assert cache.get("vision", (1, "bf16"), lambda: "vision") == ("vision", False)
    assert cache.get("vae", (2, "bf16"), lambda: "vae") == ("vae", False)
    assert cache.begin(("first", 1024, 1024, "bfloat16")) is True
    assert cache.get("vision", (1, "bf16"), lambda: "changed") == ("vision", True)
    assert cache.get("vae", (2, "bf16"), lambda: "changed") == ("vae", True)
    with pytest.raises(ValueError, match="layout"):
        cache.get("vision", (2, "bf16"), lambda: "shape changed")
    assert cache.begin(("second", 1024, 1024, "bfloat16")) is False
    assert cache.get("vision", (1, "bf16"), lambda: "other source") == ("other source", False)


def test_graph_requires_exact_real_replays_for_both_cfg_branches():
    from track_h_qwen.contracts import validate_graph

    evidence = {
        "graph_requested": True,
        "graph_captures": {"positive": 1, "negative": 1},
        "graph_replays": {"positive": 19, "negative": 19},
        "graph_fallbacks": [],
    }
    validate_graph(evidence, requested=True)
    evidence["graph_replays"]["negative"] = 18
    with pytest.raises(ValueError, match="replay"):
        validate_graph(evidence, requested=True)


def test_artifact_verification_reads_bytes(tmp_path):
    from track_h_qwen.contracts import verify_artifact

    (tmp_path / "output.png").write_bytes(b"fixture output")
    expected = hashlib.sha256(b"fixture output").hexdigest()
    verify_artifact(tmp_path, "output.png", expected)
    (tmp_path / "output.png").write_bytes(b"changed output")
    with pytest.raises(ValueError, match="SHA"):
        verify_artifact(tmp_path, "output.png", expected)
    with pytest.raises(ValueError, match="path"):
        verify_artifact(tmp_path, "../output.png", expected)


def test_resident_invariant_rejects_cpu_parameters_and_offload_hooks():
    from types import SimpleNamespace

    from track_h_qwen.native import resident_placement

    class Tensor:
        device = "cuda:0"
        dtype = "bfloat16"

        def numel(self):
            return 16

        def element_size(self):
            return 2

    class Component:
        def __init__(self):
            self.weight = Tensor()

        def named_parameters(self):
            return [("weight", self.weight)]

        def named_buffers(self):
            return []

        def named_modules(self):
            return [("", self)]

    pipeline = SimpleNamespace(transformer=Component(), text_encoder=Component(), vae=Component())
    assert resident_placement(pipeline, "bfloat16")["transformer"]["parameter_bytes"] == 32
    pipeline.vae.weight.device = "cpu"
    with pytest.raises(ValueError, match="resident"):
        resident_placement(pipeline, "bfloat16")
    pipeline.vae.weight.device = "cuda:0"
    pipeline.vae._hf_hook = object()
    with pytest.raises(ValueError, match="offload"):
        resident_placement(pipeline, "bfloat16")


def test_l3_session_requires_cold_then_three_exact_hits():
    from track_h_qwen.contracts import validate_session_pair

    ref = {
        "output_sha256": "final",
        "conditioning": {"prompt": "a", "vae": "b"},
        "cache_events": {"vision": [False, False], "vae": [False]},
    }
    cached = {**ref, "cache_events": {"vision": [True, True], "vae": [True]}}
    validate_session_pair(ref, cached, expect_hit=True)
    with pytest.raises(ValueError, match="parity"):
        validate_session_pair(ref, {**cached, "conditioning": {"prompt": "b", "vae": "b"}}, expect_hit=True)
    with pytest.raises(ValueError, match="hit"):
        validate_session_pair(ref, cached, expect_hit=False)


def test_native_warmup_receipt_is_explicitly_excluded_from_measurements(tmp_path, monkeypatch):
    from PIL import Image

    import h1

    config = tmp_path / "config.json"
    config.write_text("{}")
    run = h1.Run(config, tmp_path)
    run.receipt["model"] = {"runtime_file_sha256": "model-fingerprint"}
    run.receipt["source_sha256"] = {}
    case = {
        "case_id": "fixture", "split": "calibration", "image": "fake.png",
        "input_sha256": "input-fingerprint", "source_pixel_sha256": "pixel-fingerprint",
        "prompt": "Apply the requested edit.",
    }
    monkeypatch.setattr(h1, "native_request", lambda *a, **k: (
        Image.new("RGB", (1024, 1024)), {"request_wall_seconds": 1.0}, None, None,
    ))
    row, _, _ = run.request(None, None, case, "warmup", mode="dense")
    assert row["timing_role"] == "warmup"
    assert (tmp_path / row["output_path"]).is_file()


def test_phase_ledger_requires_exact_nine_full_nineteen_sparse_steps():
    from track_h_qwen.contracts import timing_summary

    rows = [{"segment": segment, "cuda_seconds": 0.5} for segment in
            ("vl_encode", "vl_encode", "reference_vae_encode", "vae_decode")]
    rows += [{"segment": "denoise_step", "step": i, "action": "full" if i < 8 or i == 23 else "sparse",
              "cuda_seconds": 0.125 if i < 8 or i == 23 else 0.0625}
             for i in range(28)]
    assert timing_summary(rows, dense=False) == {
        "full_steps": 9, "sparse_steps": 19, "full_cuda_seconds": 1.125, "sparse_cuda_seconds": 1.1875,
    }
    rows[4 + 23]["action"] = "sparse"
    with pytest.raises(ValueError, match="schedule"):
        timing_summary(rows, dense=False)


def test_calibration_jobs_are_source_balanced_and_precede_heldout_repeats():
    from track_h_qwen.schedule import calibration_jobs

    sources = [{"case_id": str(i), "split": "calibration"} for i in range(3)]
    jobs = calibration_jobs(sources)
    assert [j.arm for j in jobs[:4]] == ["warmup", "fresh_dense", "fresh_dense", "fresh_dense"]
    for budget in (10, 20, 35):
        for arm in ("direct_sparse", "ready_eager", "full_sparsegraph"):
            assert {j.case["case_id"] for j in jobs if j.arm == f"{arm}{budget}"} == {"0", "1", "2"}
    assert [j.case["case_id"] for j in jobs if j.arm == "full_sparsegraph40"] == ["0"]


def test_formal_jobs_mirror_source_and_arm_order_while_producing_once():
    from track_h_qwen.schedule import heldout_jobs

    sources = [{"case_id": str(i), "split": "test"} for i in range(48)]
    jobs = heldout_jobs(sources)
    producers = [j for j in jobs if j.repetition is None]
    assert [j.case["case_id"] for j in producers] == [str(i) for i in range(48)]
    assert {j.arm for j in producers} == {"direct_collect20"}
    first = [(j.case["case_id"], j.arm) for j in jobs if j.repetition == 0]
    second = [(j.case["case_id"], j.arm) for j in jobs if j.repetition == 1]
    assert len(first) == 48 * 4
    assert len(set(first)) == len(first)
    assert first[0] == ("0", "fresh_dense")
    assert first[1] == ("0", "direct_sparse20")
    assert second == first[::-1]


def test_h4_observation_excludes_cold_first_sparse_step_and_uses_full_step_total():
    from track_h_qwen.contracts import latency_observation

    row = {
        "case_id": "source", "source_pixel_sha256": "pixel-sha", "arm": "full_sparsegraph20",
        "split": "calibration", "repetition": 0, "budget_percent": 20, "timing_role": "measurement",
        "request_wall_seconds": 5.0, "denoising_wall_seconds": 4.0, "full_steps": 9, "sparse_steps": 19,
        "full_cuda_seconds": 1.8, "manifest_path": "request/manifest.json", "manifest_sha256": "manifest-hash",
        "segments": [{"segment": "denoise_step", "action": "sparse", "cuda_seconds": seconds}
                     for seconds in [1.0] + [0.0625] * 18],
    }
    observation = latency_observation(row)
    assert observation["model"] == "qwen"
    assert observation["resolution"] == 1024
    assert observation["runtime"] == "full_sparsegraph"
    assert observation["source_id"] == "pixel-sha"
    assert observation["ratio"] == 0.2
    assert observation["request_wall_s"] == 5.0
    assert observation["denoise_wall_s"] == 4.0
    assert observation["full_steps"] == 9
    assert observation["sparse_steps"] == 19
    assert observation["reuse_steps"] == 0
    assert observation["full_device_s"] == 1.8
    assert observation["steady_sparse_device_s"] == 0.0625
    assert latency_observation({**row, "timing_role": "admission"}) is None
    assert latency_observation({**row, "split": "test"}) is None
    assert latency_observation({**row, "budget_percent": 40}) is None


def test_heldout_reverse_pass_reloads_the_matching_frozen_source_trace(tmp_path, monkeypatch):
    from PIL import Image

    import h1
    from track_h_qwen.frozen.qwen_image.sparseedit_native_trace import QwenFixedMaskTrace, QwenTraceStep
    from track_h_qwen.schedule import heldout_jobs

    config = tmp_path / "config.json"
    config.write_text("{}")
    run = h1.Run(config, tmp_path)
    cases = [
        {
            "case_id": str(i), "split": "test", "image": f"{i}.png", "prompt": "Apply the requested edit.",
            "input_sha256": f"input-{i}", "source_pixel_sha256": f"pixels-{i}",
        }
        for i in range(2)
    ]
    run.receipt["model"] = {"runtime_file_sha256": "model-fingerprint"}
    run.receipt["source_sha256"] = {}

    def native(_torch, _pipeline, case, *, mode, trace, identity, **kwargs):
        if mode != "dense" and trace is None:
            rows = [
                QwenTraceStep(i, "full", [0, 1, 2, 3]) if i < 8 or i == 23
                else QwenTraceStep(i, "sparse", [int(case["case_id"])])
                for i in range(28)
            ]
            trace = QwenFixedMaskTrace.create(generated_tokens=4, active_count=1, steps=rows, identity=identity)
        if trace is not None:
            assert trace.steps[8].indices == [int(case["case_id"])]
        return Image.new("RGB", (1024, 1024)), {"request_wall_seconds": 1.0}, trace, None

    monkeypatch.setattr(h1, "native_request", native)
    run.run_jobs(None, None, heldout_jobs(cases))
    rows = run.receipt["requests"]
    assert [(r["case_id"], r["arm"], r["repetition"]) for r in rows] == [j.identity for j in heldout_jobs(cases)]
    for case in cases:
        subset = [r for r in rows if r["case_id"] == case["case_id"]]
        source = next(r for r in subset if r["arm"] == "direct_collect20")
        graph = [r for r in subset if r["arm"] == "full_sparsegraph20"]
        assert [r["repetition"] for r in graph] == [0, 1]
        assert all(r["fixed_trajectory_source"]["action_trace_sha256"] == source["action_trace_sha256"] for r in graph)
