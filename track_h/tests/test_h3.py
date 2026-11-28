from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_formal_schedule_covers_two_serial_repetitions_per_arm():
    from h3_runtime.contracts import groups, panels

    cases = [{"case_tag": f"source{i:02}", "source_pixel_sha256": str(i)} for i in range(48)]
    panel = panels()[0]
    scheduled = groups(panel, cases, "formal")
    assert len(scheduled) == 9
    for _, rows in scheduled:
        assert len(rows) == 96
        assert rows[0] == (0, cases[0])
        assert rows[47] == (0, cases[47])
        assert rows[48] == (1, cases[47])
        assert rows[95] == (1, cases[0])


def test_coverage_never_substitutes_fixed20_for_adaptive_k():
    from h3_runtime.contracts import panels

    flux768, flux1024, longcat, qwen = panels()
    assert [panel["resolution"] for panel in (flux768, flux1024, longcat, qwen)] == [768, 1024, 1024, 1024]
    assert set(flux768["arms"]) == {
        "fresh_dense", "dense_graph", "teacache", "toca", "ras", "regione",
        "sparseedit_k_only", "sparseedit_k_b_lane_e", "sparseedit_k_b_joint_cfg_lane_q",
    }
    assert "spotedit" in flux768["gaps"]
    assert "sparseedit_k_only" in flux1024["gaps"]
    assert "dense_graph" in longcat["arms"]
    assert "sparseedit_k_only" in qwen["gaps"]
    assert qwen["supplemental"] == ["sparseedit_fixed20_online_selector"]


def test_formal_cross_model_identity_rejects_a_relabelled_instruction():
    from h3_runtime.contracts import validate_cohorts

    cases = [{"case_tag": f"case-{i}", "source_pixel_sha256": str(i), "instruction": "edit"} for i in range(48)]
    validate_cohorts("formal", {"flux": cases, "longcat": cases[::-1], "qwen": cases})
    changed = [*cases[:-1], {**cases[-1], "instruction": "new edit"}]
    with pytest.raises(ValueError, match="cohort"):
        validate_cohorts("formal", {"flux": cases, "longcat": cases, "qwen": changed})


def test_source_overlay_is_derived_without_mutating_shared_flux(tmp_path):
    from h2_runtime.flux_native import RUNTIME_SOURCE
    from h3_runtime.flux import overlay_source

    rel = Path("vllm_omni/diffusion/models/flux")
    before = (RUNTIME_SOURCE / rel / "acceleration.py").read_bytes()
    with overlay_source(RUNTIME_SOURCE, tmp_path) as source:
        cfg = source / rel / "acceleration.py"
        assert not cfg.is_symlink()
        assert "cfg_ports_version" in cfg.read_text()
        assert (source / rel / "cfg_ports_runtime.py").is_file()
        assert not (source / rel / "pipeline_flux_kontext.py").is_symlink()
    assert (RUNTIME_SOURCE / rel / "acceleration.py").read_bytes() == before
    assert list(tmp_path.iterdir()) == []


def test_cuda_hidden_h3_cli_writes_failure_receipt(tmp_path, monkeypatch):
    from h3 import run

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    config = tmp_path / "config.json"
    config.write_text('{"gpu_uuid":"GPU-12345678-1234-1234-1234-123456789abc", "run_mode":"cal3"}')
    output = tmp_path / "result"
    receipt = run(config, output)
    assert receipt["phase"] == "H3"
    assert receipt["execution_status"] == "FAIL"
    assert "visible" in receipt["error"]["message"]


def test_h4_observation_exports_full_total_and_steady_sparse_mean():
    from h3_runtime.flux import latency_observation

    steps = [{"transformer_computed": True, "is_sparse_step": False} for _ in range(9)]
    steps += [{"transformer_computed": True, "is_sparse_step": True} for _ in range(3)]
    steps += [{"transformer_computed": False, "is_sparse_step": True} for _ in range(16)]
    metadata = {
        "timing": {"step_forward_device_seconds": [2.0] * 9 + [1.0, 3.0, 5.0], "denoise_wall_seconds": 29.0},
        "run_config": {"active_token_count": 460, "graph": {"branches": {"cfg_batched": {"capture_trials": 1}}}},
    }
    row = {"resolution": 768, "repetition": 1, "source_id": "source", "request_wall_s": 30.0}
    result = latency_observation("sparseedit_k_b_joint_cfg_lane_q", row, metadata, steps)
    assert (result["runtime"], result["full_steps"], result["sparse_steps"], result["reuse_steps"]) == ("joint_cfg", 9, 3, 16)
    assert result["full_device_s"] == 18.0
    assert result["steady_sparse_device_s"] == 4.0


def test_h1_dense_reuse_requires_exact_formal_repetition_and_manifest(tmp_path):
    from h1 import source_identity
    from h3_runtime.contracts import file_sha, write
    from h3_runtime.qwen import prior_dense
    from track_h_qwen.contracts import PROTOCOL, digest_json

    root = tmp_path / "H1"
    root.mkdir()
    image = root / "output.png"
    image.write_bytes(b"fixture")
    trace = root / "actions.json"
    write(trace, {"steps": [{"step": step, "action": "full"} for step in range(28)]})
    case = {"case_id": "case", "split": "test", "input_sha256": "input", "source_pixel_sha256": "source", "prompt": "edit"}
    row = {
        "arm": "fresh_dense", "case_id": "case", "split": "test", "repetition": 1, "status": "PASS",
        "input_sha256": "input", "source_pixel_sha256": "source", "prompt_sha256": digest_json("edit"),
        "model_sha256": "weights", "protocol_sha256": digest_json(PROTOCOL), "budget_percent": None,
        "timing_role": "measurement", "session_instruction_index": None, "timing_scope": "synchronized",
        "output_path": image.name, "output_sha256": file_sha(image),
        "action_trace_path": trace.name, "action_trace_sha256": file_sha(trace), "request_wall_seconds": 1.25,
    }
    manifest = root / "manifest.json"
    write(manifest, row)
    write(root / "completion.json", {
        "phase": "H1", "run_mode": "formal", "gpu_uuid": "uuid", "execution_status": "PASS",
        "protocol": PROTOCOL, "source_sha256": source_identity(), "model": {"runtime_file_sha256": "weights"},
        "requests": [{**row, "manifest_path": manifest.name, "manifest_sha256": file_sha(manifest)}],
    })
    assert prior_dense(tmp_path, case, 0, "formal", "uuid", "weights") is None
    assert prior_dense(tmp_path, case, 1, "formal", "other", "weights") is None
    assert prior_dense(tmp_path, case, 1, "formal", "uuid", "weights")["request_wall_s"] == 1.25
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        prior_dense(tmp_path, case, 1, "formal", "uuid", "weights")
