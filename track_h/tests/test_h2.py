"""CPU-only checks of the H2 fixed replay and evidence contracts."""

import json
from pathlib import Path

import pytest

import h2
from h2_runtime import flux_native


def trace_for(side, ratio):
    n = (side // 16) ** 2
    trace = {
        "kind": "flux_sparseedit_fixed_mask_trace", "schema_version": 1,
        "config": {
            "ratio": ratio, "num_inference_steps": 28, "warmup_steps": 8, "refresh_interval": 15,
            "true_cfg_scale": 4.0, "true_cfg": True, "selector_mode": "forward", "selector_protocol": "paper",
            "generated_token_count": n, "reference_token_count": n,
            "grid_height": side // 16, "grid_width": side // 16, "active_token_count": int(n * ratio),
        },
        "steps": [
            {"step": i, "phase": "warmup" if i < 8 else "refresh_full" if i == 23 else "sparse",
             "kv_action": "none" if i < 7 else "store_full" if i in (7, 23) else "update_active",
             "indices": list(range(int(n * ratio))) if i >= 8 and i != 23 else None}
            for i in range(28)
        ],
    }
    trace["sha256"] = flux_native.trace_digest(trace)
    return trace


def test_h2_frozen_sources_and_schedule():
    payload = Path(h2.__file__).parent / "payload_data/flux"
    if not payload.exists():
        pytest.skip("frozen image payload is distributed separately")
    bundle = flux_native.load_bundle(payload)
    cal3 = h2.experiment_points(bundle, "cal3", 24)
    formal = h2.experiment_points(bundle, "formal", 24)
    assert [(p.side, p.ratio) for p in cal3] == [(side, ratio) for side in h2.SIDES for ratio in h2.CAL3_RATIOS]
    assert all(len(p.cases) == 3 for p in cal3)
    assert formal[:9] == cal3
    assert [len(p.cases) for p in formal[9:]] == [24, 24, 24]
    assert [len(p.cases) for p in h2.experiment_points(bundle, "formal", 48)[9:]] == [48, 48, 48]
    assert flux_native.load_frozen_trace(payload, cal3[1].cases[0])["config"]["ratio"] == .2
    assert h2.ARMS == (
        "fresh_dense", "direct_eager", "standard_fastops_eager", "full", "joint_cfg", "segmented_int8"
    )


def test_trace_sizes_phase_indices_and_replay_identity():
    trace = trace_for(1536, .2)
    assert flux_native.validate_trace(trace, 1536, .2) == trace
    with pytest.raises(ValueError, match="shape"):
        flux_native.validate_trace(trace, 768, .2)
    trace["steps"][8]["indices"][-1] = trace["steps"][8]["indices"][-2]
    trace["sha256"] = flux_native.trace_digest(trace)
    with pytest.raises(ValueError, match="indices"):
        flux_native.validate_trace(trace, 1536, .2)


def test_arm_options_keep_distinct_runtime_paths(tmp_path):
    path = tmp_path / "trace.json"
    for arm, backend in (
        ("direct_eager", "off"), ("standard_fastops_eager", "standard_fastops_eager"),
        ("full", "off"), ("segmented_int8", "off"), ("joint_cfg", "off"),
    ):
        method, options = flux_native.request_options(arm, .2, path, side=1536)
        assert method == "sparseedit"
        assert options["sparseedit_replay_trace"] == str(path)
        assert options["sparseedit_standard_backend"] == backend
        assert options["sparseedit_graph_fallback"] is False
        assert not options["sparseedit_collect_trace"]
        assert options["sparseedit_cfg_execution"] == ("batched" if arm == "joint_cfg" else "serial")
    _, small = flux_native.request_options("joint_cfg", .2, path, side=1024)
    assert small["sparseedit_cfg_execution"] == "budget"
    _, int8 = flux_native.request_options("segmented_int8", .2, path, side=1536)
    assert int8["sparseedit_reference_kv_int8"] is True
    assert flux_native.request_options("fresh_dense")[0] == "dense"
    with pytest.raises(ValueError, match="collect"):
        flux_native.request_options("full", .2, path, collect_trace=True)


def test_distinguishes_baseline_exactness_and_capacity_results():
    def row(arm, side, digest, persistent=None):
        return {
            "status": "PASS", "arm": arm, "height": side, "active_ratio": .2, "partition": "formal",
            "case_tag": "source", "output_sha256": digest, "request_wall_s": 1,
            "persistent_kv_bytes": persistent,
        }

    rows = [row("full", 1536, "a", 24 << 30), row("segmented_int8", 1536, "b", 18 << 30),
            row("standard_fastops_eager", 1536, "a"), row("direct_eager", 1536, "b")]
    result = h2.paired_summary(rows)
    assert result["fixed_trace_comparisons"]["1536"]["direct_eager"]["matching_pngs"] == 0
    assert result["fixed_trace_comparisons"]["1536"]["standard_fastops_eager"]["matching_pngs"] == 1
    assert result["capacity"]["1536"]["persistent_kv_saved_bytes"] == [6 << 30]
    assert h2.clone_traffic_bytes(768) / 2 ** 30 == pytest.approx(13.36, abs=.01)
    assert h2.clone_traffic_bytes(1536) / 2 ** 30 == pytest.approx(49.43, abs=.01)


def test_stage_hash_manifest_binds_relative_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/manifest.json").write_text("{}")
    receipt = h2.finalize(tmp_path, {"status": "PARTIAL"})
    assert receipt["phase"] == "H2"
    assert json.loads((tmp_path / "completion.json").read_text()) == receipt
    assert (tmp_path / "sha256sums.txt").read_text().endswith(
        f"{flux_native.sha256(tmp_path / 'completion.json')}  completion.json\n"
    )
    assert receipt["files"]["sub/manifest.json"] == flux_native.sha256(tmp_path / "sub/manifest.json")


def test_native_trace_witness_accepts_no_replay_and_audits_exact_indices():
    trace = trace_for(1536, .2)
    n = trace["config"]["generated_token_count"]
    k = trace["config"]["active_token_count"]
    graph = {"graph_active": True, "fallback": False, "total_replays": 38, "cfg_branches_per_replay": 1}
    md = {
        "method": "sparseedit",
        "summary": {"dense_steps": 9, "sparse_steps": 19, "transformer_skipped_steps": 0,
                    "persistent_kv_bytes": 42 << 30, "actual_backends": ["torch_reference"]},
        "run_config": {
            **trace["config"], "latent_dtype": "torch.bfloat16", "cfg_execution_plan": {"selected": "serial"},
            "replay": None, "graph": graph,
        },
        "pipeline_timing": {"encoding_s": .2, "decode_s": .3},
        "fixed_mask_trace": trace, "fixed_mask_trace_sha256": trace["sha256"],
        "timing": {"step_forward_device_seconds": [.1] * 28,
                   "denoise_wall_seconds": 3.2, "denoise_device_seconds": 3.1},
        "step_records": [
            dict(step, cfg_branch_count=2, transformer_computed=True,
                 active_token_count=k if step["phase"] == "sparse" else n,
                 reference_policy="cached" if step["phase"] == "sparse" else "full")
            for step in trace["steps"]
        ],
    }
    point = h2.Point(1536, .2, "calibration", (), 1)
    evidence = h2.validate_execution(md, point, "full", None)
    assert evidence["replay_trace_payload_sha256"] == trace["sha256"]
    assert evidence["steady_sparse_device_s"] == pytest.approx(.1)
    assert evidence["full_device_total_s"] == pytest.approx(.9)
    md["run_config"]["replay"] = {"enabled": True, "sha256": trace["sha256"]}
    md["step_records"][8]["indices"] = list(range(1, k + 1))
    with pytest.raises(ValueError, match="executed token"):
        h2.validate_execution(md, point, "full", trace)


def test_h4_receipt_uses_total_full_device_time_and_labels_fixed_holdouts():
    row = {
        "status": "PASS", "height": 1024, "arm": "full", "partition": "formal", "source_id": "abc",
        "active_ratio": .2, "request_wall_s": 5., "denoise_wall_s": 4., "dense_steps": 9, "sparse_steps": 19,
        "reuse_steps": 0, "full_device_s": .3, "full_device_total_s": 2.7, "steady_sparse_device_s": .1,
        "repetition": 0, "manifest": "manifest.json",
    }
    obs = h2._observation(row)
    assert obs["full_device_s"] == 2.7
    assert obs["split"] == "heldout"
    assert obs["workload"] == "fixed_replay"


def test_1536_standalone_capacity_probe_records_its_own_trace():
    method, options = flux_native.request_options("segmented_int8", .2, side=1536, collect_trace=True)
    assert method == "sparseedit"
    assert options["sparseedit_collect_trace"]
    assert options["sparseedit_reference_kv_int8"]
    assert not options["sparseedit_replay_trace"]
    with pytest.raises(ValueError, match="collect"):
        flux_native.request_options("segmented_int8", .2, side=1024, collect_trace=True)


def test_standalone_capacity_probe_only_selects_missing_1536_sources(tmp_path, monkeypatch):
    from contextlib import nullcontext

    cases = tuple({"case_tag": tag} for tag in ("matched", "missing"))
    point = h2.Point(1536, .2, "formal", cases, 2)
    calls = []
    monkeypatch.setattr(flux_native, "native_worker", lambda *_a, **_k: nullcontext(object()))

    def cell(_worker, _assets, path, _point, case, arm, rep, **options):
        calls.append((case["case_tag"], arm, rep, options["witness_arm"], options.get("warmup", False)))
        return {"status": "PASS", "persistent_kv_bytes": 20 << 30}

    monkeypatch.setattr(h2, "_run_cell", cell)
    rows = h2._standalone_capacity(tmp_path, point, tmp_path, {"matched": object()}, "model", "gpu")
    assert calls == [
        ("missing", "segmented_int8_capacity_probe", 0, "segmented_int8", True),
        ("missing", "segmented_int8_capacity_probe", 0, "segmented_int8", False),
    ]
    assert rows[0]["persistent_kv_bytes"] == 20 << 30
    assert h2._standalone_capacity(tmp_path, h2.Point(1024, .2, "formal", cases, 1),
                                   tmp_path, {}, "model", "gpu") == []
