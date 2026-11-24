import json
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "scripts"))

from run_h200_window import Stage, build_stages, run_stages


def test_window_runs_other_models_and_diagnostics_after_failure(tmp_path):
    model = tmp_path / "available-model"
    model.mkdir()
    stages = [
        Stage("flux_f2", "flux.py", (), model),
        Stage("longcat_dense", "native.py", (), tmp_path / "missing-model"),
        Stage("qwen_dense", "native.py", (), model),
        Stage("flux_scaling_graph", "scaling.py", (), model),
        Stage("flux_capacity", "capacity.py", (), model),
    ]
    calls = []

    def execute(stage, output, log, timeout):
        calls.append(stage.name)
        if stage.name == "flux_scaling_graph":
            return 124
        output.mkdir()
        status = "NO_GO" if stage.name == "flux_f2" else "PASS"
        (output / "completion.json").write_text(json.dumps({"status": status}))
        return int(status != "PASS")

    results = run_stages(tmp_path, stages, execute, timeout_seconds=60)
    assert calls == ["flux_f2", "qwen_dense", "flux_scaling_graph", "flux_capacity"]
    assert [result["status"] for result in results] == ["NO_GO", "UNAVAILABLE", "PASS", "TIMEOUT", "PASS"]
    saved = json.loads((tmp_path / "session_progress.json").read_text())
    assert len(saved["stages"]) == len(stages)
    assert (tmp_path / "flux_scaling_graph/completion.json").is_file()


def test_default_window_arranges_fast_independent_models_before_long_sweeps(tmp_path):
    stages = build_stages(tmp_path, "GPU-example", tmp_path / "longcat", tmp_path / "qwen")
    names = [stage.name for stage in stages]
    assert names[:3] == ["flux_dense", "longcat_dense", "qwen_dense"]
    assert "flux_f2" in names and "flux_capacity" in names and "flux_templates" in names
    eager = next(stage for stage in stages if stage.name == "flux_scaling_eager")
    assert "--counts" in eager.args and tuple(eager.args[eager.args.index("--counts") + 1:][:4]) == (1, 2, 4, 8)
    assert names.index("qwen_dense") < names.index("flux_scaling_graph")


def test_window_deadline_skips_only_work_that_cannot_start(tmp_path, monkeypatch):
    import run_h200_window as entry

    clock = iter([0, 0, 4, 90])
    monkeypatch.setattr(entry.time, "monotonic", lambda: next(clock, 90))
    stages = [Stage("one", "a.py", ()), Stage("two", "b.py", ())]
    calls = []

    def execute(stage, output, log, timeout):
        calls.append((stage.name, timeout))
        return 1

    results = run_stages(tmp_path, stages, execute, timeout_seconds=30, budget_seconds=60)
    assert calls == [("one", 30)]
    assert results[1]["status"] == "NOT_RUN_BUDGET"


def test_window_report_excludes_non_throughput_arms(tmp_path):
    from run_h200_window import render

    stages = [Stage("templates", "f4.py", ()), Stage("dense", "dense.py", ())]

    def execute(stage, output, log, timeout):
        output.mkdir()
        arms = {"graph": {"status": "PASS"}} if stage.name == "templates" else [{
            "arm": "dense", "mode": "serial", "concurrency": 1, "images_per_minute": 5.0,
            "latency_mean_seconds": 12.0, "latency_p50_seconds": 12.0,
        }]
        (output / "completion.json").write_text(json.dumps({"status": "PASS", "arms": arms}))
        return 0

    results = run_stages(tmp_path, stages, execute, timeout_seconds=30)
    report = render(results)
    assert results[0]["performance"] == []
    assert "| dense | dense | serial | 1 | 5.000 |" in report


def test_template_probe_can_collect_diagnostics_without_f1(tmp_path, monkeypatch):
    import pytest
    import probe_template_concurrency as probe

    def missing(*args):
        raise ValueError("F1 receipts unavailable")

    monkeypatch.setattr(probe, "f1_admission", missing)
    with pytest.raises(ValueError, match="F1 receipts"):
        probe.template_admission(tmp_path, set(), {}, "GPU-example")
    result = probe.template_admission(tmp_path, set(), {}, "GPU-example", diagnostic_without_f1=True)
    assert result == {"status": "UNAVAILABLE", "reason": "F1 receipts unavailable"}
