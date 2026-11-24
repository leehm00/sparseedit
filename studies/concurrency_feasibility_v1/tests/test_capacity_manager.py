# SPDX-License-Identifier: Apache-2.0
import importlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "source"))
sys.path.insert(0, str(STUDY / "scripts"))


@lru_cache(maxsize=1)
def load_capacity():
    try:
        return importlib.import_module("vllm_omni.diffusion.models.flux.capacity_manager")
    except ModuleNotFoundError as exc:
        if exc.name == "vllm_omni.diffusion.models.flux.capacity_manager":
            pytest.fail("capacity_manager must implement the F3 footprint and admission model")
        raise


def test_shared_weights_count_once_and_private_storage_scales():
    capacity = load_capacity()
    manager = capacity.CapacityManager(
        shared_allocated_bytes=1_000,
        shared_reserved_slack_bytes=100,
        per_request_reserved_slack_bytes=20,
        budget_bytes=2_000,
    )
    footprint = capacity.RequestFootprint(200, 40, 30, 10, 35)
    state = SimpleNamespace(footprint=footprint)

    assert manager.predict_allocated_bytes([]) == 1_000
    assert manager.predict_peak_bytes([]) == 1_100
    assert manager.predict_allocated_bytes([state]) == 1_280
    assert manager.predict_allocated_bytes([state, state]) == 1_560
    assert manager.predict_reserved_bytes([state, state]) == 1_750


def test_different_request_costs_and_eager_fallback_affect_admission():
    capacity = load_capacity()
    manager = capacity.CapacityManager(shared_allocated_bytes=1_000, budget_bytes=1_600)
    graph = capacity.RequestFootprint(200, 40, 30, 10, 60)
    eager = capacity.RequestFootprint(200, 40, 30, 0)
    larger_k = capacity.RequestFootprint(200, 80, 60, 10, 60)

    assert manager.can_admit([], graph)
    assert manager.can_admit([graph], eager)
    assert not manager.can_admit([graph], graph)
    assert not manager.can_admit([eager], larger_k)
    assert manager.predict_max_concurrent(graph, 1_659) == 1
    assert manager.predict_max_concurrent(graph, 1_660) == 2
    assert manager.predict_max_concurrent(graph, 999) == 0


def test_missing_or_inconsistent_bytes_cannot_pass_admission():
    capacity = load_capacity()
    manager = capacity.CapacityManager(shared_allocated_bytes=1_000, budget_bytes=1_600)
    with pytest.raises(ValueError, match="graph_reserved_bytes"):
        capacity.RequestFootprint(200, 40, 30, 30, 10)
    with pytest.raises(ValueError, match="kv_bytes"):
        capacity.RequestFootprint(-1, 0, 0, 0)
    with pytest.raises(TypeError, match="RequestFootprint"):
        manager.can_admit([], SimpleNamespace(footprint=None))
    with pytest.raises(ValueError, match="positive"):
        manager.predict_max_concurrent(capacity.RequestFootprint(0, 0, 0, 0), 2_000)


def make_capacity_receipt(directory, name, n, *, allocated=None, reserved=None, error=None, budget=1_850):
    receipt = {
        "status": "PASS" if error is None else "OOM",
        "concurrency": n,
        "synchronized_before_return": error is None,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "device_uuid": "GPU-example",
        "environment": {"torch": "test"},
        "protocol": {"steps": 28, "K": 460},
        "source_sha": "abc",
        "memory_arm": "private_graph",
        "cases": [{"case_tag": f"case-{i}", "input_sha256": f"input-{i}", "trace_sha256": f"trace-{i}"} for i in range(n)],
        "repetition": 0 if name == "fit" else 1,
        "timing_boundary": "synchronized worker allocator peak",
        "budget_bytes": budget,
        "failure": error,
    }
    if name == "fit":
        receipt["shared_allocated_bytes"] = 1_000
        receipt["shared_reserved_bytes"] = 1_050
        receipt["footprints"] = [{
            "kv_bytes": 200, "activation_bytes": 40, "graph_bytes": 10, "graph_reserved_bytes": 35,
        }]
    path = directory / f"{name}.json"
    path.write_text(json.dumps(receipt))
    return path


def load_analyzer():
    try:
        return importlib.import_module("analyze_capacity")
    except ModuleNotFoundError as exc:
        if exc.name == "analyze_capacity":
            pytest.fail("analyze_capacity must compare held-out peaks and measured OOM bounds")
        raise


def test_capacity_analysis_validates_independent_fixed_n_and_oom(tmp_path):
    analyzer = load_analyzer()
    fit = make_capacity_receipt(tmp_path, "fit", 1, allocated=1_280, reserved=1_375)
    one = make_capacity_receipt(tmp_path, "holdout", 1, allocated=1_285, reserved=1_385)
    two = make_capacity_receipt(tmp_path, "two", 2, allocated=1_600, reserved=1_745)
    oom = make_capacity_receipt(tmp_path, "three", 3, error={"type": "OutOfMemoryError", "message": "CUDA out of memory"})
    result = analyzer.analyze_capacity(fit, [one, two, oom])

    assert result["status"] == "PASS"
    assert result["calibration"]["request_footprint"]["workspace_bytes"] == 30
    assert result["calibration"]["per_request_reserved_slack_bytes"] == 20
    assert result["fixed_n"]["status"] == "PASS"
    assert [(row["predicted_allocated_bytes"], row["predicted_reserved_bytes"]) for row in result["fixed_n"]["points"]] == [
        (1_280, 1_375), (1_560, 1_700),
    ]
    assert result["maximum_n"]["last_successful_n"] == 2
    assert result["maximum_n"]["first_oom_n"] == 3
    assert result["maximum_n"]["predicted_n"] == 2
    assert result["maximum_n"]["status"] == "PASS"


def test_capacity_analysis_preserves_error_and_never_equates_missing_oom_to_a_boundary(tmp_path):
    analyzer = load_analyzer()
    fit = make_capacity_receipt(tmp_path, "fit", 1, allocated=1_280, reserved=1_375)
    one = make_capacity_receipt(tmp_path, "holdout", 1, allocated=1_285, reserved=1_385)
    two = make_capacity_receipt(tmp_path, "two", 2, allocated=1_700, reserved=1_800)
    result = analyzer.analyze_capacity(fit, [one, two])

    assert result["status"] == "FAIL"
    assert result["fixed_n"]["status"] == "FAIL"
    assert result["fixed_n"]["points"][1]["allocated_relative_error"] > 0.05
    assert result["maximum_n"]["status"] == "CONSERVATIVE_ONLY"
    assert result["maximum_n"]["first_oom_n"] is None
    assert result["maximum_n"]["conservative_admission_n"] == 2


def test_capacity_analysis_rejects_reused_calibration_and_incompatible_budget(tmp_path):
    analyzer = load_analyzer()
    fit = make_capacity_receipt(tmp_path, "fit", 1, allocated=1_280, reserved=1_375)
    other = make_capacity_receipt(tmp_path, "other", 1, allocated=1_280, reserved=1_375, budget=2_000)

    with pytest.raises(ValueError, match="independent"):
        analyzer.analyze_capacity(fit, [fit])
    with pytest.raises(ValueError, match="budget_bytes"):
        analyzer.analyze_capacity(fit, [other])


def test_capacity_analysis_keeps_oom_when_n2_has_no_peak(tmp_path):
    analyzer = load_analyzer()
    fit = make_capacity_receipt(tmp_path, "fit", 1, allocated=1_280, reserved=1_375, budget=1_600)
    one = make_capacity_receipt(tmp_path, "holdout", 1, allocated=1_285, reserved=1_385, budget=1_600)
    two = make_capacity_receipt(tmp_path, "two", 2, error={"type": "OutOfMemoryError", "message": "CUDA out of memory"}, budget=1_600)
    result = analyzer.analyze_capacity(fit, [one, two])

    assert result["status"] == "BLOCKED"
    assert result["fixed_n"]["missing_n"] == [2]
    assert result["maximum_n"]["status"] == "PASS"
    assert result["maximum_n"]["first_oom_n"] == 2
    assert result["outcomes"][1]["failure"]["type"] == "OutOfMemoryError"


def test_f0_context_preserves_native_worker_allocator_evidence():
    analyzer = load_analyzer()
    context = analyzer.summarize_f0_capacity(STUDY / "artifacts/f0_anchor_v1/completion.json")

    assert context["device_uuid"] == "GPU-c397c43e-c0fb-328f-06f7-ed5773e129a7"
    assert context["counts"]["requests"] == 6
    assert max(row["peak_reserved_bytes"] for row in context["rows"]) == 43_597_692_928
    assert {row["kv_bytes"] for row in context["rows"]} == {7_172_259_840}
    assert {row["graph_reserved_bytes"] for row in context["rows"]} == {276_824_064}
    assert {row["condition_cache_bytes"] for row in context["rows"]} == {119_046_144, 178_569_216, 238_092_288}
    assert {row["condition_owned_bytes"] for row in context["rows"]} == {123_297_792}
    assert context["shared_allocated_bytes"] is None


def test_capacity_analysis_does_not_use_cpu_oom_as_cuda_boundary(tmp_path):
    analyzer = load_analyzer()
    fit = make_capacity_receipt(tmp_path, "fit", 1, allocated=1_280, reserved=1_375, budget=1_600)
    one = make_capacity_receipt(tmp_path, "holdout", 1, allocated=1_285, reserved=1_385, budget=1_600)
    two = make_capacity_receipt(tmp_path, "two", 2, error={"type": "OutOfMemoryError", "message": "CPU out of memory"}, budget=1_600)
    result = analyzer.analyze_capacity(fit, [one, two])

    assert result["maximum_n"]["first_oom_n"] is None
    assert result["outcomes"][1]["failure"]["message"] == "CPU out of memory"


def test_capacity_queue_admits_fifo_batches_without_changing_requests():
    analyzer = load_analyzer()
    capacity = load_capacity()
    graph = capacity.RequestFootprint(200, 40, 30, 10, 60)
    eager = capacity.RequestFootprint(200, 40, 30, 0)
    manager = capacity.CapacityManager(shared_allocated_bytes=1_000, budget_bytes=1_600)
    requests = (object(), object(), object())

    batches = analyzer.plan_admitted_batches(manager, [(requests[0], eager), (requests[1], graph), (requests[2], graph)])

    assert batches == [[requests[0], requests[1]], [requests[2]]]
    with pytest.raises(ValueError, match="cannot fit"):
        analyzer.plan_admitted_batches(manager, [(requests[0], capacity.RequestFootprint(700, 0, 0, 0))])


def test_sweep_replays_frozen_cases_for_capacity_only_and_assigns_distinct_requests():
    analyzer = load_analyzer()
    observations = (("ref0", {"case_tag": "a"}), ("ref1", {"case_tag": "b"}), ("ref2", {"case_tag": "c"}))

    selected = analyzer.sweep_inputs(observations, 5, repetition=2)
    assert [reference for reference, _ in selected] == ["ref0", "ref1", "ref2", "ref0", "ref1"]
    assert [case["case_tag"] for _, case in selected] == ["a", "b", "c", "a", "b"]
    assert len({case["request_id"] for _, case in selected}) == 5


def test_logical_capacity_ceiling_is_preserved_without_claiming_oom(tmp_path):
    analyzer = load_analyzer()
    fit = make_capacity_receipt(tmp_path, "fit", 1, allocated=1_280, reserved=1_375, budget=1_600)
    one = make_capacity_receipt(tmp_path, "holdout", 1, allocated=1_285, reserved=1_385, budget=1_600)
    ceiling = make_capacity_receipt(
        tmp_path, "two", 2, error={"type": "AdmissionBudgetExceeded", "message": "logical ceiling"}, budget=1_600,
    )
    raw = json.loads(ceiling.read_text())
    raw["status"] = "CAPACITY_BLOCKED"
    ceiling.write_text(json.dumps(raw))

    result = analyzer.analyze_capacity(fit, [one, ceiling])
    assert result["maximum_n"]["status"] == "CONSERVATIVE_ONLY"
    assert result["maximum_n"]["stop_reason"] == "PREDICTED_BUDGET_CEILING"
    assert result["maximum_n"]["first_oom_n"] is None
