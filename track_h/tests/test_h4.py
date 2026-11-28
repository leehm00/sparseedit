"""H4 calibration remains source disjoint and independent of holdout outcomes."""
import pytest

from h4 import evaluate


def row(split, source, ratio, *, sparse=19, reuse=0):
    steady = .2 + .8 * ratio
    denoise = 9 * 1.7 + sparse * steady
    return dict(
        split=split, source_id=source, ratio=ratio,
        model="flux", resolution=768, runtime="full", request_wall_s=.9 + denoise,
        denoise_wall_s=denoise, full_steps=9, sparse_steps=sparse, reuse_steps=reuse,
        full_device_s=9 * 1.7, steady_sparse_device_s=steady, capture_trials=0,
    )


def test_fit_uses_3_calibration_sources_and_scores_48_heldout_sources():
    cal = [row("calibration", f"c{i}", ratio) for i in range(3) for ratio in (.1, .2, .35)]
    heldout = [row("heldout", f"t{i}", .4, sparse=11, reuse=8) for i in range(48)]
    result = evaluate(cal + heldout)
    entry = result["comparisons"][0]
    assert entry["fit"]["alpha_s"] == pytest.approx(.2)
    assert entry["fit"]["beta_s"] == pytest.approx(.8)
    assert entry["fit"]["T_out_s"] == pytest.approx(.9)
    assert entry["mape_percent"] == pytest.approx(0)
    assert entry["heldout_sources"] == 48
    assert entry["under_5_percent"]
    heldout[0]["source_id"] = "c0"
    assert evaluate(cal + heldout)["comparisons"][0]["status"] == "BLOCKED"


def test_missing_calibration_budget_or_online_holdout_never_claims_pass():
    cal = [row("calibration", f"c{i}", .2) for i in range(3)]
    assert evaluate(cal)["status"] == "BLOCKED"
    holdout = [row("heldout", "t0", .4, sparse=12, reuse=7)]
    assert evaluate(cal + holdout)["comparisons"][0]["status"] == "BLOCKED"


def test_cold_online_trace_uses_calibration_warmup_receipt_not_heldout_time():
    from h4 import capture_features

    branch = {"warmup_trials": 2, "capture_trials": 1, "preparation_seconds": .52}
    graph = {"branches": {"positive": branch, "negative": branch}}
    capture = capture_features(graph)
    assert capture == {"capture_trials": 6, "cfg_forwards_per_step": 2, "capture_wall_s": 1.04}
    cal = [row("calibration", f"c{i}", ratio) for i in range(3) for ratio in (.1, .2, .35)]
    cal[0]["calibration_capture"] = capture
    heldout = [row("heldout", f"t{i}", .4) for i in range(48)]
    heldout[0]["capture_trials"] = 6
    heldout[0]["capture_wall_s"] = 100.0
    heldout[0]["request_wall_s"] += 3 * .52 + .20
    result = evaluate(cal + heldout)["comparisons"][0]
    assert result["fit"]["capture_dispatch_s"] == pytest.approx(.2)
    assert result["mape_percent"] == pytest.approx(0.0)
