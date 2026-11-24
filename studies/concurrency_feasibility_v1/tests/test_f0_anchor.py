import hashlib
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def test_f0_separates_anchor_from_historical_reproduction():
    anchor = json.loads((STUDY / "artifacts/f0_anchor_v1/completion.json").read_text())
    assert anchor["status"] == "PASS"
    assert anchor["fresh_stability_status"] == "PASS"
    assert anchor["historical_reproduction_status"] == "UNAVAILABLE"
    assert anchor["reference_kind"] == "fresh_same_case_native_v4_full"
    assert anchor["requests"] == 6 and anchor["all_fresh_pngs_equal"] is True
    assert anchor["max_absolute_relative_delta"] <= 0.02
    for relative_path, sha in anchor["evidence_sha256"].items():
        assert hashlib.sha256((STUDY / relative_path).read_bytes()).hexdigest() == sha
    assert json.loads((STUDY / "artifacts/baseline_v1/completion.json").read_text())["status"] == "BLOCKED"
