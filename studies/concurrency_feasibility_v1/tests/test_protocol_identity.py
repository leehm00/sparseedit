import hashlib
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
BUNDLE = STUDY.parents[1] / "backend_reference/current_v4/backend_bundle"


def test_protocol_identity():
    protocol = json.loads((STUDY / "protocol.json").read_text())
    assert (protocol["height"], protocol["width"], protocol["steps"], protocol["K"]) == (768, 768, 28, 460)
    assert protocol["true_cfg_scale"] == 4.0 and protocol["seed"] == 110
    cases = json.loads((BUNDLE / "bundle.json").read_text())["cases"]["calibration"]
    assert len(protocol["calibration_case_sha256"]) == len(cases) == 3
    assert [row["case_tag"] for row in protocol["calibration_cases"]] == [row["case_tag"] for row in cases]
    for case in cases:
        tag = case["case_tag"]
        assert protocol["calibration_case_sha256"][tag] == case["prepared_sha256"]
        assert protocol["calibration_trace_sha256"][tag] == case["trace_sha256"]
        assert hashlib.sha256((BUNDLE / "inputs" / case["input_filename"]).read_bytes()).hexdigest() == case["prepared_sha256"]
        assert hashlib.sha256((BUNDLE / "traces" / f"{tag}.json").read_bytes()).hexdigest() == case["trace_sha256"]
    assert len(protocol["baseline_source_sha256"]) == 64
