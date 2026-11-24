import json
from pathlib import Path


def test_baseline_completion_contract():
    path = Path(__file__).parents[1] / "artifacts/baseline_v1/completion.json"
    data = json.loads(path.read_text())
    assert data["status"] == "PASS"
    assert data["phase"] == "baseline"
    assert data["protocol"]["steps"] == 28
    assert data["protocol"]["K"] == 460
    assert data["protocol"]["N"] == 2304
    assert data["device_uuid"].startswith("GPU-")
    assert len(data["cases"]) == 3
    assert all("request_seconds" in c for c in data["cases"])
