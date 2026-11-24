import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bench_flux_scaling import (
    bind_cell,
    classify_outputs,
    main,
    paired_speedups,
    scaling_schedule,
    validate_execution,
)


def test_scaling_uses_paired_rotating_cases_and_alternating_order():
    tags = ("a", "b", "c")
    cells = list(scaling_schedule(tags, [1, 4, 8], 2))
    assert [cell.mode for cell in cells] == ["serial", "concurrent"] * 3 + ["concurrent", "serial"] * 3
    assert len({cell.directory for cell in cells}) == len(cells)
    assert cells[4].cases == cells[5].cases == ("a", "b", "c", "a", "b", "c", "a", "b")
    assert cells[10].cases == cells[11].cases == ("b", "c", "a", "b", "c", "a", "b", "c")


def test_bind_repeated_cases_gives_unique_ids_without_changing_trace():
    tags = ("a", "b", "c")
    bound = {tag: ({"case_tag": tag}, Path(f"{tag}.png"), Path(f"{tag}.json"), {"tag": tag}) for tag in tags}
    identities = {tag: {"case_tag": tag, "input_sha256": tag * 64} for tag in tags}
    cell = next(scaling_schedule(tags, [8], 1))
    request_ids, request_cases, records = bind_cell(cell, bound, identities)
    assert len(set(request_ids)) == 8
    assert [records[rid]["case_tag"] for rid in request_ids] == list(cell.cases)
    assert all(records[rid]["request_id"] == rid for rid in request_ids)
    assert all(request_cases[rid][0]["case_tag"] == rid for rid in request_ids)
    assert all(request_cases[rid][3] is bound[tag][3] for rid, tag in zip(request_ids, cell.cases, strict=True))
    assert bound["a"][0]["case_tag"] == "a"


@pytest.mark.parametrize(
    ("references", "status", "matches"),
    [
        ({"a": "1", "b": "2"}, "PASS", [True, True, True]),
        ({"a": "1"}, "DIAGNOSTIC", [True, None, True]),
        ({"a": "z"}, "OUTPUT_MISMATCH", [False, None, False]),
    ],
)
def test_only_complete_serial_png_matches_are_eligible(references, status, matches):
    rows = [
        {"case_tag": "a", "output_png_sha256": "1"},
        {"case_tag": "b", "output_png_sha256": "2"},
        {"case_tag": "a", "output_png_sha256": "1"},
    ]
    assert classify_outputs(rows, references) == (status, matches)


def test_paired_speedups_use_measured_walls_with_matched_cases():
    def row(rep, mode, wall, status="PASS", tags=("a", "b")):
        return {
            "concurrency": 2, "repetition": rep, "mode": mode, "status": status,
            "batch_wall_seconds": wall, "cases": [{"case_tag": tag} for tag in tags],
        }

    rows = [
        row(0, "serial", 10.0), row(0, "concurrent", 8.0),
        row(1, "serial", 10.0), row(1, "concurrent", 8.0, status="DIAGNOSTIC"),
        row(2, "serial", 10.0), row(2, "concurrent", 8.0, tags=("b", "a")),
    ]
    assert paired_speedups(rows) == [
        {"concurrency": 2, "repetition": 0, "serial_seconds": 10.0,
         "concurrent_seconds": 8.0, "throughput_speedup": 1.25},
    ]


def test_stream_evidence_is_required_for_concurrent_measurement():
    timings = [{"request_seconds": 5.0, "cuda_event_seconds": 4.0}] * 2
    raw = {
        "batch_wall_seconds": 5.0,
        "timings": timings,
        "streams": [0, 1],
        "enqueue_order": [
            {"step": step, "request_id": tag, "stream_id": stream}
            for step in range(2)
            for stream, tag in enumerate(("a", "b"))
        ],
    }
    assert validate_execution(raw, "concurrent", ("a", "b"), 2) == [5.0, 5.0]
    raw["enqueue_order"][1]["step"] = 1
    with pytest.raises(ValueError, match="step interleaving"):
        validate_execution(raw, "concurrent", ("a", "b"), 2)
    raw["streams"] = [1, 1]
    with pytest.raises(ValueError, match="distinct CUDA streams"):
        validate_execution(raw, "concurrent", ("a", "b"), 2)


def test_unbound_resolution_has_explicit_receipt(tmp_path, monkeypatch):
    import json

    import bench_flux_scaling

    monkeypatch.setattr(bench_flux_scaling, "ARTIFACTS", tmp_path)
    output = tmp_path / "1024"
    assert main([
        "--gpu-uuid", "GPU-example", "--model", str(tmp_path / "model"),
        "--output-root", str(output), "--width", "1024", "--height", "1024",
    ]) == 1
    receipt = json.loads((output / "completion.json").read_text())
    assert receipt["status"] == "UNSUPPORTED"
    assert receipt["requested_resolution"] == {"height": 1024, "width": 1024}
    assert "shape-bound" in receipt["failure"]
