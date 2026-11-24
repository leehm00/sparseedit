# SPDX-License-Identifier: Apache-2.0
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "scripts"))

PAIRS = (("a", "b"), ("b", "c"), ("c", "a"))
PROTOCOL = {"height": 768, "width": 768, "K": 460, "steps": 28, "seed": 110}
IDENTITIES = {
    tag: dict(case_tag=tag, input_sha256=(tag * 64), trace_sha256=(tag.upper().lower() * 64)) for tag in "abc"
}


def make_receipt(mode="concurrent", pair=0, repetition=0, wall=6.0, order=1):
    tags = PAIRS[pair]
    cases = [
        dict(
            IDENTITIES[tag],
            request_seconds=wall if mode == "concurrent" else wall / 2,
            cuda_event_seconds=wall / 3,
            output_png_sha256=tag * 64,
        )
        for tag in tags
    ]
    return {
        "status": "PASS",
        "concurrency": 2,
        "arm": "full",
        "mode": mode,
        "pair": pair,
        "repetition": repetition,
        "measurement_order": order,
        "protocol": copy.copy(PROTOCOL),
        "source_sha": "d" * 64,
        "device_uuid": "GPU-123",
        "environment": {"python": "3", "torch": "2", "triton": "3", "vllm": "1"},
        "runtime_lane": "private_graph",
        "timing_boundary": "end-to-end request",
        "batch_wall_seconds": wall,
        "synchronized_before_return": True,
        "cases": cases,
        "correctness": {"all_match": True, "method": "png_sha256"},
        "throughput_requests_per_second": 2 / wall,
        "latency_p50_seconds": sorted(case["request_seconds"] for case in cases)[0],
        "peak_allocated_bytes": 60,
        "peak_reserved_bytes": 100,
        "streams": [41, 42] if mode == "concurrent" else [41],
    }


def test_concurrent_receipt_has_synchronized_timing_and_identity():
    from bench_concurrency import validate_measurement

    data = make_receipt()
    validate_measurement(data, PROTOCOL, IDENTITIES)
    data["cases"][0]["input_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="identity"):
        validate_measurement(data, PROTOCOL, IDENTITIES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("synchronized_before_return", False),
        ("streams", [41, 41]),
        ("batch_wall_seconds", float("nan")),
        ("correctness", {"all_match": False}),
    ],
)
def test_rejects_partial_or_asynchronous_measurements(field, value):
    from bench_concurrency import validate_measurement

    data = make_receipt()
    data[field] = value
    with pytest.raises(ValueError):
        validate_measurement(data, PROTOCOL, IDENTITIES)


def test_serial_batch_measures_both_requests():
    from bench_concurrency import validate_measurement

    data = make_receipt(mode="serial")
    validate_measurement(data, PROTOCOL, IDENTITIES)
    data["cases"][0]["request_seconds"] = data["batch_wall_seconds"]
    with pytest.raises(ValueError, match="sum"):
        validate_measurement(data, PROTOCOL, IDENTITIES)


def test_schedule_covers_all_calibration_pairs_in_both_orders():
    from bench_concurrency import measurement_schedule

    args = argparse.Namespace(
        arm="full",
        mode="concurrent",
        concurrency=2,
        matrix=True,
        repetitions=2,
        start_repetition=0,
    )
    schedule = list(measurement_schedule(args, tuple("abc")))
    for pair, tags in enumerate(PAIRS):
        outcomes = []
        for rep in (0, 1):
            rows = [
                x for x in schedule if x.arm == "full" and x.pair == pair and x.repetition == rep and len(x.cases) == 2
            ]
            assert [x.cases for x in rows] == [tags, tags]
            outcomes.append([x.mode for x in rows])
        assert outcomes[0] == outcomes[1][::-1]


def test_analysis_pairs_repetitions_by_batch_before_bootstrap():
    from analyze_concurrency import analyze_paired

    rows = [
        make_receipt("serial", pair, rep, wall, order=rep)
        if mode == "serial"
        else make_receipt(mode, pair, rep, wall, order=1 - rep)
        for pair in range(3)
        for rep in range(2)
        for mode, wall in (("serial", (8.0 if rep == 0 else 24.0)), ("concurrent", (4.0 if rep == 0 else 24.0)))
    ]
    result = analyze_paired(rows, PROTOCOL, IDENTITIES)
    assert result["point_estimate"] == 1.5  # mean of ratios 2.0 and 1.0, not a ratio of sums
    assert result["confidence_interval_95"] == [1.5, 1.5]
    assert result["decision"] == "GO"
    assert result["bootstrap_draws"] == 5000 and result["bootstrap_seed"] == 110


def test_analysis_requests_two_more_repetitions_for_threshold_crossing():
    from analyze_concurrency import analyze_paired

    rows = [
        make_receipt(mode, pair, rep, wall, order=(rep if mode == "serial" else 1 - rep))
        for pair, ratio in enumerate((1.0, 1.1, 1.3))
        for rep in range(2)
        for mode, wall in (("serial", 10.0 * ratio), ("concurrent", 10.0))
    ]
    result = analyze_paired(rows, PROTOCOL, IDENTITIES)
    assert result["additional_repetitions"] == 2
    assert result["decision"] is None


def test_analysis_refuses_missing_pair_or_device_mix():
    from analyze_concurrency import analyze_paired

    rows = [make_receipt("serial", order=0), make_receipt()]
    rows[-1]["device_uuid"] = "GPU-other"
    with pytest.raises(ValueError):
        analyze_paired(rows, PROTOCOL, IDENTITIES)


def test_f1_gate_rejects_missing_pairs_before_measurement(tmp_path):
    from bench_concurrency import load_isolation_gate

    with pytest.raises(ValueError, match="F1"):
        load_isolation_gate(tmp_path, PAIRS, PROTOCOL, IDENTITIES, "d" * 64, "GPU-123")


@pytest.mark.skipif(
    os.environ.get("SPARSEEDIT_VERIFY_F2") != "1",
    reason="Set SPARSEEDIT_VERIFY_F2=1 to verify actual GPU evidence.",
)
def test_measured_f2_receipt_exists_and_passes():
    default_root = STUDY / "artifacts" / "concurrency_v1"
    root = Path(os.environ.get("SPARSEEDIT_F2_ROOT", default_root))
    data = json.loads((root / "full_concurrent_n2_r0" / "completion.json").read_text())
    from bench_concurrency import load_protocol_cases, validate_measurement

    protocol, cases = load_protocol_cases()
    validate_measurement(data, protocol, cases)
    assert data["status"] == "PASS" and data["concurrency"] == 2


def test_true_batch_requires_every_cfg_forward_at_batch_two():
    from bench_concurrency import validate_measurement

    data = make_receipt()
    data.update(arm="dense", mode="true_batch", streams=[41], transformer_batch_sizes=[2] * 56)
    validate_measurement(data, PROTOCOL, IDENTITIES)
    data["transformer_batch_sizes"][-1] = 1
    with pytest.raises(ValueError, match="batch-dimension"):
        validate_measurement(data, PROTOCOL, IDENTITIES)


def test_gate_uses_one_passing_lane_for_all_pairs(tmp_path):
    from bench_concurrency import load_isolation_gate

    def write_gate(tags, lane, suffix):
        receipt = tmp_path / ("__".join(tags) + suffix) / "completion.json"
        receipt.parent.mkdir()
        receipt.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "protocol": PROTOCOL,
                    "source_sha": "d" * 64,
                    "device_uuid": "GPU-123",
                    "runtime_lane": lane,
                    "correctness": {"all_match": True},
                    "cases": [dict(IDENTITIES[tag], output_png_sha256=tag * 64) for tag in tags],
                }
            )
        )

    write_gate(PAIRS[0], "private_graph", "")
    for tags in PAIRS:
        write_gate(tags, "private_eager", "__private_eager_attempt1")
    selected = load_isolation_gate(tmp_path, PAIRS, PROTOCOL, IDENTITIES, "d" * 64, "GPU-123")
    assert {row["runtime_lane"] for row in selected.values()} == {"private_eager"}
    for row in selected.values():
        assert len(row["receipt_sha256"]) == 64


def test_gpu_timeline_measures_kernel_overlap_not_launch_overlap():
    from bench_concurrency import summarize_gpu_overlap

    def kernel(stream, start, duration):
        return dict(cat="kernel", ph="X", ts=start, dur=duration, args={"device": 0, "stream": stream})

    profile = {
        "traceEvents": [
            kernel(7, 0, 10),
            kernel(8, 7, 15),
            kernel(7, 24, 9),
            dict(cat="cuda_runtime", ph="X", ts=0, dur=100, args={"device": 0, "stream": 7}),
        ]
    }
    result = summarize_gpu_overlap(profile)
    assert result["per_stream_kernel_count"] == {7: 2, 8: 1}
    assert result["gpu_kernel_active_microseconds"] == 31
    assert result["gpu_multi_stream_overlap_microseconds"] == 3


def test_analyzer_accepts_moved_f1_receipts_and_detects_changed_outputs(tmp_path):
    from analyze_concurrency import _verify_observations
    from bench_concurrency import sha256_file

    f1 = tmp_path / "returned_isolation"
    f1.mkdir()
    anchors = []
    for index, tags in enumerate(PAIRS):
        source = f1 / str(index)
        source.mkdir()
        path = source / "completion.json"
        row = make_receipt(pair=index)
        row.update(phase="F1", concurrency=2)
        serial = f1 / "serial_reference"
        serial.mkdir(exist_ok=True)
        for case in row["cases"]:
            data = case["case_tag"].encode()
            png = source / f"{case['case_tag']}.png"
            png.write_bytes(data)
            (serial / png.name).write_bytes(data)
            digest = sha256_file(png)
            case.update(output_png_sha256=digest, serial_png_sha256=digest)
        path.write_text(json.dumps(row))
        anchors.append({"path": f"/remote/original/{index}/completion.json", "sha256": sha256_file(path)})

    f2 = tmp_path / "returned_concurrency"
    f2.mkdir()
    record = make_receipt()
    record["isolation_receipts"] = anchors
    for case in record["cases"]:
        png = f2 / f"{case['case_tag']}.png"
        png.write_bytes(case["case_tag"].encode())
        case.update(output_png=png.name, output_png_sha256=sha256_file(png))

    path = f2 / "completion.json"
    _verify_observations([(path, record)], PROTOCOL, IDENTITIES, f1)
    (f1 / "0/a.png").write_bytes(b"changed F1 output")
    with pytest.raises(ValueError, match="F1 output SHA-256"):
        _verify_observations([(path, record)], PROTOCOL, IDENTITIES, f1)
    (f1 / "0/a.png").write_bytes(b"a")
    (f2 / "a.png").write_bytes(b"changed after generating receipt")
    with pytest.raises(ValueError, match="F2 output SHA-256"):
        _verify_observations([(path, record)], PROTOCOL, IDENTITIES, f1)
