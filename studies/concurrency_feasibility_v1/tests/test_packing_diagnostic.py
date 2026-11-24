from pathlib import Path
import sys

import torch

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "scripts"))

from bench_packing_diagnostic import project, summarize_trials


def test_packed_projection_preserves_request_rows():
    weight = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    rows = [torch.full((3, 2), i, dtype=weight.dtype) for i in (1, 2, 3)]
    loop = project("loop", rows, weight)
    packed = project("packed", rows, weight)
    assert len(packed) == len(loop) == 3
    for left, right in zip(loop, packed, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_packing_summary_pairs_event_times_and_reports_scope():
    row = summarize_trials("expand", 4, [
        {"loop_ms": 4.0, "packed_ms": 2.0}, {"loop_ms": 9.0, "packed_ms": 3.0},
    ])
    assert row["requests"] == 4
    assert row["median_paired_speedup"] == 2.5
    assert row["scope"] == "synthetic_projection_including_pack_and_split"
