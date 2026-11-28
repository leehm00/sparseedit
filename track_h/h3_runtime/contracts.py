# SPDX-License-Identifier: Apache-2.0
"""Frozen methods, schedules, and artifact identity for H3."""

import hashlib
import json
from pathlib import Path

ARM_NAMES = (
    "fresh_dense", "dense_graph", "teacache", "toca", "ras", "regione", "spotedit",
    "sparseedit_k_only", "sparseedit_k_b_lane_e", "sparseedit_k_b_joint_cfg_lane_q",
)
FIXED20 = "sparseedit_fixed20_online_selector"


def panels():
    no_spot = {"spotedit": "the available native FLUX SpotEdit implements CFG1, while H3 uses two branches at CFG4"}
    shape_bound = dict.fromkeys(ARM_NAMES[-3:], "the frozen FLUX W8 router and online controller use 2304+2304 tokens")
    no_acceleration = dict.fromkeys(
        ARM_NAMES[2:7], "no matching native 1024px baseline on the frozen model protocol",
    )
    no_online = {
        "sparseedit_k_only": "no admitted adaptive K-only router on this model",
        "sparseedit_k_b_lane_e": "no admitted native K+B controller on this model",
        "sparseedit_k_b_joint_cfg_lane_q": "no admitted native K+B with within-request B2 on this model",
    }
    definitions = [
        ("flux", 768, no_spot, []),
        ("flux", 1024, {**no_spot, **shape_bound}, []),
        ("longcat", 1024, {**no_acceleration, **no_online}, []),
        ("qwen", 1024, {
            **no_acceleration, **no_online,
            "dense_graph": "H1 admits sparse transformer graphs only",
        }, [FIXED20]),
    ]
    return [
        {"model": model, "resolution": side, "arms": [arm for arm in ARM_NAMES if arm not in gaps],
         "gaps": gaps, "supplemental": supplementary}
        for model, side, gaps, supplementary in definitions
    ]


def groups(panel, cases, mode):
    count = {"cal3": 3, "formal": 48}[mode]
    if len(cases) != count or len({case["source_pixel_sha256"] for case in cases}) != count:
        raise ValueError(f"{mode} requires {count} frozen distinct H3 sources")
    rows = [(rep, case) for rep, order in ((0, cases), (1, cases[::-1])) for case in order]
    return [(arm, rows) for arm in (*panel["arms"], *panel["supplemental"])]


def validate_cohorts(mode, cohorts):
    if mode == "formal":
        comparable = {
            model: {(case["case_tag"], case["source_pixel_sha256"], case["instruction"]) for case in rows}
            for model, rows in cohorts.items()
        }
        if any(rows != comparable["flux"] or len(rows) != 48 for rows in comparable.values()):
            raise ValueError("frozen H3 formal cohort differs in source identity or edit instruction")


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_sha(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def artifact(root, relative, expected_sha):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute() or file_sha(path) != expected_sha:
        raise ValueError(f"wrong artifact SHA-256 or path: {relative}")
    return path
