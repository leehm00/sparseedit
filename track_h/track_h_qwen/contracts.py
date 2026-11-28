# SPDX-License-Identifier: Apache-2.0
"""Frozen inputs, schedules and evidence contracts for Track H1."""

from __future__ import annotations

import hashlib
import json
from math import isfinite
from pathlib import Path
import re
from statistics import mean

from .schedule import CALIBRATION_BUDGETS, MEASUREMENT_ARMS

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    "cases_calibration.json": "1a9fb10be85081c3dcc11fed179f4cfa0d6e10323d9befb3348a117dc7278070",
    "cases_test.json": "f5baf1a897377782f536d4b85277d2b36a258c2e4958f7f1b998ecd71b87a247",
    "cohort_bindings.json": "a6eac849c84a7b47df40dd9b3468781ffd741906b81b7af4e523daa228d83fbe",
    "model_inventory.json": "5cf78131989ebccb2c15427a6d5153cf0dfc03f54abbec78a16460bdc1499b2f",
    "session_prompts.json": "881517a49ca93f696c88385ca39820fcdc68f7090e22b826c9bd09720a5e6c5f",
}
PROTOCOL = {
    "model": "Qwen-Image-Edit",
    "dtype": "bfloat16",
    "height": 1024,
    "width": 1024,
    "steps": 28,
    "dense_prefix_steps": 8,
    "dense_refresh_step": 23,
    "batch_size": 1,
    "num_images_per_prompt": 1,
    "seed": 110,
    "guidance_scale": 1.0,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "cpu_offload": False,
    "layerwise_offload": False,
    "kv_offload": False,
    "vae_tiling": False,
}
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
FULL_STEPS = {*range(8), 23}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def digest_json(value):
    source = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(source.encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temp.replace(path)


def verify_artifact(root, relative, expected_sha):
    root = Path(root).resolve()
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe artifact path {relative!s}")
    resolved = (root / path).resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(), f"unsafe artifact path {relative!s}")
    require(sha_file(resolved) == expected_sha, f"artifact SHA mismatch for {relative!s}")
    return resolved


def frozen_json(root, name):
    return json.loads(verify_artifact(root, name, FROZEN[name]).read_text())


def inventory(mode, assets=None):
    require(mode in ("cal3", "formal"), "H1 run_mode must be cal3 or formal")
    assets = Path(assets) if assets is not None else ROOT / "payload_data" / "h1"
    cohort = frozen_json(assets, "cohort_bindings.json")["cases"]
    selected = {}
    for split, count in (("calibration", 5 if mode == "formal" else 3),
                         ("test", 48 if mode == "formal" else 0)):
        if not count:
            continue
        manifest = frozen_json(assets, f"cases_{split}.json")
        rows = cohort[split]
        require(len(rows) == len(manifest) == (24 if split == "calibration" else 48),
                f"wrong frozen {split} cohort count")
        require(len({r["source_pixel_sha256"] for r in rows}) == len(rows),
                f"duplicate frozen {split} source")
        selected[split] = []
        for case, binding in zip(manifest[:count], rows[:count], strict=True):
            require(all(case[k] == binding[k] for k in ("case_id", "image", "prompt", "split")),
                    "frozen manifest differs from cohort binding")
            require(Path(case["case_id"]).name == case["case_id"], "unsafe frozen case ID")
            path = verify_artifact(assets, Path("inputs") / case["image"], binding["input_sha256"])
            require(path.stat().st_size == binding["input_bytes"], "wrong frozen input byte length")
            selected[split].append({**case, "input_path": path, "input_sha256": binding["input_sha256"],
                                    "source_pixel_sha256": binding["source_pixel_sha256"]})
    calibration = selected["calibration"][:3]
    if mode == "cal3":
        return {"primary": calibration, "calibration": calibration, "session": []}
    heldout = selected["test"]
    require(not ({row["source_pixel_sha256"] for row in calibration}
                 & {row["source_pixel_sha256"] for row in heldout}), "H1 heldout sources overlap cal3")
    return {"primary": heldout, "calibration": calibration, "session": selected["calibration"]}


def model_identity(model, assets):
    frozen = frozen_json(assets, "model_inventory.json")
    model = Path(model)
    require(model.is_absolute() and model.is_dir(), "models.qwen must be an absolute local model directory")
    component_names = {"processor", "scheduler", "text_encoder", "tokenizer", "transformer", "vae"}
    rows = [entry for entry in frozen["files"]
            if Path(entry["path"]).parts[0] in component_names | {"model_index.json"}]
    file_names = {entry["path"] for entry in rows}
    actual = {
        path.relative_to(model).as_posix()
        for name in component_names for path in (model / name).rglob("*")
        if path.is_file() and not any(p.startswith(".") for p in path.relative_to(model).parts)
    }
    require(actual == file_names - {"model_index.json"}, "Qwen runtime model file inventory differs")
    for entry in rows:
        path = model / entry["path"]
        require(path.is_file() and path.stat().st_size == entry["bytes"], f"wrong model file size: {entry['path']}")
        require(sha_file(path) == entry["sha256"], f"wrong model file SHA: {entry['path']}")
    index = json.loads((model / "model_index.json").read_text())
    require(index.get("_class_name") == "QwenImageEditPipeline", "original Qwen-Image-Edit model required")
    return {
        "path": str(model),
        "repository": frozen["repository"],
        "revision": frozen["head_commit"],
        "provider_manifest_sha256": frozen["manifest_sha256"],
        "runtime_file_sha256": digest_json(rows),
        "runtime_files": rows,
    }


class SessionCache:
    """Keep exactly one source on device and defer reuse until the next request."""

    def __init__(self):
        self.key = None
        self.items = {}
        self.use_previous = False

    def begin(self, key):
        self.use_previous = key == self.key and bool(self.items)
        if not self.use_previous:
            self.items.clear()
        self.key = key
        return self.use_previous

    def get(self, name, layout, compute):
        previous = self.items.get(name)
        require(previous is None or previous[0] == layout, f"cached {name} layout differs")
        if self.use_previous:
            require(previous is not None, f"cached request did not populate {name}")
            return previous[1], True
        value = compute()
        # Both CFG branches run the vision tower in the cold request.
        if previous is None:
            self.items[name] = (layout, value)
        return value, False


def validate_graph(runtime, *, requested):
    require(runtime.get("graph_requested") is requested, "graph request flag differs")
    require(not runtime.get("graph_fallbacks"), "graph run contains fallback")
    expected = {"positive": 19, "negative": 19} if requested else {"positive": 0, "negative": 0}
    require(runtime.get("graph_replays") == expected, "graph replay count differs by CFG branch")
    captures = {"positive": 1, "negative": 1} if requested else {}
    require(runtime.get("graph_captures") == captures, "graph capture count differs by CFG branch")


def validate_session_pair(uncached, cached, *, expect_hit):
    require(bool(uncached.get("conditioning")), "session lacks parity fingerprints")
    require(uncached["output_sha256"] == cached["output_sha256"]
            and uncached["conditioning"] == cached["conditioning"], "L3 output or conditioning parity failed")
    require(uncached["cache_events"] == {"vision": [False, False], "vae": [False]},
            "uncached session executed with an unexpected hit count")
    require(cached["cache_events"] == {"vision": [expect_hit, expect_hit], "vae": [expect_hit]},
            "cached session executed with an unexpected hit count")


def timing_summary(rows, *, dense):
    full = sparse = 0.0
    full_count = sparse_count = 0
    for row in rows:
        seconds = row["cuda_seconds"]
        require(isfinite(seconds) and seconds >= 0, "invalid CUDA segment duration")
        if row["segment"] == "denoise_step":
            action = "full" if dense or row["step"] in FULL_STEPS else "sparse"
            require(row["action"] == action, "segment action differs from denoising schedule")
            if action == "full":
                full += seconds
                full_count += 1
            else:
                sparse += seconds
                sparse_count += 1
    steps = [row["step"] for row in rows if row["segment"] == "denoise_step"]
    require(steps == list(range(28)), "request ledger requires one interval for each denoising step")
    for segment, count in {"vl_encode": 2, "reference_vae_encode": 1, "vae_decode": 1}.items():
        require(sum(row["segment"] == segment for row in rows) == count,
                f"request ledger requires {count} {segment} interval(s)")
    return {
        "full_steps": full_count,
        "sparse_steps": sparse_count,
        "full_cuda_seconds": full,
        "sparse_cuda_seconds": sparse,
    }


def latency_observation(row):
    budget = row["budget_percent"]
    if row["split"] != "calibration" or row["timing_role"] != "measurement" or budget not in CALIBRATION_BUDGETS:
        return None
    runtime = next((arm for arm, _, _ in MEASUREMENT_ARMS if row["arm"] == f"{arm}{budget}"), None)
    if runtime is None:
        return None
    sparse = [
        segment["cuda_seconds"]
        for segment in row["segments"]
        if segment["segment"] == "denoise_step" and segment["action"] == "sparse"
    ]
    require(row["full_steps"] == 9 and len(sparse) == row["sparse_steps"] == 19,
            "H4 calibration requires the 9-full/19-sparse schedule")
    require(all(isfinite(seconds) and seconds > 0 for seconds in sparse), "invalid H4 sparse step time")
    require(0 < row["denoising_wall_seconds"] <= row["request_wall_seconds"] and row["full_cuda_seconds"] > 0,
            "invalid H4 full-step or wall time")
    return {
        "model": "qwen",
        "resolution": PROTOCOL["height"],
        "runtime": runtime,
        "split": "calibration",
        "workload": "fixed",
        "source_id": row["source_pixel_sha256"],
        "case_id": row["case_id"],
        "ratio": budget / 100,
        "repetition": row["repetition"],
        "request_wall_s": row["request_wall_seconds"],
        "denoise_wall_s": row["denoising_wall_seconds"],
        "full_steps": row["full_steps"],
        "sparse_steps": row["sparse_steps"],
        "reuse_steps": 0,
        "full_device_s": row["full_cuda_seconds"],
        "steady_sparse_device_s": mean(sparse[1:]),
        "steady_sparse_steps": len(sparse) - 1,
        "manifest_path": row["manifest_path"],
        "manifest_sha256": row["manifest_sha256"],
    }
