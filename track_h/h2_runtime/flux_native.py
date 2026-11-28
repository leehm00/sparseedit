# SPDX-License-Identifier: Apache-2.0
"""Frozen FLUX inputs and the single-device Track A native entry for Track H."""

import base64
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from .controls import protocol

RUNTIME_SOURCE = Path(__file__).resolve().parent / "source"
BUNDLE_SHA256 = "e559de093f959c3985faf4ed911d49efb4d10b89da5a4bcf53f1a3fd4563776d"
SEED = 110
ARMS = ("fresh_dense", "direct_eager", "standard_fastops_eager", "full", "joint_cfg", "segmented_int8")
DENSE_STEPS = (0, 1, 2, 3, 4, 5, 6, 7, 23)
SPARSE_STEPS = tuple(i for i in range(28) if i not in DENSE_STEPS)


class GPUIsolationError(RuntimeError):
    """A native request cannot proceed under the configured GPU isolation."""


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def trace_digest(trace):
    return protocol.trace_digest(trace)


def token_count(side):
    return (side // 16) ** 2


def load_bundle(flux_assets):
    """Return the original v4 source order and verify all transferable files."""
    flux_assets = Path(flux_assets)
    path = flux_assets / "bundle.json"
    if sha256(path) != BUNDLE_SHA256:
        raise ValueError("FLUX frozen source bundle SHA-256 differs")
    bundle = read_json(path)
    cases = bundle["cases"]
    if len(cases["calibration"]) != 3 or len(cases["test"]) != 48:
        raise ValueError("FLUX cohort requires 3 calibration and 48 independent test sources")
    seen_sources = set()
    for case in cases["calibration"] + cases["test"]:
        source = case["source_pixel_sha256"]
        if source in seen_sources or case["input_filename"] != f"{source}.png":
            raise ValueError(f"FLUX source identity collision or filename mismatch for {case['case_tag']}")
        seen_sources.add(source)
        if sha256(flux_assets / "inputs" / case["input_filename"]) != case["prepared_sha256"]:
            raise ValueError(f"FLUX source file digest differs for {case['case_tag']}")
        load_frozen_trace(flux_assets, case)
    return bundle


def load_frozen_trace(flux_assets, case):
    path = Path(flux_assets) / "traces" / f"{case['case_tag']}.json"
    if sha256(path) != case["trace_sha256"]:
        raise ValueError(f"FLUX frozen trace file digest differs for {case['case_tag']}")
    return validate_trace(read_json(path), 768, .2)


def verify_model(model, bundle):
    """Verify the frozen Diffusers checkpoint components consumed by Omni."""
    model = Path(model)
    if not model.is_absolute() or not model.is_dir():
        raise ValueError("models.flux must be an existing absolute local checkpoint directory")
    expected = {
        name: digest for name, digest in bundle["model_hashes"].items()
        if "/" in name or name in ("configuration.json", "model_index.json")
    }
    for name, digest in expected.items():
        if sha256(model / name) != digest:
            raise ValueError(f"FLUX checkpoint SHA-256 differs for {name}")
    return expected


def validate_trace(trace, side, ratio):
    if trace.get("kind") != "flux_sparseedit_fixed_mask_trace" or trace.get("schema_version") != 1:
        raise ValueError("unexpected FLUX replay trace schema")
    if trace.get("sha256") != trace_digest(trace):
        raise ValueError("FLUX replay trace payload digest differs")
    n = token_count(side)
    k = int(n * ratio)
    expected = {
        "ratio": ratio, "num_inference_steps": 28, "warmup_steps": 8, "refresh_interval": 15,
        "true_cfg_scale": 4.0, "true_cfg": True, "selector_mode": "forward", "selector_protocol": "paper",
        "generated_token_count": n, "reference_token_count": n,
        "grid_height": side // 16, "grid_width": side // 16, "active_token_count": k,
    }
    config = trace["config"]
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("FLUX replay trace shape or protocol differs")
    steps = trace["steps"]
    if len(steps) != 28:
        raise ValueError("FLUX replay trace requires all 28 steps")
    for i, step in enumerate(steps):
        phase = "warmup" if i < 8 else "refresh_full" if i == 23 else "sparse"
        action = "none" if i < 7 else "store_full" if i in (7, 23) else "update_active"
        if (step.get("step"), step.get("phase"), step.get("kv_action")) != (i, phase, action):
            raise ValueError("FLUX replay trace chronology differs")
        indices = step.get("indices")
        if i in DENSE_STEPS:
            if indices is not None:
                raise ValueError("FLUX dense trace step carries sparse indices")
            continue
        if (not isinstance(indices, list) or len(indices) != k
                or any(type(j) is not int or not 0 <= j < n for j in indices)
                or indices != sorted(set(indices))):
            raise ValueError("FLUX replay requires unique sorted exact-K sparse indices")
        if "active_mask_b64" in step:
            mask = base64.b64decode(step["active_mask_b64"], validate=True)
            if len(mask) != n // 8 or sum(bits.bit_count() for bits in mask) != k or any(
                not mask[j // 8] & (1 << (7 - j % 8)) for j in indices
            ):
                raise ValueError("FLUX trace packed mask differs from its indices")
    return trace


def request_options(arm, ratio=.2, trace_path=None, *, collect_trace=False, side=768):
    if arm not in ARMS:
        raise ValueError(f"unknown H2 FLUX arm {arm!r}")
    can_collect = arm == "full" or (arm == "segmented_int8" and side == 1536)
    if collect_trace and (trace_path is not None or not can_collect):
        raise ValueError("FLUX trace collection requires unreplayed serial Full-v4 or the 1536 capacity probe")
    if arm == "fresh_dense":
        return "dense", {"sparseedit_return_metadata": True}
    base = arm if arm in ("direct_eager", "standard_fastops_eager") else "full"
    options = protocol.options(base, trace_path or "")
    options.update(sparseedit_ratio=ratio, sparseedit_collect_trace=collect_trace)
    if arm == "joint_cfg":
        use_budget_policy = int(token_count(side) * ratio) <= 921
        options.update(
            sparseedit_cfg_execution="budget" if use_budget_policy else "batched",
            sparseedit_template_capacity=1,
        )
    if arm == "segmented_int8":
        options["sparseedit_reference_kv_int8"] = True
    return "sparseedit", options


def _nvidia_smi(*args):
    try:
        return subprocess.check_output(["nvidia-smi", *args], text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as error:
        raise GPUIsolationError(f"cannot verify Hopper GPU using nvidia-smi: {error}") from error


def _owned_pids():
    ps = subprocess.check_output(["ps", "-e", "-o", "pid=,ppid="], text=True, timeout=10)
    parents = dict(tuple(map(int, line.split())) for line in ps.splitlines())
    own = os.getpid()
    owned = {own}
    for pid in parents:
        ancestor = parents[pid]
        while ancestor > 0 and ancestor != own:
            ancestor = parents.get(ancestor, 0)
        if ancestor == own:
            owned.add(pid)
    return owned


def verify_gpu(uuid, *, worker_active=False, allow_self=False):
    if (not re.fullmatch(r"GPU-[0-9a-fA-F-]+", uuid)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid
            or os.environ.get("SPARSEEDIT_SINGLE_REQUEST") != "1"):
        raise GPUIsolationError("run FLUX through the Track H orchestrator with one inherited Hopper UUID")
    devices = list(csv.reader(
        _nvidia_smi("--query-gpu=uuid,name", "--format=csv,noheader").splitlines(), skipinitialspace=True
    ))
    names = [name for device_uuid, name in devices if device_uuid == uuid]
    if len(names) != 1 or not re.search(r"\b(?:H20|H200)\b", names[0]):
        raise GPUIsolationError(f"configured GPU is not an NVIDIA H20/H200: {uuid}: {names}")
    apps = csv.reader(
        _nvidia_smi("--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits").splitlines(),
        skipinitialspace=True,
    )
    own = _owned_pids() if worker_active else {os.getpid()} if allow_self else set()
    compute_pids = [int(row[1]) for row in apps if row and row[0] == uuid]
    foreign = sorted(set(compute_pids) - own)
    if foreign:
        raise GPUIsolationError(f"foreign compute PIDs observed on {uuid}: {foreign}")
    return {"gpu_uuid": uuid, "device": names[0], "compute_pids": sorted(compute_pids)}


@contextmanager
def native_worker(
    model_path, *, gpu_uuid, dense_graph=False, paper_budget=False, paper_joint=False,
    profile=False, route=False, session="off", cache_backend=None, cache_config=None, source_root=None,
):
    """Keep the proven BF16, all-resident native worker on the inherited GPU."""
    verify_gpu(gpu_uuid, allow_self=True)
    from .controls.run_flux_specialized_runtime import NativeWorker

    source = Path(source_root or RUNTIME_SOURCE).resolve()
    prior_paths = list(sys.path)
    paths = [str(source), *(p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)]
    env = {
        "PYTHONPATH": os.pathsep.join(paths),
        "SPARSEEDIT_TRACK_A_GRAPH_MODE": "v4",
        "SPARSEEDIT_TRACK_A_DENSE_GRAPH": str(int(dense_graph)),
        "SPARSEEDIT_TRACK_A_PAPER_BUDGET": str(int(paper_budget)),
        "SPARSEEDIT_TRACK_A_PAPER_JOINT_CFG": str(int(paper_joint)),
        "SPARSEEDIT_TRACK_A_ROUTE": str(int(route)),
        "SPARSEEDIT_TRACK_A_SESSION": session,
        "SPARSEEDIT_TRACK_A_DEVICE_LEDGER": str(int(profile)),
        "SPARSEEDIT_TRACK_A_FORMAL_MEMORY": "1",
    }
    with patch.dict(os.environ, env):
        sys.path.insert(0, str(source))
        try:
            engine = NativeWorker(SimpleNamespace(model=model_path, cache_backend=cache_backend, cache_config=cache_config))
            engine.track_h_gpu_uuid = gpu_uuid
            try:
                yield engine
            finally:
                engine.close()
                verify_gpu(gpu_uuid, allow_self=True)
        finally:
            sys.path[:] = prior_paths


def run_native(worker, image_path, instruction, side, method, extras, output=None):
    """Use the same prepared RGB resize as Track A, outside the timed request."""
    started = time.perf_counter()
    with worker.pil_image.open(image_path) as original:
        image = original.convert("RGB").resize((side, side))
    try:
        pixel_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
        resize_s = time.perf_counter() - started
        before = verify_gpu(worker.track_h_gpu_uuid, worker_active=True)
        result = worker.run(
            image, instruction, method=method, extra_args=extras,
            seed=SEED, height=side, width=side, output_path=output,
        )
        after = verify_gpu(worker.track_h_gpu_uuid, worker_active=True)
    finally:
        image.close()
    if (result["resolved_height"], result["resolved_width"]) != (side, side):
        raise ValueError("native FLUX output resolution differs from the requested square")
    return {
        **result, "prepared_input_pixel_sha256": pixel_sha256, "input_resize_s": resize_s,
        "gpu_before": before, "gpu_after": after,
    }
