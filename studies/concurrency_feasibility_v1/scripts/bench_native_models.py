#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure standalone, resident dense throughput on a single H200."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import socket
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from analyze_concurrency import summarize_arms
from bench_concurrency import (
    BUNDLE_ROOT,
    MODEL_CLASS,
    NEGATIVE_PROMPT,
    STUDY,
    Measurement,
    _check_measurement_device,
    _environment,
    _native_forward,
    _unwrap_images,
    _write_json,
    calibration_pairs,
    load_pipeline,
    make_request,
    read_json,
    resize_case,
    sha256_file,
    source_tree_sha,
)
from run_h200_phase_a import EXPECTED

MODELS = {
    "flux": ("FLUX.1-Kontext-dev", MODEL_CLASS, 768, 1.0, 4.0),
    "longcat": ("LongCat-Image-Edit", "LongCatImageEditPipeline", 1024, 3.5, 3.5),
    "qwen": ("Qwen-Image-Edit", "QwenImageEditPipeline", 1024, 1.0, 4.0),
}
TIMING_BOUNDARY = (
    "Local worker synchronized request wall, including native preprocessing, encoding, "
    "denoising, VAE decode and image postprocessing. Initial input resize, model load, "
    "checkpoint hashing, warmup and PNG write are separate."
)


def model_protocol(kind, size=None):
    model, model_class, side, guidance, cfg = MODELS[kind]
    if size is not None:
        if size != side and not (kind == "flux" and size == 1024):
            raise ValueError(f"Unsupported {kind} dense benchmark size: {size}")
        side = size
    return {
        "model": model,
        "model_class": model_class,
        "height": side,
        "width": side,
        "dtype": "bfloat16",
        "steps": 28,
        "guidance_scale": guidance,
        "true_cfg_scale": cfg,
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": 110,
        "cpu_offload": False,
        "layerwise_offload": False,
        "vae_use_tiling": True,
        "outputs_per_request": 1,
    }


def measurement_cells(kind, tags, repetitions):
    pairs = calibration_pairs(tags)
    groups = {"serial": tuple((tag,) for tag in tags)}
    if kind == "flux":
        groups["true_batch"] = pairs
    for mode, cells in groups.items():
        for repeat in range(repetitions):
            order = list(enumerate(cells))
            if repeat % 2:
                order.reverse()
            for index, cases in order:
                yield Measurement("dense", mode, index, repeat, cases)


def identification(kind, directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Expected a local model directory: {directory}")
    model_class = model_protocol(kind)["model_class"]
    actual = read_json(directory / "model_index.json").get("_class_name")
    if actual != model_class:
        raise ValueError(f"Expected {model_class}, found {actual!r}")
    if kind == "flux":
        expected = read_json(BUNDLE_ROOT / "bundle.json")["model_hashes"]
        names = sorted(expected)
    else:
        expected = None
        names = sorted(
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and all(not part.startswith(".") for part in path.relative_to(directory).parts)
        )
        if not any(name.startswith("transformer/") and name.endswith((".safetensors", ".bin")) for name in names):
            raise FileNotFoundError(f"No local transformer checkpoint found in {directory}")
    files = {}
    for name in names:
        path = directory / name
        digest = sha256_file(path)
        if expected is not None and digest != expected[name]:
            raise ValueError(f"Frozen FLUX model hash differs: {name}")
        files[name] = digest
    digest = sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"path": str(directory), "model_class": actual, "files": files, "model_sha256": digest}


def load_calibration():
    bundle = read_json(BUNDLE_ROOT / "bundle.json")
    rows = {}
    for case in bundle["cases"]["calibration"]:
        tag = case["case_tag"]
        if tag in rows or Path(tag).name != tag or tag in ("", ".", ".."):
            raise ValueError(f"Invalid calibration case tag: {tag!r}")
        path = BUNDLE_ROOT / "inputs" / case["input_filename"]
        if sha256_file(path) != case["prepared_sha256"]:
            raise ValueError(f"Calibration input hash differs: {tag}")
        rows[tag] = (case, path, BUNDLE_ROOT / "traces" / f"{tag}.json", {})
    if len(rows) != 3 or len({case["source_pixel_sha256"] for case, *_ in rows.values()}) != 3:
        raise ValueError("Expected three distinct frozen calibration inputs")
    return rows


@contextmanager
def native_worker(kind, model, protocol):
    if kind == "flux":
        with load_pipeline(model) as (pipeline, postprocess):
            yield pipeline, None, postprocess
        return

    import torch
    from vllm.config import set_current_vllm_config
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.registry import get_diffusion_post_process_func, get_diffusion_pre_process_func
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    config = OmniDiffusionConfig.from_kwargs(
        model=str(model),
        model_class_name=protocol["model_class"],
        dtype=torch.bfloat16,
        vae_use_tiling=True,
        cache_backend=None,
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enforce_eager=True,
        parallel_config=DiffusionParallelConfig(
            ulysses_degree=1, ring_degree=1, cfg_parallel_size=1, tensor_parallel_size=1,
        ),
    )
    config.enrich_config()
    preprocess = get_diffusion_pre_process_func(config)
    postprocess = get_diffusion_post_process_func(config)
    if preprocess is None or postprocess is None:
        raise RuntimeError(f"Native processors are unavailable for {protocol['model_class']}")
    worker = DiffusionWorker(local_rank=0, rank=0, od_config=config, skip_load_model=True)
    try:
        worker.load_model(load_format=config.diffusion_load_format)
        with (
            set_forward_context(vllm_config=worker.vllm_config, omni_diffusion_config=config),
            set_current_vllm_config(worker.vllm_config),
        ):
            yield worker.model_runner.pipeline, preprocess, postprocess
    finally:
        worker.shutdown()


def _resident_parameter_bytes(pipeline):
    placement = {}
    for param in pipeline.parameters():
        device = str(param.device)
        placement[device] = placement.get(device, 0) + param.numel() * param.element_size()
    if set(placement) != {"cuda:0"}:
        raise RuntimeError(f"Model parameters are not fully resident on the selected CUDA device: {placement}")
    return placement


def _edit_request(case, image, protocol):
    import torch
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras import build_image_to_image_prompt

    params = OmniDiffusionSamplingParams(
        generator=torch.Generator(device="cuda").manual_seed(protocol["seed"]),
        true_cfg_scale=protocol["true_cfg_scale"],
        guidance_scale=protocol["guidance_scale"],
        num_inference_steps=protocol["steps"],
        num_outputs_per_prompt=1,
        resolution=protocol["height"],
        height=protocol["height"],
        width=protocol["width"],
    )
    prompt = build_image_to_image_prompt(
        model_class_name=protocol["model_class"],
        prompt=case["instruction"],
        negative_prompt=protocol["negative_prompt"],
        input_image=image,
        height=protocol["height"],
        width=protocol["width"],
    )
    return OmniDiffusionRequest(prompt=prompt, sampling_params=params, request_id=case["case_tag"])


def _run_native_edit(pipeline, preprocess, postprocess, request):
    import torch
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    device_start = torch.cuda.Event(enable_timing=True)
    device_end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        started = time.perf_counter()
        device_start.record()
        output = pipeline.forward(DiffusionRequestBatch([preprocess(request)]))
        if output.error:
            raise RuntimeError(output.error)
        images = _unwrap_images(postprocess(output.output), 1)
        device_end.record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
    return images, [{
        "request_seconds": wall,
        "cuda_event_seconds": device_start.elapsed_time(device_end) * 1e-3,
    }], []


def _run_cell(kind, cell, cases, protocol, worker, root, *, warmup=False):
    import torch

    pipeline, preprocess, postprocess = worker
    result = {
        "status": "FAILED",
        "arm": cell.arm,
        "mode": cell.mode,
        "concurrency": len(cell.cases),
        "repetition": cell.repetition,
        "requested_cases": list(cell.cases),
        "warmup": warmup,
        "timing_boundary": TIMING_BOUNDARY,
        "cases": [],
    }
    root.mkdir(parents=True, exist_ok=False)
    try:
        with ExitStack() as resources:
            prepared = []
            for tag in cell.cases:
                started = time.perf_counter()
                image = resize_case(cases[tag], protocol)
                resources.callback(image.close)
                prepared.append((image, time.perf_counter() - started))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            if kind == "flux":
                requests = [
                    make_request(cases[tag], "dense", protocol, image=img)
                    for tag, (img, _) in zip(cell.cases, prepared, strict=True)
                ]
                images, timings, sizes = _native_forward(
                    pipeline, postprocess, requests, true_batch=cell.mode == "true_batch",
                )
            else:
                request = _edit_request(cases[cell.cases[0]][0], prepared[0][0], protocol)
                images, timings, sizes = _run_native_edit(pipeline, preprocess, postprocess, request)
            for image in images:
                resources.callback(image.close)
            if kind == "flux":
                result["transformer_batch_sizes"] = sizes
                if sizes != [len(cell.cases)] * (2 * protocol["steps"]):
                    raise RuntimeError(f"FLUX transformer batch shape witness failed: {sizes}")
            for tag, image, timing, (_, resize_s) in zip(cell.cases, images, timings, prepared, strict=True):
                if image.size != (protocol["width"], protocol["height"]):
                    raise RuntimeError(f"Output size changed for {tag}: {image.size}")
                case = cases[tag][0]
                row = {
                    "case_tag": tag,
                    "input_sha256": case["prepared_sha256"],
                    "source_pixel_sha256": case["source_pixel_sha256"],
                    "input_resize_seconds": resize_s,
                    **timing,
                }
                if not warmup:
                    output = root / f"{tag}.png"
                    started = time.perf_counter()
                    image.save(output)
                    row.update(output_png=output.name, png_write_seconds=time.perf_counter() - started)
                    row["output_png_sha256"] = sha256_file(output)
                result["cases"].append(row)
            if not warmup and cell.mode == "true_batch":
                if len({row["output_png_sha256"] for row in result["cases"]}) != len(cell.cases):
                    raise RuntimeError("Distinct FLUX inputs produced identical output PNGs")
            result.update(
                status="PASS",
                batch_wall_seconds=timings[0]["request_seconds"],
                synchronized_before_return=True,
            )
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="CAPACITY_BLOCKED", failure=str(error))
    except Exception as error:
        result.update(failure=str(error), traceback=traceback.format_exc())
    finally:
        result.update(
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
        _write_json(root / "completion.json", result)
    print(f"{result['status']} {root / 'completion.json'}", flush=True)
    return result


def _configure_environment(device_uuid):
    os.environ["CUDA_VISIBLE_DEVICES"] = device_uuid
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=tuple(MODELS), required=True)
    parser.add_argument("--model", type=Path, required=True, help="Local Diffusers model directory")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--size", type=int, choices=(768, 1024), help="Dense square resolution override for FLUX")
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("Select at least one measured repetition")
    kind = args.model_kind
    try:
        protocol = model_protocol(kind, args.size)
    except ValueError as error:
        parser.error(str(error))
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    model = args.model.resolve()
    data = {
        "schema_version": 1,
        "phase": "native_dense",
        "status": "BLOCKED",
        "model_kind": kind,
        "protocol": protocol,
        "timing_boundary": TIMING_BOUNDARY,
        "device_uuid": args.gpu_uuid,
        "host": socket.gethostname(),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "source_sha": source_tree_sha(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "repetitions": args.repetitions,
        "benchmark_size": protocol["height"],
        "native_independent_request_concurrency": "UNSUPPORTED",
        "arms": [],
        "warmups": [],
        "cells": [],
    }
    timings = []
    try:
        if not model.is_dir():
            raise FileNotFoundError(f"Expected a local model directory: {model}")
        if not re.fullmatch(r"GPU-[0-9a-f-]{36}", args.gpu_uuid, re.I):
            raise ValueError("Select the physical H200 device UUID")
        data["environment"] = _environment()
        if any(data["environment"][key] != expected for key, expected in EXPECTED.items()):
            raise ValueError("Native benchmarks require the frozen H200 Python, Torch, Triton and vLLM versions")
        cases = load_calibration()
        schedule = list(measurement_cells(kind, tuple(cases), args.repetitions))
        data["planned_cells"] = len(schedule)
        lock_path = Path(f"/tmp/sparseedit-{args.gpu_uuid}.lock")
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            data.update(_check_measurement_device(args.gpu_uuid))
            if not data["device_name"].startswith("NVIDIA H200") or data["total_bytes"] < 120 * 2**30:
                raise ValueError("Select a full physical H200")
            _configure_environment(args.gpu_uuid)
            if str(STUDY / "source") not in sys.path:
                sys.path.insert(0, str(STUDY / "source"))
            started = time.perf_counter()
            print(f"Fingerprinting local {protocol['model']} at {model}", flush=True)
            data["model_identity"] = identification(kind, model)
            data["model_fingerprint_seconds"] = time.perf_counter() - started

            import torch

            data["status"] = "FAILED"
            print(f"Loading {protocol['model']} on {args.gpu_uuid}", flush=True)
            started = time.perf_counter()
            with native_worker(kind, model, protocol) as worker:
                torch.cuda.synchronize()
                data["model_load_seconds"] = time.perf_counter() - started
                data["parameter_bytes_by_device"] = _resident_parameter_bytes(worker[0])
                warmed_modes = {}
                for cell in schedule:
                    if cell.mode not in warmed_modes:
                        folder = root / "warmup" / cell.mode
                        warmup = _run_cell(kind, cell, cases, protocol, worker, folder, warmup=True)
                        warmed_modes[cell.mode] = warmup["status"]
                        data["warmups"].append({
                            "mode": cell.mode, "status": warmup["status"],
                            "receipt": (folder / "completion.json").relative_to(root).as_posix(),
                        })
                    if warmed_modes[cell.mode] != "PASS":
                        continue
                    folder = root / "cells" / cell.directory
                    result = _run_cell(kind, cell, cases, protocol, worker, folder)
                    data["cells"].append({
                        "mode": cell.mode, "status": result["status"],
                        "receipt": (folder / "completion.json").relative_to(root).as_posix(),
                    })
                    if result["status"] == "PASS":
                        timings.append(result)
            if len(timings) == len(schedule):
                data["status"] = "PASS"
            elif timings:
                data["status"] = "PARTIAL"
            else:
                statuses = {row["status"] for row in [*data["warmups"], *data["cells"]]}
                data["status"] = "CAPACITY_BLOCKED" if statuses == {"CAPACITY_BLOCKED"} else "FAILED"
    except Exception as error:
        if type(error).__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"}:
            data["status"] = "CAPACITY_BLOCKED"
        elif timings:
            data["status"] = "PARTIAL"
        data.update(failure=str(error), traceback=traceback.format_exc())
    finally:
        data.update(
            arms=summarize_arms(timings),
            finished_at_utc=datetime.now(UTC).isoformat(),
        )
        _write_json(root / "completion.json", data)
    print(f"{data['status']} {root / 'completion.json'}", flush=True)
    return 0 if data["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
