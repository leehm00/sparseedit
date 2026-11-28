#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Staged acceptance driver for the FLUX specialized SparseEdit runtime.

Subcommands
-----------
``validate`` / ``cpu-tests``
    No model and no CUDA.  Validate the frozen case manifest, input remapping,
    source hashes, variant matrix, hardware policy and (optionally) a fixed-mask
    trace artifact.
``smoke``
    Real model, 768px procedural request, 28 steps, W8, CFG4.  Runs
    ``baseline``, ``inplace``, ``transformed``, ``fused`` and ``graph`` with the
    exact baseline mask trace, saves lossless per-step audit tensors and key
    cache witnesses, and gates numerical comparisons.  ``graph`` must actually
    replay.
``benchmark``
    Real model, the three frozen requests in ``../inputs/cases.json`` at
    1024px.  Baseline first records the exact masks; every variant then replays
    the same trace.  Two repetitions per variant/request with deterministic
    rotation after an unmeasured warmup.
``analyze``
    CPU aggregation of paired quality/latency/memory, raw samples and coverage.

GPU stages re-execute this file as a child process with
``CUDA_VISIBLE_DEVICES`` set to the expected GPU UUID *only*.  The parent
prechecks hardware with ``nvidia-smi`` before any Torch/CUDA import and refuses
to start if the GPU is blocked by policy or has a foreign compute process.  It
never stops another job.

Diagnostics and timing are separated: ``smoke`` writes full audit tensors,
``benchmark`` defers all per-step host copies until after the timed denoising
interval.  Both use the same mask-trace artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
VARIANTS: tuple[str, ...] = ("baseline", "inplace", "transformed", "fused", "graph")
SPARSE_RATIO = 0.20
WARMUP_STEPS = 8
REFRESH_INTERVAL = 15
NUM_STEPS = 28
ACTIVE_RATIO = 0.20
SMOKE_HEIGHT = 768
SMOKE_WIDTH = 768
BENCH_HEIGHT = 1024
BENCH_WIDTH = 1024
SEED = 110
TRUE_CFG_SCALE = 4.0
GUIDANCE_SCALE = 1.0
NEGATIVE_PROMPT = " "
DEFAULT_MODEL = Path("/mnt/sdb/homie/flux_a40_kontext_28step_20260912/models/FLUX.1-Kontext-dev")
DEFAULT_REMOTE_PYTHON = Path("/mnt/sdb/homie/flux_a40_kontext_28step_20260912/envs/flux/bin/python")
SOURCE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES = SOURCE_ROOT.parent / "inputs" / "cases.json"
DEFAULT_INPUTS_ROOT = SOURCE_ROOT.parent / "inputs"
# node -> physical GPU indexes that are forbidden before 2026-11-25.
DEFAULT_BLOCKED_GPUS: Mapping[str, frozenset[int]] = {
    "node1": frozenset({0, 1}),
    "node3": frozenset({0, 1, 2, 3}),
}
TRACE_REQUIRED_SHAPE = {
    "schema_version": 1,
    "kind": "flux_sparseedit_fixed_mask_trace",
}
# Declared tolerances.  Copy-only paths compare bitwise.  A reorganized but
# non-fused RoPE path must reproduce the same per-row values; we still declare a
# tight bound for end-to-end BF16 velocities because kernel tiling may differ.
TRANSFORMED_OUTPUT_TOLERANCE = {"rtol": 2e-3, "atol": 2e-3}
FUSED_OUTPUT_TOLERANCE = {"rtol": 8e-3, "atol": 8e-3}
EXACT_TOLERANCE = {"rtol": 0.0, "atol": 0.0}


class DriverError(RuntimeError):
    """Raised for invalid artifacts, unsafe hardware or failed acceptance."""


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ----------------------------------------------------------------------------
# Small utilities (stdlib only at module import).
# ----------------------------------------------------------------------------
def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DriverError(f"missing artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise DriverError(f"invalid JSON at {path}: {error}") from error


def _node_name(explicit: str | None) -> str:
    if explicit:
        return explicit.strip().lower()
    host = socket.gethostname().split(".")[0].lower()
    match = re.match(r"(node\d+)", host)
    return match.group(1) if match else host


def policy_blocks_gpu(node: str | None, physical_index: int) -> bool:
    name = _node_name(node)
    return physical_index in DEFAULT_BLOCKED_GPUS.get(name, frozenset())


def _git_identity() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=SOURCE_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"git_commit": commit, "git_dirty": bool(status), "source_root": str(SOURCE_ROOT)}


def _model_artifact_hashes(model: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in ("config.json", "model_index.json", "scheduler/scheduler_config.json"):
        path = model / relative
        if path.is_file():
            hashes[relative] = sha256_file(path)
    return hashes


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    roots = [
        SOURCE_ROOT / "vllm_omni" / "diffusion" / "models" / "flux",
        SOURCE_ROOT / "examples" / "offline_inference" / "image_to_image",
    ]
    for root in roots:
        for path in sorted(root.glob("*.py")):
            digest.update(str(path.relative_to(SOURCE_ROOT)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


# ----------------------------------------------------------------------------
# Frozen cases and trace validation.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    """One frozen benchmark request.

    ``prepared_sha256`` identifies the *prepared* image bytes that are fed to
    the model (the file resolved from ``input_filename``).  ``source_pixel_sha256``
    is the source-image identity that the corpus encodes in the file-name stem;
    it is deliberately *not* the prepared file-byte digest and must never be
    compared against it.
    """

    case_tag: str
    category_slug: str
    instruction: str
    input_filename: str
    source_pixel_sha256: str
    prepared_sha256: str
    image_path: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> Case:
        required = (
            "case_tag",
            "category_slug",
            "instruction",
            "input_filename",
            "source_pixel_sha256",
            "prepared_sha256",
        )
        missing = [name for name in required if not row.get(name)]
        if missing:
            raise DriverError(f"case manifest row is missing {missing}: {dict(row)}")
        input_filename = str(row["input_filename"])
        source_pixel_sha256 = str(row["source_pixel_sha256"]).lower()
        prepared_sha256 = str(row["prepared_sha256"]).lower()
        for name, value in (
            ("source_pixel_sha256", source_pixel_sha256),
            ("prepared_sha256", prepared_sha256),
        ):
            if not _HEX64.fullmatch(value):
                raise DriverError(f"case {row['case_tag']!r} {name} is not a 64-char hex digest: {value!r}")
        # The corpus encodes the source identity in the file-name stem, so the
        # stem must agree with ``source_pixel_sha256``.
        stem = Path(input_filename).stem.lower()
        if stem != source_pixel_sha256:
            raise DriverError(
                f"case {row['case_tag']!r} input_filename stem {stem!r} does not match "
                f"source_pixel_sha256 {source_pixel_sha256!r}"
            )
        # Guard against the two identities being conflated: the prepared file
        # digest and the source identity describe different artifacts.
        if prepared_sha256 == source_pixel_sha256:
            raise DriverError(
                f"case {row['case_tag']!r} prepared_sha256 must identify the prepared image, "
                "not repeat source_pixel_sha256"
            )
        return cls(
            case_tag=str(row["case_tag"]),
            category_slug=str(row["category_slug"]),
            instruction=str(row["instruction"]),
            input_filename=input_filename,
            source_pixel_sha256=source_pixel_sha256,
            prepared_sha256=prepared_sha256,
            image_path=str(row.get("image_path", "")),
        )


def load_cases(path: Path) -> list[Case]:
    payload = read_json(path)
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise DriverError(f"unsupported case manifest schema at {path}")
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise DriverError(f"case manifest has no cases: {path}")
    cases = [Case.from_mapping(row) for row in rows]
    if len({case.case_tag for case in cases}) != len(cases):
        raise DriverError("case manifest contains duplicate case tags")
    for case in cases:
        if not case.input_filename or "/" in case.input_filename:
            raise DriverError(f"case {case.case_tag} input_filename must be a bare file name")
    return cases


def resolve_input_path(case: Case, inputs_root: Path, *, fallback_path: str | None = None) -> Path:
    candidate = (inputs_root / case.input_filename).resolve()
    if candidate.is_file():
        return candidate
    if fallback_path:
        fallback = Path(fallback_path)
        if fallback.is_file() and fallback.name == case.input_filename:
            return fallback.resolve()
    raise DriverError(
        f"input image for {case.case_tag!r} was not found by filename in {inputs_root} "
        f"(expected {case.input_filename})"
    )


def validate_cases(cases: Sequence[Case], inputs_root: Path, *, allow_missing: bool = False) -> dict[str, Any]:
    if len(cases) != 3 and not allow_missing:
        raise DriverError(f"the frozen benchmark expects exactly three cases, got {len(cases)}")
    rows = []
    for case in cases:
        if allow_missing and not (inputs_root / case.input_filename).is_file():
            rows.append(
                {
                    "case_tag": case.case_tag,
                    "input_filename": case.input_filename,
                    "status": "missing_input_allowed",
                }
            )
            continue
        path = resolve_input_path(case, inputs_root, fallback_path=case.image_path)
        digest = sha256_file(path)
        # The resolved file is the *prepared* image, so its bytes must match
        # ``prepared_sha256``.  ``source_pixel_sha256`` is the source identity
        # encoded in the stem (already checked in ``Case.from_mapping``).
        if digest != case.prepared_sha256:
            raise DriverError(
                f"prepared image hash mismatch for {case.case_tag}: {digest} != {case.prepared_sha256}"
            )
        rows.append(
            {
                "case_tag": case.case_tag,
                "input_filename": case.input_filename,
                "path": str(path),
                "sha256": digest,
                "prepared_sha256": case.prepared_sha256,
                "source_pixel_sha256": case.source_pixel_sha256,
                "status": "ok",
            }
        )
    return {"status": "ok", "case_count": len(rows), "cases": rows}


def validate_variants(variants: Sequence[str]) -> tuple[str, ...]:
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise DriverError(f"unsupported runtime variants: {unknown}; expected {list(VARIANTS)}")
    if "baseline" not in variants:
        raise DriverError("the variant matrix must include the strong index baseline")
    return tuple(variants)


def validate_trace_file(path: Path, *, cases: Sequence[Case] | None = None) -> dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from vllm_omni.diffusion.models.flux.specialized_plan import FixedMaskTrace, SparseTraceError

    try:
        trace = FixedMaskTrace.load(path)
        config = trace.config
        trace.validate(
            num_inference_steps=int(config.get("num_inference_steps", NUM_STEPS)),
            ratio=float(config.get("ratio", SPARSE_RATIO)),
            warmup_steps=int(config.get("warmup_steps", WARMUP_STEPS)),
            refresh_interval=int(config.get("refresh_interval", REFRESH_INTERVAL)),
            generated_token_count=int(config["generated_token_count"]),
            reference_token_count=int(config["reference_token_count"]),
            grid_height=int(config["grid_height"]),
            grid_width=int(config["grid_width"]),
            active_token_count=int(config["active_token_count"]),
        )
    except (SparseTraceError, KeyError, ValueError) as error:
        raise DriverError(f"invalid fixed-mask trace {path}: {error}") from error
    sparse_steps = [step for step in trace.steps if step.kv_action == "update_active"]
    return {
        "status": "ok",
        "path": str(path),
        "sha256": trace.sha256,
        "sparse_steps": len(sparse_steps),
        "refresh_steps": [step.step for step in trace.steps if step.phase == "refresh_full"],
        "index_digests": {
            step.step: hashlib.sha256(
                json.dumps(list(step.indices), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for step in sparse_steps
        },
    }


# ----------------------------------------------------------------------------
# Hardware precheck (nvidia-smi only, before any Torch import).
# ----------------------------------------------------------------------------
def _run_nvidia_smi(*query: str) -> list[list[str]]:
    command = ["nvidia-smi", *query, "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise DriverError(f"nvidia-smi precheck failed: {error}") from error
    rows = []
    for line in completed.stdout.splitlines():
        if line.strip():
            rows.append([item.strip() for item in line.split(",")])
    return rows


def hardware_precheck(
    *,
    physical_index: int,
    expected_uuid: str,
    node: str | None,
    refuse_foreign: bool = True,
) -> dict[str, Any]:
    if physical_index < 0:
        raise DriverError("--gpu-physical-index must be non-negative")
    expected = expected_uuid.strip()
    if not expected:
        raise DriverError("GPU stages require an explicit --expected-gpu-uuid")
    name = _node_name(node)
    if policy_blocks_gpu(name, physical_index):
        raise DriverError(
            f"GPU {physical_index} on {name} is blocked before 2026-11-25 by the task hardware policy"
        )
    gpus = _run_nvidia_smi("--query-gpu=index,uuid,name,memory.used")
    match = None
    for row in gpus:
        if len(row) < 3:
            continue
        if int(row[0]) == physical_index:
            match = row
            break
    if match is None:
        raise DriverError(f"physical GPU index {physical_index} was not reported by nvidia-smi")
    observed_uuid = match[1]
    if observed_uuid != expected:
        raise DriverError(
            f"GPU {physical_index} UUID mismatch: expected {expected}, observed {observed_uuid}"
        )
    foreign: list[dict[str, Any]] = []
    if refuse_foreign:
        for row in _run_nvidia_smi("--query-compute-apps=gpu_uuid,pid,process_name"):
            if len(row) >= 2 and row[0] == expected:
                foreign.append({"pid": row[1], "process_name": row[2] if len(row) > 2 else ""})
    if foreign:
        raise DriverError(
            f"GPU {physical_index} ({expected}) has foreign compute processes; refusing to interfere: {foreign}"
        )
    return {
        "node": name,
        "physical_index": physical_index,
        "uuid": expected,
        "name": match[2] if len(match) > 2 else "",
        "memory_used_mib": match[3] if len(match) > 3 else None,
        "foreign_compute_processes": foreign,
        "prechecked_at": utc_now(),
    }


def _launch_child(args: argparse.Namespace, stage: str) -> int:
    hardware = hardware_precheck(
        physical_index=args.gpu_physical_index,
        expected_uuid=args.expected_gpu_uuid,
        node=args.node,
    )
    write_json(args.output_root / "control" / "hardware.json", hardware)
    command = [
        str(args.python),
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--model",
        str(args.model),
        "--output-root",
        str(args.output_root),
        "--cases",
        str(args.cases),
        "--inputs-root",
        str(args.inputs_root),
        "--remote-inputs-root",
        str(args.remote_inputs_root or args.inputs_root),
        "--variants",
        ",".join(args.variants),
        "--repetitions",
        str(args.repetitions),
        "--seed",
        str(args.seed),
        "--gpu-physical-index",
        str(args.gpu_physical_index),
        "--expected-gpu-uuid",
        str(args.expected_gpu_uuid),
        "--node",
        str(hardware["node"]),
        "--audit",
        "1" if args.audit else "0",
        "--include-dense",
        "1" if args.include_dense else "0",
        "--resume",
        "1" if args.resume else "0",
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.expected_gpu_uuid)
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(SOURCE_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    record = {
        "stage": stage,
        "command": " ".join(shlex.quote(part) for part in command),
        "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
        "pythonpath": environment["PYTHONPATH"],
        "hardware": hardware,
        "launched_at": utc_now(),
    }
    write_json(args.output_root / "control" / f"{stage}_launch.json", record)
    print(f"Launching {stage} on GPU {args.gpu_physical_index} ({args.expected_gpu_uuid})")
    completed = subprocess.run(command, env=environment, cwd=str(SOURCE_ROOT), check=False)
    return int(completed.returncode)


# ----------------------------------------------------------------------------
# GPU worker (imports Torch only inside the child process).
# ----------------------------------------------------------------------------
class NativeWorker:
    """Persistent Omni worker with no cache backend and no parallel sharding."""

    def __init__(self, args: argparse.Namespace) -> None:
        import importlib

        self.args = args
        self.torch = importlib.import_module("torch")
        self.pil_image = importlib.import_module("PIL.Image")
        data_module = importlib.import_module("vllm_omni.diffusion.data")
        output_module = importlib.import_module("vllm_omni.diffusion.utils.image_output")
        param_module = importlib.import_module("vllm_omni.diffusion.utils.param_utils")
        omni_module = importlib.import_module("vllm_omni.entrypoints.omni")
        stage_params_module = importlib.import_module("vllm_omni.entrypoints.openai.stage_params")
        inputs_module = importlib.import_module("vllm_omni.inputs.data")
        extras_module = importlib.import_module("vllm_omni.model_extras")
        platforms_module = importlib.import_module("vllm_omni.platforms")

        self.extract_images_from_outputs = output_module.extract_images_from_outputs
        self.apply_declared_extra_args = param_module.apply_declared_extra_args
        self.clone_sampling_params = stage_params_module.clone_sampling_params
        self.OmniDiffusionSamplingParams = inputs_module.OmniDiffusionSamplingParams
        self.build_image_to_image_prompt = extras_module.build_image_to_image_prompt
        self.get_extra_body_params = extras_module.get_extra_body_params
        self.get_model_class_name = extras_module.get_model_class_name
        self.should_init_extra_args_for_non_diffusion_stages = (
            extras_module.should_init_extra_args_for_non_diffusion_stages
        )
        self.device_type = platforms_module.current_omni_platform.device_type
        parallel_config = data_module.DiffusionParallelConfig(
            ulysses_degree=1,
            ring_degree=1,
            cfg_parallel_size=1,
            tensor_parallel_size=1,
        )
        started = time.perf_counter()
        self.omni = omni_module.Omni(
            model=str(args.model),
            vae_use_tiling=True,
            cache_backend=getattr(args, "cache_backend", None),
            cache_config=getattr(args, "cache_config", None),
            parallel_config=parallel_config,
            enable_cpu_offload=False,
            enforce_eager=True,
        )
        self.model_load_seconds = time.perf_counter() - started
        self.model_class_name = self.get_model_class_name(self.omni)
        self.declared_extra_body_params = self.get_extra_body_params(self.model_class_name)

    def _sampling_params(
        self,
        *,
        method: str,
        extra_args: Mapping[str, Any],
        seed: int,
        height: int,
        width: int,
    ) -> list[Any]:
        generator = self.torch.Generator(device=self.device_type).manual_seed(int(seed))
        diffusion_params = self.OmniDiffusionSamplingParams(
            generator=generator,
            true_cfg_scale=TRUE_CFG_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_STEPS,
            num_outputs_per_prompt=1,
            resolution=max(height, width),
            height=height,
            width=width,
        )
        merged = dict(extra_args)
        merged["flux_acceleration_method"] = method
        merged.setdefault("negative_prompt", NEGATIVE_PROMPT)
        if self.declared_extra_body_params:
            self.apply_declared_extra_args(diffusion_params, self.declared_extra_body_params, merged)
        else:
            diffusion_params.extra_args.update({key: value for key, value in merged.items() if value is not None})
        defaults = list(self.omni.default_sampling_params_list or [])
        if not defaults:
            return [diffusion_params]
        sampling_params_list = [self.clone_sampling_params(params) for params in defaults]
        replaced = False
        initialize_non_diffusion = self.should_init_extra_args_for_non_diffusion_stages(self.model_class_name)
        for index, params in enumerate(sampling_params_list):
            if isinstance(params, self.OmniDiffusionSamplingParams):
                existing = dict(getattr(params, "extra_args", {}) or {})
                existing.update(diffusion_params.extra_args)
                diffusion_params.extra_args = existing
                sampling_params_list[index] = diffusion_params
                replaced = True
            elif initialize_non_diffusion and hasattr(params, "extra_args") and params.extra_args is None:
                params.extra_args = {}
        if not replaced and len(sampling_params_list) == 1:
            return [diffusion_params]
        return sampling_params_list

    def run(
        self,
        image: Any,
        instruction: str,
        *,
        method: str,
        extra_args: Mapping[str, Any],
        seed: int,
        height: int,
        width: int,
        output_path: Path | None,
    ) -> dict[str, Any]:
        prompt = self.build_image_to_image_prompt(
            model_class_name=self.model_class_name,
            prompt=instruction,
            negative_prompt=NEGATIVE_PROMPT,
            input_image=image,
            height=height,
            width=width,
        )
        sampling_params_list = self._sampling_params(
            method=method, extra_args=extra_args, seed=seed, height=height, width=width
        )
        if self.device_type == "cuda":
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        outputs = self.omni.generate(prompt, sampling_params_list=sampling_params_list, use_tqdm=False)
        if self.device_type == "cuda":
            self.torch.cuda.synchronize()
        wall = time.perf_counter() - started
        images = self.extract_images_from_outputs(outputs)
        if not images:
            raise DriverError("Omni returned no output images")
        output = images[0][0] if isinstance(images[0], list) else images[0]
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output.save(output_path)
        metadata = _extract_flux_metadata(outputs)
        memory: dict[str, Any] = {}
        if self.device_type == "cuda":
            memory = {
                "peak_allocated": int(self.torch.cuda.max_memory_allocated()),
                "peak_reserved": int(self.torch.cuda.max_memory_reserved()),
            }
        return {
            "request_wall_seconds": float(wall),
            "model_load_seconds": float(self.model_load_seconds),
            "output_image": str(output_path.resolve()) if output_path is not None else None,
            "resolved_width": int(getattr(output, "width", width)),
            "resolved_height": int(getattr(output, "height", height)),
            "runtime_metadata": metadata,
            "memory": (metadata or {}).get("memory") or {},
            "parent_process_memory_diagnostic": memory,
        }

    def close(self) -> None:
        omni = getattr(self, "omni", None)
        if omni is not None:
            omni.close()
            self.omni = None


def _extract_flux_metadata(outputs: Any) -> dict[str, Any] | None:
    values = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    for output in values:
        multimodal = getattr(output, "multimodal_output", None)
        if hasattr(multimodal, "to_dict"):
            multimodal = multimodal.to_dict()
        if isinstance(multimodal, dict):
            metadata = multimodal.get("metadata") or {}
            result = metadata.get("flux_sparseedit") if isinstance(metadata, dict) else None
            if isinstance(result, dict):
                return result
    return None


# ----------------------------------------------------------------------------
# Variant configuration.
# ----------------------------------------------------------------------------
def base_extra_args(variant: str, *, replay_trace: str = "", collect_trace: bool = False) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "flux_acceleration_method": "sparseedit",
        "sparseedit_selector_protocol": "paper",
        "sparseedit_paper_mode": "forward",
        "sparseedit_ratio": SPARSE_RATIO,
        "sparseedit_warmup_steps": WARMUP_STEPS,
        "sparseedit_refresh_interval": REFRESH_INTERVAL,
        "sparseedit_avd_enabled": False,
        "sparseedit_runtime_variant": variant,
        "sparseedit_return_metadata": True,
        "sparseedit_defer_diagnostics": True,
        "sparseedit_collect_trace": bool(collect_trace),
    }
    if replay_trace:
        extra["sparseedit_replay_trace"] = replay_trace
    return extra


def smoke_extra_args(variant: str, *, replay_trace: str, audit_dir: Path) -> dict[str, Any]:
    extra = base_extra_args(
        variant,
        replay_trace=replay_trace,
        collect_trace=variant == "baseline",
    )
    extra["sparseedit_defer_diagnostics"] = False
    extra["sparseedit_collect_audit"] = True
    extra["sparseedit_audit_dir"] = str(audit_dir)
    return extra


def _strip_metadata_for_manifest(metadata: Mapping[str, Any]) -> dict[str, Any]:
    stripped = json.loads(json.dumps(metadata, default=str))
    stripped.pop("fixed_mask_trace", None)
    for record in stripped.get("step_records", []):
        record.pop("active_mask_b64", None)
    return stripped


# ----------------------------------------------------------------------------
# Smoke stage.
# ----------------------------------------------------------------------------
def _procedural_image(path: Path, size: int) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import numpy as np
    from PIL import Image, ImageDraw

    axis = np.linspace(0, 255, size, dtype=np.uint8)
    grid = np.stack(np.meshgrid(axis, axis[::-1]), axis=-1)
    rgb = np.stack([grid[..., 0], grid[..., 1], np.full_like(grid[..., 0], 128)], axis=-1)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle([size // 8, size // 8, size // 3, size // 3], fill=(220, 40, 40))
    draw.ellipse([size // 2, size // 2, 5 * size // 6, 5 * size // 6], fill=(40, 80, 220))
    draw.line([0, size - 1, size - 1, 0], fill=(250, 250, 250), width=max(1, size // 64))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _audit_compare(
    left_dir: Path,
    right_dir: Path,
    *,
    tolerance: Mapping[str, float],
    expected_steps: int,
) -> dict[str, Any]:
    import torch

    worst = 0.0
    compared = 0
    for step in range(expected_steps):
        left_path = left_dir / f"step_{step:02d}.pt"
        right_path = right_dir / f"step_{step:02d}.pt"
        if not left_path.is_file() or not right_path.is_file():
            raise DriverError(f"audit step {step} is missing under {left_dir} or {right_dir}")
        left = torch.load(left_path, map_location="cpu")
        right = torch.load(right_path, map_location="cpu")
        for name in ("guided_velocity", "latents"):
            if name not in left or name not in right:
                raise DriverError(f"audit step {step} does not contain {name}")
            if left[name].shape != right[name].shape:
                raise DriverError(f"audit {name} shape mismatch at step {step}")
            difference = (left[name].to(torch.float32) - right[name].to(torch.float32)).abs().max().item()
            if not math.isfinite(difference) or difference > 0.02:
                raise DriverError(f"audit {name} max absolute error {difference} exceeds 0.02 at step {step}")
            worst = max(worst, float(difference))
            torch.testing.assert_close(
                left[name].to(torch.float32),
                right[name].to(torch.float32),
                rtol=float(tolerance["rtol"]),
                atol=float(tolerance["atol"]),
                msg=lambda message: f"step {step} {name} mismatch: {message}",
            )
            compared += 1
    return {"compared_tensors": compared, "max_abs_diff": worst, "tolerance": dict(tolerance)}


def _check_smoke_metadata(metadata: Mapping[str, Any], variant: str) -> dict[str, Any]:
    records = metadata.get("step_records")
    summary = metadata.get("summary")
    if not isinstance(records, list) or len(records) != NUM_STEPS:
        raise DriverError(f"{variant}: expected {NUM_STEPS} step records")
    if not isinstance(summary, Mapping):
        raise DriverError(f"{variant}: runtime summary is missing")
    sparse = [record for record in records if record.get("is_sparse_step")]
    if len(sparse) != 19:
        raise DriverError(f"{variant}: expected 19 sparse steps, got {len(sparse)}")
    phases = {record["step"]: record["phase"] for record in records}
    if phases.get(23) != "refresh_full":
        raise DriverError(f"{variant}: step 23 is not the refresh boundary: {phases.get(23)}")
    actions = {record["step"]: record["kv_action"] for record in records}
    if actions.get(7) != "store_full" or actions.get(23) != "store_full":
        raise DriverError(f"{variant}: store_full boundary is wrong: {actions}")
    digests = {record["selected_index_digest"] for record in sparse}
    if len(digests) < 2:
        raise DriverError(f"{variant}: active indices never change across sparse steps")
    if summary.get("caches_cleared") is not True:
        raise DriverError(f"{variant}: request caches were not cleared")
    if variant == "graph":
        graph = (metadata.get("run_config") or {}).get("graph") or {}
        if not graph.get("graph_active"):
            raise DriverError(f"graph variant did not capture: {graph}")
        if int(graph.get("total_replays", 0)) < 1:
            raise DriverError(f"graph variant never replayed: {graph}")
        for branch in graph.get("branches", {}).values():
            if not branch.get("graph_active") or int(branch.get("replay_count", 0)) < 1:
                raise DriverError(f"graph branch did not replay: {branch}")
            if not branch.get("captured_region"):
                raise DriverError(f"graph branch did not record its captured region: {branch}")
    return {
        "sparse_steps": len(sparse),
        "caches_cleared": summary.get("caches_cleared"),
        "distinct_index_digests": len(digests),
        "refresh_step": 23,
        "graph": (metadata.get("run_config") or {}).get("graph"),
    }


def run_smoke_worker(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    control = root / "control"
    write_json(control / "worker_start.json", {"stage": "smoke", "started_at": utc_now(), "pid": os.getpid()})
    variants = validate_variants(args.variants)
    cases = load_cases(args.cases)
    inputs_root = Path(args.remote_inputs_root)
    case_validation = validate_cases(cases, inputs_root, allow_missing=True)
    write_json(control / "case_validation.json", case_validation)
    smoke_root = root / "smoke"
    procedural = smoke_root / "input" / "procedural_768.png"
    _procedural_image(procedural, SMOKE_HEIGHT)
    instruction = "Add a small red square in the center while keeping the image otherwise unchanged."
    worker = NativeWorker(args)
    write_json(
        control / "model_load.json",
        {
            "model": str(args.model),
            "model_load_seconds": worker.model_load_seconds,
            "model_class_name": worker.model_class_name,
            "model_artifact_hashes": _model_artifact_hashes(Path(args.model)),
        },
    )
    results: dict[str, Any] = {}
    try:
        image = worker.pil_image.open(procedural).convert("RGB")
        try:
            trace_path = smoke_root / "masks" / "smoke_fixed_mask_trace.json"
            for variant in variants:
                if variant != "baseline" and not trace_path.is_file():
                    raise DriverError("baseline must run first in the smoke variant order")
                replay = str(trace_path) if variant != "baseline" else ""
                run_root = smoke_root / "runs" / variant
                extra = smoke_extra_args(variant, replay_trace=replay, audit_dir=run_root / "audit")
                manifest = worker.run(
                    image,
                    instruction,
                    method="sparseedit",
                    extra_args=extra,
                    seed=args.seed,
                    height=SMOKE_HEIGHT,
                    width=SMOKE_WIDTH,
                    output_path=run_root / "output.png",
                )
                metadata = manifest.get("runtime_metadata")
                if not isinstance(metadata, Mapping):
                    raise DriverError(f"{variant}: runtime metadata is missing")
                if metadata.get("runtime_variant") != variant:
                    raise DriverError(f"{variant}: runtime reported {metadata.get('runtime_variant')}")
                checks = _check_smoke_metadata(metadata, variant)
                if variant == "baseline":
                    trace = metadata.get("fixed_mask_trace")
                    if not isinstance(trace, Mapping):
                        raise DriverError("baseline did not record its fixed-mask trace")
                    write_json(trace_path, trace)
                    manifest["trace_sha256"] = metadata.get("fixed_mask_trace_sha256")
                else:
                    if not metadata.get("run_config", {}).get("replay", {}).get("enabled"):
                        raise DriverError(f"{variant}: replay trace was not applied")
                    manifest["trace_sha256"] = metadata["run_config"]["replay"]["sha256"]
                write_json(run_root / "manifest.json", {**manifest, "checks": checks, "metadata": _strip_metadata_for_manifest(metadata)})
                results[variant] = {"manifest": str(run_root / "manifest.json"), "checks": checks}
        finally:
            image.close()
    finally:
        worker.close()

    comparisons: dict[str, Any] = {}
    tolerance_pairs = {
        "inplace_vs_baseline": ("inplace", "baseline", EXACT_TOLERANCE),
        "transformed_vs_baseline": ("transformed", "baseline", TRANSFORMED_OUTPUT_TOLERANCE),
        "fused_vs_transformed": ("fused", "transformed", FUSED_OUTPUT_TOLERANCE),
        "graph_vs_transformed": ("graph", "transformed", EXACT_TOLERANCE),
    }
    for name, (left, right, tolerance) in tolerance_pairs.items():
        if left not in results or right not in results:
            continue
        comparisons[name] = _audit_compare(
            smoke_root / "runs" / left / "audit",
            smoke_root / "runs" / right / "audit",
            tolerance=tolerance,
            expected_steps=NUM_STEPS,
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "smoke",
        "status": "completed",
        "generated_at": utc_now(),
        "input_image": str(procedural),
        "input_sha256": sha256_file(procedural),
        "instruction": instruction,
        "variants": list(variants),
        "results": results,
        "numeric_comparisons": comparisons,
        "declared_tolerances": {
            # Copy-only paths (inplace raw cache, graph replay of the same
            # arithmetic) are gated bitwise.  Graph re-executes the identical
            # kernels on identical buffers, so any difference is a real defect;
            # the gate is not relaxed to accommodate capture.
            "inplace_vs_baseline": EXACT_TOLERANCE,
            "graph_vs_transformed": EXACT_TOLERANCE,
            "transformed_vs_baseline": TRANSFORMED_OUTPUT_TOLERANCE,
            "fused_vs_transformed": FUSED_OUTPUT_TOLERANCE,
        },
        "provenance": _provenance(args, cases=[]),
    }
    write_json(smoke_root / "smoke_report.json", report)
    write_json(root / "status.json", {"smoke": {"status": "completed", "finished_at": utc_now()}})
    print(json.dumps({"stage": "smoke", "status": "completed", "comparisons": comparisons}, indent=2))
    return 0


# ----------------------------------------------------------------------------
# Benchmark stage.
# ----------------------------------------------------------------------------
def benchmark_plan(
    cases: Sequence[Case],
    variants: Sequence[str],
    repetitions: int,
    *,
    include_dense: bool = False,
) -> list[dict[str, Any]]:
    if repetitions < 2:
        raise DriverError("the benchmark requires at least two measured repetitions")
    plan: list[dict[str, Any]] = []
    for case in cases:
        for repetition in range(repetitions):
            order = list(variants)
            shift = repetition % len(order)
            order = order[shift:] + order[:shift]
            if repetition == 0:
                order = ["baseline", *[variant for variant in order if variant != "baseline"]]
            for variant in order:
                plan.append(
                    {
                        "case_tag": case.case_tag,
                        "variant": variant,
                        "repetition": repetition,
                        "records_trace": repetition == 0 and variant == "baseline",
                    }
                )
            if include_dense:
                plan.append({"case_tag": case.case_tag, "variant": "dense", "repetition": repetition, "records_trace": False})
    return plan


def _run_manifest_path(root: Path, case: Case, variant: str, repetition: int) -> Path:
    return root / "runs" / case.case_tag / variant / f"rep{repetition}.json"


def _identity(args: argparse.Namespace, case: Case, variant: str, repetition: int, trace_sha: str | None) -> dict[str, Any]:
    input_path = resolve_input_path(case, Path(args.remote_inputs_root), fallback_path=case.image_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "case_tag": case.case_tag,
        "variant": variant,
        "repetition": repetition,
        "seed": int(args.seed),
        "num_inference_steps": NUM_STEPS,
        "height": BENCH_HEIGHT,
        "width": BENCH_WIDTH,
        "ratio": SPARSE_RATIO,
        "warmup_steps": WARMUP_STEPS,
        "refresh_interval": REFRESH_INTERVAL,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "guidance_scale": GUIDANCE_SCALE,
        "negative_prompt": NEGATIVE_PROMPT,
        "input_sha256": sha256_file(input_path),
        "prepared_sha256": case.prepared_sha256,
        "source_pixel_sha256": case.source_pixel_sha256,
        "model": str(args.model),
        "model_artifact_hashes": _model_artifact_hashes(Path(args.model)),
        "trace_sha256": trace_sha,
        "source_fingerprint": source_fingerprint(),
        "gpu_uuid": str(args.expected_gpu_uuid),
        "gpu_physical_index": int(args.gpu_physical_index),
    }


def _provenance(args: argparse.Namespace, *, cases: Sequence[Case]) -> dict[str, Any]:
    return {
        "git": _git_identity(),
        "model": str(args.model),
        "model_artifact_hashes": _model_artifact_hashes(Path(args.model)),
        "cases": str(args.cases),
        "cases_sha256": sha256_file(Path(args.cases)) if Path(args.cases).is_file() else None,
        "inputs_root": str(args.remote_inputs_root),
        "gpu_uuid": str(args.expected_gpu_uuid),
        "gpu_physical_index": int(args.gpu_physical_index),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "python": sys.executable,
        "source_fingerprint": source_fingerprint(),
        "host": platform.node(),
    }


def run_benchmark_worker(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    control = root / "control"
    write_json(control / "worker_start.json", {"stage": "benchmark", "started_at": utc_now(), "pid": os.getpid()})
    variants = validate_variants(args.variants)
    cases = load_cases(args.cases)
    case_validation = validate_cases(cases, Path(args.remote_inputs_root))
    write_json(control / "case_validation.json", case_validation)
    plan = benchmark_plan(cases, variants, args.repetitions, include_dense=bool(args.include_dense))
    scope = "full" if (len(cases) == 3 and args.repetitions >= 2 and not args.include_dense) else "subset"
    provenance = _provenance(args, cases=cases)
    write_json(
        control / "benchmark_plan.json",
        {"schema_version": SCHEMA_VERSION, "scope": scope, "request_count": len(plan), "plan": plan,
         "provenance": provenance},
    )
    worker = NativeWorker(args)
    write_json(
        control / "model_load.json",
        {
            "model": str(args.model),
            "model_load_seconds": worker.model_load_seconds,
            "model_class_name": worker.model_class_name,
        },
    )
    traces: dict[str, str] = {}
    for case in cases:
        trace_file = root / "masks" / f"{case.case_tag}.json"
        if trace_file.is_file():
            validation = validate_trace_file(trace_file)
            traces[case.case_tag] = str(validation["sha256"])
    status: dict[str, Any] = {"stage": "benchmark", "scope": scope, "started_at": utc_now(), "runs": {}}
    failures = 0
    try:
        # Unmeasured model warmup on the first frozen request.
        first = cases[0]
        with worker.pil_image.open(resolve_input_path(first, Path(args.remote_inputs_root), fallback_path=first.image_path)) as handle:
            warmup_image = handle.convert("RGB")
        try:
            for _ in range(1):
                worker.run(
                    warmup_image,
                    first.instruction,
                    method="sparseedit",
                    extra_args=base_extra_args("baseline"),
                    seed=args.seed,
                    height=BENCH_HEIGHT,
                    width=BENCH_WIDTH,
                    output_path=None,
                )
        finally:
            warmup_image.close()
        for entry in plan:
            case = next(item for item in cases if item.case_tag == entry["case_tag"])
            variant = str(entry["variant"])
            repetition = int(entry["repetition"])
            manifest_path = _run_manifest_path(root, case, variant, repetition)
            if variant == "dense":
                trace_sha = None
                extra = {"sparseedit_return_metadata": True}
                method = "dense"
            else:
                method = "sparseedit"
                trace_sha = traces.get(case.case_tag)
                if variant != "baseline" or repetition > 0:
                    if trace_sha is None:
                        raise DriverError(f"trace for {case.case_tag} is unavailable before {variant} rep{repetition}")
                    trace_path = str(root / "masks" / f"{case.case_tag}.json")
                else:
                    trace_path = ""
                extra = base_extra_args(
                    variant,
                    replay_trace=trace_path,
                    collect_trace=bool(entry["records_trace"]),
                )
            identity = _identity(args, case, variant, repetition, trace_sha)
            if args.resume and manifest_path.is_file():
                existing = read_json(manifest_path)
                if (
                    existing.get("status") == "completed"
                    and existing.get("identity_sha256") == sha256_payload(identity)
                ):
                    status["runs"][f"{case.case_tag}/{variant}/rep{repetition}"] = "skipped_completed"
                    continue
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "identity": identity,
                "identity_sha256": sha256_payload(identity),
                "scope": scope,
                "started_at": utc_now(),
            }
            try:
                with worker.pil_image.open(
                    resolve_input_path(case, Path(args.remote_inputs_root), fallback_path=case.image_path)
                ) as handle:
                    image = handle.convert("RGB")
                try:
                    manifest = worker.run(
                        image,
                        case.instruction,
                        method=method,
                        extra_args=extra,
                        seed=args.seed,
                        height=BENCH_HEIGHT,
                        width=BENCH_WIDTH,
                        output_path=manifest_path.parent / f"output_rep{repetition}.png",
                    )
                finally:
                    image.close()
                metadata = manifest.get("runtime_metadata") or {}
                if variant != "dense":
                    checks = _check_smoke_metadata(metadata, variant) if variant == "graph" else {
                        "step_count": len(metadata.get("step_records", [])),
                        "caches_cleared": (metadata.get("summary") or {}).get("caches_cleared"),
                        "graph_active": (metadata.get("summary") or {}).get("graph_active"),
                    }
                    if checks.get("caches_cleared") is not True:
                        raise DriverError(f"{variant}: caches were not cleared")
                    if variant == "baseline" and entry["records_trace"]:
                        trace = metadata.get("fixed_mask_trace")
                        if not isinstance(trace, Mapping):
                            raise DriverError("baseline benchmark run did not record its mask trace")
                        trace_path = root / "masks" / f"{case.case_tag}.json"
                        write_json(trace_path, trace)
                        traces[case.case_tag] = str(metadata.get("fixed_mask_trace_sha256"))
                        record["trace_path"] = str(trace_path)
                        record["trace_sha256"] = traces[case.case_tag]
                        record["identity"]["trace_sha256"] = traces[case.case_tag]
                        record["identity_sha256"] = sha256_payload(record["identity"])
                    elif trace_sha:
                        applied = (metadata.get("run_config") or {}).get("replay", {}).get("sha256")
                        if applied != trace_sha:
                            raise DriverError(f"{variant}: replay sha {applied} != {trace_sha}")
                record.update(
                    {
                        "status": "completed",
                        "finished_at": utc_now(),
                        "request": manifest,
                        "metadata": _strip_metadata_for_manifest(metadata),
                        "checks": checks if variant != "dense" else {},
                    }
                )
                status["runs"][f"{case.case_tag}/{variant}/rep{repetition}"] = "completed"
            except Exception as error:  # noqa: BLE001 - record and continue
                failures += 1
                record.update(
                    {
                        "status": "failed",
                        "finished_at": utc_now(),
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
                status["runs"][f"{case.case_tag}/{variant}/rep{repetition}"] = "failed"
            write_json(manifest_path, record)
            write_json(control / "state.json", {**status, "updated_at": utc_now()})
    finally:
        worker.close()
    status.update({"finished_at": utc_now(), "failures": failures})
    status["status"] = "completed" if failures == 0 else "partial"
    write_json(control / "state.json", status)
    write_json(root / "status.json", {"benchmark": status})
    print(json.dumps({"stage": "benchmark", "scope": scope, "requests": len(plan), "failures": failures}, indent=2))
    return 0 if failures == 0 else 1


# ----------------------------------------------------------------------------
# CPU analysis.
# ----------------------------------------------------------------------------
def _load_rgb(path: Path) -> Any:
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def compute_psnr(left: Any, right: Any) -> float:
    import numpy as np

    difference = left - right
    mse = float(np.mean(difference * difference))
    if mse == 0.0:
        return math.inf
    return float(10.0 * math.log10(1.0 / mse))


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p90": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": _percentile(values, 0.5),
        "mean": sum(values) / len(values),
        "p90": _percentile(values, 0.9),
        "max": max(values),
    }


def collect_runs(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("runs/*/*/rep*.json")):
        try:
            row = read_json(path)
        except DriverError:
            continue
        if not isinstance(row, Mapping):
            continue
        row = dict(row)
        row["_path"] = str(path)
        row["case_tag"] = (row.get("identity") or {}).get("case_tag")
        row["variant"] = (row.get("identity") or {}).get("variant")
        row["repetition"] = (row.get("identity") or {}).get("repetition")
        rows.append(row)
    return rows


def analyze_root(root: Path, *, tolerance: Mapping[str, float] = TRANSFORMED_OUTPUT_TOLERANCE) -> dict[str, Any]:
    root = root.resolve()
    runs = collect_runs(root)
    completed = [row for row in runs if row.get("status") == "completed"]
    failed = [row for row in runs if row.get("status") == "failed"]
    grouped: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in completed:
        key = (str(row["case_tag"]), str(row["variant"]), int(row["repetition"]))
        grouped[key] = row
    pairs: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    for (case_tag, variant, repetition), row in sorted(grouped.items()):
        if variant in {"baseline", "dense"}:
            continue
        baseline = grouped.get((case_tag, "baseline", repetition))
        if baseline is None:
            regressions.append({"case_tag": case_tag, "variant": variant, "repetition": repetition,
                                "reason": "missing_baseline"})
            continue
        candidate_path = Path(str(row.get("request", {}).get("output_image") or ""))
        baseline_path = Path(str(baseline.get("request", {}).get("output_image") or ""))
        pair: dict[str, Any] = {
            "case_tag": case_tag,
            "variant": variant,
            "repetition": repetition,
            "candidate_status": row.get("status"),
            "baseline_status": baseline.get("status"),
            "trace_sha_match": (
                (row.get("identity") or {}).get("trace_sha256")
                == (baseline.get("identity") or {}).get("trace_sha256")
            ),
            "runtime_baseline_equivalence": None,
        }
        if candidate_path.is_file() and baseline_path.is_file():
            candidate_image = _load_rgb(candidate_path)
            baseline_image = _load_rgb(baseline_path)
            if candidate_image.shape != baseline_image.shape:
                pair["error"] = "image_shape_mismatch"
                regressions.append({**pair, "reason": "image_shape_mismatch"})
            else:
                psnr = compute_psnr(candidate_image, baseline_image)
                max_diff = float(abs(candidate_image - baseline_image).max())
                pair["image_psnr"] = psnr if math.isfinite(psnr) else None
                pair["image_pixels_equal"] = max_diff == 0.0
                pair["image_max_abs_diff"] = max_diff
                if max_diff > float(tolerance["atol"]):
                    regressions.append({**pair, "reason": "image_tolerance_exceeded"})
                pair["runtime_baseline_equivalence"] = "equivalent" if max_diff <= float(tolerance["atol"]) else "regressed"
        else:
            pair["error"] = "missing_output_image"
            regressions.append({**pair, "reason": "missing_output_image"})
        metadata = row.get("metadata") or {}
        timing = metadata.get("timing") or {}
        pair["timing"] = {
            "denoise_wall_seconds": timing.get("denoise_wall_seconds"),
            "denoise_device_seconds": timing.get("denoise_device_seconds"),
            # Raw inclusive totals and the separately-measured audit capture
            # cost, so a reader can recompute the subtraction above.
            "denoise_wall_seconds_including_audit_capture": timing.get(
                "denoise_wall_seconds_including_audit_capture"
            ),
            "denoise_device_seconds_including_audit_capture": timing.get(
                "denoise_device_seconds_including_audit_capture"
            ),
            "audit_capture_device_seconds": timing.get("audit_capture_device_seconds"),
            "audit_capture_wall_seconds": timing.get("audit_capture_wall_seconds"),
            "audit_write_seconds": timing.get("audit_write_seconds"),
            "graph_capture_seconds": (
                ((metadata.get("run_config") or {}).get("graph") or {}).get("branches", {})
                .get("positive", {})
                .get("capture_seconds")
            ),
        }
        pair["memory"] = metadata.get("memory") or row.get("request", {}).get("memory") or {}
        pair["graph_replay_count"] = (
            ((metadata.get("run_config") or {}).get("graph") or {}).get("total_replays")
        )
        pairs.append(pair)
    # Dense-relative quality is reported separately and only when dense runs exist.
    dense_rows = [row for row in completed if row.get("variant") == "dense"]
    dense_relative: dict[str, Any] = {"available": bool(dense_rows), "pairs": []}
    if dense_rows:
        dense_by_case = {(row["case_tag"], row["repetition"]): row for row in dense_rows}
        for row in completed:
            if row.get("variant") == "dense":
                continue
            dense = dense_by_case.get((row["case_tag"], row["repetition"]))
            if dense is None:
                continue
            dense_path = Path(str(dense.get("request", {}).get("output_image") or ""))
            path = Path(str(row.get("request", {}).get("output_image") or ""))
            if not dense_path.is_file() or not path.is_file():
                continue
            dense_image = _load_rgb(dense_path)
            image = _load_rgb(path)
            if dense_image.shape != image.shape:
                continue
            dense_relative["pairs"].append(
                {
                    "case_tag": row["case_tag"],
                    "variant": row["variant"],
                    "repetition": row["repetition"],
                    "psnr_vs_dense": (
                        value if math.isfinite(value := compute_psnr(image, dense_image)) else None
                    ),
                }
            )
    timing_by_variant: dict[str, Any] = {}
    for variant in sorted({str(row["variant"]) for row in completed}):
        rows = [row for row in completed if row["variant"] == variant]
        timing_by_variant[variant] = {
            "request_wall_seconds": _summary(
                [float(row["request"]["request_wall_seconds"]) for row in rows if row.get("request", {}).get("request_wall_seconds") is not None]
            ),
            "denoise_wall_seconds": _summary(
                [
                    float(((row.get("metadata") or {}).get("timing") or {}).get("denoise_wall_seconds"))
                    for row in rows
                    if ((row.get("metadata") or {}).get("timing") or {}).get("denoise_wall_seconds") is not None
                ]
            ),
            "denoise_device_seconds": _summary(
                [
                    float(((row.get("metadata") or {}).get("timing") or {}).get("denoise_device_seconds"))
                    for row in rows
                    if ((row.get("metadata") or {}).get("timing") or {}).get("denoise_device_seconds") is not None
                ]
            ),
            "peak_allocated": _summary(
                [
                    float(((row.get("metadata") or {}).get("memory") or {}).get("max_memory_allocated"))
                    for row in rows
                    if ((row.get("metadata") or {}).get("memory") or {}).get("max_memory_allocated") is not None
                ]
            ),
            "peak_reserved": _summary(
                [
                    float(((row.get("metadata") or {}).get("memory") or {}).get("max_memory_reserved"))
                    for row in rows
                    if ((row.get("metadata") or {}).get("memory") or {}).get("max_memory_reserved") is not None
                ]
            ),
        }
    expected = len({(row["case_tag"], row["variant"], row["repetition"]) for row in runs})
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "stage": "analyze",
        "generated_at": utc_now(),
        "root": str(root),
        "status_coverage": {
            "discovered": len(runs),
            "completed": len(completed),
            "failed": len(failed),
            "failed_runs": [
                {
                    "path": row["_path"],
                    "error": row.get("error"),
                }
                for row in failed
            ],
            "coverage": (len(completed) / expected) if expected else 0.0,
        },
        "pairs": pairs,
        "regressions": regressions,
        "timing_by_variant": timing_by_variant,
        "runtime_baseline_equivalence": {
            "reference": "baseline",
            "tolerance": dict(tolerance),
            "pair_count": len(pairs),
            "regression_count": len(regressions),
        },
        "dense_relative_quality": dense_relative,
        "graph_replay": {
            "runs_with_replays": sum(1 for pair in pairs if (pair.get("graph_replay_count") or 0) > 0),
            "runs_expected": len([pair for pair in pairs if pair["variant"] == "graph"]),
        },
    }
    write_json(root / "analysis" / "analysis.json", analysis)
    csv_path = root / "analysis" / "pairs.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_tag",
        "variant",
        "repetition",
        "image_psnr",
        "image_max_abs_diff",
        "trace_sha_match",
        "runtime_baseline_equivalence",
        "graph_replay_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair)
    return analysis


# ----------------------------------------------------------------------------
# Argument parsing and entry points.
# ----------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        choices=["validate", "cpu-tests", "smoke", "benchmark", "analyze", "smoke-worker", "benchmark-worker"],
        default="validate",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--python", type=Path, default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--inputs-root", type=Path, default=DEFAULT_INPUTS_ROOT)
    parser.add_argument("--remote-inputs-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=SOURCE_ROOT / "outputs" / "flux_specialized_runtime")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gpu-physical-index", type=int, default=-1)
    parser.add_argument("--expected-gpu-uuid", default="")
    parser.add_argument("--node", default=None)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--audit", type=int, default=1)
    parser.add_argument("--include-dense", type=int, default=0)
    parser.add_argument("--resume", type=int, default=1)
    args = parser.parse_args(argv)
    args.variants = tuple(part.strip() for part in str(args.variants).split(",") if part.strip())
    args.remote_inputs_root = Path(args.remote_inputs_root or args.inputs_root)
    args.audit = bool(args.audit)
    args.include_dense = bool(args.include_dense)
    args.resume = bool(args.resume)
    return args


def stage_validate(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    variants = validate_variants(args.variants)
    case_validation = validate_cases(cases, Path(args.inputs_root), allow_missing=not Path(args.inputs_root).is_dir())
    matrix = benchmark_plan(cases, variants, max(2, args.repetitions), include_dense=args.include_dense)
    hardware_policy = {
        "blocked": {node: sorted(indexes) for node, indexes in DEFAULT_BLOCKED_GPUS.items()},
        "node": _node_name(args.node),
        "examples": {
            "node3_gpu7_blocked": policy_blocks_gpu("node3", 7),
            "node3_gpu3_blocked": policy_blocks_gpu("node3", 3),
            "node1_gpu4_blocked": policy_blocks_gpu("node1", 4),
        },
    }
    trace_validation = None
    if args.trace is not None:
        trace_validation = validate_trace_file(args.trace, cases=cases)
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "validate",
        "status": "ok",
        "cases": case_validation,
        "variants": list(variants),
        "benchmark_request_count": len(matrix),
        "benchmark_scope": "full" if len(matrix) == 30 else "subset",
        "hardware_policy": hardware_policy,
        "trace": trace_validation,
        "source_fingerprint": source_fingerprint(),
        "git": _git_identity(),
    }
    write_json(args.output_root / "validate.json", result)
    print(json.dumps(result, indent=2))
    return 0


def stage_analyze(args: argparse.Namespace) -> int:
    analysis = analyze_root(args.output_root)
    print(json.dumps(
        {
            "stage": "analyze",
            "completed": analysis["status_coverage"]["completed"],
            "failed": analysis["status_coverage"]["failed"],
            "pairs": len(analysis["pairs"]),
            "regressions": len(analysis["regressions"]),
            "dense_relative_available": analysis["dense_relative_quality"]["available"],
        },
        indent=2,
    ))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.stage in {"validate", "cpu-tests"}:
            return stage_validate(args)
        if args.stage == "analyze":
            return stage_analyze(args)
        if args.stage == "smoke":
            return _launch_child(args, "smoke-worker")
        if args.stage == "benchmark":
            return _launch_child(args, "benchmark-worker")
        if args.stage == "smoke-worker":
            return run_smoke_worker(args)
        if args.stage == "benchmark-worker":
            return run_benchmark_worker(args)
        raise DriverError(f"unknown stage: {args.stage}")
    except (DriverError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
