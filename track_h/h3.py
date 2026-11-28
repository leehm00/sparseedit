#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect the supported H3 native PNG and synchronized timing matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from math import isfinite
import os
from pathlib import Path
import re
import traceback

from h3_runtime import flux, longcat, qwen
from h3_runtime.contracts import FIXED20, file_sha, groups, json_sha, panels, read, validate_cohorts, write

ROOT = Path(__file__).resolve().parent
UUID_PATTERN = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
BOUNDARY = (
    "Native synchronized inference through decoded PIL response. Model loading, input file preparation, "
    "warmup, output PNG writing and offline scoring are excluded."
)


def now():
    return datetime.now(timezone.utc).isoformat()


class Run:
    def __init__(self, config_path, out):
        self.out = out
        self.receipt = {
            "schema_version": 1, "phase": "H3", "status": "FAIL", "execution_status": "FAIL",
            "batch_size": 1, "strict_serial": True, "kv_offload": False, "offline_metrics_required": True,
            "request_wall_scope": BOUNDARY, "config_sha256": file_sha(config_path), "started_at": now(),
            "requests": [], "panels": panels(), "failures": [],
        }
        self.observations = []

    def measured(self, panel, case, arm, repetition, timing, image_path, trace_path, metadata):
        from PIL import Image

        if not isinstance(timing, (int, float)) or not isfinite(timing) or timing <= 0:
            raise ValueError("native synchronized wall timing is missing or invalid")
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.size != (panel["resolution"], panel["resolution"]):
                raise ValueError("H3 native output PNG or dimensions differ")
            image.verify()
        model = panel["model"]
        row = {
            "status": "PASS", "model": model, "resolution": panel["resolution"], "arm": arm,
            "coverage": "supplemental" if arm in panel["supplemental"] else "planned",
            "repetition": repetition, "case_tag": case["case_tag"], "seed": 110,
            "source_id": case["source_pixel_sha256"], "input_sha256": case["input_sha256"],
            "instruction": case["instruction"], "instruction_sha256": json_sha(case["instruction"]),
            "request_wall_s": timing, "gpu_uuid": self.receipt["gpu_uuid"],
            "model_sha256": self.receipt["model_sha256"][model],
            "output_path": image_path.relative_to(self.out).as_posix(), "output_sha256": file_sha(image_path),
            "action_trace_path": trace_path.relative_to(self.out).as_posix(), "action_trace_sha256": file_sha(trace_path),
            "metadata": metadata,
        }
        manifest = image_path.with_name("manifest.json")
        write(manifest, row)
        self.receipt["requests"].append({
            **row, "manifest_path": manifest.relative_to(self.out).as_posix(), "manifest_sha256": file_sha(manifest),
        })
        print(f"H3 {model}/{panel['resolution']} {arm} rep{repetition} {case['case_tag']} {timing:.3f}s", flush=True)
        return row

    def directory(self, panel, arm, case, repetition):
        if Path(case["case_tag"]).name != case["case_tag"]:
            raise ValueError("unsafe case tag")
        target = (
            self.out / "requests" / panel["model"] / str(panel["resolution"]) / arm
            / f"rep{repetition}" / case["case_tag"]
        )
        target.mkdir(parents=True)
        return target

    def run_flux(self, config, panels, cases):
        from h2_runtime.flux_native import RUNTIME_SOURCE, run_native

        with flux.overlay_source(RUNTIME_SOURCE, self.out) as overlay:
            cap = self.receipt["source_sha256"]
            for path in sorted(p for p in overlay.rglob("*.py") if not p.is_symlink()):
                cap["effective_flux_comparators/" + path.relative_to(overlay).as_posix()] = file_sha(path)
            for panel in panels:
                side = panel["resolution"]
                for arm, schedule in groups(panel, cases, config["run_mode"]):
                    with flux.arm_worker(Path(config["models"]["flux"]), config["gpu_uuid"], arm, side, overlay) as args:
                        worker, method, extras = args
                        first = cases[0]
                        warmup = run_native(worker, first["input_path"], first["instruction"], side, method, extras)
                        flux.action_evidence(arm, warmup["runtime_metadata"])
                        for repetition, case in schedule:
                            directory = self.directory(panel, arm, case, repetition)
                            png, trace = directory / "output.png", directory / "actions.json"
                            result = run_native(worker, case["input_path"], case["instruction"], side, method, extras, png)
                            md = result["runtime_metadata"]
                            actions = flux.action_evidence(arm, md)
                            write(trace, actions)
                            details = {
                                "method": method, "requested_options": extras,
                                "prepared_input_pixel_sha256": result["prepared_input_pixel_sha256"],
                                "gpu_before": result["gpu_before"], "gpu_after": result["gpu_after"],
                                "model_load_seconds": result["model_load_seconds"],
                                "denoise_wall_s": md["pipeline_timing"]["denoise_wall_seconds"],
                            }
                            if arm.startswith("sparseedit_"):
                                details.update(route=md["track_a_warmup_route"], graph=md["run_config"]["graph"])
                            elif arm == "dense_graph":
                                details["dense_graph"] = md["dense_graph"]
                            elif arm in flux.PORTS:
                                details["implementation"] = md["cfg_port_identity"]
                            row = self.measured(panel, case, arm, repetition, result["request_wall_seconds"], png, trace, details)
                            if config["run_mode"] == "formal":
                                observation = flux.latency_observation(arm, row, md, actions)
                                if observation is not None:
                                    self.observations.append(observation)

    def run_longcat(self, config, panel, cases, cohort):
        from h2_runtime.flux_native import verify_gpu
        import torch

        self.receipt["longcat_environment"] = longcat.verify_environment(cohort)
        with longcat.open_model(torch, Path(config["models"]["longcat"]), cohort) as (pipeline, runtime):
            self.receipt["longcat_runtime"] = runtime
            longcat.request(torch, pipeline, cases[0], "fresh_dense")
            for arm, schedule in groups(panel, cases, config["run_mode"]):
                for repetition, case in schedule:
                    directory = self.directory(panel, arm, case, repetition)
                    png, trace = directory / "output.png", directory / "actions.json"
                    before = verify_gpu(config["gpu_uuid"], worker_active=True)
                    image, wall, actions, metadata = longcat.request(torch, pipeline, case, arm)
                    after = verify_gpu(config["gpu_uuid"], worker_active=True)
                    image.save(png, format="PNG")
                    image.close()
                    write(trace, actions)
                    self.measured(panel, case, arm, repetition, wall, png, trace,
                                  {**metadata, "gpu_before": before, "gpu_after": after})
            del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    def run_qwen(self, config, panel, cases, identity):
        from h2_runtime.flux_native import verify_gpu
        from track_h_qwen.native import load_pipeline, native_request
        from track_h_qwen.contracts import PROTOCOL
        import torch

        prior = Path(config["prior_results"]) if config.get("prior_results") else None
        pipeline, runtime = load_pipeline(torch, Path(config["models"]["qwen"]))
        self.receipt["qwen_runtime"] = runtime
        try:
            warmup, _, _, _ = native_request(torch, pipeline, cases[0], mode="dense")
            warmup.close()
            for arm, schedule in groups(panel, cases, config["run_mode"]):
                sparse = arm == FIXED20
                for repetition, case in schedule:
                    directory = self.directory(panel, arm, case, repetition)
                    png, trace = directory / "output.png", directory / "actions.json"
                    reused = None if sparse else qwen.prior_dense(
                        prior, case, repetition, config["run_mode"], config["gpu_uuid"], identity["runtime_file_sha256"],
                    )
                    if reused is not None:
                        qwen.copy_prior(reused, directory)
                        self.measured(panel, case, arm, repetition, reused["request_wall_s"], png, trace,
                                      {**reused["native_metadata"], "origin": "H1"})
                        continue
                    before = verify_gpu(config["gpu_uuid"], worker_active=True)
                    image, timing, executed, _ = native_request(
                        torch, pipeline, case, mode="direct" if sparse else "dense", budget=20 if sparse else None,
                        identity={
                            "case_id": case["case_id"], "input_sha256": case["input_sha256"],
                            "source_pixel_sha256": case["source_pixel_sha256"], "prompt_sha256": json_sha(case["prompt"]),
                            "model_sha256": identity["runtime_file_sha256"], "protocol_sha256": json_sha(PROTOCOL),
                            "budget_percent": 20 if sparse else None,
                        },
                        source_hashes=self.receipt["source_sha256"],
                    )
                    after = verify_gpu(config["gpu_uuid"], worker_active=True)
                    image.save(png, format="PNG")
                    image.close()
                    if sparse:
                        executed.save(trace)
                    else:
                        write(trace, flux.dense_actions())
                    self.measured(panel, case, arm, repetition, timing["request_wall_seconds"], png, trace,
                                  {**timing, "origin": "H3", "gpu_before": before, "gpu_after": after})
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()

    def execute(self, config):
        from h2_runtime.flux_native import RUNTIME_SOURCE, load_bundle, verify_gpu, verify_model
        from track_h_qwen.contracts import PROTOCOL, inventory, model_identity

        mode, uuid = config["run_mode"], config["gpu_uuid"]
        if mode not in ("cal3", "formal"):
            raise ValueError("H3 requires cal3 or formal run_mode")
        if not UUID_PATTERN.fullmatch(uuid) or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
            raise ValueError("H3 requires exactly one inherited visible NVIDIA GPU UUID")
        self.receipt.update(run_mode=mode, gpu_uuid=uuid, gpu=verify_gpu(uuid))
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        assets = Path(config["assets"]).resolve()
        models = config["models"]
        flux_assets = assets / "flux"
        flux_bundle = load_bundle(flux_assets)
        flux_cases = [
            {**case, "input_path": flux_assets / "inputs" / case["input_filename"], "input_sha256": case["prepared_sha256"]}
            for case in flux_bundle["cases"]["test" if mode == "formal" else "calibration"]
        ]
        longcat_bundle, longcat_cases = longcat.frozen_cases(assets / "longcat", mode)
        qwen_cases = [
            {**case, "case_tag": case["case_id"], "instruction": case["prompt"]}
            for case in inventory(mode, assets / "h1")["primary"]
        ]
        cases = {"flux": flux_cases, "longcat": longcat_cases, "qwen": qwen_cases}
        validate_cohorts(mode, cases)
        qwen_identity = model_identity(Path(models["qwen"]), assets / "h1")
        self.receipt["model_bindings"] = {
            "flux": verify_model(Path(models["flux"]), flux_bundle),
            "longcat": longcat.verify_model(Path(models["longcat"]), longcat_bundle),
            "qwen": qwen_identity,
        }
        self.receipt["model_sha256"] = {
            model: json_sha(value) if model != "qwen" else value["runtime_file_sha256"]
            for model, value in self.receipt["model_bindings"].items()
        }
        self.receipt["protocols"] = {
            "flux": {"steps": 28, "seed": 110, "cfg": 4.0, "guidance_scale": 1.0, "negative_prompt": " "},
            "longcat": longcat_bundle["protocol"], "qwen": PROTOCOL,
        }
        stems = [ROOT / "h3.py", ROOT / "h1.py", *(ROOT / "h3_runtime").glob("*.py"),
                 *(ROOT / "track_h_qwen").rglob("*.py"), *(ROOT / "h2_runtime" / "controls").glob("*.py"),
                 ROOT / "h2_runtime" / "flux_native.py", *RUNTIME_SOURCE.rglob("*.py")]
        self.receipt["source_sha256"] = {path.relative_to(ROOT).as_posix(): file_sha(path) for path in sorted(stems)}
        self.receipt["input_sha256"] = {
            name: {case["case_tag"]: case["input_sha256"] for case in values} for name, values in cases.items()
        }
        selected = self.receipt["panels"]
        for entry in selected:
            entry["scheduled_requests"] = sum(len(rows) for _, rows in groups(entry, cases[entry["model"]], mode))
        self.receipt["coverage_gaps"] = [
            {"model": panel["model"], "resolution": panel["resolution"], "arm": arm, "reason": reason}
            for panel in selected for arm, reason in panel["gaps"].items()
        ]
        self.run_flux(config, selected[:2], flux_cases)
        self.run_longcat(config, selected[2], longcat_cases, longcat_bundle)
        self.run_qwen(config, selected[3], qwen_cases, qwen_identity)
        expected = sum(panel["scheduled_requests"] for panel in selected)
        observed = self.receipt["requests"]
        if len(observed) != expected or len({
            (row["model"], row["resolution"], row["arm"], row["repetition"], row["case_tag"]) for row in observed
        }) != expected:
            raise ValueError("executed H3 request matrix differs from scheduled cells")
        self.receipt.update(status="PARTIAL" if self.receipt["coverage_gaps"] else "PASS",
                            execution_status="PASS", measured_requests=len(observed))

    def finalize(self):
        if self.observations:
            path = self.out / "latency_observations.jsonl"
            with path.open("x", encoding="utf-8") as stream:
                for observation in self.observations:
                    stream.write(json.dumps(observation, sort_keys=True, allow_nan=False) + "\n")
            self.receipt["latency_observations"] = {"path": path.name, "sha256": file_sha(path), "count": len(self.observations)}
        self.receipt["finished_at"] = now()
        write(self.out / "completion.json", self.receipt)


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    execution = Run(config, output)
    try:
        execution.execute(read(config))
    except Exception as error:
        execution.receipt["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(execution.receipt["error"]["traceback"], flush=True)
    finally:
        execution.finalize()
    return execution.receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.config, args.output)
    return 0 if result["execution_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
