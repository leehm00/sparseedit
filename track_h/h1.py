#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the resident Qwen-Image-Edit measurement matrix for Track H1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from track_h_qwen.contracts import (
    GPU_UUID,
    PROTOCOL,
    ROOT,
    SessionCache,
    digest_json,
    frozen_json,
    inventory,
    latency_observation,
    model_identity,
    require,
    sha_file,
    timing_summary,
    validate_session_pair,
    verify_artifact,
    write_json,
)
from track_h_qwen.native import admission_comparison, load_pipeline, native_request, resident_placement, select_gpu
from track_h_qwen.schedule import CALIBRATION_BUDGETS, MEASUREMENT_ARMS, calibration_jobs, heldout_jobs


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_identity():
    sources = [Path(__file__).resolve(), *(ROOT / "track_h_qwen").rglob("*.py")]
    return {path.relative_to(ROOT).as_posix(): sha_file(path) for path in sorted(sources)}


class Run:
    def __init__(self, config, output):
        self.output = output
        self.config = config
        self.receipt = {
            "schema_version": 1,
            "phase": "H1",
            "status": "FAILED",
            "execution_status": "FAILED",
            "started_at": utc_now(),
            "protocol": PROTOCOL,
            "requests": [],
            "numerical_admission": [],
            "coverage_gaps": [
                {"arm": "Full-JointCFG", "reason": "within_request_B2_port_unavailable"},
                {"arm": "online_K+B", "reason": "qwen_variable_capacity_budget_controller_unavailable"},
            ],
        }

    def request(self, torch, pipeline, case, arm, *, mode, budget=None, trace=None, producer=None,
                graph=False, admission=False, cache=None, session_index=None, prompt=None, repetition=None):
        rows = self.receipt["requests"]
        index = len(rows)
        relative = Path("requests") / f"{index:04d}_{case['case_id']}_{arm}"
        directory = self.output / relative
        directory.mkdir(parents=True, exist_ok=False)
        session = session_index is not None
        if cache is not None:
            cache.begin((case["source_pixel_sha256"], PROTOCOL["height"], PROTOCOL["width"], PROTOCOL["dtype"]))
        timing_role = "measurement"
        if arm == "warmup":
            timing_role = "warmup"
        elif admission:
            timing_role = "admission"
        elif trace is None and budget is not None:
            timing_role = "trajectory_producer"
        details = {
            "case_id": case["case_id"],
            "split": case["split"],
            "input_filename": Path(case["image"]).name,
            "input_sha256": case["input_sha256"],
            "source_pixel_sha256": case["source_pixel_sha256"],
            "prompt_sha256": digest_json(case["prompt"] if prompt is None else prompt),
            "model_sha256": self.receipt["model"]["runtime_file_sha256"],
            "protocol_sha256": digest_json(PROTOCOL),
            "budget_percent": budget,
            "arm": arm,
            "repetition": repetition,
            "model": "qwen",
            "resolution": PROTOCOL["height"],
            "warmup": arm == "warmup",
            "session_instruction_index": session_index,
            "timing_role": timing_role,
        }
        identity = {key: details[key] for key in ("case_id", "input_sha256", "source_pixel_sha256", "prompt_sha256",
                                                 "model_sha256", "protocol_sha256", "budget_percent")}
        if trace is not None:
            require(trace.identity == identity, "fixed trajectory identity differs")
            require(producer is not None, "fixed trajectory lacks source receipt")
            verify_artifact(self.output, producer["action_trace_path"], producer["action_trace_sha256"])
            details["fixed_trajectory_source"] = {
                "manifest_path": producer["manifest_path"], "manifest_sha256": producer["manifest_sha256"],
                "action_trace_path": producer["action_trace_path"], "action_trace_sha256": producer["action_trace_sha256"],
            }
        self.receipt["active_request"] = {"case_id": case["case_id"], "arm": arm, "repetition": repetition}
        image, measured, executed, states = native_request(
            torch, pipeline, case, mode=mode, budget=budget, graph=graph, trace=trace, cache=cache,
            session=session, prompt=prompt, admission=admission, identity=identity, source_hashes=self.receipt["source_sha256"],
        )
        if executed is None:
            write_json(directory / "actions.json", {
                "identity": identity,
                "steps": [{"step": step, "action": "full"} for step in range(PROTOCOL["steps"])],
            })
        else:
            executed.save(directory / "actions.json")
        image.save(directory / "output.png", format="PNG")
        manifest = {
            "status": "PASS", **details, **measured,
            "output_path": (relative / "output.png").as_posix(),
            "output_sha256": sha_file(directory / "output.png"),
            "action_trace_path": (relative / "actions.json").as_posix(),
            "action_trace_sha256": sha_file(directory / "actions.json"),
        }
        write_json(directory / "manifest.json", manifest)
        row = {
            **manifest, "manifest_path": (relative / "manifest.json").as_posix(),
            "manifest_sha256": sha_file(directory / "manifest.json"),
        }
        rows.append(row)
        self.receipt.pop("active_request")
        print(f"H1 {index:04d} {arm} {case['case_id']} {row['request_wall_seconds']:.3f}s", flush=True)
        return row, executed, states

    def run_jobs(self, torch, pipeline, jobs):
        from track_h_qwen.frozen.qwen_image.sparseedit_native_trace import load_qwen_trace

        producers = {}
        bound_key, bound_trace = None, None
        admission_rows = {}
        for job in jobs:
            key = (job.case["case_id"], job.budget)
            producer = producers.get(key)
            is_producer = job.arm.startswith("direct_collect")
            if job.budget is not None and not is_producer:
                require(producer is not None, "fixed sparse job is missing its producer")
                if bound_key != key:
                    source = verify_artifact(self.output, producer["action_trace_path"], producer["action_trace_sha256"])
                    bound_key, bound_trace = key, load_qwen_trace(source)
            row, executed, states = self.request(
                torch, pipeline, job.case, job.arm, mode=job.mode, budget=job.budget, graph=job.graph,
                trace=bound_trace if producer is not None else None, producer=producer,
                admission=job.admission, repetition=job.repetition,
            )
            if is_producer:
                require(producer is None, "fixed trajectory producer ran twice")
                producers[key] = row
                bound_key, bound_trace = key, executed
            if job.admission:
                admission_rows[job.arm] = row, states
                if job.graph:
                    self.admit(torch, job.case, job.budget, admission_rows)
                    admission_rows.clear()

    def admit(self, torch, case, budget, rows):
        direct_row, direct = rows[f"admit_direct{budget}"]
        eager_row, eager = rows[f"admit_ready{budget}"]
        graph_row, graph = rows[f"admit_graph{budget}"]
        comparisons = {}
        for name, reference, candidate, left, right in (
            ("direct_vs_ready", direct_row, eager_row, direct, eager),
            ("direct_vs_graph", direct_row, graph_row, direct, graph),
            ("ready_vs_graph", eager_row, graph_row, eager, graph),
        ):
            comparisons[name] = {
                **admission_comparison(torch, left, right),
                "reference_manifest_path": reference["manifest_path"],
                "reference_manifest_sha256": reference["manifest_sha256"],
                "candidate_manifest_path": candidate["manifest_path"],
                "candidate_manifest_sha256": candidate["manifest_sha256"],
            }
        status = "PASS" if all(item["passed"] for item in comparisons.values()) else "FAIL"
        path = f"admission-{budget}.json"
        write_json(self.output / path, {
            "case_id": case["case_id"], "budget_percent": budget, "comparisons": comparisons, "status": status,
        })
        self.receipt["numerical_admission"].append({
            "path": path, "sha256": sha_file(self.output / path), "status": status,
        })
        require(status == "PASS", f"Qwen fixed-capacity numerical admission failed for K={budget}%")

    def publish_calibration(self, cases):
        rows = self.receipt["requests"]
        observations = [value for row in rows if (value := latency_observation(row)) is not None]
        expected = {
            (case["source_pixel_sha256"], budget / 100, arm)
            for case in cases for budget in CALIBRATION_BUDGETS for arm, _, _ in MEASUREMENT_ARMS
        }
        observed = {(row["source_id"], row["ratio"], row["runtime"]) for row in observations}
        require(observed == expected and len(observations) == len(expected), "H1 H4 calibration cells differ")
        path = self.output / "latency_observations.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            for row in observations:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)
        self.receipt["latency_observations"] = {
            "path": path.name, "sha256": sha_file(path), "rows": len(observations), "scope": "calibration_only",
        }

    def session(self, torch, pipeline, cases, prompts):
        cache = SessionCache()
        checks = []
        for index, case in enumerate(cases):
            refs = self.session_arm(torch, pipeline, case, prompts, None)
            candidates = self.session_arm(torch, pipeline, case, prompts, cache)
            for step, (ref, candidate) in enumerate(zip(refs, candidates, strict=True)):
                validate_session_pair(ref, candidate, expect_hit=step > 0)
                checks.append({
                    "source_pixel_sha256": case["source_pixel_sha256"],
                    "instruction_index": step,
                    "vision_hit_count": sum(candidate["cache_events"]["vision"]),
                    "vae_hit_count": sum(candidate["cache_events"]["vae"]),
                    "uncached_manifest": ref["manifest_path"],
                    "uncached_manifest_sha256": ref["manifest_sha256"],
                    "cached_manifest": candidate["manifest_path"],
                    "cached_manifest_sha256": candidate["manifest_sha256"],
                    "sequence": index,
                    "exact_conditioning_and_png": True,
                })
        path = "session_parity.json"
        write_json(self.output / path, checks)
        self.receipt["session_parity"] = {
            "path": path,
            "sha256": sha_file(self.output / path),
            "vision_features_reused": True,
            "reference_vae_latents_reused": True,
            "language_model_forward": "full_multimodal_forward_per_cfg_branch",
        }

    def session_arm(self, torch, pipeline, case, prompts, cache):
        name = "l3_cached" if cache is not None else "l3_uncached"
        rows = []
        for step, prompt in enumerate(prompts):
            row, _, _ = self.request(torch, pipeline, case, f"{name}_{step}", mode="dense",
                                     cache=cache, session_index=step, prompt=prompt, repetition=0)
            rows.append(row)
        return rows

    def execute(self):
        raw_config = self.config.read_bytes()
        self.receipt["config_sha256"] = hashlib.sha256(raw_config).hexdigest()
        config = json.loads(raw_config)
        mode, gpu_uuid = config["run_mode"], config["gpu_uuid"]
        self.receipt.update(run_mode=mode, gpu_uuid=gpu_uuid)
        require(mode in ("cal3", "formal"), "H1 run_mode must be cal3 or formal")
        require(isinstance(gpu_uuid, str) and GPU_UUID.fullmatch(gpu_uuid), "gpu_uuid must be a full NVIDIA GPU UUID")
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "").lower() == gpu_uuid.lower(),
                "suite must bind exactly one visible GPU UUID via CUDA_VISIBLE_DEVICES")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch

        self.receipt["gpu"] = select_gpu(torch, gpu_uuid)
        assets = Path(config.get("assets", ROOT / "payload_data")) / "h1"
        cases = inventory(mode, assets)
        model = Path(config["models"]["qwen"])
        self.receipt["model"] = model_identity(model, assets)
        self.receipt["source_sha256"] = source_identity()
        pipeline, runtime = load_pipeline(torch, model)
        self.receipt["native_runtime"] = runtime
        calibration = calibration_jobs(cases["calibration"])
        formal = heldout_jobs(cases["primary"]) if mode == "formal" else []
        self.receipt["coverage"] = {
            "frozen_calibration_sources": len(cases["calibration"]),
            "frozen_heldout_sources": len(cases["primary"]) if formal else 0,
            "heldout_repetitions": 2 if formal else 0,
            "calibration_budgets_percent": sorted(CALIBRATION_BUDGETS),
            "additional_capacity_probe_budget_percent": 40,
            "session_sources": len(cases["session"]),
            "session_instructions_per_source": 4 if formal else 0,
            "coverage_counts_are_targets": True,
        }
        self.receipt["ordering"] = {
            "calibration": "warmup, dense cal3, fixed-budget producer, admission, and measurement",
            "formal": "cal3, heldout producers, mirrored source and arm passes, then S2 session",
        }

        self.run_jobs(torch, pipeline, calibration)
        self.validate_requests(self.receipt["requests"], [job.identity for job in calibration])
        for receipt in self.receipt["numerical_admission"]:
            verify_artifact(self.output, receipt["path"], receipt["sha256"])
        self.publish_calibration(cases["calibration"])

        if formal:
            start = len(self.receipt["requests"])
            self.run_jobs(torch, pipeline, formal)
            self.validate_requests(self.receipt["requests"][start:], [job.identity for job in formal])
            prompts = frozen_json(assets, "session_prompts.json")
            require(len(prompts) == len(set(prompts)) == 4, "S2 requires four distinct frozen instructions")
            start = len(self.receipt["requests"])
            self.session(torch, pipeline, cases["session"], prompts)
            expected = [
                (case["case_id"], f"l3_{variant}_{step}", 0)
                for case in cases["session"] for variant in ("uncached", "cached") for step in range(4)
            ]
            self.validate_requests(self.receipt["requests"][start:], expected)
            parity = self.receipt["session_parity"]
            verify_artifact(self.output, parity["path"], parity["sha256"])

        require(resident_placement(pipeline, torch.bfloat16) == runtime["parameter_placement"],
                "resident placement changed during measurements")
        self.receipt["coverage"].pop("coverage_counts_are_targets")
        self.receipt["execution_status"] = "PASS"
        self.receipt["status"] = "PARTIAL"

    def validate_requests(self, rows, identities):
        from PIL import Image

        actual = [(r["case_id"], r["arm"], r["repetition"]) for r in rows]
        require(actual == identities, "H1 request matrix differs from target schedule")
        for row in rows:
            path = verify_artifact(self.output, row["manifest_path"], row["manifest_sha256"])
            manifest = json.loads(path.read_text())
            require(manifest == {k: v for k, v in row.items() if k not in ("manifest_path", "manifest_sha256")},
                    "H1 request receipt manifest differs")
            output = verify_artifact(self.output, row["output_path"], row["output_sha256"])
            with Image.open(output) as image:
                require(image.format == "PNG" and image.size == (1024, 1024), "H1 output PNG has wrong format or size")
                image.verify()
            verify_artifact(self.output, row["action_trace_path"], row["action_trace_sha256"])
            totals = timing_summary(row["segments"], dense=row["budget_percent"] is None)
            require(all(row[key] == value for key, value in totals.items()), "H1 aggregate step timings differ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "completion.json").exists() or (args.output / "requests").exists():
        parser.error("H1 output already contains a run")
    lock = args.output / ".h1-run"
    lock.touch(exist_ok=False)
    run = Run(args.config, args.output)
    try:
        run.execute()
    except Exception as error:
        run.receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
    finally:
        run.receipt["finished_at"] = utc_now()
        write_json(args.output / "completion.json", run.receipt)
        lock.unlink()
    return 0 if run.receipt["execution_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
