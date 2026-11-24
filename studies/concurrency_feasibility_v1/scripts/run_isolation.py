#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the executor and private-stream isolation on frozen calibration inputs."""

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import gc
from hashlib import sha256
import importlib.util
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import sys
from time import perf_counter
import traceback

from bench_concurrency import (
    BUNDLE_ROOT, MODEL, STUDY, calibration_pairs, load_pipeline, load_protocol_cases, make_request, resize_case,
)
from check_env import query
from run_baseline import load_cases, load_json, verify_hashes, verify_model_identity
from admission import validate_trace_application

TIMING_BOUNDARY = (
    "Synchronized model worker request from encoding through VAE decode and image postprocessing. "
    "Input resize, model load, independent warmup, PNG serialization and numerical scoring separate."
)
LANES = ("private_graph", "private_eager")


def source_tree_sha():
    files = {
        str(path.relative_to(STUDY / "source")): sha256(path.read_bytes()).hexdigest()
        for path in sorted((STUDY / "source").rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        and ".pytest_cache" not in path.parts
    }
    return sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_receipt(root, receipt):
    receipt["finished_at_utc"] = datetime.now(UTC).isoformat()
    (root / "completion.json").write_text(json.dumps(receipt, indent=2) + "\n")


def image_from_output(output):
    return output["payload"]["image"][0] if isinstance(output, dict) else output[0]


def write_output(output, path):
    image = image_from_output(output)
    stream = BytesIO()
    image.save(stream, format="PNG")
    data = stream.getvalue()
    path.write_bytes(data)
    return sha256(data).hexdigest()


def tensor_hash(value):
    import torch

    return sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


@contextmanager
def frozen_full_reference(pipeline, *, capture):
    """Use the archived full selector and passively witness native Euler transitions."""
    from vllm_omni.diffusion.models.flux import paper_selector_runtime

    name = paper_selector_runtime.__package__ + "._frozen_paper_selector_reference"
    path = BUNDLE_ROOT / "runtime/vllm_omni/diffusion/models/flux/paper_selector_runtime.py"
    spec = importlib.util.spec_from_file_location(name, path)
    frozen = importlib.util.module_from_spec(spec)
    sys.modules[name] = frozen
    spec.loader.exec_module(frozen)
    original = paper_selector_runtime.diffuse_sparseedit_paper
    paper_selector_runtime.diffuse_sparseedit_paper = frozen.diffuse_sparseedit_paper
    snapshots = []
    original_step = pipeline.scheduler.step

    def step(velocity, timestep, latents, *args, **kwargs):
        from vllm_omni.diffusion.models.flux.request_state import StepSnapshot

        result = original_step(velocity, timestep, latents, *args, **kwargs)
        snapshots.append(StepSnapshot(len(snapshots), result[0].detach().clone(), velocity.detach().clone()))
        return result

    if capture:
        pipeline.scheduler.step = step
    try:
        yield snapshots
    finally:
        if capture:
            del pipeline.scheduler.step
        paper_selector_runtime.diffuse_sparseedit_paper = original
        del sys.modules[name]


def step_indices_match(candidate, reference):
    sparse = candidate["is_sparse_step"]
    got = candidate.get("selected_index_digest")
    want = reference.get("selected_index_digest")
    return sparse == reference["is_sparse_step"] and got == want and (got is not None) == sparse


def compare_steps(out, expected_steps, expected_meta, path):
    import torch

    got_meta = out.metadata["step_records"]
    want_meta = expected_meta["step_records"]
    if not len(out.steps) == len(expected_steps) == len(got_meta) == len(want_meta) == 28:
        raise ValueError("Expected all 28 denoising steps and snapshots")
    result = []
    for got, want, got_row, want_row in zip(out.steps, expected_steps, got_meta, want_meta, strict=True):
        if got.step != want.step:
            raise ValueError("Step order differs between reference and candidate")
        tensors = {}
        for name in ("latents", "velocity"):
            left = getattr(got, name)
            right = getattr(want, name)
            equal = torch.equal(left, right)
            tensors[name] = {"equal": equal, "candidate_sha256": tensor_hash(left), "reference_sha256": tensor_hash(right)}
            if not equal:
                delta = left.float() - right.float()
                filename = f"{path.name}_step{got.step:02d}_{name}_difference.pt"
                torch.save({"candidate": left.cpu(), "reference": right.cpu(), "delta": delta.cpu()}, path.with_name(filename))
                tensors[name].update(max_abs_diff=float(delta.abs().max()), raw_difference=filename)
        indices = step_indices_match(got_row, want_row)
        action = got_row["kv_action"] == want_row["kv_action"]
        result.append({
            "step": got.step, "action": got_row["kv_action"], "action_equal": action,
            "indices_equal": indices, "selected_index_digest": got_row.get("selected_index_digest"), **tensors,
        })
    return result


def step_match(row):
    return (row["action_equal"] and row["indices_equal"] and row["latents"]["equal"]
            and row["velocity"]["equal"])


def storage_ranges(state):
    import torch

    def interval(tensor):
        owned = tensor.untyped_storage()
        return owned.data_ptr(), owned.data_ptr() + owned.nbytes()

    kv = set()
    for cache in state.caches:
        for storage in cache.layers.values():
            tensors = (storage.key, storage.value) if hasattr(storage, "key") else storage
            kv.update(interval(tensor) for tensor in tensors)
    conditioning = {interval(tensor) for tensor in state.conditioning.values() if isinstance(tensor, torch.Tensor)}
    return sorted(kv), sorted(conditioning)


def disjoint(left, right):
    return not any(a < d and c < b for a, b in left for c, d in right)


def validate_lane(metadata, lane, trace):
    validate_trace_application(metadata, trace)
    active = metadata["summary"]["graph_active"]
    if active != (lane == "private_graph"):
        raise ValueError(f"Unexpected graph execution in {lane}: graph_active={active}")


def serial(executor, request, case_tag):
    import torch
    from vllm_omni.diffusion.models.flux.request_context import RequestContext

    state = executor.init_state(request, RequestContext(case_tag, -1, None, {}), stream=torch.cuda.Stream())
    try:
        for step in range(len(state.timesteps)):
            executor.run_step(state, step)
        return executor.finalize(state)
    except BaseException:
        executor.abort(state)
        raise


def run_equivalence(root, receipt, executor, pipeline, cases, images, protocol, native_postprocess):
    import torch
    from vllm_omni.diffusion.models.flux.specialized_template import clear_template

    warm = cases[0]
    def native(case, capture):
        tag = case[0]["case_tag"]
        request = make_request(case, "full", protocol, image=images[tag])
        torch.cuda.synchronize()
        started = perf_counter()
        with frozen_full_reference(pipeline, capture=capture) as steps:
            response = native_postprocess(pipeline.forward(request).output)
        torch.cuda.synchronize()
        return response, steps, pipeline._last_flux_acceleration_metadata, perf_counter() - started

    print("Task 2: independent frozen graph warmup", flush=True)
    _, _, _, native_warmup_seconds = native(warm, False)
    native_timings = {}
    for case in cases:
        tag = case[0]["case_tag"]
        _, _, metadata, seconds = native(case, False)
        native_timings[tag] = (seconds, metadata["run_config"].get("graph"))
    clear_template(pipeline)
    gc.collect()
    torch.cuda.empty_cache()

    from vllm_omni.diffusion.models.flux.request_executor import RequestExecutor
    clean = RequestExecutor(pipeline, post_process=native_postprocess)
    warmed = serial(clean, make_request(warm, "full", protocol, image=images[warm[0]["case_tag"]]), warm[0]["case_tag"])
    receipt["independent_warmup_seconds"] = {
        "native": native_warmup_seconds, "executor": warmed.timing["request_seconds"],
    }
    del warmed
    for case in cases:
        tag = case[0]["case_tag"]
        print(f"Task 2: frozen graph versus executor {tag}", flush=True)
        measured = serial(clean, make_request(case, "full", protocol, image=images[tag]), tag)
        response, snapshots, meta, _ = native(case, True)
        native_png = write_output(response, root / f"{tag}_native_graph.png")
        clear_template(pipeline)
        out = serial(executor, make_request(case, "full", protocol, image=images[tag]), tag)
        actual_png = write_output(out.output, root / f"{tag}_executor_graph.png")
        rows = compare_steps(out, snapshots, meta, root / tag)
        validate_lane(out.metadata, "private_graph", case[3])
        validate_lane(meta, "private_graph", case[3])
        match = native_png == actual_png and all(step_match(row) for row in rows)
        native_seconds, warm_graph = native_timings[tag]
        graph = measured.metadata["run_config"].get("graph") or {}
        preparation_seconds = sum(
            branch["warmup_seconds"] + branch["capture_seconds"]
            for branch in graph.get("branches", {}).values()
        )
        receipt["cases"].append({
            "case_tag": tag, "input_sha256": case[0]["prepared_sha256"], "trace_sha256": case[0]["trace_sha256"],
            "output_png_sha256": actual_png, "native_png_sha256": native_png, "all_steps_match": match,
            "steps": rows, "native_seconds": native_seconds, "executor_seconds": measured.timing["request_seconds"],
            "condition_plan": measured.metadata["run_config"].get("condition_plan"),
            "positions": measured.metadata["run_config"].get("request_positions"),
            "graph": graph, "native_graph": warm_graph, "memory": measured.memory,
            "private_graph_warmup_and_capture_seconds": preparation_seconds,
            "relative_wall_delta": measured.timing["request_seconds"] / native_seconds - 1,
        })
    receipt["requests"] = len(receipt["cases"]) * 2
    receipt["latency_offset_requires_profile"] = any(abs(case["relative_wall_delta"]) > .02 for case in receipt["cases"])
    receipt["status"] = "PASS" if all(row["all_steps_match"] for row in receipt["cases"]) else "FAILED"


def run_pair(root, receipt, executor, pipeline, cases, images, protocol, serial_references):
    import torch
    from vllm_omni.diffusion.models.flux.concurrent_scheduler import ConcurrentScheduler
    from vllm_omni.diffusion.models.flux.request_context import RequestContext

    lane = receipt["runtime_lane"]
    tags = [row["case_tag"] for row, *_ in cases]
    print(f"F1 {lane}: initialize {tags}", flush=True)
    states = []
    witnesses = []

    class IsolationScheduler(ConcurrentScheduler):
        def __init__(self, exec):
            super().__init__(exec)

        def observe(self, step):
            if step < 7:
                return
            owned = [storage_ranges(state) for state in states]
            kv = [ranges[0] for ranges in owned]
            cond = [ranges[1] for ranges in owned]
            if not all(kv) or not disjoint(*kv) or not disjoint(*cond):
                raise ValueError(f"Request-owned KV or conditioning storage overlaps at step {step}")
            witnesses.append({
                "step": step, "kv_storage_bytes": [sum(end - start for start, end in ranges) for ranges in kv],
                "kv_ranges_checked": [len(ranges) for ranges in kv],
                "kv_storage_disjoint": True, "conditioning_storage_disjoint": True,
            })
            if step in (7, 8, 23, 27):
                witnesses[-1]["kv_address_ranges"] = kv

    scheduler = IsolationScheduler(executor)
    observe = executor.run_step

    def before_step(state, step):
        observe(state, step)
        if state is states[-1]:
            scheduler.observe(step)

    executor.run_step = before_step
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    receipt["concurrency"] = 2
    receipt["shared_allocated_bytes"] = torch.cuda.memory_allocated()
    receipt["shared_reserved_bytes"] = torch.cuda.memory_reserved()
    receipt["budget_bytes"] = torch.cuda.mem_get_info()[0] + receipt["shared_reserved_bytes"]
    receipt["storage_witnesses"] = witnesses
    try:
        for case, tag in zip(cases, tags, strict=True):
            request = make_request(case, "full", protocol, image=images[tag], lane=lane)
            states.append(executor.init_state(request, RequestContext(tag, -1, None, {}), stream=torch.cuda.Stream()))
        object_ids = {
            "context": [id(state.context) for state in states],
            "scheduler": [id(state.pipeline.scheduler) for state in states],
            "template": [None if state.template is None else id(state.template) for state in states],
            "caches": [sorted(id(cache) for cache in state.caches) for state in states],
        }
        if (len(set(object_ids["context"])) != 2 or len(set(object_ids["scheduler"])) != 2
                or object_ids["template"][0] is not None and len(set(object_ids["template"])) != 2
                or set(object_ids["caches"][0]) & set(object_ids["caches"][1])):
            raise ValueError("Mutable request owners alias")
        receipt["private_owner_ids"] = object_ids
        results = scheduler.run(states)
        receipt["synchronized_before_return"] = True
    finally:
        receipt["enqueue_order"] = scheduler.enqueue_order
        receipt["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        del executor.run_step
        for state in states:
            executor.abort(state)

    complete = []
    for case, tag, got in zip(cases, tags, results, strict=True):
        want, expected_png = serial_references[tag]
        validate_lane(got.metadata, lane, case[3])
        path = root / f"{tag}.png"
        png = write_output(got.output, path)
        rows = compare_steps(got, want.steps, want.metadata, root / tag)
        row = next(row for row in receipt["cases"] if row["case_tag"] == tag)
        row.update(output_png_sha256=png, serial_png_sha256=expected_png, steps=rows,
                   final_latents_equal=torch.equal(got.latents, want.latents),
                   graph=got.metadata["run_config"].get("graph"))
        complete.append(png == expected_png and row["final_latents_equal"] and all(step_match(r) for r in rows))
    receipt["correctness"] = {"all_match": all(complete), "private_storage_disjoint": bool(witnesses)}
    receipt["status"] = "PASS" if all(complete) and witnesses else "FAILED"
    receipt["requests"] = len(complete)


def exception_status(error):
    import torch

    checked = set()
    while error is not None and id(error) not in checked:
        checked.add(id(error))
        if isinstance(error, torch.OutOfMemoryError):
            return "CAPACITY_BLOCKED", {"type": "OutOfMemoryError", "message": str(error)}
        error = error.__cause__ or error.__context__
    return "FAILED", None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--lane", choices=LANES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--equivalence", action="store_true")
    parser.add_argument("--publish-pairs", action="store_true", help="Expose local PASS entries in isolation_v1")
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--verify-full-model-hashes", action="store_true",
                        help="Use frozen file hashes for the same model on a node-local mount")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not root.is_relative_to(STUDY / "artifacts"):
        parser.error("Select a fresh study artifact directory")
    if args.equivalence and args.lane != "private_graph":
        parser.error("Task 2 uses frozen graph_reuse full")
    root.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    protocol, identities = load_protocol_cases()
    versions = {"python": sys.version.split()[0], **{name: version(name) for name in ("torch", "triton", "vllm")}}
    common = {
        "status": "FAILED", "phase": "Task2" if args.equivalence else "F1",
        "protocol": protocol, "runtime_lane": args.lane, "memory_arm": args.lane,
        "host": socket.gethostname(), "device_uuid": args.gpu_uuid, "environment": versions,
        "python_path": sys.executable,
        "source_sha": source_tree_sha(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "repetition": 0, "attempt": args.attempt, "timing_boundary": TIMING_BOUNDARY, "requests": 0,
    }
    receipt = {**common, "cases": []}
    lock_path = Path(f"/tmp/sparseedit-{args.gpu_uuid}.lock")
    try:
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt["preflight"] = {
                "device": query(args.gpu_uuid, "gpu", "name,uuid,memory.total,memory.free,driver_version"),
                "external_processes": query(args.gpu_uuid, "compute-apps", "gpu_uuid,pid,used_memory"),
            }
            if receipt["preflight"]["external_processes"]:
                receipt.update(status="BLOCKED", failure="GPU has an external compute process")
                return 2
            import torch
            from vllm_omni.diffusion.models.flux.request_executor import RequestExecutor

            bundle = load_json(BUNDLE_ROOT / "bundle.json")
            if args.verify_full_model_hashes:
                print("Verifying the same v4 model bytes on the execution node", flush=True)
                verify_hashes(MODEL, bundle["model_hashes"])
                receipt["model_identity"] = {"method": "frozen v4 file checksums", "files": len(bundle["model_hashes"])}
            else:
                receipt["model_identity"] = verify_model_identity(bundle["model_hashes"])
            case_list = load_cases(BUNDLE_ROOT, bundle)
            cases = {row["case_tag"]: bound for bound in case_list for row in (bound[0],)}
            start_resize = perf_counter()
            images = {tag: resize_case(bound, protocol) for tag, bound in cases.items()}
            receipt["input_resize_seconds"] = perf_counter() - start_resize
            started = perf_counter()
            with load_pipeline(MODEL) as (pipeline, postprocess):
                torch.cuda.synchronize()
                receipt["model_load_seconds"] = perf_counter() - started
                executor = RequestExecutor(pipeline, post_process=postprocess, capture_steps=True)
                if args.equivalence:
                    run_equivalence(root, receipt, executor, pipeline, case_list, images, protocol, postprocess)
                else:
                    references = {}
                    (root / "serial_reference").mkdir()
                    for tag, bound in cases.items():
                        print(f"F1 {args.lane}: serial reference {tag}", flush=True)
                        candidate = serial(executor, make_request(bound, "full", protocol, image=images[tag],
                                                                  lane=args.lane), tag)
                        validate_lane(candidate.metadata, args.lane, bound[3])
                        png = write_output(candidate.output, root / "serial_reference" / f"{tag}.png")
                        references[tag] = (candidate, png)
                    receipt["serial_reference_requests"] = len(references)
                    receipt["pairs"] = []
                    for tags in calibration_pairs(tuple(cases)):
                        attempt_name = "__".join(tags) + f"__{args.lane}_attempt{args.attempt}"
                        pair_root = root / attempt_name
                        pair_root.mkdir()
                        pair = {**common, "cases": [dict(identities[tag]) for tag in tags]}
                        try:
                            run_pair(pair_root, pair, executor, pipeline, [cases[tag] for tag in tags],
                                     images, protocol, references)
                        except Exception as error:
                            status, failure = exception_status(error)
                            pair.update(status=status, failure=failure or {"type": type(error).__name__, "message": str(error)})
                            (pair_root / "traceback.log").write_text(traceback.format_exc())
                            print(traceback.format_exc(), flush=True)
                        finally:
                            write_receipt(pair_root, pair)
                        if pair["status"] == "PASS" and args.publish_pairs:
                            passed_root = STUDY / "artifacts/isolation_v1" / "__".join(tags)
                            passed_root.symlink_to(os.path.relpath(pair_root, passed_root.parent), target_is_directory=True)
                        receipt["pairs"].append({"path": str(pair_root / "completion.json"), "status": pair["status"]})
                        receipt["requests"] += pair["requests"]
                        if pair["status"] != "PASS":
                            break
                    statuses = [pair["status"] for pair in receipt["pairs"]]
                    receipt["status"] = "PASS" if statuses == ["PASS"] * 3 else statuses[-1]
    except Exception as error:
        status, failure = exception_status(error)
        receipt.update(status=status, failure=failure or {"type": type(error).__name__, "message": str(error)})
        (root / "traceback.log").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        write_receipt(root, receipt)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
