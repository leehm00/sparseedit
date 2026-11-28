# SPDX-License-Identifier: Apache-2.0
"""H3 FLUX workers and the frozen serial-CFG comparator ports."""

import ast
from contextlib import contextmanager
from math import isfinite
import os
from pathlib import Path
import shutil
from statistics import mean
from tempfile import TemporaryDirectory

from .contracts import file_sha

FLUX = Path("vllm_omni/diffusion/models/flux")
TEA_HOOK = Path("vllm_omni/diffusion/cache/teacache/hook.py")
PORT_SHA256 = "18048ea34a1be20abfad09b875ab4ca0a3ad0272a6560ea19f5ddd21ee2c3310"
TEA = {"rel_l1_thresh": .2}
TOCA = {
    "toca_fresh_ratio": .1, "toca_fresh_threshold": 4, "toca_ratio_scheduler": "ToCa-ddim50",
    "toca_score_mode": "cache_frequency", "toca_soft_fresh_weight": .25,
    "toca_spatial_bonus_ratio": .6, "toca_spatial_grid_size": 2, "toca_token_scope": "generated",
}
PORTS = {
    "ras": ("ras_cfg_compose", {
        "ras_target_ratio": .25, "ras_ratio_schedule": "linear", "ras_step_ratios": None,
        "ras_k": .1, "ras_selective_start_step": 4, "ras_dense_steps": [12, 20],
        "ras_score_direction": "high_std", "ras_selection_rounding": "ceil",
    }),
    "regione": ("regione_cfg_isolated", {
        "regione_warmup_step": 6, "regione_post_step": 2, "regione_refresh_steps": [16],
        "regione_partition_threshold": .93, "regione_erosion_dilation": True,
        "regione_enable_avd": True, "regione_cache_threshold": .04,
    }),
}


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"H3 FLUX source anchor differs: {before!r}")
    return source.replace(before, after, 1)


def acceleration_overlay(source):
    source = replace_once(source, '    "ras",\n    "regione",\n',
                          '    "ras",\n    "regione",\n    "ras_cfg_compose",\n    "regione_cfg_isolated",\n')
    source = replace_once(source, '            "ras",\n            "regione",\n',
                          '            "ras",\n            "regione",\n            "ras_cfg_compose",\n            "regione_cfg_isolated",\n')
    source = replace_once(source, "    num_inference_steps: int = 28\n", """    num_inference_steps: int = 28
    cfg_ports_version: str = ""
    cfg_ports_full_execution: bool = False
    cfg_ports_admission_directory: str = ""
""")
    source = replace_once(source, "        if self.num_inference_steps <= 0:\n", """        if self.method in {"ras_cfg_compose", "regione_cfg_isolated"}:
            if self.cfg_ports_version != "serial_v1" or self.num_inference_steps != 28:
                raise ValueError("CFG ports require explicit serial_v1 and the frozen 28-step schedule")
        if self.num_inference_steps <= 0:
""")
    source = replace_once(source, 'if self.method == "ras":', 'if self.method in {"ras", "ras_cfg_compose"}:')
    source = replace_once(source, 'if self.method == "regione":', 'if self.method in {"regione", "regione_cfg_isolated"}:')
    source = replace_once(source, "            num_inference_steps=num_inference_steps,\n", """            num_inference_steps=num_inference_steps,
            cfg_ports_version=str(values.get("cfg_ports_version", "")),
""")
    source = replace_once(source, 'return self.method in {"toca", "ras", "regione", "spotedit", "sparseedit"}',
                          'return self.method in {"toca", "ras", "regione", "spotedit", "sparseedit", "ras_cfg_compose", "regione_cfg_isolated"}')
    return replace_once(source, '        prefix = f"{self.method}_"\n', """        prefix = f"{self.method}_"
        if self.method in {"ras_cfg_compose", "regione_cfg_isolated"}:
            prefix = "ras_" if self.method == "ras_cfg_compose" else "regione_"
            metadata["cfg_ports_version"] = self.cfg_ports_version
""")


def runtime_overlay(source):
    anchor = "    grid_metadata = _validate_runtime(pipeline, config, latents, image_latents, latent_ids, do_true_cfg)\n"
    return replace_once(source, anchor, """    if config.method in {"ras_cfg_compose", "regione_cfg_isolated"}:
        from .cfg_ports_runtime import diffuse_cfg_port

        return diffuse_cfg_port(**locals())
""" + anchor)


def tea_hook_overlay(source):
    source = replace_once(source, "        self._forward_cnt = 0\n        return module\n", """        self._forward_cnt = 0
        self._track_h_actions = []
        return module
""")
    anchor = "        if not should_compute and state.previous_residual is not None:\n"
    return replace_once(source, anchor, """        reuse = not should_compute and state.previous_residual is not None
        if not hasattr(self, "_track_h_actions"):
            self._track_h_actions = []
        self._track_h_actions.append({
            "step": state.cnt, "branch": cache_branch, "action": "reuse" if reuse else "compute",
        })
        if reuse:
""")


def pipeline_overlay(source):
    anchor = "            if formal_memory:\n                metadata[\"complete_request_peak_memory\"]"
    return replace_once(source, anchor, """            if acceleration_config.method == "tea_cache":
                hook = self.transformer._hook_registry.get_hook("teacache")
                metadata["track_h_teacache_actions"] = list(hook._track_h_actions)
""" + anchor)


@contextmanager
def overlay_source(base, scratch):
    base = Path(base).resolve()
    frozen = Path(__file__).with_name("frozen_cfg_ports.py")
    if file_sha(frozen) != PORT_SHA256:
        raise ValueError("frozen historical FLUX CFG ports source differs")
    with TemporaryDirectory(prefix="h3-flux-overlay-", dir=scratch) as directory:
        root = Path(directory) / "source"
        shutil.copytree(base, root, copy_function=os.symlink, ignore=shutil.ignore_patterns("__pycache__"))
        overlays = {
            FLUX / "acceleration.py": acceleration_overlay,
            FLUX / "acceleration_runtime.py": runtime_overlay,
            FLUX / "pipeline_flux_kontext.py": pipeline_overlay,
            TEA_HOOK: tea_hook_overlay,
        }
        for relative, transform in overlays.items():
            path = root / relative
            revised = transform(path.read_text())
            ast.parse(revised, filename=str(relative))
            path.unlink()
            path.write_text(revised)
        shutil.copyfile(frozen, root / FLUX / "cfg_ports_runtime.py")
        yield root


@contextmanager
def arm_worker(model, gpu_uuid, arm, side, overlay):
    from h2_runtime.flux_native import native_worker, request_options

    options = {"gpu_uuid": gpu_uuid, "dense_graph": arm == "dense_graph"}
    extras = {"sparseedit_return_metadata": True}
    method = "dense"
    if arm == "teacache":
        method = "tea_cache"
        options.update(source_root=overlay, cache_backend=method, cache_config=TEA)
        extras.update(TEA)
    elif arm == "toca":
        method = "toca"
        extras.update(TOCA)
    elif arm in PORTS:
        method, knobs = PORTS[arm]
        options["source_root"] = overlay
        extras.update(knobs, cfg_ports_version="serial_v1")
    elif arm.startswith("sparseedit_"):
        joint = arm == "sparseedit_k_b_joint_cfg_lane_q"
        only = arm == "sparseedit_k_only"
        method, extras = request_options("full", side=side)
        options.update(paper_budget=not only, paper_joint=joint, route=only, session="demand" if only else "off")
        if joint:
            extras.update(sparseedit_cfg_execution="budget", sparseedit_template_capacity=3)
    with native_worker(model, **options) as worker:
        yield worker, method, extras


def dense_actions():
    return [{"step": step, "action": "full"} for step in range(28)]


def action_evidence(arm, metadata):
    if arm == "fresh_dense":
        if metadata["method"] != "dense" or metadata["summary"]["dense_steps"] != 28:
            raise ValueError("FLUX Fresh Dense path differs")
        return dense_actions()
    if arm == "dense_graph":
        if metadata["dense_graph"]["request_replays"] != {"positive": 28, "negative": 28}:
            raise ValueError("FLUX Dense Graph replay scope differs")
        return dense_actions()
    if arm == "teacache":
        events = metadata["track_h_teacache_actions"]
        expected = [(step, branch) for step in range(28) for branch in ("positive", "negative")]
        if [(event["step"], event["branch"]) for event in events] != expected:
            raise ValueError("TeaCache lacks aligned two-branch request actions")
        if any(event["action"] not in ("compute", "reuse") or event["step"] == 0 and event["action"] != "compute"
               for event in events):
            raise ValueError("TeaCache request reset or action scope differs")
        return events
    if arm == "toca":
        records = metadata["diagnostics"]
        branch_steps = {(row.get("cfg_branch", "positive"), row["step"]) for row in records}
        if branch_steps != {(branch, step) for branch in ("positive", "negative") for step in range(28)}:
            raise ValueError("ToCa lacks two-branch layer actions")
        return records
    if arm in PORTS:
        records = metadata["diagnostics"]
        if ([row["step"] for row in records] != list(range(28)) or metadata["observation_enabled"]
                or metadata["config"]["method"] != PORTS[arm][0]):
            raise ValueError("FLUX serial CFG port lacks unobserved native 28-step evidence")
        return records
    route = metadata["track_a_warmup_route"]
    graph = metadata["run_config"]["graph"]
    steps = metadata["step_records"]
    joint = arm == "sparseedit_k_b_joint_cfg_lane_q"
    if (route["K"] not in (460, 921) or len(steps) != 28 or not graph["graph_active"]
            or graph["cfg_execution"] != ("batched" if joint else "serial")
            or graph["total_replays"] != metadata["summary"]["sparse_steps"] * (1 if joint else 2)):
        raise ValueError("FLUX routed graph execution differs")
    if arm != "sparseedit_k_only" and (metadata["paper_budget"]["K"] != route["K"]
                                      or len(metadata["paper_budget"]["steps"]) != 28):
        raise ValueError("FLUX online K+B budget witness differs")
    return steps


def latency_observation(arm, row, metadata, steps):
    if arm not in ("sparseedit_k_b_lane_e", "sparseedit_k_b_joint_cfg_lane_q"):
        return None
    computed = [step for step in steps if step["transformer_computed"]]
    timing = metadata["timing"]
    durations = timing["step_forward_device_seconds"]
    pairs = list(zip(computed, durations, strict=True))
    dense = [duration for step, duration in pairs if not step["is_sparse_step"]]
    sparse = [duration for step, duration in pairs if step["is_sparse_step"]]
    if len(dense) != 9 or len(sparse) < 2 or any(not isfinite(v) or v <= 0 for v in durations):
        raise ValueError("FLUX online K+B device timing intervals differ")
    graph = metadata["run_config"]["graph"]
    from h4 import capture_features

    capture = capture_features(graph)
    capture.pop("capture_wall_s")
    return {
        "split": "heldout", "model": "flux", "resolution": row["resolution"], "repetition": row["repetition"],
        "runtime": "joint_cfg" if arm.endswith("lane_q") else "full",
        "source_id": row["source_id"], "ratio": metadata["run_config"]["active_token_count"] / 2304,
        "request_wall_s": row["request_wall_s"], "denoise_wall_s": timing["denoise_wall_seconds"],
        "full_steps": len(dense), "sparse_steps": len(sparse), "reuse_steps": len(steps) - len(computed),
        "full_device_s": sum(dense), "steady_sparse_device_s": mean(sparse[1:]),
        **capture,
    }
