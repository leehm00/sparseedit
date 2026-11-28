# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-scoped acceleration helpers for FLUX.1 Kontext-dev.

The module keeps method configuration and tensor-only state independent from
the pipeline implementation.  It intentionally does not import model weights,
Diffusers pipelines, or CUDA-only kernels so schedules and cache semantics can
be tested on small CPU tensors.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

FluxAccelerationMethod = Literal[
    "dense",
    "tea_cache",
    "cache_dit",
    "toca",
    "ras",
    "regione",
    "spotedit",
    "sparseedit",
]
FluxSparseMethod = Literal["ras", "regione", "spotedit", "sparseedit"]
FluxSparseKVAction = Literal["none", "store_full", "update_active"]
FluxStream = Literal["dual", "single"]

FLUX_REGIONE_GAMMA_28: tuple[float, ...] = (
    0.8352,
    0.9986,
    1.0090,
    1.0097,
    1.0161,
    1.0152,
    1.0160,
    1.0173,
    1.0177,
    1.0199,
    1.0213,
    1.0203,
    1.0257,
    1.0236,
    1.0235,
    1.0278,
    1.0302,
    1.0311,
    1.0352,
    1.0371,
    1.0391,
    1.0459,
    1.0498,
    1.0581,
    1.0693,
    1.0866,
    1.1090,
)


def _tuple_of_ints(value: Any, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "none":
            return ()
        return tuple(int(item.strip()) for item in stripped.split(",") if item.strip())
    return tuple(int(item) for item in value)


def _tuple_of_strict_ints(
    value: Any, *, default: tuple[int, ...], name: str
) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "none":
            return ()
        items: Any = stripped.split(",")
    else:
        items = value

    parsed: list[int] = []
    try:
        for item in items:
            if isinstance(item, bool):
                raise ValueError(f"{name} must contain only integers")
            if isinstance(item, Integral):
                parsed.append(int(item))
                continue
            if isinstance(item, str) and re.fullmatch(r"[+-]?\d+", item.strip()):
                parsed.append(int(item.strip()))
                continue
            raise ValueError(f"{name} must contain only integers")
    except TypeError as error:
        raise ValueError(f"{name} must contain only integers") from error
    return tuple(parsed)


def _tuple_of_floats(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "none":
            return None
        return tuple(
            float(item.strip()) for item in stripped.split(",") if item.strip()
        )
    return tuple(float(item) for item in value)


@dataclass(frozen=True)
class FluxKontextAccelerationConfig:
    """Resolved method and experiment parameters for one request."""

    method: FluxAccelerationMethod = "dense"
    num_inference_steps: int = 28

    toca_fresh_ratio: float = 0.07
    toca_fresh_threshold: int = 4
    toca_ratio_scheduler: str = "ToCa-ddim50"
    toca_score_mode: str = "cache_frequency"
    toca_soft_fresh_weight: float = 0.25
    toca_spatial_bonus_ratio: float = 0.6
    toca_spatial_grid_size: int = 2
    toca_token_scope: str = "generated"

    ras_k: float = 0.1
    ras_target_ratio: float = 0.25
    ras_ratio_schedule: str = "linear"
    ras_step_ratios: tuple[float, ...] | None = None
    ras_selective_start_step: int = 4
    ras_dense_steps: tuple[int, ...] = (12, 20)
    ras_score_direction: str = "high_std"
    ras_selection_rounding: str = "ceil"

    regione_warmup_step: int = 6
    regione_post_step: int = 2
    regione_refresh_steps: tuple[int, ...] = (16,)
    regione_partition_threshold: float = 0.93
    regione_cache_threshold: float = 0.04
    regione_erosion_dilation: bool = True
    regione_enable_avd: bool = True

    spotedit_profile: str = "source"
    spotedit_threshold: float = 0.2
    spotedit_judge_method: str = "LPIPS"
    spotedit_initial_steps: int = 4
    spotedit_reset_steps: tuple[int, ...] = (13, 22)
    spotedit_dilation_radius: int = 0
    spotedit_reuse_mode: str = "velocity"
    spotedit_select_every_step: bool = True

    sparseedit_ratio: float = 0.30
    sparseedit_warmup_steps: int = 6
    sparseedit_refresh_interval: int = 15
    sparseedit_refresh_mode: str = "full"
    sparseedit_warmup_ref_freeze_step: int | None = None
    sparseedit_locality_mode: str = "default"
    sparseedit_selector_protocol: str = "legacy"
    sparseedit_paper_mode: str = "forward"
    sparseedit_paper_constant_weight: float = 0.0
    sparseedit_paper_random_seed: int = 110
    sparseedit_avd_enabled: bool = False
    sparseedit_avd_cache_threshold: float = 0.010
    sparseedit_avd_max_reuse: int = 1
    sparseedit_avd_force_final_steps: int = 4
    sparseedit_avd_gamma: tuple[float, ...] | None = None
    sparseedit_online_reuse: str = "off"
    sparseedit_collect_warmup: bool = False
    sparseedit_collect_gamma: bool = False
    sparseedit_collect_masks: bool = False
    sparseedit_return_metadata: bool = False
    # Opt-in specialized execution variants for the paper-selector protocol.
    # The default preserves the original runtime exactly.
    sparseedit_runtime_variant: str = "baseline"
    sparseedit_direct_ops: str = "off"
    sparseedit_standard_backend: str = "off"
    sparseedit_standard_audit: str = "off"
    sparseedit_standard_audit_dir: str = ""
    sparseedit_cache_layout: str = "auto"
    sparseedit_reference_kv_int8: bool = False
    sparseedit_template_capacity: int = 1
    sparseedit_prewarm_ratios: tuple[float, ...] | None = None
    sparseedit_prewarm_boundary: str = "sparse"
    sparseedit_condition_bank_shared: bool = False
    sparseedit_replay_trace: str = ""
    sparseedit_audit_dir: str = ""
    sparseedit_collect_audit: bool = False
    sparseedit_collect_trace: bool = False
    sparseedit_defer_diagnostics: bool = False
    sparseedit_graph_fallback: bool = False
    sparseedit_graph_warmup_trials: int = 2
    sparseedit_stage_fusion: bool = False
    sparseedit_layer_fusion: bool = False
    sparseedit_fuse_elementwise: bool = True
    sparseedit_share_positions: bool = True
    sparseedit_share_joint: bool = True
    sparseedit_cfg_execution: str = "serial"
    sparseedit_cfg_min_query_rows: int = 1200
    sparseedit_condition_plan_mode: str = "off"
    sparseedit_condition_prepare_graph: bool = False
    sparseedit_share_condition: bool = False
    sparseedit_request_position_reuse: bool = False
    sparseedit_profile_directory: str = ""
    sparseedit_profile_step: int = 10

    def __post_init__(self) -> None:
        if self.sparseedit_condition_prepare_graph and not (
            self.method == "sparseedit"
            and self.sparseedit_condition_plan_mode
            in {"warmup_serial", "warmup_overlap"}
        ):
            raise ValueError(
                "sparseedit_condition_prepare_graph requires SparseEdit warmup_serial or warmup_overlap"
            )
        supported = {
            "dense",
            "tea_cache",
            "cache_dit",
            "toca",
            "ras",
            "regione",
            "spotedit",
            "sparseedit",
        }
        if self.method not in supported:
            raise ValueError(f"unsupported FLUX acceleration method: {self.method}")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.method == "sparseedit":
            if self.num_inference_steps != 28:
                raise ValueError("FLUX SparseEdit routing protocol requires 28 steps")
            if (
                not math.isfinite(self.sparseedit_ratio)
                or not 0 < self.sparseedit_ratio <= 1
            ):
                raise ValueError("sparseedit_ratio must be in (0, 1]")
            if not 1 <= self.sparseedit_warmup_steps < self.num_inference_steps:
                raise ValueError(
                    "sparseedit_warmup_steps must be within the denoising schedule"
                )
            if self.sparseedit_refresh_interval < 0:
                raise ValueError("sparseedit_refresh_interval must be non-negative")
            if self.sparseedit_refresh_mode not in ("full", "generated_only"):
                raise ValueError("sparseedit_refresh_mode must be full or generated_only")
            freeze = self.sparseedit_warmup_ref_freeze_step
            if freeze is not None and (
                type(freeze) is not int or not 0 < freeze < self.sparseedit_warmup_steps
                or self.sparseedit_refresh_mode != "generated_only"
            ):
                raise ValueError("reference freeze requires generated_only and an interior warmup step")
            if self.sparseedit_refresh_mode == "generated_only" and (
                self.sparseedit_selector_protocol != "paper"
                or self.sparseedit_runtime_variant != "graph_reuse"
                or self.sparseedit_cache_layout == "raw"
                or not self.sparseedit_request_position_reuse
                or self.sparseedit_prewarm_ratios
                or not self.sparseedit_refresh_interval
                or self.sparseedit_ratio == 1
            ):
                raise ValueError("generated_only requires sparse graph_reuse with rotated positions and refresh")
            if self.sparseedit_locality_mode not in {"default", "off", "linear"}:
                raise ValueError(
                    "sparseedit_locality_mode must be 'default', 'off' or 'linear'"
                )
            if self.sparseedit_selector_protocol not in {"legacy", "paper"}:
                raise ValueError(
                    "sparseedit_selector_protocol must be 'legacy' or 'paper'"
                )
            if self.sparseedit_selector_protocol == "paper":
                if self.sparseedit_paper_mode not in {
                    "forward",
                    "reverse",
                    "constant",
                    "random",
                }:
                    raise ValueError(
                        "sparseedit_paper_mode must be 'forward', 'reverse', 'constant' or 'random'"
                    )
                if (
                    isinstance(self.sparseedit_paper_random_seed, bool)
                    or not isinstance(self.sparseedit_paper_random_seed, int)
                    or not 0 <= self.sparseedit_paper_random_seed < 2**63
                ):
                    raise ValueError(
                        "sparseedit_paper_random_seed must be a nonnegative 63-bit integer"
                    )
                if not math.isfinite(self.sparseedit_paper_constant_weight) or not (
                    0.0 <= self.sparseedit_paper_constant_weight <= 1.0
                ):
                    raise ValueError(
                        "sparseedit_paper_constant_weight must be finite and in [0, 1]"
                    )
                if (
                    self.sparseedit_avd_enabled
                    and self.sparseedit_online_reuse == "off"
                ):
                    raise ValueError(
                        "the paper selector protocol requires sparseedit AVD disabled"
                    )
            if self.sparseedit_online_reuse not in {
                "off",
                "eager",
                "prealloc",
                "fused",
            }:
                raise ValueError(
                    "sparseedit_online_reuse must be off, eager, prealloc or fused"
                )
            if self.sparseedit_online_reuse != "off" and not (
                self.sparseedit_selector_protocol == "paper"
                and self.sparseedit_avd_enabled
            ):
                raise ValueError(
                    "online reuse requires the paper selector and AVD enabled"
                )
            if (
                not math.isfinite(self.sparseedit_avd_cache_threshold)
                or self.sparseedit_avd_cache_threshold < 0
            ):
                raise ValueError(
                    "sparseedit_avd_cache_threshold must be finite and non-negative"
                )
            if self.sparseedit_avd_max_reuse < 0:
                raise ValueError("sparseedit_avd_max_reuse must be non-negative")
            if (
                not 0
                <= self.sparseedit_avd_force_final_steps
                <= self.num_inference_steps
            ):
                raise ValueError(
                    "sparseedit_avd_force_final_steps is outside the schedule"
                )
            gamma = self.sparseedit_avd_gamma
            if self.sparseedit_avd_enabled and gamma is None:
                raise ValueError(
                    "FLUX SparseEdit AVD requires explicit model-specific gamma"
                )
            if gamma is not None and (
                len(gamma) != self.num_inference_steps - 1
                or not all(math.isfinite(g) for g in gamma)
            ):
                raise ValueError("sparseedit_avd_gamma must contain 27 finite values")
            if self.sparseedit_collect_gamma and (
                self.sparseedit_ratio != 1 or self.sparseedit_avd_enabled
            ):
                raise ValueError("gamma calibration requires ratio=1 and AVD disabled")
            if self.sparseedit_direct_ops not in {"off", "qkv", "eager"}:
                raise ValueError("unsupported sparseedit_direct_ops")
            if self.sparseedit_direct_ops != "off" and (
                self.sparseedit_runtime_variant != "baseline"
                or self.sparseedit_selector_protocol != "paper"
            ):
                raise ValueError(
                    "sparseedit_direct_ops requires the paper baseline runtime"
                )
            supported_variants = {
                "baseline",
                "inplace",
                "transformed",
                "fused",
                "graph",
                "graph_reuse",
            }
            if self.sparseedit_runtime_variant not in supported_variants:
                raise ValueError(
                    "sparseedit_runtime_variant must be one of "
                    + ", ".join(sorted(supported_variants))
                )
            if self.sparseedit_prewarm_boundary not in {
                "sparse",
                "warmup_refresh",
                "startup",
            }:
                raise ValueError("unsupported prewarm boundary")
            menu = self.sparseedit_prewarm_ratios
            if menu:
                if (
                    len(menu) > self.sparseedit_template_capacity
                    or len(set(menu)) != len(menu)
                    or any(not math.isfinite(r) or not 0 < r < 1 for r in menu)
                    or not any(abs(r - self.sparseedit_ratio) < 1e-8 for r in menu)
                ):
                    raise ValueError(
                        "prewarm menu needs valid unique ratios, current ratio and sufficient capacity"
                    )
                if (
                    self.sparseedit_runtime_variant != "graph_reuse"
                    or self.sparseedit_cfg_execution != "serial"
                    or self.sparseedit_cache_layout == "raw"
                    or not self.sparseedit_layer_fusion
                    or not self.sparseedit_stage_fusion
                    or not self.sparseedit_share_positions
                    or not self.sparseedit_share_joint
                    or not self.sparseedit_request_position_reuse
                    or self.sparseedit_condition_plan_mode != "reuse"
                    or not self.sparseedit_condition_bank_shared
                    or self.sparseedit_graph_fallback
                ):
                    raise ValueError(
                        "prewarm requires rotated serial graph with shared fused plans and exact bank"
                    )
            if (
                isinstance(self.sparseedit_template_capacity, bool)
                or not 1 <= self.sparseedit_template_capacity <= 3
            ):
                raise ValueError("sparseedit_template_capacity must be in [1, 3]")
            if (
                self.sparseedit_template_capacity > 1
                and self.sparseedit_cfg_execution != "serial"
            ):
                from .cfg_execution import paper_joint_cfg_enabled

                if self.sparseedit_cfg_execution != "budget" or not paper_joint_cfg_enabled():
                    raise ValueError("multi-template shared arena requires serial CFG or the Track A online Joint-CFG protocol")
            if self.sparseedit_cache_layout not in {"auto", "raw", "rotated"}:
                raise ValueError("sparseedit_cache_layout must be auto, raw or rotated")
            if (
                self.sparseedit_cache_layout != "auto"
                and self.sparseedit_runtime_variant == "baseline"
            ):
                raise ValueError(
                    "sparseedit_cache_layout requires a specialized runtime"
                )
            if (
                self.sparseedit_runtime_variant != "baseline"
                and self.sparseedit_selector_protocol != "paper"
            ):
                raise ValueError(
                    "specialized sparseedit runtime variants require sparseedit_selector_protocol='paper'"
                )
            if (
                self.sparseedit_stage_fusion
                and self.sparseedit_runtime_variant == "baseline"
            ):
                raise ValueError(
                    "sparseedit_stage_fusion requires a specialized runtime variant"
                )
            if (
                self.sparseedit_layer_fusion
                and self.sparseedit_runtime_variant == "baseline"
            ):
                raise ValueError(
                    "sparseedit_layer_fusion requires a specialized runtime variant"
                )
            if not 0 <= self.sparseedit_profile_step < self.num_inference_steps:
                raise ValueError("sparseedit_profile_step is outside the schedule")
            if (
                self.sparseedit_profile_directory
                and self.sparseedit_runtime_variant in {"graph", "graph_reuse"}
            ):
                raise ValueError("forward profiling requires eager execution")
            if self.sparseedit_cfg_execution not in {"serial", "batched", "auto", "budget"}:
                raise ValueError("unsupported sparseedit_cfg_execution")
            if self.sparseedit_reference_kv_int8 and (
                self.sparseedit_runtime_variant != "graph_reuse"
                or self.sparseedit_cfg_execution != "serial"
                or self.sparseedit_cache_layout == "raw"
                or self.sparseedit_prewarm_ratios
                or self.sparseedit_ratio == 1
            ):
                raise ValueError("reference INT8 KV requires rotated graph_reuse and serial CFG")
            if self.sparseedit_cfg_min_query_rows < 1:
                raise ValueError("sparseedit_cfg_min_query_rows must be positive")
            if self.sparseedit_cfg_execution != "serial" and not (
                self.sparseedit_runtime_variant in {"transformed", "graph_reuse"}
                and self.sparseedit_layer_fusion
                and self.sparseedit_fuse_elementwise
                and self.sparseedit_request_position_reuse
            ):
                raise ValueError(
                    "CFG batching requires transformed/graph_reuse, fused layers and request positions"
                )
            if (
                self.sparseedit_condition_prepare_graph
                and self.sparseedit_condition_plan_mode
                not in {"warmup_serial", "warmup_overlap"}
            ):
                raise ValueError(
                    "sparseedit_condition_prepare_graph requires warmup_serial or warmup_overlap"
                )
            if self.sparseedit_condition_plan_mode not in {
                "off",
                "serial",
                "cfg",
                "schedule",
                "reuse",
                "warmup_overlap",
                "warmup_serial",
            }:
                raise ValueError("unsupported sparseedit_condition_plan_mode")
            if self.sparseedit_condition_plan_mode != "off" and not (
                self.sparseedit_layer_fusion and self.sparseedit_fuse_elementwise
            ):
                raise ValueError(
                    "condition projection plan requires fused layer modulation"
                )
            if self.sparseedit_condition_plan_mode in {
                "warmup_overlap",
                "warmup_serial",
            } and (
                self.sparseedit_ratio == 1
                or self.sparseedit_warmup_steps < 2
                or self.sparseedit_cfg_execution != "serial"
                or not self.sparseedit_request_position_reuse
            ):
                raise ValueError(
                    f"{self.sparseedit_condition_plan_mode} requires sparse execution, "
                    "warmup, request positions and serial CFG"
                )
            if self.sparseedit_share_condition and not (
                self.sparseedit_layer_fusion and self.sparseedit_fuse_elementwise
            ):
                raise ValueError("condition sharing requires fused layer modulation")
            if self.sparseedit_request_position_reuse and not (
                self.sparseedit_layer_fusion and self.sparseedit_share_positions
            ):
                raise ValueError(
                    "request position reuse requires layer-level position sharing"
                )
            for name in ("sparseedit_replay_trace", "sparseedit_audit_dir"):
                value = getattr(self, name)
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"{name} must be a filesystem path string")
            if self.sparseedit_graph_warmup_trials < 1:
                raise ValueError("sparseedit_graph_warmup_trials must be at least 1")
            if self.sparseedit_defer_diagnostics and self.sparseedit_collect_audit:
                raise ValueError(
                    "deferred diagnostics cannot also collect per-step audit tensors"
                )
        if self.method == "toca":
            if not 0.0 <= self.toca_fresh_ratio <= 1.0:
                raise ValueError("toca_fresh_ratio must be in [0, 1]")
            if self.toca_fresh_threshold <= 0:
                raise ValueError("toca_fresh_threshold must be positive")
            supported_toca_schedulers = {
                "constant",
                "linear",
                "exp",
                "linear-mode",
                "layerwise",
                "linear-layerwise",
                "ToCa-ddim50",
                "ToCa-ddpm250",
            }
            if self.toca_ratio_scheduler not in supported_toca_schedulers:
                raise ValueError(
                    f"unsupported ToCa ratio scheduler: {self.toca_ratio_scheduler}"
                )
            if self.toca_score_mode not in {
                "cache_frequency",
                "latent_diff_cache_frequency",
            }:
                raise ValueError(
                    "FLUX ToCa currently supports cache_frequency and latent_diff_cache_frequency scores"
                )
            if self.toca_token_scope != "generated":
                raise ValueError(
                    "FLUX Kontext ToCa currently requires toca_token_scope='generated'"
                )
        if self.method == "ras":
            if not 0.0 < self.ras_target_ratio <= 1.0:
                raise ValueError("ras_target_ratio must be in (0, 1]")
            if self.ras_ratio_schedule not in {"linear", "constant"}:
                raise ValueError(
                    f"unsupported RAS ratio schedule: {self.ras_ratio_schedule}"
                )
            if self.ras_step_ratios is not None:
                if len(self.ras_step_ratios) != self.num_inference_steps:
                    raise ValueError(
                        "ras_step_ratios must contain one value per denoising step"
                    )
                if any(not 0.0 < ratio <= 1.0 for ratio in self.ras_step_ratios):
                    raise ValueError("ras_step_ratios values must be in (0, 1]")
            if self.ras_k < 0.0:
                raise ValueError("ras_k must be non-negative")
            if self.ras_score_direction not in {"high_std", "low_std"}:
                raise ValueError(
                    f"unsupported RAS score direction: {self.ras_score_direction}"
                )
            if self.ras_selection_rounding not in {"ceil", "floor", "round"}:
                raise ValueError(
                    f"unsupported RAS selection rounding: {self.ras_selection_rounding}"
                )
            if not 0 <= self.ras_selective_start_step < self.num_inference_steps:
                raise ValueError(
                    "ras_selective_start_step must be within the denoising schedule"
                )
            if tuple(sorted(set(self.ras_dense_steps))) != self.ras_dense_steps:
                raise ValueError("ras_dense_steps must be sorted and unique")
            if any(
                step < 1 or step > self.num_inference_steps
                for step in self.ras_dense_steps
            ):
                raise ValueError(
                    "ras_dense_steps use 1-based values within the denoising schedule"
                )
        if self.method == "regione":
            if self.regione_warmup_step < 1:
                raise ValueError("regione_warmup_step must be at least 1")
            if self.regione_post_step < 0:
                raise ValueError("regione_post_step must be non-negative")
            if (
                tuple(sorted(set(self.regione_refresh_steps)))
                != self.regione_refresh_steps
            ):
                raise ValueError("regione_refresh_steps must be sorted and unique")
            if not self.regione_refresh_steps:
                raise ValueError("regione_refresh_steps must contain at least one step")
            if min(self.regione_refresh_steps) <= self.regione_warmup_step + 1:
                raise ValueError(
                    "the first RegionE refresh step must be later than warmup_step + 1"
                )
            if (
                max(self.regione_refresh_steps)
                > self.num_inference_steps - self.regione_post_step - 1
            ):
                raise ValueError(
                    "RegionE refresh steps must finish before the smooth stage"
                )
            if not 0.0 <= self.regione_partition_threshold <= 1.0:
                raise ValueError("regione_partition_threshold must be in [0, 1]")
            if self.regione_cache_threshold < 0.0:
                raise ValueError("regione_cache_threshold must be non-negative")
            if self.regione_enable_avd and self.num_inference_steps != 28:
                raise ValueError(
                    "the migrated FLUX RegionE AVD coefficients cover the 28-step schedule"
                )
        if self.method == "spotedit":
            if self.spotedit_profile not in {"source", "source_legacy", "paper"}:
                raise ValueError(
                    f"unsupported SpotEdit profile: {self.spotedit_profile}"
                )
            if self.spotedit_threshold < 0.0:
                raise ValueError("spotedit_threshold must be non-negative")
            if self.spotedit_judge_method not in {
                "L4",
                "cosine",
                "LPIPS",
                "LPIPS_kmeans",
            }:
                raise ValueError(
                    f"unsupported SpotEdit judge method: {self.spotedit_judge_method}"
                )
            if self.spotedit_initial_steps < 1:
                raise ValueError("spotedit_initial_steps must be at least 1")
            normalized_reset_steps = _tuple_of_strict_ints(
                self.spotedit_reset_steps,
                default=(13, 22),
                name="spotedit_reset_steps",
            )
            object.__setattr__(self, "spotedit_reset_steps", normalized_reset_steps)
            if (
                tuple(sorted(set(self.spotedit_reset_steps)))
                != self.spotedit_reset_steps
            ):
                raise ValueError("spotedit_reset_steps must be sorted and unique")
            if any(
                step < 0 or step >= self.num_inference_steps
                for step in self.spotedit_reset_steps
            ):
                raise ValueError(
                    "spotedit_reset_steps must use 0-based indices within the denoising schedule"
                )
            if self.spotedit_dilation_radius < 0:
                raise ValueError("spotedit_dilation_radius must be non-negative")
            if self.spotedit_reuse_mode not in {"velocity", "overwrite"}:
                raise ValueError(
                    f"unsupported SpotEdit reuse mode: {self.spotedit_reuse_mode}"
                )

    @classmethod
    def from_extra_args(
        cls,
        extra_args: dict[str, Any] | None,
        *,
        num_inference_steps: int,
    ) -> FluxKontextAccelerationConfig:
        values = dict(extra_args or {})
        method = str(
            values.get(
                "flux_acceleration_method", values.get("acceleration_method", "dense")
            )
        ).lower()
        aliases = {
            "none": "dense",
            "teacache": "tea_cache",
            "tea-cache": "tea_cache",
            "cachedit": "cache_dit",
            "cache-dit": "cache_dit",
            "sparse_edit": "sparseedit",
            "sparse-edit": "sparseedit",
        }
        method = aliases.get(method, method)

        profile = str(values.get("spotedit_profile", "source"))
        source_defaults = {
            "spotedit_threshold": 0.2,
            "spotedit_judge_method": "LPIPS",
            "spotedit_initial_steps": 4,
            "spotedit_reset_steps": (13, 22),
            "spotedit_dilation_radius": 0,
            "spotedit_reuse_mode": "velocity",
            "spotedit_select_every_step": True,
        }
        spot_defaults = {
            "source": source_defaults,
            "source_legacy": source_defaults,
            "paper": {
                "spotedit_threshold": 0.2,
                "spotedit_judge_method": "LPIPS",
                "spotedit_initial_steps": 4,
                "spotedit_reset_steps": (13, 22),
                "spotedit_dilation_radius": 1,
                "spotedit_reuse_mode": "overwrite",
                "spotedit_select_every_step": True,
            },
        }.get(profile, {})

        def resolved(name: str, default: Any) -> Any:
            return (
                values[name] if name in values and values[name] is not None else default
            )

        return cls(
            method=method,  # type: ignore[arg-type]
            num_inference_steps=num_inference_steps,
            toca_fresh_ratio=float(resolved("toca_fresh_ratio", 0.07)),
            toca_fresh_threshold=int(resolved("toca_fresh_threshold", 4)),
            toca_ratio_scheduler=str(resolved("toca_ratio_scheduler", "ToCa-ddim50")),
            toca_score_mode=str(resolved("toca_score_mode", "cache_frequency")),
            toca_soft_fresh_weight=float(resolved("toca_soft_fresh_weight", 0.25)),
            toca_spatial_bonus_ratio=float(resolved("toca_spatial_bonus_ratio", 0.6)),
            toca_spatial_grid_size=int(resolved("toca_spatial_grid_size", 2)),
            toca_token_scope=str(resolved("toca_token_scope", "generated")),
            ras_k=float(resolved("ras_k", 0.1)),
            ras_target_ratio=float(resolved("ras_target_ratio", 0.25)),
            ras_ratio_schedule=str(resolved("ras_ratio_schedule", "linear")),
            ras_step_ratios=_tuple_of_floats(values.get("ras_step_ratios")),
            ras_selective_start_step=int(resolved("ras_selective_start_step", 4)),
            ras_dense_steps=_tuple_of_ints(
                values.get("ras_dense_steps"), default=(12, 20)
            ),
            ras_score_direction=str(resolved("ras_score_direction", "high_std")),
            ras_selection_rounding=str(resolved("ras_selection_rounding", "ceil")),
            regione_warmup_step=int(resolved("regione_warmup_step", 6)),
            regione_post_step=int(resolved("regione_post_step", 2)),
            regione_refresh_steps=_tuple_of_ints(
                values.get("regione_refresh_steps"), default=(16,)
            ),
            regione_partition_threshold=float(
                resolved("regione_partition_threshold", 0.93)
            ),
            regione_cache_threshold=float(resolved("regione_cache_threshold", 0.04)),
            regione_erosion_dilation=bool(resolved("regione_erosion_dilation", True)),
            regione_enable_avd=bool(resolved("regione_enable_avd", True)),
            spotedit_profile=profile,
            spotedit_threshold=float(
                resolved(
                    "spotedit_threshold", spot_defaults.get("spotedit_threshold", 0.2)
                )
            ),
            spotedit_judge_method=str(
                resolved(
                    "spotedit_judge_method",
                    spot_defaults.get("spotedit_judge_method", "LPIPS"),
                )
            ),
            spotedit_initial_steps=int(
                resolved(
                    "spotedit_initial_steps",
                    spot_defaults.get("spotedit_initial_steps", 4),
                )
            ),
            spotedit_reset_steps=_tuple_of_strict_ints(
                values.get("spotedit_reset_steps"),
                default=spot_defaults.get("spotedit_reset_steps", (13, 22)),
                name="spotedit_reset_steps",
            ),
            spotedit_dilation_radius=int(
                resolved(
                    "spotedit_dilation_radius",
                    spot_defaults.get("spotedit_dilation_radius", 0),
                )
            ),
            spotedit_reuse_mode=str(
                resolved(
                    "spotedit_reuse_mode",
                    spot_defaults.get("spotedit_reuse_mode", "velocity"),
                )
            ),
            spotedit_select_every_step=bool(
                resolved(
                    "spotedit_select_every_step",
                    spot_defaults.get("spotedit_select_every_step", True),
                )
            ),
            sparseedit_ratio=float(resolved("sparseedit_ratio", 0.30)),
            sparseedit_warmup_steps=int(resolved("sparseedit_warmup_steps", 6)),
            sparseedit_refresh_interval=int(
                resolved("sparseedit_refresh_interval", 15)
            ),
            sparseedit_refresh_mode=str(resolved("sparseedit_refresh_mode", "full")),
            sparseedit_warmup_ref_freeze_step=resolved("sparseedit_warmup_ref_freeze_step", None),
            sparseedit_locality_mode=str(
                resolved("sparseedit_locality_mode", "default")
            ),
            sparseedit_selector_protocol=str(
                resolved("sparseedit_selector_protocol", "legacy")
            ),
            sparseedit_paper_mode=str(resolved("sparseedit_paper_mode", "forward")),
            sparseedit_paper_constant_weight=float(
                resolved("sparseedit_paper_constant_weight", 0.0)
            ),
            sparseedit_paper_random_seed=resolved("sparseedit_paper_random_seed", 110),
            sparseedit_avd_enabled=bool(resolved("sparseedit_avd_enabled", False)),
            sparseedit_avd_cache_threshold=float(
                resolved("sparseedit_avd_cache_threshold", 0.010)
            ),
            sparseedit_avd_max_reuse=int(resolved("sparseedit_avd_max_reuse", 1)),
            sparseedit_avd_force_final_steps=int(
                resolved("sparseedit_avd_force_final_steps", 4)
            ),
            sparseedit_avd_gamma=_tuple_of_floats(values.get("sparseedit_avd_gamma")),
            sparseedit_online_reuse=str(resolved("sparseedit_online_reuse", "off")),
            sparseedit_collect_warmup=bool(
                resolved("sparseedit_collect_warmup", False)
            ),
            sparseedit_collect_gamma=bool(resolved("sparseedit_collect_gamma", False)),
            sparseedit_collect_masks=bool(resolved("sparseedit_collect_masks", False)),
            sparseedit_return_metadata=bool(
                resolved("sparseedit_return_metadata", False)
            ),
            sparseedit_runtime_variant=str(
                resolved(
                    "sparseedit_runtime_variant",
                    "graph_reuse"
                    if resolved("sparseedit_online_reuse", "off") != "off"
                    else "baseline",
                )
            ),
            sparseedit_direct_ops=str(resolved("sparseedit_direct_ops", "off")),
            sparseedit_standard_backend=str(resolved("sparseedit_standard_backend", "off")),
            sparseedit_standard_audit=str(resolved("sparseedit_standard_audit", "off")),
            sparseedit_standard_audit_dir=str(resolved("sparseedit_standard_audit_dir", "")),
            sparseedit_cache_layout=str(resolved("sparseedit_cache_layout", "auto")),
            sparseedit_prewarm_boundary=str(
                resolved("sparseedit_prewarm_boundary", "sparse")
            ),
            sparseedit_prewarm_ratios=_tuple_of_floats(
                values.get("sparseedit_prewarm_ratios")
            ),
            sparseedit_template_capacity=int(
                resolved("sparseedit_template_capacity", 1)
            ),
            sparseedit_condition_bank_shared=bool(
                resolved("sparseedit_condition_bank_shared", False)
            ),
            sparseedit_replay_trace=str(resolved("sparseedit_replay_trace", "") or ""),
            sparseedit_audit_dir=str(resolved("sparseedit_audit_dir", "") or ""),
            sparseedit_collect_audit=bool(resolved("sparseedit_collect_audit", False)),
            sparseedit_collect_trace=bool(resolved("sparseedit_collect_trace", False)),
            sparseedit_defer_diagnostics=bool(
                resolved("sparseedit_defer_diagnostics", False)
            ),
            sparseedit_graph_fallback=bool(
                resolved("sparseedit_graph_fallback", False)
            ),
            sparseedit_graph_warmup_trials=int(
                resolved("sparseedit_graph_warmup_trials", 2)
            ),
            sparseedit_stage_fusion=bool(resolved("sparseedit_stage_fusion", False)),
            sparseedit_layer_fusion=bool(resolved("sparseedit_layer_fusion", False)),
            sparseedit_fuse_elementwise=bool(
                resolved("sparseedit_fuse_elementwise", True)
            ),
            sparseedit_share_positions=bool(
                resolved("sparseedit_share_positions", True)
            ),
            sparseedit_share_joint=bool(resolved("sparseedit_share_joint", True)),
            sparseedit_cfg_execution=str(
                resolved("sparseedit_cfg_execution", "serial")
            ),
            sparseedit_reference_kv_int8=bool(resolved("sparseedit_reference_kv_int8", False)),
            sparseedit_cfg_min_query_rows=int(
                resolved("sparseedit_cfg_min_query_rows", 1200)
            ),
            sparseedit_condition_plan_mode=str(
                resolved("sparseedit_condition_plan_mode", "off")
            ),
            sparseedit_condition_prepare_graph=bool(
                resolved("sparseedit_condition_prepare_graph", False)
            ),
            sparseedit_share_condition=bool(
                resolved("sparseedit_share_condition", False)
            ),
            sparseedit_request_position_reuse=bool(
                resolved("sparseedit_request_position_reuse", False)
            ),
            sparseedit_profile_directory=str(
                resolved("sparseedit_profile_directory", "") or ""
            ),
            sparseedit_profile_step=int(resolved("sparseedit_profile_step", 10)),
        )

    @property
    def uses_pipeline_cache_backend(self) -> bool:
        return self.method in {"tea_cache", "cache_dit"}

    @property
    def uses_custom_transformer_path(self) -> bool:
        return self.method in {"toca", "ras", "regione", "spotedit", "sparseedit"}

    @property
    def uses_specialized_runtime(self) -> bool:
        return (
            self.method == "sparseedit"
            and self.sparseedit_runtime_variant != "baseline"
        )

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "method": self.method,
            "num_inference_steps": self.num_inference_steps,
        }
        prefix = f"{self.method}_"
        for name, value in self.__dict__.items():
            if name.startswith(prefix):
                metadata[name] = list(value) if isinstance(value, tuple) else value
        return metadata


@dataclass
class FluxSparseKVCache:
    """Per-branch normalized image-key and value cache for both FLUX streams."""

    layers: dict[tuple[FluxStream, int], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )

    def store_full(
        self,
        stream: FluxStream,
        layer_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if key.shape != value.shape:
            raise ValueError(
                f"FLUX sparse key/value shape mismatch: {key.shape} and {value.shape}"
            )
        self.layers[(stream, layer_index)] = (
            key.detach().contiguous().clone(),
            value.detach().contiguous().clone(),
        )

    def update_active(
        self,
        stream: FluxStream,
        layer_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
        image_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self.layers.get((stream, layer_index))
        if cached is None:
            raise RuntimeError(
                f"FLUX sparse {stream} layer {layer_index} was used before a full K/V cache was stored"
            )
        cached_key, cached_value = cached
        if image_indices.ndim != 2 or image_indices.shape[0] != key.shape[0]:
            raise ValueError(
                "FLUX sparse image_indices must have shape [batch, active_image_tokens]"
            )
        if (
            key.shape[:2] != image_indices.shape
            or value.shape[:2] != image_indices.shape
        ):
            raise ValueError("FLUX sparse active K/V length must match image_indices")
        updated_key = cached_key.clone()
        updated_value = cached_value.clone()
        scatter_index = (
            image_indices.to(device=key.device, dtype=torch.long)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        scatter_index = scatter_index.expand(-1, -1, key.shape[-2], key.shape[-1])
        updated_key.scatter_(1, scatter_index, key)
        updated_value.scatter_(1, scatter_index, value)
        self.layers[(stream, layer_index)] = (
            updated_key.detach(),
            updated_value.detach(),
        )
        return updated_key, updated_value

    def spotedit_fuse(
        self,
        stream: FluxStream,
        layer_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
        image_indices: torch.Tensor,
        *,
        generated_token_count: int,
        reference_token_count: int,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if generated_token_count != reference_token_count:
            raise ValueError(
                "SpotEdit requires token-aligned generated and reference images"
            )
        updated_key, updated_value = self.update_active(
            stream, layer_index, key, value, image_indices
        )
        generated_key = updated_key[:, :generated_token_count]
        generated_value = updated_value[:, :generated_token_count]
        reference_key = updated_key[:, generated_token_count:]
        reference_value = updated_value[:, generated_token_count:]
        active = image_indices.to(device=key.device, dtype=torch.long)
        if bool((active >= generated_token_count).any()):
            raise ValueError("SpotEdit sparse steps update generated tokens only")
        reference_seed_key = reference_key.clone()
        reference_seed_value = reference_value.clone()
        scatter_index = (
            active.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, key.shape[-2], key.shape[-1])
        )
        reference_seed_key.scatter_(1, scatter_index, key)
        reference_seed_value.scatter_(1, scatter_index, value)
        fused_generated_key = (1.0 - alpha) * reference_seed_key + alpha * generated_key
        fused_generated_value = (
            1.0 - alpha
        ) * reference_seed_value + alpha * generated_value
        return (
            torch.cat([fused_generated_key, reference_key], dim=1),
            torch.cat([fused_generated_value, reference_value], dim=1),
        )

    def clear(self) -> None:
        self.layers.clear()


@dataclass
class FluxSparseAttentionContext:
    """Attention metadata for one transformer call and one CFG branch."""

    method: FluxSparseMethod
    kv_cache: FluxSparseKVCache
    kv_action: FluxSparseKVAction
    image_indices: torch.Tensor
    full_img_ids: torch.Tensor
    generated_token_count: int
    reference_token_count: int
    spotedit_alpha: float = 0.0
    direct_ops: str = "off"
    # Opt-in specialized execution: an owned cache from
    # ``specialized_cache``. ``None`` preserves the original runtime.
    specialized_cache: Any | None = None
    layer_fusion: bool = False
    layer_elementwise: bool = True
    layer_positions: bool = True
    layer_joint: bool = True
    request_position_view: Any | None = None
    condition_view: Any | None = None
    layer_share_condition: bool = False
    layer_condition: torch.Tensor | None = field(default=None, init=False)
    stream: FluxStream | None = field(default=None, init=False)
    layer_index: int | None = field(default=None, init=False)
    full_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = field(
        default=None, init=False
    )
    layer_kv_positions: torch.Tensor | None = field(default=None, init=False)
    layer_image_rotary: tuple[torch.Tensor, torch.Tensor] | None = field(
        default=None, init=False
    )

    @property
    def full_image_token_count(self) -> int:
        return self.generated_token_count + self.reference_token_count

    def set_layer(self, stream: FluxStream, layer_index: int) -> None:
        self.stream = stream
        self.layer_index = layer_index

    def resolve_layer(self) -> tuple[FluxStream, int]:
        if self.stream is None or self.layer_index is None:
            raise RuntimeError(
                "FLUX sparse attention context did not receive a transformer layer index"
            )
        return self.stream, self.layer_index


def _scheduled_toca_ratio(
    base_ratio: float,
    *,
    step: int,
    layer: int,
    module: str,
    num_steps: int,
    last_layer_index: int,
    scheduler: str,
    fresh_threshold: int,
) -> float:
    weight = 0.9
    if scheduler == "constant":
        ratio = base_ratio
    elif scheduler == "linear":
        ratio = base_ratio * (1 + weight - 2 * weight * step / max(1, num_steps))
    elif scheduler == "exp":
        ratio = base_ratio * weight ** (step / max(1, num_steps))
    elif scheduler == "linear-mode":
        mode = (step % fresh_threshold) / fresh_threshold - 0.5
        ratio = base_ratio * (
            1 + weight - 2 * weight * step / max(1, num_steps) + 0.1 * mode
        )
    elif scheduler == "layerwise":
        ratio = base_ratio * (
            1 + weight - 2 * weight * layer / max(1, last_layer_index)
        )
    elif scheduler in {"linear-layerwise", "ToCa-ddim50", "ToCa-ddpm250"}:
        step_weight = 2.0 if scheduler == "ToCa-ddim50" else 0.4
        layer_weight = -0.2 if scheduler == "ToCa-ddim50" else 0.8
        module_weight = 2.5
        module_time_weight = 0.6
        module_factor = (
            1 - (1 - module_time_weight) * module_weight
            if module == "attn"
            else 1 + module_time_weight * module_weight
        )
        step_factor = 1 + step_weight - 2 * step_weight * step / max(1, num_steps)
        layer_factor = (
            1 + layer_weight - 2 * layer_weight * layer / max(1, last_layer_index)
        )
        ratio = base_ratio * step_factor * layer_factor * module_factor
    else:
        raise ValueError(f"unrecognized ToCa ratio scheduler: {scheduler}")
    return float(max(0.0, min(1.0, ratio)))


@dataclass
class FluxToCaState:
    """Feature caches and token-age scores for one FLUX CFG branch."""

    config: FluxKontextAccelerationConfig
    num_layers: int
    generated_token_count: int
    reference_token_count: int
    grid_height: int
    grid_width: int
    attention_outputs: dict[
        tuple[FluxStream, int], tuple[torch.Tensor, torch.Tensor | None]
    ] = field(default_factory=dict)
    ffn_outputs: dict[tuple[FluxStream, int], torch.Tensor] = field(
        default_factory=dict
    )
    projected_outputs: dict[tuple[FluxStream, int], torch.Tensor] = field(
        default_factory=dict
    )
    cache_frequency: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)
    previous_tokens: dict[tuple[int, str], torch.Tensor] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def full_image_token_count(self) -> int:
        return self.generated_token_count + self.reference_token_count

    def is_full_step(self, step_index: int) -> bool:
        if step_index == self.config.num_inference_steps - 1:
            return True
        base = self.config.toca_fresh_threshold
        step_weight = 0.0 if self.config.toca_fresh_ratio == 0 else 0.4
        factor = (
            1
            + step_weight
            - 2 * step_weight * step_index / self.config.num_inference_steps
        )
        threshold = max(1, int(round(base / factor)))  # noqa: RUF046
        if (
            step_index
            in range(
                int(self.config.num_inference_steps * 0.2),
                int(self.config.num_inference_steps * 0.4),
            )
            and self.config.toca_fresh_ratio != 0
        ):
            threshold = 2
        return step_index % threshold == 0

    def has_layer(self, stream: FluxStream, layer_index: int) -> bool:
        key = (stream, layer_index)
        if stream == "dual":
            return key in self.attention_outputs and key in self.ffn_outputs
        return key in self.attention_outputs and key in self.projected_outputs

    def reset_cache_frequency(
        self,
        global_layer_index: int,
        module: str,
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.cache_frequency[(global_layer_index, module)] = torch.zeros(
            batch_size,
            self.generated_token_count,
            dtype=torch.long,
            device=device,
        )

    def select_generated_indices(
        self,
        tokens: torch.Tensor,
        *,
        step_index: int,
        global_layer_index: int,
        module: str,
    ) -> tuple[torch.Tensor, float]:
        generated = tokens[:, : self.generated_token_count]
        key = (global_layer_index, module)
        frequency = self.cache_frequency.get(key)
        if (
            frequency is None
            or frequency.shape != generated.shape[:2]
            or frequency.device != generated.device
        ):
            frequency = torch.zeros(
                generated.shape[:2], device=generated.device, dtype=torch.long
            )
        if self.config.toca_score_mode == "latent_diff_cache_frequency":
            previous = self.previous_tokens.get(key)
            score = (
                torch.zeros(
                    generated.shape[:2], device=generated.device, dtype=generated.dtype
                )
                if previous is None
                else torch.linalg.vector_norm(
                    generated - previous.to(generated), dim=-1
                )
            )
            self.previous_tokens[key] = generated.detach().clone()
        else:
            score = torch.zeros(
                generated.shape[:2], device=generated.device, dtype=generated.dtype
            )
        score = (
            score
            + self.config.toca_soft_fresh_weight
            * frequency.to(score.dtype)
            / self.config.toca_fresh_threshold
        )
        if (
            self.config.toca_spatial_bonus_ratio > 0
            and self.grid_height * self.grid_width == score.shape[1]
        ):
            score = _apply_spatial_bonus(
                score,
                grid_height=self.grid_height,
                grid_width=self.grid_width,
                grid_size=self.config.toca_spatial_grid_size,
                bonus_ratio=self.config.toca_spatial_bonus_ratio,
            )
        ratio = _scheduled_toca_ratio(
            self.config.toca_fresh_ratio,
            step=step_index,
            layer=global_layer_index,
            module=module,
            num_steps=self.config.num_inference_steps,
            last_layer_index=max(1, self.num_layers - 1),
            scheduler=self.config.toca_ratio_scheduler,
            fresh_threshold=self.config.toca_fresh_threshold,
        )
        count = min(
            self.generated_token_count, max(0, int(ratio * self.generated_token_count))
        )
        indices = torch.argsort(score, dim=-1, descending=True, stable=True)[:, :count]
        frequency.add_(1)
        if indices.numel() > 0:
            frequency.scatter_(
                1, indices, torch.zeros_like(indices, dtype=frequency.dtype)
            )
        self.cache_frequency[key] = frequency
        return indices, ratio


def _apply_spatial_bonus(
    score: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    grid_size: int,
    bonus_ratio: float,
) -> torch.Tensor:
    if grid_size <= 0 or grid_height % grid_size or grid_width % grid_size:
        return score
    batch_size = score.shape[0]
    blocks = score.view(
        batch_size,
        grid_height // grid_size,
        grid_size,
        grid_width // grid_size,
        grid_size,
    )
    blocks = (
        blocks.permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch_size, -1, grid_size * grid_size)
    )
    maxima, indices = blocks.max(dim=-1, keepdim=True)
    bonus = torch.zeros_like(blocks).scatter(-1, indices, maxima * bonus_ratio)
    blocks = blocks + bonus
    blocks = blocks.view(
        batch_size,
        grid_height // grid_size,
        grid_width // grid_size,
        grid_size,
        grid_size,
    )
    return blocks.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, -1)


@dataclass(frozen=True)
class FluxRASDecision:
    step_index: int
    active_indices: torch.Tensor
    requested_ratio: float
    dense: bool
    dense_reason: str | None
    scores: torch.Tensor

    @property
    def active_count(self) -> int:
        return int(self.active_indices.shape[1])


@dataclass
class FluxRASState:
    config: FluxKontextAccelerationConfig
    generated_token_count: int
    previous_full_noise: torch.Tensor | None = None
    drop_count: torch.Tensor = field(init=False)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.drop_count = torch.zeros(self.generated_token_count, dtype=torch.long)

    def ratio_for_step(self, step_index: int) -> float:
        if (
            step_index < self.config.ras_selective_start_step
            or step_index + 1 in self.config.ras_dense_steps
        ):
            return 1.0
        if self.config.ras_step_ratios is not None:
            return float(self.config.ras_step_ratios[step_index])
        if self.config.ras_ratio_schedule == "constant":
            return self.config.ras_target_ratio
        span = (
            self.config.num_inference_steps - 1 - self.config.ras_selective_start_step
        )
        if span <= 0:
            return self.config.ras_target_ratio
        progress = (step_index - self.config.ras_selective_start_step + 1) / (span + 1)
        return 1.0 + (self.config.ras_target_ratio - 1.0) * progress

    def decision(self, step_index: int, *, device: torch.device) -> FluxRASDecision:
        ratio = self.ratio_for_step(step_index)
        dense_reason = None
        if step_index < self.config.ras_selective_start_step:
            dense_reason = "initial_dense"
        elif step_index + 1 in self.config.ras_dense_steps:
            dense_reason = "periodic_dense"
        elif ratio >= 1.0:
            dense_reason = "ratio_dense"
        dense = dense_reason is not None
        noise_std = self._noise_std(device)
        age = torch.exp(self.drop_count.to(device=device).float() * self.config.ras_k)
        scores = (
            noise_std * age
            if self.config.ras_score_direction == "high_std"
            else age / noise_std.clamp_min(1e-6)
        )
        if dense:
            indices = torch.arange(
                self.generated_token_count, device=device, dtype=torch.long
            ).unsqueeze(0)
        else:
            if self.previous_full_noise is None:
                raise RuntimeError(
                    "RAS sparse selection requires a preceding full-noise reconstruction"
                )
            raw = self.generated_token_count * ratio
            if self.config.ras_selection_rounding == "ceil":
                count = math.ceil(raw)
            elif self.config.ras_selection_rounding == "floor":
                count = math.floor(raw)
            else:
                count = round(raw)
            count = min(self.generated_token_count, max(1, count))
            indices = torch.argsort(scores, descending=True, stable=True)[
                :count
            ].unsqueeze(0)
        return FluxRASDecision(
            step_index, indices, ratio, dense, dense_reason, scores.unsqueeze(0)
        )

    def reconstruct(
        self, decision: FluxRASDecision, compact_noise: torch.Tensor
    ) -> torch.Tensor:
        if decision.dense:
            full = compact_noise.clone()
        else:
            if self.previous_full_noise is None:
                raise RuntimeError(
                    "RAS cannot reconstruct sparse noise before a dense step"
                )
            full = self.previous_full_noise.clone()
            full[:, decision.active_indices.squeeze(0)] = compact_noise
        self.previous_full_noise = full.detach().clone()
        if decision.dense:
            self.drop_count.zero_()
        else:
            active = decision.active_indices.squeeze(0).to(
                device=self.drop_count.device
            )
            self.drop_count.add_(1)
            self.drop_count[active] = 0
        return full

    def _noise_std(self, device: torch.device) -> torch.Tensor:
        if self.previous_full_noise is None:
            return torch.zeros(self.generated_token_count, device=device)
        noise = self.previous_full_noise.float()
        if noise.shape[-1] % 4:
            raise RuntimeError(
                "FLUX packed velocity channels must be divisible by four"
            )
        latent_channels = noise.shape[-1] // 4
        return (
            noise.unflatten(-1, (latent_channels, 4))
            .std(dim=-2, correction=0)
            .mean(dim=-1)
            .squeeze(0)
        )


@dataclass
class FluxRegionEState:
    config: FluxKontextAccelerationConfig
    generated_token_count: int
    grid_height: int
    grid_width: int
    edited_indices: torch.Tensor | None = None
    unedited_indices: torch.Tensor | None = None
    unedited_latents: torch.Tensor | None = None
    velocity_cache: torch.Tensor | None = None
    avd_accumulate: float = 1.0
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def partition_step_index(self) -> int:
        return self.config.regione_warmup_step - 1

    @property
    def smooth_start_step_index(self) -> int:
        return self.config.num_inference_steps - self.config.regione_post_step

    @property
    def forced_step_indices(self) -> tuple[int, ...]:
        return tuple(step - 1 for step in self.config.regione_refresh_steps)

    @property
    def merge_step_indices(self) -> tuple[int, ...]:
        return (*self.forced_step_indices, self.smooth_start_step_index)

    def is_local_step(self, step_index: int) -> bool:
        return (
            self.config.regione_warmup_step <= step_index < self.smooth_start_step_index
            and (step_index not in self.forced_step_indices)
        )

    def is_full_cache_step(self, step_index: int) -> bool:
        return (
            step_index == self.partition_step_index
            or step_index in self.forced_step_indices
        )

    def set_partition(
        self, estimated_x0: torch.Tensor, reference_latents: torch.Tensor
    ) -> None:
        edited_mask = select_regione_mask(
            estimated_x0,
            reference_latents,
            threshold=self.config.regione_partition_threshold,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            erosion_dilation=self.config.regione_erosion_dilation,
        )
        all_indices = torch.arange(
            self.generated_token_count, device=estimated_x0.device
        ).unsqueeze(0)
        self.edited_indices = all_indices[edited_mask].reshape(1, -1)
        self.unedited_indices = all_indices[~edited_mask].reshape(1, -1)
        if self.edited_indices.shape[1] == 0:
            self.edited_indices = all_indices
            self.unedited_indices = all_indices[:, :0]

    def avd_decision(
        self,
        step_index: int,
        timestep: torch.Tensor,
        previous_timestep: torch.Tensor | None,
    ) -> tuple[bool, float | None, str]:
        if not self.config.regione_enable_avd:
            self.avd_accumulate = 1.0
            return True, None, "avd_disabled"
        if (
            previous_timestep is None
            or step_index <= self.config.regione_warmup_step
            or not self.is_local_step(step_index)
        ):
            self.avd_accumulate = 1.0
            return True, None, "forced_compute"
        current_t = float(timestep.detach().float().item())
        previous_t = float(previous_timestep.detach().float().item())
        gamma = float(
            torch.tensor(
                FLUX_REGIONE_GAMMA_28[step_index - 1], dtype=torch.float16
            ).item()
        )
        ratio = gamma * (1.0 + (current_t - previous_t) / 1000.0)
        if ratio >= 1.0:
            self.avd_accumulate = 1.0
            return True, ratio, "ratio_ge_one"
        candidate = self.avd_accumulate * ratio
        if 1.0 - candidate > self.config.regione_cache_threshold:
            self.avd_accumulate = 1.0
            return True, ratio, "error_threshold"
        self.avd_accumulate = candidate
        return False, ratio, "cache_hit"

    def cached_velocity(self, ratio: float) -> torch.Tensor:
        if self.velocity_cache is None or self.edited_indices is None:
            raise RuntimeError("RegionE velocity cache is unavailable")
        velocity = self.velocity_cache
        if velocity.shape[1] != self.edited_indices.shape[1]:
            velocity = gather_tokens(velocity, self.edited_indices)
        return velocity * ratio

    def update_velocity(self, velocity: torch.Tensor) -> None:
        self.velocity_cache = velocity.detach().clone()

    def merge(self, edited_latents: torch.Tensor) -> torch.Tensor:
        if (
            self.edited_indices is None
            or self.unedited_indices is None
            or self.unedited_latents is None
        ):
            raise RuntimeError("RegionE merge state is incomplete")
        merged = edited_latents.new_zeros(
            edited_latents.shape[0],
            self.generated_token_count,
            edited_latents.shape[-1],
        )
        scatter_tokens_(merged, self.edited_indices, edited_latents)
        scatter_tokens_(merged, self.unedited_indices, self.unedited_latents)
        return merged


def select_regione_mask(
    estimated_x0: torch.Tensor,
    reference_latents: torch.Tensor,
    *,
    threshold: float,
    grid_height: int,
    grid_width: int,
    erosion_dilation: bool,
) -> torch.Tensor:
    if estimated_x0.shape != reference_latents.shape:
        raise ValueError(
            "RegionE requires token-aligned predicted and reference latents"
        )
    similarity = F.cosine_similarity(estimated_x0, reference_latents, dim=-1)
    edited = similarity <= threshold
    if erosion_dilation:
        mask = edited.float().view(1, 1, grid_height, grid_width)
        erosion_kernel = torch.zeros(1, 1, 3, 3, device=mask.device)
        erosion_kernel[0, 0, 1, :] = 1
        erosion_kernel[0, 0, :, 1] = 1
        eroded = F.conv2d(mask, erosion_kernel, padding=1) == erosion_kernel.sum()
        edited = (
            F.conv2d(
                eroded.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2
            )
            > 0
        ).view(1, -1)
    return edited


@dataclass(frozen=True)
class FluxSpotEditConfig:
    profile: str
    threshold: float
    judge_method: str
    initial_steps: int
    reset_steps: tuple[int, ...]
    dilation_radius: int
    reuse_mode: str
    select_every_step: bool
    min_threshold: float = 1e-4
    threshold_decay: float = 0.5

    @classmethod
    def from_acceleration_config(
        cls, config: FluxKontextAccelerationConfig
    ) -> FluxSpotEditConfig:
        return cls(
            profile=config.spotedit_profile,
            threshold=config.spotedit_threshold,
            judge_method=config.spotedit_judge_method,
            initial_steps=config.spotedit_initial_steps,
            reset_steps=config.spotedit_reset_steps,
            dilation_radius=config.spotedit_dilation_radius,
            reuse_mode=config.spotedit_reuse_mode,
            select_every_step=config.spotedit_select_every_step,
        )

    def is_full_step(self, step_index: int) -> bool:
        return step_index < self.initial_steps or step_index in self.reset_steps


class FluxSpotEditTokenLPIPS:
    """FLUX VAE feature distance used by the SpotEdit reference implementation."""

    def __init__(self, vae: Any) -> None:
        self.vae = vae
        self.reference_tokens: torch.Tensor | None = None
        self.reference_features: dict[str, torch.Tensor] | None = None

    def _unpack(
        self, tokens: torch.Tensor, image_size: tuple[int, int], vae_scale_factor: int
    ) -> torch.Tensor:
        batch, token_count, packed_channels = tokens.shape
        latent_height = 2 * (image_size[0] // (vae_scale_factor * 2))
        latent_width = 2 * (image_size[1] // (vae_scale_factor * 2))
        if (
            token_count != (latent_height // 2) * (latent_width // 2)
            or packed_channels % 4
        ):
            raise ValueError(
                "SpotEdit token layout does not match the requested FLUX image size"
            )
        latent = tokens.view(
            batch, latent_height // 2, latent_width // 2, packed_channels // 4, 2, 2
        )
        return latent.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, packed_channels // 4, latent_height, latent_width
        )

    def _prepare(
        self, tokens: torch.Tensor, image_size: tuple[int, int], vae_scale_factor: int
    ) -> torch.Tensor:
        latent = self._unpack(tokens, image_size, vae_scale_factor)
        parameter = next(self.vae.decoder.parameters())
        latent = latent.to(device=parameter.device, dtype=parameter.dtype)
        scaling = float(getattr(self.vae.config, "scaling_factor", 1.0))
        shift = float(getattr(self.vae.config, "shift_factor", 0.0) or 0.0)
        return latent / scaling + shift

    def _features(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        decoder = self.vae.decoder
        hidden = decoder.conv_in(latent)
        features = {"conv_in": hidden}
        hidden = decoder.mid_block(hidden, None)
        features["mid_block"] = hidden
        hidden = decoder.up_blocks[0](hidden, None)
        features["up_blocks.0"] = hidden
        return features

    @torch.no_grad()
    def __call__(
        self,
        predicted: torch.Tensor,
        reference: torch.Tensor,
        *,
        image_size: tuple[int, int],
        vae_scale_factor: int,
    ) -> torch.Tensor:
        if self.reference_tokens is None or not torch.equal(
            self.reference_tokens, reference
        ):
            self.reference_tokens = reference.detach().clone()
            self.reference_features = self._features(
                self._prepare(reference, image_size, vae_scale_factor)
            )
        if self.reference_features is None:
            raise RuntimeError("SpotEdit reference features were not initialized")
        predicted_latent = self._prepare(predicted, image_size, vae_scale_factor)
        predicted_features = self._features(predicted_latent)
        differences = []
        for name in ("conv_in", "mid_block", "up_blocks.0"):
            first = F.normalize(predicted_features[name], dim=1)
            second = F.normalize(self.reference_features[name], dim=1)
            difference = (first - second).pow(2).sum(dim=1, keepdim=True)
            difference = F.interpolate(
                difference,
                size=predicted_latent.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            differences.append(difference)
        score = torch.stack(differences).mean(dim=0)
        return F.avg_pool2d(score, kernel_size=2, stride=2).flatten(start_dim=1)


@dataclass
class FluxSpotEditState:
    config: FluxSpotEditConfig
    vae: Any
    generated_token_count: int
    grid_height: int
    grid_width: int
    reuse_mask: torch.Tensor | None = None
    previous_x0: torch.Tensor | None = None
    previous_full_noise: torch.Tensor | None = None
    final_reuse_mask: torch.Tensor | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    lpips: FluxSpotEditTokenLPIPS = field(init=False)

    def __post_init__(self) -> None:
        self.lpips = FluxSpotEditTokenLPIPS(self.vae)

    def select(
        self,
        reference_latents: torch.Tensor,
        *,
        image_size: tuple[int, int],
        vae_scale_factor: int,
    ) -> torch.Tensor:
        if self.previous_x0 is None:
            raise RuntimeError("SpotEdit selection requires a previous x0 prediction")
        threshold = self.config.threshold
        mask = self._judge(reference_latents, threshold, image_size, vae_scale_factor)
        while bool(mask.all()) and threshold > self.config.min_threshold:
            threshold *= self.config.threshold_decay
            mask = self._judge(
                reference_latents, threshold, image_size, vae_scale_factor
            )
        if bool(mask.all()):
            mask = torch.zeros_like(mask)
        mask = dilate_uncached_mask(
            mask, self.grid_height, self.grid_width, self.config.dilation_radius
        )
        self.reuse_mask = mask
        if bool(mask.any()):
            self.final_reuse_mask = mask.detach().clone()
        return mask

    def _judge(
        self,
        reference: torch.Tensor,
        threshold: float,
        image_size: tuple[int, int],
        vae_scale_factor: int,
    ) -> torch.Tensor:
        predicted = self.previous_x0
        if predicted is None or predicted.shape != reference.shape:
            raise ValueError("SpotEdit requires aligned predicted and reference tokens")
        if self.config.judge_method == "L4":
            score = (predicted - reference).abs().pow(4).mean(dim=-1).mean(dim=0)
            return score < threshold
        if self.config.judge_method == "cosine":
            score = F.cosine_similarity(
                reference.flatten(0, 1), predicted.flatten(0, 1), dim=-1
            )
            return score > threshold
        score = self.lpips(
            predicted,
            reference,
            image_size=image_size,
            vae_scale_factor=vae_scale_factor,
        ).mean(dim=0)
        if self.config.judge_method == "LPIPS_kmeans":
            return _kmeans2_reuse(score)
        return score < threshold


def _kmeans2_reuse(scores: torch.Tensor) -> torch.Tensor:
    values = np.sort(scores.detach().float().cpu().numpy().astype(np.float64))
    if len(values) < 2 or values[-1] <= values[0]:
        return scores <= float(values[-1])
    prefix = np.cumsum(values)
    prefix_sq = np.cumsum(values**2)
    best_index = 1
    best_sse = np.inf
    for index in range(1, len(values)):
        left_count = index
        right_count = len(values) - index
        left_sum = prefix[index - 1]
        right_sum = prefix[-1] - left_sum
        left_sq = prefix_sq[index - 1]
        right_sq = prefix_sq[-1] - left_sq
        sse = (
            left_sq
            - left_sum * left_sum / left_count
            + right_sq
            - right_sum * right_sum / right_count
        )
        if sse < best_sse:
            best_sse = sse
            best_index = index
    threshold = float((values[best_index - 1] + values[best_index]) / 2)
    return scores <= threshold


def dilate_uncached_mask(
    reuse_mask: torch.Tensor,
    grid_height: int,
    grid_width: int,
    dilation_radius: int,
) -> torch.Tensor:
    if dilation_radius <= 0:
        return reuse_mask.reshape(-1)
    uncached = (~reuse_mask).float().view(1, 1, grid_height, grid_width)
    dilated = F.max_pool2d(
        uncached,
        kernel_size=2 * dilation_radius + 1,
        stride=1,
        padding=dilation_radius,
    )
    return (~dilated.squeeze().bool()).reshape(-1)


def spotedit_alpha(normalized_timestep: float, *, profile: str) -> float:
    timestep = min(max(float(normalized_timestep), 0.0), 1.0)
    if profile == "source_legacy":
        timestep /= 1000.0
    return math.cos(0.5 * math.pi * timestep) ** 2


def boundary_aware_smoothing(
    generated: torch.Tensor,
    reference: torch.Tensor,
    reuse_mask: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    weight: float = 0.8,
) -> torch.Tensor:
    if generated.ndim != 3 or generated.shape != reference.shape:
        raise ValueError(
            "SpotEdit smoothing requires matching [batch, tokens, channels] tensors"
        )
    batch, token_count, channels = generated.shape
    if grid_height <= 0 or grid_width <= 0 or grid_height * grid_width != token_count:
        raise ValueError(
            f"SpotEdit smoothing grid {grid_height}x{grid_width} does not match {token_count} tokens"
        )
    if reuse_mask.numel() != batch * token_count:
        raise ValueError(
            "SpotEdit smoothing reuse mask does not match the generated token grid"
        )
    reuse = reuse_mask.bool().reshape(batch, 1, grid_height, grid_width)
    edited = (~reuse).float()
    boundary = reuse & (F.max_pool2d(edited, kernel_size=3, stride=1, padding=1) > 0)
    interior = reuse & ~boundary
    output = (
        generated.reshape(batch, grid_height, grid_width, channels)
        .permute(0, 3, 1, 2)
        .clone()
    )
    source = reference.reshape(batch, grid_height, grid_width, channels).permute(
        0, 3, 1, 2
    )
    output = torch.where(interior.expand_as(output), source, output)
    blended = weight * source + (1.0 - weight) * output
    output = torch.where(boundary.expand_as(output), blended, output)
    return output.permute(0, 2, 3, 1).reshape_as(generated)


def gather_tokens(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_index = (
        indices.to(device=tensor.device, dtype=torch.long)
        .unsqueeze(-1)
        .expand(-1, -1, tensor.shape[-1])
    )
    return torch.gather(tensor, 1, gather_index)


def scatter_tokens_(
    destination: torch.Tensor, indices: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    scatter_index = (
        indices.to(device=destination.device, dtype=torch.long)
        .unsqueeze(-1)
        .expand_as(source)
    )
    destination.scatter_(1, scatter_index, source)
    return destination


def gather_ids(ids: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if ids.ndim != 2:
        raise ValueError("FLUX image ids must have shape [tokens, axes]")
    if indices.shape[0] != 1:
        raise ValueError("FLUX compact acceleration currently supports batch size 1")
    return ids.index_select(
        0, indices.squeeze(0).to(device=ids.device, dtype=torch.long)
    )


def with_acceleration_context(
    attention_kwargs: dict[str, Any] | None,
    context: FluxSparseAttentionContext | FluxToCaState | None,
    *,
    step_index: int,
) -> dict[str, Any]:
    result = dict(attention_kwargs or {})
    if context is not None:
        result["_flux_acceleration_context"] = context
        result["_flux_acceleration_step"] = step_index
    return result


__all__ = [
    "FLUX_REGIONE_GAMMA_28",
    "FluxKontextAccelerationConfig",
    "FluxRASDecision",
    "FluxRASState",
    "FluxRegionEState",
    "FluxSparseAttentionContext",
    "FluxSparseKVCache",
    "FluxSpotEditConfig",
    "FluxSpotEditState",
    "FluxToCaState",
    "boundary_aware_smoothing",
    "gather_ids",
    "gather_tokens",
    "scatter_tokens_",
    "spotedit_alpha",
    "with_acceleration_context",
]
