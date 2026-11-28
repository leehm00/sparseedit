# SPDX-License-Identifier: Apache-2.0
"""Online-controller replay and immutable-anchor velocity reuse primitives.

The AVD per-step ratio is applied to the last computed velocity. Accumulation
is controller state only, never an implicit multiplicative reuse factor.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm_omni.diffusion.models.flux.reuse_stage_core import (
    _launch_fused_stage,
    _native_euler_scaled,
    _validate_stage,
)

TRACE_KIND = "flux_sparseedit_online_controller_trace"
TRACE_SCHEMA_VERSION = 1


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ControllerStep:
    step: int
    action: str
    ratio: float | None
    accumulate: float
    force_reason: str | None = None
    indices: tuple[int, ...] | None = None
    anchor_step: int | None = None

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 0:
            raise ValueError("controller step must be a nonnegative integer")
        if self.anchor_step is not None and (
            type(self.anchor_step) is not int or self.anchor_step < 0
        ):
            raise ValueError("controller anchor must be a nonnegative integer")
        if self.force_reason is not None and not isinstance(self.force_reason, str):
            raise ValueError("controller reason must be a string")
        if self.indices is not None and any(type(i) is not int for i in self.indices):
            raise ValueError("controller mask indices must be integers")
        if self.action not in {"dense", "sparse", "reuse"}:
            raise ValueError(f"unsupported controller action: {self.action}")
        if not math.isfinite(float(self.accumulate)) or float(self.accumulate) <= 0:
            raise ValueError("accumulate must be finite and positive")
        if self.action == "reuse":
            if (
                self.ratio is None
                or not math.isfinite(float(self.ratio))
                or float(self.ratio) <= 0
            ):
                raise ValueError("reuse requires a finite positive per-step ratio")
            if self.indices is not None:
                raise ValueError("reuse cannot carry a sparse mask")
        else:
            if self.ratio is not None:
                raise ValueError("computed steps do not have a reuse ratio")
            if self.action == "sparse":
                if self.indices is None or len(set(self.indices)) != len(self.indices):
                    raise ValueError("sparse action requires a unique mask")
            elif self.indices is not None:
                raise ValueError("dense action cannot carry a sparse mask")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "step": self.step,
            "action": self.action,
            "ratio": self.ratio,
            "accumulate": self.accumulate,
            "force_reason": self.force_reason,
            "anchor_step": self.anchor_step,
        }
        if self.indices is not None:
            payload["indices"] = list(self.indices)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ControllerStep:
        if not isinstance(payload, Mapping):
            raise ValueError("controller step must be an object")  # noqa: TRY004
        raw_indices = payload.get("indices")
        if raw_indices is not None and (
            not isinstance(raw_indices, Sequence)
            or isinstance(raw_indices, (str, bytes))
        ):
            raise ValueError("controller mask must be an index sequence")
        try:
            return cls(
                step=payload["step"],
                action=str(payload["action"]),
                ratio=None if payload.get("ratio") is None else float(payload["ratio"]),
                accumulate=float(payload["accumulate"]),
                force_reason=payload.get("force_reason"),
                anchor_step=payload.get("anchor_step"),
                indices=None if raw_indices is None else tuple(raw_indices),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed controller step: {dict(payload)}") from error


@dataclass(frozen=True)
class OnlineReuseTrace:
    config: dict[str, Any]
    steps: tuple[ControllerStep, ...]
    sha256: str

    @classmethod
    def from_steps(
        cls, *, config: Mapping[str, Any], steps: Sequence[ControllerStep]
    ) -> OnlineReuseTrace:
        body = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "kind": TRACE_KIND,
            "config": dict(config),
            "steps": [step.to_dict() for step in steps],
        }
        return cls(config=dict(config), steps=tuple(steps), sha256=_digest(body))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "kind": TRACE_KIND,
            "config": dict(self.config),
            "steps": [step.to_dict() for step in self.steps],
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OnlineReuseTrace:
        if not isinstance(payload, Mapping):
            raise ValueError("online trace root must be an object")  # noqa: TRY004
        if (
            payload.get("schema_version") != TRACE_SCHEMA_VERSION
            or payload.get("kind") != TRACE_KIND
        ):
            raise ValueError("unsupported online trace schema or kind")
        if not isinstance(payload.get("config"), Mapping) or not isinstance(
            payload.get("steps"), list
        ):
            raise ValueError("online trace config/steps are malformed")  # noqa: TRY004
        body = {
            key: payload[key] for key in ("schema_version", "kind", "config", "steps")
        }
        if payload.get("sha256") != _digest(body):
            raise ValueError("online trace sha256/digest mismatch")
        return cls(
            config=dict(payload["config"]),
            steps=tuple(ControllerStep.from_dict(item) for item in payload["steps"]),
            sha256=str(payload["sha256"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> OnlineReuseTrace:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot read online controller trace: {path}: {error}"
            ) from error
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output


def validate_controller_replay(
    trace: OnlineReuseTrace,
    *,
    num_inference_steps: int,
    generated_token_count: int | None = None,
    active_token_count: int | None = None,
    warmup_steps: int | None = None,
    refresh_interval: int | None = None,
    gamma: Sequence[float] | None = None,
) -> None:
    if len(trace.steps) != num_inference_steps:
        raise ValueError("online trace does not cover the complete scheduler")
    config = trace.config
    required = {
        "num_inference_steps",
        "generated_token_count",
        "reference_token_count",
        "active_token_count",
        "grid_height",
        "grid_width",
        "ratio",
        "warmup_steps",
        "refresh_interval",
        "true_cfg",
        "true_cfg_scale",
        "selector_protocol",
        "selector_mode",
        "selector_constant_weight",
        "avd_cache_threshold",
        "avd_max_reuse",
        "avd_force_final_steps",
        "gamma_sha256",
        "request_sha256",
        "schedule_sha256",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(
            f"online trace missing required config/provenance: {sorted(missing)}"
        )
    integer_fields = (
        "num_inference_steps",
        "generated_token_count",
        "reference_token_count",
        "active_token_count",
        "grid_height",
        "grid_width",
        "warmup_steps",
        "refresh_interval",
        "avd_max_reuse",
        "avd_force_final_steps",
    )
    if any(
        type(config[name]) is not int or config[name] < 0 for name in integer_fields
    ):
        raise ValueError("online trace integer configuration is malformed")
    count = config["generated_token_count"]
    if (
        count <= 0
        or config["grid_height"] * config["grid_width"] != count
        or not 1 <= config["active_token_count"] <= count
        or not 2 <= config["warmup_steps"] < num_inference_steps
        or config["avd_force_final_steps"] > num_inference_steps
    ):
        raise ValueError(
            "online trace dimensions or forced-compute configuration is malformed"
        )
    if (
        not isinstance(config["true_cfg"], bool)
        or not config["true_cfg"]
        or config["selector_protocol"] != "paper"
        or config["selector_mode"] not in {"forward", "reverse", "constant"}
    ):
        raise ValueError("online trace policy configuration is malformed")
    for name in (
        "ratio",
        "true_cfg_scale",
        "selector_constant_weight",
        "avd_cache_threshold",
    ):
        if not isinstance(config[name], (float, int)) or not math.isfinite(
            float(config[name])
        ):
            raise ValueError(f"online trace malformed config {name}")
    if not 0 < config["ratio"] <= 1 or config["avd_cache_threshold"] < 0:
        raise ValueError("online trace budget/threshold is malformed")
    if config["active_token_count"] != max(1, math.floor(count * config["ratio"])):
        raise ValueError("online trace budget and active count disagree")
    for name in ("gamma_sha256", "request_sha256", "schedule_sha256"):
        if (
            not isinstance(config[name], str)
            or len(config[name]) != 64
            or any(c not in "0123456789abcdef" for c in config[name])
        ):
            raise ValueError(f"online trace malformed provenance {name}")
    for name, expected in (
        ("num_inference_steps", num_inference_steps),
        ("generated_token_count", generated_token_count),
        ("active_token_count", active_token_count),
        ("warmup_steps", warmup_steps),
        ("refresh_interval", refresh_interval),
    ):
        if expected is not None and config[name] != expected:
            raise ValueError(f"stale online trace config {name}")
    if gamma is not None and config["gamma_sha256"] != _digest(list(gamma)):
        raise ValueError("stale online trace gamma hash")
    last_compute = None
    previous_dense = False
    since_dense = hits = 0
    accumulate = 1.0
    for expected_step, step in enumerate(trace.steps):
        if step.step != expected_step:
            raise ValueError("online trace steps must be unique and consecutive")
        refresh = bool(
            expected_step >= config["warmup_steps"]
            and config["refresh_interval"]
            and since_dense > 0
            and since_dense % config["refresh_interval"] == 0
        )
        must_dense = (
            config["ratio"] == 1 or expected_step < config["warmup_steps"] or refresh
        )
        if must_dense and step.action != "dense":
            raise ValueError(
                f"online trace violates forced dense action at step {expected_step}"
            )
        if (
            expected_step >= num_inference_steps - config["avd_force_final_steps"]
            and step.action == "reuse"
        ):
            raise ValueError(
                f"online trace final boundary requires transformer compute at step {expected_step}"
            )
        if step.action == "reuse":
            if (
                last_compute is None
                or previous_dense
                or step.anchor_step != last_compute
            ):
                raise ValueError(
                    f"online trace impossible reuse anchor chronology at step {expected_step}"
                )
            hits += 1
            accumulate *= float(step.ratio)
            if (
                hits > config["avd_max_reuse"]
                or float(step.ratio) >= 1
                or 1 - accumulate > config["avd_cache_threshold"] + 1e-12
                or not math.isclose(
                    step.accumulate, accumulate, rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise ValueError(
                    f"online trace invalid reuse ratio/accumulation at step {expected_step}"
                )
        else:
            if step.anchor_step != expected_step or step.accumulate != 1.0:
                raise ValueError(
                    f"online trace impossible compute anchor chronology at step {expected_step}"
                )
            last_compute = expected_step
            hits = 0
            accumulate = 1.0
        if step.action == "sparse":
            assert step.indices is not None
            if len(step.indices) != config["active_token_count"]:
                raise ValueError("online trace sparse mask has the wrong active count")
        if step.indices is not None and any(
            not 0 <= index < count for index in step.indices
        ):
            raise ValueError("online trace mask index is out of bounds")
        previous_dense = step.action == "dense"
        since_dense = 0 if previous_dense else since_dense + 1


def native_dt_at(scheduler: Any, step: int, device: torch.device | str) -> torch.Tensor:
    """Native FP32/FP64 sigma subtraction without advancing scheduler state."""
    if not 0 <= step < len(scheduler.sigmas) - 1:
        raise IndexError("scheduler dt index outside schedule")
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)
    return sigmas[step + 1] - sigmas[step]


@dataclass(frozen=True)
class PreparedEulerDt:
    """A request-prepared operand matching native scalar dispatch for one sample."""

    value: float | torch.Tensor
    sample_dtype: torch.dtype
    sample_device: torch.device


def native_euler_dt_semantics(scheduler: Any, latent: torch.Tensor) -> dict[str, str]:
    sigmas = scheduler.sigmas
    if sigmas.device.type == "cuda":
        category = "cuda_0d_common_dtype"
    elif sigmas.device.type == "cpu" and latent.device.type == "cuda":
        category = "cpu_host_scalar_on_cuda"
    elif sigmas.device == latent.device and sigmas.device.type == "cpu":
        category = "cpu_tensor_native_dispatch"
    else:
        raise ValueError("unsupported native scheduler sigma/sample device combination")
    return {
        "scalar_category": category,
        "sigma_dtype": str(sigmas.dtype),
        "sample_dtype": str(latent.dtype),
    }


def prepare_native_euler_dts(
    scheduler: Any,
    latent: torch.Tensor,
) -> tuple[tuple[PreparedEulerDt, ...], dict[str, Any]]:
    """Prepare scalar promotion once; run() binds each operand without GPU work.

    A native CUDA zero-dimensional dt participates in TensorIterator's common
    model dtype before opmath multiplication. A CPU scalar used on CUDA keeps
    the host scalar's precision. CPU tensor execution keeps its native dispatch.
    """
    sigmas = scheduler.sigmas
    if sigmas.ndim != 1 or sigmas.dtype not in (torch.float32, torch.float64):
        raise ValueError("native sigma schedule must be one-dimensional FP32/FP64")
    semantics = native_euler_dt_semantics(scheduler, latent)
    native = sigmas[1:] - sigmas[:-1]
    if semantics["scalar_category"] == "cuda_0d_common_dtype":
        if sigmas.device != latent.device:
            raise ValueError("native CUDA sigmas must share the sample device")
        prepared = native.to(dtype=latent.dtype).to(dtype=torch.float32)
        values = tuple(prepared.unbind())
        operand = "round_to_sample_dtype_then_fp32"
    elif semantics["scalar_category"] == "cpu_host_scalar_on_cuda":
        values = tuple(native.tolist())
        operand = "host_scalar_full_precision"
    else:
        values = tuple(native.unbind())
        operand = "native_cpu_tensor"
    bindings = tuple(
        PreparedEulerDt(value, latent.dtype, latent.device) for value in values
    )
    return bindings, {
        "semantics": semantics,
        "original_sigma_device": str(sigmas.device),
        "original_sigma_dtype": str(sigmas.dtype),
        "operand_preparation": operand,
        "preparation_scope": "once_per_request_before_denoising_loop",
    }


class AnchoredReuseProgram:
    """One-step immutable-anchor Euler update with optional owned output slots."""

    def __init__(self, latent: torch.Tensor, anchor: torch.Tensor, mode: str) -> None:
        if mode not in {"eager", "prealloc", "fused"}:
            raise ValueError("online reuse mode must be eager, prealloc or fused")
        if mode == "fused" and not latent.is_cuda:
            raise RuntimeError("fused online reuse requires CUDA")
        _validate_stage(latent, anchor, 0.0, None)
        self.mode = mode
        # Two recurrent latent slots and three velocity slots keep the two
        # preceding complete velocities alive while the next result is written.
        self.latent_slots = (
            tuple(torch.empty_like(latent) for _ in range(2)) if mode != "eager" else ()
        )
        self.velocity_slots = (
            tuple(torch.empty_like(anchor) for _ in range(3)) if mode != "eager" else ()
        )
        self.calls = 0
        self.fused_kernel_launches = 0
        self._sample_fp32 = (
            torch.empty_like(latent, dtype=torch.float32)
            if mode == "prealloc"
            else None
        )
        self._increment = torch.empty_like(latent) if mode == "prealloc" else None
        self.age_slots = (
            tuple(
                torch.empty(latent.shape[:-1], dtype=torch.int64, device=latent.device)
                for _ in range(3)
            )
            if mode != "eager"
            else ()
        )
        self._generated = torch.ones(
            latent.shape[:-1], dtype=torch.bool, device=latent.device
        )

    def run(
        self,
        latent: torch.Tensor,
        anchor: torch.Tensor,
        ratio: float,
        dt: float | torch.Tensor | PreparedEulerDt,
        *,
        age: torch.Tensor | None = None,
        generated: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("reuse ratio must be finite and positive")
        prepared_dt = isinstance(dt, PreparedEulerDt)
        if prepared_dt:
            if dt.sample_dtype != latent.dtype or dt.sample_device != latent.device:
                raise ValueError(
                    "prepared dt binding does not match sample dtype/device"
                )
            dt = dt.value
        if (
            self.mode == "fused"
            and isinstance(dt, torch.Tensor)
            and dt.device.type == "cuda"
            and not prepared_dt
        ):
            raise ValueError(
                "fused CUDA dt requires prepare_native_euler_dts to preserve native scalar promotion"
            )
        if age is None and (self.mode == "fused" or generated is not None):
            raise ValueError("fused or masked reuse requires actual token ages")
        if age is not None:
            if (
                age.shape != latent.shape[:-1]
                or age.dtype != torch.int64
                or age.device != latent.device
                or not age.is_contiguous()
            ):
                raise ValueError(
                    "reuse age must match latent tokens as contiguous int64"
                )
            if generated is not None and (
                generated.shape != age.shape
                or generated.dtype != torch.bool
                or generated.device != age.device
                or not generated.is_contiguous()
            ):
                raise ValueError("generated mask must match reuse ages")
        if self.mode == "eager":
            velocity = anchor * ratio
            latent = _native_euler_scaled(latent, velocity, dt)
            self.calls += 1
            if age is None:
                return latent, velocity
            age_out = (
                age + 1 if generated is None else torch.where(generated, age + 1, age)
            )
            return latent, velocity, age_out
        latent_out = self.latent_slots[self.calls % 2]
        velocity_out = self.velocity_slots[self.calls % 3]
        age_out = self.age_slots[self.calls % 3] if age is not None else None
        _validate_stage(latent, anchor, dt, latent_out)
        if age_out is not None and torch._C._overlaps(age_out, age):
            raise ValueError("reuse age output must not alias input")
        if self.mode == "prealloc":
            torch.mul(anchor, ratio, out=velocity_out)
            self._sample_fp32.copy_(latent)
            torch.mul(dt, velocity_out, out=self._increment)
            self._sample_fp32.add_(self._increment)
            latent_out.copy_(self._sample_fp32)
            if age_out is not None:
                torch.add(age, 1, out=age_out)
                if generated is not None:
                    torch.where(generated, age_out, age, out=age_out)
        else:
            _launch_fused_stage(
                latent,
                anchor,
                age,
                self._generated if generated is None else generated,
                ratio,
                dt,
                latent_out,
                velocity_out,
                age_out,
                block_size=1024,
            )
            self.fused_kernel_launches += 1
        self.calls += 1
        if age_out is None:
            return latent_out, velocity_out
        return latent_out, velocity_out, age_out


__all__ = [
    "AnchoredReuseProgram",
    "ControllerStep",
    "OnlineReuseTrace",
    "PreparedEulerDt",
    "native_dt_at",
    "native_euler_dt_semantics",
    "prepare_native_euler_dts",
    "validate_controller_replay",
]
