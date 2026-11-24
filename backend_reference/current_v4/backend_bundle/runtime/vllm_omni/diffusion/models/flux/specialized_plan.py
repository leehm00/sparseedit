# SPDX-License-Identifier: Apache-2.0
"""Fixed-mask replay artifacts and GPU-resident request buffers for FLUX.

The paper selector is the only source of the active set.  A trace records the
exact per-step phase/action plus the sorted active generated indices for every
sparse step so later runtime variants can replay the *same* action trace with
different execution mechanics.

``SpecializedExecutionPlan`` owns the fixed-K compact buffers used by the
transformer entry points.  Indices, gathered latents and compact position IDs
are written in place every sparse step; no per-layer host readback is needed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

TRACE_SCHEMA_VERSION = 1
TRACE_KIND = "flux_sparseedit_fixed_mask_trace"


class SparseTraceError(RuntimeError):
    """Raised when a fixed-mask replay artifact is invalid or stale."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def expected_schedule(
    *,
    num_inference_steps: int,
    ratio: float,
    warmup_steps: int,
    refresh_interval: int,
) -> list[tuple[str, str]]:
    """Return the paper-runtime ``(phase, kv_action)`` schedule for every step."""
    schedule: list[tuple[str, str]] = []
    since_dense = 0
    for step in range(num_inference_steps):
        refresh = bool(
            step >= warmup_steps and refresh_interval and since_dense > 0 and since_dense % refresh_interval == 0
        )
        dense = ratio == 1 or step < warmup_steps or refresh
        if ratio == 1:
            phase = "full"
        elif step < warmup_steps:
            phase = "warmup"
        elif refresh:
            phase = "refresh_full"
        else:
            phase = "sparse"
        if dense:
            kv_action = "store_full" if ratio < 1 and (step == warmup_steps - 1 or refresh) else "none"
            since_dense = 0
        else:
            kv_action = "update_active"
            since_dense += 1
        schedule.append((phase, kv_action))
    return schedule


@dataclass(frozen=True)
class TraceStep:
    step: int
    phase: str
    kv_action: str
    indices: tuple[int, ...] | None = None
    active_mask_b64: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step": self.step,
            "phase": self.phase,
            "kv_action": self.kv_action,
        }
        if self.indices is not None:
            payload["indices"] = list(self.indices)
        if self.active_mask_b64 is not None:
            payload["active_mask_b64"] = self.active_mask_b64
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceStep:
        try:
            step = int(payload["step"])
            phase = str(payload["phase"])
            kv_action = str(payload["kv_action"])
        except (KeyError, TypeError, ValueError) as error:
            raise SparseTraceError(f"malformed trace step: {dict(payload)}") from error
        raw_indices = payload.get("indices")
        indices: tuple[int, ...] | None
        if raw_indices is None:
            indices = None
        elif isinstance(raw_indices, Sequence) and not isinstance(raw_indices, (str, bytes)):
            indices = tuple(int(value) for value in raw_indices)
        else:
            raise SparseTraceError(f"trace step {step} indices must be a sequence")
        mask = payload.get("active_mask_b64")
        if mask is not None and not isinstance(mask, str):
            raise SparseTraceError(f"trace step {step} active_mask_b64 must be a string")
        return cls(step=step, phase=phase, kv_action=kv_action, indices=indices, active_mask_b64=mask)


@dataclass(frozen=True)
class FixedMaskTrace:
    """Validated, canonical fixed-mask replay artifact."""

    config: dict[str, Any]
    steps: tuple[TraceStep, ...]
    sha256: str

    # ------------------------------------------------------------------
    # Construction and persistence.
    # ------------------------------------------------------------------
    @classmethod
    def from_step_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        config: Mapping[str, Any],
    ) -> FixedMaskTrace:
        steps: list[TraceStep] = []
        for record in records:
            step = int(record["step"])
            phase = str(record["phase"])
            kv_action = str(record["kv_action"])
            indices = record.get("indices")
            if indices is not None:
                if isinstance(indices, torch.Tensor):
                    indices = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1).tolist()
                else:
                    indices = [int(value) for value in indices]
            mask = record.get("active_mask_b64")
            steps.append(
                TraceStep(
                    step=step,
                    phase=phase,
                    kv_action=kv_action,
                    indices=tuple(indices) if indices is not None else None,
                    active_mask_b64=str(mask) if mask is not None else None,
                )
            )
        steps.sort(key=lambda item: item.step)
        payload = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "kind": TRACE_KIND,
            "config": {key: _jsonable(value) for key, value in dict(config).items()},
            "steps": [step.to_dict() for step in steps],
        }
        return cls(config=payload["config"], steps=tuple(steps), sha256=_sha256(payload))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FixedMaskTrace:
        if payload.get("kind") != TRACE_KIND:
            raise SparseTraceError(f"unexpected trace kind: {payload.get('kind')!r}")
        if int(payload.get("schema_version", -1)) != TRACE_SCHEMA_VERSION:
            raise SparseTraceError("unsupported fixed-mask trace schema version")
        config = payload.get("config")
        if not isinstance(config, Mapping):
            raise SparseTraceError("fixed-mask trace is missing its config object")
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise SparseTraceError("fixed-mask trace is missing steps")
        steps = tuple(TraceStep.from_dict(row) for row in raw_steps)
        expected_digest = _sha256({"schema_version": TRACE_SCHEMA_VERSION, "kind": TRACE_KIND,
                                   "config": dict(config), "steps": [step.to_dict() for step in steps]})
        declared = payload.get("sha256")
        if declared is not None and str(declared) != expected_digest:
            raise SparseTraceError("fixed-mask trace hash does not match its payload")
        return cls(config=dict(config), steps=steps, sha256=expected_digest)

    @classmethod
    def load(cls, path: str | Path) -> FixedMaskTrace:
        trace_path = Path(path)
        try:
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise SparseTraceError(f"fixed-mask trace does not exist: {trace_path}") from error
        except json.JSONDecodeError as error:
            raise SparseTraceError(f"invalid fixed-mask trace JSON at {trace_path}: {error}") from error
        if not isinstance(payload, Mapping):
            raise SparseTraceError("fixed-mask trace root must be an object")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "kind": TRACE_KIND,
            "config": dict(self.config),
            "steps": [step.to_dict() for step in self.steps],
            "sha256": self.sha256,
        }

    def save(self, path: str | Path) -> Path:
        trace_path = Path(path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return trace_path

    # ------------------------------------------------------------------
    # Validation and access.
    # ------------------------------------------------------------------
    def validate(
        self,
        *,
        num_inference_steps: int,
        ratio: float,
        warmup_steps: int,
        refresh_interval: int,
        generated_token_count: int,
        reference_token_count: int,
        grid_height: int,
        grid_width: int,
        active_token_count: int,
    ) -> None:
        if grid_height * grid_width != generated_token_count:
            raise SparseTraceError(
                f"grid {grid_height}x{grid_width} does not match {generated_token_count} generated tokens"
            )
        if len(self.steps) != num_inference_steps:
            raise SparseTraceError(
                f"fixed-mask trace covers {len(self.steps)} steps, runtime has {num_inference_steps}"
            )
        config = self.config
        checks = {
            "num_inference_steps": num_inference_steps,
            "generated_token_count": generated_token_count,
            "reference_token_count": reference_token_count,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "active_token_count": active_token_count,
        }
        for name, expected in checks.items():
            if name in config and int(config[name]) != int(expected):
                raise SparseTraceError(
                    f"stale fixed-mask trace: config[{name!r}]={config[name]} does not match {expected}"
                )
        for name, expected in (("ratio", float(ratio)), ("warmup_steps", int(warmup_steps)),
                               ("refresh_interval", int(refresh_interval))):
            if name in config and not math.isclose(float(config[name]), float(expected), rel_tol=0.0, abs_tol=1e-9):
                raise SparseTraceError(
                    f"stale fixed-mask trace: config[{name!r}]={config[name]} does not match {expected}"
                )
        schedule = expected_schedule(
            num_inference_steps=num_inference_steps,
            ratio=float(ratio),
            warmup_steps=int(warmup_steps),
            refresh_interval=int(refresh_interval),
        )
        seen: set[int] = set()
        for trace_step in self.steps:
            if trace_step.step in seen:
                raise SparseTraceError(f"duplicate trace step {trace_step.step}")
            seen.add(trace_step.step)
            if not 0 <= trace_step.step < num_inference_steps:
                raise SparseTraceError(f"trace step outside schedule: {trace_step.step}")
            expected_phase, expected_action = schedule[trace_step.step]
            if trace_step.phase != expected_phase:
                raise SparseTraceError(
                    f"stale fixed-mask trace at step {trace_step.step}: phase {trace_step.phase!r} "
                    f"does not match schedule {expected_phase!r}"
                )
            if trace_step.kv_action != expected_action:
                raise SparseTraceError(
                    f"stale fixed-mask trace at step {trace_step.step}: kv_action {trace_step.kv_action!r} "
                    f"does not match schedule {expected_action!r}"
                )
            if expected_action == "update_active":
                indices = trace_step.indices
                if indices is None:
                    raise SparseTraceError(f"sparse trace step {trace_step.step} is missing indices")
                if len(indices) != active_token_count:
                    raise SparseTraceError(
                        f"sparse trace step {trace_step.step} has {len(indices)} indices, expected {active_token_count}"
                    )
                if len(set(indices)) != len(indices):
                    raise SparseTraceError(f"sparse trace step {trace_step.step} contains duplicate indices")
                if min(indices) < 0 or max(indices) >= generated_token_count:
                    raise SparseTraceError(
                        f"sparse trace step {trace_step.step} indexes outside [0, {generated_token_count})"
                    )
                if trace_step.active_mask_b64 is not None:
                    decoded = _unpack_mask(trace_step.active_mask_b64, generated_token_count)
                    if np.flatnonzero(decoded).tolist() != list(indices):
                        raise SparseTraceError(
                            f"sparse trace step {trace_step.step} mask does not match its sorted indices"
                        )
            elif trace_step.indices is not None:
                raise SparseTraceError(
                    f"dense trace step {trace_step.step} unexpectedly carries sparse indices"
                )

    def step(self, index: int) -> TraceStep:
        if index < 0 or index >= len(self.steps):
            raise SparseTraceError(f"trace step {index} is outside the recorded schedule")
        return self.steps[index]

    def indices_for(self, index: int, *, device: torch.device | str | None = None) -> torch.Tensor:
        trace_step = self.step(index)
        if trace_step.indices is None:
            raise SparseTraceError(f"trace step {index} does not carry a sparse mask")
        tensor = torch.tensor(trace_step.indices, dtype=torch.long)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor.unsqueeze(0)

    def masks(self) -> list[tuple[int, np.ndarray]]:
        token_count = int(self.config["generated_token_count"])
        output: list[tuple[int, np.ndarray]] = []
        for trace_step in self.steps:
            if trace_step.active_mask_b64 is None:
                continue
            output.append((trace_step.step, _unpack_mask(trace_step.active_mask_b64, token_count)))
        return output


def _unpack_mask(encoded: str, token_count: int) -> np.ndarray:
    try:
        raw = base64.b64decode(encoded)
    except Exception as error:  # noqa: BLE001 - malformed user artifact
        raise SparseTraceError("invalid base64 active mask") from error
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:token_count]
    if bits.size != token_count:
        raise SparseTraceError(f"active mask covers {bits.size} tokens, expected {token_count}")
    return bits.astype(bool)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


class SpecializedExecutionPlan:
    """Owned fixed-K buffers bound once per request and reused every sparse step."""

    def __init__(
        self,
        *,
        variant: str,
        generated_token_count: int,
        reference_token_count: int,
        active_token_count: int,
        latent_channels: int,
        dtype: torch.dtype,
        device: torch.device,
        generated_ids: torch.Tensor | None = None,
        fuse_stage: bool = False,
    ) -> None:
        if active_token_count <= 0 or active_token_count > generated_token_count:
            raise ValueError("active_token_count must be in [1, generated_token_count]")
        self.variant = variant
        self.generated_token_count = int(generated_token_count)
        self.reference_token_count = int(reference_token_count)
        self.active_token_count = int(active_token_count)
        self.latent_channels = int(latent_channels)
        self.dtype = dtype
        self.device = torch.device(device)
        self.fuse_stage = fuse_stage
        self.stage_backend: str | None = None
        self.row_map = (
            torch.empty(self.generated_token_count, dtype=torch.int32, device=self.device) if fuse_stage else None
        )
        if fuse_stage and generated_ids is None:
            raise ValueError("stage packing requires generated position IDs")
        self.active_indices = torch.zeros((1, self.active_token_count), dtype=torch.long, device=self.device)
        self.compact_latents = torch.zeros(
            (1, self.active_token_count, self.latent_channels), dtype=dtype, device=self.device
        )
        if generated_ids is not None:
            if generated_ids.ndim != 2 or generated_ids.shape[0] != self.generated_token_count:
                raise ValueError("generated_ids must have shape [generated_token_count, axes]")
            self.compact_ids = torch.zeros(
                (self.active_token_count, generated_ids.shape[1]),
                dtype=generated_ids.dtype,
                device=self.device,
            )
            self._generated_ids = generated_ids.to(device=self.device)
        else:
            self.compact_ids = torch.zeros((self.active_token_count, 3), dtype=torch.float32, device=self.device)
            self._generated_ids = None
        self.timestep = torch.zeros((1,), dtype=dtype, device=self.device)
        self.bind_count = 0
        self.released = False

    def _ensure_live(self) -> None:
        if self.released:
            raise RuntimeError("specialized execution plan was released")

    def bind_sparse(
        self,
        latents: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update the fixed compact buffers in place and return their views."""
        self._ensure_live()
        if latents.shape != (1, self.generated_token_count, self.latent_channels):
            raise ValueError(
                "stale plan/shape mismatch: latents must be "
                f"{(1, self.generated_token_count, self.latent_channels)}, got {tuple(latents.shape)}"
            )
        flat_input = indices.reshape(-1) if isinstance(indices, torch.Tensor) else None
        if flat_input is None:
            raise TypeError("indices must be a torch tensor")
        if flat_input.numel() != self.active_token_count:
            raise ValueError(
                f"stale plan/shape mismatch: got {flat_input.numel()} indices, plan expects {self.active_token_count}"
            )
        if indices.ndim == 2 and indices.shape[0] != 1:
            raise ValueError("specialized runtime supports batch size 1")
        if self.fuse_stage:
            from vllm_omni.diffusion.models.flux.specialized_stage import pack_sparse_inputs

            self.stage_backend = pack_sparse_inputs(
                latents, self._generated_ids, indices, self.active_indices,
                self.compact_latents, self.compact_ids, self.row_map,
            )
            self.bind_count += 1
            return self.active_indices, self.compact_latents, self.compact_ids
        self.active_indices.copy_(flat_input.reshape(1, -1).to(device=self.device, dtype=torch.long))
        gather_index = self.active_indices.unsqueeze(-1).expand(-1, -1, self.latent_channels)
        torch.gather(latents, 1, gather_index, out=self.compact_latents)
        if self._generated_ids is not None:
            self.compact_ids.copy_(self._generated_ids.index_select(0, self.active_indices.reshape(-1)))
        self.bind_count += 1
        return self.active_indices, self.compact_latents, self.compact_ids

    def bind_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        self._ensure_live()
        source = timestep.reshape(-1)
        if source.numel() != 1:
            raise ValueError("graph timestep buffer requires a scalar timestep")
        self.timestep.copy_(source.to(device=self.device, dtype=self.dtype))
        return self.timestep

    def witness(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "generated_token_count": self.generated_token_count,
            "reference_token_count": self.reference_token_count,
            "active_token_count": self.active_token_count,
            "latent_channels": self.latent_channels,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "bind_count": self.bind_count,
            "active_indices_ptr": self.active_indices.data_ptr(),
            "compact_latents_ptr": self.compact_latents.data_ptr(),
            "compact_ids_ptr": self.compact_ids.data_ptr(),
        }

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.active_indices = None  # type: ignore[assignment]
        self.compact_latents = None  # type: ignore[assignment]
        self.compact_ids = None  # type: ignore[assignment]
        self.timestep = None  # type: ignore[assignment]
        self._generated_ids = None
        self.row_map = None


__all__ = [
    "FixedMaskTrace",
    "SpecializedExecutionPlan",
    "SparseTraceError",
    "TRACE_KIND",
    "TRACE_SCHEMA_VERSION",
    "TraceStep",
    "expected_schedule",
]
