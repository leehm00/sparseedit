"""Standalone controlled SparseEdit velocity-reuse action.

This is not an online ratio controller and does not modify or claim speedup for
FLUX. Ratios and dt are frozen controlled inputs. The numerical contract follows
installed FlowMatchEulerDiscreteScheduler.step:

1. velocity * ratio, rounded in the velocity dtype;
2. sample.to(float32);
3. dt * scaled_velocity, rounded according to native scalar promotion;
4. the FP32 addition, then one cast back to model_output.dtype.

A zero-dimensional FP32/FP64 tensor dt retains native PyTorch scalar-tensor
promotion. Other real scalars are converted to Python float (documented
controlled FP64 host value); PyTorch then applies its normal scalar promotion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch

SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@dataclass(frozen=True)
class ControlledRatios:
    """One frozen ratio, expanded without online-controller state."""

    value: float

    def __post_init__(self) -> None:
        value = float(self.value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("ratio must be finite and positive")
        object.__setattr__(self, "value", value)

    @classmethod
    def uniform(cls, value: float) -> ControlledRatios:
        return cls(value)

    def expanded(self, length: int) -> tuple[float, ...]:
        if length <= 0:
            raise ValueError("reuse length must be positive")
        return (self.value,) * length


@dataclass(frozen=True)
class ReuseAction:
    ratios: ControlledRatios
    dt: float | torch.Tensor
    length: int

    def __post_init__(self) -> None:
        if not isinstance(self.length, int) or self.length <= 0:
            raise ValueError("reuse length must be positive")
        if isinstance(self.dt, torch.Tensor) and self.dt.is_cuda:
            # Validate on a temporary CPU scalar: never call item() on the CUDA
            # tensor during a captured run. The original device scalar is kept.
            control_dt(self.dt.detach().cpu())
        object.__setattr__(self, "dt", control_dt(self.dt))


@dataclass
class ReuseState:
    latent: torch.Tensor
    history: tuple[torch.Tensor, torch.Tensor]
    age: torch.Tensor
    generated: torch.Tensor

    def validate(self) -> None:
        tensors = (self.latent, *self.history, self.age, self.generated)
        if not all(isinstance(item, torch.Tensor) for item in tensors):
            raise TypeError("all state fields must be tensors")
        if self.latent.ndim < 2:
            raise ValueError("latent must have batch and token dimensions")
        shape = self.latent.shape
        token_shape = shape[:-1]
        if self.history[0].shape != shape or self.history[1].shape != shape:
            raise ValueError("velocity history shape must equal latent shape")
        if self.age.shape != token_shape or self.generated.shape != token_shape:
            raise ValueError("age/generated shape must equal latent shape without channels")
        if self.latent.dtype not in SUPPORTED_DTYPES:
            raise TypeError("latent dtype must be FP32, FP16 or BF16")
        if any(item.dtype != self.latent.dtype for item in self.history):
            raise TypeError("velocity history dtype must equal latent dtype")
        if self.age.dtype != torch.int64:
            raise TypeError("age dtype must be int64")
        if self.generated.dtype != torch.bool:
            raise TypeError("generated mask dtype must be bool")
        device = self.latent.device
        if any(item.device != device for item in tensors[1:]):
            raise ValueError("all state tensors must be on one device")
        if not all(item.is_contiguous() for item in tensors):
            raise ValueError("state tensors must be contiguous")
        storage_tensors = (self.latent, *self.history, self.age, self.generated)
        for index, left in enumerate(storage_tensors):
            for right in storage_tensors[index + 1 :]:
                if torch._C._overlaps(left, right):
                    raise ValueError("input state tensors must not alias")

    def clone(self) -> ReuseState:
        return ReuseState(
            latent=self.latent.clone(),
            history=(self.history[0].clone(), self.history[1].clone()),
            age=self.age.clone(),
            generated=self.generated.clone(),
        )


def _validate_stage_dt(dt: float | torch.Tensor, device: torch.device) -> None:
    """Non-synchronizing launch checks; values are validated by ReuseAction."""

    if isinstance(dt, torch.Tensor):
        if dt.ndim != 0:
            raise ValueError("dt tensor must be zero-dimensional")
        if dt.dtype not in (torch.float32, torch.float64):
            raise TypeError("native dt tensor must be FP32 or FP64")
        if dt.device != device:
            raise ValueError("native scalar dt tensor must share the stage device")
    elif not isinstance(dt, Real) or not math.isfinite(float(dt)):
        raise ValueError("dt must be a finite real scalar")


def control_dt(dt: float | torch.Tensor) -> float | torch.Tensor:
    """Validate and retain the scalar category needed for native promotion."""

    if isinstance(dt, torch.Tensor):
        if dt.ndim != 0:
            raise ValueError("dt tensor must be zero-dimensional")
        if dt.dtype not in (torch.float32, torch.float64):
            raise TypeError("native dt tensor must be FP32 or FP64")
        if not math.isfinite(float(dt.item())):
            raise ValueError("dt must be finite")
        return dt
    if not isinstance(dt, Real):
        raise TypeError("dt must be a real scalar or zero-dimensional tensor")
    value = float(dt)
    if not math.isfinite(value):
        raise ValueError("dt must be finite")
    return value


def _validate_stage(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    dt: float | torch.Tensor,
    out: torch.Tensor | None,
) -> None:
    if latent.shape != velocity.shape:
        raise ValueError("latent and velocity shape must match")
    if latent.dtype != velocity.dtype or latent.dtype not in SUPPORTED_DTYPES:
        raise TypeError("latent and velocity must share a supported dtype")
    if latent.device != velocity.device or not latent.is_contiguous() or not velocity.is_contiguous():
        raise ValueError("stage inputs must be contiguous and on one device")
    _validate_stage_dt(dt, latent.device)
    if out is not None:
        if out.shape != latent.shape or out.dtype != latent.dtype or out.device != latent.device:
            raise ValueError("output must match latent shape, dtype and device")
        if not out.is_contiguous():
            raise ValueError("output must be contiguous")
        if torch._C._overlaps(out, latent) or torch._C._overlaps(out, velocity):
            raise ValueError("output must not alias a stage input")


def _native_euler_scaled(
    latent: torch.Tensor,
    scaled_velocity: torch.Tensor,
    dt: float | torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Native operation boundaries for an already ratio-scaled velocity."""

    _validate_stage(latent, scaled_velocity, dt, out)
    sample_fp32 = latent.to(torch.float32)
    # Keep this multiplication separate: for BF16/FP16 it normally rounds before
    # the FP32 addition. Replacing it with one FP32 FMA is numerically invalid.
    increment = dt * scaled_velocity
    updated = sample_fp32 + increment
    result = updated.to(scaled_velocity.dtype)
    if out is None:
        return result
    out.copy_(result)
    return out


def apply_euler_stage(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    ratio: float,
    dt: float | torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Simplest eager reference: allocation at every arithmetic stage."""

    ControlledRatios.uniform(ratio)
    _validate_stage(latent, velocity, dt, out)
    scaled_velocity = velocity * ratio
    return _native_euler_scaled(latent, scaled_velocity, dt, out)


def eager_reuse_trace(initial: ReuseState, action: ReuseAction) -> ReuseState:
    """Several consecutive native stages with non-aliasing history rotation."""

    initial.validate()
    latent = initial.latent
    previous, previous_previous = initial.history
    age = initial.age
    for ratio in action.ratios.expanded(action.length):
        # This allocation is the velocity retained by the next selector.
        scaled_velocity = previous * ratio
        latent = _native_euler_scaled(latent, scaled_velocity, action.dt)
        previous, previous_previous = scaled_velocity, previous
        age = torch.where(initial.generated, age + 1, age)
    return ReuseState(
        latent=latent,
        history=(previous, previous_previous),
        age=age,
        generated=torch.zeros_like(initial.generated),
    )


class ReuseProgram:
    """Preallocated recurrent state program with ping-pong output storage.

    Inputs are cloned during preparation. run() never overwrites an input or a
    history value needed by a later stage. Returned output views remain valid
    until the next run of this same program; callers retaining them must clone.
    A separate program/graph is required for each length and scalar setting.
    """

    def __init__(self, initial: ReuseState, block_size: int = 1024) -> None:
        initial.validate()
        self.input = initial.clone()
        self.block_size = block_size
        self.latent_slots = tuple(torch.empty_like(initial.latent) for _ in range(2))
        self.velocity_slots = tuple(torch.empty_like(initial.latent) for _ in range(2))
        self.age_slots = tuple(torch.empty_like(initial.age) for _ in range(2))
        self.latent_fp32_scratch = torch.empty_like(initial.latent, dtype=torch.float32)
        self.increment_scratch = torch.empty_like(initial.latent)
        self.empty_generated = torch.zeros_like(initial.generated)
        self._last_length: int | None = None
        self._last_fused: bool | None = None

    def restore_input(self) -> None:
        """Reset prepared input before graph capture or an independent replay."""
        self.input = self.input.clone()

    def run(self, action: ReuseAction, fused: bool = False) -> ReuseState:
        if fused and not self.input.latent.is_cuda:
            raise RuntimeError("fused Triton execution requires CUDA; CPU is reference-only")
        latent = self.input.latent
        previous, previous_previous = self.input.history
        age = self.input.age
        for index, ratio in enumerate(action.ratios.expanded(action.length)):
            slot = index % 2
            next_velocity = self.velocity_slots[slot]
            next_latent = self.latent_slots[slot]
            next_age = self.age_slots[slot]
            if fused:
                _launch_fused_stage(
                    latent,
                    previous,
                    age,
                    self.input.generated,
                    ratio,
                    action.dt,
                    next_latent,
                    next_velocity,
                    next_age,
                    self.block_size,
                )
            else:
                torch.mul(previous, ratio, out=next_velocity)
                # Native out= operations retain each rounding boundary while
                # avoiding the reference path's temporary allocations.
                self.latent_fp32_scratch.copy_(latent)
                torch.mul(next_velocity, action.dt, out=self.increment_scratch)
                torch.add(
                    self.latent_fp32_scratch,
                    self.increment_scratch,
                    out=self.latent_fp32_scratch,
                )
                next_latent.copy_(self.latent_fp32_scratch)
                torch.add(age, self.input.generated, out=next_age)
            latent = next_latent
            previous, previous_previous = next_velocity, previous
            age = next_age
        self._last_length = action.length
        self._last_fused = fused
        return ReuseState(
            latent=latent,
            history=(previous, previous_previous),
            age=age,
            generated=self.empty_generated,
        )


def reference_reuse_trace(
    initial: ReuseState,
    action: ReuseAction,
    preallocate: bool = False,
) -> ReuseState:
    result = (
        ReuseProgram(initial).run(action)
        if preallocate
        else eager_reuse_trace(initial, action)
    )
    return result.clone()


def fused_reuse_trace(initial: ReuseState, action: ReuseAction) -> ReuseState:
    """CUDA-only fused path; importing this module never imports Triton."""

    return ReuseProgram(initial).run(action, fused=True).clone()


_TRITON_KERNEL = None


def _get_triton_kernel():
    global _TRITON_KERNEL
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def reuse_stage_kernel(
        latent_ptr,
        velocity_ptr,
        age_ptr,
        generated_ptr,
        dt_ptr,
        latent_out_ptr,
        velocity_out_ptr,
        age_out_ptr,
        ratio,
        latent_numel,
        age_numel,
        age_blocks,
        IS_NATIVE_DT: tl.constexpr,
        DTYPE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < latent_numel
        velocity = tl.load(velocity_ptr + offsets, mask=mask, other=0.0)
        latent = tl.load(latent_ptr + offsets, mask=mask, other=0.0)
        if IS_NATIVE_DT:
            dt = tl.load(dt_ptr).to(tl.float32)
        else:
            dt = dt_ptr.to(tl.float32)
        # libdevice rounding intrinsics prevent unintended FP32 FMA contraction.
        velocity_fp32 = velocity.to(tl.float32)
        scaled_fp32 = libdevice.mul_rn(velocity_fp32, ratio)
        scaled_native = scaled_fp32.to(DTYPE)
        increment_fp32 = libdevice.mul_rn(dt, scaled_native.to(tl.float32))
        increment_native = increment_fp32.to(DTYPE)
        latent_fp32 = latent.to(tl.float32)
        updated_fp32 = libdevice.add_rn(latent_fp32, increment_native.to(tl.float32))
        updated_native = updated_fp32.to(DTYPE)
        tl.store(velocity_out_ptr + offsets, scaled_native, mask=mask)
        tl.store(latent_out_ptr + offsets, updated_native, mask=mask)
        if pid < age_blocks:
            age_offsets = pid * BLOCK + tl.arange(0, BLOCK)
            age_mask = age_offsets < age_numel
            generated = tl.load(generated_ptr + age_offsets, mask=age_mask, other=0)
            ages = tl.load(age_ptr + age_offsets, mask=age_mask, other=0)
            tl.store(
                age_out_ptr + age_offsets,
                tl.where(generated != 0, ages + 1, ages),
                mask=age_mask,
            )

    _TRITON_KERNEL = reuse_stage_kernel
    return reuse_stage_kernel


def _launch_fused_stage(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    age: torch.Tensor,
    generated: torch.Tensor,
    ratio: float,
    dt: float | torch.Tensor,
    latent_out: torch.Tensor,
    velocity_out: torch.Tensor,
    age_out: torch.Tensor,
    block_size: int,
) -> None:
    _validate_stage(latent, velocity, dt, latent_out)
    if not latent.is_cuda:
        raise RuntimeError("Triton reuse stage requires CUDA")
    if age_out.shape != age.shape or age_out.dtype != age.dtype or age_out.device != age.device:
        raise ValueError("age output must match age")
    if velocity_out.shape != velocity.shape or velocity_out.dtype != velocity.dtype:
        raise ValueError("velocity output must match velocity")
    if velocity_out.device != velocity.device or not velocity_out.is_contiguous():
        raise ValueError("velocity output must be contiguous and on the input device")
    if not age_out.is_contiguous() or torch._C._overlaps(age_out, age):
        raise ValueError("age output must be contiguous and non-aliasing")
    if torch._C._overlaps(velocity_out, velocity) or torch._C._overlaps(velocity_out, latent):
        raise ValueError("velocity output must not alias inputs")
    import triton

    kernel = _get_triton_kernel()
    dtype = {
        torch.float32: triton.language.float32,
        torch.float16: triton.language.float16,
        torch.bfloat16: triton.language.bfloat16,
    }[latent.dtype]
    native_dt = isinstance(dt, torch.Tensor)
    dt_argument = dt if native_dt else float(dt)
    latent_numel = latent.numel()
    age_numel = age.numel()
    age_blocks = triton.cdiv(age_numel, block_size)
    grid = (triton.cdiv(latent_numel, block_size),)
    kernel[grid](
        latent,
        velocity,
        age,
        generated,
        dt_argument,
        latent_out,
        velocity_out,
        age_out,
        float(ratio),
        latent_numel,
        age_numel,
        age_blocks,
        IS_NATIVE_DT=native_dt,
        DTYPE=dtype,
        BLOCK=block_size,
        num_warps=4,
    )


def validate_snapshot_state(raw: dict) -> ReuseState:
    """Load only the explicit admission-state schema; never infer tensor roles."""

    names = (
        "latents",
        "prev_velocity",
        "prev_prev_velocity",
        "ages",
        "generated_mask",
    )
    missing = [name for name in names if name not in raw]
    if missing:
        raise ValueError(f"snapshot missing required fields: {missing}")
    state = ReuseState(
        latent=raw["latents"],
        history=(raw["prev_velocity"], raw["prev_prev_velocity"]),
        age=raw["ages"],
        generated=raw["generated_mask"],
    )
    state.validate()
    if state.latent.ndim != 3 or state.latent.shape[1:] != (2304, 64):
        raise ValueError("snapshot must have N=2304 tokens and channel=64")
    return state


def snapshot_sequence(root: str, steps: Sequence[int]) -> list[str]:
    """Return deterministic early/middle/late snapshot names for a driver."""

    return [f"{root}/step_{step:02d}.pt" for step in steps]
