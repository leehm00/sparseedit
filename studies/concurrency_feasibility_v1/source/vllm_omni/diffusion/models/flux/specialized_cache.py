# SPDX-License-Identifier: Apache-2.0
"""Owned, request-scoped K/V caches for the opt-in FLUX specialized runtime.

Two layouts are supported:

``raw``
    Per-layer image K/V cache holding the column-normalized (pre-RoPE) FLUX
    keys.  ``store_full`` fills pre-allocated storage in place and
    ``update_active`` scatters the current active rows in place.  No clone and
    no source view is retained, so the cache owns stable pointers for the
    lifetime of the request.

``rotated`` (attention-ready)
    Per-layer joint K/V buffers laid out as ``[text | generated | reference]``.
    Text rows are rewritten on every transformer forward; image rows are
    rewritten only for the active positions (or all rows on a dense refresh).
    The cached K is already rotated, so attention does not re-rotate the full
    image context.

The cache never re-allocates after its first (or explicitly bound) allocation:
refresh and sparse updates copy/scatter into the same storage.  Any shape or
token-count change is rejected instead of silently invalidating a graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

SpecializedCacheMode = Literal["raw", "rotated"]

_CACHE_MODES = frozenset({"raw", "rotated"})


class SpecializedCacheError(RuntimeError):
    """Raised when a specialized cache is used inconsistently."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


@dataclass
class _LayerStorage:
    key: torch.Tensor
    value: torch.Tensor
    pointer_before_refresh: tuple[int, int]
    writes: int = 0

    @property
    def pointers(self) -> tuple[int, int]:
        return (self.key.data_ptr(), self.value.data_ptr())


@dataclass
class SpecializedKVCache:
    """Owned per-branch K/V storage shared by every layer of one stream."""

    mode: SpecializedCacheMode
    text_token_count: int
    generated_token_count: int
    reference_token_count: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    label: str = ""
    #: ``torch`` keeps the reference rotation/write-back; ``fused`` routes
    #: active normalized-K rotation plus K/V submission through the targeted
    #: Triton kernel (with a declared PyTorch fallback).
    submission: Literal["torch", "fused"] = "torch"
    batch_size: int = 1
    shared_pool: Any | None = field(default=None, repr=False)
    batch_slot: int | None = None
    layers: dict[tuple[str, int], _LayerStorage] = field(default_factory=dict)
    operated_rows: int = 0
    write_events: int = 0
    text_writes: int = 0
    backends: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.batch_size not in (1, 2):
            raise ValueError("specialized cache supports one or two CFG branches")
        if self.shared_pool is not None and (
            (self.batch_size == 1 and self.batch_slot not in (0, 1))
            or (self.batch_size == 2 and self.batch_slot is not None)
        ):
            raise ValueError("CFG pool needs a serial slot or a joint B2 view")
        if self.mode not in _CACHE_MODES:
            raise ValueError(f"unsupported specialized cache mode: {self.mode!r}")
        if self.submission not in {"torch", "fused"}:
            raise ValueError(f"unsupported specialized submission mode: {self.submission!r}")
        self.text_token_count = _positive_int(self.text_token_count, "text_token_count")
        self.generated_token_count = _positive_int(self.generated_token_count, "generated_token_count")
        self.reference_token_count = _positive_int(self.reference_token_count, "reference_token_count")
        if self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_kv_heads and head_dim must be positive")
        if self.mode == "rotated" and self.text_token_count <= 0:
            raise ValueError("rotated specialized cache requires a non-empty text segment")
        if self.mode == "raw" and self.text_token_count != 0:
            raise ValueError("raw specialized cache stores image K/V only; text_token_count must be 0")

    @property
    def full_image_token_count(self) -> int:
        return self.generated_token_count + self.reference_token_count

    @property
    def joint_token_count(self) -> int:
        return self.text_token_count + self.full_image_token_count

    @property
    def owns_storage(self) -> bool:
        return True

    def sequence_length(self, mode: str | None = None) -> int:
        layout = mode or self.mode
        return self.joint_token_count if layout == "rotated" else self.full_image_token_count

    def _validate_input(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        rows: int,
        stream: str,
    ) -> None:
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise SpecializedCacheError("specialized cache expects torch tensors")
        if key.shape != value.shape:
            raise SpecializedCacheError(f"FLUX specialized key/value shape mismatch: {key.shape} and {value.shape}")
        expected_tail = (self.num_kv_heads, self.head_dim)
        if key.ndim != 4 or (key.shape[0], key.shape[2], key.shape[3]) != (self.batch_size, *expected_tail):
            raise SpecializedCacheError(
                f"FLUX specialized {stream} K/V must be "
                f"[{self.batch_size}, tokens, {expected_tail[0]}, {expected_tail[1]}], "
                f"got {tuple(key.shape)}"
            )
        if key.shape[1] != rows:
            raise SpecializedCacheError(
                f"FLUX specialized {stream} K/V row count mismatch: got {key.shape[1]}, expected {rows}"
            )
        if key.device != self.device or key.dtype != self.dtype:
            raise SpecializedCacheError(
                f"FLUX specialized {stream} K/V device/dtype mismatch: {key.device}/{key.dtype} vs "
                f"{self.device}/{self.dtype}"
            )

    def _allocate(self, stream: str, layer_index: int) -> _LayerStorage:
        rows = self.sequence_length()
        shape = (self.batch_size, rows, self.num_kv_heads, self.head_dim)
        if self.shared_pool is None:
            key = torch.empty(shape, dtype=self.dtype, device=self.device)
            value = torch.empty(shape, dtype=self.dtype, device=self.device)
        else:
            key, value = self.shared_pool.allocate((stream, layer_index), shape, dtype=self.dtype, device=self.device)
            if self.batch_slot is not None:
                key, value = key[self.batch_slot : self.batch_slot + 1], value[self.batch_slot : self.batch_slot + 1]
        storage = _LayerStorage(key=key, value=value, pointer_before_refresh=storage_pointers(key, value))
        self.layers[(stream, layer_index)] = storage
        return storage

    def ensure_layer(self, stream: str, layer_index: int) -> _LayerStorage:
        storage = self.layers.get((stream, layer_index))
        if storage is None:
            storage = self._allocate(stream, layer_index)
        return storage

    def submission_backend(self) -> str:
        """Return the configured submission path for this cache."""
        return self.submission

    def require_layer(self, stream: str, layer_index: int) -> _LayerStorage:
        if self.shared_pool is not None:
            self.shared_pool.require_full((stream, layer_index), self.batch_slot)
            if (stream, layer_index) not in self.layers:
                self._allocate(stream, layer_index)
        storage = self.layers.get((stream, layer_index))
        if storage is None:
            raise SpecializedCacheError(
                f"FLUX specialized {stream} layer {layer_index} was updated before its full cache was stored"
            )
        return storage

    # ------------------------------------------------------------------
    # Raw image cache (the strong in-place baseline layout).
    # ------------------------------------------------------------------
    def store_full_image(
        self,
        stream: str,
        layer_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode != "raw":
            raise SpecializedCacheError("store_full_image is only valid for the raw specialized layout")
        self._validate_input(key, value, rows=self.full_image_token_count, stream=stream)
        storage = self.ensure_layer(stream, layer_index)
        storage.key.copy_(key.detach())
        storage.value.copy_(value.detach())
        if self.shared_pool is not None:
            self.shared_pool.mark_full((stream, layer_index), self.batch_slot)
        storage.writes += 1
        self.write_events += 1
        self.operated_rows += int(key.shape[1])
        return storage.key, storage.value

    def update_active_image(
        self,
        stream: str,
        layer_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
        image_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode != "raw":
            raise SpecializedCacheError("update_active_image is only valid for the raw specialized layout")
        storage = self.require_layer(stream, layer_index)
        indices = _validate_scatter_indices(
            image_indices,
            rows=int(key.shape[1]) if isinstance(key, torch.Tensor) else -1,
            limit=self.full_image_token_count,
            device=self.device,
        )
        self._validate_input(key, value, rows=int(indices.numel()), stream=stream)
        storage.key.index_copy_(1, indices, key.detach())
        storage.value.index_copy_(1, indices, value.detach())
        storage.writes += 1
        self.write_events += 2
        self.operated_rows += int(key.shape[1])
        return storage.key, storage.value

    # ------------------------------------------------------------------
    # Attention-ready joint cache ([text | generated | reference]).
    # ------------------------------------------------------------------
    def store_full_joint(
        self,
        stream: str,
        layer_index: int,
        rotated_image_key: torch.Tensor,
        image_value: torch.Tensor,
        rotated_text_key: torch.Tensor,
        text_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode != "rotated":
            raise SpecializedCacheError("store_full_joint is only valid for the rotated specialized layout")
        self._validate_input(rotated_image_key, image_value, rows=self.full_image_token_count, stream=stream)
        self._validate_input(
            rotated_text_key,
            text_value,
            rows=self.text_token_count,
            stream=f"{stream} text",
        )
        storage = self.ensure_layer(stream, layer_index)
        storage.key[:, : self.text_token_count].copy_(rotated_text_key.detach())
        storage.value[:, : self.text_token_count].copy_(text_value.detach())
        storage.key[:, self.text_token_count :].copy_(rotated_image_key.detach())
        storage.value[:, self.text_token_count :].copy_(image_value.detach())
        if self.shared_pool is not None:
            self.shared_pool.mark_full((stream, layer_index), self.batch_slot)
        storage.writes += 1
        self.write_events += 4
        self.text_writes += 1
        self.operated_rows += int(rotated_image_key.shape[1])
        return storage.key, storage.value

    def update_active_joint(
        self,
        stream: str,
        layer_index: int,
        rotated_image_key: torch.Tensor,
        image_value: torch.Tensor,
        rotated_text_key: torch.Tensor,
        text_value: torch.Tensor,
        scatter_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode != "rotated":
            raise SpecializedCacheError("update_active_joint is only valid for the rotated specialized layout")
        storage = self.require_layer(stream, layer_index)
        positions = _validate_scatter_indices(
            scatter_positions,
            rows=int(rotated_image_key.shape[1]) if isinstance(rotated_image_key, torch.Tensor) else -1,
            limit=self.joint_token_count,
            device=self.device,
        )
        self._validate_input(
            rotated_image_key,
            image_value,
            rows=int(positions.numel()),
            stream=stream,
        )
        self._validate_input(
            rotated_text_key,
            text_value,
            rows=self.text_token_count,
            stream=f"{stream} text",
        )
        storage.key[:, : self.text_token_count].copy_(rotated_text_key.detach())
        storage.value[:, : self.text_token_count].copy_(text_value.detach())
        storage.key.index_copy_(1, positions, rotated_image_key.detach())
        storage.value.index_copy_(1, positions, image_value.detach())
        storage.writes += 1
        self.write_events += 4
        self.text_writes += 1
        self.operated_rows += int(rotated_image_key.shape[1])
        return storage.key, storage.value

    def record_submission(self, rows: int, *, backend: str | None = None, text: bool = True) -> None:
        """Account for a low-level fused submission without touching storage."""
        self.write_events += 4 if text else 2
        self.operated_rows += int(rows)
        if text:
            self.text_writes += 1
        if backend is not None:
            self.backends.add(str(backend))

    # ------------------------------------------------------------------
    # Diagnostics and lifecycle.
    # ------------------------------------------------------------------
    def pointer_witness(self) -> dict[str, Any]:
        witness: dict[str, Any] = {
            "label": self.label,
            "batch_size": self.batch_size,
            "shared_cfg_pool": self.shared_pool is not None,
            "batch_slot": self.batch_slot,
            "mode": self.mode,
            "shape": {
                "text_token_count": self.text_token_count,
                "generated_token_count": self.generated_token_count,
                "reference_token_count": self.reference_token_count,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
            },
            "layers": {},
            "operated_rows": self.operated_rows,
            "write_events": self.write_events,
            "text_writes": self.text_writes,
            "backends": sorted(self.backends),
        }
        for (stream, layer_index), storage in sorted(self.layers.items(), key=lambda item: (item[0][0], item[0][1])):
            key_ptr, value_ptr = storage.pointers
            witness["layers"][f"{stream}:{layer_index}"] = {
                "key_ptr": key_ptr,
                "value_ptr": value_ptr,
                "fully_owned": storage.key.is_contiguous() and storage.value.is_contiguous(),
                "writes": storage.writes,
            }
        return witness

    def pointers_for(self, stream: str, layer_index: int) -> tuple[int, int]:
        return self.require_layer(stream, layer_index).pointers

    def clear(self) -> None:
        self.layers.clear()

    def release(self) -> None:
        self.clear()


def storage_pointers(key: torch.Tensor, value: torch.Tensor) -> tuple[int, int]:
    return (key.data_ptr(), value.data_ptr())


def _validate_scatter_indices(
    indices: Any,
    *,
    rows: int,
    limit: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(indices, torch.Tensor):
        raise SpecializedCacheError("specialized scatter indices must be a torch tensor")
    if indices.ndim == 2 and indices.shape[0] == 1:
        flat = indices[0]
    elif indices.ndim == 1:
        flat = indices
    else:
        raise SpecializedCacheError(
            f"specialized scatter indices must be [1, rows] or [rows], got {tuple(indices.shape)}"
        )
    if flat.numel() != rows:
        raise SpecializedCacheError(f"specialized scatter index count {flat.numel()} does not match active rows {rows}")
    if flat.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise SpecializedCacheError(f"unsupported specialized scatter index dtype: {flat.dtype}")
    if flat.numel() == 0:
        raise SpecializedCacheError("specialized scatter requires at least one active row")
    if limit <= 0:
        raise SpecializedCacheError("specialized scatter limit must be positive")
    # Deliberately no ``min()/max()/unique()`` here: those would synchronize
    # the host once per layer.  Exact-K validity is checked once when a trace
    # is recorded/loaded (see ``specialized_plan``).
    return flat.to(device=device, dtype=torch.long)


def cache_allocation_bytes(
    *,
    mode: SpecializedCacheMode,
    text_token_count: int,
    generated_token_count: int,
    reference_token_count: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    num_layers: int,
) -> int:
    """Return the persistent K/V capacity implied by one branch cache."""
    rows = (
        text_token_count + generated_token_count + reference_token_count
        if mode == "rotated"
        else (generated_token_count + reference_token_count)
    )
    element_size = torch.empty(0, dtype=dtype).element_size()
    return int(rows * num_kv_heads * head_dim * element_size * 2 * num_layers)


def parse_cache_mode(variant: str, override: str = "auto") -> SpecializedCacheMode:
    if variant not in {"inplace", "transformed", "fused", "graph", "graph_reuse"}:
        raise ValueError(f"variant {variant!r} does not use a specialized cache")
    if override not in {"auto", "raw", "rotated"}:
        raise ValueError("cache layout override must be auto, raw or rotated")
    if override != "auto":
        return override
    if variant in {"inplace"}:
        return "raw"
    if variant in {"transformed", "fused", "graph", "graph_reuse"}:
        return "rotated"
    raise ValueError(f"variant {variant!r} does not use a specialized cache")


def finite_check(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)


__all__ = [
    "SpecializedCacheError",
    "SpecializedCacheMode",
    "SpecializedKVCache",
    "cache_allocation_bytes",
    "parse_cache_mode",
    "storage_pointers",
]
