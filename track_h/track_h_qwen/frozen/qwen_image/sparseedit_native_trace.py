# SPDX-License-Identifier: Apache-2.0
"""Strict Qwen-specific fixed-mask/action trajectory traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

QWEN_TRACE_KIND = "qwen_native_sparseedit_fixed_mask_trace"
QWEN_TRACE_SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class QwenTraceStep:
    step: int
    action: str
    indices: list[int]
    flow_time: float | None = None


@dataclass
class QwenFixedMaskTrace:
    model_family: str
    generated_tokens: int
    active_count: int
    steps: list[QwenTraceStep]
    identity: dict[str, Any]
    source_hashes: dict[str, str]

    @classmethod
    def create(
        cls,
        *,
        generated_tokens: int,
        active_count: int,
        steps: list[QwenTraceStep],
        identity=None,
        source_hashes=None,
    ):
        trace = cls(
            "Qwen-Image-Edit",
            generated_tokens,
            active_count,
            steps,
            dict(identity or {}),
            dict(source_hashes or {}),
        )
        trace.validate()
        return trace

    def validate(self) -> None:
        if self.model_family != "Qwen-Image-Edit":
            raise ValueError(
                "fixed-mask trace must be Qwen-Image-Edit; another Qwen version is rejected"
            )
        if (
            self.generated_tokens <= 0
            or not 0 < self.active_count <= self.generated_tokens
        ):
            raise ValueError("invalid Qwen generated/active token counts")
        if len(self.steps) != 28 or [row.step for row in self.steps] != list(range(28)):
            raise ValueError("Qwen trace must contain 28 contiguous actions")
        for row in self.steps:
            if row.action not in ("full", "sparse"):
                raise ValueError("trace action must be full or sparse")
            expected = (
                self.generated_tokens if row.action == "full" else self.active_count
            )
            if any(type(index) is not int for index in row.indices):
                raise ValueError(f"trace indices must be integers at step {row.step}")
            if len(row.indices) != expected or sorted(set(row.indices)) != row.indices:
                raise ValueError(f"malformed trace indices at step {row.step}")
            if row.indices and (
                row.indices[0] < 0 or row.indices[-1] >= self.generated_tokens
            ):
                raise ValueError(f"out-of-range trace indices at step {row.step}")
            if row.action == "full" and row.indices != list(
                range(self.generated_tokens)
            ):
                raise ValueError("full steps must cover every generated token")
        full = {row.step for row in self.steps if row.action == "full"}
        if full not in (set(range(8)) | {23}, set(range(28))):
            raise ValueError("Qwen trace must use W8/refresh23 or force-dense actions")

    def payload(self) -> dict[str, Any]:
        return {
            "kind": QWEN_TRACE_KIND,
            "schema_version": QWEN_TRACE_SCHEMA_VERSION,
            "model_family": self.model_family,
            "generated_tokens": self.generated_tokens,
            "active_count": self.active_count,
            "steps": [asdict(row) for row in self.steps],
            "identity": self.identity,
            "source_hashes": self.source_hashes,
        }

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(canonical_json(self.payload()) + "\n", encoding="utf-8")
        temporary.replace(path)


def load_qwen_trace(
    path: str | Path, *, expected_identity: dict[str, Any] | None = None
) -> QwenFixedMaskTrace:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            payload.get("kind") != QWEN_TRACE_KIND
            or payload.get("schema_version") != QWEN_TRACE_SCHEMA_VERSION
        ):
            raise ValueError("not a supported Qwen native SparseEdit trace")
        trace = QwenFixedMaskTrace(
            model_family=payload["model_family"],
            generated_tokens=payload["generated_tokens"],
            active_count=payload["active_count"],
            steps=[QwenTraceStep(**row) for row in payload["steps"]],
            identity=payload.get("identity", {}),
            source_hashes=payload.get("source_hashes", {}),
        )
        trace.validate()
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed Qwen trace: {path}") from error
    if expected_identity is not None:
        for key, value in expected_identity.items():
            if trace.identity.get(key) != value:
                raise ValueError(f"Qwen trace identity mismatch for {key}")
    return trace


__all__ = [
    "QWEN_TRACE_KIND",
    "QwenFixedMaskTrace",
    "QwenTraceStep",
    "canonical_json",
    "load_qwen_trace",
    "sha256_file",
]
