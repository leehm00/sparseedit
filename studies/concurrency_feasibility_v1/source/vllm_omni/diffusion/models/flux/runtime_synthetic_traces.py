#!/usr/bin/env python3
"""Deterministic synthetic token traces and pure-CPU descriptor utilities.

Trace contract
--------------
``make_trace(n, k, morphology, seed)`` returns a sorted, distinct ``int64``
array of exactly ``k`` positions in ``[0, n)``.  Morphologies are generation
recipes only; the runtime policy never receives the morphology label and uses
only measured trace features.

Descriptor contract
-------------------
``build_compact_chunks(indices, g)`` partitions the *compact* row order into
consecutive chunks of size ``g``.  For each chunk it reports the compact start
row, the first source position (``base``), the chunk length and whether the
source positions are contiguous.  Because indices are sorted and distinct,
``indices[start + length - 1] - indices[start] == length - 1`` is a valid
contiguity test.

No PyTorch/Triton import occurs in this module.
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable, Sequence

import numpy as np

MORPHOLOGIES = ("clustered", "mixed", "scattered")
INT64 = np.int64
RUN_BIN_EDGES = (0.0, 2.0, 8.0)  # bins: [0,2), [2,8), [8, inf)
RUN_BIN_LABELS = ("0-2", "2-8", "8+")


class TraceError(ValueError):
    """Raised for invalid trace specifications or malformed indices."""


def _validate_spec(n: int, k: int) -> tuple[int, int]:
    if not isinstance(n, (int, np.integer)) or int(n) <= 0:
        raise TraceError(f"n must be a positive integer, got {n!r}")
    if not isinstance(k, (int, np.integer)) or int(k) <= 0:
        raise TraceError(f"k must be a positive integer, got {k!r}")
    n, k = int(n), int(k)
    if k > n:
        raise TraceError(f"k={k} exceeds n={n}")
    return n, k


def run_lengths(indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return run start offsets and run lengths for sorted distinct indices."""
    arr = np.asarray(indices, dtype=INT64)
    if arr.ndim != 1:
        raise TraceError("indices must be one-dimensional")
    if arr.size == 0:
        return np.empty(0, dtype=INT64), np.empty(0, dtype=INT64)
    breaks = np.flatnonzero(np.diff(arr) != 1) + 1
    starts = np.concatenate((np.asarray([0], dtype=INT64), breaks.astype(INT64)))
    ends = np.concatenate((breaks.astype(INT64), np.asarray([arr.size], dtype=INT64)))
    return starts.astype(INT64), (ends - starts).astype(INT64)


def _validate_indices(indices: Sequence[int], n: int | None = None) -> np.ndarray:
    arr = np.asarray(indices)
    if arr.ndim != 1:
        raise TraceError("indices must be one-dimensional")
    if arr.size == 0:
        return arr.astype(INT64)
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.equal(np.mod(arr, 1), 0)):
            raise TraceError("indices must be integers")
    arr = arr.astype(INT64)
    if np.any(arr < 0):
        raise TraceError("indices must be non-negative")
    if n is not None and np.any(arr >= int(n)):
        raise TraceError(f"indices must lie in [0, {n})")
    if arr.size > 1 and np.any(np.diff(arr) <= 0):
        raise TraceError("indices must be sorted and strictly increasing (distinct)")
    return arr


def _clustered_positions(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """One/few contiguous regions with variable origins, exact K total."""
    n_clusters = int(rng.integers(1, 4))
    n_clusters = min(n_clusters, k)
    chosen: set[int] = set()
    remaining = k
    for cluster_index in range(n_clusters):
        clusters_left = n_clusters - cluster_index
        if clusters_left == 1:
            size = remaining
        else:
            max_size = remaining - (clusters_left - 1)
            size = int(rng.integers(1, max_size + 1))
        remaining -= size
        max_origin = n - size
        origin = int(rng.integers(0, max_origin + 1))
        for position in range(origin, origin + size):
            if 0 <= position < n:
                chosen.add(position)
    if len(chosen) < k:
        available = np.setdiff1d(
            np.arange(n, dtype=INT64),
            np.fromiter(sorted(chosen), dtype=INT64, count=len(chosen)),
            assume_unique=True,
        )
        need = k - len(chosen)
        if need > available.size:
            raise TraceError("internal clustered generation could not fill exact K")
        chosen.update(int(v) for v in rng.choice(available, size=need, replace=False))
    if len(chosen) != k:
        raise TraceError(f"clustered generation produced {len(chosen)} != k={k}")
    return np.sort(np.fromiter(sorted(chosen), dtype=INT64, count=k))


def make_trace(n: int, k: int, morphology: str, seed: int) -> np.ndarray:
    """Generate a deterministic, sorted, distinct exact-K trace.

    - ``clustered``: one to three contiguous regions with variable origins.
    - ``mixed``: roughly half clustered positions plus uniform positions drawn
      from the disjoint complement.
    - ``scattered``: uniform sampling without replacement.
    """
    n, k = _validate_spec(n, k)
    if morphology not in MORPHOLOGIES:
        raise TraceError(
            f"unknown morphology {morphology!r}; expected one of {MORPHOLOGIES}"
        )
    if not isinstance(seed, (int, np.integer)):
        raise TraceError(f"seed must be an integer, got {seed!r}")
    rng = np.random.default_rng(int(seed))
    if morphology == "clustered":
        positions = _clustered_positions(rng, n, k)
    elif morphology == "scattered":
        positions = np.sort(
            rng.choice(n, size=k, replace=False).astype(INT64)
        )
    else:  # mixed
        n_clustered = (k + 1) // 2
        clustered = _clustered_positions(rng, n, n_clustered)
        if k - n_clustered:
            available = np.setdiff1d(
                np.arange(n, dtype=INT64), clustered, assume_unique=True
            )
            need = k - n_clustered
            fill = rng.choice(available, size=need, replace=False).astype(INT64)
            positions = np.sort(np.concatenate((clustered, fill)))
        else:
            positions = clustered
    positions = _validate_indices(positions, n)
    if positions.size != k:
        raise TraceError(f"trace has {positions.size} positions, expected {k}")
    return positions.astype(INT64)


def trace_features(indices: Sequence[int], n: int) -> dict[str, float | int]:
    """Measured trace features; morphology labels are intentionally absent."""
    n, k = _validate_spec(int(n), int(len(indices)))
    arr = _validate_indices(indices, n)
    starts, lengths = run_lengths(arr)
    run_count = int(lengths.size)
    positions = float(arr.size)
    positions_in_ge4 = float(lengths[lengths >= 4].sum()) if run_count else 0.0
    positions_in_ge16 = float(lengths[lengths >= 16].sum()) if run_count else 0.0
    runs_ge4 = float((lengths >= 4).sum()) if run_count else 0.0
    runs_ge16 = float((lengths >= 16).sum()) if run_count else 0.0
    return {
        "n": int(n),
        "k": int(arr.size),
        "density": float(arr.size) / float(n),
        "run_count": run_count,
        "mean_run_length": float(arr.size) / run_count if run_count else 0.0,
        "median_run_length": float(np.median(lengths)) if run_count else 0.0,
        "max_run_length": int(lengths.max()) if run_count else 0,
        "frac_positions_in_runs_ge4": positions_in_ge4 / positions,
        "frac_positions_in_runs_ge16": positions_in_ge16 / positions,
        "frac_runs_ge4": runs_ge4 / run_count if run_count else 0.0,
        "frac_runs_ge16": runs_ge16 / run_count if run_count else 0.0,
    }


def build_compact_chunks(indices: Sequence[int], g: int) -> dict[str, object]:
    """CPU block descriptor over consecutive compact chunks of size ``g``.

    Returns ``chunk_start`` (compact row offset), ``base`` (first source index),
    ``length`` (rows in this chunk, including an incomplete tail) and
    ``contiguous`` (whether the source rows are exactly base..base+length-1).
    """
    if not isinstance(g, (int, np.integer)) or int(g) <= 0:
        raise TraceError(f"g must be a positive integer, got {g!r}")
    g = int(g)
    arr = _validate_indices(indices)
    k = int(arr.size)
    if k == 0:
        empty = np.empty(0, dtype=INT64)
        return {
            "chunk_start": empty,
            "base": empty,
            "length": empty,
            "contiguous": np.empty(0, dtype=bool),
            "chunk_size": g,
            "n_chunks": 0,
            "k": 0,
        }
    n_chunks = (k + g - 1) // g
    chunk_start = np.arange(n_chunks, dtype=INT64) * g
    length = np.minimum(g, k - chunk_start).astype(INT64)
    base = arr[chunk_start].astype(INT64)
    last = arr[chunk_start + length - 1].astype(INT64)
    contiguous = (last - base) == (length - 1)
    return {
        "chunk_start": chunk_start,
        "base": base,
        "length": length,
        "contiguous": contiguous.astype(bool),
        "chunk_size": g,
        "n_chunks": int(n_chunks),
        "k": k,
    }


def descriptor_stats(descriptor: dict[str, object]) -> dict[str, float | int]:
    n_chunks = int(descriptor["n_chunks"])
    contiguous = np.asarray(descriptor["contiguous"], dtype=bool)
    length = np.asarray(descriptor["length"], dtype=INT64)
    rows_contiguous = int(length[contiguous].sum()) if n_chunks else 0
    k = int(descriptor["k"])
    return {
        "n_chunks": n_chunks,
        "chunk_size": int(descriptor["chunk_size"]),
        "k": k,
        "contiguous_chunks": int(contiguous.sum()),
        "contiguous_fraction_chunks": (
            float(contiguous.mean()) if n_chunks else 0.0
        ),
        "rows_in_contiguous_chunks": rows_contiguous,
        "contiguous_fraction_rows": (rows_contiguous / k if k else 0.0),
    }


def mean_run_length_bin(mean_run_length: float) -> str:
    """Frozen measured-feature binning [0,2), [2,8), [8, inf)."""
    value = float(mean_run_length)
    if value < 0.0:
        raise TraceError("mean_run_length must be non-negative")
    if value < RUN_BIN_EDGES[1]:
        return RUN_BIN_LABELS[0]
    if value < RUN_BIN_EDGES[2]:
        return RUN_BIN_LABELS[1]
    return RUN_BIN_LABELS[2]


def policy_cell_key(n: int, d: int, k: int, mean_run_length: float) -> str:
    """Policy key: known geometry plus measured mean-run-length bin.

    The morphology label is deliberately not part of this key.
    """
    return f"{int(n)}|{int(d)}|{int(k)}|{mean_run_length_bin(mean_run_length)}"


def cpu_reference_gather(src: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Pure-NumPy gather reference used by CPU tests."""
    array = np.asarray(src)
    idx = _validate_indices(indices, n=array.shape[0])
    return np.array(array[idx], copy=True)


def cpu_reference_scatter_kv(
    k_src: np.ndarray,
    v_src: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy paired scatter reference; inactive rows are preserved."""
    k_out = np.array(k_cache, copy=True)
    v_out = np.array(v_cache, copy=True)
    idx = _validate_indices(indices, n=k_out.shape[0])
    if v_out.shape[0] != k_out.shape[0]:
        raise TraceError("K and V caches must have the same row count")
    k_out[idx] = np.asarray(k_src)
    v_out[idx] = np.asarray(v_src)
    return k_out, v_out


def _demo(args: argparse.Namespace) -> int:
    k = int(args.k) if args.k is not None else max(1, int(round(args.n * args.rho)))
    print(
        json.dumps(
            {
                "n": args.n,
                "k": k,
                "seed": args.seed,
                "chunk_size": args.chunk_size,
                "traces": {
                    morphology: {
                        "features": trace_features(
                            make_trace(args.n, k, morphology, args.seed), args.n
                        ),
                        "descriptor": descriptor_stats(
                            build_compact_chunks(
                                make_trace(args.n, k, morphology, args.seed),
                                args.chunk_size,
                            )
                        ),
                    }
                    for morphology in MORPHOLOGIES
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return _demo(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
