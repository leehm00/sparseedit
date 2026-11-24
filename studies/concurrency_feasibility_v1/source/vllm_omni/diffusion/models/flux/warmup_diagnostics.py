# SPDX-License-Identifier: Apache-2.0
"""Tensor-only warmup and sparse-mask diagnostics for FLUX routing."""

from __future__ import annotations

import math
from typing import Any

import torch


def infer_token_grid(num_tokens: int) -> tuple[int, int]:
    """Return the most square integer grid whose area equals ``num_tokens``."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    height = math.isqrt(num_tokens)
    while height > 1 and num_tokens % height:
        height -= 1
    return height, num_tokens // height


def binary_mask_spatial_stats(mask: torch.Tensor, *, grid_height: int, grid_width: int) -> dict[str, float]:
    """Measure connected components, largest component, and active bounding box."""
    if mask.ndim == 2:
        mask = mask[0]
    flat = mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
    if flat.numel() != grid_height * grid_width:
        raise ValueError(f"mask has {flat.numel()} tokens, expected grid area {grid_height * grid_width}")
    active_count = int(flat.sum().item())
    if not active_count:
        return {"component_count": 0.0, "largest_component_ratio": 0.0, "active_bbox_area_ratio": 0.0}
    grid = flat.reshape(grid_height, grid_width)
    visited = torch.zeros_like(grid)
    components: list[int] = []
    positions = torch.nonzero(grid, as_tuple=False)
    for row_value, column_value in positions.tolist():
        if visited[row_value, column_value]:
            continue
        stack = [(row_value, column_value)]
        visited[row_value, column_value] = True
        size = 0
        while stack:
            row, column = stack.pop()
            size += 1
            for next_row, next_column in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                if (
                    0 <= next_row < grid_height
                    and 0 <= next_column < grid_width
                    and grid[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
        components.append(size)
    bbox = (int(positions[:, 0].max()) - int(positions[:, 0].min()) + 1) * (
        int(positions[:, 1].max()) - int(positions[:, 1].min()) + 1
    )
    return {
        "component_count": float(len(components)),
        "largest_component_ratio": float(max(components) / active_count),
        "active_bbox_area_ratio": float(bbox / (grid_height * grid_width)),
    }


def _spatial_energy(values: torch.Tensor, *, grid_height: int, grid_width: int) -> tuple[float, float]:
    spatial = values.float().reshape(values.shape[0], grid_height, grid_width, -1)
    horizontal = spatial[:, :, 1:, :] - spatial[:, :, :-1, :]
    vertical = spatial[:, 1:, :, :] - spatial[:, :-1, :, :]
    edge_energy = 0.5 * (horizontal.square().mean() + vertical.square().mean())
    laplacian = torch.zeros_like(spatial)
    laplacian[:, 1:-1, 1:-1, :] = (
        4 * spatial[:, 1:-1, 1:-1, :]
        - spatial[:, :-2, 1:-1, :]
        - spatial[:, 2:, 1:-1, :]
        - spatial[:, 1:-1, :-2, :]
        - spatial[:, 1:-1, 2:, :]
    )
    high_frequency = laplacian.square().mean()
    return float(edge_energy.item()), float((high_frequency / spatial.square().mean().clamp(min=1e-12)).item())


def collect_warmup_step_diagnostics(
    *,
    latents: torch.Tensor,
    previous_latents: torch.Tensor | None,
    previous_previous_latents: torch.Tensor | None,
    pred_x0: torch.Tensor,
    reference_latents: torch.Tensor | None,
    grid_height: int | None = None,
    grid_width: int | None = None,
    topk_fraction: float = 0.10,
) -> dict[str, Any]:
    """Compute routing descriptors from one tensor-only FLUX warmup step."""
    num_tokens = int(latents.shape[1])
    if grid_height is None or grid_width is None or grid_height * grid_width != num_tokens:
        grid_height, grid_width = infer_token_grid(num_tokens)
    delta = torch.zeros_like(latents) if previous_latents is None else latents - previous_latents
    delta_norm = torch.linalg.vector_norm(delta.float(), dim=-1)
    latent_norm = torch.linalg.vector_norm(latents.float(), dim=-1).mean().clamp(min=1e-12)
    instability = 0.0
    if previous_latents is not None and previous_previous_latents is not None:
        previous_delta = previous_latents - previous_previous_latents
        cosine = torch.nn.functional.cosine_similarity(delta.float(), previous_delta.float(), dim=-1, eps=1e-8)
        instability = float((1.0 - cosine).mean())
    active_count = max(1, min(num_tokens, math.ceil(num_tokens * topk_fraction)))
    indices = torch.topk(delta_norm, active_count, dim=1).indices
    topk_mask = torch.zeros_like(delta_norm, dtype=torch.bool)
    topk_mask.scatter_(1, indices, True)
    spatial = binary_mask_spatial_stats(topk_mask, grid_height=grid_height, grid_width=grid_width)
    threshold = delta_norm.mean(dim=1, keepdim=True) + delta_norm.std(dim=1, keepdim=True)
    pred_x0_generated = pred_x0[:, :num_tokens, :]
    edge_energy, high_frequency = _spatial_energy(pred_x0_generated, grid_height=grid_height, grid_width=grid_width)
    ref_diff = None
    if reference_latents is not None:
        count = min(num_tokens, int(reference_latents.shape[1]))
        difference = pred_x0_generated[:, :count, :].float() - reference_latents[:, :count, :].float()
        ref_diff = float(torch.linalg.vector_norm(difference, dim=-1).mean())
    return {
        "latent_step_speed": float(delta_norm.mean()),
        "delta_relative": float(delta_norm.mean() / latent_norm),
        "latent_update_direction_instability": instability,
        "latent_change_topk_active_bbox_area_ratio": spatial["active_bbox_area_ratio"],
        "latent_change_topk_component_count": spatial["component_count"],
        "latent_change_topk_largest_component_ratio": spatial["largest_component_ratio"],
        "latent_change_residual_active_fraction": float((delta_norm > threshold).float().mean()),
        "pred_x0_spatial_std": float(pred_x0_generated.float().std()),
        "edge_energy": edge_energy,
        "high_freq_ratio": high_frequency,
        "ref_diff": ref_diff,
        "diagnostic_grid_height": int(grid_height),
        "diagnostic_grid_width": int(grid_width),
    }


__all__ = ["binary_mask_spatial_stats", "collect_warmup_step_diagnostics", "infer_token_grid"]
