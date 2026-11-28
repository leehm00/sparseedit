# SPDX-License-Identifier: Apache-2.0
"""Optional CPU mask diagnostics for FLUX SparseEdit ablation runs.

Collection synchronizes tensors to the CPU and belongs in separate diagnostic
runs, outside latency measurements. No model or pipeline imports are required.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.models.flux.sparseedit_primitives import select_sparseedit_spatial_error_mask


def four_connected_component_count(mask: torch.Tensor, *, grid_height: int, grid_width: int) -> int:
    """Count components using only horizontal and vertical grid neighbors."""
    from scipy.ndimage import label

    grid = mask.detach().to(device="cpu", dtype=torch.bool).reshape(grid_height, grid_width).numpy()
    _, count = label(grid, structure=((0, 1, 0), (1, 1, 1), (0, 1, 0)))
    return int(count)


def collect_sparseedit_mask_diagnostics(
    *,
    active_mask: torch.Tensor,
    scores: torch.Tensor,
    grid_height: int,
    grid_width: int,
    active_count: int,
    progress: float,
    selection_progress: float,
) -> dict[str, Any]:
    """Compare the active mask with raw top-k on the same age-weighted scores."""
    raw_mask, _ = select_sparseedit_spatial_error_mask(
        update_strategy="spatial_error_smooth_decay",
        scores=scores,
        grid_height=grid_height,
        grid_width=grid_width,
        active_count=active_count,
        progress=1.0,
    )
    active_cpu = active_mask.detach().cpu()
    raw_cpu = raw_mask.detach().cpu()
    active_indices = active_cpu.reshape(-1).nonzero().flatten().tolist()
    raw_indices = raw_cpu.reshape(-1).nonzero().flatten().tolist()
    return {
        "grid_height": grid_height,
        "grid_width": grid_width,
        "progress": progress,
        "selection_progress": selection_progress,
        "active_count": len(active_indices),
        "raw_topk_active_count": len(raw_indices),
        "active_indices": active_indices,
        "raw_topk_indices": raw_indices,
        "active_component_count": four_connected_component_count(
            active_cpu, grid_height=grid_height, grid_width=grid_width
        ),
        "raw_topk_component_count": four_connected_component_count(
            raw_cpu, grid_height=grid_height, grid_width=grid_width
        ),
    }
