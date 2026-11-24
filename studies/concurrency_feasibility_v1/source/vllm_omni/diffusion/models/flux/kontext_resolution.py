# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resolution selection helpers for FLUX.1 Kontext image editing."""

from __future__ import annotations

from typing import Any

import torch

PREFERRED_KONTEXT_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
)


def _four_dimensional_latent_pixel_dimensions(
    latents: torch.Tensor | None,
    *,
    vae_scale_factor: int,
) -> tuple[int, int] | None:
    if latents is None or latents.ndim != 4:
        return None
    if vae_scale_factor <= 0:
        raise ValueError("FLUX Kontext vae_scale_factor must be positive")
    latent_height, latent_width = latents.shape[-2:]
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError(f"FLUX Kontext latent spatial dimensions must be positive, got {tuple(latents.shape)}")
    return int(latent_height) * vae_scale_factor, int(latent_width) * vae_scale_factor


def normalize_kontext_reference_latents(
    image: Any,
    *,
    latent_channels: int,
) -> torch.Tensor | None:
    """Return reference latents as ``[batch, channels, height, width]`` when present."""

    if latent_channels <= 0:
        raise ValueError("FLUX Kontext latent_channels must be positive")
    candidates = image if isinstance(image, (list, tuple)) else [image]
    if not candidates or not all(isinstance(candidate, torch.Tensor) for candidate in candidates):
        return None

    normalized: list[torch.Tensor] = []
    for candidate in candidates:
        if candidate.ndim == 3 and candidate.shape[0] == latent_channels:
            candidate = candidate.unsqueeze(0)
        elif candidate.ndim == 4 and candidate.shape[1] == latent_channels:
            pass
        else:
            return None
        latent_height, latent_width = candidate.shape[-2:]
        if latent_height <= 0 or latent_width <= 0 or latent_height % 2 or latent_width % 2:
            raise ValueError(
                "FLUX Kontext reference latent spatial dimensions must be positive and divisible by 2, "
                f"got {tuple(candidate.shape)}"
            )
        if normalized:
            first = normalized[0]
            if candidate.shape[1:] != first.shape[1:]:
                raise ValueError("FLUX Kontext reference latent list elements must have the same shape")
            if candidate.dtype != first.dtype:
                raise ValueError("FLUX Kontext reference latent list elements must have the same dtype")
            if candidate.device != first.device:
                raise ValueError("FLUX Kontext reference latent list elements must use the same device")
        normalized.append(candidate)

    result = torch.cat(normalized, dim=0)
    if result.shape[0] != 1:
        raise ValueError(
            f"FLUX Kontext currently requires exactly one reference latent, got batch size {result.shape[0]}"
        )
    return result


def resolve_kontext_dimensions(
    image_height: int | None,
    image_width: int | None,
    height: int | None,
    width: int | None,
    *,
    generated_latents: torch.Tensor | None = None,
    reference_latents: torch.Tensor | None = None,
    vae_scale_factor: int = 8,
) -> tuple[int, int]:
    """Resolve generated image dimensions, preferring the reference aspect ratio."""

    if (height is None) != (width is None):
        raise ValueError("FLUX Kontext height and width must be specified together")
    generated_dimensions = _four_dimensional_latent_pixel_dimensions(
        generated_latents,
        vae_scale_factor=vae_scale_factor,
    )
    if height is not None and width is not None:
        requested_dimensions = int(height), int(width)
        if generated_dimensions is not None and generated_dimensions != requested_dimensions:
            raise ValueError(
                "FLUX Kontext 4D generated latents resolve to "
                f"{generated_dimensions[0]}x{generated_dimensions[1]}, but requested dimensions are "
                f"{requested_dimensions[0]}x{requested_dimensions[1]}"
            )
        return requested_dimensions
    if generated_dimensions is not None:
        return generated_dimensions
    reference_dimensions = _four_dimensional_latent_pixel_dimensions(
        reference_latents,
        vae_scale_factor=vae_scale_factor,
    )
    if reference_dimensions is not None:
        return reference_dimensions
    if image_height is None or image_width is None:
        return 1024, 1024

    aspect_ratio = image_width / image_height
    resolved_width, resolved_height = min(
        PREFERRED_KONTEXT_RESOLUTIONS,
        key=lambda size: abs(aspect_ratio - size[0] / size[1]),
    )
    return resolved_height, resolved_width


def validate_kontext_generated_latents(
    latents: torch.Tensor | None,
    *,
    batch_size: int,
    num_channels_latents: int,
    latent_height: int,
    latent_width: int,
    generated_id_count: int,
) -> None:
    """Validate unpacked or packed generated latents before reshaping or denoising."""

    if latents is None:
        return
    if latents.ndim == 4:
        expected_shape = (batch_size, num_channels_latents, latent_height, latent_width)
        if tuple(latents.shape) != expected_shape:
            raise ValueError(
                f"FLUX Kontext 4D generated latents have shape {tuple(latents.shape)}; expected {expected_shape} "
                "for the resolved dimensions"
            )
        return
    if latents.ndim == 3:
        if latents.shape[0] != batch_size:
            raise ValueError(
                f"FLUX Kontext packed generated latent batch size is {latents.shape[0]}, expected {batch_size}"
            )
        expected_channels = num_channels_latents * 4
        if latents.shape[2] != expected_channels:
            raise ValueError(
                f"FLUX Kontext packed generated latent channels are {latents.shape[2]}, expected {expected_channels}"
            )
        if latents.shape[1] != generated_id_count:
            raise ValueError(
                f"FLUX Kontext packed generated latent token count {latents.shape[1]} does not match "
                f"generated latent id count {generated_id_count}"
            )
        return
    raise ValueError(
        f"FLUX Kontext generated latents must have shape [batch, channels, height, width] or "
        f"[batch, tokens, channels], got {tuple(latents.shape)}"
    )


__all__ = [
    "PREFERRED_KONTEXT_RESOLUTIONS",
    "normalize_kontext_reference_latents",
    "resolve_kontext_dimensions",
    "validate_kontext_generated_latents",
]
