# SPDX-License-Identifier: Apache-2.0
"""Ordinary eager operators for the direct SparseEdit implementation control.

Keep weights, sparse semantics, attention backend and precision shared. These
operators remove the inherited packed-QKV and custom normalization/rotation
implementations. No artificial delays, dense inactive-token computation or
per-token Python loops are introduced.
"""

import torch
import torch.nn.functional as F


def project_qkv(module, x: torch.Tensor, sizes: tuple[int, int, int]):
    if sum(sizes) != module.weight.shape[0]:
        raise ValueError("direct QKV requires unsharded, unquantized projection weights")
    weight_views = module.weight.split(sizes, dim=0)
    bias_views = (None,) * 3 if module.bias is None else module.bias.split(sizes, dim=0)
    return tuple(F.linear(x, weight, bias) for weight, bias in zip(weight_views, bias_views))


def rms_norm(module, x: torch.Tensor) -> torch.Tensor:
    values = x.float()
    values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + module.variance_epsilon)
    # Match the model CUDA kernel: multiply the learned weight before BF16 conversion.
    return (values * module.weight.float()).to(x.dtype)


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved FLUX rotation with FP32 products and the shared BF16 inputs."""
    if cos.ndim == 3:
        cos, sin = cos[0], sin[0]
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    cosine, sine = cos.float()[None, :, None], sin.float()[None, :, None]
    real = pairs[..., 0] * cosine - pairs[..., 1] * sine
    imag = pairs[..., 1] * cosine + pairs[..., 0] * sine
    return torch.stack((real, imag), dim=-1).flatten(-2).to(x.dtype)
