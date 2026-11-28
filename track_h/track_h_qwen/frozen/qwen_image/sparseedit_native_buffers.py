# SPDX-License-Identifier: Apache-2.0
"""Stable Qwen model-call buffers and genuine CUDA graph replay counters."""

from __future__ import annotations

from collections.abc import Callable

import torch


class QwenStableBuffers:
    def __init__(
        self,
        active_count: int,
        channels: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        text_tokens: int,
        text_channels: int,
        timestep_dtype: torch.dtype | None = None,
    ) -> None:
        if active_count <= 0 or channels <= 0 or text_tokens <= 0 or text_channels <= 0:
            raise ValueError("stable Qwen buffer dimensions must be positive")
        self.latents = torch.empty(
            (1, active_count, channels), dtype=dtype, device=device
        )
        self.indices = torch.empty((active_count,), dtype=torch.long, device=device)
        self.timestep = torch.empty((1,), dtype=timestep_dtype or dtype, device=device)
        self.text = torch.empty(
            (1, text_tokens, text_channels), dtype=dtype, device=device
        )
        self.text_mask = torch.empty((1, text_tokens), dtype=torch.bool, device=device)
        self.graph = None
        self.graph_output = None
        self.capture_count = 0
        self.replay_count = 0

    def bind(
        self,
        *,
        latents: torch.Tensor,
        indices: torch.Tensor,
        timestep: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor | None,
    ) -> None:
        for name, source in (
            ("latents", latents),
            ("timestep", timestep),
            ("text", text),
        ):
            target = getattr(self, name)
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f"bound {name} differs from captured shape or dtype")
            target.copy_(source.detach())
        if indices.shape != self.indices.shape or indices.dtype != torch.long:
            raise ValueError("bound indices differ from captured active shape")
        self.indices.copy_(indices.detach())
        if text_mask is None:
            self.text_mask.fill_(True)
        else:
            if text_mask.shape != self.text_mask.shape:
                raise ValueError("text mask differs from captured branch")
            self.text_mask.copy_(text_mask.detach().to(dtype=torch.bool))

    def pointers(self) -> dict[str, int]:
        return {
            name: getattr(self, name).data_ptr()
            for name in ("latents", "indices", "timestep", "text", "text_mask")
        }

    def capture(
        self, forward: Callable[[], torch.Tensor], *, warmup_trials: int = 2
    ) -> None:
        if self.latents.device.type != "cuda":
            raise RuntimeError("CUDA graph capture requires CUDA stable buffers")
        stream = torch.cuda.Stream(device=self.latents.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup_trials):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward()
        self.graph, self.graph_output = graph, output
        self.capture_count += 1

    def replay(self) -> torch.Tensor:
        if self.graph is None or self.graph_output is None:
            raise RuntimeError("no captured Qwen graph is available")
        self.graph.replay()
        self.replay_count += 1
        return self.graph_output

    def reset_request(self) -> None:
        self.graph = None
        self.graph_output = None
        self.capture_count = 0
        self.replay_count = 0


__all__ = ["QwenStableBuffers"]
