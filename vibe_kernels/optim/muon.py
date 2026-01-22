# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muon optimizer backed by Triton Newton–Schulz kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from vibe_kernels.common.tensor_types import TensorLike

from .impl import triton_impl

Tensor = TensorLike
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@dataclass
class MuonConfig:
    """Static configuration switches for the Triton Muon kernels."""

    ns_steps: int = 5


class TritonMuon(torch.optim.Optimizer):
    """Single-GPU Muon optimizer using Triton Newton–Schulz primitives."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        *,
        nesterov: bool = True,
        ns_steps: Optional[int] = None,
        weight_decay: float = 0.01,
        config: Optional[MuonConfig] = None,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)
        self._config = config or MuonConfig()

    def _maybe_init_state(self, p: Tensor, state: dict) -> None:
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)

    @staticmethod
    def _should_use_triton(param: Tensor) -> bool:
        return param.device.type == "cuda" and param.dtype in _SUPPORTED_DTYPES

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group.get("weight_decay", 0.0)
            ns_steps = group.get("ns_steps") or self._config.ns_steps

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]
                self._maybe_init_state(p, state)
                buf = state["momentum_buffer"]

                if not grad.is_contiguous():
                    grad = grad.contiguous()

                if buf.device != grad.device:
                    buf = buf.to(device=grad.device)
                    state["momentum_buffer"] = buf

                # Standard momentum update (dampening = 0)
                buf.mul_(momentum).add_(grad)
                update_matrix = (grad + momentum * buf) if nesterov else buf

                if update_matrix.ndim < 2:
                    # 1D parameters fall back to momentum SGD style update.
                    p.add_(update_matrix, alpha=-lr)
                    continue

                original_shape = update_matrix.shape
                matrix_2d = update_matrix.reshape(update_matrix.shape[0], -1)

                if not self._should_use_triton(p):
                    # Fallback: use PyTorch orthogonalization
                    orth = triton_impl.fast_newton_schulz(
                        matrix_2d.float(), steps=ns_steps
                    )
                    orth = orth.to(matrix_2d.dtype)
                else:
                    orth = triton_impl.fast_newton_schulz(matrix_2d, steps=ns_steps)

                scale = max(1.0, matrix_2d.size(0) / matrix_2d.size(1)) ** 0.5
                orth_view = orth.view(original_shape)

                p.mul_(1 - lr * weight_decay)
                p.add_(orth_view, alpha=-lr * scale)

        return loss


class TritonDistMuon(TritonMuon):
    """Placeholder distributed Muon optimizer (not yet implemented)."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        *,
        nesterov: bool = True,
        ns_steps: Optional[int] = None,
        weight_decay: float = 0.01,
        config: Optional[MuonConfig] = None,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            config=config,
        )

    def step(self, closure=None):  # type: ignore[override]
        raise NotImplementedError("Distributed Triton Muon is not implemented")
