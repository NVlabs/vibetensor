# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradient norm utilities (planning scaffold)."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch.nn import Parameter
from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

from vibe_kernels.common.tensor_types import TensorLike

Tensor = TensorLike


def _collect_grads(parameters: Sequence[Tensor | Parameter]) -> list[Tensor]:
    grads: list[Tensor] = []
    for item in parameters:
        grad: Tensor | None
        grad_attr = getattr(item, "grad", None)
        if grad_attr is not None:
            grad = grad_attr
        else:
            grad = item
        if grad is None:
            continue
        grads.append(grad)
    return grads


def compute_global_grad_norm(
    parameters: Iterable[Tensor | Parameter], *, ord: float = 2.0
) -> Tensor:
    """Return the global L2 gradient norm for the provided parameters."""

    if ord != 2:
        raise NotImplementedError("Only L2 norm is currently supported")
    params_list = list(parameters)
    grads = _collect_grads(params_list)
    if not grads:
        return torch.zeros((), dtype=torch.float32)
    total_norm = _get_total_norm(
        grads, norm_type=ord, error_if_nonfinite=False, foreach=None
    )
    return total_norm.to(torch.float32)


def clip_grad_norm_(
    parameters: Iterable[Tensor | Parameter], max_norm: float, *, eps: float = 1e-6
) -> Tensor:
    """Clip gradients in-place and return the applied scaling factor."""

    params_list = list(parameters)
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    grads = _collect_grads(params_list)
    if not grads:
        return torch.ones((), dtype=torch.float32)
    total_norm = _get_total_norm(
        grads, norm_type=2.0, error_if_nonfinite=False, foreach=None
    )
    if total_norm == 0:
        return torch.ones((), dtype=torch.float32)
    clip_coef = max_norm / (total_norm + eps)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    _clip_grads_with_norm_(params_list, max_norm, total_norm, foreach=None)
    return clip_coef_clamped.to(torch.float32)
