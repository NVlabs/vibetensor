# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def layernorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """PyTorch reference implementation of LayerNorm forward pass."""
    weight_cast = weight.to(x.dtype)
    bias_cast = bias.to(x.dtype) if bias is not None else None
    return F.layer_norm(x, weight_cast.shape, weight_cast, bias_cast, eps)


def layernorm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference implementation using float32 accumulation."""
    x_f32 = x.float()
    bias_f32 = bias.float() if bias is not None else None
    return F.layer_norm(x_f32, weight.shape, weight, bias_f32, eps).to(x.dtype)


def rstd_ref(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference implementation of reciprocal standard deviation."""
    x_f32 = x.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = ((x_f32 - mean) ** 2).mean(dim=-1)
    return 1.0 / torch.sqrt(var + eps)


def mean_ref(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation of mean."""
    return x.float().mean(dim=-1)
