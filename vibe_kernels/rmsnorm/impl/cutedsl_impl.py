# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib

import torch

from vibe_kernels.common.tensor_types import TensorLike

_rms_module = importlib.import_module(
    "kernel_factory.rmsnorm.impl.cutedsl_rmsnorm.rmsnorm"
)
_CUTEDSL_CLASS = getattr(_rms_module, "CuTeDSLRMSNorm", None)
if _CUTEDSL_CLASS is not None:
    CuTeDSLRMSNorm = _CUTEDSL_CLASS
else:  # pragma: no cover - optional backend

    class CuTeDSLRMSNorm(torch.nn.Module):  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("CuTeDSL RMSNorm backend is not available")

        def forward(self, x: TensorLike):  # type: ignore[override]
            raise RuntimeError("CuTeDSL RMSNorm backend is not available")


def rmsnorm(
    x: TensorLike,
    weight: TensorLike | None = None,
    *,
    bias: TensorLike | None = None,
    residual: TensorLike | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    eps: float = 1e-6,
    prenorm: bool = False,
):
    return _rms_module.rmsnorm(
        x,
        weight=weight,
        bias=bias,
        residual=residual,
        out_dtype=out_dtype,
        residual_dtype=residual_dtype,
        eps=eps,
        prenorm=prenorm,
    )


def rmsnorm_fwd(
    x: TensorLike,
    weight: TensorLike | None = None,
    *,
    bias: TensorLike | None = None,
    residual: TensorLike | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
):
    return _rms_module.rmsnorm_fwd(
        x,
        weight=weight,
        bias=bias,
        residual=residual,
        out_dtype=out_dtype,
        residual_dtype=residual_dtype,
        eps=eps,
        store_rstd=store_rstd,
    )


def rmsnorm_bwd(
    x: TensorLike,
    weight: TensorLike | None,
    dout: TensorLike,
    rstd: TensorLike,
    *,
    dresidual_out: TensorLike | None = None,
    has_bias: bool = False,
    has_residual: bool = False,
):
    return _rms_module.rmsnorm_bwd(
        x,
        weight,
        dout,
        rstd,
        dresidual_out=dresidual_out,
        has_bias=has_bias,
        has_residual=has_residual,
    )


def is_available() -> bool:
    return True


__all__ = [
    "rmsnorm",
    "rmsnorm_fwd",
    "rmsnorm_bwd",
    "CuTeDSLRMSNorm",
    "is_available",
]
