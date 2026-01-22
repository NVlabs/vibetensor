# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from vibe_kernels.common.tensor_types import TensorLike

from .impl.triton_impl import _RMSNormFn, TritonRMSNorm

# Expose Triton implementation as the default RMSNorm class
RMSNorm = TritonRMSNorm

# Optional CuTeDSL backend
try:
    from .impl import cutedsl_impl

    _cutedsl_available = True
except ImportError:
    cutedsl_impl = None  # type: ignore[assignment]
    _cutedsl_available = False


def is_cutedsl_available() -> bool:
    return _cutedsl_available


def _require_cutedsl() -> None:
    if not _cutedsl_available:
        raise RuntimeError("CuTeDSL RMSNorm backend is not available")


def cutedsl_rmsnorm(
    x: TensorLike,
    weight: Optional[TensorLike] = None,
    *,
    bias: Optional[TensorLike] = None,
    residual: Optional[TensorLike] = None,
    out_dtype: Any = None,
    residual_dtype: Any = None,
    eps: float = 1e-6,
    prenorm: bool = False,
) -> TensorLike:
    _require_cutedsl()
    assert cutedsl_impl is not None
    return cutedsl_impl.rmsnorm(
        x,
        weight=weight,
        bias=bias,
        residual=residual,
        out_dtype=out_dtype,
        residual_dtype=residual_dtype,
        eps=eps,
        prenorm=prenorm,
    )


def cutedsl_rmsnorm_forward(
    x: TensorLike,
    weight: Optional[TensorLike] = None,
    *,
    bias: Optional[TensorLike] = None,
    residual: Optional[TensorLike] = None,
    out_dtype: Any = None,
    residual_dtype: Any = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
) -> Tuple[TensorLike, TensorLike, Optional[TensorLike]]:
    _require_cutedsl()
    assert cutedsl_impl is not None
    return cutedsl_impl.rmsnorm_fwd(
        x,
        weight=weight,
        bias=bias,
        residual=residual,
        out_dtype=out_dtype,
        residual_dtype=residual_dtype,
        eps=eps,
        store_rstd=store_rstd,
    )


def cutedsl_rmsnorm_backward(
    x: TensorLike,
    weight: Optional[TensorLike],
    dout: TensorLike,
    rstd: TensorLike,
    *,
    dresidual_out: Optional[TensorLike] = None,
    has_bias: bool = False,
    has_residual: bool = False,
) -> Tuple[
    TensorLike, Optional[TensorLike], Optional[TensorLike], Optional[TensorLike]
]:
    _require_cutedsl()
    assert cutedsl_impl is not None
    return cutedsl_impl.rmsnorm_bwd(
        x,
        weight,
        dout,
        rstd,
        dresidual_out=dresidual_out,
        has_bias=has_bias,
        has_residual=has_residual,
    )


if _cutedsl_available:
    CuTeDSLRMSNorm = cutedsl_impl.CuTeDSLRMSNorm
else:

    class CuTeDSLRMSNorm(torch.nn.Module):  # type: ignore[misc]
        def __init__(
            self, *args: Any, **kwargs: Any
        ) -> None:  # pragma: no cover - optional backend
            raise RuntimeError("CuTeDSL RMSNorm backend is not available")

        def forward(
            self, x: TensorLike
        ) -> TensorLike:  # pragma: no cover - optional backend
            raise RuntimeError("CuTeDSL RMSNorm backend is not available")


__all__ = [
    "RMSNorm",
    "is_cutedsl_available",
    "cutedsl_rmsnorm",
    "cutedsl_rmsnorm_forward",
    "cutedsl_rmsnorm_backward",
    "CuTeDSLRMSNorm",
]
