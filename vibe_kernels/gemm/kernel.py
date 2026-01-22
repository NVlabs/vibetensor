# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

from vibe_kernels.common.tensor_types import TensorLike

from .impl.triton_impl import (
    GEMMTiling,
    is_triton_available,
    make_default_gemm_configs,
    triton_gemm,
    triton_gemm_backward,
)

try:
    from .impl.cutedsl_impl import (
        gemm_backward as cutedsl_gemm_backward,
        gemm_out as cutedsl_gemm,
    )

    _cutedsl_available = True
except ImportError:
    cutedsl_gemm = None
    cutedsl_gemm_backward = None
    _cutedsl_available = False


def is_cutedsl_available() -> bool:
    """Return True if the CuTeDSL GEMM backend can be imported."""
    return _cutedsl_available


def gemm(
    a: TensorLike,
    b: TensorLike,
    bias: Optional[TensorLike] = None,
    out: Optional[TensorLike] = None,
    *,
    backend: str = "auto",
) -> TensorLike:
    """Matrix multiplication C = A @ B + bias.

    Args:
        a: Input tensor A.
        b: Input tensor B.
        bias: Optional bias tensor.
        out: Optional output tensor.
        backend: "triton", "cutedsl", or "auto".

    Returns:
        Result tensor.
    """
    if backend == "auto":
        backend = "triton"

    if backend == "triton":
        return triton_gemm(a, b, bias=bias, out=out)

    if backend == "cutedsl":
        if not _cutedsl_available:
            raise RuntimeError("CuTeDSL backend is not available")
        return cutedsl_gemm(a, b, bias=bias, out=out)

    raise ValueError(f"Unknown backend: {backend}")


__all__ = [
    "gemm",
    "GEMMTiling",
    "is_triton_available",
    "make_default_gemm_configs",
    "triton_gemm",
    "triton_gemm_backward",
    "cutedsl_gemm",
    "cutedsl_gemm_backward",
    "is_cutedsl_available",
]
