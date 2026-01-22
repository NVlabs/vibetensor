# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from importlib import import_module
from typing import Optional, Tuple

from vibe_kernels.common import alloc
from vibe_kernels.common.tensor_types import TensorLike

_GEMM_INTERFACE = None


def _get_gemm_interface():
    global _GEMM_INTERFACE
    if _GEMM_INTERFACE is None:
        _GEMM_INTERFACE = import_module(
            "kernel_factory.gemm.impl.cutedsl_gemm.gemm_interface"
        )
    return _GEMM_INTERFACE


def _normalize_tensor(t: TensorLike):
    if t.device.type != "cuda":
        raise ValueError("CuTeDSL GEMM expects CUDA tensors")
    if t.dim() != 2:
        raise ValueError("CuTeDSL GEMM only supports 2D matrices")
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def gemm_out(
    a: TensorLike,
    b: TensorLike,
    *,
    bias: TensorLike | None = None,
    out: TensorLike | None = None,
):
    gemm_interface = _get_gemm_interface()

    a = _normalize_tensor(a)
    b = _normalize_tensor(b)

    if a.dtype != b.dtype:
        raise ValueError("Input dtypes must match for CuTeDSL GEMM")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Inner dimensions must agree")

    if out is None:
        out = alloc.empty((a.shape[0], b.shape[1]), like=a, dtype=a.dtype)
    else:
        if out.shape != (a.shape[0], b.shape[1]):
            raise ValueError("Output tensor has incorrect shape")
        if out.device != a.device or out.dtype != a.dtype:
            raise ValueError("Output tensor must match input device and dtype")

    gemm_interface = _get_gemm_interface()
    GemmConfig = getattr(gemm_interface, "GemmConfig")

    def _parse_override(env_value: str) -> tuple[int, int]:
        parts = [int(x.strip()) for x in env_value.split(",")]
        if len(parts) != 2:
            raise ValueError("Expected two comma-separated integers")
        return parts[0], parts[1]

    def _select_config(m: int, n: int, k: int) -> object:
        override_tile = os.getenv("CUTEDSL_GEMM_TILE_SHAPE")
        override_cluster = os.getenv("CUTEDSL_GEMM_CLUSTER_SHAPE")
        override_pingpong = os.getenv("CUTEDSL_GEMM_PINGPONG")

        if override_tile:
            tile_m, tile_n = _parse_override(override_tile)
        else:
            size = max(m, n, k)
            if size <= 1024:
                tile_m, tile_n = 128, 256
            elif size <= 2048:
                tile_m, tile_n = 128, 192
            elif size <= 3072:
                tile_m, tile_n = 64, 192
            else:
                tile_m, tile_n = 128, 256

        if override_cluster:
            cluster_m, cluster_n = _parse_override(override_cluster)
        else:
            size = max(m, n, k)
            if size <= 1024:
                cluster_m, cluster_n = 2, 1
            elif size <= 2048:
                cluster_m, cluster_n = 1, 1
            elif size <= 3072:
                cluster_m, cluster_n = 2, 1
            elif size <= 4096:
                cluster_m, cluster_n = 1, 2
            else:
                cluster_m, cluster_n = 2, 1

        if override_pingpong is not None:
            pingpong = override_pingpong.lower() in {"1", "true", "yes"}
        else:
            size = max(m, n, k)
            if size <= 1024:
                pingpong = False
            elif size <= 4096:
                pingpong = False
            else:
                pingpong = False

        return GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            pingpong=pingpong,
        )

    config = _select_config(a.shape[0], b.shape[1], a.shape[1])

    fused_bias_override = os.getenv("CUTEDSL_GEMM_FORCE_FUSED_BIAS")
    use_fused_bias = (
        fused_bias_override is not None
        and fused_bias_override.lower() in {"1", "true", "yes"}
    )
    bias_arg = bias if use_fused_bias else None

    gemm_interface.gemm_tuned(
        a,
        b,
        out,
        C=None,
        bias=bias_arg,
        alpha=1.0,
        beta=0.0,
        config=config,
    )
    if bias is not None and not use_fused_bias:
        out.add_(bias)
    return out


def gemm_backward(
    grad_output: TensorLike,
    a: TensorLike,
    b: TensorLike,
    *,
    compute_grad_a: bool = True,
    compute_grad_b: bool = True,
    compute_grad_bias: bool = False,
) -> Tuple[Optional[TensorLike], Optional[TensorLike], Optional[TensorLike]]:
    """Compute gradients for C = a @ b + bias using CuTeDSL GEMM kernels."""

    grad_output = _normalize_tensor(grad_output)
    a = _normalize_tensor(a)
    b = _normalize_tensor(b)

    if grad_output.dtype != a.dtype or grad_output.dtype != b.dtype:
        raise TypeError("All tensors must share the same dtype")

    M, N = grad_output.shape
    if a.shape[0] != M or b.shape[1] != N:
        raise ValueError("Shape mismatch between grad_output and inputs")
    K = a.shape[1]
    if b.shape[0] != K:
        raise ValueError("Input matrix shapes are incompatible")

    grad_a: Optional[TensorLike] = None
    grad_b: Optional[TensorLike] = None
    grad_bias: Optional[TensorLike] = None

    if compute_grad_a and K > 0 and N > 0:
        grad_a = alloc.empty_like(a)
        b_t = b.transpose(0, 1).contiguous()
        gemm_out(grad_output, b_t, out=grad_a)

    if compute_grad_b and M > 0 and K > 0:
        grad_b = alloc.empty_like(b)
        a_t = a.transpose(0, 1).contiguous()
        gemm_out(a_t, grad_output, out=grad_b)

    if compute_grad_bias:
        grad_bias = grad_output.sum(dim=0).contiguous()

    return grad_a, grad_b, grad_bias
