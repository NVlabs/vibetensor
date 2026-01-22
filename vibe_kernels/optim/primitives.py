# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Triton primitive kernels used across optimizers.

This module provides both Triton-backed kernels and CPU/GPU fallbacks for the
basic vector operations used by the optimizer implementations.  The Triton
kernels can be refined/tuned later; the current variants favour clarity so the
surrounding optimizer code can start integrating against the new API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vibe_kernels.common.tensor_types import TensorLike

if TYPE_CHECKING:  # pragma: no cover - static analyzers
    import triton  # type: ignore[import]
    import triton.language as tl  # type: ignore[import]

    _HAS_TRITON = True
else:  # pragma: no cover - runtime import handling
    try:
        import triton  # type: ignore[import]
        import triton.language as tl  # type: ignore[import]

        _HAS_TRITON = True
    except Exception:
        triton = None  # type: ignore[assignment]
        tl = None  # type: ignore[assignment]
        _HAS_TRITON = False

Tensor = TensorLike

_DEFAULT_BLOCK_SIZE = 256

if _HAS_TRITON:

    @triton.jit  # type: ignore[misc]
    def _axpby_kernel(
        x_ptr,
        y_ptr,
        alpha,
        beta,
        n_elements,
        BLOCK_SIZE,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        alpha_cast = tl.full([1], alpha, dtype=x.dtype)
        beta_cast = tl.full([1], beta, dtype=y.dtype)
        out = alpha_cast * x + beta_cast * y
        tl.store(x_ptr + offsets, out, mask=mask)

    @triton.jit  # type: ignore[misc]
    def _weight_decay_kernel(x_ptr, decay, n_elements, BLOCK_SIZE):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        decay_cast = tl.full([1], 1 - decay, dtype=x.dtype)
        tl.store(x_ptr + offsets, decay_cast * x, mask=mask)

    def _launch_grid(numel: int, block_size: int = _DEFAULT_BLOCK_SIZE) -> tuple[int]:
        return (triton.cdiv(numel, block_size),)

else:

    def _launch_grid(
        numel: int, block_size: int = _DEFAULT_BLOCK_SIZE
    ) -> tuple[int]:  # pragma: no cover - fallback
        raise RuntimeError("Triton is required for GPU primitives")


def vector_axpby(x: Tensor, y: Tensor, *, alpha: float, beta: float) -> None:
    """Compute ``x = alpha * x + beta * y`` in-place."""

    if x.shape != y.shape:
        raise ValueError("vector_axpby expects tensors with identical shapes")
    if not _HAS_TRITON or x.device.type != "cuda":
        x.mul_(alpha).add_(y, alpha=beta)
        return
    n = x.numel()
    if n == 0:
        return
    grid = _launch_grid(n)
    _axpby_kernel[grid](
        x,
        y,
        alpha=float(alpha),
        beta=float(beta),
        n_elements=n,
        BLOCK_SIZE=_DEFAULT_BLOCK_SIZE,
    )


def apply_weight_decay(param: Tensor, *, decay: float) -> None:
    """Apply decoupled weight decay ``param *= (1 - decay)`` in-place."""

    if decay == 0.0:
        return
    if not _HAS_TRITON or param.device.type != "cuda":
        param.mul_(1 - decay)
        return
    n = param.numel()
    if n == 0:
        return
    grid = _launch_grid(n)
    _weight_decay_kernel[grid](
        param,
        decay=float(decay),
        n_elements=n,
        BLOCK_SIZE=_DEFAULT_BLOCK_SIZE,
    )


def rsqrt_with_eps(value: Tensor, *, eps: float) -> Tensor:
    """Return ``1 / sqrt(value + eps)`` with appropriate precision."""

    base = value.to(torch.float32)
    result = torch.rsqrt(base + eps)
    return result.to(value.dtype)


def reduce_l2_norm_sq(value: Tensor) -> Tensor:
    """Reduce ``sum(value**2)`` over the flattened tensor on the device."""

    return value.to(torch.float32).pow(2).sum()


def matmul_transpose(matrix: Tensor) -> Tensor:
    """Return ``matrix @ matrix.T``; used during Newton–Schulz iterations."""

    return matrix @ matrix.transpose(-1, -2)
