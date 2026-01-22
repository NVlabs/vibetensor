# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional, Tuple

from vibe_kernels.common.tensor_types import TensorLike

from .impl import triton_impl

try:
    from .impl import cutedsl_impl

    _cutedsl_available = True
except (ImportError, ValueError, RuntimeError):
    cutedsl_impl = None  # type: ignore[assignment]
    _cutedsl_available = False


def is_cutedsl_available() -> bool:
    return _cutedsl_available


def _require_cutedsl() -> None:
    if not _cutedsl_available:
        raise RuntimeError("CuTeDSL Softmax backend is not available")


def _prepare_cutedsl_input(
    x: TensorLike, dim: int
) -> Tuple[TensorLike, Optional[int]]:
    if dim < 0:
        dim += x.ndim
    if dim == x.ndim - 1:
        return x.contiguous(), None
    return x.movedim(dim, -1).contiguous(), dim


def cutedsl_softmax(x: TensorLike, dim: int = -1) -> TensorLike:
    _require_cutedsl()
    assert cutedsl_impl is not None
    x_in, moved_dim = _prepare_cutedsl_input(x, dim)
    out = cutedsl_impl.softmax(x_in)
    if moved_dim is not None:
        return out.movedim(-1, moved_dim)
    return out


def cutedsl_log_softmax(x: TensorLike, dim: int = -1) -> TensorLike:
    _require_cutedsl()
    assert cutedsl_impl is not None
    x_in, moved_dim = _prepare_cutedsl_input(x, dim)
    out = cutedsl_impl.log_softmax(x_in)
    if moved_dim is not None:
        return out.movedim(-1, moved_dim)
    return out


def _torch_softmax_fallback(x: TensorLike, dim: int, dtype: Any) -> TensorLike:
    import torch.nn.functional as F

    return F.softmax(x, dim=dim, dtype=dtype)


def _torch_log_softmax_fallback(x: TensorLike, dim: int, dtype: Any) -> TensorLike:
    import torch.nn.functional as F

    return F.log_softmax(x, dim=dim, dtype=dtype)


def softmax(
    x: TensorLike,
    dim: int = -1,
    dtype: Any = None,
    *,
    backend: str = "auto",
) -> TensorLike:
    """
    Unified Softmax dispatcher.

    Args:
        x: Input tensor
        dim: Dimension along which Softmax will be computed (default: -1)
        dtype: Desired data type of returned tensor
        backend: Execution backend ("auto", "torch", "cutedsl", "triton")
    """
    if backend == "cutedsl":
        return cutedsl_softmax(x, dim=dim)

    if backend == "triton":
        return triton_impl.softmax(x, dim=dim)

    if backend == "auto":
        pass

    return _torch_softmax_fallback(x, dim=dim, dtype=dtype)


def log_softmax(
    x: TensorLike,
    dim: int = -1,
    dtype: Any = None,
    *,
    backend: str = "auto",
) -> TensorLike:
    if backend == "cutedsl":
        return cutedsl_log_softmax(x, dim=dim)

    if backend == "triton":
        return triton_impl.log_softmax(x, dim=dim)

    return _torch_log_softmax_fallback(x, dim=dim, dtype=dtype)


__all__ = [
    "softmax",
    "log_softmax",
    "cutedsl_softmax",
    "cutedsl_log_softmax",
    "is_cutedsl_available",
]
