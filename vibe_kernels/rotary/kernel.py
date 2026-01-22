# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

from vibe_kernels.common.tensor_types import TensorLike
from vibe_kernels.common.utils import BackendType, validate_backend

from .impl.torch_impl import apply_rotary_torch
from .impl.triton_impl import apply_rotary_triton

try:
    from .impl.cutedsl_impl import apply_rotary_cutedsl
except ImportError:
    apply_rotary_cutedsl = None  # type: ignore


def apply_rotary_embedding(
    q: TensorLike,
    k: TensorLike,
    cos: TensorLike,
    sin: TensorLike,
    positions: Optional[TensorLike] = None,
    *,
    backend: BackendType = "triton",
) -> Tuple[TensorLike, TensorLike]:
    """Apply rotary position embeddings to query/key projections.

    Args:
        q, k: Query/key tensors of shape ``(..., seqlen, head_dim)``. In practice, the
            rotary kernels are used with ``(batch, heads, seqlen, head_dim)``.
        cos, sin: Precomputed cosine/sine tables where the last dimension is
            ``head_dim // 2``. Typically these have shape ``(max_position, head_dim // 2)``.
        positions: Optional integer tensor of shape ``(*leading, seqlen)`` giving the
            position index for each token into the first dimension of ``cos``/``sin``.
        backend: ``"triton"`` (default), ``"cutedsl"``, or ``"torch"``.

    Returns:
        Tuple ``(q_rot, k_rot)`` with the same shape/dtype as ``q``/``k``.
    """
    backend = validate_backend(backend)

    if backend == "auto":
        backend = "triton"

    if backend == "triton":
        return apply_rotary_triton(q, k, cos, sin, positions)

    if backend == "cutedsl":
        if apply_rotary_cutedsl is None:
            raise RuntimeError("CuTeDSL backend is not available (import failed)")
        return apply_rotary_cutedsl(q, k, cos, sin, positions)

    if backend == "torch":
        return apply_rotary_torch(q, k, cos, sin, positions)

    raise ValueError(f"Unknown backend: {backend}")


__all__ = [
    "apply_rotary_embedding",
]
