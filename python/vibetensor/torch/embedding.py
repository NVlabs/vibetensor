# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from vibetensor import _C


def embedding(
    weight: Any,
    indices: Any,
    *,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
):
    """Gather rows from a 2D embedding matrix.

    This is a thin wrapper around the dispatcher op `vt::embedding`.

    Note: we intentionally materialize scalar args as *typed* CPU 0-d tensors
    instead of relying on `vibetensor.torch.ops` scalar auto-wrapping.
    """

    needs_grad = False
    if sparse:
        try:
            from vibetensor.autograd import is_grad_enabled
        except Exception:  # pragma: no cover
            is_grad_enabled = lambda: False  # type: ignore[assignment]
        needs_grad = bool(getattr(weight, "requires_grad", False)) and is_grad_enabled()

    if sparse and needs_grad:
        raise NotImplementedError(
            "vibetensor.embedding: sparse gradients are not supported"
        )

    pad_t = _C._cpu_full([], "int64", int(padding_idx))
    scale_t = _C._cpu_full([], "bool", bool(scale_grad_by_freq))
    sparse_t = _C._cpu_full([], "bool", bool(sparse))
    return _C._call_op("vt::embedding", weight, indices, pad_t, scale_t, sparse_t)
