# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from vibetensor import _C
import vibetensor.autograd as _A

from ..embedding import embedding as _embedding


def embedding(
    input: Any,
    weight: Any,
    padding_idx=None,
    max_norm=None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
):
    """Mirror ``torch.nn.functional.embedding``.

    Notes:
    - ``sparse=True`` is accepted for API parity, but sparse gradients are not supported.
    - ``max_norm`` uses the dispatcher op ``vt::embedding_renorm_``.
    """

    # --- padding_idx normalization (exact PyTorch behavior) ---
    if padding_idx is None:
        padding_idx_norm = -1
    else:
        # Prefer a canonical `.size(0)` if available; fall back to `.sizes[0]`.
        num_embeddings = None
        size_m = getattr(weight, "size", None)
        if callable(size_m):
            try:
                num_embeddings = int(size_m(0))
            except Exception:
                num_embeddings = None
        if num_embeddings is None:
            num_embeddings = int(getattr(weight, "sizes")[0])

        pad = int(padding_idx)
        if pad < 0:
            assert pad >= -num_embeddings, "Padding_idx must be within num_embeddings"
            padding_idx_norm = num_embeddings + pad
        else:
            assert pad < num_embeddings, "Padding_idx must be within num_embeddings"
            padding_idx_norm = pad

    # --- max_norm renorm (in-place, non-differentiable) ---
    if max_norm is not None:
        # Match PyTorch's locality hint.
        contig = getattr(input, "contiguous", None)
        if callable(contig):
            input = contig()

        if not _C._has_op("vt::embedding_renorm_"):
            raise NotImplementedError(
                "vibetensor.nn.functional.embedding: max_norm is not implemented"
            )

        max_norm_t = _C._cpu_full([], "float32", float(max_norm))
        norm_type_t = _C._cpu_full([], "float32", float(norm_type))
        with _A.no_grad():
            _C._call_op(
                "vt::embedding_renorm_",
                weight.detach(),
                input,
                max_norm_t,
                norm_type_t,
            )

    return _embedding(
        weight,
        input,
        padding_idx=padding_idx_norm,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
    )
