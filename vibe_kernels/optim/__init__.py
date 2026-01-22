# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton-based training optimizer interfaces.

This package groups the forthcoming optimizer kernels (AdamW, Muon) and
supporting gradient utilities used during NanoChat training.  The actual
Triton kernels are implemented in the sibling modules; this package exposes a
PyTorch-Optimizer compatible facade so the training scripts can switch between
PyTorch and Triton backends without reshaping parameter groups.

The concrete implementations will arrive in subsequent iterations.  For now the
classes and functions below are thin placeholders that surface the intended API
shape and document expected behaviour.
"""

from __future__ import annotations

from .adamw import TritonAdamW, TritonDistAdamW  # type: ignore[import]
from .clip import clip_grad_norm_, compute_global_grad_norm  # type: ignore[import]
from .muon import TritonDistMuon, TritonMuon  # type: ignore[import]

__all__ = [
    "TritonAdamW",
    "TritonDistAdamW",
    "TritonMuon",
    "TritonDistMuon",
    "compute_global_grad_norm",
    "clip_grad_norm_",
]
