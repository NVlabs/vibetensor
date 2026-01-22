# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding-related Triton kernels."""

__all__ = [
    "FusedEmbeddingRMSNorm",
]


def __getattr__(name):
    if name == "FusedEmbeddingRMSNorm":
        from .kernel import FusedEmbeddingRMSNorm
        return FusedEmbeddingRMSNorm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
