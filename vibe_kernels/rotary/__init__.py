# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rotary embedding Triton kernels."""

__all__ = [
    "apply_rotary_embedding",
]


def __getattr__(name):
    if name == "apply_rotary_embedding":
        from .kernel import apply_rotary_embedding
        return apply_rotary_embedding
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
