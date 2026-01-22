# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention kernels built on Triton."""

__all__ = ["fused_attention", "KVCache", "reshape_for_gqa", "restore_from_gqa"]

_kernel_exports = None
_utils_exports = None


def __getattr__(name):
    global _kernel_exports, _utils_exports
    if name == "fused_attention":
        if _kernel_exports is None:
            from .kernel import fused_attention
            _kernel_exports = {"fused_attention": fused_attention}
        return _kernel_exports["fused_attention"]
    if name in ("KVCache", "reshape_for_gqa", "restore_from_gqa"):
        if _utils_exports is None:
            from .utils import KVCache, reshape_for_gqa, restore_from_gqa
            _utils_exports = {
                "KVCache": KVCache,
                "reshape_for_gqa": reshape_for_gqa,
                "restore_from_gqa": restore_from_gqa,
            }
        return _utils_exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
