# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GEMM kernel package exposing Triton implementations and helpers."""

__all__ = [
    "GEMMTiling",
    "is_triton_available",
    "is_cutedsl_available",
    "make_default_gemm_configs",
    "triton_gemm",
    "triton_gemm_backward",
    "cutedsl_gemm",
    "cutedsl_gemm_backward",
]

_kernel_exports = None


def __getattr__(name):
    global _kernel_exports
    if name in __all__:
        if _kernel_exports is None:
            from .kernel import (
                cutedsl_gemm,
                cutedsl_gemm_backward,
                GEMMTiling,
                is_cutedsl_available,
                is_triton_available,
                make_default_gemm_configs,
                triton_gemm,
                triton_gemm_backward,
            )
            _kernel_exports = {
                "GEMMTiling": GEMMTiling,
                "is_triton_available": is_triton_available,
                "is_cutedsl_available": is_cutedsl_available,
                "make_default_gemm_configs": make_default_gemm_configs,
                "triton_gemm": triton_gemm,
                "triton_gemm_backward": triton_gemm_backward,
                "cutedsl_gemm": cutedsl_gemm,
                "cutedsl_gemm_backward": cutedsl_gemm_backward,
            }
        return _kernel_exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
