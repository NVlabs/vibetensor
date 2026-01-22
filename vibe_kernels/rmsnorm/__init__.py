# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm Triton kernels."""

__all__ = [
    "RMSNorm",
    "CuTeDSLRMSNorm",
    "is_cutedsl_available",
    "cutedsl_rmsnorm",
    "cutedsl_rmsnorm_forward",
    "cutedsl_rmsnorm_backward",
]

_kernel_exports = None


def __getattr__(name):
    global _kernel_exports
    if name in __all__:
        if _kernel_exports is None:
            from .kernel import (
                cutedsl_rmsnorm,
                cutedsl_rmsnorm_backward,
                cutedsl_rmsnorm_forward,
                CuTeDSLRMSNorm,
                is_cutedsl_available,
                RMSNorm,
            )
            _kernel_exports = {
                "RMSNorm": RMSNorm,
                "CuTeDSLRMSNorm": CuTeDSLRMSNorm,
                "is_cutedsl_available": is_cutedsl_available,
                "cutedsl_rmsnorm": cutedsl_rmsnorm,
                "cutedsl_rmsnorm_forward": cutedsl_rmsnorm_forward,
                "cutedsl_rmsnorm_backward": cutedsl_rmsnorm_backward,
            }
        return _kernel_exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
