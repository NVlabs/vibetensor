# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loss-focused Triton kernels."""

__all__ = ["cross_entropy_loss", "softmax", "log_softmax", "is_cutedsl_available"]

_kernel_exports = None


def __getattr__(name):
    global _kernel_exports
    if name in __all__:
        if _kernel_exports is None:
            from .kernel import cross_entropy_loss, is_cutedsl_available, log_softmax, softmax
            _kernel_exports = {
                "cross_entropy_loss": cross_entropy_loss,
                "softmax": softmax,
                "log_softmax": log_softmax,
                "is_cutedsl_available": is_cutedsl_available,
            }
        return _kernel_exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
