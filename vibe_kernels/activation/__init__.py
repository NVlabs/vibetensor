# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Activation-focused Triton kernels for NanoChat."""

__all__ = [
    "relu_squared",
    "softcap_tanh_projection",
    "elementwise_add",
    "elementwise_mul",
    "elementwise_where",
    "elementwise_lerp",
    "rowwise_l2_norm",
]

_kernel_exports = None


def __getattr__(name):
    global _kernel_exports
    if name in __all__:
        if _kernel_exports is None:
            from .kernel import (
                elementwise_add,
                elementwise_lerp,
                elementwise_mul,
                elementwise_where,
                relu_squared,
                rowwise_l2_norm,
                softcap_tanh_projection,
            )
            _kernel_exports = {
                "relu_squared": relu_squared,
                "softcap_tanh_projection": softcap_tanh_projection,
                "elementwise_add": elementwise_add,
                "elementwise_mul": elementwise_mul,
                "elementwise_where": elementwise_where,
                "elementwise_lerp": elementwise_lerp,
                "rowwise_l2_norm": rowwise_l2_norm,
            }
        return _kernel_exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
