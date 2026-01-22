# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from . import cutedsl_softmax

_softmax_mod = cutedsl_softmax.softmax

softmax = _softmax_mod.softmax
softmax_forward = _softmax_mod.softmax_forward
softmax_backward = _softmax_mod.softmax_backward
log_softmax = _softmax_mod.log_softmax
log_softmax_forward = _softmax_mod.log_softmax_forward
log_softmax_backward = _softmax_mod.log_softmax_backward


def is_available() -> bool:
    return True


__all__ = [
    "softmax",
    "softmax_forward",
    "softmax_backward",
    "log_softmax",
    "log_softmax_forward",
    "log_softmax_backward",
    "is_available",
]
