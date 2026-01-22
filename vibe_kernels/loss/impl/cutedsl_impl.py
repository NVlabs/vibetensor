# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from . import cutedsl_cross_entropy

_ce_mod = cutedsl_cross_entropy

CrossEntropyFunction = _ce_mod.CrossEntropyFunction
cross_entropy = _ce_mod.cross_entropy
cross_entropy_forward = _ce_mod.cross_entropy_forward
cross_entropy_backward = _ce_mod.cross_entropy_backward


def is_available() -> bool:  # pragma: no cover - convenience hook
    return True


__all__ = [
    "CrossEntropyFunction",
    "cross_entropy",
    "cross_entropy_forward",
    "cross_entropy_backward",
    "is_available",
]
