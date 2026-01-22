# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vendored CuTeDSL RMSNorm implementation."""

from __future__ import annotations

import importlib
import importlib.util
import sys


def _load_with_alias(alias: str, qualified: str):
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.find_spec(qualified)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot find module {qualified}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


cutedsl_rmsnorm = _load_with_alias("cutedsl_rmsnorm", f"{__name__}.rmsnorm")

__all__ = ["cutedsl_rmsnorm"]
