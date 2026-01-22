# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

_DTYPE_STR_TO_CUTE = {
    "float16": "f16",
    "torch.float16": "f16",
    "f16": "f16",
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
    "bf16": "bf16",
    "float32": "f32",
    "torch.float32": "f32",
    "f32": "f32",
}

BackendType = Literal["auto", "triton", "cutedsl", "torch"]


def map_dtype_to_cute(dtype: Any) -> str:
    """Return CuTeDSL-compatible dtype identifier from any dtype representation."""
    dtype_str = str(dtype)
    if dtype_str in _DTYPE_STR_TO_CUTE:
        return _DTYPE_STR_TO_CUTE[dtype_str]
    raise ValueError(f"Unsupported dtype for CuTeDSL kernels: {dtype}")


def validate_backend(backend: str, allowed: list[str] | None = None) -> str:
    """Validate backend string."""
    if allowed is None:
        valid_backends = ["auto", "triton", "cutedsl", "torch"]
    else:
        valid_backends = allowed + ["auto"] if "auto" not in allowed else allowed

    if backend not in valid_backends:
        raise ValueError(
            f"Invalid backend '{backend}'. Must be one of {valid_backends}"
        )
    return backend
