# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore[assignment]

from vibetensor import _C as _C


# Supported NumPy dtypes for CPU factories/H2D copy.
if _np is not None:
    _bf16 = None
    try:
        _bf16 = _np.dtype("bfloat16")  # type: ignore[attr-defined]
    except Exception:
        _bf16 = None

    SUPPORTED_NP_DTYPES = tuple(
        dt
        for dt in (
            getattr(_np, "float32", None),
            getattr(_np, "float64", None),
            getattr(_np, "complex64", None),
            getattr(_np, "complex128", None),
            getattr(_np, "int32", None),
            getattr(_np, "int64", None),
            getattr(_np, "bool_", None),
            getattr(_np, "float16", None),
            _bf16,
        )
        if dt is not None
    )
else:
    SUPPORTED_NP_DTYPES = ()


def _expected_dtype_message(*, for_full: bool) -> str:
    # Keep this in sync with C++ dtype token parsing in
    # src/vbt/python/factory_bindings.cc:dtype_from_token.
    base = ["float32", "float64", "complex64", "complex128", "int32", "int64", "bool"]

    # float16/bfloat16 are allowed for empty/zeros, but full() does not
    # implement float16/bfloat16 filling yet.
    if not for_full:
        base.insert(2, "float16")
        if bool(getattr(_C, "_has_bf16", lambda: False)()):
            base.insert(3, "bfloat16")

    return "unsupported dtype: expected one of {" + ",".join(base) + "}"


def normalize_dtype_token(dtype: Any | None, *, for_full: bool = False) -> str:
    """Normalize a dtype token into a canonical string.

    Supported canonical tokens:
      - float32, float64, complex64, complex128, float16, bfloat16, int32, int64, bool

    Notes:
      - float16/bfloat16 are rejected for full() because C++ fill kernels
        are not implemented for them yet.
    """

    # Accept strings and NumPy dtypes
    if dtype is None:
        return "float32"

    if _np is not None and isinstance(dtype, _np.dtype):
        dtype = dtype.name

    if _np is not None and dtype in (
        getattr(_np, "float32", object),
        getattr(_np, "float64", object),
        getattr(_np, "complex64", object),
        getattr(_np, "complex128", object),
        getattr(_np, "int32", object),
        getattr(_np, "int64", object),
        getattr(_np, "bool_", object),
        getattr(_np, "float16", object),
    ):
        dtype = _np.dtype(dtype).name  # type: ignore[assignment]

    if isinstance(dtype, str):
        s = dtype.lower()

        if s in ("float32", "float64", "complex64", "complex128", "int32", "int64", "bool"):
            return s

        if s == "cfloat":
            return "complex64"

        if s == "cdouble":
            return "complex128"

        if s == "float16":
            if for_full:
                raise TypeError(_expected_dtype_message(for_full=True))
            return s

        if s == "bfloat16":
            if not getattr(_C, "_has_bf16", lambda: False)():
                raise TypeError(_expected_dtype_message(for_full=False))
            if for_full:
                raise TypeError(_expected_dtype_message(for_full=True))
            return s

    raise TypeError(_expected_dtype_message(for_full=for_full))
