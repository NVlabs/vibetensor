# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C

try:
    import torch
except Exception:  # pragma: no cover - torch is an optional test dependency
    torch = None


def _skip_if_no_torch() -> None:
    if torch is None:
        pytest.skip("torch not available for tensor_iter promotion tests")


def test_vt_add_mixed_dtypes_raises_value_error():
    """Public vt.add should not enable TI promotion.

    Mixed-dtype inputs are still rejected; promotion remains an
    internal-only mechanism exercised by C++ tests.
    """
    _skip_if_no_torch()
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((4,), dtype=np.float32)
    b_np = rng.integers(-5, 5, size=(4,), dtype=np.int64)

    # Build raw TensorImpls via DLPack and call the C++ vt::add binding
    # is not accidentally enabled on the public surface.
    a = C._from_dlpack(a_np.__dlpack__())
    b = C._from_dlpack(b_np.__dlpack__())

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _ = C.vt.add(a, b)


def test_vt_mul_mixed_dtypes_raises_value_error():
    """Public vt.mul should also reject mixed dtypes."""
    _skip_if_no_torch()
    rng = np.random.default_rng(1)
    a_np = rng.standard_normal((4,), dtype=np.float32)
    b_np = rng.integers(-3, 3, size=(4,), dtype=np.int64)

    a = C._from_dlpack(a_np.__dlpack__())
    b = C._from_dlpack(b_np.__dlpack__())

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        _ = C.vt.mul(a, b)


def test_reduction_mean_int64_still_raises_value_error():
    """Regression guard: reductions keep existing dtype rules.

    Even after enabling TI promotion flags internally, mean on
    int64 tensors should continue to raise instead of promoting
    silently to float.
    """
    _skip_if_no_torch()
    arr = np.arange(12, dtype=np.int64).reshape(3, 4)
    t = vt.from_numpy(arr)

    with pytest.raises(ValueError):
        _ = t.mean()
