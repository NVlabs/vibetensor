# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


def test_full_float64_fills_deterministically() -> None:
    t = vt.full((2, 3), 1.25, dtype="float64")
    assert t.dtype == "float64"

    arr = np.from_dlpack(t)
    assert arr.dtype == np.float64
    np.testing.assert_allclose(arr, np.full((2, 3), 1.25, dtype=np.float64))


def test_cpu_from_numpy_copy_rejects_dtype_mismatch_bytes() -> None:
    x32 = np.arange(4, dtype=np.float32)
    with pytest.raises(ValueError, match="byte size mismatch"):
        _ = C._cpu_from_numpy_copy(x32, "float64")


def test_as_tensor_numpy_casts_when_dtype_differs() -> None:
    x32 = np.array([1.25, -2.5], dtype=np.float32)
    t = vt.as_tensor(x32, dtype="float64")
    out = np.from_dlpack(t)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, x32.astype(np.float64))
