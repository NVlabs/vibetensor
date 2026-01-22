# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


def test_tensor_dtype_inference_and_preserve_ndarray():
    # Python sequences: any-float -> float32; all-bool -> bool; else int64
    t1 = vt.tensor([1, 2, 3])
    assert t1.dtype == "int64"
    t2 = vt.tensor([1.0, 2.0])
    assert t2.dtype == "float32"
    t3 = vt.tensor([True, False])
    assert t3.dtype == "bool"

    # NumPy ndarray: preserve supported dtype
    a32 = np.array([1, 2], dtype=np.int32)
    t32 = vt.tensor(a32)
    assert t32.dtype == "int32"
    a64 = np.array([1.0, 2.0], dtype=np.float64)
    t64 = vt.tensor(a64)
    assert t64.dtype == "float64"
    b64 = np.from_dlpack(t64)
    assert b64.dtype == np.float64
    np.testing.assert_allclose(b64, a64)


def test_as_tensor_and_from_numpy_cpu_copy_only_aliasing():
    a = np.arange(5, dtype=np.float32)
    t = vt.as_tensor(a)
    # Mutate source; tensor should reflect change (zero-copy in P2)
    a[0] = 123.0
    b = np.from_dlpack(t)
    assert b[0] == 123.0

    f = vt.from_numpy(np.array([1, 2], dtype=np.int64))
    assert f.dtype == "int64"


def test_tensor_rejects_vbt_tensor_input_and_device_reject():
    # Build a simple tensor via zeros then ensure tensor() rejects it
    z = vt.zeros((2,))
    with pytest.raises(TypeError):
        _ = vt.tensor(z)
    # Device normalization negative path
    with pytest.raises(ValueError):
        _ = vt.tensor([1, 2], device="cuda")


def test_ones_like_micro():
    base = vt.zeros((3,), dtype="int64")
    o = vt.ones_like(base)
    assert o.dtype == base.dtype
    # Roundtrip to numpy to check contents are ones
    arr = np.from_dlpack(o)
    assert arr.dtype == np.int64
    assert arr.tolist() == [1, 1, 1]


def test_tensor_dtype_override_precedence():
    t = vt.tensor([1, 2], dtype="float32")
    assert t.dtype == "float32"


def test_as_tensor_numpy_zero_size_is_supported():
    # NumPy can expose zero-size arrays via DLPack in a way that our importer
    # rejects; as_tensor() must still accept the array (via a copy fallback).
    a = np.empty((0, 3), dtype=np.int64)
    t = vt.as_tensor(a)
    assert t.dtype == "int64"
    assert tuple(t.sizes) == (0, 3)


def test_as_tensor_numpy_dtype_override_casts_values():
    a = np.array([1, 2], dtype=np.int32)
    t = vt.as_tensor(a, dtype="float32")
    assert t.dtype == "float32"

    out = np.from_dlpack(t)
    assert out.dtype == np.float32
    assert out.tolist() == [1.0, 2.0]
