# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt


def test_numpy_negative_stride_error_text():
    a = np.arange(6, dtype=np.int32)
    b = a[::-1]
    with pytest.raises(ValueError) as ei:
        _ = vt.from_numpy(b)
    assert "At least one stride in the given numpy array is negative" in str(ei.value)


def test_numpy_stride_multiple_precedence_over_negative():
    base = np.arange(6, dtype=np.int32).reshape(2, 3)
    # Craft a view with stride bytes that are not a multiple of itemsize (unsafe view)
    bad = np.lib.stride_tricks.as_strided(base, shape=(2, 3), strides=(base.strides[0], base.strides[1] // 2))
    # Also make a negative-stride view on top to ensure first error is stride-multiple
    bad2 = bad[:, ::-1]
    with pytest.raises(ValueError) as ei:
        _ = vt.from_numpy(bad2)
    assert "given numpy array strides not a multiple of the element byte size" in str(ei.value)


def test_numpy_byte_order_rejected_when_non_native():
    # Choose non-native byte order for the platform
    import sys
    if sys.byteorder == "little":
        dt = np.dtype(">i4")
    else:
        dt = np.dtype("<i4")
    a = np.array([1, 2, 3, 4], dtype=dt)
    with pytest.raises(ValueError) as ei:
        _ = vt.from_numpy(a)
    assert "given numpy array has byte order different from the native byte order" in str(ei.value)


def test_numpy_fortran_and_zero_stride_preserve_strides():
    # Fortran-contiguous: accepted and sizes preserved
    f = np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    t = vt.from_numpy(f)
    assert t.dtype == "float32"
    assert tuple(t.sizes) == f.shape

    # Zero-stride broadcast: accept and sizes preserved
    base = np.array([7], dtype=np.int32)
    b = np.broadcast_to(base, (3, 3))
    tb = vt.from_numpy(b)
    assert tb.dtype == "int32"
    assert tuple(tb.sizes) == b.shape
