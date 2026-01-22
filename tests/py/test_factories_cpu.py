# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import vibetensor.torch as vt

def test_device_rejection_and_negative_sizes():
    with pytest.raises(ValueError):
        _ = vt.zeros((2,), device="cuda")
    with pytest.raises(ValueError):
        _ = vt.zeros((-1,))


def test_dtype_normalization_errors_and_fp16_bf16_behavior():
    # float64 is supported (copy-only for NumPy interop until DLPack importer gains float64)
    z64 = vt.zeros((1,), dtype="float64")
    assert z64.dtype == "float64"
    # float16 accepted for zeros/empty
    z16 = vt.zeros((1,), dtype="float16")
    assert z16.dtype == "float16"
    with pytest.raises(TypeError):
        _ = vt.full((1,), 1, dtype="float16")
    # BF16 gate
    if getattr(C, "_has_dlpack_bf16", False):
        zb = vt.zeros((0,), dtype="bfloat16")
        assert zb.dtype == "bfloat16"
        with pytest.raises(TypeError):
            _ = vt.full((1,), 1, dtype="bfloat16")
    else:
        with pytest.raises(TypeError):
            _ = vt.zeros((1,), dtype="bfloat16")
import pytest

from vibetensor import _C as C


def test_ones_default_dtype_float32():
    t = vt.ones((2, 3))
    assert t.dtype == "float32"
    assert t.device == (1, 0)


def test_zeros_and_full_scalar_enforcement():
    z = vt.zeros(4)
    assert z.dtype == "float32"
    f = vt.full((2,), 3, dtype="int64")
    assert f.dtype == "int64"
    with pytest.raises(TypeError):
        _ = vt.full((2,), [1, 2])


def test_overflow_text_on_large_shape():
    # Choose a shape that will overflow when computing numel
    big = (2**62, 4)
    with pytest.raises(ValueError) as ei:
        _ = vt.zeros(big, dtype="int32")
    assert "numel*itemsize overflow" in str(ei.value)
