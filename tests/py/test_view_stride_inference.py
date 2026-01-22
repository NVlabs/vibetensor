# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt


def test_view_stride_inference_allows_noncontiguous_dense_chunk():
    base = vt.arange(24, dtype="float32").view((2, 3, 4))

    # A non-contiguous view that still forms a single "contiguous chunk" in
    # PyTorch's computeStride sense.
    x = base.as_strided((2, 3, 2), (12, 4, 2), 0)
    assert not x.is_contiguous()

    y = x.view((12,))
    assert tuple(y.sizes) == (12,)
    assert tuple(y.strides) == (2,)

    # Mutate the base tensor (contiguous) and ensure the view reflects it.
    base.add_(vt.ones_like(base))

    arr = np.from_dlpack(y.contiguous())
    expected = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1)[..., ::2].reshape(12)
    np.testing.assert_array_equal(arr, expected)


def test_reshape_stride_inference_aliases_when_possible():
    base = vt.arange(24, dtype="float32").view((2, 3, 4))
    x = base.as_strided((2, 3, 2), (12, 4, 2), 0)

    y = x.reshape((6, 2))
    assert tuple(y.sizes) == (6, 2)
    assert tuple(y.strides) == (4, 2)

    # Mutate the base tensor (contiguous) and ensure the reshaped view reflects it.
    base.add_(vt.ones_like(base))

    arr = np.from_dlpack(y.contiguous())
    expected = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1)[..., ::2].reshape(6, 2)
    np.testing.assert_array_equal(arr, expected)


def test_view_incompatible_strides_raises():
    base = vt.arange(6, dtype="int64").view((2, 3))
    t = base.transpose(0, 1)

    with pytest.raises(ValueError, match="view size is not compatible"):
        _ = t.view((6,))


def test_view_zero_numel_stride_rules_match_pytorch():
    # Build a 0-numel view that still has backing storage (as_strided requires it).
    base = vt.zeros((1, 2), dtype="float32").narrow(0, 0, 0)
    # Give it an unusual (but legal) stride pattern.
    t = base.as_strided((0, 2), (123, 1), 0)

    same = t.view((0, 2))
    assert tuple(same.strides) == (123, 1)

    reshaped = t.view((0, 1, 2))
    assert tuple(reshaped.sizes) == (0, 1, 2)
    assert tuple(reshaped.strides) == (2, 2, 1)


def test_view_stride_inference_accepts_negative_strides():
    base = vt.arange(6, dtype="float32")
    rev = base.as_strided((6,), (-1,), 5)

    m = rev.view((2, 3))
    assert tuple(m.sizes) == (2, 3)
    assert tuple(m.strides) == (-3, -1)

    arr = np.from_dlpack(m.contiguous())
    expected = np.array([5, 4, 3, 2, 1, 0], dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(arr, expected)

