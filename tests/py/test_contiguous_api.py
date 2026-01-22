# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


def test_contiguous_copies_noncontiguous_and_preserves_values():
    x = vt.arange(6, dtype="int64").view((2, 3)).transpose(0, 1)
    assert not x.is_contiguous()

    y = x.contiguous()
    assert y.is_contiguous()
    assert y.sizes == x.sizes
    assert y.strides == (2, 1)

    np.testing.assert_array_equal(y.numpy(), x.numpy())


def test_is_contiguous_memory_format_channels_last_and_preserve():
    base = vt.arange(24, dtype="int64").view((2, 3, 2, 2))
    cl = base.as_strided((2, 3, 2, 2), (12, 1, 6, 3), 0)

    assert not cl.is_contiguous()
    assert cl.is_contiguous("channels_last")
    assert cl.is_contiguous("preserve")

    out = cl.contiguous("preserve")
    assert out.strides == cl.strides
    assert out.is_contiguous("channels_last")

    with pytest.raises(NotImplementedError):
        _ = base.contiguous("channels_last")


def test_memory_format_enum_is_accepted():
    x = vt.zeros((2, 3), dtype="float32")
    assert x.is_contiguous(C.MemoryFormat.contiguous)
    assert x.is_contiguous(C.MemoryFormat.preserve)
    assert not x.is_contiguous(C.MemoryFormat.channels_last)
