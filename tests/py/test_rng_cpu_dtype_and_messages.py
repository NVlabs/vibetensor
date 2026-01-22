# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import vibetensor.torch as vt
from vibetensor import _C


def test_normal_dtype_guard():
    t = vt.zeros([4], dtype=np.int32)
    with pytest.raises(TypeError, match="expected floating dtype for normal_"):
        _C._normal_(t, 0.0, 1.0)


def test_bernoulli_dtype_and_p_guard():
    t = vt.zeros([4], dtype=np.int32)
    with pytest.raises(TypeError, match="expected floating dtype for bernoulli_"):
        _C._bernoulli_(t, 0.5)
    t2 = vt.zeros([4], dtype=np.float32)
    with pytest.raises(ValueError, match="bernoulli_: p must be in \\[0, 1\\]"):
        _C._bernoulli_(t2, -0.1)
    with pytest.raises(ValueError, match="bernoulli_: p must be in \\[0, 1\\]"):
        _C._bernoulli_(t2, 1.1)


def test_randint_dtype_and_range_guard():
    t = vt.zeros([4], dtype=np.float32)
    with pytest.raises(TypeError, match="randint: output dtype must be int64"):
        _C._randint_(t, 0, 10)
    t2 = vt.zeros([4], dtype=np.int64)
    with pytest.raises(
        ValueError,
        match=r"randint: require low < high and \(high - low\) in \[1, 2\^63 - 1\]",
    ):
        _C._randint_(t2, 5, 5)


def test_uniform_range_validation():
    t = vt.zeros([4], dtype=np.float32)
    # Non-finite bounds
    with pytest.raises(ValueError, match="uniform_: low and high must be finite"):
        _C._uniform_(t, float("nan"), 1.0)
    with pytest.raises(ValueError, match="uniform_: low and high must be finite"):
        _C._uniform_(t, 0.0, float("inf"))
    # low > high
    with pytest.raises(ValueError, match="uniform_: low must be <= high"):
        _C._uniform_(t, 2.0, -1.0)


def test_randint_int64_boundary_cases():
    int64_min = -(2**63)
    int64_max = 2**63 - 1
    t = vt.zeros([4], dtype=np.int64)
    # Extremely wide interval covering full int64 range should be rejected
    with pytest.raises(
        ValueError,
        match=r"randint: require low < high and \(high - low\) in \[1, 2\^63 - 1\]",
    ):
        _C._randint_(t, int64_min, int64_max)
    # Largest allowed interval (size 2^63-1) should be accepted
    t2 = vt.zeros([4], dtype=np.int64)
    _C._randint_(t2, 0, 2**63 - 1)
