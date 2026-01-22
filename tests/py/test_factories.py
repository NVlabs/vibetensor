# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import vibetensor.torch as vt
import pytest


def test_arange_int_and_float_dtypes():
    a = vt.arange(5)
    assert a.dtype == "int64"
    assert tuple(a.sizes) == (5,)
    b = vt.arange(1, 2.5, 0.5)
    assert b.dtype == "float32"
    assert tuple(b.sizes) == (3,)
    # keyword step
    c = vt.arange(1, 2.5, step=0.5)
    assert c.dtype == "float32"
    assert tuple(c.sizes) == (3,)
    with pytest.raises(ValueError):
        _ = vt.arange(0, 10, 0)


def test_linspace_basic_cases():
    # steps == 0 -> empty
    e = vt.linspace(0.0, 1.0, steps=0)
    assert e.dtype == "float32"
    assert tuple(e.sizes) == (0,)
    # steps == 1 -> [start]
    s1 = vt.linspace(3.0, 10.0, steps=1)
    assert s1.dtype == "float32"
    assert tuple(s1.sizes) == (1,)
    # steps >= 2 inclusive endpoints
    s = vt.linspace(3, 10, steps=5)
    assert s.dtype == "float32"
    assert tuple(s.sizes) == (5,)


def test_eye_basic_and_type_checks():
    m = vt.eye(3)
    assert m.dtype == "float32"
    assert tuple(m.sizes) == (3, 3)
    with pytest.raises(TypeError):
        _ = vt.eye(3.2)
    with pytest.raises(TypeError):
        _ = vt.eye(3, m=2.5)
