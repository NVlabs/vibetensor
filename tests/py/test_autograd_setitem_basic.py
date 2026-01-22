# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


if not hasattr(C, "autograd"):
    pytest.skip("autograd disabled in this build", allow_module_level=True)


def _to_numpy_cpu(t):
    return np.from_dlpack(t).reshape(tuple(int(s) for s in t.sizes))


@pytest.fixture(autouse=True)
def _enable_autograd_indexing_v2():
    prev = C._autograd_indexing_v2_enabled()  # type: ignore[attr-defined]
    prev_neg = C._autograd_indexing_v2_negstride_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_enabled_for_tests(True)  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_negstride_enabled_for_tests(False)  # type: ignore[attr-defined]
    try:
        yield
    finally:
        C._set_autograd_indexing_v2_enabled_for_tests(prev)  # type: ignore[attr-defined]
        C._set_autograd_indexing_v2_negstride_enabled_for_tests(prev_neg)  # type: ignore[attr-defined]


def test_setitem_overwrite_zeroes_grad_region_for_base():
    x = vt.ones((5,), dtype="float32")
    x.requires_grad = True

    y = C.vt.add(x, vt.full((5,), 2.0, dtype="float32"))  # non-leaf
    assert y.is_leaf is False

    y[1] = 2.0

    y.backward(vt.ones_like(y))

    assert x.grad is not None
    g = _to_numpy_cpu(x.grad)

    expected = np.ones(5, dtype=np.float32)
    expected[1] = 0.0
    np.testing.assert_allclose(g, expected)


def test_setitem_overwrite_propagates_rhs_grad_when_same_shape():
    x = vt.arange(5, dtype="float32")
    x.requires_grad = True

    y = C.vt.add(x, vt.zeros((5,), dtype="float32"))  # non-leaf
    assert y.is_leaf is False

    v = vt.tensor([10.0, 20.0], dtype="float32")
    v.requires_grad = True

    y[1:3] = v

    y.backward(vt.ones_like(y))

    assert x.grad is not None
    gx = _to_numpy_cpu(x.grad)
    expected_x = np.ones(5, dtype=np.float32)
    expected_x[1:3] = 0.0
    np.testing.assert_allclose(gx, expected_x)

    assert v.grad is not None
    np.testing.assert_allclose(_to_numpy_cpu(v.grad), np.ones(2, dtype=np.float32))


def test_setitem_overwrite_reduces_rhs_grad_for_broadcast():
    x = vt.arange(5, dtype="float32")
    x.requires_grad = True

    y = C.vt.add(x, vt.zeros((5,), dtype="float32"))  # non-leaf

    v = vt.tensor([10.0], dtype="float32")
    v.requires_grad = True

    y[1:3] = v

    y.backward(vt.ones_like(y))

    assert v.grad is not None
    np.testing.assert_allclose(_to_numpy_cpu(v.grad), np.array([2.0], dtype=np.float32))


def test_setitem_leaf_requires_grad_raises_under_v2():
    x = vt.ones((5,), dtype="float32")
    x.requires_grad = True

    with pytest.raises(RuntimeError, match="in-place on leaf tensors"):
        x[1] = 2.0
