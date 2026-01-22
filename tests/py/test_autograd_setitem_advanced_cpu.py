# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


if not hasattr(C, "autograd"):
    pytest.skip("autograd disabled in this build", allow_module_level=True)


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


def _to_numpy_cpu(t):
    return np.from_dlpack(t).reshape(tuple(int(s) for s in t.sizes))


def test_setitem_advanced_overwrite_zeroes_grad_region_and_propagates_rhs_grad():
    x = vt.arange(5, dtype="float32")
    x.requires_grad = True

    # Non-leaf base for in-place autograd.
    y = C.vt.add(x, vt.zeros((5,), dtype="float32"))
    assert y.is_leaf is False

    idx = vt.tensor([1, 3], dtype="int64")

    v = vt.tensor([10.0, 20.0], dtype="float32")
    v.requires_grad = True

    y[idx] = v

    y.backward(vt.ones_like(y))

    assert x.grad is not None
    gx = _to_numpy_cpu(x.grad)
    expected_x = np.ones(5, dtype=np.float32)
    expected_x[[1, 3]] = 0.0
    np.testing.assert_allclose(gx, expected_x)

    assert v.grad is not None
    np.testing.assert_allclose(_to_numpy_cpu(v.grad), np.ones(2, dtype=np.float32))


def test_index_put_advanced_accumulate_true_allows_duplicates_and_grad_self_is_unchanged():
    x = vt.arange(5, dtype="float32")
    x.requires_grad = True

    y = C.vt.add(x, vt.zeros((5,), dtype="float32"))

    idx = vt.tensor([1, 1, 3], dtype="int64")

    v = vt.ones((3,), dtype="float32")
    v.requires_grad = True

    y.index_put_((idx,), v, accumulate=True)

    y.backward(vt.ones_like(y))

    assert x.grad is not None
    np.testing.assert_allclose(_to_numpy_cpu(x.grad), np.ones(5, dtype=np.float32))

    assert v.grad is not None
    np.testing.assert_allclose(_to_numpy_cpu(v.grad), np.ones(3, dtype=np.float32))


def test_setitem_advanced_broadcast_reduces_grad_value():
    x = vt.arange(6, dtype="float32").reshape((2, 3)).detach()
    x.requires_grad = True

    y = C.vt.add(x, vt.zeros((2, 3), dtype="float32"))

    idx = vt.tensor([0, 2], dtype="int64")

    v = vt.ones((2, 1), dtype="float32")
    v.requires_grad = True

    y[:, idx] = v

    y.backward(vt.ones_like(y))

    assert x.grad is not None
    gx = _to_numpy_cpu(x.grad)
    expected_x = np.ones((2, 3), dtype=np.float32)
    expected_x[:, [0, 2]] = 0.0
    np.testing.assert_allclose(gx, expected_x)

    assert v.grad is not None
    np.testing.assert_allclose(_to_numpy_cpu(v.grad), np.full((2, 1), 2.0, dtype=np.float32))


def test_index_put_advanced_overwrite_rejects_duplicates_under_autograd():
    x = vt.arange(5, dtype="float32")
    x.requires_grad = True

    y = C.vt.add(x, vt.zeros((5,), dtype="float32"))

    idx = vt.tensor([1, 1, 3], dtype="int64")
    v = vt.ones((3,), dtype="float32")

    y_before = _to_numpy_cpu(y).copy()

    with pytest.raises(
        RuntimeError,
        match="duplicate indices are not supported.*accumulate=False",
    ):
        y.index_put_((idx,), v, accumulate=False)

    np.testing.assert_allclose(_to_numpy_cpu(y), y_before)


def test_index_put_autograd_does_not_mutate_user_index_tensor():
    x = vt.arange(5, dtype="float32")
    x.requires_grad = True

    y = C.vt.add(x, vt.zeros((5,), dtype="float32"))

    idx = vt.tensor([-1, 0], dtype="int64")
    idx_before = _to_numpy_cpu(idx).copy()

    v = vt.tensor([1.0, 2.0], dtype="float32")

    y.index_put_((idx,), v, accumulate=False)

    y.backward(vt.ones_like(y))

    idx_after = _to_numpy_cpu(idx)
    np.testing.assert_allclose(idx_after, idx_before)


def test_setitem_advanced_autograd_rejects_when_v2_disabled():
    x = vt.arange(5, dtype="float32")
    x.requires_grad = True

    y = C.vt.add(x, vt.zeros((5,), dtype="float32"))

    idx = vt.tensor([1, 3], dtype="int64")
    v = vt.tensor([10.0, 20.0], dtype="float32")
    v.requires_grad = True

    prev = C._autograd_indexing_v2_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_enabled_for_tests(False)  # type: ignore[attr-defined]
    try:
        with pytest.raises(
            RuntimeError,
            match="autograd for in-place advanced indexing is not supported",
        ):
            y[idx] = v
    finally:
        C._set_autograd_indexing_v2_enabled_for_tests(prev)  # type: ignore[attr-defined]
