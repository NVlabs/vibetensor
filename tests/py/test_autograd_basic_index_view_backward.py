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


def test_basic_getitem_select_grad_matches_base_shape_and_placement():
    x = vt.arange(12, dtype="float32").reshape((3, 4)).detach()
    x.requires_grad = True

    y = x[1]
    z = C.vt.mul(y, vt.ones_like(y))
    z.backward(vt.ones_like(z))

    assert x.grad is not None
    assert tuple(x.grad.sizes) == tuple(x.sizes)

    expected = np.zeros((3, 4), dtype=np.float32)
    expected[1, :] = 1.0
    np.testing.assert_allclose(_to_numpy_cpu(x.grad), expected)


def test_basic_getitem_slice_grad_matches_base_shape_and_placement():
    x = vt.arange(12, dtype="float32").reshape((3, 4)).detach()
    x.requires_grad = True

    y = x[:2]
    z = C.vt.mul(y, vt.ones_like(y))
    z.backward(vt.ones_like(z))

    assert x.grad is not None
    assert tuple(x.grad.sizes) == tuple(x.sizes)

    expected = np.zeros((3, 4), dtype=np.float32)
    expected[:2, :] = 1.0
    np.testing.assert_allclose(_to_numpy_cpu(x.grad), expected)


def test_basic_getitem_newaxis_grad_matches_base_shape_and_placement():
    x = vt.arange(4, dtype="float32")
    x.requires_grad = True

    y = x[None]
    z = C.vt.mul(y, vt.ones_like(y))
    z.backward(vt.ones_like(z))

    assert x.grad is not None
    assert tuple(x.grad.sizes) == tuple(x.sizes)

    expected = np.ones((4,), dtype=np.float32)
    np.testing.assert_allclose(_to_numpy_cpu(x.grad), expected)


def test_basic_getitem_select_backward_accepts_expanded_grad_out():
    x = vt.arange(12, dtype="float32").reshape((3, 4)).detach()
    x.requires_grad = True

    y = x[1]

    # Create a stride-0 expanded gradient (still shape-compatible).
    g0 = vt.ones((1,), dtype="float32")
    g = g0.expand(tuple(int(s) for s in y.sizes))

    y.backward(g)

    expected = np.zeros((3, 4), dtype=np.float32)
    expected[1, :] = 1.0
    np.testing.assert_allclose(_to_numpy_cpu(x.grad), expected)


def test_basic_getitem_slice_backward_accepts_expanded_grad_out():
    x = vt.arange(12, dtype="float32").reshape((3, 4)).detach()
    x.requires_grad = True

    y = x[:2]

    g0 = vt.ones((1, 1), dtype="float32")
    g = g0.expand(tuple(int(s) for s in y.sizes))

    y.backward(g)

    expected = np.zeros((3, 4), dtype=np.float32)
    expected[:2, :] = 1.0
    np.testing.assert_allclose(_to_numpy_cpu(x.grad), expected)


def test_basic_getitem_negative_stride_backward_raises_when_disabled():
    x = vt.arange(4, dtype="float32")
    x.requires_grad = True

    y = x[::-1]

    with pytest.raises(
        RuntimeError,
        match=r"basic index view backward: negative strides are not supported",
    ):
        y.backward(vt.ones_like(y))


def test_basic_getitem_negative_stride_backward_scatter_adds_when_enabled():
    prev = C._autograd_indexing_v2_negstride_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_negstride_enabled_for_tests(True)  # type: ignore[attr-defined]
    try:
        x = vt.arange(4, dtype="float32")
        x.requires_grad = True

        y = x[::-1]
        g = vt.arange(4, dtype="float32")
        y.backward(g)

        assert x.grad is not None
        np.testing.assert_allclose(
            _to_numpy_cpu(x.grad),
            np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32),
        )
    finally:
        C._set_autograd_indexing_v2_negstride_enabled_for_tests(prev)  # type: ignore[attr-defined]


def test_basic_getitem_negative_stride_last_dim_backward_scatter_adds_when_enabled():
    prev = C._autograd_indexing_v2_negstride_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_negstride_enabled_for_tests(True)  # type: ignore[attr-defined]
    try:
        x = vt.arange(12, dtype="float32").reshape((3, 4)).detach()
        x.requires_grad = True

        y = x[:, ::-1]
        g = vt.arange(12, dtype="float32").reshape((3, 4))
        y.backward(g)

        assert x.grad is not None
        expected = _to_numpy_cpu(g)[:, ::-1]
        np.testing.assert_allclose(_to_numpy_cpu(x.grad), expected)
    finally:
        C._set_autograd_indexing_v2_negstride_enabled_for_tests(prev)  # type: ignore[attr-defined]


def test_basic_getitem_negative_stride_first_dim_backward_scatter_adds_when_enabled():
    prev = C._autograd_indexing_v2_negstride_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_negstride_enabled_for_tests(True)  # type: ignore[attr-defined]
    try:
        x = vt.arange(12, dtype="float32").reshape((3, 4)).detach()
        x.requires_grad = True

        y = x[::-1, :]
        g = vt.arange(12, dtype="float32").reshape((3, 4))
        y.backward(g)

        assert x.grad is not None
        expected = _to_numpy_cpu(g)[::-1, :]
        np.testing.assert_allclose(_to_numpy_cpu(x.grad), expected)
    finally:
        C._set_autograd_indexing_v2_negstride_enabled_for_tests(prev)  # type: ignore[attr-defined]


def test_basic_getitem_negative_stride_both_dims_backward_scatter_adds_when_enabled():
    prev = C._autograd_indexing_v2_negstride_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_negstride_enabled_for_tests(True)  # type: ignore[attr-defined]
    try:
        x = vt.arange(12, dtype="float32").reshape((3, 4)).detach()
        x.requires_grad = True

        y = x[::-1, ::-1]
        g = vt.arange(12, dtype="float32").reshape((3, 4))
        y.backward(g)

        assert x.grad is not None
        expected = _to_numpy_cpu(g)[::-1, ::-1]
        np.testing.assert_allclose(_to_numpy_cpu(x.grad), expected)
    finally:
        C._set_autograd_indexing_v2_negstride_enabled_for_tests(prev)  # type: ignore[attr-defined]


def test_basic_getitem_negative_stride_step_gt1_backward_scatter_adds_when_enabled():
    prev = C._autograd_indexing_v2_negstride_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_negstride_enabled_for_tests(True)  # type: ignore[attr-defined]
    try:
        x = vt.arange(6, dtype="float32")
        x.requires_grad = True

        y = x[::-2]
        g = C.vt.add(vt.arange(3, dtype="float32"), vt.ones((3,), dtype="float32"))
        y.backward(g)

        assert x.grad is not None
        np.testing.assert_allclose(
            _to_numpy_cpu(x.grad),
            np.array([0.0, 3.0, 0.0, 2.0, 0.0, 1.0], dtype=np.float32),
        )
    finally:
        C._set_autograd_indexing_v2_negstride_enabled_for_tests(prev)  # type: ignore[attr-defined]


def test_basic_getitem_ellipsis_backward_identity_grad():
    x = vt.arange(12, dtype="float32").reshape((3, 4)).detach()
    x.requires_grad = True

    y = x[...]
    z = C.vt.mul(y, vt.ones_like(y))
    z.backward(vt.ones_like(z))

    assert x.grad is not None
    np.testing.assert_allclose(
        _to_numpy_cpu(x.grad),
        np.ones((3, 4), dtype=np.float32),
    )


def test_basic_getitem_empty_tuple_backward_identity_grad():
    x = vt.arange(12, dtype="float32").reshape((3, 4)).detach()
    x.requires_grad = True

    y = x[()]
    z = C.vt.mul(y, vt.ones_like(y))
    z.backward(vt.ones_like(z))

    assert x.grad is not None
    np.testing.assert_allclose(
        _to_numpy_cpu(x.grad),
        np.ones((3, 4), dtype=np.float32),
    )
