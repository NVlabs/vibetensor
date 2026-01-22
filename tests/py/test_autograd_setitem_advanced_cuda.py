# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


if not hasattr(C, "autograd"):
    pytest.skip("autograd disabled in this build", allow_module_level=True)


def _cuda_only() -> bool:
    return bool(getattr(C, "_has_cuda", False)) and int(getattr(C, "_cuda_device_count", lambda: 0)()) > 0


@pytest.fixture(autouse=True)
def _enable_autograd_indexing_v2():
    prev = C._autograd_indexing_v2_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_enabled_for_tests(True)  # type: ignore[attr-defined]
    try:
        yield
    finally:
        C._set_autograd_indexing_v2_enabled_for_tests(prev)  # type: ignore[attr-defined]


@pytest.mark.cuda
def test_setitem_advanced_overwrite_zeroes_grad_region_cuda_and_propagates_rhs_grad():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    ag = C.autograd
    prev_cuda = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    prev_grad = bool(ag.is_grad_enabled())
    try:
        ag.set_grad_enabled(True)
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]

        x0_np = np.arange(5, dtype=np.float32)
        x = vt.cuda.to_device(x0_np)
        x.set_requires_grad(True)

        # Non-leaf base for in-place autograd.
        zeros = vt.cuda.to_device(np.zeros_like(x0_np))
        y = C.vt.add(x, zeros)
        assert y.is_leaf is False

        idx = vt.tensor([1, 3], dtype="int64")

        v_np = np.array([10.0, 20.0], dtype=np.float32)
        v = vt.cuda.to_device(v_np)
        v.set_requires_grad(True)

        y[idx] = v

        ones = vt.cuda.to_device(np.ones(5, dtype=np.float32))
        y.backward(ones)

        assert x.grad is not None
        gx = vt.cuda.from_device(x.grad())
        expected_x = np.ones(5, dtype=np.float32)
        expected_x[[1, 3]] = 0.0
        np.testing.assert_allclose(gx, expected_x)

        assert v.grad is not None
        np.testing.assert_allclose(vt.cuda.from_device(v.grad()), np.ones(2, dtype=np.float32))
    finally:
        ag.set_cuda_autograd_enabled(prev_cuda)  # type: ignore[attr-defined]
        ag.set_grad_enabled(prev_grad)


@pytest.mark.cuda
def test_setitem_advanced_broadcast_reduces_grad_value_cuda():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    ag = C.autograd
    prev_cuda = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    prev_grad = bool(ag.is_grad_enabled())
    try:
        ag.set_grad_enabled(True)
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]

        x0_np = np.arange(6, dtype=np.float32).reshape((2, 3))
        x = vt.cuda.to_device(x0_np)
        x.set_requires_grad(True)

        y = C.vt.add(x, vt.cuda.to_device(np.zeros_like(x0_np)))

        idx = vt.tensor([0, 2], dtype="int64")

        v_np = np.ones((2, 1), dtype=np.float32)
        v = vt.cuda.to_device(v_np)
        v.set_requires_grad(True)

        y[:, idx] = v

        y.backward(vt.cuda.to_device(np.ones((2, 3), dtype=np.float32)))

        assert x.grad is not None
        gx = vt.cuda.from_device(x.grad())
        expected_x = np.ones((2, 3), dtype=np.float32)
        expected_x[:, [0, 2]] = 0.0
        np.testing.assert_allclose(gx, expected_x)

        assert v.grad is not None
        np.testing.assert_allclose(vt.cuda.from_device(v.grad()), np.full((2, 1), 2.0, dtype=np.float32))
    finally:
        ag.set_cuda_autograd_enabled(prev_cuda)  # type: ignore[attr-defined]
        ag.set_grad_enabled(prev_grad)


@pytest.mark.cuda
def test_index_put_advanced_accumulate_true_allows_duplicates_cuda():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    ag = C.autograd
    prev_cuda = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    prev_grad = bool(ag.is_grad_enabled())
    try:
        ag.set_grad_enabled(True)
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]

        x0_np = np.arange(5, dtype=np.float32)
        x = vt.cuda.to_device(x0_np)
        x.set_requires_grad(True)

        y = C.vt.add(x, vt.cuda.to_device(np.zeros_like(x0_np)))

        idx = vt.tensor([1, 1, 3], dtype="int64")

        v = vt.cuda.to_device(np.ones((3,), dtype=np.float32))
        v.set_requires_grad(True)

        y.index_put_((idx,), v, accumulate=True)

        y.backward(vt.cuda.to_device(np.ones((5,), dtype=np.float32)))

        assert x.grad is not None
        np.testing.assert_allclose(vt.cuda.from_device(x.grad()), np.ones(5, dtype=np.float32))

        assert v.grad is not None
        np.testing.assert_allclose(vt.cuda.from_device(v.grad()), np.ones(3, dtype=np.float32))
    finally:
        ag.set_cuda_autograd_enabled(prev_cuda)  # type: ignore[attr-defined]
        ag.set_grad_enabled(prev_grad)


@pytest.mark.cuda
def test_index_put_advanced_overwrite_rejects_duplicates_under_autograd_cuda():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    x0_np = np.arange(5, dtype=np.float32)
    x = vt.cuda.to_device(x0_np)
    x.set_requires_grad(True)

    y = C.vt.add(x, vt.cuda.to_device(np.zeros_like(x0_np)))

    idx = vt.tensor([1, 1, 3], dtype="int64")
    v = vt.cuda.to_device(np.ones((3,), dtype=np.float32))

    y_before = vt.cuda.from_device(y).copy()

    with pytest.raises(
        RuntimeError,
        match="duplicate indices are not supported.*accumulate=False",
    ):
        y.index_put_((idx,), v, accumulate=False)

    np.testing.assert_allclose(vt.cuda.from_device(y), y_before)


@pytest.mark.cuda
def test_index_put_autograd_does_not_mutate_user_index_tensor_cuda():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    ag = C.autograd
    prev_cuda = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    prev_grad = bool(ag.is_grad_enabled())
    try:
        ag.set_grad_enabled(True)
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]

        x0_np = np.arange(5, dtype=np.float32)
        x = vt.cuda.to_device(x0_np)
        x.set_requires_grad(True)

        y = C.vt.add(x, vt.cuda.to_device(np.zeros_like(x0_np)))

        idx_cuda = vt.cuda.to_device(np.array([-1, 0], dtype=np.int64))
        idx_before = vt.cuda.from_device(idx_cuda).copy()

        v = vt.cuda.to_device(np.array([1.0, 2.0], dtype=np.float32))

        y.index_put_((idx_cuda,), v, accumulate=False)
        y.backward(vt.cuda.to_device(np.ones((5,), dtype=np.float32)))

        idx_after = vt.cuda.from_device(idx_cuda)
        np.testing.assert_allclose(idx_after, idx_before)
    finally:
        ag.set_cuda_autograd_enabled(prev_cuda)  # type: ignore[attr-defined]
        ag.set_grad_enabled(prev_grad)


@pytest.mark.cuda
def test_index_put_autograd_rejects_suffix_full_slice_cuda():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    ag = C.autograd
    prev_cuda = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    prev_grad = bool(ag.is_grad_enabled())
    try:
        ag.set_grad_enabled(True)
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]

        x0_np = np.arange(6, dtype=np.float32).reshape((2, 3))
        x = vt.cuda.to_device(x0_np)
        x.set_requires_grad(True)

        # Non-leaf base for in-place autograd.
        y = C.vt.add(x, vt.cuda.to_device(np.zeros_like(x0_np)))
        assert y.is_leaf is False

        idx = vt.tensor([0], dtype="int64")
        v = vt.cuda.to_device(np.ones((1, 3), dtype=np.float32))

        y_before = vt.cuda.from_device(y).copy()

        with pytest.raises(
            RuntimeError,
            match="autograd for in-place advanced indexing is not supported",
        ):
            y.index_put_((idx, slice(None)), v, accumulate=False)

        np.testing.assert_allclose(vt.cuda.from_device(y), y_before)
    finally:
        ag.set_cuda_autograd_enabled(prev_cuda)  # type: ignore[attr-defined]
        ag.set_grad_enabled(prev_grad)
