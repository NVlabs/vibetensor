# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


def _to_numpy_cpu(t):
    cap = vt.to_dlpack(t)
    # NumPy's from_dlpack historically accepted raw DLPack capsules
    # directly, but newer versions expect an object with a __dlpack__
    # method. Support both by wrapping the capsule when needed.
    try:
        arr = np.from_dlpack(cap)  # type: ignore[arg-type]
    except AttributeError:
        class _CapsuleWrapper:
            def __init__(self, inner):
                self._inner = inner

            def __dlpack__(self):  # pragma: no cover - tiny adapter
                return self._inner

        arr = np.from_dlpack(_CapsuleWrapper(cap))  # type: ignore[arg-type]
    return arr.reshape(tuple(int(s) for s in t.sizes))


def test_advanced_index_write_tensor_index_cpu_updates_values_and_version():
    x = vt.zeros((4,), dtype="float32")
    idx = vt.tensor([1, 3], dtype="int64")
    v = vt.tensor([5.0, 7.0], dtype="float32")

    v0 = x.version()
    x[idx] = v
    assert x.version() == v0 + 1

    x_np = _to_numpy_cpu(x)
    expected = np.zeros(4, dtype=np.float32)
    expected[[1, 3]] = [5.0, 7.0]
    np.testing.assert_allclose(x_np, expected)


def test_advanced_index_write_2d_tensor_index_suffix_full_slice_cpu():
    x = vt.zeros((4, 3), dtype="float32")
    idx = vt.tensor([1, 3], dtype="int64")
    v = vt.arange(6, dtype="float32").reshape((2, 3))

    x[idx] = v

    expected = np.zeros((4, 3), dtype=np.float32)
    expected[[1, 3], :] = _to_numpy_cpu(v)
    np.testing.assert_allclose(_to_numpy_cpu(x), expected)


def test_advanced_index_write_rejects_autograd_when_requires_grad():
    x = vt.zeros((3,), dtype="float32")
    x.set_requires_grad(True)
    idx = vt.tensor([0, 2], dtype="int64")
    v = vt.ones((2,), dtype="float32")

    ag = C.autograd
    prev = ag.is_grad_enabled()
    try:
        ag.set_grad_enabled(True)
        with pytest.raises(
            RuntimeError,
            match="autograd for in-place advanced indexing is not supported",
        ):
            x[idx] = v
    finally:
        ag.set_grad_enabled(prev)

    # Data should remain unchanged on failure.
    np.testing.assert_allclose(_to_numpy_cpu(x), np.zeros_like(_to_numpy_cpu(x)))


def test_advanced_index_write_allows_no_grad_context():
    x = vt.zeros((3,), dtype="float32")
    x.set_requires_grad(True)
    idx = vt.tensor([0, 2], dtype="int64")
    v = vt.tensor([1.0, 2.0], dtype="float32")

    expected = np.zeros(3, dtype=np.float32)
    expected[[0, 2]] = [1.0, 2.0]

    ag = C.autograd
    prev = ag.is_grad_enabled()
    try:
        with ag.no_grad():  # type: ignore[attr-defined]
            x[idx] = v
    finally:
        ag.set_grad_enabled(prev)

    x_np = _to_numpy_cpu(x)
    np.testing.assert_allclose(x_np, expected)


def test_advanced_index_write_allows_inference_mode_context():
    x = vt.zeros((3,), dtype="float32")
    x.set_requires_grad(True)
    idx = vt.tensor([0, 2], dtype="int64")
    v = vt.tensor([1.0, 2.0], dtype="float32")

    expected = np.zeros(3, dtype=np.float32)
    expected[[0, 2]] = [1.0, 2.0]

    # Under inference_mode, graph construction is suppressed even though the
    # leaf requires grad, so the in-place write should behave like no_grad.
    with vt.inference_mode():
        x[idx] = v

    x_np = _to_numpy_cpu(x)
    np.testing.assert_allclose(x_np, expected)


def test_advanced_index_write_dtype_device_mismatch():
    x = vt.zeros((3,), dtype="float32")
    idx = vt.tensor([0, 2], dtype="int64")
    v_bad = vt.ones((2,), dtype="int64")  # dtype mismatch

    with pytest.raises(ValueError, match="index assignment: dtype/device mismatch"):
        x[idx] = v_bad


def test_advanced_index_write_multi_tensor_indices_not_supported():
    x = vt.zeros((3, 3), dtype="float32")
    idx0 = vt.tensor([0, 2], dtype="int64")
    idx1 = vt.tensor([0, 1], dtype="int64")
    v = vt.ones((2, 2), dtype="float32")

    with pytest.raises(
        NotImplementedError,
        match="multiple tensor/bool indices",
    ):
        x[idx0, idx1] = v


def test_advanced_index_write_suffix_basic_indices_not_supported():
    x = vt.zeros((2, 3), dtype="float32")
    idx = vt.tensor([0, 2], dtype="int64")
    v = vt.ones((2,), dtype="float32")

    with pytest.raises(
        NotImplementedError,
        match="suffix basic indices",
    ):
        x[idx, 0] = v


def test_advanced_index_write_sequence_of_scalars_not_supported():
    x = vt.zeros((4,), dtype="float32")
    v = vt.ones((2,), dtype="float32")

    with pytest.raises(
        NotImplementedError,
        match="advanced indexing pattern is not supported",
    ):
        x[[0, 1]] = v


def test_advanced_index_write_zero_dim_advanced_uses_core_error():
    x0 = C.vt.unit()
    idx = vt.tensor([0], dtype="int64")
    v = vt.tensor([1.0], dtype="float32")

    prev = C._advanced_indexing_enabled()
    try:
        C._set_advanced_indexing_enabled_for_tests(True)
        with pytest.raises(
            RuntimeError,
            match="advanced indexing is not supported for 0-d tensors",
        ):
            x0[idx] = v
    finally:
        C._set_advanced_indexing_enabled_for_tests(prev)


def test_advanced_index_write_scalar_bool_under_implemented_vt():
    x = vt.zeros((3,), dtype="float32")

    prev = C._advanced_indexing_enabled()
    try:
        C._set_advanced_indexing_enabled_for_tests(True)
        with pytest.raises(
            NotImplementedError,
            match="advanced indexing pattern is not supported",
        ):
            x[True] = 1.0
    finally:
        C._set_advanced_indexing_enabled_for_tests(prev)
