# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt


def _to_numpy_cpu(t):
    # vibetensor patches numpy.from_dlpack to accept capsules and return a copy.
    return np.from_dlpack(vt.to_dlpack(t))


def test_embedding_forward_shapes_and_values_cpu():
    weight = vt.arange(12, dtype="float32").reshape((4, 3))
    w_np = _to_numpy_cpu(weight)

    # 0-d indices -> (D,)
    idx0 = vt.tensor(2, dtype="int64")
    out0 = vt.embedding(weight, idx0)
    assert tuple(out0.sizes) == (3,)
    np.testing.assert_allclose(_to_numpy_cpu(out0), w_np[2])

    # 1-d indices -> (N, D)
    idx1 = vt.tensor([0, 2, 1], dtype="int64")
    out1 = vt.embedding(weight, idx1)
    assert tuple(out1.sizes) == (3, 3)
    np.testing.assert_allclose(_to_numpy_cpu(out1), w_np[[0, 2, 1]])

    # 2-d indices -> (S1, S2, D)
    idx2 = vt.tensor([[0, 1], [1, 3]], dtype="int64")
    out2 = vt.embedding(weight, idx2)
    assert tuple(out2.sizes) == (2, 2, 3)
    np.testing.assert_allclose(_to_numpy_cpu(out2), w_np[_to_numpy_cpu(idx2)])

    # int32 indices are accepted
    idx1_i32 = vt.tensor([0, 2, 1], dtype="int32")
    out1_i32 = vt.embedding(weight, idx1_i32)
    np.testing.assert_allclose(_to_numpy_cpu(out1_i32), w_np[[0, 2, 1]])


def test_embedding_forward_bounds_checks_cpu():
    weight = vt.arange(12, dtype="float32").reshape((4, 3))

    with pytest.raises(IndexError, match="index out of range in self"):
        _ = vt.embedding(weight, vt.tensor([-1], dtype="int64"))

    with pytest.raises(IndexError, match="index out of range in self"):
        _ = vt.embedding(weight, vt.tensor([4], dtype="int64"))

    with pytest.raises(IndexError, match="index out of range in self"):
        _ = vt.embedding(weight, vt.tensor([-1], dtype="int32"))

    with pytest.raises(IndexError, match="index out of range in self"):
        _ = vt.embedding(weight, vt.tensor([4], dtype="int32"))


def test_embedding_forward_bounds_check_runs_when_D_is_zero():
    weight = vt.zeros((4, 0), dtype="float32")

    with pytest.raises(IndexError, match="index out of range in self"):
        _ = vt.embedding(weight, vt.tensor([4], dtype="int64"))

    with pytest.raises(IndexError, match="index out of range in self"):
        _ = vt.embedding(weight, vt.tensor([4], dtype="int32"))


def test_embedding_weight_numel_overflow_is_overflowerror():
    # Construct a view where V*D overflows int64, which makes TensorImpl::numel()==0
    # and TensorImpl::data()==nullptr. The embedding kernel must throw std::overflow_error
    # that surfaces as Python OverflowError.
    base = vt.tensor([1.0], dtype="float32")
    weight = base.as_strided([2**62, 4], [0, 0], 0)

    with pytest.raises(OverflowError, match="weight numel overflow"):
        _ = vt.embedding(weight, vt.tensor([0], dtype="int64"))


def test_embedding_forward_strided_weight_cpu():
    # Non-contiguous 2D weight view via as_strided (ws0=10, ws1=2)
    base = vt.arange(100, dtype="float32")
    weight = base.as_strided([4, 3], [10, 2], 0)

    idx = vt.tensor([0, 3, 1], dtype="int64")
    out = vt.embedding(weight, idx)

    w_np = _to_numpy_cpu(weight)
    np.testing.assert_allclose(_to_numpy_cpu(out), w_np[_to_numpy_cpu(idx)])


def test_embedding_forward_strided_indices_cpu():
    weight = vt.arange(12, dtype="float32").reshape((4, 3))
    w_np = _to_numpy_cpu(weight)

    idx_base = vt.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype="int64")
    # Shape (2, 3), but non-contiguous due to stride1=2.
    idx = idx_base.as_strided([2, 3], [5, 2], 0)

    out = vt.embedding(weight, idx)
    np.testing.assert_allclose(_to_numpy_cpu(out), w_np[_to_numpy_cpu(idx)])

    # Same, but int32 indices.
    idx_base_i32 = vt.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype="int32")
    idx_i32 = idx_base_i32.as_strided([2, 3], [5, 2], 0)
    out_i32 = vt.embedding(weight, idx_i32)
    np.testing.assert_allclose(_to_numpy_cpu(out_i32), w_np[_to_numpy_cpu(idx_i32)])


def test_embedding_forward_negative_stride_indices_cpu():
    weight = vt.arange(12, dtype="float32").reshape((4, 3))
    w_np = _to_numpy_cpu(weight)

    idx_base = vt.tensor([0, 1, 2, 3], dtype="int64")
    idx_rev = idx_base.as_strided([4], [-1], 3)

    out = vt.embedding(weight, idx_rev)
    np.testing.assert_allclose(_to_numpy_cpu(out), w_np[[3, 2, 1, 0]])
