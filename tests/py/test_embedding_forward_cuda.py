# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


def _has_cuda() -> bool:
    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) > 0
    except Exception:
        return False


def test_embedding_forward_shapes_and_values_cuda():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    w_cpu = np.arange(12, dtype=np.float32).reshape((4, 3))
    weight = vt.cuda.to_device(w_cpu)

    # 0-d indices -> (D,)
    idx0 = vt.cuda.to_device(np.array([2], dtype=np.int64)).as_strided([], [], 0)
    out0 = vt.embedding(weight, idx0)
    assert tuple(out0.sizes) == (3,)
    np.testing.assert_allclose(vt.cuda.from_device(out0), w_cpu[2])

    # 1-d indices -> (N, D)
    idx1_cpu = np.array([0, 2, 1], dtype=np.int64)
    idx1 = vt.cuda.to_device(idx1_cpu)
    out1 = vt.embedding(weight, idx1)
    assert tuple(out1.sizes) == (3, 3)
    np.testing.assert_allclose(vt.cuda.from_device(out1), w_cpu[idx1_cpu])

    # 2-d indices -> (S1, S2, D)
    idx2_cpu = np.array([[0, 1], [1, 3]], dtype=np.int64)
    idx2 = vt.cuda.to_device(idx2_cpu)
    out2 = vt.embedding(weight, idx2)
    assert tuple(out2.sizes) == (2, 2, 3)
    np.testing.assert_allclose(vt.cuda.from_device(out2), w_cpu[idx2_cpu])


def test_embedding_forward_bounds_checks_cuda():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    w_cpu = np.arange(12, dtype=np.float32).reshape((4, 3))
    weight = vt.cuda.to_device(w_cpu)

    with pytest.raises(IndexError, match="index out of range in self"):
        _ = vt.embedding(weight, vt.cuda.to_device(np.array([-1], dtype=np.int64)))

    with pytest.raises(IndexError, match="index out of range in self"):
        _ = vt.embedding(weight, vt.cuda.to_device(np.array([4], dtype=np.int64)))


def test_embedding_forward_bounds_check_runs_when_D_is_zero_cuda():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    weight = vt.cuda.to_device(np.zeros((4, 0), dtype=np.float32))

    with pytest.raises(IndexError, match="index out of range in self"):
        _ = vt.embedding(weight, vt.cuda.to_device(np.array([4], dtype=np.int64)))


def test_embedding_forward_strided_weight_cuda():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    base_cpu = np.arange(100, dtype=np.float32)
    base = vt.cuda.to_device(base_cpu)
    weight = base.as_strided([4, 3], [10, 2], 0)

    idx_cpu = np.array([0, 3, 1], dtype=np.int64)
    idx = vt.cuda.to_device(idx_cpu)

    out = vt.embedding(weight, idx)

    # Build the expected strided 2D view on the host.
    w_view = np.lib.stride_tricks.as_strided(
        base_cpu,
        shape=(4, 3),
        strides=(10 * base_cpu.itemsize, 2 * base_cpu.itemsize),
    )
    np.testing.assert_allclose(vt.cuda.from_device(out), w_view[idx_cpu])


def test_embedding_forward_rejects_noncontig_indices_cuda():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    w_cpu = np.arange(12, dtype=np.float32).reshape((4, 3))
    weight = vt.cuda.to_device(w_cpu)

    idx_base = vt.cuda.to_device(
        np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=np.int64)
    )
    # Shape (2, 3) but non-contiguous due to stride1=2.
    idx = idx_base.as_strided([2, 3], [5, 2], 0)

    with pytest.raises(ValueError, match="CUDA indices must be contiguous"):
        _ = vt.embedding(weight, idx)

