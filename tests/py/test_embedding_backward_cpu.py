# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt


def _to_numpy_cpu(t):
    return np.from_dlpack(t).copy()


def test_embedding_backward_duplicates_sum_and_padding_idx_cpu():
    weight = vt.arange(15, dtype="float32").reshape((5, 3)).detach()
    weight.requires_grad = True

    # Duplicate index 2, and include padding_idx=1.
    idx = vt.tensor([0, 2, 2, 1], dtype="int64")
    out = vt.embedding(weight, idx, padding_idx=1, scale_grad_by_freq=False)

    grad_out = vt.arange(12, dtype="float32").reshape((4, 3))
    grad_out_np = _to_numpy_cpu(grad_out)

    out.backward(grad_out)

    assert weight.grad is not None
    g_np = _to_numpy_cpu(weight.grad)

    expected = np.zeros((5, 3), dtype=np.float32)
    expected[0] += grad_out_np[0]
    expected[2] += grad_out_np[1] + grad_out_np[2]
    # padding_idx row is always zero in backward
    expected[1].fill(0.0)

    np.testing.assert_allclose(g_np, expected)


def test_embedding_backward_scale_grad_by_freq_cpu():
    weight = vt.arange(8, dtype="float32").reshape((4, 2)).detach()
    weight.requires_grad = True

    # Duplicate index 0 (freq=3), include padding_idx=1 (freq=2).
    idx = vt.tensor([1, 1, 0, 0, 0, 3], dtype="int64")
    out = vt.embedding(weight, idx, padding_idx=1, scale_grad_by_freq=True)

    grad_out = vt.arange(12, dtype="float32").reshape((6, 2))
    grad_out_np = _to_numpy_cpu(grad_out)

    out.backward(grad_out)

    assert weight.grad is not None
    g_np = _to_numpy_cpu(weight.grad)

    expected = np.zeros((4, 2), dtype=np.float32)
    expected[0] += (grad_out_np[2] + grad_out_np[3] + grad_out_np[4]) / 3.0
    expected[3] += grad_out_np[5]
    # padding_idx row is always zero in backward
    expected[1].fill(0.0)

    np.testing.assert_allclose(g_np, expected)


def test_embedding_backward_scale_grad_by_freq_int32_indices_cpu():
    weight = vt.arange(12, dtype="float32").reshape((4, 3)).detach()
    weight.requires_grad = True

    idx = vt.tensor([0, 1, 1, 3], dtype="int32")
    out = vt.embedding(weight, idx, scale_grad_by_freq=True)

    grad_out = vt.arange(12, dtype="float32").reshape((4, 3))
    grad_out_np = _to_numpy_cpu(grad_out)

    out.backward(grad_out)

    assert weight.grad is not None
    g_np = _to_numpy_cpu(weight.grad)

    expected = np.zeros((4, 3), dtype=np.float32)
    expected[0] += grad_out_np[0]
    expected[1] += (grad_out_np[1] + grad_out_np[2]) / 2.0
    expected[3] += grad_out_np[3]

    np.testing.assert_allclose(g_np, expected)


def test_embedding_backward_scale_grad_by_freq_strided_indices_cpu():
    weight = vt.arange(12, dtype="float32").reshape((4, 3)).detach()
    weight.requires_grad = True

    idx_base = vt.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype="int64")
    # Shape (2, 3), but non-contiguous due to stride1=2.
    idx = idx_base.as_strided([2, 3], [5, 2], 0)

    out = vt.embedding(weight, idx, scale_grad_by_freq=True)

    out_sizes = tuple(out.sizes)
    grad_out = vt.arange(int(np.prod(out_sizes)), dtype="float32").reshape(out_sizes)
    grad_out_np = _to_numpy_cpu(grad_out)

    out.backward(grad_out)
    assert weight.grad is not None
    g_np = _to_numpy_cpu(weight.grad)

    idx_np = _to_numpy_cpu(idx)
    counts = np.zeros((4,), dtype=np.int64)
    for pos in np.ndindex(*idx_np.shape):
        counts[int(idx_np[pos])] += 1

    expected = np.zeros((4, 3), dtype=np.float32)
    for pos in np.ndindex(*idx_np.shape):
        row = int(idx_np[pos])
        expected[row] += grad_out_np[pos] / float(counts[row])

    np.testing.assert_allclose(g_np, expected)


def test_embedding_backward_scale_grad_by_freq_noncontiguous_grad_out_cpu():
    weight = vt.arange(15, dtype="float32").reshape((5, 3)).detach()
    weight.requires_grad = True

    idx = vt.tensor([0, 2, 2, 1], dtype="int64")
    out = vt.embedding(weight, idx, scale_grad_by_freq=True)

    # Non-contiguous grad_out with the correct shape.
    grad_base = vt.arange(24, dtype="float32").reshape((4, 6))
    grad_out = grad_base.as_strided([4, 3], [6, 2], 0)
    grad_out_np = _to_numpy_cpu(grad_out)

    out.backward(grad_out)

    assert weight.grad is not None
    g_np = _to_numpy_cpu(weight.grad)

    expected = np.zeros((5, 3), dtype=np.float32)
    expected[0] += grad_out_np[0]
    expected[2] += (grad_out_np[1] + grad_out_np[2]) / 2.0
    expected[1] += grad_out_np[3]

    np.testing.assert_allclose(g_np, expected)


def test_embedding_backward_scale_grad_by_freq_negative_stride_indices_cpu():
    weight = vt.arange(12, dtype="float32").reshape((4, 3)).detach()
    weight.requires_grad = True

    idx_base = vt.tensor([0, 1, 2, 2], dtype="int64")
    idx_rev = idx_base.as_strided([4], [-1], 3)

    out = vt.embedding(weight, idx_rev, scale_grad_by_freq=True)

    grad_out = vt.arange(12, dtype="float32").reshape((4, 3))
    grad_out_np = _to_numpy_cpu(grad_out)

    out.backward(grad_out)

    assert weight.grad is not None
    g_np = _to_numpy_cpu(weight.grad)

    expected = np.zeros((4, 3), dtype=np.float32)
    # idx_rev is [2, 2, 1, 0]
    expected[2] += (grad_out_np[0] + grad_out_np[1]) / 2.0
    expected[1] += grad_out_np[2]
    expected[0] += grad_out_np[3]

    np.testing.assert_allclose(g_np, expected)


def test_embedding_backward_runs_when_D_is_zero_with_scale_grad_by_freq_cpu():
    weight = vt.zeros((4, 0), dtype="float32").detach()
    weight.requires_grad = True

    idx = vt.tensor([0, 3], dtype="int64")
    out = vt.embedding(weight, idx, scale_grad_by_freq=True)

    assert tuple(out.sizes) == (2, 0)

    grad_out = vt.zeros((2, 0), dtype="float32")
    out.backward(grad_out)

    assert weight.grad is not None
    np.testing.assert_allclose(
        _to_numpy_cpu(weight.grad), np.zeros((4, 0), dtype=np.float32)
    )


def test_embedding_backward_empty_indices_with_scale_grad_by_freq_cpu():
    weight = vt.arange(12, dtype="float32").reshape((4, 3)).detach()
    weight.requires_grad = True

    idx = vt.zeros((0,), dtype="int64")
    out = vt.embedding(weight, idx, scale_grad_by_freq=True)

    assert tuple(out.sizes) == (0, 3)

    grad_out = vt.zeros((0, 3), dtype="float32")
    out.backward(grad_out)

    assert weight.grad is not None
    g_np = _to_numpy_cpu(weight.grad)
    np.testing.assert_allclose(g_np, np.zeros((4, 3), dtype=np.float32))


def test_embedding_autograd_gating_ignores_indices_requires_grad_cpu():
    weight = vt.arange(6, dtype="float32").reshape((3, 2))
    assert weight.requires_grad is False

    idx = vt.tensor([0, 2, 1], dtype="int64")
    idx.requires_grad = True  # VibeTensor allows this even for integer tensors.

    out = vt.embedding(weight, idx)

    assert out.requires_grad is False
    assert out.grad_fn is None


def test_embedding_backward_strided_indices_cpu():
    weight = vt.arange(12, dtype="float32").reshape((4, 3)).detach()
    weight.requires_grad = True

    idx_base = vt.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype="int64")
    # Shape (2, 3), but non-contiguous due to stride1=2.
    idx = idx_base.as_strided([2, 3], [5, 2], 0)

    out = vt.embedding(weight, idx)

    out_sizes = tuple(out.sizes)
    grad_out = vt.arange(int(np.prod(out_sizes)), dtype="float32").reshape(out_sizes)
    grad_out_np = _to_numpy_cpu(grad_out)

    out.backward(grad_out)
    assert weight.grad is not None
    g_np = _to_numpy_cpu(weight.grad)

    idx_np = _to_numpy_cpu(idx)
    expected = np.zeros((4, 3), dtype=np.float32)
    for pos in np.ndindex(*idx_np.shape):
        expected[int(idx_np[pos])] += grad_out_np[pos]

    np.testing.assert_allclose(g_np, expected)


def test_embedding_backward_noncontiguous_grad_out_cpu():
    weight = vt.arange(15, dtype="float32").reshape((5, 3)).detach()
    weight.requires_grad = True

    idx = vt.tensor([0, 2, 2, 1], dtype="int64")
    out = vt.embedding(weight, idx)

    # Non-contiguous grad_out with the correct shape.
    grad_base = vt.arange(24, dtype="float32").reshape((4, 6))
    grad_out = grad_base.as_strided([4, 3], [6, 2], 0)
    grad_out_np = _to_numpy_cpu(grad_out)

    out.backward(grad_out)

    assert weight.grad is not None
    g_np = _to_numpy_cpu(weight.grad)

    expected = np.zeros((5, 3), dtype=np.float32)
    expected[0] += grad_out_np[0]
    expected[2] += grad_out_np[1] + grad_out_np[2]
    expected[1] += grad_out_np[3]

    np.testing.assert_allclose(g_np, expected)


def test_embedding_backward_negative_stride_indices_cpu():
    weight = vt.arange(12, dtype="float32").reshape((4, 3)).detach()
    weight.requires_grad = True

    idx_base = vt.tensor([0, 1, 2, 3], dtype="int64")
    idx_rev = idx_base.as_strided([4], [-1], 3)

    out = vt.embedding(weight, idx_rev)

    grad_out = vt.arange(12, dtype="float32").reshape((4, 3))
    grad_out_np = _to_numpy_cpu(grad_out)

    out.backward(grad_out)

    assert weight.grad is not None
    g_np = _to_numpy_cpu(weight.grad)

    expected = np.zeros((4, 3), dtype=np.float32)
    expected[3] += grad_out_np[0]
    expected[2] += grad_out_np[1]
    expected[1] += grad_out_np[2]
    expected[0] += grad_out_np[3]

    np.testing.assert_allclose(g_np, expected)


def test_embedding_backward_runs_when_D_is_zero_cpu():
    weight = vt.zeros((4, 0), dtype="float32").detach()
    weight.requires_grad = True

    idx = vt.tensor([0, 3], dtype="int64")
    out = vt.embedding(weight, idx)

    assert tuple(out.sizes) == (2, 0)

    grad_out = vt.zeros((2, 0), dtype="float32")
    out.backward(grad_out)

    assert weight.grad is not None
    np.testing.assert_allclose(_to_numpy_cpu(weight.grad), np.zeros((4, 0), dtype=np.float32))


def test_embedding_backward_empty_indices_cpu():
    weight = vt.arange(12, dtype="float32").reshape((4, 3)).detach()
    weight.requires_grad = True

    idx = vt.zeros((0,), dtype="int64")
    out = vt.embedding(weight, idx)

    assert tuple(out.sizes) == (0, 3)

    grad_out = vt.zeros((0, 3), dtype="float32")
    out.backward(grad_out)

    assert weight.grad is not None
    g_np = _to_numpy_cpu(weight.grad)
    np.testing.assert_allclose(g_np, np.zeros((4, 3), dtype=np.float32))
