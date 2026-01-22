# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C

pytestmark = pytest.mark.cuda


def _has_cuda() -> bool:
    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) > 0
    except Exception:
        return False


@pytest.fixture
def cuda_autograd_enabled():
    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    try:
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]
        yield
    finally:
        ag.set_cuda_autograd_enabled(prev)  # type: ignore[attr-defined]


def test_embedding_backward_duplicates_sum_and_padding_idx_cuda(cuda_autograd_enabled):
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    w_cpu = np.arange(15, dtype=np.float32).reshape((5, 3))
    weight = vt.cuda.to_device(w_cpu).detach()
    weight.requires_grad = True

    # Duplicate index 2 (freq=3), and include padding_idx=1.
    idx_cpu = np.array([0, 2, 2, 2, 1], dtype=np.int64)
    idx = vt.cuda.to_device(idx_cpu)

    out = vt.embedding(weight, idx, padding_idx=1, scale_grad_by_freq=False)

    grad_out_cpu = np.arange(15, dtype=np.float32).reshape((5, 3))
    grad_out = vt.cuda.to_device(grad_out_cpu)

    out.backward(grad_out)

    assert weight.grad is not None
    g_np = vt.cuda.from_device(weight.grad())

    expected = np.zeros((5, 3), dtype=np.float32)
    expected[0] += grad_out_cpu[0]
    expected[2] += grad_out_cpu[1] + grad_out_cpu[2] + grad_out_cpu[3]
    # padding_idx row is always zero in backward
    expected[1].fill(0.0)

    np.testing.assert_allclose(g_np, expected, rtol=1e-6, atol=1e-6)


def test_embedding_backward_scale_grad_by_freq_cuda(cuda_autograd_enabled):
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    w_cpu = np.arange(8, dtype=np.float32).reshape((4, 2))
    weight = vt.cuda.to_device(w_cpu).detach()
    weight.requires_grad = True

    # Duplicate index 0 (freq=3), include padding_idx=1 (freq=2).
    idx_cpu = np.array([1, 1, 0, 0, 0, 3], dtype=np.int64)
    idx = vt.cuda.to_device(idx_cpu)

    out = vt.embedding(weight, idx, padding_idx=1, scale_grad_by_freq=True)

    grad_out_cpu = np.arange(12, dtype=np.float32).reshape((6, 2))
    grad_out = vt.cuda.to_device(grad_out_cpu)

    out.backward(grad_out)

    assert weight.grad is not None
    g_np = vt.cuda.from_device(weight.grad())

    expected = np.zeros((4, 2), dtype=np.float32)
    expected[0] += (grad_out_cpu[2] + grad_out_cpu[3] + grad_out_cpu[4]) / 3.0
    expected[3] += grad_out_cpu[5]
    # padding_idx row is always zero in backward
    expected[1].fill(0.0)

    np.testing.assert_allclose(g_np, expected, rtol=1e-6, atol=1e-6)


def test_embedding_backward_multidim_indices_cuda(cuda_autograd_enabled):
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    w_cpu = np.arange(8, dtype=np.float32).reshape((4, 2))
    weight = vt.cuda.to_device(w_cpu).detach()
    weight.requires_grad = True

    # 2D indices -> grad_out shape (2, 3, D). Include padding_idx and duplicates.
    idx_cpu = np.array([[0, 1, 3], [1, 0, 2]], dtype=np.int64)
    idx = vt.cuda.to_device(idx_cpu)

    out = vt.embedding(weight, idx, padding_idx=1, scale_grad_by_freq=False)

    grad_out_cpu = np.arange(12, dtype=np.float32).reshape((2, 3, 2))
    grad_out = vt.cuda.to_device(grad_out_cpu)

    out.backward(grad_out)

    assert weight.grad is not None
    g_np = vt.cuda.from_device(weight.grad())

    expected = np.zeros((4, 2), dtype=np.float32)
    expected[0] += grad_out_cpu[0, 0] + grad_out_cpu[1, 1]
    expected[2] += grad_out_cpu[1, 2]
    expected[3] += grad_out_cpu[0, 2]
    expected[1].fill(0.0)  # padding_idx row

    np.testing.assert_allclose(g_np, expected, rtol=1e-6, atol=1e-6)


def test_embedding_backward_noncontiguous_grad_out_cuda(cuda_autograd_enabled):
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    w_cpu = np.arange(15, dtype=np.float32).reshape((5, 3))
    weight = vt.cuda.to_device(w_cpu).detach()
    weight.requires_grad = True

    idx_cpu = np.array([0, 2, 2, 1], dtype=np.int64)
    idx = vt.cuda.to_device(idx_cpu)

    out = vt.embedding(weight, idx)

    grad_base_cpu = np.arange(24, dtype=np.float32).reshape((4, 6))
    grad_base = vt.cuda.to_device(grad_base_cpu)
    grad_out = grad_base.as_strided([4, 3], [6, 2], 0)
    assert not grad_out.is_contiguous()

    grad_out_cpu = np.lib.stride_tricks.as_strided(
        grad_base_cpu, shape=(4, 3), strides=(6 * 4, 2 * 4)
    )

    out.backward(grad_out)

    assert weight.grad is not None
    g_np = vt.cuda.from_device(weight.grad())

    expected = np.zeros((5, 3), dtype=np.float32)
    expected[0] += grad_out_cpu[0]
    expected[2] += grad_out_cpu[1] + grad_out_cpu[2]
    expected[1] += grad_out_cpu[3]

    np.testing.assert_allclose(g_np, expected, rtol=1e-6, atol=1e-6)


def test_embedding_backward_runs_when_D_is_zero_cuda(cuda_autograd_enabled):
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    weight = vt.cuda.to_device(np.zeros((4, 0), dtype=np.float32)).detach()
    weight.requires_grad = True

    idx = vt.cuda.to_device(np.array([0, 3], dtype=np.int64))
    out = vt.embedding(weight, idx)
    assert tuple(out.sizes) == (2, 0)

    grad_out = vt.cuda.to_device(np.zeros((2, 0), dtype=np.float32))
    out.backward(grad_out)

    assert weight.grad is not None
    np.testing.assert_allclose(
        vt.cuda.from_device(weight.grad()), np.zeros((4, 0), dtype=np.float32)
    )


def test_embedding_backward_empty_indices_cuda(cuda_autograd_enabled):
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    w_cpu = np.arange(12, dtype=np.float32).reshape((4, 3))
    weight = vt.cuda.to_device(w_cpu).detach()
    weight.requires_grad = True

    idx = vt.cuda.to_device(np.zeros((0,), dtype=np.int64))
    out = vt.embedding(weight, idx)
    assert tuple(out.sizes) == (0, 3)

    grad_out = vt.cuda.to_device(np.zeros((0, 3), dtype=np.float32))
    out.backward(grad_out)

    assert weight.grad is not None
    np.testing.assert_allclose(
        vt.cuda.from_device(weight.grad()), np.zeros((4, 3), dtype=np.float32)
    )
