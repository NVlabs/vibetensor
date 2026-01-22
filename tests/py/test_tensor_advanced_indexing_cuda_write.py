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


def _has_cuda() -> bool:
    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) > 0
    except Exception:
        return False


def test_advanced_index_write_1d_tensor_index_cuda_matches_cpu():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    # CPU baseline.
    x_cpu = vt.zeros((4,), dtype="float32")
    idx_cpu = vt.tensor([1, 3], dtype="int64")
    v_cpu = vt.tensor([5.0, 7.0], dtype="float32")

    x_cpu[idx_cpu] = v_cpu
    x_cpu_np = _to_numpy_cpu(x_cpu)

    # CUDA path mirrors the same update starting from zeros.
    x0_np = np.zeros_like(x_cpu_np, dtype=np.float32)
    idx_np = _to_numpy_cpu(idx_cpu).astype(np.int64)
    v_np = _to_numpy_cpu(v_cpu)

    x_cuda = vt.cuda.to_device(x0_np)
    idx_cuda = vt.cuda.to_device(idx_np)
    v_cuda = vt.cuda.to_device(v_np)

    x_cuda[idx_cuda] = v_cuda
    x_cuda_np = vt.cuda.from_device(x_cuda)

    np.testing.assert_allclose(x_cuda_np, x_cpu_np)


def test_advanced_index_write_2d_prefix_tensor_index_cuda_matches_cpu():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    x_cpu = vt.zeros((2, 3), dtype="float32")
    idx_cpu = vt.tensor([0, 2], dtype="int64")
    v_cpu = vt.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")

    x_cpu[:, idx_cpu] = v_cpu
    x_cpu_np = _to_numpy_cpu(x_cpu)

    x0_np = np.zeros_like(x_cpu_np, dtype=np.float32)
    idx_np = _to_numpy_cpu(idx_cpu).astype(np.int64)
    v_np = _to_numpy_cpu(v_cpu)

    x_cuda = vt.cuda.to_device(x0_np)
    idx_cuda = vt.cuda.to_device(idx_np)
    v_cuda = vt.cuda.to_device(v_np)

    x_cuda[:, idx_cuda] = v_cuda
    x_cuda_np = vt.cuda.from_device(x_cuda)

    np.testing.assert_allclose(x_cuda_np, x_cpu_np)


def test_advanced_index_write_cuda_rejects_autograd_when_requires_grad():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    x0_np = np.zeros(3, dtype=np.float32)
    x_cuda = vt.cuda.to_device(x0_np)
    x_cuda.set_requires_grad(True)

    idx_np = np.array([0, 2], dtype=np.int64)
    v_np = np.array([1.0, 2.0], dtype=np.float32)

    idx_cuda = vt.cuda.to_device(idx_np)
    v_cuda = vt.cuda.to_device(v_np)

    ag = C.autograd
    prev = ag.is_grad_enabled()
    try:
        ag.set_grad_enabled(True)
        with pytest.raises(
            RuntimeError,
            match="autograd for in-place advanced indexing is not supported",
        ):
            x_cuda[idx_cuda] = v_cuda
    finally:
        ag.set_grad_enabled(prev)

    # Data should remain unchanged on failure.
    np.testing.assert_allclose(vt.cuda.from_device(x_cuda), x0_np)


def test_advanced_index_write_allows_inference_mode_context_cuda():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    x0_np = np.zeros(3, dtype=np.float32)
    x_cuda = vt.cuda.to_device(x0_np)
    x_cuda.set_requires_grad(True)

    idx_np = np.array([0, 2], dtype=np.int64)
    v_np = np.array([1.0, 2.0], dtype=np.float32)

    idx_cuda = vt.cuda.to_device(idx_np)
    v_cuda = vt.cuda.to_device(v_np)

    expected = x0_np.copy()
    expected[[0, 2]] = [1.0, 2.0]

    # Under inference_mode, the in-place CUDA write should behave like no_grad
    # even though the tensor requires grad.
    with vt.inference_mode():
        x_cuda[idx_cuda] = v_cuda

    np.testing.assert_allclose(vt.cuda.from_device(x_cuda), expected)


def test_tensor_index_put_1d_tensor_index_cuda_matches_cpu():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    # CPU baseline.
    x_cpu = vt.zeros((4,), dtype="float32")
    idx_cpu = vt.tensor([1, 3], dtype="int64")
    v_cpu = vt.tensor([5.0, 7.0], dtype="float32")

    x_cpu.index_put_((idx_cpu,), v_cpu, accumulate=False)
    x_cpu_np = _to_numpy_cpu(x_cpu)

    # CUDA path mirrors the same update starting from zeros.
    x0_np = np.zeros_like(x_cpu_np, dtype=np.float32)
    idx_np = _to_numpy_cpu(idx_cpu).astype(np.int64)
    v_np = _to_numpy_cpu(v_cpu)

    x_cuda = vt.cuda.to_device(x0_np)
    idx_cuda = vt.cuda.to_device(idx_np)
    v_cuda = vt.cuda.to_device(v_np)

    x_cuda.index_put_((idx_cuda,), v_cuda, accumulate=False)
    x_cuda_np = vt.cuda.from_device(x_cuda)

    np.testing.assert_allclose(x_cuda_np, x_cpu_np)


def test_tensor_index_put_cuda_rejects_autograd_when_requires_grad():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    x0_np = np.zeros(3, dtype=np.float32)
    x_cuda = vt.cuda.to_device(x0_np)
    x_cuda.set_requires_grad(True)

    idx_np = np.array([0, 2], dtype=np.int64)
    v_np = np.array([1.0, 2.0], dtype=np.float32)

    idx_cuda = vt.cuda.to_device(idx_np)
    v_cuda = vt.cuda.to_device(v_np)

    ag = C.autograd
    prev = ag.is_grad_enabled()
    try:
        ag.set_grad_enabled(True)
        with pytest.raises(
            RuntimeError,
            match="autograd for in-place advanced indexing is not supported",
        ):
            x_cuda.index_put_((idx_cuda,), v_cuda, accumulate=False)
    finally:
        ag.set_grad_enabled(prev)

    # Data should remain unchanged on failure.
    np.testing.assert_allclose(vt.cuda.from_device(x_cuda), x0_np)


def test_tensor_index_put_accumulate_true_cuda_matches_cpu():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    x_cpu = vt.zeros((5,), dtype="float32")
    idx_cpu = vt.tensor([0, 1, 0, 4], dtype="int64")
    v_cpu = vt.tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")

    x_cpu.index_put_((idx_cpu,), v_cpu, accumulate=True)
    x_cpu_np = _to_numpy_cpu(x_cpu)

    x0_np = np.zeros_like(x_cpu_np, dtype=np.float32)
    idx_np = _to_numpy_cpu(idx_cpu).astype(np.int64)
    v_np = _to_numpy_cpu(v_cpu)

    x_cuda = vt.cuda.to_device(x0_np)
    idx_cuda = vt.cuda.to_device(idx_np)
    v_cuda = vt.cuda.to_device(v_np)

    x_cuda.index_put_((idx_cuda,), v_cuda, accumulate=True)
    x_cuda_np = vt.cuda.from_device(x_cuda)

    np.testing.assert_allclose(x_cuda_np, x_cpu_np, rtol=1e-5, atol=1e-5)
