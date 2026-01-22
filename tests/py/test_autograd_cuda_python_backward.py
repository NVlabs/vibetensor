# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PythonBackwardNode CUDA support (PyFunctionNode CUDA).

These tests verify that Python-registered backward functions work correctly
with CUDA tensors when CUDA autograd is enabled.
"""
import pytest
import numpy as np

import vibetensor._C as C
import vibetensor.torch as vt


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0


@pytest.mark.cuda
def test_python_backward_cuda_simple_identity():
    """Test that a simple identity Python backward works on CUDA."""
    if not _cuda_available():
        pytest.skip("CUDA not available")

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())
    try:
        ag.set_cuda_autograd_enabled(True)

        # Define a custom op schema (use getattr since 'def' is a Python keyword)
        c_def = getattr(C, "def")
        try:
            c_def("test_py_cuda::identity(Tensor) -> Tensor")
        except Exception:
            pass  # Already defined

        # Forward: just return input
        def _fwd(x):
            return x.clone()

        # Register forward
        C._try_register_boxed_python_override("test_py_cuda::identity", _fwd)

        # Backward: identity gradient
        def _bw(grads_in, saved):
            (grad_out,) = grads_in
            return (grad_out,)

        # Register backward (ok=False means already registered, which is fine)
        C._try_register_boxed_autograd_fallback("test_py_cuda::identity", _bw)

        # Create CUDA tensor with requires_grad
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x = vt.from_numpy(x_np).cuda()
        x.set_requires_grad(True)

        # Forward
        y = C._call_op("test_py_cuda::identity", x)

        # Verify output has grad_fn
        meta = ag._debug_tensor_meta(y)
        assert meta.get("has_grad_fn") is True, "Output should have grad_fn"

        # Create CUDA gradient
        grad = vt.from_numpy(np.array([1.0, 1.0, 1.0], dtype=np.float32)).cuda()

        # Backward - this should NOT throw "CpuOnly node received CUDA gradient"
        y.backward(grad)

        # Verify gradient was computed
        x_grad = x.grad
        assert x_grad is not None, "x.grad should not be None"

        # Verify gradient values (identity backward should give same gradient)
        x_grad_np = np.from_dlpack(x_grad.cpu())
        np.testing.assert_array_almost_equal(
            x_grad_np, [1.0, 1.0, 1.0],
            err_msg="Gradient should match input gradient (identity backward)"
        )
    finally:
        ag.set_cuda_autograd_enabled(prev)
        C._reset_autograd_py()


@pytest.mark.cuda
def test_python_backward_cuda_scale():
    """Test Python backward with scaling on CUDA."""
    if not _cuda_available():
        pytest.skip("CUDA not available")

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())
    try:
        ag.set_cuda_autograd_enabled(True)

        # Define a custom op schema (use getattr since 'def' is a Python keyword)
        c_def = getattr(C, "def")
        try:
            c_def("test_py_cuda::scale2x(Tensor) -> Tensor")
        except Exception:
            pass

        # Forward: scale by 2
        def _fwd(x):
            return C.vt.add(x, x)  # 2x

        C._try_register_boxed_python_override("test_py_cuda::scale2x", _fwd)

        # Backward: gradient is 2 * upstream
        def _bw(grads_in, saved):
            (grad_out,) = grads_in
            grad_x = C.vt.add(grad_out, grad_out)  # 2 * grad_out
            return (grad_x,)

        C._try_register_boxed_autograd_fallback("test_py_cuda::scale2x", _bw)

        # Create CUDA tensor
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x = vt.from_numpy(x_np).cuda()
        x.set_requires_grad(True)

        # Forward
        y = C._call_op("test_py_cuda::scale2x", x)

        # Backward
        grad = vt.from_numpy(np.array([1.0, 1.0, 1.0], dtype=np.float32)).cuda()
        y.backward(grad)

        # Verify gradient (should be 2x the upstream gradient)
        x_grad = x.grad
        assert x_grad is not None
        x_grad_np = np.from_dlpack(x_grad.cpu())
        np.testing.assert_array_almost_equal(
            x_grad_np, [2.0, 2.0, 2.0],
            err_msg="Gradient should be 2x (scale backward)"
        )
    finally:
        ag.set_cuda_autograd_enabled(prev)
        C._reset_autograd_py()


@pytest.mark.cuda
def test_python_backward_cuda_multi_input():
    """Test Python backward with multiple inputs on CUDA."""
    if not _cuda_available():
        pytest.skip("CUDA not available")

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())
    try:
        ag.set_cuda_autograd_enabled(True)

        # Define a custom op schema (use getattr since 'def' is a Python keyword)
        c_def = getattr(C, "def")
        try:
            c_def("test_py_cuda::add2(Tensor, Tensor) -> Tensor")
        except Exception:
            pass

        # Forward: add two tensors
        def _fwd(a, b):
            return C.vt.add(a, b)

        C._try_register_boxed_python_override("test_py_cuda::add2", _fwd)

        # Backward: gradient flows to both inputs
        def _bw(grads_in, saved):
            (grad_out,) = grads_in
            return (grad_out, grad_out)

        C._try_register_boxed_autograd_fallback("test_py_cuda::add2", _bw)

        # Create CUDA tensors
        a_np = np.array([1.0, 2.0], dtype=np.float32)
        b_np = np.array([3.0, 4.0], dtype=np.float32)
        a = vt.from_numpy(a_np).cuda()
        b = vt.from_numpy(b_np).cuda()
        a.set_requires_grad(True)
        b.set_requires_grad(True)

        # Forward
        y = C._call_op("test_py_cuda::add2", a, b)

        # Backward
        grad = vt.from_numpy(np.array([1.0, 1.0], dtype=np.float32)).cuda()
        y.backward(grad)

        # Verify gradients
        a_grad = a.grad
        b_grad = b.grad
        assert a_grad is not None
        assert b_grad is not None

        a_grad_np = np.from_dlpack(a_grad.cpu())
        b_grad_np = np.from_dlpack(b_grad.cpu())
        np.testing.assert_array_almost_equal(a_grad_np, [1.0, 1.0])
        np.testing.assert_array_almost_equal(b_grad_np, [1.0, 1.0])
    finally:
        ag.set_cuda_autograd_enabled(prev)
        C._reset_autograd_py()


@pytest.mark.cuda
def test_python_backward_cuda_stream_kind_is_cuda_allowlisted():
    """Test that PythonBackwardNode reports CudaAllowlisted stream_kind for CUDA inputs."""
    if not _cuda_available():
        pytest.skip("CUDA not available")

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())
    try:
        ag.set_cuda_autograd_enabled(True)

        # PythonBackwardNode should have stream_kind() = CudaAllowlisted when
        # inputs are CUDA tensors, not the default CpuOnly.
        
        # The fact that backward() doesn't throw
        # "CpuOnly node received CUDA gradient" proves the fix works.

        # (use getattr since 'def' is a Python keyword)
        c_def = getattr(C, "def")
        try:
            c_def("test_py_cuda::passthrough(Tensor) -> Tensor")
        except Exception:
            pass

        def _fwd(x):
            return x.clone()

        C._try_register_boxed_python_override("test_py_cuda::passthrough", _fwd)

        def _bw(grads_in, saved):
            return grads_in

        C._try_register_boxed_autograd_fallback("test_py_cuda::passthrough", _bw)

        # CUDA tensor
        x = vt.from_numpy(np.array([1.0], dtype=np.float32)).cuda()
        x.set_requires_grad(True)

        y = C._call_op("test_py_cuda::passthrough", x)
        grad = vt.from_numpy(np.array([1.0], dtype=np.float32)).cuda()

        # This should NOT raise "CpuOnly node received CUDA gradient"
        y.backward(grad)

        # Success - the fix is working
        assert x.grad is not None
    finally:
        ag.set_cuda_autograd_enabled(prev)
        C._reset_autograd_py()
