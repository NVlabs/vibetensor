# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PyFunctionNode CUDA support (PyFunctionNode CUDA).

These tests verify that vibetensor.autograd.Function works correctly
with CUDA tensors when CUDA autograd is enabled.
"""
import pytest
import numpy as np

import vibetensor._C as C
import vibetensor.torch as vt
import vibetensor.autograd as vag


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0


@pytest.mark.cuda
def test_pyfunction_cuda_identity():
    """Test that a simple identity Function works on CUDA."""
    if not _cuda_available():
        pytest.skip("CUDA not available")

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())
    try:
        ag.set_cuda_autograd_enabled(True)

        # Define a simple Function that returns input unchanged
        class IdentityFn(vag.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def backward(ctx, grad_out):
                return (grad_out,)

        # Create CUDA tensor with requires_grad
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x = vt.from_numpy(x_np).cuda()
        x.set_requires_grad(True)

        # Forward through Function
        y = IdentityFn.apply(x)

        # Verify output has grad_fn
        meta = ag._debug_tensor_meta(y)
        assert meta.get("has_grad_fn") is True, "Output should have grad_fn"

        # Create CUDA gradient
        grad = vt.from_numpy(np.array([1.0, 1.0, 1.0], dtype=np.float32)).cuda()

        # Backward - should NOT throw "CpuOnly node received CUDA gradient"
        y.backward(grad)

        # Verify gradient was computed
        x_grad = x.grad
        assert x_grad is not None, "x.grad should not be None"

        # Verify gradient values (identity backward)
        x_grad_np = np.from_dlpack(x_grad.cpu())
        np.testing.assert_array_almost_equal(
            x_grad_np, [1.0, 1.0, 1.0],
            err_msg="Gradient should match input gradient (identity backward)"
        )
    finally:
        ag.set_cuda_autograd_enabled(prev)


@pytest.mark.cuda
def test_pyfunction_cuda_scale():
    """Test Function with scaling on CUDA."""
    if not _cuda_available():
        pytest.skip("CUDA not available")

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())
    try:
        ag.set_cuda_autograd_enabled(True)

        class Scale2xFn(vag.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return C.vt.add(x, x)  # 2x

            @staticmethod
            def backward(ctx, grad_out):
                # Gradient is 2 * upstream
                return (C.vt.add(grad_out, grad_out),)

        # Create CUDA tensor
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x = vt.from_numpy(x_np).cuda()
        x.set_requires_grad(True)

        # Forward
        y = Scale2xFn.apply(x)

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


@pytest.mark.cuda
def test_pyfunction_cuda_requires_cuda_autograd_enabled():
    """Test that CUDA Function fails when CUDA autograd is disabled."""
    if not _cuda_available():
        pytest.skip("CUDA not available")

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())
    try:
        # Disable CUDA autograd
        ag.set_cuda_autograd_enabled(False)

        class IdentityFn(vag.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, grad_out):
                return (grad_out,)

        # Create CUDA tensor
        x = vt.from_numpy(np.array([1.0], dtype=np.float32)).cuda()
        x.set_requires_grad(True)

        # Should raise error because CUDA autograd is disabled
        # Python layer: "supports autograd only for CPU float32 tensors (or CUDA float32 when CUDA autograd is enabled)"
        # C++ layer: "CUDA output requires CUDA autograd to be enabled"
        with pytest.raises(
            (TypeError, RuntimeError),
            match=r"CUDA (float32 when CUDA )?autograd.*(is )?enabled"
        ):
            IdentityFn.apply(x)
    finally:
        ag.set_cuda_autograd_enabled(prev)


@pytest.mark.cuda
def test_pyfunction_cuda_stream_kind_is_cuda_allowlisted():
    """Test that PyFunctionNode reports CudaAllowlisted stream_kind for CUDA inputs."""
    if not _cuda_available():
        pytest.skip("CUDA not available")

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())
    try:
        ag.set_cuda_autograd_enabled(True)

        # PyFunctionNode should have stream_kind() = CudaAllowlisted when
        # inputs are CUDA tensors, not the default CpuOnly.

        class PassthroughFn(vag.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, grad_out):
                return (grad_out,)

        # CUDA tensor
        x = vt.from_numpy(np.array([1.0], dtype=np.float32)).cuda()
        x.set_requires_grad(True)

        y = PassthroughFn.apply(x)
        grad = vt.from_numpy(np.array([1.0], dtype=np.float32)).cuda()

        # This should NOT raise "CpuOnly node received CUDA gradient"
        y.backward(grad)

        # Success - the fix is working
        assert x.grad is not None
    finally:
        ag.set_cuda_autograd_enabled(prev)


@pytest.mark.cuda
def test_pyfunction_cuda_multi_input():
    """Test that a 2-input Function works correctly on CUDA.
    
    This tests the num_incoming_grad_slots() fix - without it, a 2-input
    Function would wait for 2 grad slots but only receive 1.
    """
    if not _cuda_available():
        pytest.skip("CUDA not available")

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())
    try:
        ag.set_cuda_autograd_enabled(True)

        class AddFn(vag.Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                # Use vt.ops for CUDA-compatible add
                return C._call_op("vt::add", a, b)

            @staticmethod
            def backward(ctx, grad_out):
                # d(a+b)/da = 1, d(a+b)/db = 1
                return (grad_out, grad_out)

        a = vt.from_numpy(np.array([1.0, 2.0], dtype=np.float32)).cuda()
        b = vt.from_numpy(np.array([3.0, 4.0], dtype=np.float32)).cuda()
        a.requires_grad = True
        b.requires_grad = True

        y = AddFn.apply(a, b)
        grad = vt.from_numpy(np.array([1.0, 1.0], dtype=np.float32)).cuda()
        y.backward(grad)

        # Both inputs should have gradients
        assert a.grad is not None, "a.grad should be set"
        assert b.grad is not None, "b.grad should be set"
    finally:
        ag.set_cuda_autograd_enabled(prev)
