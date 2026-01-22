# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor._C as C
import vibetensor.torch as vt


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0


pytestmark = pytest.mark.cuda


def test_ci1_cuda_backward_rejects_cpu_grad_tensor_mixed_device() -> None:
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    try:
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]

        x = C._make_cuda_tensor([4], "float32", 1.0)
        x.set_requires_grad(True)
        y = C._call_op("vt::add", x, x)

        grad_cpu = vt.tensor([1.0, 1.0, 1.0, 1.0], dtype="float32")
        with pytest.raises(ValueError) as ei:
            y.backward(grad_cpu)
        assert "gradient dtype/device/shape must match tensor" in str(ei.value)
    finally:
        ag.set_cuda_autograd_enabled(prev)  # type: ignore[attr-defined]


def test_ci1_cuda_backward_rejects_non_dense_grad_tensor() -> None:
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    try:
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]

        # Build a graph that causes duplicate accumulation into a CUDA leaf.
        x = C._make_cuda_tensor([4], "float32", 1.0)
        x.set_requires_grad(True)
        y = C._call_op("vt::add", x, x)

        # Construct a non-dense 1D view with the *same* sizes as y.
        base = C._make_cuda_tensor([7], "float32", 1.0)
        grad = base.as_strided([4], [2], 0)
        assert grad.sizes == y.sizes
        assert grad.is_non_overlapping_and_dense() is False

        with pytest.raises(RuntimeError) as ei:
            y.backward(grad)
        msg = str(ei.value)
        assert (
            "layout mismatch" in msg
            or "non-dense" in msg
            or "metadata mismatch" in msg
        )
    finally:
        ag.set_cuda_autograd_enabled(prev)  # type: ignore[attr-defined]


def test_ci1_cuda_backward_rejects_non_float32_roots() -> None:
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    # This test exercises the Python binding guardrail: roots on CUDA must be float32/float16.
    x = C._make_cuda_tensor([4], "int64", 1.0)
    x.set_requires_grad(True)
    y = C._call_op("vt::add", x, x)

    grad = C._make_cuda_tensor([4], "int64", 1.0)
    with pytest.raises(ValueError) as ei:
        y.backward(grad)
    assert "Float32/Float16 CUDA" in str(ei.value)
