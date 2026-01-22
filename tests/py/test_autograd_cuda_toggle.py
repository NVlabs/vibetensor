# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor._C as C
import vibetensor.autograd as A


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0


def test_cuda_autograd_toggle_roundtrip_overlay_and_bindings():
    ag = C.autograd

    # CPU-only builds may not expose CUDA; toggle should degrade to a stub.
    if not getattr(C, "_has_cuda", False):
        assert A.is_cuda_autograd_enabled() is False
        A.set_cuda_autograd_enabled(True)  # no-op
        assert A.is_cuda_autograd_enabled() is False
        return

    prev = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    try:
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]
        assert ag.is_cuda_autograd_enabled() is True  # type: ignore[attr-defined]
        assert A.is_cuda_autograd_enabled() is True

        A.set_cuda_autograd_enabled(False)
        assert ag.is_cuda_autograd_enabled() is False  # type: ignore[attr-defined]
        assert A.is_cuda_autograd_enabled() is False
    finally:
        ag.set_cuda_autograd_enabled(prev)  # type: ignore[attr-defined]


@pytest.mark.cuda
def test_cuda_backward_rejected_when_toggle_disabled():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    try:
        ag.set_cuda_autograd_enabled(False)  # type: ignore[attr-defined]

        x = C._make_cuda_tensor([2], "float32", 1.0)
        x.set_requires_grad(True)

        # Trigger wrapper construction so the output has grad_fn.
        y = C._call_op("vt::add", x, x)
        meta = ag._debug_tensor_meta(y)  # type: ignore[attr-defined]
        assert meta.get("has_grad_fn") is True

        grad = C._make_cuda_tensor([2], "float32", 1.0)
        with pytest.raises(ValueError) as ei:
            y.backward(grad)
        assert "Float32/Float16 CUDA when CUDA autograd is enabled" in str(ei.value)
    finally:
        ag.set_cuda_autograd_enabled(prev)  # type: ignore[attr-defined]
