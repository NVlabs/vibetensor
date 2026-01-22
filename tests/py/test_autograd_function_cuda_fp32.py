# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor._C as C
from vibetensor.autograd import Function


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0  # type: ignore[attr-defined]


class Square(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        # Avoid CPU scalar ops: factories are CPU-only.
        return ((x + x) * grad_out,)


@pytest.mark.cuda
def test_autograd_function_cuda_fp32_forward_backward_explicit_grad() -> None:
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    try:
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]

        x = C._make_cuda_tensor([], "float32", 3.0)
        x.requires_grad = True

        y = Square.apply(x)
        assert isinstance(y, C.Tensor)
        assert bool(getattr(y, "requires_grad", False))

        # Sanity: PyFunctionNode should be CUDA allowlisted and have stream info.
        handle, _output_nr = ag._graph_get_gradient_edge(y)  # type: ignore[attr-defined]
        info = ag._grad_fn_stream_info(handle)  # type: ignore[attr-defined]
        assert info["stream_kind"] == "cuda_allowlisted"
        assert bool(info["has_canonical_stream"]) is True
        assert int(info["device_type"]) == 2  # kDLCUDA
        assert int(info["device_index"]) == int(x.device[1])

        grad = C._make_cuda_tensor([], "float32", 1.0)
        y.backward(grad)

        gx = x.grad
        assert gx is not None
        assert gx.dtype == "float32"
        assert gx.device[0] == 2
        assert gx.device[1] == x.device[1]
        assert float(gx.cpu().item()) == pytest.approx(6.0)
    finally:
        ag.set_cuda_autograd_enabled(prev)  # type: ignore[attr-defined]
