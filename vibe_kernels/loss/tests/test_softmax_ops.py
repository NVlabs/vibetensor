# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from vibe_kernels.loss import (
    log_softmax as triton_log_softmax,
    softmax as triton_softmax,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _tolerance(dtype: torch.dtype) -> float:
    if dtype == torch.float32:
        return 5e-5
    if dtype == torch.bfloat16:
        return 1.5e-2
    return 5e-3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim", [-1, 1])
@pytest.mark.parametrize("backend", ["triton", "cutedsl"])
def test_softmax_matches_torch(dtype: torch.dtype, dim: int, backend: str) -> None:
    if backend == "cutedsl":
        from vibe_kernels.softmax.kernel import is_cutedsl_available

        if not is_cutedsl_available():
            pytest.skip("CuTeDSL softmax backend is not available")
    torch.manual_seed(0)
    x_backend = torch.randn(3, 5, 7, device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x_backend.detach().clone().to(torch.float32).requires_grad_(True)

    ours = triton_softmax(x_backend, dim=dim, backend=backend)
    ref = F.softmax(x_ref, dim=dim)

    tol = _tolerance(dtype)
    assert torch.allclose(ours.float(), ref, atol=tol, rtol=0)

    grad = torch.randn_like(ref)
    ours.backward(grad.to(dtype))
    ref.backward(grad)

    assert x_backend.grad is not None and x_ref.grad is not None
    assert torch.allclose(x_backend.grad.float(), x_ref.grad, atol=tol, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("backend", ["triton", "cutedsl"])
def test_log_softmax_matches_torch(dtype: torch.dtype, dim: int, backend: str) -> None:
    if backend == "cutedsl":
        from vibe_kernels.softmax.kernel import is_cutedsl_available

        if not is_cutedsl_available():
            pytest.skip("CuTeDSL softmax backend is not available")
    torch.manual_seed(1)
    x_backend = torch.randn(4, 6, device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x_backend.detach().clone().to(torch.float32).requires_grad_(True)

    ours = triton_log_softmax(x_backend, dim=dim, backend=backend)
    ref = F.log_softmax(x_ref, dim=dim)

    tol = _tolerance(dtype)
    assert torch.allclose(ours.float(), ref, atol=tol, rtol=0)

    grad = torch.randn_like(ref)
    ours.backward(grad.to(dtype))
    ref.backward(grad)

    assert x_backend.grad is not None and x_ref.grad is not None
    assert torch.allclose(x_backend.grad.float(), x_ref.grad, atol=tol, rtol=0)
