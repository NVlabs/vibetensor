# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vibe_kernels.optim import TritonMuon  # type: ignore[import]
from vibe_kernels.optim.impl.triton_impl import (
    fast_newton_schulz,
    matmul_transpose,
    matmul_transpose_assign,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Muon kernels"
)
def test_matmul_transpose_matches_torch():
    torch.manual_seed(0)
    matrix = torch.randn(128, 64, device="cuda", dtype=torch.float16)
    gram_triton = matmul_transpose(matrix)
    gram_ref = (matrix @ matrix.transpose(-1, -2)).to(gram_triton.dtype)
    torch.testing.assert_close(gram_triton, gram_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Muon kernels"
)
def test_matmul_transpose_assign_matches_function():
    torch.manual_seed(1)
    matrix = torch.randn(96, 48, device="cuda", dtype=torch.float16)
    out = torch.empty(
        (matrix.size(0), matrix.size(0)), device=matrix.device, dtype=matrix.dtype
    )
    matmul_transpose_assign(matrix, out)
    via_function = matmul_transpose(matrix)
    torch.testing.assert_close(out, via_function)
    torch.testing.assert_close(out, out.transpose(-1, -2))


def _reference_newton_schulz(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = matrix.to(torch.bfloat16)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.mT
        transposed = True
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        buf1 = X @ X.transpose(-1, -2)
        buf2 = buf1 @ buf1.transpose(-1, -2)
        B = b * buf1 + c * buf2
        X = a * X + torch.matmul(B.to(X.dtype), X)
    if transposed:
        X = X.mT
    return X.to(matrix.dtype)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Muon kernels"
)
def test_newton_schulz_matches_reference():
    torch.manual_seed(2)
    matrix = torch.randn(64, 80, device="cuda", dtype=torch.float32)
    result = fast_newton_schulz(matrix, steps=4)
    reference = _reference_newton_schulz(matrix, steps=4)
    torch.testing.assert_close(result, reference, atol=1.5e-2, rtol=1e-2)

    gram_result = (result @ result.transpose(-1, -2)).float()
    gram_reference = (reference @ reference.transpose(-1, -2)).float()
    torch.testing.assert_close(gram_result, gram_reference, atol=7e-2, rtol=8e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Muon kernels"
)
def test_triton_muon_matches_reference_step():
    device = torch.device("cuda")
    torch.manual_seed(3)

    param_ref = torch.randn(
        64, 32, device=device, dtype=torch.float32, requires_grad=True
    )
    param_new = param_ref.detach().clone().requires_grad_(True)
    grad = torch.randn_like(param_ref)
    param_ref.grad = grad.clone()
    param_new.grad = grad.clone()

    lr = 0.02
    momentum = 0.95
    weight_decay = 0.01
    ns_steps = 5

    # Reference update
    buf_ref = torch.zeros_like(param_ref)
    buf_ref.mul_(momentum).add_(grad)
    update_ref = grad + momentum * buf_ref
    matrix_ref = update_ref.reshape(update_ref.shape[0], -1)
    orth_ref = fast_newton_schulz(matrix_ref, steps=ns_steps).reshape_as(update_ref)
    scale = max(1.0, matrix_ref.size(0) / matrix_ref.size(1)) ** 0.5
    with torch.no_grad():
        param_ref.mul_(1 - lr * weight_decay)
        param_ref.add_(orth_ref, alpha=-lr * scale)

    optimizer = TritonMuon(
        [param_new],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        ns_steps=ns_steps,
    )
    optimizer.step()

    torch.cuda.synchronize()
    torch.testing.assert_close(param_ref, param_new, atol=2e-3, rtol=1e-3)
