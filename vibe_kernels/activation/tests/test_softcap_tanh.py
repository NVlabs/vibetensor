# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vibe_kernels.activation import softcap_tanh_projection

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _softcap_reference(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + x.abs())


def _softcap_tanh_reference(x: torch.Tensor, weight: torch.Tensor):
    proj = x * weight
    return _softcap_reference(proj), torch.tanh(proj)


def test_softcap_tanh_projection_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(3, 5, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(5, device="cuda", dtype=torch.float16)
    softcap_out, tanh_out = softcap_tanh_projection(x, weight)
    ref_softcap, ref_tanh = _softcap_tanh_reference(x, weight)
    assert torch.allclose(softcap_out, ref_softcap, atol=2e-3, rtol=0)
    assert torch.allclose(tanh_out, ref_tanh, atol=2e-3, rtol=0)

    grad_softcap = torch.randn_like(softcap_out)
    grad_tanh = torch.randn_like(tanh_out)
    torch.autograd.backward((softcap_out, tanh_out), (grad_softcap, grad_tanh))

    x2 = x.detach().clone().requires_grad_(True)
    sc_ref, tanh_ref = _softcap_tanh_reference(x2, weight)
    torch.autograd.backward((sc_ref, tanh_ref), (grad_softcap, grad_tanh))

    assert torch.allclose(x.grad, x2.grad, atol=2e-3, rtol=0)
