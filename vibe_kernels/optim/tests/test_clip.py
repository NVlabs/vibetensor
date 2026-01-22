# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from vibe_kernels.optim import (  # type: ignore[import]
    clip_grad_norm_,
    compute_global_grad_norm,
)


def test_compute_global_grad_norm_matches_torch():
    torch.manual_seed(0)
    params = [
        torch.randn(4, 5, requires_grad=True),
        torch.randn(3, 2, requires_grad=True),
    ]
    grads = []
    for p in params:
        g = torch.randn_like(p)
        p.grad = g
        grads.append(g)

    expected_sq = torch.zeros((), dtype=torch.float64)
    for g in grads:
        expected_sq += g.double().pow(2).sum()
    expected = expected_sq.sqrt().to(torch.float32)
    custom = compute_global_grad_norm(params)
    assert torch.allclose(expected.to(custom.dtype), custom, atol=1e-6)


def test_clip_grad_norm_in_place():
    torch.manual_seed(0)
    params_ref = [
        torch.randn(6, 4, requires_grad=True),
        torch.randn(2, 3, requires_grad=True),
    ]
    params_new = [p.detach().clone().requires_grad_(True) for p in params_ref]

    grads = [torch.randn_like(p) for p in params_ref]
    for g, p_ref, p_new in zip(grads, params_ref, params_new):
        p_ref.grad = g.clone()
        p_new.grad = g.clone()

    torch.nn.utils.clip_grad_norm_(params_ref, max_norm=0.5)
    scale = clip_grad_norm_(params_new, max_norm=0.5)

    assert scale <= 1
    for p_ref, p_new in zip(params_ref, params_new):
        assert p_ref.grad is not None and p_new.grad is not None
        assert torch.allclose(p_ref.grad, p_new.grad, atol=1e-6)
