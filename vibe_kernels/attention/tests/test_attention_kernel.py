# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from vibe_kernels.attention import fused_attention, reshape_for_gqa

pytestmark = pytest.mark.skipif(  # type: ignore[assignment]
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _sdpa_reference(q, k, v, *, causal: bool, scale: float) -> torch.Tensor:
    batch, hq, seqlen, dim = q.shape
    hk = k.shape[1]
    group = hq // hk
    qg = (
        q.view(batch, hk, group, seqlen, dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * group, hk, seqlen, dim)
    )
    kg = (
        k.unsqueeze(1).repeat(1, group, 1, 1, 1).reshape(batch * group, hk, seqlen, dim)
    )
    vg = (
        v.unsqueeze(1).repeat(1, group, 1, 1, 1).reshape(batch * group, hk, seqlen, dim)
    )
    out = torch.nn.functional.scaled_dot_product_attention(
        qg,
        kg,
        vg,
        is_causal=causal,
        dropout_p=0.0,
        scale=scale,
    )
    out = (
        out.view(batch, group, hk, seqlen, dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, hq, seqlen, dim)
    )
    return out


def test_fused_attention_matches_torch():
    torch.manual_seed(0)
    batch, hq, hk, seqlen, dim = 2, 4, 2, 32, 32
    scale = 1.0 / (dim**0.5)
    q = torch.randn(
        batch, hq, seqlen, dim, device="cuda", dtype=torch.float32, requires_grad=True
    )
    k = torch.randn(
        batch, hk, seqlen, dim, device="cuda", dtype=torch.float32, requires_grad=True
    )
    v = torch.randn_like(k, requires_grad=True)

    out = fused_attention(q, k, v, causal=True, sm_scale=scale, warp_specialize=False)
    ref = _sdpa_reference(q, k, v, causal=True, scale=scale)

    assert torch.allclose(out, ref, atol=2e-3, rtol=0)

    loss = torch.nn.functional.mse_loss(out, ref)
    loss.backward()

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    ref2 = _sdpa_reference(q2, k2, v2, causal=True, scale=scale)
    loss2 = torch.nn.functional.mse_loss(ref2, ref.detach())
    loss2.backward()

    assert torch.allclose(q.grad, q2.grad, atol=5e-4, rtol=0)
    assert torch.allclose(k.grad, k2.grad, atol=5e-4, rtol=0)
    assert torch.allclose(v.grad, v2.grad, atol=5e-4, rtol=0)


def test_fused_attention_bfloat16_matches_torch():
    torch.manual_seed(2)
    batch, hq, hk, seqlen, dim = 2, 4, 2, 32, 64
    scale = 1.0 / (dim**0.5)
    q = torch.randn(
        batch, hq, seqlen, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(
        batch, hk, seqlen, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn_like(k, requires_grad=True)

    out = fused_attention(q, k, v, causal=True, sm_scale=scale, warp_specialize=False)
    ref_inputs = [t.detach().float().requires_grad_(True) for t in (q, k, v)]
    ref = _sdpa_reference(
        ref_inputs[0], ref_inputs[1], ref_inputs[2], causal=True, scale=scale
    )

    assert torch.allclose(out.float(), ref, atol=1e-2, rtol=0)

    grad = torch.randn_like(ref)
    out.backward(grad.to(out.dtype))
    ref.backward(grad)

    for grad_a, grad_b in zip(
        (q.grad, k.grad, v.grad),
        (ref_inputs[0].grad, ref_inputs[1].grad, ref_inputs[2].grad),
    ):
        assert grad_a is not None and grad_b is not None
        assert torch.allclose(grad_a.float(), grad_b, atol=5e-2, rtol=0)


def test_fused_attention_disable_warp_specialize_env():
    torch.manual_seed(1)
    batch, hq, hk, seqlen, dim = 1, 2, 2, 32, 16
    scale = 1.0 / (dim**0.5)
    q = torch.randn(batch, hq, seqlen, dim, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, hk, seqlen, dim, device="cuda", dtype=torch.float32)
    v = torch.randn_like(k)

    os.environ["AIKF_DISABLE_WARP_SPECIALIZE"] = "1"
    try:
        out = fused_attention(
            q, k, v, causal=False, sm_scale=scale, warp_specialize=True
        )
        ref = _sdpa_reference(q, k, v, causal=False, scale=scale)
        assert torch.allclose(out, ref, atol=1e-3, rtol=0)
    finally:
        os.environ.pop("AIKF_DISABLE_WARP_SPECIALIZE", None)
