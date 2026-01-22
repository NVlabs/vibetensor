# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vibe_kernels.activation import (
    elementwise_add,
    elementwise_lerp,
    elementwise_mul,
    elementwise_where,
    rowwise_l2_norm,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_elementwise_add_mul():
    torch.manual_seed(0)
    a = torch.randn(16, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    assert torch.allclose(elementwise_add(a, b), a + b, atol=1e-6, rtol=0)
    assert torch.allclose(elementwise_mul(a, b), a * b, atol=1e-6, rtol=0)


def test_elementwise_where():
    torch.manual_seed(1)
    a = torch.randn(8, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    cond = torch.rand_like(a) > 0.5
    assert torch.allclose(
        elementwise_where(cond, a, b), torch.where(cond, a, b), atol=1e-6, rtol=0
    )


def test_elementwise_lerp_scalar_and_tensor():
    torch.manual_seed(2)
    a = torch.randn(32, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    weight_scalar = 0.25
    weight_tensor = torch.rand_like(a)
    assert torch.allclose(
        elementwise_lerp(a, b, weight_scalar),
        torch.lerp(a, b, weight_scalar),
        atol=1e-6,
        rtol=0,
    )
    assert torch.allclose(
        elementwise_lerp(a, b, weight_tensor),
        torch.lerp(a, b, weight_tensor),
        atol=1e-6,
        rtol=0,
    )


def test_rowwise_l2_norm():
    torch.manual_seed(3)
    x = torch.randn(4, 6, device="cuda", dtype=torch.float32)
    assert torch.allclose(
        rowwise_l2_norm(x), torch.linalg.norm(x, dim=-1), atol=1e-6, rtol=0
    )
