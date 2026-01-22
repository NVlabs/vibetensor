# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vibe_kernels.activation import relu_squared

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_relu_squared_forward_backward():
    torch.manual_seed(0)
    x = torch.randn(4, 5, device="cuda", dtype=torch.float32, requires_grad=True)
    y = relu_squared(x)
    ref = torch.relu(x) ** 2
    assert torch.allclose(y, ref, atol=1e-6, rtol=0)

    loss = y.sum()
    loss.backward()

    x2 = x.detach().clone().requires_grad_(True)
    ref_loss = (torch.relu(x2) ** 2).sum()
    ref_loss.backward()
    assert torch.allclose(x.grad, x2.grad, atol=1e-6, rtol=0)
