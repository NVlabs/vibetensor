# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# pyright: reportMissingImports=false

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "tmp"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - optional dependency
    from vibe_kernels.layernorm import (  # type: ignore[import]
        layernorm as cutedsl_layernorm,
        layernorm_ref,
    )
except ImportError:  # pragma: no cover
    pytest.skip("kernel_factory package is not available", allow_module_level=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_layernorm_backward_matches_pytorch(
    dtype: torch.dtype, with_bias: bool
) -> None:
    if not torch.cuda.is_available():  # pragma: no cover - GPU required
        pytest.skip("CUDA is required for CuTeDSL layernorm")

    device = torch.device("cuda")
    m, n = 4, 64
    eps = 1e-6

    x = torch.randn(m, n, device=device, dtype=dtype)
    weight = torch.randn(n, device=device, dtype=torch.float32)
    bias = torch.randn(n, device=device, dtype=torch.float32) if with_bias else None
    grad_out = torch.randn_like(x)

    # CuTeDSL forward/backward
    x_cute = x.detach().clone().requires_grad_(True)
    weight_cute = weight.detach().clone().requires_grad_(True)
    bias_cute = bias.detach().clone().requires_grad_(True) if bias is not None else None

    out_cute = cutedsl_layernorm(x_cute, weight_cute, bias=bias_cute, eps=eps)
    out_cute.backward(grad_out)

    # PyTorch reference using float32 accumulation
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True) if bias is not None else None

    out_ref = torch.nn.functional.layer_norm(
        x_ref.to(torch.float32),
        (n,),
        weight_ref,
        bias_ref,
        eps,
    ).to(dtype)
    out_ref.backward(grad_out)

    assert x_ref.grad is not None
    assert weight_ref.grad is not None
    if dtype is torch.bfloat16:
        atol = rtol = 6e-2
    else:
        atol = rtol = 1e-3

    torch.testing.assert_close(out_cute, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(x_cute.grad, x_ref.grad.to(dtype), atol=atol, rtol=rtol)
    weight_atol = 1e-2 if dtype is torch.bfloat16 else 1e-3
    torch.testing.assert_close(
        weight_cute.grad, weight_ref.grad, atol=weight_atol, rtol=weight_atol
    )
    if with_bias:
        assert bias_cute is not None and bias_ref is not None
        assert bias_ref.grad is not None
        torch.testing.assert_close(
            bias_cute.grad, bias_ref.grad, atol=weight_atol, rtol=weight_atol
        )
    else:
        assert bias_cute is None or bias_cute.grad is None
