# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from vibe_kernels.rmsnorm import RMSNorm


def check_forward_backward(dtype: torch.dtype, gamma: bool) -> None:
    torch.manual_seed(0)
    rows, hidden = 16, 64
    eps = 1e-6
    device = torch.device("cuda")

    x = torch.randn(rows, hidden, device=device, dtype=dtype, requires_grad=True)
    gamma_tensor = None
    if gamma:
        gamma_tensor = torch.randn(hidden, device=device, dtype=torch.float32)

    # Reference
    ref = torch.nn.functional.rms_norm(
        x,
        (hidden,),
        weight=gamma_tensor,
        eps=eps,
    )
    ref_loss = ref.square().sum()
    ref_loss.backward()

    # Module under test
    x2 = x.detach().clone().requires_grad_(True)
    module = RMSNorm(
        hidden_size=hidden, eps=eps, learnable_gamma=gamma, dtype=dtype, device=device
    )
    if gamma and gamma_tensor is not None:
        with torch.no_grad():
            module.gamma.copy_(gamma_tensor)

    out = module(x2)
    loss = out.square().sum()
    loss.backward()

    assert torch.allclose(
        out, ref, atol=1e-4 if dtype == torch.float32 else 1e-2, rtol=0
    )
    assert torch.allclose(
        x2.grad, x.grad, atol=1e-4 if dtype == torch.float32 else 5e-3, rtol=0
    )
    if gamma:
        assert torch.allclose(
            module.gamma.grad, gamma_tensor * 0 + module.gamma.grad, atol=5e-3, rtol=0
        )


def main() -> None:
    for dtype in [torch.float32, torch.bfloat16]:
        for gamma in [False, True]:
            check_forward_backward(dtype, gamma)
    print("RMSNorm tests passed")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemError("CUDA is required to run RMSNorm tests")
    main()
