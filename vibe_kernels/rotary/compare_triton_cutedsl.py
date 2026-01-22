# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from vibe_kernels.rotary import apply_rotary_embedding


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    B, H, S, D = 4, 8, 2048, 128
    torch.manual_seed(0)

    q = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    cos = torch.randn(S, D // 2, device=device, dtype=torch.float32)
    sin = torch.randn_like(cos)
    positions = (
        torch.arange(S, device=device, dtype=torch.int32).view(1, 1, S).expand(B, H, S)
    )

    q_tri, k_tri = apply_rotary_embedding(
        q.clone(), k.clone(), cos, sin, positions, backend="triton"
    )
    q_cut, k_cut = apply_rotary_embedding(
        q.clone(), k.clone(), cos, sin, positions, backend="cutedsl"
    )

    max_diff_q = (q_tri - q_cut).abs().max().item()
    max_diff_k = (k_tri - k_cut).abs().max().item()

    print("Triton vs CuTeDSL (float32)")
    print("============================")
    print(f"B={B}, H={H}, S={S}, D={D}")
    print(f"max_diff_q = {max_diff_q}")
    print(f"max_diff_k = {max_diff_k}")
    print(
        "allclose =",
        torch.allclose(q_tri, q_cut, atol=1e-6, rtol=0)
        and torch.allclose(k_tri, k_cut, atol=1e-6, rtol=0),
    )


if __name__ == "__main__":
    main()
