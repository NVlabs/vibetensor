# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from vibe_kernels.rotary import apply_rotary_embedding


def _torch_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
):
    head_dim_half = q.size(-1) // 2
    cos_gather = cos.reshape(-1, cos.shape[-1])[positions].to(torch.float32)
    sin_gather = sin.reshape(-1, cos.shape[-1])[positions].to(torch.float32)

    q1 = q[..., :head_dim_half].to(torch.float32)
    q2 = q[..., head_dim_half:].to(torch.float32)
    k1 = k[..., :head_dim_half].to(torch.float32)
    k2 = k[..., head_dim_half:].to(torch.float32)

    q_out1 = q1 * cos_gather + q2 * sin_gather
    q_out2 = -q1 * sin_gather + q2 * cos_gather
    k_out1 = k1 * cos_gather + k2 * sin_gather
    k_out2 = -k1 * sin_gather + k2 * cos_gather

    q_out = torch.cat([q_out1, q_out2], dim=-1).to(q.dtype)
    k_out = torch.cat([k_out1, k_out2], dim=-1).to(k.dtype)
    return q_out, k_out


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

    ref_q, ref_k = _torch_rotary(q, k, cos, sin, positions)
    tri_q, tri_k = apply_rotary_embedding(q, k, cos, sin, positions, backend="triton")

    max_diff_q = (ref_q - tri_q).abs().max().item()
    max_diff_k = (ref_k - tri_k).abs().max().item()

    print("Triton vs Torch (float32)")
    print("==========================")
    print(f"B={B}, H={H}, S={S}, D={D}")
    print(f"max_diff_q = {max_diff_q}")
    print(f"max_diff_k = {max_diff_k}")
    print(
        "allclose =",
        torch.allclose(ref_q, tri_q, atol=1e-6, rtol=0)
        and torch.allclose(ref_k, tri_k, atol=1e-6, rtol=0),
    )


if __name__ == "__main__":
    main()
