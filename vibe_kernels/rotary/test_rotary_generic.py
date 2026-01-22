# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from vibe_kernels.rotary import apply_rotary_embedding


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    # D=98 means pairs_per_head=49, which is NOT a power of 2.
    # This forces the generic kernel path.
    B, H, S, D = 2, 4, 128, 98
    torch.manual_seed(0)

    q = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    cos = torch.randn(S, D // 2, device=device, dtype=torch.float32)
    sin = torch.randn_like(cos)
    positions = (
        torch.arange(S, device=device, dtype=torch.int32).view(1, 1, S).expand(B, H, S)
    )

    print(f"Testing Generic Path with B={B}, H={H}, S={S}, D={D}")

    # Run Triton as reference
    try:
        q_tri, k_tri = apply_rotary_embedding(
            q.clone(), k.clone(), cos, sin, positions, backend="triton"
        )
    except Exception as e:
        print(f"Triton failed (might not support this shape?): {e}")
        # Fallback to Torch if Triton fails on odd shapes (though Triton usually handles it)
        q_tri, k_tri = apply_rotary_embedding(
            q.clone(), k.clone(), cos, sin, positions, backend="torch"
        )
        print("Using Torch as reference instead.")

    # Run CuTeDSL
    q_cut, k_cut = apply_rotary_embedding(
        q.clone(), k.clone(), cos, sin, positions, backend="cutedsl"
    )

    max_diff_q = (q_tri - q_cut).abs().max().item()
    max_diff_k = (k_tri - k_cut).abs().max().item()

    print(f"max_diff_q = {max_diff_q}")
    print(f"max_diff_k = {max_diff_k}")

    is_close = torch.allclose(q_tri, q_cut, atol=1e-5, rtol=1e-5) and torch.allclose(
        k_tri, k_cut, atol=1e-5, rtol=1e-5
    )

    print(f"allclose = {is_close}")

    if not is_close:
        print("MISMATCH DETECTED!")
        exit(1)


if __name__ == "__main__":
    main()
