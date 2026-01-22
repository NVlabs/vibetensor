# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import torch

from vibe_kernels.rotary import apply_rotary_embedding


CONFIGS = [
    ("A", 1, 8, 2048, 128),
    ("B", 4, 8, 2048, 128),
    ("C", 4, 16, 2048, 128),
]


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


def _time_fn(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    dtype = torch.bfloat16

    print("Rotary QK CuTeDSL vs Torch (bf16)")
    print("=================================")

    for name, B, H, S, D in CONFIGS:
        torch.manual_seed(0)

        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn_like(q)
        cos = torch.randn(S, D // 2, device=device, dtype=dtype)
        sin = torch.randn_like(cos)
        positions = (
            torch.arange(S, device=device, dtype=torch.int32)
            .view(1, 1, S)
            .expand(B, H, S)
        )

        def torch_fn():
            return _torch_rotary(q, k, cos, sin, positions)

        def cutedsl_fn():
            return apply_rotary_embedding(q, k, cos, sin, positions, backend="cutedsl")

        torch_ms = _time_fn(torch_fn)
        cut_ms = _time_fn(cutedsl_fn)

        ref_q, ref_k = torch_fn()
        cut_q, cut_k = cutedsl_fn()

        max_diff_q = (ref_q - cut_q).abs().max().item()
        max_diff_k = (ref_k - cut_k).abs().max().item()

        print(f"Config {name}: B={B}, H={H}, S={S}, D={D}")
        print(f"  torch_mean_ms   = {torch_ms:.4f}")
        print(f"  cutedsl_mean_ms = {cut_ms:.4f}")
        print(f"  cutedsl_speedup = {torch_ms / cut_ms:.3f}x vs torch")
        print(f"  max_diff_q      = {max_diff_q}")
        print(f"  max_diff_k      = {max_diff_k}")
        print()


if __name__ == "__main__":
    main()
