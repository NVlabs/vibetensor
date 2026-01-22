# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import time

import torch

from .kernel import apply_rotary_embedding


def benchmark_isolation():
    batch, heads, seqlen, head_dim = 32, 32, 4096, 128
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Benchmarking Isolation: B={batch}, H={heads}, S={seqlen}, D={head_dim}")

    q = torch.randn(batch, heads, seqlen, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    cos = torch.randn(seqlen, head_dim // 2, device=device, dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    positions = torch.arange(seqlen, device=device, dtype=torch.int32)
    positions_3d = positions.unsqueeze(0).unsqueeze(0).expand(batch, heads, seqlen)

    # 1. Triton Baseline
    def _triton():
        apply_rotary_embedding(q, k, cos, sin, positions_3d, backend="triton")

    # 2. Torch Baseline
    def _torch():
        apply_rotary_embedding(q, k, cos, sin, positions_3d, backend="torch")

    # 3. Full CuTeDSL (End-to-End)
    def _cutedsl_full():
        apply_rotary_embedding(q, k, cos, sin, positions_3d, backend="cutedsl")

    # Warmup compile
    try:
        apply_rotary_embedding(q, k, cos, sin, positions_3d, backend="cutedsl")
    except Exception as e:
        print(f"Compilation failed: {e}")
        return

    # Timing Helper
    def bench(fn, name, iters=20):
        torch.cuda.synchronize()
        # warmup
        for _ in range(5):
            fn()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000 / iters
        print(f"{name}: {ms:.4f} ms")
        return ms

    try:
        triton_ms = bench(_triton, "Triton Baseline")
        torch_ms = bench(_torch, "Torch Baseline")
        full_ms = bench(_cutedsl_full, "CuTeDSL Full")

        print(f"\nAnalysis:")
        print(
            f"CuTeDSL vs Triton: {full_ms:.4f} vs {triton_ms:.4f} (Gap: {full_ms/triton_ms:.2f}x)"
        )
        print(
            f"CuTeDSL vs Torch:  {full_ms:.4f} vs {torch_ms:.4f} (Speedup: {torch_ms/full_ms:.2f}x)"
        )

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    benchmark_isolation()
