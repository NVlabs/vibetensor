#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive benchmark: CuTe (ours) vs QuACK (original) vs Triton (ours) vs PyTorch

Usage:
    python benchmark_final.py
"""

import os
import sys
import time
from typing import Optional, Tuple

import torch

sys.path.insert(0, "/workspace/terry/nano-cursor/tmp/quack")


def benchmark_fn(fn, warmup=10, iters=100):
    """Benchmark a function."""
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / iters * 1000


def benchmark_all(
    M: int, N: int, k: int, dtype=torch.bfloat16, warmup=10, iters=100, verbose=True
):
    """Benchmark all implementations."""

    if verbose:
        print(f"\n{'='*100}")
        print(f"Config: M={M}, N={N}, k={k}, dtype={dtype}")
        print(f"{'='*100}")

    torch.manual_seed(42)
    x = torch.randn(M, N, device="cuda", dtype=dtype)

    results = {}

    # 1. PyTorch
    if verbose:
        print("\n1. PyTorch torch.topk")
    time_torch = benchmark_fn(lambda: torch.topk(x, k, dim=-1), warmup, iters)
    results["pytorch"] = time_torch
    if verbose:
        print(f"   Time: {time_torch:.4f} ms")

    # 2. Our CuTe
    try:
        from cute_topk import topk as cute_topk_ours

        if verbose:
            print("\n2. CuTe (our standalone)")
        time_cute_ours = benchmark_fn(lambda: cute_topk_ours(x, k), warmup, iters)
        results["cute_ours"] = time_cute_ours
        if verbose:
            print(f"   Time: {time_cute_ours:.4f} ms")
            print(f"   Speedup vs PyTorch: {time_torch / time_cute_ours:.2f}x")
    except Exception as e:
        if verbose:
            print(f"\n2. CuTe (our standalone): Error - {e}")
        results["cute_ours"] = None

    # 3. QuACK original
    try:
        from quack.topk import topk as quack_topk

        if verbose:
            print("\n3. QuACK (original)")
        time_quack = benchmark_fn(lambda: quack_topk(x, k), warmup, iters)
        results["quack"] = time_quack
        if verbose:
            print(f"   Time: {time_quack:.4f} ms")
            print(f"   Speedup vs PyTorch: {time_torch / time_quack:.2f}x")
    except Exception as e:
        if verbose:
            print(f"\n3. QuACK (original): Not available")
        results["quack"] = None

    # 4. Triton (stream - 2-stage)
    try:
        os.environ["AIKF_TOPK_IMPL"] = "stream"
        if "sampling.kernel" in sys.modules:
            del sys.modules["sampling.kernel"]
        from sampling.kernel import _select_topk

        if verbose:
            print("\n4. Triton (stream - 2-stage)")
        time_triton = benchmark_fn(lambda: _select_topk(x, k), warmup, iters)
        results["triton_stream"] = time_triton
        if verbose:
            print(f"   Time: {time_triton:.4f} ms")
            print(f"   Speedup vs PyTorch: {time_torch / time_triton:.2f}x")
    except Exception as e:
        if verbose:
            print(f"\n4. Triton (stream): Error - {e}")
        results["triton_stream"] = None

    return results


def run_sweep():
    """Run comprehensive sweep."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE TOP-K BENCHMARK")
    print("Comparing: CuTe (ours) vs QuACK (original) vs Triton (ours) vs PyTorch")
    print("=" * 100)

    configs = [
        # (M, N, k) - Power-of-2 sizes for CuTe/QuACK
        (4096, 64, 8),
        (4096, 64, 16),
        (4096, 64, 32),
        (4096, 128, 8),
        (4096, 128, 16),
        (4096, 128, 32),
        (4096, 256, 8),
        (4096, 256, 16),
        (4096, 256, 32),
        (4096, 512, 8),
        (4096, 512, 16),
        (4096, 512, 32),
        (4096, 1024, 8),
        (4096, 1024, 16),
        (4096, 1024, 32),
        (4096, 1024, 64),
        (4096, 2048, 32),
        (4096, 2048, 64),
        (4096, 2048, 128),
        (4096, 4096, 32),
        (4096, 4096, 64),
        (4096, 4096, 128),
    ]

    all_results = []

    print("\n" + "=" * 100)
    print(
        f"{'M':>5} {'N':>5} {'k':>4} | {'PyTorch':>10} | {'CuTe(ours)':>15} | {'QuACK':>15} | {'Triton':>15} | {'Best':>10}"
    )
    print(
        f"{'':>5} {'':>5} {'':>4} | {'(ms)':>10} | {'(ms) Spdup':>15} | {'(ms) Spdup':>15} | {'(ms) Spdup':>15} | {'':>10}"
    )
    print("=" * 100)

    for M, N, k in configs:
        results = benchmark_all(M, N, k, warmup=5, iters=50, verbose=False)

        pytorch = results["pytorch"]
        cute_ours = results.get("cute_ours")
        quack = results.get("quack")
        triton = results.get("triton_stream")

        # Find best (excluding None values)
        times = {
            "PyTorch": pytorch,
            "CuTe(ours)": cute_ours,
            "QuACK": quack,
            "Triton": triton,
        }
        valid = {k: v for k, v in times.items() if v is not None}
        best = min(valid, key=valid.get) if valid else "N/A"

        print(f"{M:5d} {N:5d} {k:4d} | ", end="")
        print(f"{pytorch:8.4f} | ", end="")

        if cute_ours:
            print(f"{cute_ours:8.4f} {pytorch/cute_ours:5.2f}x | ", end="")
        else:
            print(f"{'N/A':>15} | ", end="")

        if quack:
            print(f"{quack:8.4f} {pytorch/quack:5.2f}x | ", end="")
        else:
            print(f"{'N/A':>15} | ", end="")

        if triton:
            print(f"{triton:8.4f} {pytorch/triton:5.2f}x | ", end="")
        else:
            print(f"{'N/A':>15} | ", end="")

        print(f"{best:>10}")

        all_results.append((M, N, k, results))

    print("=" * 100)

    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    cute_speedups = []
    quack_speedups = []
    triton_speedups = []

    for M, N, k, results in all_results:
        pytorch = results["pytorch"]
        if results.get("cute_ours"):
            cute_speedups.append(pytorch / results["cute_ours"])
        if results.get("quack"):
            quack_speedups.append(pytorch / results["quack"])
        if results.get("triton_stream"):
            triton_speedups.append(pytorch / results["triton_stream"])

    if cute_speedups:
        print(f"\nCuTe (ours) vs PyTorch:")
        print(f"  Min speedup:  {min(cute_speedups):.2f}x")
        print(f"  Max speedup:  {max(cute_speedups):.2f}x")
        print(f"  Mean speedup: {sum(cute_speedups)/len(cute_speedups):.2f}x")

    if quack_speedups:
        print(f"\nQuACK (original) vs PyTorch:")
        print(f"  Min speedup:  {min(quack_speedups):.2f}x")
        print(f"  Max speedup:  {max(quack_speedups):.2f}x")
        print(f"  Mean speedup: {sum(quack_speedups)/len(quack_speedups):.2f}x")

    if triton_speedups:
        print(f"\nTriton (stream) vs PyTorch:")
        print(f"  Min speedup:  {min(triton_speedups):.2f}x")
        print(f"  Max speedup:  {max(triton_speedups):.2f}x")
        print(f"  Mean speedup: {sum(triton_speedups)/len(triton_speedups):.2f}x")

    if cute_speedups and quack_speedups:
        print(f"\nCuTe (ours) vs QuACK (original):")
        cute_vs_quack = [c / q for c, q in zip(cute_speedups, quack_speedups)]
        print(f"  Min ratio:  {min(cute_vs_quack):.2f}x")
        print(f"  Max ratio:  {max(cute_vs_quack):.2f}x")
        print(f"  Mean ratio: {sum(cute_vs_quack)/len(cute_vs_quack):.2f}x")
        if sum(cute_vs_quack) / len(cute_vs_quack) > 1:
            print(f"  → Our CuTe is faster than QuACK on average!")
        else:
            print(f"  → QuACK is faster than our CuTe on average")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    run_sweep()
