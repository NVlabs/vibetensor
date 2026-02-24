#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive benchmark comparing:
1. Our Triton implementation (kernel_factory)
2. QuACK CuTe implementation
3. PyTorch baseline

Based on problem sizes from QuACK's benchmark.
"""

import argparse
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import torch

# Add paths
sys.path.insert(0, "/workspace/terry/nano-cursor/kernel_factory")
sys.path.insert(0, "/workspace/terry/nano-cursor/tmp/quack")


def benchmark_torch(
    x: torch.Tensor, k: int, warmup: int = 10, iters: int = 100
) -> float:
    """Benchmark PyTorch topk."""
    # Warmup
    for _ in range(warmup):
        vals, idx = torch.topk(x, k, dim=-1, largest=True, sorted=True)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        vals, idx = torch.topk(x, k, dim=-1, largest=True, sorted=True)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / iters * 1000
    return avg_ms


def benchmark_triton(
    x: torch.Tensor, k: int, warmup: int = 10, iters: int = 100, impl: str = "stream"
) -> Optional[float]:
    """Benchmark our Triton implementation."""
    try:
        os.environ["AIKF_TOPK_IMPL"] = impl

        # Reload to pick up env changes
        if "sampling.kernel" in sys.modules:
            del sys.modules["sampling.kernel"]
        from sampling.kernel import _select_topk

        # Warmup
        for _ in range(warmup):
            vals, idx = _select_topk(x, k)
            torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            vals, idx = _select_topk(x, k)
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_ms = (end - start) / iters * 1000
        return avg_ms
    except Exception as e:
        print(f"    Triton {impl} error: {e}")
        return None


def benchmark_quack(
    x: torch.Tensor, k: int, warmup: int = 10, iters: int = 100
) -> Optional[float]:
    """Benchmark QuACK CuTe implementation."""
    try:
        # Check constraints
        N = x.shape[1]
        if N != 2 ** int(math.log2(N)):
            # print(f"    QuACK requires N to be power of 2, got {N}")
            return None
        if k != 2 ** int(math.log2(k)):
            # print(f"    QuACK requires k to be power of 2, got {k}")
            return None
        if k > 128:
            # print(f"    QuACK requires k <= 128, got {k}")
            return None
        if N > 4096:
            # print(f"    QuACK requires N <= 4096, got {N}")
            return None

        # Try to import cutlass cuda module (may not be available)
        try:
            import cutlass

            # Check if cutlass has the cuda attribute
            if hasattr(cutlass, "cuda") and hasattr(
                cutlass.cuda, "initialize_cuda_context"
            ):
                cutlass.cuda.initialize_cuda_context()
            else:
                # Cutlass doesn't have cuda module in this version
                return None
        except Exception:
            return None

        from quack.topk import topk

        # Warmup
        for _ in range(warmup):
            vals, idx = topk(x, k)
            torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            vals, idx = topk(x, k)
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_ms = (end - start) / iters * 1000
        return avg_ms
    except Exception as e:
        # print(f"    QuACK error: {e}")
        return None


def calculate_bandwidth(
    M: int, N: int, k: int, time_ms: float, dtype_bytes: int = 2
) -> float:
    """Calculate memory bandwidth in GB/s."""
    # Read: M*N elements, Write: M*k values + M*k indices
    bytes_read = M * N * dtype_bytes
    bytes_write = M * k * dtype_bytes + M * k * 4  # indices are int32
    total_bytes = bytes_read + bytes_write
    return (total_bytes / 1e9) / (time_ms / 1000)


def run_single_benchmark(
    M: int,
    N: int,
    k: int,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 10,
    iters: int = 100,
    verbose: bool = True,
):
    """Run benchmark for a single configuration."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Config: M={M}, N={N}, k={k}, dtype={dtype}")
        print(f"{'='*80}")

    # Create input
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    dtype_bytes = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    results = {}

    # 1. PyTorch
    if verbose:
        print("\n1. PyTorch topk:")
    time_torch = benchmark_torch(x, k, warmup, iters)
    bw_torch = calculate_bandwidth(M, N, k, time_torch, dtype_bytes)
    results["torch"] = (time_torch, bw_torch)
    if verbose:
        print(f"   Time: {time_torch:.4f} ms")
        print(f"   Bandwidth: {bw_torch:.2f} GB/s")

    # 2. Triton (stream - default 2-stage)
    if verbose:
        print("\n2. Triton (stream - 2-stage):")
    time_triton_stream = benchmark_triton(x, k, warmup, iters, impl="stream")
    if time_triton_stream:
        bw_triton_stream = calculate_bandwidth(M, N, k, time_triton_stream, dtype_bytes)
        speedup_stream = time_torch / time_triton_stream
        results["triton_stream"] = (
            time_triton_stream,
            bw_triton_stream,
            speedup_stream,
        )
        if verbose:
            print(f"   Time: {time_triton_stream:.4f} ms")
            print(f"   Bandwidth: {bw_triton_stream:.2f} GB/s")
            print(f"   Speedup vs PyTorch: {speedup_stream:.2f}x")

    # 3. Triton (singlepass)
    if verbose:
        print("\n3. Triton (singlepass):")
    time_triton_single = benchmark_triton(x, k, warmup, iters, impl="singlepass")
    if time_triton_single:
        bw_triton_single = calculate_bandwidth(M, N, k, time_triton_single, dtype_bytes)
        speedup_single = time_torch / time_triton_single
        results["triton_singlepass"] = (
            time_triton_single,
            bw_triton_single,
            speedup_single,
        )
        if verbose:
            print(f"   Time: {time_triton_single:.4f} ms")
            print(f"   Bandwidth: {bw_triton_single:.2f} GB/s")
            print(f"   Speedup vs PyTorch: {speedup_single:.2f}x")

    # 4. QuACK
    if verbose:
        print("\n4. QuACK (CuTe):")
    time_quack = benchmark_quack(x, k, warmup, iters)
    if time_quack:
        bw_quack = calculate_bandwidth(M, N, k, time_quack, dtype_bytes)
        speedup_quack = time_torch / time_quack
        results["quack"] = (time_quack, bw_quack, speedup_quack)
        if verbose:
            print(f"   Time: {time_quack:.4f} ms")
            print(f"   Bandwidth: {bw_quack:.2f} GB/s")
            print(f"   Speedup vs PyTorch: {speedup_quack:.2f}x")

    return results


def run_sweep(M: int = 8192, dtype: torch.dtype = torch.bfloat16):
    """Run comprehensive sweep across problem sizes."""
    print(f"\n{'='*80}")
    print(f"Top-K Benchmark Sweep: M={M}, dtype={dtype}")
    print(f"{'='*80}")

    # Test configurations from QuACK benchmark
    configs = [
        # (N, k) - N and k must be power of 2 for QuACK
        (64, 8),
        (64, 16),
        (64, 32),
        (128, 8),
        (128, 16),
        (128, 32),
        (256, 8),
        (256, 16),
        (256, 32),
        (512, 8),
        (512, 16),
        (512, 32),
        (1024, 8),
        (1024, 16),
        (1024, 32),
        (1024, 64),
        (2048, 32),
        (2048, 64),
        (2048, 128),
        (4096, 32),
        (4096, 64),
        (4096, 128),
    ]

    # Also test sizes that don't fit QuACK constraints (only Triton/PyTorch)
    non_power2_configs = [
        (50000, 50),  # Typical LLM vocab size
        (32000, 40),  # Common vocab size
        (100, 10),  # Small test
    ]

    all_results = []

    print("\n" + "=" * 100)
    print(
        f"{'N':>6} {'k':>4} | {'PyTorch':>12} | {'Triton-2st':>12} | {'Triton-1st':>12} | {'QuACK':>12} | {'Best':>12}"
    )
    print(
        f"{'':>6} {'':>4} | {'(ms)':>12} | {'(ms) Spdup':>12} | {'(ms) Spdup':>12} | {'(ms) Spdup':>12} | {'':>12}"
    )
    print("=" * 100)

    # Power-of-2 configs (all implementations)
    for N, k in configs:
        results = run_single_benchmark(
            M, N, k, dtype, warmup=5, iters=50, verbose=False
        )

        torch_time = results["torch"][0]
        triton_stream = results.get("triton_stream", (None, None, None))
        triton_single = results.get("triton_singlepass", (None, None, None))
        quack = results.get("quack", (None, None, None))

        # Find best
        times = {
            "PyTorch": torch_time,
            "Triton-2st": triton_stream[0],
            "Triton-1st": triton_single[0],
            "QuACK": quack[0],
        }
        valid_times = {k: v for k, v in times.items() if v is not None}
        best = min(valid_times, key=valid_times.get) if valid_times else "N/A"

        print(f"{N:6d} {k:4d} | ", end="")
        print(f"{torch_time:8.4f} ms | ", end="")

        if triton_stream[0]:
            print(f"{triton_stream[0]:8.4f} {triton_stream[2]:>4.2f}x | ", end="")
        else:
            print(f"{'N/A':>14} | ", end="")

        if triton_single[0]:
            print(f"{triton_single[0]:8.4f} {triton_single[2]:>4.2f}x | ", end="")
        else:
            print(f"{'N/A':>14} | ", end="")

        if quack[0]:
            print(f"{quack[0]:8.4f} {quack[2]:>4.2f}x | ", end="")
        else:
            print(f"{'N/A':>14} | ", end="")

        print(f"{best:>12}")

        all_results.append((N, k, results))

    # Non-power-of-2 configs (PyTorch and Triton only)
    print("\n--- Non-power-of-2 sizes (QuACK not supported) ---")
    for N, k in non_power2_configs:
        results = run_single_benchmark(
            M, N, k, dtype, warmup=5, iters=50, verbose=False
        )

        torch_time = results["torch"][0]
        triton_stream = results.get("triton_stream", (None, None, None))
        triton_single = results.get("triton_singlepass", (None, None, None))

        # Find best
        times = {
            "PyTorch": torch_time,
            "Triton-2st": triton_stream[0],
            "Triton-1st": triton_single[0],
        }
        valid_times = {k: v for k, v in times.items() if v is not None}
        best = min(valid_times, key=valid_times.get) if valid_times else "N/A"

        print(f"{N:6d} {k:4d} | ", end="")
        print(f"{torch_time:8.4f} ms | ", end="")

        if triton_stream[0]:
            print(f"{triton_stream[0]:8.4f} {triton_stream[2]:>4.2f}x | ", end="")
        else:
            print(f"{'N/A':>14} | ", end="")

        if triton_single[0]:
            print(f"{triton_single[0]:8.4f} {triton_single[2]:>4.2f}x | ", end="")
        else:
            print(f"{'N/A':>14} | ", end="")

        print(f"{'N/A':>14} | ", end="")
        print(f"{best:>12}")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Benchmark top-k implementations")
    parser.add_argument("--M", type=int, default=8192, help="Batch size")
    parser.add_argument("--N", type=int, default=1024, help="Vocabulary size")
    parser.add_argument("--k", type=int, default=32, help="Top-k value")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--sweep", action="store_true", help="Run sweep across sizes")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    torch.manual_seed(42)

    if args.sweep:
        run_sweep(M=args.M, dtype=dtype)
    else:
        run_single_benchmark(args.M, args.N, args.k, dtype, args.warmup, args.iters)


if __name__ == "__main__":
    main()
