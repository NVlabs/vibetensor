#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark for gather and scatter_add Triton kernels."""

import argparse
import time
from typing import Callable

import torch

from vibe_kernels.indexing import gather, scatter_add


def benchmark_fn(
    fn: Callable,
    *args,
    warmup: int = 10,
    iterations: int = 100,
    **kwargs,
) -> float:
    """Benchmark a function and return mean time in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000  # ms


def benchmark_gather(
    batch_sizes: list[int],
    gather_sizes: list[int],
    inner_sizes: list[int],
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    iterations: int = 100,
):
    """Benchmark gather operation."""
    print(f"\n{'='*60}")
    print(f"Gather Benchmark (dtype={dtype})")
    print(f"{'='*60}")
    print(f"{'Src Shape':<20} {'Idx Size':<10} {'Triton (ms)':<12} {'PyTorch (ms)':<12} {'Speedup':<10}")
    print("-" * 60)
    
    device = torch.device("cuda")
    
    for batch in batch_sizes:
        for inner in inner_sizes:
            for idx_size in gather_sizes:
                src = torch.randn(batch, inner, dtype=dtype, device=device)
                idx = torch.randint(0, batch, (idx_size,), dtype=torch.int64, device=device)
                
                # Triton
                triton_time = benchmark_fn(
                    gather, src, 0, idx,
                    warmup=warmup, iterations=iterations,
                )
                
                # PyTorch reference
                def torch_gather():
                    return src[idx]
                
                torch_time = benchmark_fn(
                    torch_gather,
                    warmup=warmup, iterations=iterations,
                )
                
                speedup = torch_time / triton_time if triton_time > 0 else float('inf')
                
                print(f"({batch}, {inner}){'':<10} {idx_size:<10} {triton_time:<12.4f} {torch_time:<12.4f} {speedup:<10.2f}x")


def benchmark_scatter_add(
    out_sizes: list[int],
    scatter_sizes: list[int],
    inner_sizes: list[int],
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    iterations: int = 100,
):
    """Benchmark scatter_add operation."""
    print(f"\n{'='*60}")
    print(f"Scatter Add Benchmark (dtype={dtype})")
    print(f"{'='*60}")
    print(f"{'Out Shape':<20} {'Src Size':<10} {'Triton (ms)':<12} {'PyTorch (ms)':<12} {'Speedup':<10}")
    print("-" * 60)
    
    device = torch.device("cuda")
    
    for out_dim in out_sizes:
        for inner in inner_sizes:
            for src_size in scatter_sizes:
                # Random indices with potential duplicates
                idx = torch.randint(0, out_dim, (src_size,), dtype=torch.int64, device=device)
                src = torch.randn(src_size, inner, dtype=dtype, device=device)
                
                # Triton
                def triton_scatter():
                    out = torch.zeros(out_dim, inner, dtype=dtype, device=device)
                    scatter_add(out, 0, idx, src)
                    return out
                
                triton_time = benchmark_fn(
                    triton_scatter,
                    warmup=warmup, iterations=iterations,
                )
                
                # PyTorch reference
                def torch_scatter():
                    out = torch.zeros(out_dim, inner, dtype=dtype, device=device)
                    out.scatter_add_(0, idx.view(-1, 1).expand(-1, inner), src)
                    return out
                
                torch_time = benchmark_fn(
                    torch_scatter,
                    warmup=warmup, iterations=iterations,
                )
                
                speedup = torch_time / triton_time if triton_time > 0 else float('inf')
                
                print(f"({out_dim}, {inner}){'':<10} {src_size:<10} {triton_time:<12.4f} {torch_time:<12.4f} {speedup:<10.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark gather and scatter_add kernels")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    args = parser.parse_args()
    
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    print(f"Running benchmarks with {args.iterations} iterations, {args.warmup} warmup")
    
    # Gather benchmarks
    benchmark_gather(
        batch_sizes=[1024, 4096, 16384],
        gather_sizes=[128, 512, 2048],
        inner_sizes=[256, 1024],
        dtype=dtype,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    
    # Scatter add benchmarks
    benchmark_scatter_add(
        out_sizes=[1024, 4096, 16384],
        scatter_sizes=[128, 512, 2048],
        inner_sizes=[256, 1024],
        dtype=dtype,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
