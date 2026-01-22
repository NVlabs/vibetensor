# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GEMM Kernel Benchmark: cuTile vs PyTorch

This script implements a tiled GEMM (General Matrix Multiply) kernel using cuTile
and benchmarks it against PyTorch's native matmul implementation.
"""

import math
import time

import cuda.tile as ct
import torch


@ct.kernel
def gemm_kernel(
    A,
    B,
    C,
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    """
    Tiled GEMM kernel: C = A @ B

    A has shape (M, K)
    B has shape (K, N)
    C has shape (M, N)
    """
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # Initialize accumulator in float32 for precision
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

    # Iterate over K dimension in tiles
    num_k_tiles = A.shape[1] // BLOCK_K
    for k in range(num_k_tiles):
        # Load tiles from A and B
        a_tile = ct.load(A, index=(bid_m, k), shape=(BLOCK_M, BLOCK_K))
        b_tile = ct.load(B, index=(k, bid_n), shape=(BLOCK_K, BLOCK_N))
        # Matrix multiply-accumulate
        acc = ct.mma(a_tile, b_tile, acc)

    # Convert to output dtype and store
    acc = ct.astype(acc, C.dtype)
    ct.store(C, index=(bid_m, bid_n), tile=acc)


def pytorch_gemm(A, B):
    """PyTorch reference implementation."""
    return torch.matmul(A, B)


def run_cutile_gemm(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K):
    """Launch cuTile GEMM kernel."""
    M, K = A.shape
    _, N = B.shape

    grid = (math.ceil(M / BLOCK_M), math.ceil(N / BLOCK_N))
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        gemm_kernel,
        (A, B, C, BLOCK_M, BLOCK_N, BLOCK_K),
    )
    return C


def validate(cutile_output, pytorch_output, rtol=1e-2, atol=1e-2):
    """Validate cuTile output against PyTorch reference."""
    try:
        torch.testing.assert_close(cutile_output, pytorch_output, rtol=rtol, atol=atol)
        return True, 0.0
    except AssertionError as e:
        max_diff = (cutile_output - pytorch_output).abs().max().item()
        return False, max_diff


def benchmark(func, warmup=10, iterations=100):
    """Benchmark a function with warmup and timing."""
    # Warmup
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        func()
        end_events[i].record()

    torch.cuda.synchronize()

    times = [start_events[i].elapsed_time(end_events[i]) for i in range(iterations)]
    return sum(times) / len(times)  # Return average time in ms


def run_benchmark(M, K, N, dtype=torch.float16):
    """Run GEMM benchmark for given dimensions."""
    print(f"\n{'='*60}")
    print(f"Matrix dimensions: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")
    print(f"Data type: {dtype}")
    print(f"{'='*60}")

    # Block sizes (must be powers of 2)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    # Ensure dimensions are divisible by block sizes
    M_padded = math.ceil(M / BLOCK_K) * BLOCK_K
    K_padded = math.ceil(K / BLOCK_K) * BLOCK_K
    N_padded = math.ceil(N / BLOCK_K) * BLOCK_K

    # Create input tensors
    A = torch.randn(M_padded, K_padded, dtype=dtype, device="cuda")
    B = torch.randn(K_padded, N_padded, dtype=dtype, device="cuda")
    C_cutile = torch.zeros(M_padded, N_padded, dtype=dtype, device="cuda")

    # Run cuTile GEMM
    run_cutile_gemm(A, B, C_cutile, BLOCK_M, BLOCK_N, BLOCK_K)

    # Run PyTorch GEMM
    C_pytorch = pytorch_gemm(A, B)

    # Validate correctness
    passed, max_diff = validate(C_cutile, C_pytorch)
    status = "PASSED" if passed else f"FAILED (max diff: {max_diff:.6f})"
    print(f"Correctness check: {status}")

    # Benchmark cuTile
    cutile_time = benchmark(
        lambda: run_cutile_gemm(A, B, C_cutile, BLOCK_M, BLOCK_N, BLOCK_K)
    )

    # Benchmark PyTorch
    pytorch_time = benchmark(lambda: pytorch_gemm(A, B))

    # Calculate TFLOPS
    flops = 2 * M_padded * N_padded * K_padded  # 2 ops per multiply-add
    cutile_tflops = flops / (cutile_time * 1e-3) / 1e12
    pytorch_tflops = flops / (pytorch_time * 1e-3) / 1e12

    print(f"\nPerformance Results:")
    print(f"  cuTile:  {cutile_time:.3f} ms  ({cutile_tflops:.2f} TFLOPS)")
    print(f"  PyTorch: {pytorch_time:.3f} ms  ({pytorch_tflops:.2f} TFLOPS)")
    print(
        f"  Speedup: {pytorch_time/cutile_time:.2f}x"
        if cutile_time > 0
        else "  Speedup: N/A"
    )

    return {
        "M": M_padded,
        "K": K_padded,
        "N": N_padded,
        "cutile_ms": cutile_time,
        "pytorch_ms": pytorch_time,
        "cutile_tflops": cutile_tflops,
        "pytorch_tflops": pytorch_tflops,
        "passed": passed,
    }


def main():
    print("=" * 60)
    print("GEMM Benchmark: cuTile vs PyTorch")
    print("=" * 60)

    # Print GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Test various matrix sizes
    test_cases = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (1024, 4096, 1024),  # Tall-skinny
        (4096, 1024, 4096),  # Wide
    ]

    results = []
    for M, K, N in test_cases:
        result = run_benchmark(M, K, N, dtype=torch.float16)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Size':<20} {'cuTile (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 60)
    for r in results:
        size_str = f"{r['M']}x{r['K']}x{r['N']}"
        speedup = r["pytorch_ms"] / r["cutile_ms"] if r["cutile_ms"] > 0 else 0
        print(
            f"{size_str:<20} {r['cutile_ms']:<15.3f} {r['pytorch_ms']:<15.3f} {speedup:<10.2f}x"
        )


if __name__ == "__main__":
    main()
