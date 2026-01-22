# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GEMM Kernel Benchmark: cuTile vs PyTorch

This script implements an optimized tiled GEMM kernel using cuTile
with swizzle pattern, larger tiles, and multi-CTA support.
"""

import math
import time

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


def swizzle_2d(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M):
    """2D swizzle pattern for better L2 cache locality."""
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def gemm_kernel_optimized(
    A,
    B,
    C,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    TILE_SIZE_K: ConstInt,
):
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M)

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(TILE_SIZE_M, TILE_SIZE_K))
    accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for k in range(num_tiles_k):
        a = ct.load(
            A, index=(bidx, k), shape=(TILE_SIZE_M, TILE_SIZE_K), padding_mode=zero_pad
        ).astype(dtype)
        b = ct.load(
            B, index=(k, bidy), shape=(TILE_SIZE_K, TILE_SIZE_N), padding_mode=zero_pad
        ).astype(dtype)
        accumulator = ct.mma(a, b, accumulator)

    accumulator = ct.astype(accumulator, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=accumulator)


def _compute_bid(tile_id, num_bid_in_group, num_bid_m, GROUP_SIZE_M):
    group_id = tile_id // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = ct.minimum(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (tile_id % group_size_m)
    bid_n = (tile_id % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@ct.kernel(num_ctas=2)
def persistent_gemm_kernel(
    A,
    B,
    C,
    M: int,
    N: int,
    K: int,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    TILE_SIZE_K: ConstInt,
    GROUP_SIZE_M: ConstInt,
):
    start_bid = ct.bid(0)

    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    k_tiles = ct.cdiv(K, TILE_SIZE_K)
    num_tiles = num_bid_m * num_bid_n
    num_programs = ct.num_blocks(0)

    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    for tile_id in range(start_bid, num_tiles, num_programs):
        num_bid_in_group = GROUP_SIZE_M * num_bid_n
        bid_m, bid_n = _compute_bid(tile_id, num_bid_in_group, num_bid_m, GROUP_SIZE_M)

        accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)

        for k_tile in range(k_tiles):
            a = ct.load(A, index=(bid_m, k_tile), shape=(TILE_SIZE_M, TILE_SIZE_K))
            b = ct.load(B, index=(k_tile, bid_n), shape=(TILE_SIZE_K, TILE_SIZE_N))
            a = ct.astype(a, dtype)
            b = ct.astype(b, dtype)
            accumulator = ct.mma(a, b, acc=accumulator)

        result = ct.astype(accumulator, C.dtype)
        ct.store(C, index=(bid_m, bid_n), tile=result)


def pytorch_gemm(A, B):
    return torch.matmul(A, B)


def get_optimal_tile_config(M, K, N):
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        if max(M, N, K) >= 4096:
            return 128, 64, 64, 8, 2, True
        else:
            return 128, 64, 64, 8, 1, False
    else:
        if max(M, N, K) >= 4096:
            return 256, 256, 64, 8, 1, True
        elif max(M, N, K) >= 2048:
            return 128, 128, 32, 8, 1, False
        else:
            return 128, 128, 32, 8, 1, False


def run_cutile_gemm(
    A, B, C, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M=8, use_persistent=False
):
    M, K = A.shape
    _, N = B.shape

    if use_persistent:
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_tiles = math.ceil(M / TILE_M) * math.ceil(N / TILE_N)
        grid = (min(NUM_SMS, num_tiles), 1, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            persistent_gemm_kernel,
            (A, B, C, M, N, K, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M),
        )
    else:
        grid = (math.ceil(M / TILE_M) * math.ceil(N / TILE_N),)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            gemm_kernel_optimized,
            (A, B, C, TILE_M, TILE_N, TILE_K),
        )
    return C


def validate(cutile_output, pytorch_output, rtol=1e-2, atol=1e-2):
    try:
        torch.testing.assert_close(cutile_output, pytorch_output, rtol=rtol, atol=atol)
        return True, 0.0
    except AssertionError as e:
        max_diff = (cutile_output - pytorch_output).abs().max().item()
        return False, max_diff


def benchmark(func, warmup=10, iterations=100):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        func()
        end_events[i].record()

    torch.cuda.synchronize()

    times = [start_events[i].elapsed_time(end_events[i]) for i in range(iterations)]
    return sum(times) / len(times)


def run_benchmark(M, K, N, dtype=torch.float16):
    print(f"\n{'='*60}")
    print(f"Matrix dimensions: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")
    print(f"Data type: {dtype}")
    print(f"{'='*60}")

    TILE_M, TILE_N, TILE_K, GROUP_SIZE_M, occupancy, use_persistent = (
        get_optimal_tile_config(M, K, N)
    )
    print(
        f"Config: TILE=({TILE_M}x{TILE_N}x{TILE_K}), GROUP_SIZE_M={GROUP_SIZE_M}, persistent={use_persistent}"
    )

    M_padded = math.ceil(M / TILE_K) * TILE_K
    K_padded = math.ceil(K / TILE_K) * TILE_K
    N_padded = math.ceil(N / TILE_K) * TILE_K

    A = torch.randn(M_padded, K_padded, dtype=dtype, device="cuda")
    B = torch.randn(K_padded, N_padded, dtype=dtype, device="cuda")
    C_cutile = torch.zeros(M_padded, N_padded, dtype=dtype, device="cuda")

    run_cutile_gemm(
        A, B, C_cutile, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M, use_persistent
    )
    C_pytorch = pytorch_gemm(A, B)

    passed, max_diff = validate(C_cutile, C_pytorch)
    status = "PASSED" if passed else f"FAILED (max diff: {max_diff:.6f})"
    print(f"Correctness check: {status}")

    cutile_time = benchmark(
        lambda: run_cutile_gemm(
            A, B, C_cutile, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M, use_persistent
        )
    )

    pytorch_time = benchmark(lambda: pytorch_gemm(A, B))

    flops = 2 * M_padded * N_padded * K_padded
    cutile_tflops = flops / (cutile_time * 1e-3) / 1e12
    pytorch_tflops = flops / (pytorch_time * 1e-3) / 1e12

    speedup_vs_pytorch = pytorch_time / cutile_time if cutile_time > 0 else 0

    print(f"\nPerformance Results:")
    print(f"  cuTile:  {cutile_time:.3f} ms  ({cutile_tflops:.2f} TFLOPS)")
    print(f"  PyTorch: {pytorch_time:.3f} ms  ({pytorch_tflops:.2f} TFLOPS)")
    print(f"  vs PyTorch: {speedup_vs_pytorch:.2f}x")

    return {
        "M": M_padded,
        "K": K_padded,
        "N": N_padded,
        "cutile_ms": cutile_time,
        "pytorch_ms": pytorch_time,
        "cutile_tflops": cutile_tflops,
        "pytorch_tflops": pytorch_tflops,
        "speedup_vs_pytorch": speedup_vs_pytorch,
        "passed": passed,
    }


def main():
    print("=" * 60)
    print("GEMM Benchmark: cuTile vs PyTorch (Optimized)")
    print("=" * 60)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Capability: {torch.cuda.get_device_capability()}")
    print(f"SM Count: {torch.cuda.get_device_properties('cuda').multi_processor_count}")

    test_cases = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (1024, 4096, 1024),
        (4096, 1024, 4096),
    ]

    results = []
    for M, K, N in test_cases:
        result = run_benchmark(M, K, N, dtype=torch.float16)
        results.append(result)

    print("\n" + "=" * 95)
    print("Summary")
    print("=" * 95)
    print(
        f"{'Size':<20} {'cuTile (ms)':<12} {'cuTile TFLOPS':<14} {'PyTorch (ms)':<13} {'PyTorch TFLOPS':<15} {'vs PyTorch':<12}"
    )
    print("-" * 95)
    for r in results:
        size_str = f"{r['M']}x{r['K']}x{r['N']}"
        print(
            f"{size_str:<20} {r['cutile_ms']:<12.3f} {r['cutile_tflops']:<14.2f} {r['pytorch_ms']:<13.3f} {r['pytorch_tflops']:<15.2f} {r['speedup_vs_pytorch']:<12.2f}x"
        )


if __name__ == "__main__":
    main()
