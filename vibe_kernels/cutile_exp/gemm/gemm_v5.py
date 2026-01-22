# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GEMM Benchmark v5: Using TileGym's exact configs without autotuning overhead.
"""

import math
from math import ceil

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


def swizzle_2d(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M):
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


# Pre-compiled kernels with different num_ctas settings
@ct.kernel(num_ctas=1)
def matmul_kernel_cta1(A, B, C, TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt):
    GROUP_SIZE_M = 8
    M, N = A.shape[0], B.shape[1]
    bidx, bidy = swizzle_2d(M, N, TILE_M, TILE_N, GROUP_SIZE_M)
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(TILE_M, TILE_K))
    acc = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    for k in range(num_tiles_k):
        a = ct.load(
            A, index=(bidx, k), shape=(TILE_M, TILE_K), padding_mode=zero_pad
        ).astype(dtype)
        b = ct.load(
            B, index=(k, bidy), shape=(TILE_K, TILE_N), padding_mode=zero_pad
        ).astype(dtype)
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(bidx, bidy), tile=ct.astype(acc, C.dtype))


@ct.kernel(num_ctas=2)
def matmul_kernel_cta2(A, B, C, TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt):
    GROUP_SIZE_M = 8
    M, N = A.shape[0], B.shape[1]
    bidx, bidy = swizzle_2d(M, N, TILE_M, TILE_N, GROUP_SIZE_M)
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(TILE_M, TILE_K))
    acc = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    for k in range(num_tiles_k):
        a = ct.load(
            A, index=(bidx, k), shape=(TILE_M, TILE_K), padding_mode=zero_pad
        ).astype(dtype)
        b = ct.load(
            B, index=(k, bidy), shape=(TILE_K, TILE_N), padding_mode=zero_pad
        ).astype(dtype)
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(bidx, bidy), tile=ct.astype(acc, C.dtype))


@ct.kernel(num_ctas=4)
def matmul_kernel_cta4(A, B, C, TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt):
    GROUP_SIZE_M = 8
    M, N = A.shape[0], B.shape[1]
    bidx, bidy = swizzle_2d(M, N, TILE_M, TILE_N, GROUP_SIZE_M)
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(TILE_M, TILE_K))
    acc = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    for k in range(num_tiles_k):
        a = ct.load(
            A, index=(bidx, k), shape=(TILE_M, TILE_K), padding_mode=zero_pad
        ).astype(dtype)
        b = ct.load(
            B, index=(k, bidy), shape=(TILE_K, TILE_N), padding_mode=zero_pad
        ).astype(dtype)
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(bidx, bidy), tile=ct.astype(acc, C.dtype))


def _compute_bid(tile_id, num_bid_in_group, num_bid_m, GROUP_SIZE_M):
    group_id = tile_id // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = ct.minimum(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (tile_id % group_size_m)
    bid_n = (tile_id % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@ct.kernel(num_ctas=2)
def persistent_kernel_cta2(
    A,
    B,
    C,
    M: int,
    N: int,
    K: int,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    GROUP_SIZE_M: ConstInt,
):
    start_bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, TILE_M)
    num_bid_n = ct.cdiv(N, TILE_N)
    k_tiles = ct.cdiv(K, TILE_K)
    num_tiles = num_bid_m * num_bid_n
    num_programs = ct.num_blocks(0)
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    zero_pad = ct.PaddingMode.ZERO

    for tile_id in range(start_bid, num_tiles, num_programs):
        num_bid_in_group = GROUP_SIZE_M * num_bid_n
        bid_m, bid_n = _compute_bid(tile_id, num_bid_in_group, num_bid_m, GROUP_SIZE_M)
        acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        for k_tile in range(k_tiles):
            a = ct.load(
                A, index=(bid_m, k_tile), shape=(TILE_M, TILE_K), padding_mode=zero_pad
            ).astype(dtype)
            b = ct.load(
                B, index=(k_tile, bid_n), shape=(TILE_K, TILE_N), padding_mode=zero_pad
            ).astype(dtype)
            acc = ct.mma(a, b, acc=acc)
        ct.store(C, index=(bid_m, bid_n), tile=ct.astype(acc, C.dtype))


@ct.kernel(num_ctas=4)
def persistent_kernel_cta4(
    A,
    B,
    C,
    M: int,
    N: int,
    K: int,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    GROUP_SIZE_M: ConstInt,
):
    start_bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, TILE_M)
    num_bid_n = ct.cdiv(N, TILE_N)
    k_tiles = ct.cdiv(K, TILE_K)
    num_tiles = num_bid_m * num_bid_n
    num_programs = ct.num_blocks(0)
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    zero_pad = ct.PaddingMode.ZERO

    for tile_id in range(start_bid, num_tiles, num_programs):
        num_bid_in_group = GROUP_SIZE_M * num_bid_n
        bid_m, bid_n = _compute_bid(tile_id, num_bid_in_group, num_bid_m, GROUP_SIZE_M)
        acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        for k_tile in range(k_tiles):
            a = ct.load(
                A, index=(bid_m, k_tile), shape=(TILE_M, TILE_K), padding_mode=zero_pad
            ).astype(dtype)
            b = ct.load(
                B, index=(k_tile, bid_n), shape=(TILE_K, TILE_N), padding_mode=zero_pad
            ).astype(dtype)
            acc = ct.mma(a, b, acc=acc)
        ct.store(C, index=(bid_m, bid_n), tile=ct.astype(acc, C.dtype))


def get_best_config(M, K, N):
    """Select best config based on TileGym's tuned values for sm100."""
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # Large square matrices - use persistent with 128x512x64 cta4 (TileGym's best)
    if M >= 4096 and N >= 4096:
        return 128, 512, 64, 8, 4, True
    # Medium-large
    elif M >= 2048 and N >= 2048:
        return 256, 256, 64, 8, 2, True
    # Medium
    elif M >= 1024 and N >= 1024:
        return 128, 128, 32, 8, 1, False
    # Small
    else:
        return 128, 128, 32, 8, 1, False


def run_gemm(A, B, C, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M, num_ctas, use_persistent):
    M, K = A.shape
    _, N = B.shape

    if use_persistent:
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_tiles = ceil(M / TILE_M) * ceil(N / TILE_N)
        grid = (min(NUM_SMS // num_ctas, num_tiles),)
        kernel = persistent_kernel_cta4 if num_ctas == 4 else persistent_kernel_cta2
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            kernel,
            (A, B, C, M, N, K, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M),
        )
    else:
        grid = (ceil(M / TILE_M) * ceil(N / TILE_N),)
        if num_ctas == 4:
            kernel = matmul_kernel_cta4
        elif num_ctas == 2:
            kernel = matmul_kernel_cta2
        else:
            kernel = matmul_kernel_cta1
        ct.launch(
            torch.cuda.current_stream(), grid, kernel, (A, B, C, TILE_M, TILE_N, TILE_K)
        )
    return C


def pytorch_gemm(A, B):
    return torch.matmul(A, B)


def validate(out1, out2, rtol=1e-2, atol=1e-2):
    try:
        torch.testing.assert_close(out1, out2, rtol=rtol, atol=atol)
        return True, 0.0
    except AssertionError:
        return False, (out1 - out2).abs().max().item()


def benchmark(func, warmup=10, iterations=100):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        starts[i].record()
        func()
        ends[i].record()

    torch.cuda.synchronize()
    return sum(starts[i].elapsed_time(ends[i]) for i in range(iterations)) / iterations


def run_benchmark(M, K, N, dtype=torch.float16):
    print(f"\n{'='*60}")
    print(f"Matrix: A({M}x{K}) @ B({K}x{N})")
    print(f"{'='*60}")

    TILE_M, TILE_N, TILE_K, GROUP_SIZE_M, num_ctas, use_persistent = get_best_config(
        M, K, N
    )
    print(
        f"Config: ({TILE_M}x{TILE_N}x{TILE_K}), ctas={num_ctas}, persistent={use_persistent}"
    )

    # Pad dimensions
    pad = max(TILE_M, TILE_N, TILE_K)
    M_pad = ceil(M / pad) * pad
    K_pad = ceil(K / pad) * pad
    N_pad = ceil(N / pad) * pad

    A = torch.randn(M_pad, K_pad, dtype=dtype, device="cuda")
    B = torch.randn(K_pad, N_pad, dtype=dtype, device="cuda")
    C = torch.zeros(M_pad, N_pad, dtype=dtype, device="cuda")

    # Run and validate
    run_gemm(A, B, C, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M, num_ctas, use_persistent)
    C_ref = pytorch_gemm(A, B)
    passed, diff = validate(C, C_ref)
    print(f"Correctness: {'PASSED' if passed else f'FAILED ({diff:.6f})'}")

    # Benchmark
    cutile_time = benchmark(
        lambda: run_gemm(
            A, B, C, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M, num_ctas, use_persistent
        )
    )
    pytorch_time = benchmark(lambda: pytorch_gemm(A, B))

    flops = 2 * M_pad * N_pad * K_pad
    cutile_tflops = flops / (cutile_time * 1e-3) / 1e12
    pytorch_tflops = flops / (pytorch_time * 1e-3) / 1e12
    speedup = pytorch_time / cutile_time

    print(f"cuTile:  {cutile_time:.3f} ms ({cutile_tflops:.2f} TFLOPS)")
    print(f"PyTorch: {pytorch_time:.3f} ms ({pytorch_tflops:.2f} TFLOPS)")
    print(f"vs PyTorch: {speedup:.2f}x")

    return {
        "M": M_pad,
        "K": K_pad,
        "N": N_pad,
        "cutile_ms": cutile_time,
        "pytorch_ms": pytorch_time,
        "cutile_tflops": cutile_tflops,
        "pytorch_tflops": pytorch_tflops,
        "speedup": speedup,
        "passed": passed,
    }


def main():
    print("=" * 60)
    print("GEMM Benchmark v5: TileGym-optimized configs")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"SM: {torch.cuda.get_device_capability()}, SMs: {torch.cuda.get_device_properties('cuda').multi_processor_count}"
    )

    results = []
    for M, K, N in [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (1024, 4096, 1024),
        (4096, 1024, 4096),
        (8192, 8192, 8192),
    ]:
        results.append(run_benchmark(M, K, N))

    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    print(
        f"{'Size':<20} {'cuTile (ms)':<12} {'cuTile TFLOPS':<14} {'PyTorch TFLOPS':<15} {'vs PyTorch':<12}"
    )
    print("-" * 90)
    for r in results:
        size = f"{r['M']}x{r['K']}x{r['N']}"
        print(
            f"{size:<20} {r['cutile_ms']:<12.3f} {r['cutile_tflops']:<14.2f} {r['pytorch_tflops']:<15.2f} {r['speedup']:<12.2f}x"
        )


if __name__ == "__main__":
    main()
