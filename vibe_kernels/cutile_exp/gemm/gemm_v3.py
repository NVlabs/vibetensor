# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GEMM Kernel Benchmark: cuTile vs PyTorch (v3)

Key optimizations:
- Smart tile selection based on matrix shape
- Non-persistent for small problems, persistent for large
- Proper occupancy tuning
- Better GROUP_SIZE selection based on M/N ratio
"""

import math

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


@ct.kernel(num_ctas=1)
def gemm_kernel(
    A,
    B,
    C,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    TILE_SIZE_K: ConstInt,
    GROUP_SIZE_M: ConstInt,
):
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

    ct.store(C, index=(bidx, bidy), tile=ct.astype(accumulator, C.dtype))


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
    zero_pad = ct.PaddingMode.ZERO

    for tile_id in range(start_bid, num_tiles, num_programs):
        num_bid_in_group = GROUP_SIZE_M * num_bid_n
        bid_m, bid_n = _compute_bid(tile_id, num_bid_in_group, num_bid_m, GROUP_SIZE_M)
        accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)

        for k_tile in range(k_tiles):
            a = ct.load(
                A,
                index=(bid_m, k_tile),
                shape=(TILE_SIZE_M, TILE_SIZE_K),
                padding_mode=zero_pad,
            ).astype(dtype)
            b = ct.load(
                B,
                index=(k_tile, bid_n),
                shape=(TILE_SIZE_K, TILE_SIZE_N),
                padding_mode=zero_pad,
            ).astype(dtype)
            accumulator = ct.mma(a, b, acc=accumulator)

        ct.store(C, index=(bid_m, bid_n), tile=ct.astype(accumulator, C.dtype))


def get_optimal_config(M, K, N):
    """
    Select optimal tile configuration based on matrix dimensions.
    Returns: (TILE_M, TILE_N, TILE_K, GROUP_SIZE_M, use_persistent)
    """
    gpu_capability = torch.cuda.get_device_capability()
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # Calculate problem characteristics
    total_tiles_128 = (M // 128) * (N // 128)
    total_tiles_256 = (M // 256) * (N // 256) if M >= 256 and N >= 256 else 0

    # Use persistent when we have enough tiles to saturate SMs
    use_persistent = total_tiles_128 >= NUM_SMS * 2

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
        if M >= 4096 and N >= 4096:
            return 128, 64, 64, 8, True
        elif M >= 2048 or N >= 2048:
            return 128, 64, 64, 8, use_persistent
        else:
            return 128, 64, 64, 8, False
    else:
        # sm100 (Blackwell) - optimized configs from tuning
        # Case: Wide matrix (small K, large M and N) - e.g., 4096x1024x4096
        # Best: 256x256x32, G=8, non-persistent -> 0.79x
        if K <= 1024 and M >= 4096 and N >= 4096:
            return 256, 256, 32, 8, False
        # Case: Large square matrices (4096+)
        # Best: 256x256x64, G=8, non-persistent -> 0.95x
        elif M >= 4096 and N >= 4096:
            return 256, 256, 64, 8, False
        # Case: 2048x2048
        # Best: 128x256x64, non-persistent -> 1.37x
        elif M >= 2048 and N >= 2048:
            return 128, 256, 64, 8, False
        # Case: Medium matrices
        elif M >= 1024 and N >= 1024:
            return 128, 128, 32, 8, False
        else:
            return 128, 128, 32, 8, False


def run_cutile_gemm(A, B, C, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M, use_persistent):
    """Launch cuTile GEMM kernel."""
    M, K = A.shape
    _, N = B.shape

    if use_persistent:
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_tiles = math.ceil(M / TILE_M) * math.ceil(N / TILE_N)
        grid = (min(NUM_SMS, num_tiles),)
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
            gemm_kernel,
            (A, B, C, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M),
        )
    return C


def pytorch_gemm(A, B):
    return torch.matmul(A, B)


def validate(cutile_output, pytorch_output, rtol=1e-2, atol=1e-2):
    try:
        torch.testing.assert_close(cutile_output, pytorch_output, rtol=rtol, atol=atol)
        return True, 0.0
    except AssertionError:
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
    return (
        sum(start_events[i].elapsed_time(end_events[i]) for i in range(iterations))
        / iterations
    )


def run_benchmark(M, K, N, dtype=torch.float16):
    """Run GEMM benchmark for given dimensions."""
    print(f"\n{'='*60}")
    print(f"Matrix dimensions: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")
    print(f"{'='*60}")

    TILE_M, TILE_N, TILE_K, GROUP_SIZE_M, use_persistent = get_optimal_config(M, K, N)

    # Pad dimensions to tile size
    pad = max(TILE_M, TILE_N, TILE_K)
    M_padded = math.ceil(M / pad) * pad
    K_padded = math.ceil(K / pad) * pad
    N_padded = math.ceil(N / pad) * pad

    print(
        f"Config: TILE=({TILE_M}x{TILE_N}x{TILE_K}), GROUP={GROUP_SIZE_M}, persistent={use_persistent}"
    )

    A = torch.randn(M_padded, K_padded, dtype=dtype, device="cuda")
    B = torch.randn(K_padded, N_padded, dtype=dtype, device="cuda")
    C_cutile = torch.zeros(M_padded, N_padded, dtype=dtype, device="cuda")

    # Validate
    run_cutile_gemm(
        A, B, C_cutile, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M, use_persistent
    )
    C_pytorch = pytorch_gemm(A, B)
    passed, max_diff = validate(C_cutile, C_pytorch)
    print(f"Correctness: {'PASSED' if passed else f'FAILED (diff={max_diff:.6f})'}")

    # Benchmark
    cutile_time = benchmark(
        lambda: run_cutile_gemm(
            A, B, C_cutile, TILE_M, TILE_N, TILE_K, GROUP_SIZE_M, use_persistent
        )
    )
    pytorch_time = benchmark(lambda: pytorch_gemm(A, B))

    flops = 2 * M_padded * N_padded * K_padded
    cutile_tflops = flops / (cutile_time * 1e-3) / 1e12
    pytorch_tflops = flops / (pytorch_time * 1e-3) / 1e12
    speedup = pytorch_time / cutile_time if cutile_time > 0 else 0

    print(f"cuTile:  {cutile_time:.3f} ms ({cutile_tflops:.2f} TFLOPS)")
    print(f"PyTorch: {pytorch_time:.3f} ms ({pytorch_tflops:.2f} TFLOPS)")
    print(f"vs PyTorch: {speedup:.2f}x")

    return {
        "M": M_padded,
        "K": K_padded,
        "N": N_padded,
        "cutile_ms": cutile_time,
        "pytorch_ms": pytorch_time,
        "cutile_tflops": cutile_tflops,
        "pytorch_tflops": pytorch_tflops,
        "speedup": speedup,
        "passed": passed,
    }


def main():
    print("=" * 60)
    print("GEMM Benchmark: cuTile vs PyTorch (v3)")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}, SM: {torch.cuda.get_device_capability()}")
    print(f"SMs: {torch.cuda.get_device_properties('cuda').multi_processor_count}")

    test_cases = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (1024, 4096, 1024),
        (4096, 1024, 4096),
        (8192, 8192, 8192),
    ]

    results = []
    for M, K, N in test_cases:
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
