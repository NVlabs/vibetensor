#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark streaming top-k kernel with different configurations."""

import os
import sys
import time
from typing import List, Tuple

import torch


# Import after setting env vars
def benchmark_config(
    batch_size: int,
    vocab_size: int,
    k: int,
    impl: str,
    block_n: int = 128,
    num_warps: int = 4,
) -> Tuple[float, float]:
    """Benchmark a specific configuration."""
    os.environ["AIKF_TOPK_IMPL"] = impl
    os.environ["AIKF_STREAMING_BLOCK_N"] = str(block_n)
    os.environ["AIKF_STREAMING_WARPS"] = str(num_warps)

    # Reload module to pick up env changes
    if "kernel" in sys.modules:
        del sys.modules["kernel"]
    from kernel import _select_topk

    # Warmup
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float16)
    for _ in range(5):
        vals, idx = _select_topk(logits, k)
        torch.cuda.synchronize()

    # Benchmark
    trials = 20
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(trials):
        vals, idx = _select_topk(logits, k)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / trials * 1000
    return avg_ms


def main():
    print("Benchmarking streaming top-k kernel")
    print("=" * 80)

    # Test configuration: b=1, vocab=50k, k=50 (typical LLM sampling)
    batch_size = 1
    vocab_size = 50000
    k = 50

    print(f"\nConfiguration: batch={batch_size}, vocab={vocab_size}, k={k}")
    print("-" * 80)

    # Baseline: PyTorch
    print("\n1. PyTorch baseline:")
    torch_time = benchmark_config(batch_size, vocab_size, k, "torch")
    print(f"   Time: {torch_time:.4f} ms")

    # Streaming configurations
    print("\n2. Streaming kernel (autotune BLOCK_N and num_warps):")

    configs = []
    for block_n in [64, 128, 256]:
        for num_warps in [4, 8]:
            stream_time = benchmark_config(
                batch_size, vocab_size, k, "streaming", block_n, num_warps
            )
            speedup = torch_time / stream_time
            configs.append((block_n, num_warps, stream_time, speedup))
            print(
                f"   BLOCK_N={block_n:3d}, num_warps={num_warps}: {stream_time:.4f} ms  (speedup: {speedup:.2f}x)"
            )

    # Find best config
    best_config = min(configs, key=lambda x: x[2])
    print(f"\n3. Best streaming config:")
    print(f"   BLOCK_N={best_config[0]}, num_warps={best_config[1]}")
    print(
        f"   Time: {best_config[2]:.4f} ms  (speedup: {best_config[3]:.2f}x vs PyTorch)"
    )

    # Test with larger batch
    print(f"\n4. Larger batch test (batch=8):")
    batch_size = 8
    torch_time_b8 = benchmark_config(batch_size, vocab_size, k, "torch")
    stream_time_b8 = benchmark_config(
        batch_size, vocab_size, k, "streaming", best_config[0], best_config[1]
    )
    speedup_b8 = torch_time_b8 / stream_time_b8
    print(f"   PyTorch: {torch_time_b8:.4f} ms")
    print(f"   Streaming: {stream_time_b8:.4f} ms  (speedup: {speedup_b8:.2f}x)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
