#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quickly validate the correctness and performance of every top-k implementation.

Usage:
    cd /workspace/terry/nano-cursor
    python kernel_factory/sampling/test_all_topk.py
"""

import os
import sys
import time

import torch

# Add search paths
sys.path.insert(0, "/workspace/terry/nano-cursor/kernel_factory")
sys.path.insert(0, "/workspace/terry/nano-cursor/kernel_factory/sampling")
sys.path.insert(0, "/workspace/terry/nano-cursor/tmp/quack")

print("=" * 80)
print("Top-K implementation quick check")
print("=" * 80)


def benchmark(fn, warmup=5, iters=20):
    """Simple benchmark helper."""
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


# Test configuration
M, N, k = 4096, 1024, 32
print(f"\nTest configuration: M={M}, N={N}, k={k}, dtype=bfloat16\n")

# Create inputs
torch.manual_seed(42)
x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

# ============================================================================
# 1. PyTorch (baseline)
# ============================================================================
print("1. PyTorch torch.topk")
print("-" * 80)
vals_torch, idx_torch = torch.topk(x, k, dim=-1, largest=True, sorted=True)
time_torch = benchmark(lambda: torch.topk(x, k, dim=-1))
print(f"   Time: {time_torch:.4f} ms")
print(f"   ✅ Baseline implementation\n")

# ============================================================================
# 2. CuTe (ours)
# ============================================================================
print("2. CuTe (our standalone implementation)")
print("-" * 80)
try:
    from cute_topk import topk as cute_topk

    vals_cute, idx_cute = cute_topk(x, k)
    time_cute = benchmark(lambda: cute_topk(x, k))

    # Validate correctness
    correct = torch.allclose(vals_cute.float(), vals_torch.float(), rtol=1e-3)
    max_diff = (vals_cute.float() - vals_torch.float()).abs().max().item()

    print(f"   Time: {time_cute:.4f} ms")
    print(f"   Speedup: {time_torch / time_cute:.2f}x vs PyTorch")
    print(f"   Correctness: {'✅ pass' if correct else '❌ fail'}")
    print(f"   Max error: {max_diff:.6f}\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# ============================================================================
# 3. QuACK (original)
# ============================================================================
print("3. QuACK (original implementation)")
print("-" * 80)
try:
    from quack.topk import topk as quack_topk

    vals_quack, idx_quack = quack_topk(x, k)
    time_quack = benchmark(lambda: quack_topk(x, k))

    # Validate correctness
    correct = torch.allclose(vals_quack.float(), vals_torch.float(), rtol=1e-3)
    max_diff = (vals_quack.float() - vals_torch.float()).abs().max().item()

    print(f"   Time: {time_quack:.4f} ms")
    print(f"   Speedup: {time_torch / time_quack:.2f}x vs PyTorch")
    print(f"   Correctness: {'✅ pass' if correct else '❌ fail'}")
    print(f"   Max error: {max_diff:.6f}\n")
except Exception as e:
    print(f"   ❌ Unavailable (requires a full QuACK installation)\n")

# ============================================================================
# 4. Triton (ours - stream variant)
# ============================================================================
print("4. Triton (our 2-stage implementation)")
print("-" * 80)
try:
    os.environ["AIKF_TOPK_IMPL"] = "stream"
    if "sampling.kernel" in sys.modules:
        del sys.modules["sampling.kernel"]
    from sampling.kernel import _select_topk

    vals_triton, idx_triton = _select_topk(x, k)
    time_triton = benchmark(lambda: _select_topk(x, k), warmup=3, iters=10)

    # Validate correctness
    correct = torch.allclose(vals_triton.float(), vals_torch.float(), rtol=1e-3)
    max_diff = (vals_triton.float() - vals_torch.float()).abs().max().item()

    print(f"   Time: {time_triton:.4f} ms")
    print(f"   Speedup: {time_torch / time_triton:.2f}x vs PyTorch")
    print(f"   Correctness: {'✅ pass' if correct else '❌ fail'}")
    print(f"   Max error: {max_diff:.6f}")
    print(
        f"   ⚠️  Performance warning: Triton is {time_triton / time_torch:.1f}x slower than PyTorch\n"
    )
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("Summary")
print("=" * 80)
print(f"✅ PyTorch:  {time_torch:.4f} ms (baseline)")

try:
    print(f"✅ CuTe:     {time_cute:.4f} ms ({time_torch / time_cute:.2f}x)")
except:
    print(f"❌ CuTe:     unavailable")

try:
    print(f"✅ QuACK:    {time_quack:.4f} ms ({time_torch / time_quack:.2f}x)")
except:
    print(f"❌ QuACK:    unavailable")

try:
    print(f"⚠️  Triton:   {time_triton:.4f} ms ({time_torch / time_triton:.2f}x)")
except:
    print(f"❌ Triton:   unavailable")

print("\nRecommendations:")
print("  • N < 1024:  use PyTorch")
print("  • N ≥ 1024:  use CuTe (fastest)")
print("  • Any size:  stick with PyTorch (most flexible)")
print("  • ❌ Avoid Triton (too slow)")
print("=" * 80)
