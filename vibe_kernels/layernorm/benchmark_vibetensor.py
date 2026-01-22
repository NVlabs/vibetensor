# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np
import torch
import triton

# Ensure we can import kernel_factory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming the script is in kernel_factory/layernorm/
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    import vibetensor.torch as vt
    from vibetensor import _C as C

    VIBETENSOR_AVAILABLE = True
except ImportError:
    VIBETENSOR_AVAILABLE = False
    print("VibeTensor not found. Skipping VibeTensor benchmarks.")

try:
    from vibe_kernels.layernorm.impl import triton_impl

    # Import vibetensor_impl to trigger registration
    if VIBETENSOR_AVAILABLE:
        from vibe_kernels.layernorm.impl import vibetensor_impl
except ImportError:
    print("Kernel Factory imports failed")
    sys.exit(1)


def benchmark_layernorm(M=4096, N=4096, dtype=torch.float16, eps=1e-5):
    if not torch.cuda.is_available():
        print("No CUDA available")
        return

    print(f"Benchmarking LayerNorm | Shape: ({M}, {N}) | Dtype: {dtype}")

    # Inputs
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    b = torch.randn(N, device="cuda", dtype=dtype)

    # Warmup & Correctness
    y_ref = torch.nn.functional.layer_norm(x, (N,), w, b, eps=eps)

    providers = []

    # 1. PyTorch Native
    providers.append(
        (
            "PyTorch Native",
            lambda: torch.nn.functional.layer_norm(x, (N,), w, b, eps=eps),
        )
    )

    # 2. Kernel Factory (Direct Triton)
    providers.append(
        ("KF Direct (Triton)", lambda: triton_impl.layernorm(x, w, b, eps))
    )

    # 3. VibeTensor
    if VIBETENSOR_AVAILABLE:
        # Prepare VibeTensor inputs
        x_vt = vt.from_dlpack(x)
        w_vt = vt.from_dlpack(w)
        b_vt = vt.from_dlpack(b)
        # Scalar eps as tensor for dispatch
        eps_vt = C._cuda_h2d_alloc_copy(
            np.array(eps, dtype=np.float32), "float32", 0, False
        )

        def run_vbt():
            out = C._call_op("kf::layernorm", x_vt, w_vt, b_vt, eps_vt)
            # Explicit sync for accurate timing of async dispatch in do_bench
            vt.cuda.current_stream().synchronize()
            return out

        # Verify VBT correctness first
        y_vbt = run_vbt()
        y_vbt_torch = torch.from_dlpack(y_vbt)
        if not torch.allclose(y_vbt_torch, y_ref, atol=1e-2, rtol=1e-2):
            diff = (y_vbt_torch - y_ref).abs().max().item()
            print(f"WARNING: VibeTensor output mismatch! Max Diff: {diff}")

        providers.append(("VibeTensor (Adapter)", run_vbt))

    # Run Benchmarks
    print(f"\n{'Provider':<25} | {'Time (ms)':<10} | {'GB/s':<10}")
    print("-" * 55)

    results = []
    quantiles = [0.5, 0.2, 0.8]
    element_size = x.element_size()
    # Read X, Read W, Read B, Write Y
    total_bytes = 2 * M * N * element_size

    for name, fn in providers:
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        gbps = (total_bytes * 1e-9) / (ms * 1e-3)
        results.append((name, ms, gbps))
        print(f"{name:<25} | {ms:<10.4f} | {gbps:<10.2f}")


if __name__ == "__main__":
    benchmark_layernorm()
