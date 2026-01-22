# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np
import torch
import triton

# Ensure we can import kernel_factory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    import vibetensor.torch as vt
    from vibetensor import _C as C

    VIBETENSOR_AVAILABLE = True
except ImportError:
    VIBETENSOR_AVAILABLE = False
    print("VibeTensor not found.")

# Import implementation
try:
    # Import vibetensor_impl first to register op
    if VIBETENSOR_AVAILABLE:
        from vibe_kernels.rmsnorm.impl import vibetensor_impl
    from vibe_kernels.rmsnorm.impl import triton_impl
except ImportError:
    print("Kernel Factory RMSNorm impl not found")
    sys.exit(1)


# Reference RMSNorm (Torch Native)
def torch_rmsnorm(x, w, eps):
    if hasattr(torch.nn.functional, "rms_norm"):
        return torch.nn.functional.rms_norm(x, x.shape[-1:], w, eps=eps)
    else:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return w * x.to(input_dtype)


def benchmark_rmsnorm(M=4096, N=4096, dtype=torch.float16, eps=1e-5):
    if not torch.cuda.is_available():
        print("No CUDA")
        return

    print(f"Benchmarking RMSNorm | Shape: ({M}, {N}) | Dtype: {dtype}")

    # Inputs
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)  # Gamma

    # Warmup & Reference
    y_ref = torch_rmsnorm(x, w, eps)

    providers = []
    providers.append(("PyTorch Native", lambda: torch_rmsnorm(x, w, eps)))

    # Kernel Factory Direct
    # The triton_impl exposes _RMSNormFn.apply(x, gamma, eps) or a class TritonRMSNorm
    providers.append(
        ("KF Direct (Triton)", lambda: triton_impl._RMSNormFn.apply(x, w, eps))
    )

    # VibeTensor
    if VIBETENSOR_AVAILABLE:
        x_vt = vt.from_dlpack(x)
        w_vt = vt.from_dlpack(w)
        # Pass eps as tensor (0-d) for dispatch; it will be unwrapped by register_triton_op helper
        eps_vt = C._cuda_h2d_alloc_copy(
            np.array(eps, dtype=np.float32), "float32", 0, False
        )

        def run_vbt():
            # Call using schema registered in vibetensor_impl.py: kf::rmsnorm_fwd
            out = C._call_op("kf::rmsnorm_fwd", x_vt, w_vt, eps_vt)
            vt.cuda.current_stream().synchronize()
            return out

        # Verify
        y_vbt = run_vbt()
        y_vbt_torch = torch.from_dlpack(y_vbt)
        if not torch.allclose(y_vbt_torch, y_ref, atol=1e-2, rtol=1e-2):
            diff = (y_vbt_torch - y_ref).abs().max().item()
            print(f"WARNING: VibeTensor output mismatch! Max Diff: {diff}")

        providers.append(("VibeTensor (Adapter)", run_vbt))

    # Run Benchmarks
    print(f"\n{'Provider':<25} | {'Time (ms)':<10} | {'GB/s':<10}")
    print("-" * 55)

    quantiles = [0.5, 0.2, 0.8]
    element_size = x.element_size()
    # Read X, Read W, Write Y
    total_bytes = 2 * M * N * element_size

    for name, fn in providers:
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        gbps = (total_bytes * 1e-9) / (ms * 1e-3)
        print(f"{name:<25} | {ms:<10.4f} | {gbps:<10.2f}")


if __name__ == "__main__":
    benchmark_rmsnorm()
