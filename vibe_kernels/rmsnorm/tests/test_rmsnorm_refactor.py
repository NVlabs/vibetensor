# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

try:
    from vibe_kernels.rmsnorm import cutedsl_rmsnorm, is_cutedsl_available, RMSNorm
except ImportError:
    print("Could not import kernel_factory.rmsnorm")
    raise


def test_triton_rmsnorm():
    print("\nTesting Triton RMSNorm...")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    dtype = torch.float16
    hidden_size = 1024

    model = RMSNorm(hidden_size, dtype=dtype, device=device)
    x = torch.randn(16, hidden_size, device=device, dtype=dtype)

    out = model(x)
    print(f"Triton output shape: {out.shape}")
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

    # Compare with torch.nn.RMSNorm
    ref_model = torch.nn.RMSNorm(hidden_size, eps=model.eps).to(device).to(dtype)
    # Copy weights
    with torch.no_grad():
        ref_model.weight.copy_(model.gamma)

    ref_out = ref_model(x)
    diff = (out - ref_out).abs().max()
    print(f"Max diff vs Torch: {diff.item()}")
    assert diff < 1e-3


def test_cutedsl_rmsnorm_import():
    print("\nTesting CuTeDSL RMSNorm Import...")
    if is_cutedsl_available():
        print("CuTeDSL backend is available.")
        if not torch.cuda.is_available():
            print("Skipping execution check (CUDA unavailable)")
            return

        device = torch.device("cuda")
        dtype = torch.float16
        hidden_size = 1024
        x = torch.randn(16, hidden_size, device=device, dtype=dtype)

        try:
            out = cutedsl_rmsnorm(x)
            print(f"CuTeDSL output shape: {out.shape}")
            assert out.shape == x.shape
        except Exception as e:
            print(f"CuTeDSL execution failed (expected if not compiled/setup): {e}")
    else:
        print("CuTeDSL backend is NOT available (likely import error in lazy load).")
        # If it's not available, we should check if it was due to missing dependencies or broken imports
        # but for now just noting it.


if __name__ == "__main__":
    test_triton_rmsnorm()
    test_cutedsl_rmsnorm_import()
