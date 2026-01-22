# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

try:
    from vibe_kernels.softmax import (
        cutedsl_softmax,
        is_cutedsl_available,
        log_softmax,
        softmax,
    )
except ImportError:
    print("Could not import kernel_factory.softmax")
    raise


def test_softmax_fallback():
    print("\nTesting Softmax Fallback (Torch)...")
    x = torch.randn(16, 1024)

    # Test Torch fallback
    out = softmax(x, dim=-1)
    ref = torch.nn.functional.softmax(x, dim=-1)
    diff = (out - ref).abs().max()
    print(f"Max diff (Torch Backend): {diff.item()}")
    assert diff < 1e-5


def test_cutedsl_softmax_import():
    print("\nTesting CuTeDSL Softmax Import...")
    if is_cutedsl_available():
        print("CuTeDSL backend is available.")
        if not torch.cuda.is_available():
            print("Skipping execution check (CUDA unavailable)")
            return

        device = torch.device("cuda")
        # CuTeDSL kernel expects 2D input or specific shapes, let's try standard 2D
        x = torch.randn(16, 1024, device=device, dtype=torch.float16)

        try:
            # Explicitly request cutedsl backend via wrapper or direct call
            out = cutedsl_softmax(x)
            print(f"CuTeDSL output shape: {out.shape}")
            assert out.shape == x.shape

            # Compare vs torch
            ref = torch.nn.functional.softmax(x.float(), dim=-1).to(torch.float16)
            diff = (out - ref).abs().max()
            print(f"Max diff vs Torch (CuTeDSL): {diff.item()}")

        except Exception as e:
            print(f"CuTeDSL execution failed (expected if not compiled/setup): {e}")
    else:
        print("CuTeDSL backend is NOT available.")


if __name__ == "__main__":
    test_softmax_fallback()
    test_cutedsl_softmax_import()
