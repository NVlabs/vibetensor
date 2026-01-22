# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark helper for Muon Newton–Schulz iterations."""

from __future__ import annotations

import argparse
import time

import torch

from vibe_kernels.optim.impl.triton_impl import (  # type: ignore[import]
    fast_newton_schulz,
)


def _reference_newton_schulz(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = matrix.to(torch.bfloat16)
    transpose_result = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transpose_result = True

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = X @ X.mT
        gram2 = gram @ gram.mT
        B = b * gram + c * gram2
        X = a * X + torch.matmul(B.to(X.dtype), X)

    if transpose_result:
        X = X.mT
    return X.to(matrix.dtype)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Muon Newton–Schulz benchmark")
    parser.add_argument(
        "--dim", type=int, default=2048, help="Matrix dimension (square)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Input dtype",
    )
    parser.add_argument(
        "--steps", type=int, default=5, help="Number of Newton–Schulz iterations"
    )
    parser.add_argument("--runs", type=int, default=10, help="Timed runs after warmup")
    parser.add_argument(
        "--warmup", type=int, default=3, help="Warmup runs before timing"
    )
    return parser.parse_args()


def _time_fn(fn, matrix: torch.Tensor, steps: int, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn(matrix, steps)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        fn(matrix, steps)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / runs * 1000.0


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)

    torch.manual_seed(0)
    matrix = torch.randn(args.dim, args.dim, device=device, dtype=dtype)

    triton_time = _time_fn(
        fast_newton_schulz, matrix, args.steps, args.warmup, args.runs
    )
    ref_time = _time_fn(
        _reference_newton_schulz, matrix, args.steps, args.warmup, args.runs
    )

    result_triton = fast_newton_schulz(matrix.clone(), args.steps)
    result_ref = _reference_newton_schulz(matrix.clone(), args.steps)
    gram_triton = (result_triton @ result_triton.mT).float()
    gram_ref = (result_ref @ result_ref.mT).float()

    torch.testing.assert_close(result_triton, result_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(gram_triton, gram_ref, atol=5e-2, rtol=5e-2)

    print("Muon Newton–Schulz Benchmark (device: cuda)")
    print(f"dim={args.dim}, dtype={args.dtype}, iterations={args.steps}")
    print(f"Triton fast_newton_schulz : {triton_time:.4f} ms/iteration")
    print(f"PyTorch reference        : {ref_time:.4f} ms/iteration")
    print(f"Speedup                   : {ref_time / triton_time:.2f}x")


if __name__ == "__main__":
    main()
