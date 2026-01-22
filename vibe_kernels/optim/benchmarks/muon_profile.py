# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Break down Muon Newton–Schulz runtime into granular components."""

from __future__ import annotations

import argparse
import time

import torch

from vibe_kernels.optim.muon_kernels import (  # type: ignore[import]
    fast_newton_schulz,
    matmul_transpose_assign,
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


def _time_fn(fn, *args, **kwargs) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0


def _profile_steps(matrix: torch.Tensor, steps: int) -> None:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = matrix.clone()
    buf1 = torch.empty((X.size(-2), X.size(-2)), device=X.device, dtype=X.dtype)
    buf2 = torch.empty_like(buf1)

    times = []
    for _ in range(steps):
        t1 = _time_fn(matmul_transpose_assign, X, buf1)
        t2 = _time_fn(matmul_transpose_assign, buf1, buf2)
        B = b * buf1 + c * buf2
        t3 = _time_fn(torch.matmul, B.to(X.dtype), X)
        X = torch.add(a * X, torch.matmul(B.to(X.dtype), X))
        times.append((t1, t2, t3))

    print("Per-iteration breakdown (ms):")
    for idx, (t1, t2, t3) in enumerate(times, 1):
        print(f"  iter {idx}: matmul_transpose {t1:.4f} / {t2:.4f} | gemm {t3:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Muon components")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    matrix = torch.randn(args.dim, args.dim, device=device, dtype=dtype)

    triton_time = _time_fn(fast_newton_schulz, matrix.clone(), args.steps)
    ref_time = _time_fn(_reference_newton_schulz, matrix.clone(), args.steps)
    print(f"Total Triton fast_newton_schulz: {triton_time:.4f} ms")
    print(f"Total PyTorch reference         : {ref_time:.4f} ms")

    _profile_steps(matrix.clone(), args.steps)


if __name__ == "__main__":
    main()
