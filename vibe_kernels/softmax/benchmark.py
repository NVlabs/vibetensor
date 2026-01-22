# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, Tuple

import torch

from .kernel import log_softmax, softmax


def _benchmark_case(
    rows: int,
    cols: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
) -> None:
    torch.manual_seed(0)
    if dim == -1:
        shape = (rows, cols)
        x = torch.randn(rows, cols, device=device, dtype=dtype)
    else:
        # If dim=0, shape is (cols, rows) effectively for reduction?
        # Let's stick to simple (rows, cols) and softmax along dim
        shape = (rows, cols)
        x = torch.randn(rows, cols, device=device, dtype=dtype)

    print(f"Benchmarking Softmax shape={shape} dim={dim} dtype={dtype}")

    def _baseline():
        return torch.nn.functional.softmax(x, dim=dim, dtype=torch.float32).to(dtype)

    def _triton():
        return softmax(x, dim=dim, backend="triton")

    def _cutedsl():
        return softmax(x, dim=dim, backend="cutedsl")

    # Validation
    ref = _baseline()

    try:
        tri = _triton()
        tri_diff = (tri - ref).abs().max().item()
        tri_close = torch.allclose(tri, ref, atol=1e-2, rtol=0)
    except Exception as e:
        print(f"Triton failed: {e}")
        tri_diff = -1
        tri_close = False

    try:
        cut = _cutedsl()
        cut_diff = (cut - ref).abs().max().item()
        cut_close = torch.allclose(cut, ref, atol=1e-2, rtol=0)
    except Exception as e:
        print(f"CuTeDSL failed: {e}")
        cut_diff = -1
        cut_close = False

    print(
        f"Validation: Triton diff={tri_diff:.6f} ({tri_close}), CuTeDSL diff={cut_diff:.6f} ({cut_close})"
    )

    def _time(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0

    # Warmup
    for _ in range(warmup):
        _baseline()
        if tri_close:
            _triton()
        if cut_close:
            _cutedsl()
    torch.cuda.synchronize()

    # Measure
    baseline_times = [_time(_baseline) for _ in range(iters)]
    baseline_mean = statistics.fmean(baseline_times)
    print(f"Baseline: {baseline_mean:.4f} ms")

    if tri_close:
        triton_times = [_time(_triton) for _ in range(iters)]
        triton_mean = statistics.fmean(triton_times)
        print(
            f"Triton:   {triton_mean:.4f} ms (Speedup: {baseline_mean/triton_mean:.2f}x)"
        )

    if cut_close:
        cutedsl_times = [_time(_cutedsl) for _ in range(iters)]
        cutedsl_mean = statistics.fmean(cutedsl_times)
        print(
            f"CuTeDSL:  {cutedsl_mean:.4f} ms (Speedup: {baseline_mean/cutedsl_mean:.2f}x)"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)

    _benchmark_case(
        args.rows, args.cols, args.dim, dtype, device, args.warmup, args.iters
    )


if __name__ == "__main__":
    main()
