# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import dataclasses
import math
import statistics
import time
from typing import Iterable, List, Sequence, Tuple

import torch  # type: ignore[import]

from .kernel import (
    cutedsl_gemm,
    cutedsl_gemm_backward,
    is_cutedsl_available,
    triton_gemm,
    triton_gemm_backward,
)


def _torch_backward(
    grad_output: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_a = torch.matmul(grad_output, b.transpose(0, 1))
    grad_b = torch.matmul(a.transpose(0, 1), grad_output)
    grad_bias = grad_output.sum(dim=0)
    return grad_a, grad_b, grad_bias


@dataclasses.dataclass(frozen=True)
class BenchmarkCase:
    label: str
    shape: Tuple[int, int, int]
    iterations: int
    warmup: int

    @property
    def flops(self) -> int:
        m, n, k = self.shape
        return 2 * m * n * k


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_single_case(
    case: BenchmarkCase, dtype: torch.dtype, device: torch.device, backend: str
) -> dict:
    m, n, k = case.shape
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=dtype, generator=generator)
    b = torch.randn((k, n), device=device, dtype=dtype, generator=generator)
    bias = torch.randn((n,), device=device, dtype=dtype, generator=generator)
    grad_output = torch.randn((m, n), device=device, dtype=dtype, generator=generator)

    if backend == "triton":
        backend_label = "triton"

        def fwd_impl() -> torch.Tensor:
            return triton_gemm(a, b, bias=bias)

        def bwd_impl():
            return triton_gemm_backward(grad_output, a, b, compute_grad_bias=True)

    elif backend == "cutedsl":
        if not is_cutedsl_available():
            raise RuntimeError("CuTeDSL backend requested but not available")
        backend_label = "cutedsl"

        def fwd_impl() -> torch.Tensor:
            return cutedsl_gemm(a, b, bias=bias)

        def bwd_impl():
            return cutedsl_gemm_backward(grad_output, a, b, compute_grad_bias=True)

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Warmup
    for _ in range(case.warmup):
        fwd_impl()
    for _ in range(case.warmup):
        bwd_impl()
    for _ in range(case.warmup):
        _ = torch.matmul(a, b) + bias
    for _ in range(case.warmup):
        _torch_backward(grad_output, a, b)

    def _time(fn) -> List[float]:
        timings: List[float] = []
        for _ in range(case.iterations):
            _synchronize()
            start = time.perf_counter()
            fn()
            _synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
        return timings

    backend_fwd_timings = _time(fwd_impl)
    torch_fwd_timings = _time(lambda: torch.matmul(a, b) + bias)

    backend_bwd_timings = _time(bwd_impl)
    torch_bwd_timings = _time(lambda: _torch_backward(grad_output, a, b))

    backend_out = fwd_impl()
    torch_out = torch.matmul(a, b) + bias
    max_abs_diff = (backend_out - torch_out).abs().max().item()
    allclose = torch.allclose(backend_out, torch_out, atol=1e-2, rtol=0)

    grad_a_backend, grad_b_backend, grad_bias_backend = bwd_impl()
    assert (
        grad_a_backend is not None
        and grad_b_backend is not None
        and grad_bias_backend is not None
    )
    grad_a_torch, grad_b_torch, grad_bias_torch = _torch_backward(grad_output, a, b)

    max_grad_a_diff = (grad_a_backend - grad_a_torch).abs().max().item()
    max_grad_b_diff = (grad_b_backend - grad_b_torch).abs().max().item()
    max_grad_bias_diff = (grad_bias_backend - grad_bias_torch).abs().max().item()
    allclose_bwd = (
        torch.allclose(grad_a_backend, grad_a_torch, atol=1e-2, rtol=0)
        and torch.allclose(grad_b_backend, grad_b_torch, atol=1e-2, rtol=0)
        and torch.allclose(grad_bias_backend, grad_bias_torch, atol=1e-2, rtol=0)
    )

    def _summary(samples: Sequence[float]) -> Tuple[float, float]:
        mean = statistics.fmean(samples)
        p50 = statistics.median(samples)
        return mean, p50

    backend_fwd_mean, backend_fwd_p50 = _summary(backend_fwd_timings)
    torch_fwd_mean, torch_fwd_p50 = _summary(torch_fwd_timings)
    backend_bwd_mean, backend_bwd_p50 = _summary(backend_bwd_timings)
    torch_bwd_mean, torch_bwd_p50 = _summary(torch_bwd_timings)

    fwd_speedup = torch_fwd_mean / backend_fwd_mean if backend_fwd_mean else math.nan
    bwd_speedup = torch_bwd_mean / backend_bwd_mean if backend_bwd_mean else math.nan
    fwd_gflops = (case.flops / 1e12) / (backend_fwd_mean / 1000.0)
    bwd_flops = 4 * m * n * k + m * n
    bwd_gflops = (bwd_flops / 1e12) / (backend_bwd_mean / 1000.0)

    return {
        "label": case.label,
        "backend": backend_label,
        "shape": case.shape,
        "dtype": str(dtype).split(".")[-1],
        "backend_ms": backend_fwd_mean,
        "backend_p50_ms": backend_fwd_p50,
        "torch_ms": torch_fwd_mean,
        "torch_p50_ms": torch_fwd_p50,
        "speedup": fwd_speedup,
        "gflops": fwd_gflops,
        "max_diff": float(max_abs_diff),
        "allclose": bool(allclose),
        "backend_bwd_ms": backend_bwd_mean,
        "backend_bwd_p50_ms": backend_bwd_p50,
        "torch_bwd_ms": torch_bwd_mean,
        "torch_bwd_p50_ms": torch_bwd_p50,
        "bwd_speedup": bwd_speedup,
        "bwd_gflops": bwd_gflops,
        "max_grad_a_diff": float(max_grad_a_diff),
        "max_grad_b_diff": float(max_grad_b_diff),
        "max_grad_bias_diff": float(max_grad_bias_diff),
        "allclose_bwd": bool(allclose_bwd),
        "iterations": case.iterations,
        "warmup": case.warmup,
    }


def _format_result(row: dict) -> str:
    m, n, k = row["shape"]
    backend = row["backend"]
    return (
        f"{row['label']:<8} [{backend}] | {m:5d}x{k:5d} * {k:5d}x{n:5d} | "
        f"fwd {backend} {row['backend_ms']:.3f} ms (p50 {row['backend_p50_ms']:.3f}) | "
        f"fwd torch {row['torch_ms']:.3f} ms (p50 {row['torch_p50_ms']:.3f}) | "
        f"bwd {backend} {row['backend_bwd_ms']:.3f} ms (p50 {row['backend_bwd_p50_ms']:.3f}) | "
        f"bwd torch {row['torch_bwd_ms']:.3f} ms (p50 {row['torch_bwd_p50_ms']:.3f}) | "
        f"speedup fwd {row['speedup']:.3f}x bwd {row['bwd_speedup']:.3f}x | "
        f"diff fwd {row['max_diff']:.6f} ga {row['max_grad_a_diff']:.6f} gb {row['max_grad_b_diff']:.6f} bias {row['max_grad_bias_diff']:.6f} | "
        f"allclose fwd {row['allclose']} bwd {row['allclose_bwd']}"
    )


def run_benchmarks(
    cases: Iterable[BenchmarkCase],
    dtype: torch.dtype,
    device: torch.device,
    backend: str,
) -> List[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for benchmarking")
    torch.cuda.reset_peak_memory_stats(device)
    return [_run_single_case(case, dtype, device, backend) for case in cases]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton GEMM kernel against torch.matmul"
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16"],
        help="Data type to benchmark",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run benchmarks on (default: cuda)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs=3,
        metavar=("I2048", "I4096", "I8192"),
        default=[50, 20, 10],
        help="Iterations for (2048,4096,8192) square cases",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations per case",
    )
    parser.add_argument(
        "--backend",
        default="triton",
        choices=["triton", "cutedsl"],
        help="Kernel backend to benchmark (default: triton)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    cases = [
        BenchmarkCase("2048", (2048, 2048, 2048), args.iterations[0], args.warmup),
        BenchmarkCase(
            "4096", (4096, 4096, 4096), args.iterations[1], max(args.warmup, 5)
        ),
        BenchmarkCase(
            "8192", (8192, 8192, 8192), args.iterations[2], max(args.warmup, 3)
        ),
    ]

    backend = args.backend
    results = run_benchmarks(cases, dtype, device, backend)
    title = f"{backend.upper()} GEMM Benchmark"
    print(f"\n{title}\n{'=' * len(title)}")
    for row in results:
        print(_format_result(row))

    print("\nSummary Table")
    header = (
        "label",
        "backend",
        "shape",
        "dtype",
        "backend_ms",
        "torch_ms",
        "speedup",
        "gflops",
        "backend_bwd_ms",
        "torch_bwd_ms",
        "bwd_speedup",
        "bwd_gflops",
        "max_diff",
        "max_grad_a_diff",
        "max_grad_b_diff",
        "max_grad_bias_diff",
        "allclose",
        "allclose_bwd",
    )
    print(",".join(header))
    for row in results:
        print(
            ",".join(
                [
                    row["label"],
                    row["backend"],
                    "x".join(str(dim) for dim in row["shape"]),
                    row["dtype"],
                    f"{row['backend_ms']:.4f}",
                    f"{row['torch_ms']:.4f}",
                    f"{row['speedup']:.3f}",
                    f"{row['gflops']:.2f}",
                    f"{row['backend_bwd_ms']:.4f}",
                    f"{row['torch_bwd_ms']:.4f}",
                    f"{row['bwd_speedup']:.3f}",
                    f"{row['bwd_gflops']:.2f}",
                    f"{row['max_diff']:.6f}",
                    f"{row['max_grad_a_diff']:.6f}",
                    f"{row['max_grad_b_diff']:.6f}",
                    f"{row['max_grad_bias_diff']:.6f}",
                    str(row["allclose"]),
                    str(row["allclose_bwd"]),
                ]
            )
        )


if __name__ == "__main__":
    main()
