# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Tuple

import torch  # type: ignore[import]

from .kernel import relu_squared, softcap_tanh_projection


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class BenchmarkResult:
    baseline_ms: float
    triton_ms: float
    speedup: float
    max_abs_diff: float


def _parse_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{name}'")
    return _DTYPE_MAP[name]


def _time_function(iters: int, fn: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _benchmark_relu_squared(
    batch: int, features: int, dtype: torch.dtype, warmup: int, iters: int
) -> BenchmarkResult:
    x = torch.randn(batch, features, device="cuda", dtype=dtype)

    def baseline():
        torch.relu(x) ** 2

    def triton_impl():
        relu_squared(x)

    for _ in range(warmup):
        baseline()
        triton_impl()

    baseline_ms = _time_function(iters, baseline)
    triton_ms = _time_function(iters, triton_impl)

    ref = torch.relu(x) ** 2
    out = relu_squared(x)
    max_abs = torch.max(torch.abs(out - ref)).item()
    speedup = baseline_ms / triton_ms if triton_ms > 0 else float("inf")
    return BenchmarkResult(baseline_ms, triton_ms, speedup, max_abs)


def _softcap_reference(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + x.abs())


def _benchmark_softcap_tanh(
    batch: int, features: int, dtype: torch.dtype, warmup: int, iters: int
) -> BenchmarkResult:
    x = torch.randn(batch, features, device="cuda", dtype=dtype)
    weight = torch.randn(features, device="cuda", dtype=dtype)

    def baseline():
        proj = x * weight
        _softcap_reference(proj)
        torch.tanh(proj)

    def triton_impl():
        softcap_tanh_projection(x, weight)

    for _ in range(warmup):
        baseline()
        triton_impl()

    baseline_ms = _time_function(iters, baseline)
    triton_ms = _time_function(iters, triton_impl)

    proj = x * weight
    ref_softcap = _softcap_reference(proj)
    ref_tanh = torch.tanh(proj)
    softcap_out, tanh_out = softcap_tanh_projection(x, weight)
    max_abs = max(
        torch.max(torch.abs(softcap_out - ref_softcap)).item(),
        torch.max(torch.abs(tanh_out - ref_tanh)).item(),
    )
    speedup = baseline_ms / triton_ms if triton_ms > 0 else float("inf")
    return BenchmarkResult(baseline_ms, triton_ms, speedup, max_abs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark activation kernels against PyTorch baselines"
    )
    parser.add_argument(
        "--activation", choices=["relu_squared", "softcap_tanh"], default="relu_squared"
    )
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--features", type=int, default=4096)
    parser.add_argument("--dtype", choices=list(_DTYPE_MAP.keys()), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():  # pragma: no cover
        raise SystemError("CUDA is required for activation benchmarks")

    torch.manual_seed(0)

    dtype = _parse_dtype(args.dtype)

    if args.activation == "relu_squared":
        result = _benchmark_relu_squared(
            args.batch, args.features, dtype, args.warmup, args.iters
        )
    else:
        result = _benchmark_softcap_tanh(
            args.batch, args.features, dtype, args.warmup, args.iters
        )

    print("Activation Benchmark")
    print("=====================")
    print(f"activation       : {args.activation}")
    print(f"batch            : {args.batch}")
    print(f"features         : {args.features}")
    print(f"dtype            : {args.dtype}")
    print(f"baseline mean    : {result.baseline_ms:.4f} ms")
    print(f"triton mean      : {result.triton_ms:.4f} ms")
    print(f"speedup          : {result.speedup:.3f}x")
    print(f"max |diff|       : {result.max_abs_diff:.6f}")


if __name__ == "__main__":
    main()
