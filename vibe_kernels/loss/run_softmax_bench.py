# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from vibe_kernels.loss import log_softmax as kf_log_softmax, softmax as kf_softmax

try:
    from quack.softmax import softmax as quack_softmax  # type: ignore[import]

    HAS_QUACK = True
except ImportError:  # pragma: no cover - optional dependency
    quack_softmax = None
    HAS_QUACK = False


def ensure_cuda() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - guardrail
        raise SystemExit("CUDA device is required for benchmark")


@dataclass
class BenchConfig:
    rows: int
    cols: int
    dtype: torch.dtype
    name: str


@dataclass
class BenchResult:
    forward_ms: Optional[float]
    backward_ms: Optional[float]


def _time_function(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def _compute_forward(x: torch.Tensor, backend: str, op: str) -> torch.Tensor:
    if backend == "torch":
        if op == "softmax":
            return F.softmax(x, dim=-1)
        if op == "log_softmax":
            return F.log_softmax(x, dim=-1)
        raise ValueError(op)  # pragma: no cover
    if backend == "triton":
        if op == "softmax":
            return kf_softmax(x, dim=-1, backend="triton")
        if op == "log_softmax":
            return kf_log_softmax(x, dim=-1, backend="triton")
        raise ValueError(op)  # pragma: no cover
    if backend == "cutedsl":
        if op == "softmax":
            return kf_softmax(x, dim=-1, backend="cutedsl")
        if op == "log_softmax":
            return kf_log_softmax(x, dim=-1, backend="cutedsl")
        raise ValueError(op)  # pragma: no cover
    if backend == "quack":
        if not HAS_QUACK:
            raise RuntimeError("Quack backend requested but not available")
        if op != "softmax":
            raise RuntimeError("Quack backend does not implement log_softmax")
        return quack_softmax(x)  # type: ignore[operator]
    raise ValueError(f"Unknown backend {backend}")


def _bench_config(
    config: BenchConfig, backend: str, op: str, warmup: int, iters: int
) -> BenchResult:
    if op == "log_softmax" and backend == "quack":
        return BenchResult(None, None)

    device = torch.device("cuda")
    x_base = torch.randn(config.rows, config.cols, device=device, dtype=config.dtype)
    grad_base = torch.randn_like(x_base, dtype=torch.float32)

    def forward_once() -> None:
        with torch.no_grad():
            _compute_forward(x_base, backend, op)

    forward_ms = _time_function(forward_once, warmup, iters)

    def backward_once() -> None:
        x = x_base.detach().clone().requires_grad_(True)
        y = _compute_forward(x, backend, op)
        grad = grad_base.to(y.dtype)
        y.backward(grad)

    backward_ms = _time_function(backward_once, warmup, iters)

    return BenchResult(forward_ms, backward_ms)


def run_benchmarks(
    configs: List[BenchConfig],
    backends: List[str],
    op: str,
    warmup: int,
    iters: int,
) -> Dict[str, List[BenchResult]]:
    results: Dict[str, List[BenchResult]] = {backend: [] for backend in backends}
    for cfg in configs:
        for backend in backends:
            results[backend].append(_bench_config(cfg, backend, op, warmup, iters))
    return results


def _format_ms(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def print_table(
    title: str,
    configs: List[BenchConfig],
    backends: List[str],
    results: Dict[str, List[BenchResult]],
) -> None:
    print(title)
    header = ["Backend", "Rows", "Cols", "Dtype", "Forward (ms)", "Backward (ms)"]
    print("| " + " | ".join(header) + " |")
    print("|" + "-" * (len(header) * 12) + "|")
    for backend in backends:
        for cfg, res in zip(configs, results[backend]):
            print(
                "| {} | {} | {} | {} | {} | {} |".format(
                    backend,
                    cfg.rows,
                    cfg.cols,
                    cfg.dtype.__repr__().split(".")[-1],
                    _format_ms(res.forward_ms),
                    _format_ms(res.backward_ms),
                )
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark softmax backends")
    parser.add_argument(
        "--iters", type=int, default=20, help="Iterations per measurement"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations per measurement"
    )
    args = parser.parse_args()

    ensure_cuda()

    configs = [
        BenchConfig(4096, 8192, torch.float16, "4096x8192_fp16"),
        BenchConfig(4096, 8192, torch.bfloat16, "4096x8192_bf16"),
        BenchConfig(4096, 16384, torch.float16, "4096x16384_fp16"),
    ]

    softmax_backends = ["torch", "triton", "cutedsl"] + (["quack"] if HAS_QUACK else [])
    log_softmax_backends = ["torch", "triton", "cutedsl"]

    softmax_results = run_benchmarks(
        configs, softmax_backends, op="softmax", warmup=args.warmup, iters=args.iters
    )
    log_results = run_benchmarks(
        configs,
        log_softmax_backends,
        op="log_softmax",
        warmup=args.warmup,
        iters=args.iters,
    )

    print_table("Softmax Forward/Backward", configs, softmax_backends, softmax_results)
    print_table(
        "Log-Softmax Forward/Backward", configs, log_softmax_backends, log_results
    )


if __name__ == "__main__":
    main()
