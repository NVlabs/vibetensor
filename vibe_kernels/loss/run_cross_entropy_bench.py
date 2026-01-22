# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from vibe_kernels.loss import cross_entropy_loss as kf_cross_entropy

try:  # pragma: no cover - optional dependency
    from quack.cross_entropy import (  # type: ignore[import]
        cross_entropy as quack_cross_entropy,
    )

    HAS_QUACK = True
except ImportError:  # pragma: no cover - optional dependency
    quack_cross_entropy = None
    HAS_QUACK = False

IGNORE_INDEX = -1


@dataclass
class BenchConfig:
    batch: int
    vocab: int
    dtype: torch.dtype
    ignore_ratio: float = 0.1


@dataclass
class BenchResult:
    forward_ms: float
    backward_ms: float


def ensure_cuda() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - guardrail
        raise SystemExit("CUDA device is required for benchmark")


def _time_function(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(iters, 1)


def _prepare_targets(config: BenchConfig, device: torch.device) -> torch.Tensor:
    targets = torch.randint(
        0, config.vocab, (config.batch,), device=device, dtype=torch.long
    )
    if config.ignore_ratio > 0:
        mask = torch.rand(config.batch, device=device) < config.ignore_ratio
        if mask.all():
            mask[0] = False
        targets = targets.clone()
        targets[mask] = IGNORE_INDEX
    return targets


def _compute_forward(
    logits: torch.Tensor, targets: torch.Tensor, backend: str
) -> torch.Tensor:
    if backend == "torch":
        return F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)
    if backend == "triton":
        return kf_cross_entropy(
            logits, targets, ignore_index=IGNORE_INDEX, backend="triton"
        )
    if backend == "cutedsl":
        return kf_cross_entropy(
            logits, targets, ignore_index=IGNORE_INDEX, backend="cutedsl"
        )
    if backend == "quack":
        if not HAS_QUACK:
            raise RuntimeError("Quack backend requested but quack is not available")
        return quack_cross_entropy(  # type: ignore[operator]
            logits, targets, ignore_index=IGNORE_INDEX, reduction="mean"
        )
    raise ValueError(f"Unknown backend {backend}")


def _bench_single(
    config: BenchConfig,
    backend: str,
    warmup: int,
    iters: int,
    logits_base: torch.Tensor,
    targets: torch.Tensor,
) -> BenchResult:
    def forward_once() -> None:
        with torch.no_grad():
            _compute_forward(logits_base, targets, backend)

    forward_ms = _time_function(forward_once, warmup, iters)

    def backward_once() -> None:
        logits = logits_base.detach().clone().requires_grad_(True)
        loss = _compute_forward(logits, targets, backend)
        loss.backward()

    backward_ms = _time_function(backward_once, warmup, iters)
    return BenchResult(forward_ms, backward_ms)


def run_benchmarks(
    configs: List[BenchConfig],
    backends: List[str],
    warmup: int,
    iters: int,
) -> Dict[str, List[BenchResult]]:
    device = torch.device("cuda")
    results: Dict[str, List[BenchResult]] = {backend: [] for backend in backends}
    for config in configs:
        logits_base = torch.randn(
            config.batch, config.vocab, device=device, dtype=config.dtype
        )
        targets = _prepare_targets(config, device)
        for backend in backends:
            results[backend].append(
                _bench_single(config, backend, warmup, iters, logits_base, targets)
            )
    return results


def _format_ms(value: float) -> str:
    return f"{value:.4f}"


def _format_speedup(torch_value: float, backend_value: float) -> str:
    speed = torch_value / backend_value
    return f"{speed:.2f}×"


def print_table(
    title: str,
    configs: List[BenchConfig],
    backends: List[str],
    results: Dict[str, List[BenchResult]],
) -> None:
    print(title)
    header = [
        "Backend",
        "Batch",
        "Vocab",
        "Dtype",
        "Forward (ms)",
        "Speedup",
        "Backward (ms)",
        "Speedup",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "-" * (len(header) * 12) + "|")
    for backend in backends:
        for idx, config in enumerate(configs):
            res = results[backend][idx]
            torch_res = results["torch"][idx]
            forward_speed = _format_speedup(torch_res.forward_ms, res.forward_ms)
            backward_speed = _format_speedup(torch_res.backward_ms, res.backward_ms)
            print(
                "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    backend,
                    config.batch,
                    config.vocab,
                    config.dtype.__repr__().split(".")[-1],
                    _format_ms(res.forward_ms),
                    forward_speed,
                    _format_ms(res.backward_ms),
                    backward_speed,
                )
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cross-entropy backends")
    parser.add_argument(
        "--iters", type=int, default=20, help="Iterations per measurement"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations per measurement"
    )
    args = parser.parse_args()

    ensure_cuda()
    torch.manual_seed(0)

    configs = [
        BenchConfig(4096, 8192, torch.float16),
        BenchConfig(4096, 8192, torch.bfloat16),
        BenchConfig(4096, 16384, torch.float16),
    ]

    backends = ["torch", "triton", "cutedsl"] + (["quack"] if HAS_QUACK else [])
    if not HAS_QUACK:
        print("Quack backend not available; skipping Quack measurements.\n")
    results = run_benchmarks(configs, backends, args.warmup, args.iters)
    print_table("Cross-Entropy Forward/Backward", configs, backends, results)


if __name__ == "__main__":
    main()
