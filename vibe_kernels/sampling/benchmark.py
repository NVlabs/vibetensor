# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch  # type: ignore[import]
import torch.nn.functional as F

from .kernel import sample_logits


@dataclass
class BenchmarkResult:
    baseline_ms: float
    triton_ms: float
    speedup: float


def _time_fn(iters: int, fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _baseline(
    logits: torch.Tensor, top_k: int, temperature: float, generator: torch.Generator
) -> torch.Tensor:
    if top_k is not None and top_k < logits.size(-1):
        values, indices = torch.topk(logits, top_k, dim=-1)
        probs = F.softmax(values / temperature, dim=-1)
        draws = torch.multinomial(probs, num_samples=1, generator=generator)
        return indices.gather(-1, draws).squeeze(-1)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _benchmark(
    batch: int, vocab: int, top_k: int, temperature: float, warmup: int, iters: int
) -> BenchmarkResult:
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(batch, vocab, device="cuda", dtype=torch.float32)

    def baseline():
        _baseline(logits, top_k, temperature, generator)

    def triton_impl():
        sample_logits(logits, top_k=top_k, temperature=temperature, generator=generator)

    for _ in range(warmup):
        baseline()
        triton_impl()

    baseline_ms = _time_fn(iters, baseline)
    triton_ms = _time_fn(iters, triton_impl)
    speedup = baseline_ms / triton_ms if triton_ms > 0 else float("inf")
    return BenchmarkResult(baseline_ms, triton_ms, speedup)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton sampling against torch.multinomial"
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():  # pragma: no cover
        raise SystemError("CUDA is required for sampling benchmark")

    result = _benchmark(
        args.batch, args.vocab, args.top_k, args.temperature, args.warmup, args.iters
    )

    print("Sampling Benchmark")
    print("===================")
    print(f"batch            : {args.batch}")
    print(f"vocab size       : {args.vocab}")
    print(f"top_k            : {args.top_k}")
    print(f"temperature      : {args.temperature}")
    print(f"baseline mean    : {result.baseline_ms:.4f} ms")
    print(f"triton mean      : {result.triton_ms:.4f} ms")
    print(f"speedup          : {result.speedup:.3f}x")


if __name__ == "__main__":
    main()
