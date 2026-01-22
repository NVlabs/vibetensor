# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch  # type: ignore[import]
import torch.nn.functional as F

from .kernel import log_softmax


@dataclass
class BenchmarkResult:
    baseline_ms: float
    triton_ms: float
    speedup: float
    max_abs_diff: float


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


def _prepare(batch: int, seq: int, vocab: int, dtype: torch.dtype) -> torch.Tensor:
    shape = (batch, seq, vocab) if seq > 0 else (batch, vocab)
    return torch.randn(*shape, device="cuda", dtype=dtype)


def _benchmark(
    batch: int,
    seq: int,
    vocab: int,
    dtype: torch.dtype,
    dim: int,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    logits = _prepare(batch, seq, vocab, dtype)

    def baseline() -> torch.Tensor:
        return F.log_softmax(logits.float(), dim=dim)

    def triton_impl() -> torch.Tensor:
        return log_softmax(logits, dim=dim)

    for _ in range(warmup):
        baseline()
        triton_impl()

    baseline_ms = _time_fn(iters, baseline)
    triton_ms = _time_fn(iters, triton_impl)

    with torch.no_grad():
        ref = F.log_softmax(logits.float(), dim=dim)
        ours = log_softmax(logits, dim=dim).float()
        max_abs = torch.max(torch.abs(ref - ours)).item()

    speedup = baseline_ms / triton_ms if triton_ms > 0 else float("inf")
    return BenchmarkResult(baseline_ms, triton_ms, speedup, max_abs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark Triton log_softmax")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--dim", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():  # pragma: no cover
        raise SystemError("CUDA is required for log_softmax benchmarking")

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    result = _benchmark(
        args.batch, args.seq, args.vocab, dtype, args.dim, args.warmup, args.iters
    )

    print("LogSoftmax Benchmark")
    print("=====================")
    print(f"batch            : {args.batch}")
    if args.seq > 0:
        print(f"sequence length  : {args.seq}")
    print(f"vocab size       : {args.vocab}")
    print(f"dtype            : {args.dtype}")
    print(f"dim              : {args.dim}")
    print(f"baseline mean    : {result.baseline_ms:.4f} ms")
    print(f"triton mean      : {result.triton_ms:.4f} ms")
    print(f"speedup          : {result.speedup:.3f}x")
    print(f"max |diff|       : {result.max_abs_diff:.6f}")


if __name__ == "__main__":
    main()
