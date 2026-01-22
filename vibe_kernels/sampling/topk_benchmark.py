# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch  # type: ignore[import]

from .kernel import _select_topk


@dataclass
class BenchmarkResult:
    torch_ms: float
    triton_ms: float
    speedup: float
    max_abs_diff: float
    indices_equal: bool


def _time_fn(iters: int, fn, *args) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        fn(*args)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _benchmark(
    batch: int, vocab: int, top_k: int, dtype: torch.dtype, iters: int
) -> BenchmarkResult:
    logits = torch.randn(batch, vocab, device="cuda", dtype=dtype)

    def torch_impl() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.topk(logits, top_k, dim=-1)

    def triton_impl() -> tuple[torch.Tensor, torch.Tensor]:
        return _select_topk(logits, top_k)

    torch_ms = _time_fn(iters, torch_impl)
    triton_ms = _time_fn(iters, triton_impl)

    torch_vals, torch_idx = torch_impl()
    triton_vals, triton_idx = triton_impl()

    max_abs = (torch_vals.to(torch.float32) - triton_vals).abs().max().item()
    indices_equal = torch.equal(torch_idx.to(torch.int32), triton_idx)
    speedup = torch_ms / triton_ms if triton_ms > 0 else float("inf")
    return BenchmarkResult(torch_ms, triton_ms, speedup, max_abs, indices_equal)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton top-k selector vs torch.topk"
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--top-k", dest="top_k", type=int, default=40)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float32"
    )
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():  # pragma: no cover
        raise SystemError("CUDA is required for top-k benchmarking")

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)

    result = _benchmark(args.batch, args.vocab, args.top_k, dtype, args.iters)

    print("Top-k Benchmark")
    print("================")
    print(f"batch            : {args.batch}")
    print(f"vocab size       : {args.vocab}")
    print(f"top_k            : {args.top_k}")
    print(f"dtype            : {args.dtype}")
    print(f"torch.topk       : {result.torch_ms:.4f} ms")
    print(f"triton select    : {result.triton_ms:.4f} ms")
    print(f"speedup          : {result.speedup:.3f}x")
    print(f"max |diff|       : {result.max_abs_diff:.3e}")
    print(f"indices equal    : {result.indices_equal}")


if __name__ == "__main__":
    main()
