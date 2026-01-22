# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Literal

import torch  # type: ignore[import]
import torch.nn.functional as F

from .kernel import cross_entropy_loss

ReductionArg = Literal["mean", "none"]


@dataclass
class BenchmarkResult:
    baseline_ms: float
    triton_ms: float
    cutedsl_ms: float
    speedup_tri: float
    speedup_cut: float
    max_abs_diff: float
    cut_max_abs_diff: float


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


def _prepare(
    batch: int, seq: int, vocab: int, dtype: torch.dtype, ignore_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(batch, seq, vocab, device="cuda", dtype=dtype)
    targets = torch.randint(0, vocab, (batch, seq), device="cuda", dtype=torch.long)
    targets[:, 0] = ignore_index
    return logits, targets


def _benchmark(
    batch: int,
    seq: int,
    vocab: int,
    dtype: torch.dtype,
    reduction: ReductionArg,
    warmup: int,
    iters: int,
) -> BenchmarkResult:
    ignore_index = -1
    logits, targets = _prepare(batch, seq, vocab, dtype, ignore_index)

    def baseline():
        F.cross_entropy(
            logits.float().view(-1, vocab),
            targets.view(-1),
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def triton_impl():
        cross_entropy_loss(
            logits,
            targets,
            ignore_index=ignore_index,
            reduction=reduction,
            backend="triton",
        )

    def cutedsl_impl():
        cross_entropy_loss(
            logits,
            targets,
            ignore_index=ignore_index,
            reduction=reduction,
            backend="cutedsl",
        )

    # Warmup
    for _ in range(warmup):
        baseline()
        try:
            triton_impl()
        except:
            pass
        try:
            cutedsl_impl()
        except:
            pass

    # --- Quack ---
    print("Benchmarking Quack...")
    import importlib
    import sys

    quack_module = None
    if "/workspace/terry/nano-cursor/tmp/quack" not in sys.path:
        sys.path.insert(0, "/workspace/terry/nano-cursor/tmp/quack")
    try:
        quack_module = importlib.import_module("quack.cross_entropy")
    except ImportError:
        print("Quack module not found")

    def quack_impl():
        if quack_module:
            # Quack expects flattened inputs (M, N)
            logits_flat = logits.view(-1, vocab)
            targets_flat = targets.view(-1)
            quack_module.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=ignore_index,
                reduction=reduction,
            )

    try:
        for _ in range(warmup):
            quack_impl()
        torch.cuda.synchronize()
        quack_samples = [_time_fn(1, quack_impl) for _ in range(iters)]
        quack_mean = statistics.fmean(quack_samples)
    except Exception as e:
        print(f"Quack failed: {e}")
        quack_mean = float("inf")

    # Measure
    baseline_ms = _time_fn(iters, baseline)

    try:
        triton_ms = _time_fn(iters, triton_impl)
    except RuntimeError as e:
        print(f"Triton failed: {e}")
        triton_ms = float("inf")

    try:
        cutedsl_ms = _time_fn(iters, cutedsl_impl)
    except RuntimeError as e:
        print(f"CuTeDSL failed: {e}")
        cutedsl_ms = float("inf")

    # Verification
    with torch.no_grad():
        # Reference: PyTorch (Float32 computation)
        ref = (
            F.cross_entropy(
                logits.float().view(-1, vocab),
                targets.view(-1),
                ignore_index=ignore_index,
                reduction="none",
            )
            .float()
            .view(-1)
        )

        # Triton
        try:
            ours_tri = (
                cross_entropy_loss(
                    logits,
                    targets,
                    ignore_index=ignore_index,
                    reduction="none",
                    backend="triton",
                )
                .float()
                .view(-1)
            )
            tri_diff = torch.max(torch.abs(ref - ours_tri)).item()
        except:
            tri_diff = -1.0

        # CuTeDSL
        try:
            ours_cut = (
                cross_entropy_loss(
                    logits,
                    targets,
                    ignore_index=ignore_index,
                    reduction="none",
                    backend="cutedsl",
                )
                .float()
                .view(-1)
            )
            cut_diff = torch.max(torch.abs(ref - ours_cut)).item()
        except Exception as e:
            print(f"CuTeDSL verify failed: {e}")
            cut_diff = -1.0

    speedup_tri = baseline_ms / triton_ms if triton_ms > 0 else 0.0
    speedup_cut = baseline_ms / cutedsl_ms if cutedsl_ms > 0 else 0.0
    speedup_quack = (
        baseline_ms / quack_mean
        if quack_mean > 0 and quack_mean != float("inf")
        else 0.0
    )

    print(f"Quack mean: {quack_mean:.4f} ms")
    print(f"Quack speedup: {speedup_quack:.3f}x")

    return BenchmarkResult(
        baseline_ms, triton_ms, cutedsl_ms, speedup_tri, speedup_cut, tri_diff, cut_diff
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark Triton cross-entropy loss")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--reduction", choices=["mean", "none"], default="mean")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():  # pragma: no cover
        raise SystemError("CUDA is required for cross entropy benchmarking")

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    result = _benchmark(
        args.batch, args.seq, args.vocab, dtype, args.reduction, args.warmup, args.iters
    )

    print("Cross Entropy Benchmark")
    print("========================")
    print(f"batch            : {args.batch}")
    print(f"sequence length  : {args.seq}")
    print(f"vocab size       : {args.vocab}")
    print(f"dtype            : {args.dtype}")
    print(f"reduction        : {args.reduction}")
    print(f"baseline mean    : {result.baseline_ms:.4f} ms")
    print(f"triton mean      : {result.triton_ms:.4f} ms")
    print(f"cutedsl mean     : {result.cutedsl_ms:.4f} ms")
    print(f"triton speedup   : {result.speedup_tri:.3f}x")
    print(f"cutedsl speedup  : {result.speedup_cut:.3f}x")
    print(f"triton |diff|    : {result.max_abs_diff:.6f}")
    print(f"cutedsl |diff|   : {result.cut_max_abs_diff:.6f}")


if __name__ == "__main__":
    main()
