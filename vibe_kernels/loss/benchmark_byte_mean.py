# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]

from .kernel import cross_entropy_loss  # type: ignore[import]


@dataclass
class BenchmarkResult:
    baseline_ms: float
    triton_ms: float
    speedup: float
    max_abs_diff: float


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype {name}")


def _time_fn(warmup: int, iters: int, fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _load_token_bytes(
    vocab: int, path: str | None, synthetic_period: int
) -> torch.Tensor:
    if path is not None:
        tensor = torch.load(path, map_location="cuda")
        if tensor.numel() < vocab:
            raise ValueError("token_bytes tensor shorter than vocab")
        return tensor.to(torch.int32)
    pattern = (
        torch.arange(synthetic_period, device="cuda", dtype=torch.int32)
        % synthetic_period
    )
    tensor = pattern.repeat((vocab + synthetic_period - 1) // synthetic_period)[
        :vocab
    ].clone()
    tensor[0] = 0
    return tensor


def _run_benchmark(
    batch: int,
    seq: int,
    vocab: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    seed: int,
    ignore_index: int,
    token_bytes_path: str | None,
    synthetic_period: int,
) -> BenchmarkResult:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    logits = torch.randn(batch, seq, vocab, device="cuda", dtype=dtype)
    targets = torch.randint(0, vocab, (batch, seq), device="cuda", dtype=torch.long)
    if ignore_index >= 0:
        targets[:, 0] = ignore_index
    token_bytes = _load_token_bytes(vocab, token_bytes_path, synthetic_period)

    logits_f32 = logits.detach().clone().to(torch.float32).requires_grad_(True)
    logits_triton = logits.detach().clone().requires_grad_(True)

    flat_targets = targets.view(-1)

    def _baseline_step() -> torch.Tensor:
        loss_none = F.cross_entropy(
            logits_f32.view(-1, vocab),
            flat_targets,
            ignore_index=ignore_index,
            reduction="none",
        )
        weights = torch.zeros_like(loss_none)
        valid = flat_targets >= 0
        weights[valid] = token_bytes[flat_targets[valid]].to(weights.dtype)
        total_bytes = weights.sum().clamp(min=1)
        return (loss_none * weights).sum() / total_bytes

    def _baseline_call() -> torch.Tensor:
        loss = _baseline_step()
        loss.backward()
        if logits_f32.grad is not None:
            logits_f32.grad.zero_()
        return loss

    def _triton_call() -> torch.Tensor:
        loss = cross_entropy_loss(
            logits_triton,
            targets,
            ignore_index=ignore_index,
            reduction="byte_mean",
            token_bytes=token_bytes,
        )
        loss.backward()
        if logits_triton.grad is not None:
            logits_triton.grad.zero_()
        return loss

    baseline_ms = _time_fn(warmup, iters, _baseline_call)
    triton_ms = _time_fn(warmup, iters, _triton_call)

    loss_base = _baseline_step().detach()
    loss_triton = cross_entropy_loss(
        logits_triton.detach(),
        targets,
        ignore_index=ignore_index,
        reduction="byte_mean",
        token_bytes=token_bytes,
    ).detach()
    max_abs_diff = (loss_base - loss_triton).abs().item()

    speedup = baseline_ms / triton_ms if triton_ms > 0 else float("inf")
    return BenchmarkResult(baseline_ms, triton_ms, speedup, max_abs_diff)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark byte-weighted Triton cross-entropy"
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--vocab", type=int, default=50304)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-index", type=int, default=-1)
    parser.add_argument(
        "--token-bytes", type=str, default=None, help="Optional path to token_bytes.pt"
    )
    parser.add_argument(
        "--synthetic-period",
        type=int,
        default=5,
        help="Period for synthetic byte pattern when no tensor is provided",
    )
    args = parser.parse_args(argv)

    if (
        args.token_bytes is not None and not Path(args.token_bytes).exists()
    ):  # pragma: no cover
        raise FileNotFoundError(f"token_bytes file not found: {args.token_bytes}")

    if not torch.cuda.is_available():  # pragma: no cover
        raise SystemError("CUDA is required for byte-mean benchmarking")

    dtype = _torch_dtype(args.dtype)

    result = _run_benchmark(
        batch=args.batch,
        seq=args.seq,
        vocab=args.vocab,
        dtype=dtype,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        ignore_index=args.ignore_index,
        token_bytes_path=args.token_bytes,
        synthetic_period=args.synthetic_period,
    )

    print("Byte-mean Cross Entropy Benchmark")
    print("==================================")
    print(f"batch            : {args.batch}")
    print(f"sequence length  : {args.seq}")
    print(f"vocab size       : {args.vocab}")
    print(f"dtype            : {args.dtype}")
    print(f"ignore_index     : {args.ignore_index}")
    if args.token_bytes:
        print(f"token_bytes      : {args.token_bytes}")
    else:
        print(f"synthetic period : {args.synthetic_period}")
    print(f"baseline mean    : {result.baseline_ms:.4f} ms")
    print(f"triton mean      : {result.triton_ms:.4f} ms")
    print(f"speedup          : {result.speedup:.3f}x")
    print(f"max |diff|       : {result.max_abs_diff:.6f}")


if __name__ == "__main__":  # pragma: no cover
    main()
