# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import statistics
import time
from typing import Tuple

import torch  # type: ignore[import]

from .kernel import FusedEmbeddingRMSNorm


def _benchmark_case(
    num_tokens: int,
    vocab: int,
    hidden: int,
    eps: float,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> Tuple[float, float, bool, float]:
    rng = torch.Generator(device=device)
    rng.manual_seed(0)

    tokens = torch.randint(0, vocab, (num_tokens,), device=device, generator=rng)
    embed = torch.nn.Embedding(vocab, hidden, device=device, dtype=dtype)
    torch.nn.init.normal_(embed.weight, mean=0.0, std=1.0 / hidden**0.5)

    fused = FusedEmbeddingRMSNorm(vocab, hidden, eps=eps, dtype=dtype, device=device)
    fused.weight.data.copy_(embed.weight.data)

    def _baseline():
        out = embed(tokens).to(torch.float32)
        out = torch.nn.functional.rms_norm(out, (hidden,), eps=eps)
        return out.to(dtype)

    def _fused():
        return fused(tokens)

    for _ in range(warmup):
        _baseline()
        _fused()

    def _time(fn) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0

    baseline_samples = [_time(_baseline) for _ in range(iters)]
    fused_samples = [_time(_fused) for _ in range(iters)]

    baseline_mean = statistics.fmean(baseline_samples)
    fused_mean = statistics.fmean(fused_samples)

    ref = _baseline()
    hyp = _fused()
    diff = (ref - hyp).abs()
    max_diff = diff.max().item()
    allclose = torch.allclose(ref, hyp, atol=1e-2, rtol=0)

    return baseline_mean, fused_mean, allclose, max_diff


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fused embedding RMSNorm kernel"
    )
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=50304)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    if not torch.cuda.is_available():  # pragma: no cover - requires GPU
        raise RuntimeError("CUDA is required to benchmark the fused embedding kernel")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)

    baseline_mean, fused_mean, allclose, max_diff = _benchmark_case(
        args.tokens,
        args.vocab,
        args.hidden,
        args.eps,
        device,
        dtype,
        args.warmup,
        args.iters,
    )

    print("Fused Embedding + RMSNorm Benchmark")
    print("====================================")
    print(f"tokens          : {args.tokens}")
    print(f"vocab size      : {args.vocab}")
    print(f"hidden dim      : {args.hidden}")
    print(f"dtype           : {args.dtype}")
    print(f"baseline mean   : {baseline_mean:.4f} ms")
    print(f"fused mean      : {fused_mean:.4f} ms")
    print(f"speedup         : {baseline_mean / fused_mean:.3f}x")
    print(f"max |diff|      : {max_diff:.6f}")
    print(f"allclose        : {allclose}")


if __name__ == "__main__":
    main()
