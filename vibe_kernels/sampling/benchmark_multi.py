# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]

from .kernel import sample_logits  # type: ignore[import]


@dataclass
class BenchmarkResult:
    baseline_ms: float
    triton_ms: float
    speedup: float
    max_freq_diff: float


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


def _make_generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g


def _run_benchmark(
    batch: int,
    vocab: int,
    top_k: int,
    num_samples: int,
    temperature: float,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    seed: int,
    repeats: int,
) -> BenchmarkResult:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    logits = torch.randn(batch, vocab, device="cuda", dtype=dtype)
    scaled_logits = logits / max(temperature, 1e-6)

    baseline_gen = _make_generator(seed)
    triton_gen = _make_generator(seed)

    def _baseline_call() -> torch.Tensor:
        state = baseline_gen.get_state()
        if top_k is not None and top_k < vocab:
            vals, idx = torch.topk(scaled_logits, top_k, dim=-1)
            probs = F.softmax(vals, dim=-1)
            draws = torch.multinomial(
                probs,
                num_samples=num_samples,
                replacement=True,
                generator=baseline_gen,
            )
            tokens = idx.gather(-1, draws)
        else:
            probs = F.softmax(scaled_logits, dim=-1)
            tokens = torch.multinomial(
                probs,
                num_samples=num_samples,
                replacement=True,
                generator=baseline_gen,
            )
        baseline_gen.set_state(state)
        return tokens

    def _triton_call() -> torch.Tensor:
        state = triton_gen.get_state()
        tokens = sample_logits(
            logits,
            top_k=top_k,
            temperature=temperature,
            generator=triton_gen,
            num_samples=num_samples,
        )
        triton_gen.set_state(state)
        return tokens

    baseline_ms = _time_fn(warmup, iters, _baseline_call)
    triton_ms = _time_fn(warmup, iters, _triton_call)

    # Distribution check on a replicated single row
    row_logits = logits[:1].expand(repeats, -1).contiguous()
    ref_gen = _make_generator(seed)
    ours_gen = _make_generator(seed)

    if top_k is not None and top_k < vocab:
        vals, idx = torch.topk(row_logits / max(temperature, 1e-6), top_k, dim=-1)
        probs = F.softmax(vals, dim=-1)
        ref_draws = torch.multinomial(
            probs,
            num_samples=num_samples,
            replacement=True,
            generator=ref_gen,
        )
        ref_tokens = idx.gather(-1, ref_draws)
    else:
        probs = F.softmax(row_logits / max(temperature, 1e-6), dim=-1)
        ref_tokens = torch.multinomial(
            probs,
            num_samples=num_samples,
            replacement=True,
            generator=ref_gen,
        )
        idx = torch.arange(vocab, device=row_logits.device)[None, :]

    ours_tokens = sample_logits(
        row_logits,
        top_k=top_k,
        temperature=temperature,
        generator=ours_gen,
        num_samples=num_samples,
    )

    vocab_range = idx[0]
    ref_counts = torch.bincount(
        ref_tokens.view(-1).to(torch.long), minlength=vocab
    ).float()
    ours_counts = torch.bincount(
        ours_tokens.view(-1).to(torch.long), minlength=vocab
    ).float()
    ref_freq = ref_counts / ref_counts.sum()
    ours_freq = ours_counts / ours_counts.sum()
    max_freq_diff = (ref_freq[vocab_range] - ours_freq[vocab_range]).abs().max().item()

    speedup = baseline_ms / triton_ms if triton_ms > 0 else float("inf")
    return BenchmarkResult(baseline_ms, triton_ms, speedup, max_freq_diff)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark multi-sample Triton sampler vs PyTorch"
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float32"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():  # pragma: no cover
        raise SystemError("CUDA is required for multi-sample benchmarking")

    dtype = _torch_dtype(args.dtype)

    result = _run_benchmark(
        batch=args.batch,
        vocab=args.vocab,
        top_k=args.top_k,
        num_samples=args.num_samples,
        temperature=args.temperature,
        dtype=dtype,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        repeats=args.repeats,
    )

    print("Multi-sample Sampling Benchmark")
    print("================================")
    print(f"batch            : {args.batch}")
    print(f"vocab size       : {args.vocab}")
    print(f"top_k            : {args.top_k}")
    print(f"num_samples      : {args.num_samples}")
    print(f"temperature      : {args.temperature}")
    print(f"dtype            : {args.dtype}")
    print(f"baseline mean    : {result.baseline_ms:.4f} ms")
    print(f"triton mean      : {result.triton_ms:.4f} ms")
    print(f"speedup          : {result.speedup:.3f}x")
    print(f"max |freq diff|  : {result.max_freq_diff:.5f}")


if __name__ == "__main__":  # pragma: no cover
    main()
