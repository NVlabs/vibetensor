# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Ensure kernel_factory and quack are importable
sys.path.insert(0, "/workspace/terry/nano-cursor/tmp/quack")

from quack.topk import topk as quack_topk  # type: ignore[import]

from vibe_kernels.sampling.kernel import (  # type: ignore[import]
    _select_topk,
    _select_topk_cutedsl,
    sample_logits,
)


@dataclass
class TopKResult:
    name: str
    ms: float


@dataclass
class SamplingResult:
    name: str
    ms: float


def _bench(fn, iters: int = 50, warmup: int = 10) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def numeric_check_topk() -> None:
    assert torch.cuda.is_available(), "CUDA is required"
    torch.manual_seed(0)
    batch, vocab, k = 4, 1024, 32
    for dtype in (torch.float16, torch.bfloat16):
        logits = torch.randn(batch, vocab, device="cuda", dtype=dtype)
        vals_cute, idx_cute = _select_topk_cutedsl(logits, k)
        vals_ref, idx_ref = torch.topk(logits, k, dim=-1)
        max_diff = (vals_cute.float() - vals_ref.float()).abs().max().item()
        indices_equal = torch.equal(idx_cute, idx_ref.to(torch.int32))
        print(
            f"[topk numeric] dtype={dtype}, max |Δvals|={max_diff:.3e}, indices_equal={indices_equal}"
        )


def numeric_check_sampling() -> None:
    assert torch.cuda.is_available(), "CUDA is required"
    torch.manual_seed(1)
    vocab = 64
    top_k = 8
    temperature = 0.9
    repeats = 4096

    logits = torch.arange(vocab, device="cuda", dtype=torch.float32).unsqueeze(0)
    tiled = logits.expand(repeats, -1).contiguous()

    gen_cut = torch.Generator(device="cuda").manual_seed(123)
    draws_cut = sample_logits(
        tiled,
        top_k=top_k,
        temperature=temperature,
        generator=gen_cut,
        num_samples=1,
        backend="cutedsl",
    )
    counts_cut = torch.bincount(draws_cut.view(-1), minlength=vocab).float()
    freq_cut = counts_cut / counts_cut.sum()

    gen_torch = torch.Generator(device="cuda").manual_seed(123)
    draws_torch = sample_logits(
        tiled,
        top_k=top_k,
        temperature=temperature,
        generator=gen_torch,
        num_samples=1,
        backend="torch",
    )
    counts_torch = torch.bincount(draws_torch.view(-1), minlength=vocab).float()
    freq_torch = counts_torch / counts_torch.sum()

    max_diff = (freq_cut - freq_torch).abs().max().item()
    print(
        f"[sampling numeric] vocab={vocab}, top_k={top_k}, temp={temperature}, max |Δfreq|={max_diff:.3e}"
    )


def benchmark_topk() -> None:
    assert torch.cuda.is_available(), "CUDA is required"
    M, N, K = 4096, 1024, 32
    dtype = torch.bfloat16
    torch.manual_seed(0)
    logits = torch.randn(M, N, device="cuda", dtype=dtype)

    def torch_impl():
        return torch.topk(logits, K, dim=-1)

    def triton_impl():
        return _select_topk(logits, K)

    def cutedsl_impl():
        return _select_topk_cutedsl(logits, K)

    def quack_impl():
        return quack_topk(logits, K)

    # numeric sanity
    vals_ref, _ = torch_impl()
    for name, fn in (
        ("triton", triton_impl),
        ("cutedsl", cutedsl_impl),
        ("quack", quack_impl),
    ):
        vals, _ = fn()
        max_diff = (vals.float() - vals_ref.float()).abs().max().item()
        print(f"[topk numeric vs torch] {name:7s} max |Δvals|={max_diff:.3e}")

    results: list[TopKResult] = []
    results.append(TopKResult("torch", _bench(torch_impl)))
    results.append(TopKResult("triton", _bench(triton_impl)))
    results.append(TopKResult("cutedsl", _bench(cutedsl_impl)))
    results.append(TopKResult("quack", _bench(quack_impl)))

    base = results[0].ms
    print("\nTopK benchmark (M=4096, N=1024, K=32, dtype=bfloat16)")
    for r in results:
        speedup = base / r.ms if r.ms > 0 else float("inf")
        print(f"  {r.name:7s}: {r.ms:.4f} ms  (speedup vs torch: {speedup:.2f}x)")


def benchmark_sampling() -> None:
    assert torch.cuda.is_available(), "CUDA is required"
    batch, vocab, top_k = 32, 32000, 40
    temperature = 1.0
    torch.manual_seed(0)
    logits = torch.randn(batch, vocab, device="cuda", dtype=torch.float32)
    gen = torch.Generator(device="cuda").manual_seed(123)

    def baseline():
        # Torch reference: top-k + softmax + multinomial or full softmax
        if top_k < vocab:
            values, indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(values / temperature, dim=-1)
            draws = torch.multinomial(probs, num_samples=1, generator=gen)
            return indices.gather(-1, draws).squeeze(-1)
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)

    def triton_impl():
        return sample_logits(
            logits,
            top_k=top_k,
            temperature=temperature,
            generator=gen,
            num_samples=1,
            backend="triton",
        )

    def cutedsl_impl():
        return sample_logits(
            logits,
            top_k=top_k,
            temperature=temperature,
            generator=gen,
            num_samples=1,
            backend="cutedsl",
        )

    results: list[SamplingResult] = []
    results.append(SamplingResult("torch", _bench(baseline)))
    results.append(SamplingResult("triton", _bench(triton_impl)))
    results.append(SamplingResult("cutedsl", _bench(cutedsl_impl)))

    base = results[0].ms
    print("\nSampling benchmark (batch=32, vocab=32000, top_k=40, temp=1.0)")
    for r in results:
        speedup = base / r.ms if r.ms > 0 else float("inf")
        print(f"  {r.name:7s}: {r.ms:.4f} ms  (speedup vs torch: {speedup:.2f}x)")


if __name__ == "__main__":
    if not torch.cuda.is_available():  # pragma: no cover
        raise SystemExit("CUDA GPU is required for these benchmarks")
    numeric_check_topk()
    numeric_check_sampling()
    benchmark_topk()
    benchmark_sampling()
