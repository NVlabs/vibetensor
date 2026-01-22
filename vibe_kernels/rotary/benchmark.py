# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import statistics
import time
from typing import Tuple

import torch  # type: ignore[import]

from .kernel import apply_rotary_embedding


def _torch_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    head_dim_half = q.size(-1) // 2
    cos_gather = cos.reshape(-1, cos.shape[-1])[positions].to(torch.float32)
    sin_gather = sin.reshape(-1, sin.shape[-1])[positions].to(torch.float32)
    q1 = q[..., :head_dim_half].to(torch.float32)
    q2 = q[..., head_dim_half:].to(torch.float32)
    k1 = k[..., :head_dim_half].to(torch.float32)
    k2 = k[..., head_dim_half:].to(torch.float32)
    q_out1 = q1 * cos_gather + q2 * sin_gather
    q_out2 = -q1 * sin_gather + q2 * cos_gather
    k_out1 = k1 * cos_gather + k2 * sin_gather
    k_out2 = -k1 * sin_gather + k2 * cos_gather
    q_out = torch.cat([q_out1, q_out2], dim=-1).to(q.dtype)
    k_out = torch.cat([k_out1, k_out2], dim=-1).to(k.dtype)
    return q_out, k_out


def _benchmark_case(
    batch: int,
    heads: int,
    seqlen: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> Tuple[float, float, float, bool, bool, float, float]:
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seqlen, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    cos = torch.randn(seqlen, head_dim // 2, device=device, dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    positions = torch.arange(seqlen, device=device, dtype=torch.int32)
    positions = positions.unsqueeze(0).unsqueeze(0).expand(batch, heads, seqlen)

    def _baseline():
        return _torch_rotary(q, k, cos, sin, positions)

    def _triton():
        return apply_rotary_embedding(
            q.clone(), k.clone(), cos, sin, positions, backend="triton"
        )

    def _cutedsl():
        return apply_rotary_embedding(
            q.clone(), k.clone(), cos, sin, positions, backend="cutedsl"
        )

    def _time(fn) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0

    # --- Baseline ---
    print("Benchmarking Baseline...")
    for _ in range(warmup):
        _baseline()
    torch.cuda.synchronize()
    baseline_samples = [_time(_baseline) for _ in range(iters)]
    baseline_mean = statistics.fmean(baseline_samples)

    # --- Triton ---
    print("Benchmarking Triton...")
    for _ in range(warmup):
        _triton()
    torch.cuda.synchronize()
    try:
        triton_samples = [_time(_triton) for _ in range(iters)]
        triton_mean = statistics.fmean(triton_samples)
    except RuntimeError as e:
        print(f"Triton failed: {e}")
        triton_mean = float("inf")

    # --- CuTeDSL ---
    print("Benchmarking CuTeDSL...")
    for _ in range(warmup):
        _cutedsl()
    torch.cuda.synchronize()
    try:
        cutedsl_samples = [_time(_cutedsl) for _ in range(iters)]
        cutedsl_mean = statistics.fmean(cutedsl_samples)
    except RuntimeError as e:
        print(f"CuTeDSL failed: {e}")
        cutedsl_mean = float("inf")

    # --- Verification ---
    print("Verifying...")
    ref_q, ref_k = _baseline()

    def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
        return (a - b).abs().max().item()

    try:
        tri_q, tri_k = _triton()
        tri_diff = max(_max_diff(ref_q, tri_q), _max_diff(ref_k, tri_k))
        tri_allclose = torch.allclose(
            ref_q, tri_q, atol=1e-2, rtol=0
        ) and torch.allclose(ref_k, tri_k, atol=1e-2, rtol=0)
    except Exception as e:
        print(f"Triton verification failed: {e}")
        tri_diff = -1.0
        tri_allclose = False

    try:
        cut_q, cut_k = _cutedsl()
        cut_diff = max(_max_diff(ref_q, cut_q), _max_diff(ref_k, cut_k))
        cut_allclose = torch.allclose(
            ref_q, cut_q, atol=1e-2, rtol=0
        ) and torch.allclose(ref_k, cut_k, atol=1e-2, rtol=0)
    except Exception as e:
        print(f"CuTeDSL verification failed: {e}")
        cut_diff = -1.0
        cut_allclose = False

    return (
        baseline_mean,
        triton_mean,
        cutedsl_mean,
        tri_allclose,
        cut_allclose,
        tri_diff,
        cut_diff,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton and CuTeDSL rotary embedding kernels"
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError("CUDA is required for rotary benchmark")

    torch.cuda.set_device(0)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)

    (
        baseline_mean,
        triton_mean,
        cutedsl_mean,
        tri_allclose,
        cut_allclose,
        tri_diff,
        cut_diff,
    ) = _benchmark_case(
        args.batch,
        args.heads,
        args.seqlen,
        args.headdim,
        device,
        dtype,
        args.warmup,
        args.iters,
    )

    print("Rotary Embedding Benchmark")
    print("==========================")
    print(f"batch            : {args.batch}")
    print(f"heads            : {args.heads}")
    print(f"sequence length  : {args.seqlen}")
    print(f"head dim         : {args.headdim}")
    print(f"dtype            : {args.dtype}")
    print(f"baseline mean    : {baseline_mean:.4f} ms")
    print(f"triton mean      : {triton_mean:.4f} ms")
    print(f"cutedsl mean     : {cutedsl_mean:.4f} ms")

    if triton_mean > 0 and triton_mean != float("inf"):
        print(f"triton speedup   : {baseline_mean / triton_mean:.3f}x")
    else:
        print("triton speedup   : N/A")

    if cutedsl_mean > 0 and cutedsl_mean != float("inf"):
        print(f"cutedsl speedup  : {baseline_mean / cutedsl_mean:.3f}x")
    else:
        print("cutedsl speedup  : N/A")

    if cutedsl_mean > 0 and triton_mean > 0 and triton_mean != float("inf"):
        print(f"cutedsl vs triton: {triton_mean / cutedsl_mean:.3f}x")
    else:
        print("cutedsl vs triton: N/A")

    print(f"triton max |diff|: {tri_diff:.6f}, allclose={tri_allclose}")
    print(f"cutedsl max |diff|: {cut_diff:.6f}, allclose={cut_allclose}")


if __name__ == "__main__":
    main()
