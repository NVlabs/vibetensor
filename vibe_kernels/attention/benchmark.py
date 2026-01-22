# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention kernel benchmark: PyTorch SDPA vs Triton vs VBT-Triton.

This benchmark compares three implementations:
1. PyTorch SDPA (scaled_dot_product_attention) - baseline
2. Triton kernel (kernel.py) - uses PyTorch tensors with Triton launcher
3. VBT-Triton (vbt_native.py) - pure VibeTensor, no PyTorch dependency

Note: VBT-Triton currently only supports float32.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Tuple, Optional, Dict, Any

import torch


def _sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """PyTorch SDPA baseline."""
    batch, hq, seqlen, dim = q.shape
    hk = k.shape[1]
    group = hq // hk
    qg = (
        q.view(batch, hk, group, seqlen, dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * group, hk, seqlen, dim)
    )
    kg = (
        k.unsqueeze(1).repeat(1, group, 1, 1, 1).reshape(batch * group, hk, seqlen, dim)
    )
    vg = (
        v.unsqueeze(1).repeat(1, group, 1, 1, 1).reshape(batch * group, hk, seqlen, dim)
    )
    out = torch.nn.functional.scaled_dot_product_attention(
        qg,
        kg,
        vg,
        is_causal=causal,
        dropout_p=0.0,
        scale=sm_scale,
    )
    out = (
        out.view(batch, group, hk, seqlen, dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, hq, seqlen, dim)
    )
    return out


def _benchmark_case(
    batch: int,
    heads: int,
    kv_heads: int,
    seqlen: int,
    head_dim: int,
    dtype: torch.dtype,
    causal: bool,
    warmup: int,
    iters: int,
    include_vbt: bool = True,
) -> Dict[str, Any]:
    """Benchmark all three implementations."""
    from .kernel import fused_attention
    
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = torch.randn(batch, heads, seqlen, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, kv_heads, seqlen, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    sm_scale = 1.0 / (head_dim**0.5)

    results: Dict[str, Any] = {
        "config": {
            "batch": batch,
            "heads": heads,
            "kv_heads": kv_heads,
            "seqlen": seqlen,
            "head_dim": head_dim,
            "dtype": str(dtype).split(".")[-1],
            "causal": causal,
        }
    }

    def _time(fn) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0

    # PyTorch SDPA
    def _pytorch():
        return _sdpa_reference(q, k, v, causal=causal, sm_scale=sm_scale)

    for _ in range(warmup):
        _pytorch()
    pytorch_samples = [_time(_pytorch) for _ in range(iters)]
    results["pytorch_ms"] = statistics.fmean(pytorch_samples)
    ref_out = _pytorch()

    # Triton kernel
    def _triton():
        return fused_attention(q, k, v, causal=causal, sm_scale=sm_scale)

    for _ in range(warmup):
        _triton()
    triton_samples = [_time(_triton) for _ in range(iters)]
    results["triton_ms"] = statistics.fmean(triton_samples)
    triton_out = _triton()
    
    triton_diff = (ref_out - triton_out).abs().max().item()
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    results["triton_diff"] = triton_diff
    results["triton_allclose"] = torch.allclose(ref_out, triton_out, atol=tol, rtol=0)

    # VBT-Triton (only float32, same heads)
    vbt_supported = (dtype == torch.float32 and heads == kv_heads and 
                     head_dim in {16, 32, 64, 128})
    
    if include_vbt and vbt_supported:
        try:
            from . import vbt_native
            import vibetensor.torch as vt
            
            q_vt = vt.from_dlpack(q)
            k_vt = vt.from_dlpack(k)
            v_vt = vt.from_dlpack(v)
            
            def _vbt():
                return vbt_native.attention(q_vt, k_vt, v_vt, causal=causal, sm_scale=sm_scale)
            
            for _ in range(warmup):
                _vbt()
            vbt_samples = [_time(_vbt) for _ in range(iters)]
            results["vbt_ms"] = statistics.fmean(vbt_samples)
            
            vbt_out = torch.from_dlpack(_vbt())
            vbt_diff = (ref_out - vbt_out).abs().max().item()
            results["vbt_diff"] = vbt_diff
            results["vbt_allclose"] = torch.allclose(ref_out, vbt_out, atol=1e-2, rtol=1e-3)
        except Exception as e:
            results["vbt_error"] = str(e)
    elif include_vbt:
        results["vbt_skip"] = f"VBT requires float32, same heads, head_dim in [16,32,64,128]"

    return results


def _benchmark_backward(
    batch: int,
    heads: int,
    kv_heads: int,
    seqlen: int,
    head_dim: int,
    dtype: torch.dtype,
    causal: bool,
    warmup: int,
    iters: int,
    include_vbt: bool = True,
) -> Dict[str, Any]:
    """Benchmark backward pass for all three implementations."""
    from .kernel import fused_attention
    
    torch.manual_seed(0)
    device = torch.device("cuda")
    
    sm_scale = 1.0 / (head_dim**0.5)

    q_base = torch.randn(batch, heads, seqlen, head_dim, device=device, dtype=dtype)
    k_base = torch.randn(batch, kv_heads, seqlen, head_dim, device=device, dtype=dtype)
    v_base = torch.randn(batch, kv_heads, seqlen, head_dim, device=device, dtype=dtype)
    grad_out = torch.randn(batch, heads, seqlen, head_dim, device=device, dtype=dtype)

    results: Dict[str, Any] = {
        "config": {
            "batch": batch, "heads": heads, "kv_heads": kv_heads,
            "seqlen": seqlen, "head_dim": head_dim,
            "dtype": str(dtype).split(".")[-1], "causal": causal,
        }
    }

    def _time(fn) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0

    # PyTorch backward
    def _pytorch_bwd():
        q = q_base.clone().requires_grad_(True)
        k = k_base.clone().requires_grad_(True)
        v = v_base.clone().requires_grad_(True)
        out = _sdpa_reference(q, k, v, causal=causal, sm_scale=sm_scale)
        out.backward(grad_out)
        return q.grad, k.grad, v.grad

    for _ in range(warmup):
        _pytorch_bwd()
    pytorch_samples = [_time(_pytorch_bwd) for _ in range(iters)]
    results["pytorch_ms"] = statistics.fmean(pytorch_samples)
    ref_dq, ref_dk, ref_dv = _pytorch_bwd()

    # Triton backward
    def _triton_bwd():
        q = q_base.clone().requires_grad_(True)
        k = k_base.clone().requires_grad_(True)
        v = v_base.clone().requires_grad_(True)
        out = fused_attention(q, k, v, causal=causal, sm_scale=sm_scale)
        out.backward(grad_out)
        return q.grad, k.grad, v.grad

    for _ in range(warmup):
        _triton_bwd()
    triton_samples = [_time(_triton_bwd) for _ in range(iters)]
    results["triton_ms"] = statistics.fmean(triton_samples)
    triton_dq, triton_dk, triton_dv = _triton_bwd()
    
    tol = 5e-2 if dtype == torch.bfloat16 else 2e-2
    results["triton_diff"] = max(
        (ref_dq - triton_dq).abs().max().item(),
        (ref_dk - triton_dk).abs().max().item(),
        (ref_dv - triton_dv).abs().max().item()
    )
    results["triton_allclose"] = (
        torch.allclose(ref_dq, triton_dq, atol=tol, rtol=0) and
        torch.allclose(ref_dk, triton_dk, atol=tol, rtol=0) and
        torch.allclose(ref_dv, triton_dv, atol=tol, rtol=0)
    )

    # VBT backward (float32 only, same heads)
    vbt_supported = (dtype == torch.float32 and heads == kv_heads and 
                     head_dim in {16, 32, 64, 128})
    
    if include_vbt and vbt_supported:
        try:
            from . import vbt_native
            import vibetensor.torch as vt
            
            q_vt = vt.from_dlpack(q_base)
            k_vt = vt.from_dlpack(k_base)
            v_vt = vt.from_dlpack(v_base)
            grad_out_vt = vt.from_dlpack(grad_out)
            
            def _vbt_bwd():
                out_vt, lse_vt = vbt_native.attention_with_lse(
                    q_vt, k_vt, v_vt, causal=causal, sm_scale=sm_scale
                )
                dq, dk, dv = vbt_native.attention_backward(
                    grad_out_vt, q_vt, k_vt, v_vt, out_vt, lse_vt,
                    causal=causal, sm_scale=sm_scale
                )
                return dq, dk, dv
            
            for _ in range(warmup):
                _vbt_bwd()
            vbt_samples = [_time(_vbt_bwd) for _ in range(iters)]
            results["vbt_ms"] = statistics.fmean(vbt_samples)
            
            vbt_dq, vbt_dk, vbt_dv = _vbt_bwd()
            vbt_dq_t = torch.from_dlpack(vbt_dq)
            vbt_dk_t = torch.from_dlpack(vbt_dk)
            vbt_dv_t = torch.from_dlpack(vbt_dv)
            
            results["vbt_diff"] = max(
                (ref_dq - vbt_dq_t).abs().max().item(),
                (ref_dk - vbt_dk_t).abs().max().item(),
                (ref_dv - vbt_dv_t).abs().max().item()
            )
            results["vbt_allclose"] = (
                torch.allclose(ref_dq, vbt_dq_t, atol=2e-2, rtol=1e-3) and
                torch.allclose(ref_dk, vbt_dk_t, atol=2e-2, rtol=1e-3) and
                torch.allclose(ref_dv, vbt_dv_t, atol=2e-2, rtol=1e-3)
            )
        except Exception as e:
            results["vbt_error"] = str(e)
    elif include_vbt:
        results["vbt_skip"] = "VBT requires float32, same heads, head_dim in [16,32,64,128]"

    return results


def run_benchmark_suite(warmup: int = 10, iters: int = 50) -> list:
    """Run benchmark suite with standard problem sizes."""
    configs = [
        # (batch, heads, kv_heads, seqlen, head_dim, dtype, causal)
        # Small - testing
        (1, 8, 8, 512, 64, torch.float32, True),
        (1, 8, 8, 512, 64, torch.float16, True),
        # Medium - typical inference
        (4, 32, 32, 1024, 64, torch.float32, True),
        (4, 32, 32, 1024, 64, torch.float16, True),
        (4, 32, 32, 1024, 64, torch.bfloat16, True),
        # Large - training
        (8, 32, 32, 2048, 128, torch.float32, True),
        (8, 32, 32, 2048, 128, torch.float16, True),
        (8, 32, 32, 2048, 128, torch.bfloat16, True),
        # GQA configs
        (4, 32, 8, 2048, 128, torch.float16, True),
        (4, 32, 8, 2048, 128, torch.bfloat16, True),
    ]
    
    results = []
    for batch, heads, kv_heads, seqlen, head_dim, dtype, causal in configs:
        try:
            result = _benchmark_case(
                batch, heads, kv_heads, seqlen, head_dim, dtype, causal, warmup, iters
            )
            results.append(result)
        except Exception as e:
            results.append({
                "config": {
                    "batch": batch, "heads": heads, "kv_heads": kv_heads,
                    "seqlen": seqlen, "head_dim": head_dim,
                    "dtype": str(dtype).split(".")[-1], "causal": causal,
                },
                "error": str(e)
            })
    return results


def print_results_table(results: list) -> None:
    """Print results as a markdown table."""
    print("\n## Attention Benchmark Results\n")
    print("| Batch | Heads | KV | SeqLen | Dim | Dtype | PyTorch (ms) | Triton (ms) | VBT (ms) | Triton Speedup | VBT Speedup | Numerics |")
    print("|-------|-------|-----|--------|-----|-------|--------------|-------------|----------|----------------|-------------|----------|")
    
    for r in results:
        if "error" in r:
            c = r["config"]
            print(f"| {c['batch']} | {c['heads']} | {c['kv_heads']} | {c['seqlen']} | {c['head_dim']} | {c['dtype']} | ERROR | - | - | - | - | {r['error'][:20]} |")
            continue
            
        c = r["config"]
        pytorch_ms = r.get("pytorch_ms", 0)
        triton_ms = r.get("triton_ms", 0)
        vbt_ms = r.get("vbt_ms")
        
        triton_speedup = pytorch_ms / triton_ms if triton_ms > 0 else 0
        vbt_speedup = pytorch_ms / vbt_ms if vbt_ms else "-"
        vbt_str = f"{vbt_ms:.3f}" if vbt_ms else "N/A"
        vbt_speedup_str = f"{vbt_speedup:.2f}x" if isinstance(vbt_speedup, float) else vbt_speedup
        
        triton_ok = "OK" if r.get("triton_allclose") else f"DIFF:{r.get('triton_diff', 0):.2e}"
        vbt_ok = ""
        if vbt_ms:
            vbt_ok = "OK" if r.get("vbt_allclose") else f"DIFF:{r.get('vbt_diff', 0):.2e}"
        
        numerics = f"T:{triton_ok}"
        if vbt_ok:
            numerics += f" V:{vbt_ok}"
        
        print(f"| {c['batch']} | {c['heads']} | {c['kv_heads']} | {c['seqlen']} | {c['head_dim']} | {c['dtype']} | {pytorch_ms:.3f} | {triton_ms:.3f} | {vbt_str} | {triton_speedup:.2f}x | {vbt_speedup_str} | {numerics} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark attention: PyTorch vs Triton vs VBT-Triton"
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--backward", action="store_true", help="Benchmark backward pass")
    parser.add_argument("--both", action="store_true", help="Benchmark both forward and backward")
    parser.add_argument("--suite", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--no-vbt", action="store_true", help="Skip VBT-Triton benchmark")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for attention benchmark")

    if args.suite:
        results = run_benchmark_suite(args.warmup, args.iters)
        print_results_table(results)
        return

    dtype = getattr(torch, args.dtype)
    kv_heads = args.kv_heads if args.kv_heads else args.heads
    
    run_fwd = not args.backward or args.both
    run_bwd = args.backward or args.both

    if run_fwd:
        results = _benchmark_case(
            args.batch,
            args.heads,
            kv_heads,
            args.seqlen,
            args.headdim,
            dtype,
            args.causal,
            args.warmup,
            args.iters,
            include_vbt=not args.no_vbt,
        )

        print("Attention Benchmark (Forward)")
        print("=" * 50)
        c = results["config"]
        print(f"Config: batch={c['batch']}, heads={c['heads']}, kv_heads={c['kv_heads']}, "
              f"seqlen={c['seqlen']}, head_dim={c['head_dim']}, dtype={c['dtype']}, causal={c['causal']}")
        print()
        print(f"PyTorch SDPA : {results['pytorch_ms']:.4f} ms")
        print(f"Triton       : {results['triton_ms']:.4f} ms  "
              f"(speedup: {results['pytorch_ms']/results['triton_ms']:.2f}x, "
              f"diff: {results['triton_diff']:.2e}, allclose: {results['triton_allclose']})")
        
        if "vbt_ms" in results:
            print(f"VBT-Triton   : {results['vbt_ms']:.4f} ms  "
                  f"(speedup: {results['pytorch_ms']/results['vbt_ms']:.2f}x, "
                  f"diff: {results['vbt_diff']:.2e}, allclose: {results['vbt_allclose']})")
        elif "vbt_skip" in results:
            print(f"VBT-Triton   : SKIPPED ({results['vbt_skip']})")
        elif "vbt_error" in results:
            print(f"VBT-Triton   : ERROR ({results['vbt_error']})")

    if run_bwd:
        if run_fwd:
            print()
        bwd_results = _benchmark_backward(
            args.batch,
            args.heads,
            kv_heads,
            args.seqlen,
            args.headdim,
            dtype,
            args.causal,
            args.warmup,
            args.iters,
            include_vbt=not args.no_vbt,
        )

        print("Attention Benchmark (Backward)")
        print("=" * 50)
        c = bwd_results["config"]
        print(f"Config: batch={c['batch']}, heads={c['heads']}, kv_heads={c['kv_heads']}, "
              f"seqlen={c['seqlen']}, head_dim={c['head_dim']}, dtype={c['dtype']}, causal={c['causal']}")
        print()
        print(f"PyTorch SDPA : {bwd_results['pytorch_ms']:.4f} ms")
        print(f"Triton       : {bwd_results['triton_ms']:.4f} ms  "
              f"(speedup: {bwd_results['pytorch_ms']/bwd_results['triton_ms']:.2f}x, "
              f"diff: {bwd_results['triton_diff']:.2e}, allclose: {bwd_results['triton_allclose']})")
        
        if "vbt_ms" in bwd_results:
            print(f"VBT-Triton   : {bwd_results['vbt_ms']:.4f} ms  "
                  f"(speedup: {bwd_results['pytorch_ms']/bwd_results['vbt_ms']:.2f}x, "
                  f"diff: {bwd_results['vbt_diff']:.2e}, allclose: {bwd_results['vbt_allclose']})")
        elif "vbt_skip" in bwd_results:
            print(f"VBT-Triton   : SKIPPED ({bwd_results['vbt_skip']})")
        elif "vbt_error" in bwd_results:
            print(f"VBT-Triton   : ERROR ({bwd_results['vbt_error']})")


if __name__ == "__main__":
    main()
