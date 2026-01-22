# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import statistics
import sys
import time
from typing import Callable, cast, Dict, Tuple

import torch

from .impl import triton_impl
from .kernel import (  # type: ignore[import]
    cutedsl_layernorm,
    is_cutedsl_available,
    layernorm_mean_ref,
    layernorm_ref,
    layernorm_rstd_ref,
)

DEFAULT_QUACK_PATH = "/workspace/terry/nano-cursor/tmp/quack"

BackendResult = Tuple[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def _time_backend(
    fn: Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    warmup: int,
    iters: int,
) -> BackendResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    capture: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    for _ in range(iters):
        capture = fn()
    torch.cuda.synchronize()
    assert capture is not None
    out, rstd, mean = capture
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(iters, 1)
    return elapsed_ms, (out.detach().cpu(), rstd.detach().cpu(), mean.detach().cpu())


def _prepare_inputs(
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    *,
    device: torch.device,
    use_bias: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    torch.manual_seed(0)
    x = torch.randn(rows, hidden, device=device, dtype=dtype)
    weight = torch.randn(hidden, device=device, dtype=torch.float32)
    bias = torch.randn(hidden, device=device, dtype=dtype) if use_bias else None
    return x, weight, bias


def _make_torch_backend(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    eps: float,
) -> Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    def _run() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight_cast = weight.to(x.dtype)
        bias_cast = bias.to(x.dtype) if bias is not None else None
        out = torch.nn.functional.layer_norm(
            x, weight_cast.shape, weight_cast, bias_cast, eps
        )
        rstd = layernorm_rstd_ref(x, eps=eps)
        mean = layernorm_mean_ref(x)
        return out, rstd, mean

    return _run


def _make_cutedsl_backend(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    eps: float,
) -> Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not is_cutedsl_available():  # pragma: no cover
        raise RuntimeError("CuTeDSL LayerNorm backend is not available")

    def _run() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out_t = cast(
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            cutedsl_layernorm(
                x,
                weight,
                bias=bias,
                eps=eps,
                return_rstd=True,
                return_mean=True,
            ),
        )
        out, rstd, mean = out_t
        return out, rstd, mean

    return _run


def _make_triton_backend(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    eps: float,
) -> Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    def _run() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, rstd, mean = triton_impl.layernorm(
            x,
            weight,
            bias=bias,
            eps=eps,
            return_rstd=True,
            return_mean=True,
        )
        return out, rstd, mean

    return _run


def _make_quack_backend(
    module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    eps: float,
) -> Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    def _run() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out_t = cast(
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            module.layernorm(
                x,
                weight,
                eps=eps,
                return_rstd=True,
                return_mean=True,
            ),
        )
        out, rstd, mean = out_t
        if bias is not None:
            bias_to_add = bias.to(out.dtype)
            out = out + bias_to_add
        return out, rstd, mean

    return _run


def _format_backend_results(results: Dict[str, BackendResult], reference: str) -> None:
    ref_time, (ref_out, ref_rstd, ref_mean) = results[reference]
    print("Backend Results")
    print("==============")
    print(f"reference backend : {reference}")
    print(f"reference time    : {ref_time:.4f} ms")
    print()
    header = f"{'backend':<10}{'mean_ms':>12}{'speedup_vs_ref':>18}{'max|Δ|_out':>16}{'max|Δ|_rstd':>16}{'max|Δ|_mean':>16}"
    print(header)
    print("-" * len(header))
    for name, (elapsed, (out, rstd, mean)) in results.items():
        diff_out = (out - ref_out).abs().max().item()
        diff_rstd = (rstd - ref_rstd).abs().max().item()
        diff_mean = (mean - ref_mean).abs().max().item()
        speedup = ref_time / elapsed if elapsed > 0 else float("inf")
        print(
            f"{name:<10}{elapsed:>12.4f}{speedup:>18.3f}{diff_out:>16.3e}{diff_rstd:>16.3e}{diff_mean:>16.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LayerNorm backends")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="torch,triton,cutedsl,quack",
        help="Comma-separated list of backends to benchmark",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Enable an additive bias vector",
    )
    parser.add_argument(
        "--quack-path",
        type=str,
        default=DEFAULT_QUACK_PATH,
        help="Path to the QQuack repository for reference implementation",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="quack",
        choices=["torch", "cutedsl", "quack"],
        help="Backend used as numeric reference",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError("CUDA is required for LayerNorm benchmarking")

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    x, weight, bias = _prepare_inputs(
        args.rows,
        args.hidden,
        dtype,
        device=device,
        use_bias=args.bias,
    )

    backends_requested = [
        name.strip().lower() for name in args.backends.split(",") if name.strip()
    ]
    if args.reference not in backends_requested:
        backends_requested.append(args.reference)

    quack_module = None
    if "quack" in backends_requested:
        if args.quack_path not in sys.path:
            sys.path.insert(0, args.quack_path)
        quack_module = importlib.import_module("quack.layernorm")

    backend_fns: Dict[
        str, Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = {}
    for backend in backends_requested:
        if backend == "torch":
            backend_fns[backend] = _make_torch_backend(x, weight, bias, eps=args.eps)
        elif backend == "cutedsl":
            backend_fns[backend] = _make_cutedsl_backend(x, weight, bias, eps=args.eps)
        elif backend == "triton":
            backend_fns[backend] = _make_triton_backend(x, weight, bias, eps=args.eps)
        elif backend == "quack":
            if quack_module is None:  # pragma: no cover
                raise RuntimeError("Quack module failed to import")
            backend_fns[backend] = _make_quack_backend(
                quack_module, x, weight, bias, eps=args.eps
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    results: Dict[str, BackendResult] = {}
    for name, fn in backend_fns.items():
        elapsed, outputs = _time_backend(fn, args.warmup, args.iters)
        results[name] = (elapsed, outputs)

    if args.reference not in results:
        raise RuntimeError(f"Reference backend {args.reference} did not run")

    print("LayerNorm Benchmark")
    print("===================")
    print(f"rows             : {args.rows}")
    print(f"hidden dim       : {args.hidden}")
    print(f"dtype            : {args.dtype}")
    print(f"bias             : {args.bias}")
    print(f"eps              : {args.eps}")
    print(f"warmup           : {args.warmup}")
    print(f"iters            : {args.iters}")
    print()

    _format_backend_results(results, args.reference)


if __name__ == "__main__":
    main()
