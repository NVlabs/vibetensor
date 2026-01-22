# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import statistics
import sys
import time
from typing import Callable, Dict, Tuple

import torch  # type: ignore[import]

from .kernel import cutedsl_rmsnorm, is_cutedsl_available, RMSNorm

DEFAULT_QUACK_PATH = "/workspace/terry/nano-cursor/tmp/quack"


def _time_backend(
    fn: Callable[[], torch.Tensor], warmup: int, iters: int
) -> Tuple[float, torch.Tensor]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    result: torch.Tensor | None = None
    for _ in range(iters):
        result = fn()
    torch.cuda.synchronize()
    assert result is not None
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(iters, 1)
    return elapsed_ms, result


def _prepare_inputs(
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    *,
    device: torch.device,
    use_weight: bool,
    use_bias: bool,
    use_residual: bool,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    torch.manual_seed(0)
    x = torch.randn(rows, hidden, device=device, dtype=dtype)
    weight = (
        torch.randn(hidden, device=device, dtype=torch.float32) if use_weight else None
    )
    bias = torch.randn(hidden, device=device, dtype=dtype) if use_bias else None
    residual = (
        torch.randn(rows, hidden, device=device, dtype=dtype) if use_residual else None
    )
    return x, weight, bias, residual


def _make_torch_backend(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    *,
    eps: float,
) -> Callable[[], torch.Tensor]:
    hidden = x.shape[-1]

    def _run() -> torch.Tensor:
        out = torch.nn.functional.rms_norm(x, (hidden,), weight=weight, eps=eps)
        if bias is not None:
            out = out + bias
        if residual is not None:
            out = out + residual
        return out

    return _run


def _make_triton_backend(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    *,
    eps: float,
) -> Callable[[], torch.Tensor]:
    module = RMSNorm(
        x.shape[-1],
        eps=eps,
        learnable_gamma=weight is not None,
        dtype=x.dtype,
        device=x.device,
    )
    if weight is not None:
        with torch.no_grad():
            module.gamma.copy_(weight)

    def _run() -> torch.Tensor:
        return module(x)

    return _run


def _make_cutedsl_backend(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    *,
    eps: float,
) -> Callable[[], torch.Tensor]:
    if not is_cutedsl_available():  # pragma: no cover
        raise RuntimeError("CuTeDSL RMSNorm backend is not available")

    def _run() -> torch.Tensor:
        return cutedsl_rmsnorm(
            x,
            weight,
            bias=bias,
            residual=residual,
            eps=eps,
        )

    return _run


def _make_quack_backend(
    quack_module,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    *,
    eps: float,
) -> Callable[[], torch.Tensor]:
    def _run() -> torch.Tensor:
        return quack_module.rmsnorm(
            x,
            weight,
            bias=bias,
            residual=residual,
            eps=eps,
        )

    return _run


def _format_backend_results(
    results: Dict[str, Tuple[float, torch.Tensor]], reference: str
) -> None:
    ref_time, ref_output = results[reference]
    print("Backend Results")
    print("==============")
    print(f"reference backend : {reference}")
    print(f"reference time    : {ref_time:.4f} ms")
    print()
    header = (
        f"{'backend':<10}{'mean_ms':>12}{'speedup_vs_ref':>18}{'max_diff_vs_ref':>20}"
    )
    print(header)
    print("-" * len(header))
    for name, (elapsed, output) in results.items():
        diff = (output - ref_output).abs().max().item()
        speedup = ref_time / elapsed if elapsed > 0 else float("inf")
        print(f"{name:<10}{elapsed:>12.4f}{speedup:>18.3f}{diff:>20.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RMSNorm backends")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
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
        "--no-weight", action="store_true", help="disable learnable weight"
    )
    parser.add_argument("--bias", action="store_true", help="add bias term")
    parser.add_argument(
        "--residual", action="store_true", help="add residual connection"
    )
    parser.add_argument(
        "--quack-path",
        type=str,
        default=DEFAULT_QUACK_PATH,
        help="Path to the quack repository for reference implementation",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="quack",
        choices=["torch", "triton", "cutedsl", "quack"],
        help="Backend used as numeric reference",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError("CUDA is required for RMSNorm benchmarking")

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")

    x, weight, bias, residual = _prepare_inputs(
        args.rows,
        args.hidden,
        dtype,
        device=device,
        use_weight=not args.no_weight,
        use_bias=args.bias,
        use_residual=args.residual,
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
        try:
            quack_module = importlib.import_module("quack.rmsnorm")
        except ModuleNotFoundError:
            backends_requested = [b for b in backends_requested if b != "quack"]
            if not backends_requested:
                backends_requested = ["torch"]
            if args.reference == "quack":
                args.reference = "torch" if "torch" in backends_requested else backends_requested[0]

    backend_fns: Dict[str, Callable[[], torch.Tensor]] = {}
    for backend in backends_requested:
        if backend == "torch":
            backend_fns[backend] = _make_torch_backend(
                x, weight, bias, residual, eps=args.eps
            )
        elif backend == "triton":
            backend_fns[backend] = _make_triton_backend(x, weight, eps=args.eps)
        elif backend == "cutedsl":
            backend_fns[backend] = _make_cutedsl_backend(
                x, weight, bias, residual, eps=args.eps
            )
        elif backend == "quack":
            if quack_module is None:  # pragma: no cover
                raise RuntimeError("Quack module failed to import")
            backend_fns[backend] = _make_quack_backend(
                quack_module, x, weight, bias, residual, eps=args.eps
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    results: Dict[str, Tuple[float, torch.Tensor]] = {}
    for name, fn in backend_fns.items():
        elapsed, output = _time_backend(fn, args.warmup, args.iters)
        results[name] = (elapsed, output.detach().cpu())

    if args.reference not in results:
        raise RuntimeError(f"Reference backend {args.reference} did not run")

    print("RMSNorm Benchmark")
    print("==================")
    print(f"rows             : {args.rows}")
    print(f"hidden dim       : {args.hidden}")
    print(f"dtype            : {args.dtype}")
    print(f"weight           : {not args.no_weight}")
    print(f"bias             : {args.bias}")
    print(f"residual         : {args.residual}")
    print(f"eps              : {args.eps}")
    print(f"warmup           : {args.warmup}")
    print(f"iters            : {args.iters}")
    print()

    _format_backend_results(results, args.reference)


if __name__ == "__main__":
    main()
