# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark script comparing TritonAdamW against torch.optim.AdamW."""

from __future__ import annotations

import argparse
import time

import torch

from vibe_kernels.optim import TritonAdamW  # type: ignore[import]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdamW optimizer benchmark")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=[4096, 1024],
        help="Parameter matrix shape",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Parameter dtype",
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of benchmark steps"
    )
    parser.add_argument(
        "--warmup", type=int, default=20, help="Warmup steps before timing"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay value"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--betas", type=float, nargs=2, default=[0.9, 0.999], help="AdamW betas"
    )
    return parser.parse_args()


def _dtype_from_string(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _time_steps(
    optimizer: torch.optim.Optimizer, param: torch.Tensor, steps: int, warmup: int
) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        optimizer.step()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(steps):
        optimizer.step()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / steps * 1000.0  # ms/step


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    dtype = _dtype_from_string(args.dtype)

    torch.manual_seed(0)
    param_ref = torch.randn(*args.shape, device=device, dtype=dtype, requires_grad=True)
    param_new = param_ref.detach().clone().requires_grad_(True)
    grad = torch.randn_like(param_ref)
    param_ref.grad = grad.clone()
    param_new.grad = grad.clone()

    lr = args.lr
    betas = tuple(args.betas)  # type: ignore[assignment]
    eps = 1e-8
    weight_decay = args.weight_decay

    opt_ref = torch.optim.AdamW(
        [param_ref], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )
    opt_new = TritonAdamW(
        [param_new], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )

    ref_time = _time_steps(opt_ref, param_ref, args.steps, args.warmup)
    new_time = _time_steps(opt_new, param_new, args.steps, args.warmup)

    # Relax tolerances for bfloat16
    atol = 1e-5
    rtol = 5e-4
    if dtype == torch.bfloat16:
        atol = 1e-2
        rtol = 1e-2

    try:
        torch.testing.assert_close(param_ref, param_new, atol=atol, rtol=rtol)
        print("Validation: PASSED")
    except AssertionError as e:
        print(f"Validation: FAILED (atol={atol}, rtol={rtol})")
        print(e)
        # Continue to print results even if validation fails
        pass

    print("AdamW Benchmark (device: cuda)")
    print(f"Shape: {tuple(args.shape)}, dtype: {args.dtype}")
    print(f"PyTorch AdamW: {ref_time:.4f} ms/step")
    print(f"TritonAdamW : {new_time:.4f} ms/step")
    print(f"Speedup      : {ref_time / new_time:.2f}x")


if __name__ == "__main__":
    main()
