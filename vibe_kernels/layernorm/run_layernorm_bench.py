# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import torch

from .kernel import layernorm


def ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")


def bench(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


@dataclass
class Config:
    rows: int
    hidden: int
    dtype: torch.dtype
    bias: bool


def run(configs: Iterable[Config]) -> list[dict[str, float | int | str | bool]]:
    ensure_cuda()
    torch.manual_seed(0)
    results = []
    device = torch.device("cuda")

    for cfg in configs:
        x = torch.randn(cfg.rows, cfg.hidden, device=device, dtype=cfg.dtype)
        weight = torch.randn(cfg.hidden, device=device, dtype=torch.float32)
        bias = (
            torch.randn(cfg.hidden, device=device, dtype=torch.float32)
            if cfg.bias
            else None
        )
        grad = torch.randn_like(x)

        def forward_cutedsl() -> None:
            with torch.no_grad():
                layernorm(x, weight, bias=bias, eps=1e-6)

        def forward_torch() -> None:
            with torch.no_grad():
                torch.nn.functional.layer_norm(
                    x,
                    (cfg.hidden,),
                    weight.to(x.dtype),
                    bias.to(x.dtype) if bias is not None else None,
                    1e-6,
                )

        def backward_cutedsl() -> None:
            xc = x.detach().clone().requires_grad_(True)
            wc = weight.detach().clone().requires_grad_(True)
            bc = (
                bias.detach().clone().requires_grad_(True) if bias is not None else None
            )
            out = layernorm(xc, wc, bias=bc, eps=1e-6)
            out.backward(grad)

        def backward_torch() -> None:
            xt = x.detach().clone().requires_grad_(True)
            wt = weight.detach().clone().requires_grad_(True)
            bt = (
                bias.detach().clone().requires_grad_(True) if bias is not None else None
            )
            out = torch.nn.functional.layer_norm(
                xt,
                (cfg.hidden,),
                wt.to(xt.dtype),
                bt.to(xt.dtype) if bt is not None else None,
                1e-6,
            )
            out.backward(grad)

        results.append(
            {
                "rows": cfg.rows,
                "hidden": cfg.hidden,
                "dtype": str(cfg.dtype).split(".")[-1],
                "bias": cfg.bias,
                "forward_torch_ms": bench(forward_torch),
                "forward_cutedsl_ms": bench(forward_cutedsl),
                "backward_torch_ms": bench(backward_torch),
                "backward_cutedsl_ms": bench(backward_cutedsl),
            }
        )

    return results


def main() -> None:
    cfgs = [
        Config(4096, 8192, torch.float16, True),
        Config(4096, 8192, torch.bfloat16, False),
        Config(4096, 16384, torch.float16, False),
    ]
    for row in run(cfgs):
        print(
            "rows={rows} hidden={hidden} dtype={dtype} bias={bias} | "
            "fwd torch={forward_torch_ms:.4f}ms cutedsl={forward_cutedsl_ms:.4f}ms | "
            "bwd torch={backward_torch_ms:.4f}ms cutedsl={backward_cutedsl_ms:.4f}ms".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
