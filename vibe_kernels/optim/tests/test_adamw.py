# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vibe_kernels.optim import TritonAdamW, TritonDistAdamW  # type: ignore[import]


def _setup_parameters(
    shape: tuple[int, ...] = (8, 8),
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
):
    device = device or torch.device("cpu")
    torch.manual_seed(0)
    param_ref = torch.randn(*shape, dtype=dtype, device=device, requires_grad=True)
    param_new = param_ref.detach().clone().requires_grad_(True)
    grad = torch.randn_like(param_ref)
    param_ref.grad = grad.clone()
    param_new.grad = grad.clone()
    return param_ref, param_new


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _distributed_adamw_worker(
    rank: int,
    world_size: int,
    port: int,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        param_ref, param_new = _setup_parameters()
        opt_ref = torch.optim.AdamW(
            [param_ref], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        opt_new = TritonDistAdamW(
            [param_new], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

        opt_ref.step()
        opt_new.step()

        dist.barrier()

        torch.testing.assert_close(param_ref, param_new, atol=1e-6, rtol=1e-6)

        state_ref = opt_ref.state[param_ref]
        state_new = opt_new.state[param_new]
        shard_len = state_new["exp_avg"].numel()
        offset = shard_len * rank

        reference_exp_avg = state_ref["exp_avg"].view(-1).narrow(0, offset, shard_len)
        reference_exp_avg_sq = (
            state_ref["exp_avg_sq"].view(-1).narrow(0, offset, shard_len)
        )

        torch.testing.assert_close(
            reference_exp_avg, state_new["exp_avg"], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            reference_exp_avg_sq, state_new["exp_avg_sq"], atol=1e-6, rtol=1e-6
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for TritonAdamW tests"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_adamw_matches_torch(dtype: torch.dtype):
    device = torch.device("cuda")
    param_ref, param_new = _setup_parameters(dtype=dtype, device=device)

    lr = 1e-2
    betas = (0.9, 0.95)
    eps = 1e-8
    weight_decay = 1e-2

    opt_ref = torch.optim.AdamW(
        [param_ref], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )
    opt_new = TritonAdamW(
        [param_new], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )

    opt_ref.step()
    opt_new.step()
    torch.cuda.synchronize()

    if dtype == torch.float32:
        atol, rtol = 1e-7, 1e-7
    else:
        atol, rtol = 1e-3, 1e-3
    torch.testing.assert_close(param_ref, param_new, atol=atol, rtol=rtol)

    state_ref = opt_ref.state[param_ref]
    state_new = opt_new.state[param_new]
    torch.testing.assert_close(
        state_ref["exp_avg"].to(torch.float32),
        state_new["exp_avg"].to(torch.float32),
        atol=1e-6,
        rtol=5e-4,
    )
    torch.testing.assert_close(
        state_ref["exp_avg_sq"].to(torch.float32),
        state_new["exp_avg_sq"].to(torch.float32),
        atol=1e-6,
        rtol=5e-4,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for TritonAdamW tests"
)
def test_triton_adamw_maximize():
    device = torch.device("cuda")
    param_ref, param_new = _setup_parameters(device=device)

    lr = 1e-3
    betas = (0.8, 0.999)
    eps = 1e-9

    opt_ref = torch.optim.AdamW([param_ref], lr=lr, betas=betas, eps=eps, maximize=True)
    opt_new = TritonAdamW([param_new], lr=lr, betas=betas, eps=eps, maximize=True)

    opt_ref.step()
    opt_new.step()
    torch.cuda.synchronize()

    torch.testing.assert_close(param_ref, param_new, atol=3e-5, rtol=5e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for TritonAdamW tests"
)
def test_triton_adamw_multiple_params():
    torch.manual_seed(1234)
    device = torch.device("cuda")
    params_ref = [
        torch.randn(4, 6, device=device, requires_grad=True),
        torch.randn(3, 5, device=device, requires_grad=True),
    ]
    params_new = [p.detach().clone().requires_grad_(True) for p in params_ref]

    grads = [torch.randn_like(p) for p in params_ref]
    for g, p_ref, p_new in zip(grads, params_ref, params_new):
        p_ref.grad = g.clone()
        p_new.grad = g.clone()

    lr = 5e-3
    betas = (0.91, 0.987)
    eps = 1e-8
    weight_decay = 2e-2

    opt_ref = torch.optim.AdamW(
        params_ref, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )
    opt_new = TritonAdamW(
        params_new, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )

    opt_ref.step()
    opt_new.step()
    torch.cuda.synchronize()

    for p_ref, p_new in zip(params_ref, params_new):
        torch.testing.assert_close(p_ref, p_new, atol=1e-6, rtol=1e-4)

    for p_ref, p_new in zip(params_ref, params_new):
        state_ref = opt_ref.state[p_ref]
        state_new = opt_new.state[p_new]
        torch.testing.assert_close(
            state_ref["exp_avg"].to(torch.float32),
            state_new["exp_avg"].to(torch.float32),
            atol=1e-5,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            state_ref["exp_avg_sq"].to(torch.float32),
            state_new["exp_avg_sq"].to(torch.float32),
            atol=1e-5,
            rtol=1e-4,
        )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
def test_triton_dist_adamw_matches_torch_distributed():
    world_size = 2
    port = _find_free_port()
    lr = 5e-3
    betas = (0.9, 0.95)
    eps = 1e-8
    weight_decay = 1e-2

    mp.spawn(  # type: ignore[attr-defined]
        _distributed_adamw_worker,
        args=(world_size, port, lr, betas, eps, weight_decay),
        nprocs=world_size,
        join=True,
    )
