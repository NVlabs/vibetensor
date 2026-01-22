# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton-backed AdamW optimizers.

This module now provides a single-GPU Triton implementation of AdamW. The
``TritonAdamW`` class mirrors ``torch.optim.AdamW`` while executing the update
step through a fused Triton kernel when the parameters are CUDA tensors of a
supported dtype (fp32/fp16/bf16). Unsupported configurations transparently
fall back to the PyTorch functional implementation.

Distributed variants still delegate to ``torch.distributed`` utilities; only the
single-rank kernel is implemented here per the current milestone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, Iterable, Optional, Tuple

import torch
import triton
from torch.optim._functional import adamw as functional_adamw  # type: ignore[import]

from vibe_kernels.common.tensor_types import TensorLike

from .impl import triton_impl
from .metadata import ShardMetadata

Tensor = TensorLike
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _is_bias_parameter(param: Tensor) -> bool:
    return param.ndim <= 1


@dataclass
class AdamWConfig:
    """Static configuration parameters for the Triton AdamW kernels."""

    chunk_size: int = 4096
    ilp: int = 4
    allow_tf32: bool = False


class TritonAdamW(torch.optim.Optimizer):
    """Optimizer exposing the AdamW API while using Triton kernels when possible."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        maximize: bool = False,
        config: Optional[AdamWConfig] = None,
    ) -> None:
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, maximize=maximize
        )
        super().__init__(params, defaults)
        self._config = config or AdamWConfig()

    def _init_state(self, p: Tensor, state: dict[str, Any]) -> None:
        state["step"] = torch.tensor(0.0, device=p.device)
        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state["is_bias"] = _is_bias_parameter(p)

    @staticmethod
    def _can_use_triton(param: Tensor, grad: Tensor | None) -> bool:
        if grad is None:
            return False
        if param.device.type != "cuda" or grad.device.type != "cuda":
            return False
        if param.dtype not in _SUPPORTED_DTYPES or grad.dtype not in _SUPPORTED_DTYPES:
            return False
        return True

    def step(self, closure=None):  # type: ignore[override]
        """Perform a single optimization step."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            maximize = group.get("maximize", False)

            for p in params:
                grad = p.grad
                if grad is None:
                    continue

                if grad.is_sparse:  # pragma: no cover - sparse grads unsupported
                    raise RuntimeError("TritonAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, state)

                # Ensure state tensors are on the correct device
                if state["exp_avg"].device != p.device:
                    state["exp_avg"] = state["exp_avg"].to(device=p.device)
                if state["exp_avg_sq"].device != p.device:
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(device=p.device)
                if state["step"].device != p.device:
                    state["step"] = state["step"].to(device=p.device)

                lr_mul = getattr(p, "lr_mul", 1.0)
                wd_mul = getattr(p, "wd_mul", 1.0)
                lr_local = lr * lr_mul
                wd_local = weight_decay * wd_mul

                step_tensor: Tensor = state["step"]
                step_val = int(step_tensor.item()) + 1

                use_triton = self._can_use_triton(p, grad)

                if not use_triton:
                    # Fallback to PyTorch functional implementation (keeps state tensors).
                    functional_adamw(
                        [p],
                        [grad],
                        [state["exp_avg"]],
                        [state["exp_avg_sq"]],
                        [],
                        [step_tensor],
                        foreach=False,
                        capturable=False,
                        differentiable=False,
                        fused=None,
                        grad_scale=None,
                        found_inf=None,
                        has_complex=torch.is_complex(p),
                        amsgrad=False,
                        beta1=beta1,
                        beta2=beta2,
                        lr=lr_local,
                        weight_decay=wd_local,
                        eps=eps,
                        maximize=maximize,
                    )
                    continue

                if not p.is_contiguous():
                    p.data = p.data.contiguous()
                if not grad.is_contiguous():
                    grad = grad.contiguous()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if not exp_avg.is_contiguous():
                    exp_avg = exp_avg.contiguous()
                    state["exp_avg"] = exp_avg
                if not exp_avg_sq.is_contiguous():
                    exp_avg_sq = exp_avg_sq.contiguous()
                    state["exp_avg_sq"] = exp_avg_sq

                param_flat = p.data.view(-1)
                grad_flat = grad.view(-1)
                exp_avg_flat = exp_avg.view(-1)
                exp_avg_sq_flat = exp_avg_sq.view(-1)

                n_elements = param_flat.numel()
                if n_elements == 0:
                    continue

                step_size = lr_local / (1.0 - beta1**step_val)
                inv_bias_correction2 = 1.0 / (1.0 - beta2**step_val)
                decay = lr_local * wd_local
                if state["is_bias"]:
                    decay = 0.0
                maximize_sign = -1.0 if maximize else 1.0

                step_tensor.add_(1.0)
                block_size = triton_impl.pick_block_size(n_elements)
                grid = (triton.cdiv(n_elements, block_size),)

                triton_impl._adamw_update_kernel[grid](
                    param_flat,
                    grad_flat,
                    exp_avg_flat,
                    exp_avg_sq_flat,
                    step_size,
                    beta1,
                    1.0 - beta1,
                    beta2,
                    1.0 - beta2,
                    inv_bias_correction2,
                    eps,
                    decay,
                    maximize_sign,
                    n_elements,
                    BLOCK_SIZE=block_size,
                    num_warps=triton_impl.num_warps(block_size),
                )

        return loss


class TritonDistAdamW(TritonAdamW):
    """Distributed AdamW mirroring ``nanochat.adamw.DistAdamW`` semantics."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        maximize: bool = False,
        config: Optional[AdamWConfig] = None,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            config=config,
        )

    def step(self, closure=None):  # type: ignore[override]
        """Distributed AdamW step with ZeRO-2 style sharding.

        For the distributed variant we continue to reuse the PyTorch functional
        update within the sharded communication pattern while single-rank kernels
        execute the fused Triton update.
        """

        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return super().step(closure)

        world_size = dist.get_world_size()
        if world_size <= 1:
            return super().step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            rank = dist.get_rank()

            for group in self.param_groups:
                params = group["params"]
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                maximize = group.get("maximize", False)

                for p in params:
                    grad = p.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:  # pragma: no cover - sparse grads unsupported
                        raise RuntimeError(
                            "TritonDistAdamW does not support sparse gradients"
                        )
                    if not p.is_contiguous():
                        raise ValueError(
                            "TritonDistAdamW requires contiguous parameters"
                        )
                    if not grad.is_contiguous():
                        raise ValueError(
                            "TritonDistAdamW requires contiguous gradients"
                        )

                    param_flat = p.view(-1)
                    grad_flat = grad.view(-1)
                    if grad_flat.numel() != param_flat.numel():
                        raise ValueError(
                            "Gradient and parameter must have identical numel"
                        )
                    if grad_flat.numel() % world_size != 0:
                        raise ValueError(
                            "Parameter numel must be divisible by world_size for TritonDistAdamW"
                        )

                    shard_len = grad_flat.numel() // world_size
                    if shard_len == 0:
                        continue

                    shard_offset = shard_len * rank
                    shard_info = ShardMetadata(
                        parameter=p, offset=shard_offset, length=shard_len
                    )

                    avg_grad = grad_flat.clone()
                    dist.all_reduce(avg_grad, op=dist.ReduceOp.AVG)
                    grad_slice = avg_grad.narrow(
                        0, shard_offset, shard_len
                    ).contiguous()

                    state = self.state[p]
                    if len(state) == 0:
                        # Initialize with sharded state directly
                        state["step"] = torch.tensor(0.0, device=p.device)
                        state["shard_metadata"] = shard_info
                        state["exp_avg"] = torch.zeros(
                            shard_len, dtype=p.dtype, device=p.device
                        )
                        state["exp_avg_sq"] = torch.zeros(
                            shard_len, dtype=p.dtype, device=p.device
                        )
                        state["is_bias"] = _is_bias_parameter(p)
                    elif state.get("shard_metadata") != shard_info:
                        state["shard_metadata"] = shard_info
                        state["exp_avg"] = torch.zeros(
                            shard_len, dtype=p.dtype, device=p.device
                        )
                        state["exp_avg_sq"] = torch.zeros(
                            shard_len, dtype=p.dtype, device=p.device
                        )

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    step_tensor: Tensor = state["step"]

                    functional_adamw(
                        [param_flat.narrow(0, shard_offset, shard_len)],
                        [grad_slice],
                        [exp_avg],
                        [exp_avg_sq],
                        [],
                        [step_tensor],
                        foreach=False,
                        capturable=False,
                        differentiable=False,
                        fused=None,
                        grad_scale=None,
                        found_inf=None,
                        has_complex=torch.is_complex(p),
                        amsgrad=False,
                        beta1=beta1,
                        beta2=beta2,
                        lr=lr,
                        weight_decay=weight_decay,
                        eps=eps,
                        maximize=maximize,
                    )

                    gather_list = [
                        torch.empty_like(param_flat.narrow(0, shard_offset, shard_len))
                        for _ in range(world_size)
                    ]
                    dist.all_gather(
                        gather_list, param_flat.narrow(0, shard_offset, shard_len)
                    )
                    param_flat.copy_(torch.cat(gather_list, dim=0))

        return loss
