# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch  # type: ignore[import]
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _pick_num_warps(block_size: int) -> int:
    if block_size >= 32768:
        return 32
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    if block_size >= 1024:
        return 4
    if block_size >= 256:
        return 2
    return 1


@triton.jit
def _embedding_rmsnorm_fwd(
    output_ptr,
    output_stride,
    inv_rms_ptr,
    embedding_ptr,
    embedding_stride,
    gamma_ptr,
    token_ptr,
    n_cols,
    eps,
    HAS_GAMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    token = tl.load(token_ptr + row_id)
    emb_row = embedding_ptr + token * embedding_stride
    x = tl.load(emb_row + col_offsets, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.math.rsqrt(mean_sq + eps)
    norm = x * inv_rms

    if HAS_GAMMA:
        gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        norm = norm * gamma

    out_row = output_ptr + row_id * output_stride
    tl.store(out_row + col_offsets, norm.to(tl.float32), mask=mask)
    tl.store(inv_rms_ptr + row_id, inv_rms)


@triton.jit
def _embedding_rmsnorm_bwd(
    grad_weight_ptr,
    grad_weight_stride,
    grad_gamma_ptr,
    grad_output_ptr,
    grad_output_stride,
    embedding_ptr,
    embedding_stride,
    inv_rms_ptr,
    gamma_ptr,
    token_ptr,
    n_cols,
    HAS_GAMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    token = tl.load(token_ptr + row_id)
    emb_row = embedding_ptr + token * embedding_stride
    x = tl.load(emb_row + col_offsets, mask=mask, other=0.0).to(tl.float32)

    grad_row = grad_output_ptr + row_id * grad_output_stride
    dy = tl.load(grad_row + col_offsets, mask=mask, other=0.0).to(tl.float32)

    inv_rms = tl.load(inv_rms_ptr + row_id).to(tl.float32)
    normed = x * inv_rms

    if HAS_GAMMA:
        gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        dy_scaled = dy * gamma
        grad_gamma = tl.sum(dy * normed, axis=0)
        tl.atomic_add(grad_gamma_ptr + col_offsets, grad_gamma, mask=mask)
    else:
        dy_scaled = dy

    dot = tl.sum(dy_scaled * normed, axis=0)
    dx = inv_rms / n_cols * (n_cols * dy_scaled - normed * dot)

    grad_weight_row = grad_weight_ptr + token * grad_weight_stride
    tl.atomic_add(grad_weight_row + col_offsets, dx, mask=mask)


class _EmbeddingRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        weight: torch.Tensor,
        token_ids: torch.Tensor,
        gamma: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        if token_ids.device.type != "cuda":  # pragma: no cover - GPU only
            raise RuntimeError("Triton fused embedding RMSNorm requires CUDA inputs")

        tokens = token_ids.contiguous().view(-1).to(torch.int32)
        weight = weight
        has_gamma = gamma is not None
        gamma_buf = (
            gamma.contiguous()
            if has_gamma
            else torch.empty(0, device=weight.device, dtype=torch.float32)
        )

        n_rows = tokens.numel()
        hidden = weight.shape[1]

        block = _next_power_of_2(hidden)
        num_warps = _pick_num_warps(block)

        out = torch.empty((n_rows, hidden), dtype=weight.dtype, device=weight.device)
        inv_rms = torch.empty((n_rows,), dtype=torch.float32, device=weight.device)

        _embedding_rmsnorm_fwd[(n_rows,)](  # type: ignore[misc]
            out,
            out.stride(0),
            inv_rms,
            weight,
            weight.stride(0),
            gamma_buf if has_gamma else weight,  # unused when HAS_GAMMA == False
            tokens,
            hidden,
            eps,
            HAS_GAMMA=has_gamma,
            BLOCK_SIZE=block,
            num_warps=num_warps,
        )

        ctx.save_for_backward(tokens, inv_rms, gamma_buf)
        ctx.weight = weight
        ctx.has_gamma = has_gamma
        ctx.hidden = hidden
        ctx.block = block
        ctx.num_warps = num_warps
        return out.view(*token_ids.shape, hidden)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        tokens, inv_rms, gamma_buf = ctx.saved_tensors
        weight = ctx.weight
        has_gamma = ctx.has_gamma
        hidden = ctx.hidden
        n_rows = tokens.numel()

        grad_output = grad_output.contiguous().view(n_rows, hidden).to(torch.float32)

        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        grad_gamma = (
            torch.zeros_like(gamma_buf, dtype=torch.float32)
            if has_gamma and gamma_buf.numel() > 0
            else grad_output.new_empty(0)
        )

        _embedding_rmsnorm_bwd[(n_rows,)](  # type: ignore[misc]
            grad_weight,
            grad_weight.stride(0),
            grad_gamma,
            grad_output,
            grad_output.stride(0),
            weight,
            weight.stride(0),
            inv_rms,
            gamma_buf if has_gamma else weight,
            tokens,
            hidden,
            HAS_GAMMA=has_gamma,
            BLOCK_SIZE=ctx.block,
            num_warps=ctx.num_warps,
        )

        grad_weight = grad_weight.to(weight.dtype)
        grad_gamma = (
            grad_gamma.to(gamma_buf.dtype)
            if has_gamma and gamma_buf.numel() > 0
            else None
        )

        return grad_weight, None, grad_gamma, None


class FusedEmbeddingRMSNorm(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        eps: float = 1e-6,
        learnable_gamma: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.learnable_gamma = learnable_gamma

        self.weight = torch.nn.Parameter(
            torch.empty(
                self.num_embeddings, self.embedding_dim, device=device, dtype=dtype
            )
        )
        torch.nn.init.normal_(self.weight, mean=0.0, std=1.0 / self.embedding_dim**0.5)

        if self.learnable_gamma:
            self.gamma = torch.nn.Parameter(
                torch.ones(self.embedding_dim, device=device, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "gamma",
                torch.ones(self.embedding_dim, device=device, dtype=torch.float32),
                persistent=False,
            )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        gamma = self.gamma if self.learnable_gamma else None
        return _EmbeddingRMSNormFn.apply(self.weight, token_ids, gamma, self.eps)


__all__ = [
    "FusedEmbeddingRMSNorm",
]
