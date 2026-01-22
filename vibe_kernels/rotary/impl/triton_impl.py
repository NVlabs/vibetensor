# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from vibe_kernels.common.tensor_types import TensorLike


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _pick_num_warps(block: int) -> int:
    if block >= 8192:
        return 16
    if block >= 2048:
        return 8
    if block >= 1024:
        return 4
    if block >= 512:
        return 4
    if block >= 256:
        return 2
    return 1


@triton.jit
def _apply_rotary(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    rows,
    q_row_stride,
    k_row_stride,
    cos_stride,
    head_dim_half,
    Q_IS_FP16: tl.constexpr,
    Q_IS_BF16: tl.constexpr,
    K_IS_FP16: tl.constexpr,
    K_IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_offsets = pid * ROWS_PER_CTA + tl.arange(0, ROWS_PER_CTA)
    row_mask = row_offsets < rows

    col = tl.arange(0, BLOCK_SIZE)
    col_mask = col < head_dim_half
    mask_2d = row_mask[:, None] & col_mask[None, :]

    pos = tl.load(pos_ptr + row_offsets, mask=row_mask, other=0).to(tl.int32)

    cos_base = cos_ptr + pos[:, None] * cos_stride
    sin_base = sin_ptr + pos[:, None] * cos_stride
    cos = tl.load(cos_base + col[None, :], mask=mask_2d, other=0.0).to(tl.float32)
    sin = tl.load(sin_base + col[None, :], mask=mask_2d, other=0.0).to(tl.float32)

    q_base = q_ptr + row_offsets[:, None] * q_row_stride
    k_base = k_ptr + row_offsets[:, None] * k_row_stride

    q1 = tl.load(q_base + col[None, :], mask=mask_2d, other=0.0).to(tl.float32)
    q2 = tl.load(q_base + head_dim_half + col[None, :], mask=mask_2d, other=0.0).to(
        tl.float32
    )
    k1 = tl.load(k_base + col[None, :], mask=mask_2d, other=0.0).to(tl.float32)
    k2 = tl.load(k_base + head_dim_half + col[None, :], mask=mask_2d, other=0.0).to(
        tl.float32
    )

    q_out1 = tl.math.fma(q2, sin, q1 * cos)
    q_out2 = tl.math.fma(q2, cos, (-q1) * sin)
    k_out1 = tl.math.fma(k2, sin, k1 * cos)
    k_out2 = tl.math.fma(k2, cos, (-k1) * sin)

    if Q_IS_FP16:
        q_out1 = q_out1.to(tl.float16)
        q_out2 = q_out2.to(tl.float16)
    elif Q_IS_BF16:
        q_out1 = q_out1.to(tl.bfloat16)
        q_out2 = q_out2.to(tl.bfloat16)

    if K_IS_FP16:
        k_out1 = k_out1.to(tl.float16)
        k_out2 = k_out2.to(tl.float16)
    elif K_IS_BF16:
        k_out1 = k_out1.to(tl.bfloat16)
        k_out2 = k_out2.to(tl.bfloat16)

    tl.store(q_base + col[None, :], q_out1, mask=mask_2d)
    tl.store(q_base + head_dim_half + col[None, :], q_out2, mask=mask_2d)
    tl.store(k_base + col[None, :], k_out1, mask=mask_2d)
    tl.store(k_base + head_dim_half + col[None, :], k_out2, mask=mask_2d)


class _RotaryEmbeddingFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: TensorLike,
        k: TensorLike,
        cos: TensorLike,
        sin: TensorLike,
        positions: Optional[TensorLike],
    ):
        if q.device.type != "cuda":  # pragma: no cover - GPU only
            raise RuntimeError("Rotary kernel requires CUDA tensors")
        assert q.shape == k.shape, "q and k must have the same shape"
        assert q.dtype == k.dtype, "q and k must share dtype"
        assert q.is_contiguous(), "q must be contiguous"
        assert k.is_contiguous(), "k must be contiguous"

        *leading, seqlen, head_dim = q.shape
        assert head_dim % 2 == 0, "head dimension must be even"
        head_dim_half = head_dim // 2

        q_flat = q.view(-1, head_dim).contiguous()
        k_flat = k.view(-1, head_dim).contiguous()
        rows = q_flat.shape[0]

        cos_flat = cos.reshape(-1, cos.shape[-1]).contiguous()
        sin_flat = sin.reshape(-1, sin.shape[-1]).contiguous()
        assert cos_flat.shape[1] == head_dim_half, "cos dimension mismatch"

        if positions is not None:
            pos_flat = positions.reshape(-1).to(torch.int32).contiguous()
        else:
            assert rows % seqlen == 0
            offsets = torch.arange(seqlen, device=q.device, dtype=torch.int32)
            pos_flat = offsets.repeat(rows // seqlen)

        block = _next_power_of_2(head_dim_half)
        num_warps = _pick_num_warps(block)

        q_dtype = q.dtype
        k_dtype = k.dtype
        q_is_fp16 = q_dtype == torch.float16
        q_is_bf16 = q_dtype == torch.bfloat16
        k_is_fp16 = k_dtype == torch.float16
        k_is_bf16 = k_dtype == torch.bfloat16

        rows_per_cta = 4
        grid = (triton.cdiv(rows, rows_per_cta),)
        _apply_rotary[grid](  # type: ignore[misc]
            q_flat,
            k_flat,
            cos_flat,
            sin_flat,
            pos_flat,
            rows,
            q_flat.stride(0),
            k_flat.stride(0),
            cos_flat.stride(0),
            head_dim_half,
            Q_IS_FP16=q_is_fp16,
            Q_IS_BF16=q_is_bf16,
            K_IS_FP16=k_is_fp16,
            K_IS_BF16=k_is_bf16,
            BLOCK_SIZE=block,
            ROWS_PER_CTA=rows_per_cta,
            num_warps=num_warps,
            num_stages=2,
        )

        q_out = q_flat.view(*leading, seqlen, head_dim)
        k_out = k_flat.view(*leading, seqlen, head_dim)

        ctx.save_for_backward(cos_flat, sin_flat, pos_flat)
        ctx.head_dim_half = head_dim_half
        ctx.shape = q.shape
        ctx.dtype = q.dtype
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q_out, grad_k_out):  # type: ignore[override]
        cos_flat, sin_flat, pos_flat = ctx.saved_tensors
        head_dim_half = ctx.head_dim_half
        shape = ctx.shape
        dtype = ctx.dtype

        grad_q = grad_q_out.contiguous().view(-1, head_dim_half * 2).to(torch.float32)
        grad_k = grad_k_out.contiguous().view(-1, head_dim_half * 2).to(torch.float32)

        cos = cos_flat[pos_flat.long()].to(torch.float32)
        sin = sin_flat[pos_flat.long()].to(torch.float32)

        def _rotate_back(grad: TensorLike):
            g1 = grad[:, :head_dim_half]
            g2 = grad[:, head_dim_half:]
            dx1 = g1 * cos - g2 * sin
            dx2 = g1 * sin + g2 * cos
            return torch.cat([dx1, dx2], dim=1)

        grad_q_in = _rotate_back(grad_q).to(dtype).view(shape)
        grad_k_in = _rotate_back(grad_k).to(dtype).view(shape)
        return grad_q_in, grad_k_in, None, None, None


def apply_rotary_triton(
    q: TensorLike,
    k: TensorLike,
    cos: TensorLike,
    sin: TensorLike,
    positions: Optional[TensorLike],
):
    return _RotaryEmbeddingFn.apply(q, k, cos, sin, positions)
