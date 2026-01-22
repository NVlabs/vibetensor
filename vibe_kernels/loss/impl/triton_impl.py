# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch  # type: ignore[import]
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

from vibe_kernels.common import alloc
from vibe_kernels.common.tensor_types import TensorLike

Reduction = Literal["none", "mean", "byte_mean"]


def _flatten_last_dim(tensor: TensorLike) -> Tuple[TensorLike, int, int]:
    vocab_size = tensor.shape[-1]
    rows = tensor.numel() // vocab_size if vocab_size > 0 else 0
    if rows == 0:
        return tensor.new_empty((0, vocab_size)), 0, vocab_size
    return tensor.reshape(rows, vocab_size).contiguous(), rows, vocab_size


@triton.jit
def _cross_entropy_forward(
    logits_ptr,
    targets_ptr,
    logsumexp_ptr,
    mask_ptr,
    rows,
    cols,
    stride_logits,
    IGNORE_INDEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    row_offset = row * stride_logits
    offsets = tl.arange(0, BLOCK_SIZE)

    target = tl.load(targets_ptr + row)
    is_active = target != IGNORE_INDEX

    max_val = tl.full([1], -float("inf"), dtype=tl.float32)

    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(logits_ptr + row_offset + idx, mask=mask, other=-float("inf"))
        vals = vals.to(tl.float32)
        block_max = tl.max(tl.where(mask, vals, -float("inf")), axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = tl.zeros([1], dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(logits_ptr + row_offset + idx, mask=mask, other=-float("inf"))
        vals = vals.to(tl.float32)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0), axis=0)

    sum_exp = tl.maximum(sum_exp, 1e-20)
    logsumexp = (tl.log(sum_exp) + max_val).to(tl.float32)
    target_logit = tl.load(
        logits_ptr + row_offset + target, mask=is_active, other=0.0
    ).to(tl.float32)
    active_mask = tl.where(is_active, 1.0, 0.0).to(tl.float32)

    out_idx = row + tl.arange(0, 1)
    tl.store(logsumexp_ptr + out_idx, logsumexp, mask=out_idx < rows)
    tl.store(mask_ptr + out_idx, active_mask, mask=out_idx < rows)


@triton.jit
def _cross_entropy_backward(
    grad_ptr,
    logits_ptr,
    targets_ptr,
    dloss_ptr,
    logsumexp_ptr,
    mask_ptr,
    rows,
    cols,
    stride_logits,
    LOGITS_IS_FP16: tl.constexpr,
    LOGITS_IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    row_offset = row * stride_logits
    offsets = tl.arange(0, BLOCK_SIZE)

    mask_val = tl.load(mask_ptr + row)
    if mask_val == 0.0:
        zero = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        zero = tl.where(offsets < cols, zero, 0.0)
        for start in range(0, cols, BLOCK_SIZE):
            idx = start + offsets
            mask = idx < cols
            out = zero
            if LOGITS_IS_FP16:
                out = out.to(tl.float16)
            elif LOGITS_IS_BF16:
                out = out.to(tl.bfloat16)
            tl.store(grad_ptr + row_offset + idx, out, mask=mask)
        return

    logsumexp = tl.load(logsumexp_ptr + row)
    scale = tl.load(dloss_ptr + row)
    target = tl.load(targets_ptr + row)

    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        logits = tl.load(logits_ptr + row_offset + idx, mask=mask, other=-float("inf"))
        logits = logits.to(tl.float32)
        probs = tl.exp(logits - logsumexp)
        grad = probs * scale
        if LOGITS_IS_FP16:
            grad = grad.to(tl.float16)
        elif LOGITS_IS_BF16:
            grad = grad.to(tl.bfloat16)
        tl.store(grad_ptr + row_offset + idx, grad, mask=mask)

    target_offset = row_offset + target
    current = tl.load(grad_ptr + target_offset)
    current_f32 = current.to(tl.float32)
    adjusted = current_f32 - scale
    if LOGITS_IS_FP16:
        adjusted = adjusted.to(tl.float16)
    elif LOGITS_IS_BF16:
        adjusted = adjusted.to(tl.bfloat16)
    tl.store(grad_ptr + target_offset, adjusted)


class _CrossEntropyLossFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logits: TensorLike,
        targets: TensorLike,
        ignore_index: int,
        reduction: Reduction,
        token_bytes: Optional[TensorLike],
    ):
        if logits.device.type != "cuda":  # pragma: no cover
            raise RuntimeError("cross_entropy_loss Triton kernel requires CUDA tensors")
        if logits.ndim < 2:
            raise ValueError("logits must have at least 2 dimensions")
        if targets.dtype != torch.int64 and targets.dtype != torch.int32:
            raise TypeError("targets must be int64 or int32")

        orig_shape = logits.shape
        logits_2d, rows, vocab_size = _flatten_last_dim(logits)
        targets_flat = targets.reshape(rows).to(torch.int32)

        BLOCK = 128
        logsumexp = alloc.empty((rows,), like=logits, dtype="float32")
        mask = alloc.empty_like(logsumexp)

        stride = logits_2d.stride(0)
        grid = (rows,)
        _cross_entropy_forward[grid](  # type: ignore[misc]
            logits_2d,
            targets_flat,
            logsumexp,
            mask,
            rows,
            vocab_size,
            stride,
            IGNORE_INDEX=ignore_index,
            BLOCK_SIZE=BLOCK,
        )

        logits_float = logits_2d.to(torch.float32)
        targets_clamped = torch.clamp(targets_flat.to(torch.int64), min=0)
        target_logits = torch.gather(
            logits_float, 1, targets_clamped.unsqueeze(1)
        ).squeeze(1)
        losses = (logsumexp - target_logits) * mask

        byte_weights: Optional[TensorLike] = None
        if token_bytes is not None:
            if token_bytes.ndim != 1:
                raise ValueError("token_bytes must be a 1D tensor matching vocab size")
            if token_bytes.shape[0] < vocab_size:
                raise ValueError(
                    "token_bytes length must be at least the vocabulary size"
                )
            bytes_table = token_bytes.to(device=logits.device, dtype=torch.float32)
            gathered_bytes = torch.gather(
                bytes_table, 0, targets_clamped.to(torch.int64)
            )
            byte_weights = gathered_bytes * mask

        valid_count = mask.sum()
        if byte_weights is not None:
            ctx.save_for_backward(
                logits_2d, targets_flat, logsumexp, mask, byte_weights
            )
            ctx.byte_sum = byte_weights.sum()
        else:
            ctx.save_for_backward(logits_2d, targets_flat, logsumexp, mask)
            ctx.byte_sum = None
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.rows = rows
        ctx.vocab_size = vocab_size
        ctx.orig_shape = orig_shape
        ctx.valid_count = valid_count
        ctx.has_byte_weights = byte_weights is not None

        if reduction == "none":
            return losses.view(*orig_shape[:-1])
        if reduction == "mean":
            if valid_count.item() == 0:
                return losses.new_zeros(())
            return losses.sum() / valid_count
        if reduction == "byte_mean":
            if byte_weights is None:
                raise ValueError(
                    "reduction='byte_mean' requires token_bytes to be provided"
                )
            byte_sum = ctx.byte_sum
            if byte_sum is None or byte_sum.item() == 0:
                return losses.new_zeros(())
            weighted = (losses * byte_weights).sum()
            return weighted / byte_sum
        raise ValueError(
            "Unsupported reduction: expected 'mean', 'none', or 'byte_mean'"
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_output: TensorLike,
    ):
        saved = ctx.saved_tensors
        if ctx.has_byte_weights:
            logits_2d, targets_flat, logsumexp, mask, byte_weights = saved
        else:
            logits_2d, targets_flat, logsumexp, mask = saved
            byte_weights = None

        rows = ctx.rows
        vocab_size = ctx.vocab_size
        BLOCK = 128

        if ctx.reduction == "none":
            dloss = grad_output.contiguous().view(rows).to(torch.float32)
        elif ctx.reduction == "mean":
            if ctx.valid_count.item() == 0:
                zero = logits_2d.new_zeros(logits_2d.shape).view(ctx.orig_shape)
                return zero, None, None, None, None
            scale = grad_output.to(torch.float32) / ctx.valid_count
            dloss = mask * scale
        elif ctx.reduction == "byte_mean":
            if byte_weights is None:
                raise RuntimeError(
                    "byte_mean reduction expected byte weights in context"
                )
            byte_sum = ctx.byte_sum
            if byte_sum is None or byte_sum.item() == 0:
                zero = logits_2d.new_zeros(logits_2d.shape).view(ctx.orig_shape)
                return zero, None, None, None, None
            scale = grad_output.to(torch.float32) / byte_sum
            dloss = byte_weights * scale
        else:
            raise ValueError("Unsupported reduction in backward pass")

        grad_logits = alloc.empty_like(logits_2d)
        stride = logits_2d.stride(0)
        logits_dtype = logits_2d.dtype
        grid = (rows,)
        _cross_entropy_backward[grid](  # type: ignore[misc]
            grad_logits,
            logits_2d,
            targets_flat,
            dloss,
            logsumexp,
            mask,
            rows,
            vocab_size,
            stride,
            LOGITS_IS_FP16=logits_dtype == torch.float16,
            LOGITS_IS_BF16=logits_dtype == torch.bfloat16,
            BLOCK_SIZE=BLOCK,
        )

        grad_logits = grad_logits.view(ctx.orig_shape)
        return grad_logits, None, None, None, None


def cross_entropy_loss(
    logits: TensorLike,
    targets: TensorLike,
    ignore_index: int,
    reduction: Reduction,
    token_bytes: Optional[TensorLike],
):
    return _CrossEntropyLossFn.apply(
        logits, targets, ignore_index, reduction, token_bytes
    )
