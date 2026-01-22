# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

import torch  # type: ignore[import]
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

from vibe_kernels.common import alloc
from vibe_kernels.common.tensor_types import TensorLike


@triton.jit
def _softmax_kernel(
    out_ptr,
    logits_ptr,
    rows,
    cols,
    stride_logits,
    stride_out,
    LOGITS_IS_FP16: tl.constexpr,
    LOGITS_IS_BF16: tl.constexpr,
    OUTPUT_LOG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    row_offset_logits = row * stride_logits
    row_offset_out = row * stride_out
    offsets = tl.arange(0, BLOCK_SIZE)

    max_val = tl.full([1], -float("inf"), dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(
            logits_ptr + row_offset_logits + idx, mask=mask, other=-float("inf")
        )
        vals = vals.to(tl.float32)
        block_max = tl.max(tl.where(mask, vals, -float("inf")), axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = tl.zeros([1], dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(
            logits_ptr + row_offset_logits + idx, mask=mask, other=-float("inf")
        )
        vals = vals.to(tl.float32)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0), axis=0)

    sum_exp = tl.maximum(sum_exp, 1e-20)
    logsumexp = tl.log(sum_exp) + max_val

    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(
            logits_ptr + row_offset_logits + idx, mask=mask, other=-float("inf")
        )
        vals = vals.to(tl.float32)
        if OUTPUT_LOG:
            out = vals - logsumexp
        else:
            out = tl.exp(vals - logsumexp)
        if LOGITS_IS_FP16:
            out = out.to(tl.float16)
        elif LOGITS_IS_BF16:
            out = out.to(tl.bfloat16)
        else:
            out = out.to(tl.float32)
        tl.store(out_ptr + row_offset_out + idx, out, mask=mask)


class _TritonSoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: TensorLike):  # type: ignore[override]
        if logits.device.type != "cuda":  # pragma: no cover
            raise RuntimeError("softmax Triton kernel requires CUDA tensors")
        if logits.ndim == 0:
            raise ValueError("softmax input must have at least one dimension")
        cols = logits.shape[-1]
        if cols == 0:
            empty = logits.new_empty((0, 0))
            ctx.save_for_backward(empty)
            ctx.shape = logits.shape
            ctx.rows = 0
            ctx.cols = 0
            return logits.clone()
        rows = logits.numel() // cols
        if rows == 0:
            empty = logits.new_empty((0, cols))
            ctx.save_for_backward(empty)
            ctx.shape = logits.shape
            ctx.rows = 0
            ctx.cols = cols
            return logits.clone()
        logits_2d = logits.contiguous().view(rows, cols)
        output = alloc.empty_like(logits_2d)
        stride_logits = logits_2d.stride(0)
        stride_out = output.stride(0)
        grid = (rows,)
        _softmax_kernel[grid](  # type: ignore[misc]
            output,
            logits_2d,
            rows,
            cols,
            stride_logits,
            stride_out,
            LOGITS_IS_FP16=logits_2d.dtype == torch.float16,
            LOGITS_IS_BF16=logits_2d.dtype == torch.bfloat16,
            OUTPUT_LOG=0,
            BLOCK_SIZE=128,
        )
        ctx.save_for_backward(output)
        ctx.shape = logits.shape
        ctx.rows = rows
        ctx.cols = cols
        return output.view(logits.shape)

    @staticmethod
    def backward(ctx, grad_output: TensorLike):  # type: ignore[override]
        if ctx.rows == 0:
            return (grad_output,)
        (probs,) = ctx.saved_tensors
        grad = grad_output.contiguous().view(ctx.rows, ctx.cols)
        probs = probs.view(ctx.rows, ctx.cols)
        grad = grad - (grad * probs).sum(dim=1, keepdim=True)
        grad = grad * probs
        return (grad.view(ctx.shape),)


class _TritonLogSoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: TensorLike):  # type: ignore[override]
        if logits.device.type != "cuda":  # pragma: no cover
            raise RuntimeError("log_softmax Triton kernel requires CUDA tensors")
        if logits.ndim == 0:
            raise ValueError("log_softmax input must have at least one dimension")
        cols = logits.shape[-1]
        if cols == 0:
            empty = logits.new_empty((0, 0))
            ctx.save_for_backward(empty)
            ctx.shape = logits.shape
            ctx.rows = 0
            ctx.cols = 0
            return logits.clone()
        rows = logits.numel() // cols
        if rows == 0:
            empty = logits.new_empty((0, cols))
            ctx.save_for_backward(empty)
            ctx.shape = logits.shape
            ctx.rows = 0
            ctx.cols = cols
            return logits.clone()
        logits_2d = logits.contiguous().view(rows, cols)
        output = alloc.empty_like(logits_2d)
        stride_logits = logits_2d.stride(0)
        stride_out = output.stride(0)
        grid = (rows,)
        _softmax_kernel[grid](  # type: ignore[misc]
            output,
            logits_2d,
            rows,
            cols,
            stride_logits,
            stride_out,
            LOGITS_IS_FP16=logits_2d.dtype == torch.float16,
            LOGITS_IS_BF16=logits_2d.dtype == torch.bfloat16,
            OUTPUT_LOG=1,
            BLOCK_SIZE=128,
        )
        ctx.save_for_backward(output)
        ctx.shape = logits.shape
        ctx.rows = rows
        ctx.cols = cols
        return output.view(logits.shape)

    @staticmethod
    def backward(ctx, grad_output: TensorLike):  # type: ignore[override]
        if ctx.rows == 0:
            return (grad_output,)
        (log_probs,) = ctx.saved_tensors
        grad = grad_output.contiguous().view(ctx.rows, ctx.cols)
        log_probs = log_probs.view(ctx.rows, ctx.cols)
        probs = torch.exp(log_probs)
        grad = grad - probs * grad.sum(dim=1, keepdim=True)
        return (grad.view(ctx.shape),)


def _prepare_last_dim(
    tensor: TensorLike, dim: int
) -> Tuple[TensorLike, Optional[int]]:
    if tensor.ndim == 0:
        raise ValueError("tensor must have at least one dimension")
    if dim < 0:
        dim += tensor.ndim
    if dim < 0 or dim >= tensor.ndim:
        raise ValueError("dim out of range")
    if dim == tensor.ndim - 1:
        return tensor.contiguous(), None
    return tensor.movedim(dim, -1).contiguous(), dim


def softmax(x: TensorLike, dim: int = -1):
    prepared, moved = _prepare_last_dim(x, dim)
    result = _TritonSoftmaxFn.apply(prepared)
    if moved is None:
        return result
    return result.movedim(-1, moved)


def log_softmax(x: TensorLike, dim: int = -1):
    prepared, moved = _prepare_last_dim(x, dim)
    result = _TritonLogSoftmaxFn.apply(prepared)
    if moved is None:
        return result
    return result.movedim(-1, moved)
