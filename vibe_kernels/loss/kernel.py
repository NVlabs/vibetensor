# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import torch

from vibe_kernels.common import alloc
from vibe_kernels.common.tensor_types import TensorLike
from vibe_kernels.softmax import kernel as softmax_kernel

from .impl import torch_impl, triton_impl

try:
    from .impl import cutedsl_impl

    _cutedsl_available = True
except (ImportError, ValueError, RuntimeError):
    cutedsl_impl = None  # type: ignore[assignment]
    _cutedsl_available = False

Reduction = Literal["none", "mean", "byte_mean"]


def is_cutedsl_available() -> bool:
    return _cutedsl_available


class _CutedslCrossEntropyLossFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int,
        reduction: Reduction,
        token_bytes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if logits.device.type != "cuda":  # pragma: no cover
            raise RuntimeError(
                "cross_entropy_loss CuTeDSL kernel requires CUDA tensors"
            )
        if logits.ndim < 2:
            raise ValueError("logits must have at least 2 dimensions")
        if targets.dtype not in (torch.int64, torch.int32):
            raise TypeError("targets must be int64 or int32")

        assert cutedsl_impl is not None

        orig_shape = logits.shape
        # Reuse helper from triton_impl to avoid duplication
        logits_2d, rows, vocab_size = triton_impl._flatten_last_dim(logits)
        targets_flat = targets.reshape(rows)

        need_dx_cache = ctx.needs_input_grad[0]

        # Fast path: common training/inference case
        fast_mean = reduction == "mean" and token_bytes is None
        if fast_mean:
            losses, _, dx = cutedsl_impl.cross_entropy_forward(
                logits_2d,
                targets_flat,
                ignore_index=ignore_index,
                return_lse=False,
                return_dx=need_dx_cache,
            )
            valid_count = (targets_flat != ignore_index).sum().to(torch.float32)

            if need_dx_cache:
                ctx.save_for_backward(dx)
            else:
                ctx.save_for_backward()

            ctx.ignore_index = ignore_index
            ctx.reduction = reduction
            ctx.rows = rows
            ctx.vocab_size = vocab_size
            ctx.orig_shape = orig_shape
            ctx.valid_count = valid_count
            ctx.has_byte_weights = False
            ctx.needs_dx_cache = need_dx_cache
            ctx.byte_sum = None
            ctx.use_mean_fastpath = True

            if valid_count.item() == 0:
                return losses.new_zeros(())
            return losses.sum() / valid_count

        # Generic path with byte weighting support
        losses, _, dx = cutedsl_impl.cross_entropy_forward(
            logits_2d,
            targets_flat,
            ignore_index=ignore_index,
            return_lse=False,
            return_dx=need_dx_cache,
        )
        mask = targets_flat != ignore_index
        mask_f32 = mask.to(torch.float32)

        byte_weights: Optional[torch.Tensor] = None
        if token_bytes is not None:
            if token_bytes.ndim != 1:
                raise ValueError("token_bytes must be a 1D tensor matching vocab size")
            if token_bytes.shape[0] < vocab_size:
                raise ValueError(
                    "token_bytes length must be at least the vocabulary size"
                )
            bytes_table = token_bytes.to(device=logits.device, dtype=torch.float32)
            targets_clamped = torch.clamp(targets_flat.to(torch.int64), min=0)
            gathered_bytes = torch.gather(bytes_table, 0, targets_clamped)
            byte_weights = gathered_bytes * mask_f32

        valid_count = mask_f32.sum()
        if need_dx_cache:
            if byte_weights is not None:
                ctx.save_for_backward(dx, mask, byte_weights)
                ctx.byte_sum = byte_weights.sum()
            else:
                ctx.save_for_backward(dx, mask)
                ctx.byte_sum = None
        else:
            ctx.save_for_backward()
            ctx.byte_sum = None

        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.rows = rows
        ctx.vocab_size = vocab_size
        ctx.orig_shape = orig_shape
        ctx.valid_count = valid_count
        ctx.has_byte_weights = need_dx_cache and byte_weights is not None
        ctx.needs_dx_cache = need_dx_cache

        losses_view = losses.view(*orig_shape[:-1]) if orig_shape[:-1] else losses
        if reduction == "none":
            return losses_view
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
    ) -> Tuple[TensorLike, None, None, None, None]:
        if not getattr(ctx, "needs_dx_cache", False):
            raise RuntimeError(
                "CuTeDSL cross_entropy_loss was evaluated with gradients disabled; backward is unavailable."
            )

        saved = ctx.saved_tensors

        # Fast path for mean reduction without byte weights
        if getattr(ctx, "use_mean_fastpath", False) and ctx.reduction == "mean":
            (dx,) = saved
            if ctx.valid_count.item() == 0:
                zero = dx.new_zeros(dx.shape).view(ctx.orig_shape)
                return zero, None, None, None, None
            scale = grad_output.to(dx.dtype) / ctx.valid_count
            grad_logits = dx * scale
            grad_logits = grad_logits.view(ctx.orig_shape)
            return grad_logits, None, None, None, None

        if ctx.has_byte_weights:
            dx, mask, byte_weights = saved
        else:
            dx, mask = saved
            byte_weights = None

        rows = ctx.rows
        mask_float = mask.to(dx.dtype)
        if ctx.reduction == "none":
            dloss = grad_output.contiguous().view(rows).to(dx.dtype)
        elif ctx.reduction == "mean":
            if ctx.valid_count.item() == 0:
                zero = dx.new_zeros(dx.shape).view(ctx.orig_shape)
                return zero, None, None, None, None
            scale = grad_output.to(dx.dtype) / ctx.valid_count
            dloss = mask_float.view(rows) * scale
        elif ctx.reduction == "byte_mean":
            if byte_weights is None:
                raise RuntimeError(
                    "byte_mean reduction expected byte weights in context"
                )
            byte_sum = ctx.byte_sum
            if byte_sum is None or byte_sum.item() == 0:
                zero = dx.new_zeros(dx.shape).view(ctx.orig_shape)
                return zero, None, None, None, None
            scale = grad_output.to(dx.dtype) / byte_sum
            dloss = byte_weights.to(dx.dtype) * scale
        else:
            raise ValueError("Unsupported reduction in backward pass")

        grad_logits = dx * dloss.view(-1, 1)
        grad_logits = grad_logits.view(ctx.orig_shape)
        return grad_logits, None, None, None, None


def cross_entropy_loss(
    logits: TensorLike,
    targets: TensorLike,
    *,
    ignore_index: int = -1,
    reduction: Reduction = "mean",
    token_bytes: Optional[TensorLike] = None,
    backend: Literal["triton", "cutedsl", "torch"] = "triton",
) -> TensorLike:
    """Cross entropy loss with optional masking and byte-weighting.

    Args:
        logits: Tensor of shape ``(..., vocab)`` in fp16/bf16/fp32.
        targets: Tensor broadcastable to ``logits.shape[:-1]`` with int indices.
        ignore_index: Label value to skip in the loss/gradients.
        reduction: ``"mean"`` (default), ``"none"``, or ``"byte_mean"``.
        token_bytes: Optional 1-D tensor of shape ``(vocab,)`` providing the byte length per
            token. Required when ``reduction="byte_mean"`` to weight losses by byte length.
        backend: Implementation to use (``"triton"`` default, ``"cutedsl"``, or ``"torch"``).
    """

    if logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match logits batch and sequence dimensions")
    if reduction == "byte_mean" and token_bytes is None:
        raise ValueError("token_bytes must be provided when reduction='byte_mean'")

    if backend == "triton":
        return triton_impl.cross_entropy_loss(
            logits, targets, ignore_index, reduction, token_bytes
        )
    if backend == "cutedsl":
        if not _cutedsl_available or cutedsl_impl is None:
            raise RuntimeError("CuTeDSL backend not available")

        # Inference-only fast path: avoid autograd wrapper when no gradients are needed
        if not logits.requires_grad and reduction == "mean" and token_bytes is None:
            # Match the CuTeDSL kernel expectations closely
            logits_2d, rows, vocab_size = triton_impl._flatten_last_dim(logits)
            targets_flat = targets.reshape(rows)
            losses, _, _ = cutedsl_impl.cross_entropy_forward(
                logits_2d,
                targets_flat,
                ignore_index=ignore_index,
                return_lse=False,
                return_dx=False,
            )
            valid_count = (targets_flat != ignore_index).sum().to(torch.float32)
            if valid_count.item() == 0:
                return losses.new_zeros(())
            return losses.sum() / valid_count

        return _CutedslCrossEntropyLossFn.apply(
            logits, targets, ignore_index, reduction, token_bytes
        )
    if backend == "torch":
        return torch_impl.cross_entropy_loss(
            logits, targets, ignore_index, reduction, token_bytes
        )
    raise ValueError("backend must be one of 'triton', 'cutedsl', or 'torch'")


def softmax(
    logits: TensorLike,
    *,
    dim: int = -1,
    backend: Literal["triton", "cutedsl", "torch"] = "triton",
) -> TensorLike:
    return softmax_kernel.softmax(logits, dim=dim, backend=backend)


def log_softmax(
    logits: TensorLike,
    *,
    dim: int = -1,
    backend: Literal["triton", "cutedsl", "torch"] = "triton",
) -> TensorLike:
    return softmax_kernel.log_softmax(logits, dim=dim, backend=backend)


__all__ = ["cross_entropy_loss", "softmax", "log_softmax", "is_cutedsl_available"]
