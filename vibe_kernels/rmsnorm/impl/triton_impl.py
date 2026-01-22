# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch  # type: ignore[import]
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

from vibe_kernels.common import alloc
from vibe_kernels.common.tensor_types import TensorLike


def _vt_torch_ones_1d(n: int, *, device: Optional[torch.device]) -> TensorLike:
    # Allocate storage via VibeTensor and wrap as a Torch view (DLPack).
    # This avoids Torch allocator usage for parameter/buffer init.
    try:
        from vibetensor import _C as _VTC  # type: ignore[import]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("VibeTensor extension is required for VT-backed allocations") from e

    dev = device
    if dev is None or str(dev) == "cpu" or getattr(dev, "type", "cpu") == "cpu":
        vt = _VTC._cpu_full([int(n)], "float32", 1.0)  # type: ignore[attr-defined]
        t = torch.utils.dlpack.from_dlpack(vt)  # type: ignore[attr-defined]
        setattr(t, "_vbt_backing", vt)
        return t

    if getattr(dev, "type", None) != "cuda":
        raise ValueError("only cpu/cuda devices are supported")
    idx = getattr(dev, "index", None)
    if idx is None:
        idx = int(torch.cuda.current_device())
    vt = _VTC._cuda_zeros([int(n)], "float32", int(idx))  # type: ignore[attr-defined]
    t = torch.utils.dlpack.from_dlpack(vt)  # type: ignore[attr-defined]
    t.fill_(1.0)
    setattr(t, "_vbt_backing", vt)
    return t


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _pick_num_warps(block_size: int) -> int:
    if block_size >= 2048:
        return 8
    if block_size >= 1024:
        return 4
    if block_size >= 512:
        return 4
    if block_size >= 256:
        return 2
    return 1


@triton.jit
def _rmsnorm_fwd(
    output_ptr,
    output_stride,
    inv_rms_ptr,
    input_ptr,
    input_stride,
    gamma_ptr,
    n_cols,
    eps,
    HAS_GAMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNKS: tl.constexpr,
    OUT_IS_FP16: tl.constexpr,
    OUT_IS_BF16: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    in_row = input_ptr + row_id * input_stride
    out_row = output_ptr + row_id * output_stride

    sum_sq = 0.0
    for chunk in tl.static_range(0, CHUNKS):
        col_start = chunk * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(in_row + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    mean_sq = sum_sq / n_cols
    inv_rms = tl.math.rsqrt(mean_sq + eps)

    for chunk in tl.static_range(0, CHUNKS):
        col_start = chunk * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(in_row + cols, mask=mask, other=0.0).to(tl.float32)
        norm = x * inv_rms
        if HAS_GAMMA:
            gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            norm = norm * gamma
        if OUT_IS_FP16:
            norm = norm.to(tl.float16)
        elif OUT_IS_BF16:
            norm = norm.to(tl.bfloat16)
        else:
            norm = norm.to(tl.float32)
        tl.store(out_row + cols, norm, mask=mask)

    tl.store(inv_rms_ptr + row_id, inv_rms)


@triton.jit
def _rmsnorm_bwd(
    grad_input_ptr,
    grad_input_stride,
    grad_gamma_ptr,
    grad_output_ptr,
    grad_output_stride,
    input_ptr,
    input_stride,
    inv_rms_ptr,
    gamma_ptr,
    n_cols,
    HAS_GAMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNKS: tl.constexpr,
    GRAD_IS_FP16: tl.constexpr,
    GRAD_IS_BF16: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    in_row = input_ptr + row_id * input_stride
    grad_out_row = grad_output_ptr + row_id * grad_output_stride
    grad_in_row = grad_input_ptr + row_id * grad_input_stride

    inv_rms = tl.load(inv_rms_ptr + row_id).to(tl.float32)
    dot = 0.0

    for chunk in tl.static_range(0, CHUNKS):
        col_start = chunk * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(in_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(grad_out_row + cols, mask=mask, other=0.0).to(tl.float32)
        normed = x * inv_rms
        if HAS_GAMMA:
            gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            dy_scaled = dy * gamma
            grad_gamma_chunk = tl.sum(dy * normed, axis=0)
            tl.atomic_add(grad_gamma_ptr + cols, grad_gamma_chunk, mask=mask)
        else:
            dy_scaled = dy
        dot += tl.sum(dy_scaled * normed, axis=0)

    dot_mean = dot / n_cols

    for chunk in tl.static_range(0, CHUNKS):
        col_start = chunk * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(in_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(grad_out_row + cols, mask=mask, other=0.0).to(tl.float32)
        normed = x * inv_rms
        if HAS_GAMMA:
            gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            dy_scaled = dy * gamma
        else:
            dy_scaled = dy
        dx = inv_rms * (dy_scaled - normed * dot_mean)
        if GRAD_IS_FP16:
            dx = dx.to(tl.float16)
        elif GRAD_IS_BF16:
            dx = dx.to(tl.bfloat16)
        else:
            dx = dx.to(tl.float32)
        tl.store(grad_in_row + cols, dx, mask=mask)


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        input: TensorLike,
        gamma: Optional[TensorLike],
        eps: float,
    ):
        if input.device.type != "cuda":  # pragma: no cover - GPU only
            raise RuntimeError("RMSNorm Triton kernel requires CUDA input")

        x = input.contiguous()
        hidden = x.shape[-1]
        rows = x.numel() // hidden
        x_mat = x.view(rows, hidden)

        has_gamma = gamma is not None
        if has_gamma:
            gamma_buf = gamma.contiguous()
        else:
            gamma_buf = alloc.empty((0,), like=x, dtype="float32")

        block = min(256, _next_power_of_2(hidden))
        chunks = (hidden + block - 1) // block
        num_warps = _pick_num_warps(block)

        out = alloc.empty_like(x_mat, dtype=input.dtype)
        inv_rms = alloc.empty((rows,), like=x, dtype="float32")

        out_is_fp16 = input.dtype == torch.float16
        out_is_bf16 = input.dtype == torch.bfloat16

        _rmsnorm_fwd[(rows,)](  # type: ignore[misc]
            out,
            out.stride(0),
            inv_rms,
            x_mat,
            x_mat.stride(0),
            gamma_buf if has_gamma else x_mat,
            hidden,
            eps,
            HAS_GAMMA=has_gamma,
            BLOCK_SIZE=block,
            CHUNKS=chunks,
            OUT_IS_FP16=out_is_fp16,
            OUT_IS_BF16=out_is_bf16,
            num_warps=num_warps,
        )

        ctx.save_for_backward(x_mat, inv_rms, gamma_buf)
        ctx.has_gamma = has_gamma
        ctx.hidden = hidden
        ctx.block = block
        ctx.chunks = chunks
        ctx.num_warps = num_warps
        ctx.out_is_fp16 = out_is_fp16
        ctx.out_is_bf16 = out_is_bf16
        ctx.eps = eps
        ctx.input_shape = input.shape
        ctx.input_dtype = input.dtype

        return out.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        x_mat, inv_rms, gamma_buf = ctx.saved_tensors
        hidden = ctx.hidden
        rows = x_mat.shape[0]

        grad_out = grad_output.contiguous().view(rows, hidden)

        grad_input = alloc.empty_like(grad_out, dtype="float32")
        grad_gamma = (
            alloc.zeros_like(gamma_buf, dtype="float32")
            if ctx.has_gamma and gamma_buf.numel() > 0
            else grad_output.new_empty(0)
        )

        grad_is_fp16 = ctx.input_dtype == torch.float16
        grad_is_bf16 = ctx.input_dtype == torch.bfloat16

        _rmsnorm_bwd[(rows,)](  # type: ignore[misc]
            grad_input,
            grad_input.stride(0),
            grad_gamma,
            grad_out,
            grad_out.stride(0),
            x_mat,
            x_mat.stride(0),
            inv_rms,
            gamma_buf if ctx.has_gamma else x_mat,
            hidden,
            HAS_GAMMA=ctx.has_gamma,
            BLOCK_SIZE=ctx.block,
            CHUNKS=ctx.chunks,
            GRAD_IS_FP16=grad_is_fp16,
            GRAD_IS_BF16=grad_is_bf16,
            num_warps=ctx.num_warps,
        )

        grad_input = grad_input.to(ctx.input_dtype).view(ctx.input_shape)
        grad_gamma_out: Optional[TensorLike]
        if ctx.has_gamma and gamma_buf.numel() > 0:
            grad_gamma_out = grad_gamma
        else:
            grad_gamma_out = None

        return grad_input, grad_gamma_out, None


class TritonRMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float = 1e-6,
        learnable_gamma: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.learnable_gamma = learnable_gamma

        if learnable_gamma:
            self.gamma = torch.nn.Parameter(_vt_torch_ones_1d(hidden_size, device=device))
        else:
            self.register_buffer(
                "gamma",
                _vt_torch_ones_1d(hidden_size, device=device),
                persistent=False,
            )
        self.dtype = dtype

    def forward(self, x: TensorLike):  # type: ignore[override]
        gamma = self.gamma if self.learnable_gamma else None
        return _RMSNormFn.apply(x, gamma, self.eps)
