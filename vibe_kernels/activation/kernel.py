# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Tuple

import torch  # type: ignore[import]
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]
from torch import Tensor


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


@triton.jit
def _relu_squared_fwd(out_ptr, inp_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    relu = tl.where(x > 0, x, 0.0)
    y = relu * relu
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _relu_squared_bwd(
    grad_in_ptr, grad_out_ptr, inp_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    relu = tl.where(x > 0, x, 0.0)
    grad = grad_out * (2.0 * relu)
    tl.store(grad_in_ptr + offsets, grad, mask=mask)


class _ReLUSquaredFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if input.device.type != "cuda":  # pragma: no cover
            raise RuntimeError("relu_squared Triton kernel requires CUDA input")

        x = input.contiguous()
        out = torch.empty_like(x)
        n_elements = out.numel()
        block_size = _next_power_of_2(min(1024, max(32, n_elements)))
        grid = ((n_elements + block_size - 1) // block_size,)

        _relu_squared_fwd[grid](  # type: ignore[misc]
            out,
            x,
            n_elements,
            BLOCK_SIZE=block_size,
        )

        ctx.save_for_backward(x)
        ctx.block_size = block_size
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        grad_out = grad_output.contiguous()
        grad_input = torch.empty_like(grad_out)
        n_elements = grad_out.numel()
        grid = ((n_elements + ctx.block_size - 1) // ctx.block_size,)
        _relu_squared_bwd[grid](  # type: ignore[misc]
            grad_input,
            grad_out,
            x,
            n_elements,
            BLOCK_SIZE=ctx.block_size,
        )
        return grad_input


def relu_squared(input: torch.Tensor) -> torch.Tensor:
    """Elementwise ReLU-squared activation with fused backward."""

    return _ReLUSquaredFn.apply(input)


@triton.jit
def _softcap_tanh_fwd(
    softcap_ptr,
    tanh_ptr,
    input_ptr,
    weight_ptr,
    rows,
    cols,
    stride_in,
    stride_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= rows:
        return
    row_offset = pid * stride_in
    out_offset = pid * stride_out
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < cols
        x = tl.load(input_ptr + row_offset + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        proj = x * w
        abs_proj = tl.abs(proj)
        softcap = proj / (1.0 + abs_proj)
        exp_neg = tl.math.exp(-2.0 * abs_proj)
        sign = tl.where(proj >= 0, 1.0, -1.0)
        tanh = sign * (1.0 - exp_neg) / (1.0 + exp_neg)
        tl.store(softcap_ptr + out_offset + offs, softcap, mask=mask)
        tl.store(tanh_ptr + out_offset + offs, tanh, mask=mask)


class _SoftcapTanhProjFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input.device.type != "cuda":  # pragma: no cover
            raise RuntimeError("softcap_tanh_projection requires CUDA tensors")
        if weight.device != input.device:
            raise RuntimeError("weight must be on the same device as input")
        if input.shape[-1] != weight.shape[-1]:
            raise ValueError("weight must match the last dimension of input")

        x = input.contiguous()
        w = weight.contiguous()
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        cols = x.shape[-1]
        x_matrix = x.view(rows, cols)
        softcap = torch.empty_like(x_matrix, dtype=torch.float32)
        tanh = torch.empty_like(x_matrix, dtype=torch.float32)

        block = min(256, _next_power_of_2(max(1, cols)))
        grid = (rows,)
        _softcap_tanh_fwd[grid](  # type: ignore[misc]
            softcap,
            tanh,
            x_matrix,
            w,
            rows,
            cols,
            x_matrix.stride(0),
            softcap.stride(0),
            BLOCK=block,
        )

        softcap_f32 = softcap.view_as(x)
        tanh_f32 = tanh.view_as(x)
        softcap_out = softcap_f32.to(dtype=x.dtype)
        tanh_out = tanh_f32.to(dtype=x.dtype)

        ctx.save_for_backward(x, w, softcap_f32, tanh_f32)
        return softcap_out, tanh_out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_softcap: torch.Tensor,
        grad_tanh: torch.Tensor,
    ):
        x, w, softcap_f32, tanh_f32 = ctx.saved_tensors
        grad_softcap = grad_softcap.to(dtype=torch.float32)
        grad_tanh = grad_tanh.to(dtype=torch.float32)
        x_f32 = x.to(dtype=torch.float32)
        w_f32 = w.to(dtype=torch.float32)
        proj = x_f32 * w_f32
        abs_proj = proj.abs()
        dsoftcap_dproj = 1.0 / (1.0 + abs_proj) ** 2
        dtanh_dproj = 1.0 - tanh_f32**2
        grad_proj = grad_softcap * dsoftcap_dproj + grad_tanh * dtanh_dproj
        grad_input = (grad_proj * w_f32).to(dtype=x.dtype)
        grad_weight = (
            (grad_proj * x_f32).reshape(-1, x.shape[-1]).sum(dim=0).to(dtype=w.dtype)
        )
        return grad_input, grad_weight


def softcap_tanh_projection(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply biasless per-feature projection followed by softcap and tanh transforms."""

    return _SoftcapTanhProjFn.apply(input, weight)


def _launch_grid(n_elements: int, block: int) -> tuple[int]:
    return ((n_elements + block - 1) // block,)


@triton.jit
def _binary_kernel(
    out_ptr, a_ptr, b_ptr, n_elements, OP: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    if OP == 0:
        res = a + b
    else:
        res = a * b
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def _where_kernel(out_ptr, cond_ptr, a_ptr, b_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    cond = tl.load(cond_ptr + offsets, mask=mask, other=0).to(tl.int1)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    res = tl.where(cond, a, b)
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def _lerp_kernel(
    out_ptr,
    a_ptr,
    b_ptr,
    w_ptr,
    weight_scalar,
    n_elements,
    BLOCK: tl.constexpr,
    WEIGHT_IS_SCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    if WEIGHT_IS_SCALAR:
        weight = weight_scalar
    else:
        weight = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    res = a + weight * (b - a)
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def _rowwise_norm_kernel(out_ptr, x_ptr, rows, cols, stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= rows:
        return
    row_offset = pid * stride
    acc = 0.0
    for start in range(0, cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < cols
        x = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        acc += tl.sum(x * x, axis=0)
    tl.store(out_ptr + pid, tl.sqrt(acc))


def _flatten_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_contiguous():
        return tensor
    return tensor.contiguous()


def _binary_op(a: torch.Tensor, b: torch.Tensor, op: int) -> torch.Tensor:
    if a.device.type != "cuda":  # pragma: no cover
        raise RuntimeError("elementwise ops require CUDA tensors")
    if a.shape != b.shape:
        raise ValueError("shapes must match for elementwise operations")
    out = torch.empty_like(a)
    a_flat = _flatten_if_needed(a).view(-1)
    b_flat = _flatten_if_needed(b).view(-1)
    out_flat = out.view(-1)
    block = 256
    grid = _launch_grid(out_flat.numel(), block)
    _binary_kernel[grid](  # type: ignore[misc]
        out_flat,
        a_flat,
        b_flat,
        out_flat.numel(),
        op,
        BLOCK=block,
    )
    return out


def elementwise_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _binary_op(a, b, op=0)


def elementwise_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _binary_op(a, b, op=1)


def elementwise_where(
    cond: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    if cond.device.type != "cuda":  # pragma: no cover
        raise RuntimeError("elementwise_where requires CUDA tensors")
    if cond.dtype != torch.bool:
        raise TypeError("condition tensor must be bool")
    if a.shape != b.shape or a.shape != cond.shape:
        raise ValueError("all tensors must share the same shape")
    out = torch.empty_like(a)
    cond_flat = _flatten_if_needed(cond.to(torch.int8)).view(-1)
    a_flat = _flatten_if_needed(a).view(-1)
    b_flat = _flatten_if_needed(b).view(-1)
    out_flat = out.view(-1)
    block = 256
    grid = _launch_grid(out_flat.numel(), block)
    _where_kernel[grid](  # type: ignore[misc]
        out_flat,
        cond_flat,
        a_flat,
        b_flat,
        out_flat.numel(),
        BLOCK=block,
    )
    return out


def elementwise_lerp(a: Tensor, b: Tensor, weight: Tensor | float) -> Tensor:
    if a.device.type != "cuda":  # pragma: no cover
        raise RuntimeError("elementwise_lerp requires CUDA tensors")
    if a.shape != b.shape:
        raise ValueError("a and b must share shape")
    out = torch.empty_like(a)
    a_flat = _flatten_if_needed(a).view(-1)
    b_flat = _flatten_if_needed(b).view(-1)
    out_flat = out.view(-1)
    block = 256
    grid = _launch_grid(out_flat.numel(), block)
    if not isinstance(weight, Tensor):
        weight_scalar = float(weight)
        _lerp_kernel[grid](  # type: ignore[misc]
            out_flat,
            a_flat,
            b_flat,
            a_flat,  # unused
            weight_scalar,
            out_flat.numel(),
            BLOCK=block,
            WEIGHT_IS_SCALAR=True,
        )
        return out

    weight_tensor: Tensor = weight
    if weight_tensor.device != a.device:
        raise RuntimeError("weight tensor must be on the same device")
    if weight_tensor.shape != a.shape:
        raise ValueError("weight tensor must match input shape")
    w_flat = _flatten_if_needed(weight_tensor).view(-1)
    _lerp_kernel[grid](  # type: ignore[misc]
        out_flat,
        a_flat,
        b_flat,
        w_flat,
        0.0,
        out_flat.numel(),
        BLOCK=block,
        WEIGHT_IS_SCALAR=False,
    )
    return out


def rowwise_l2_norm(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cuda":  # pragma: no cover
        raise RuntimeError("rowwise_l2_norm requires CUDA tensors")
    if x.ndim < 2:
        raise ValueError("rowwise_l2_norm expects at least 2 dimensions")
    rows = x.reshape(-1, x.shape[-1]).shape[0]
    cols = x.shape[-1]
    x_matrix = x.view(rows, cols)
    out = torch.empty(rows, device=x.device, dtype=torch.float32)
    block = 256
    grid = (rows,)
    _rowwise_norm_kernel[grid](  # type: ignore[misc]
        out,
        x_matrix,
        rows,
        cols,
        x_matrix.stride(0),
        BLOCK=block,
    )
    return out.view(*x.shape[:-1])


__all__ = [
    "relu_squared",
    "softcap_tanh_projection",
    "elementwise_add",
    "elementwise_mul",
    "elementwise_where",
    "elementwise_lerp",
    "rowwise_l2_norm",
]
