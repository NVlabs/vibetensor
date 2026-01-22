# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton kernels for gather and scatter_add operations.

These kernels provide efficient indexed memory access operations:
- gather: Select elements from a tensor along a dimension using an index tensor
- scatter_add: Scatter values into a tensor with accumulation using an index tensor
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor


def _next_power_of_2(x: int) -> int:
    """Round up to the next power of 2."""
    return 1 << (x - 1).bit_length()


# -----------------------------------------------------------------------------
# Gather Kernel
# -----------------------------------------------------------------------------

@triton.jit
def _gather_1d_kernel(
    out_ptr,
    src_ptr,
    idx_ptr,
    src_stride,      # stride of src along the gather dimension
    out_stride,      # stride of out along the gather dimension  
    num_indices,     # number of indices (output size along gather dim)
    inner_size,      # product of dimensions after the gather dimension
    BLOCK_SIZE: tl.constexpr,
):
    """1D gather kernel: out[i, j] = src[idx[i], j] for gather along dim 0."""
    pid = tl.program_id(0)
    
    # Each program handles one index
    if pid >= num_indices:
        return
    
    # Load the index for this row
    idx = tl.load(idx_ptr + pid)
    
    # Process inner dimension in blocks
    for start in range(0, inner_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < inner_size
        
        # Load from src at idx position
        src_offset = idx * src_stride + offsets
        vals = tl.load(src_ptr + src_offset, mask=mask, other=0.0)
        
        # Store to output at pid position
        out_offset = pid * out_stride + offsets
        tl.store(out_ptr + out_offset, vals, mask=mask)


@triton.jit
def _gather_2d_kernel(
    out_ptr,
    src_ptr,
    idx_ptr,
    batch_size,
    src_dim_size,    # size of src along gather dimension
    idx_dim_size,    # size of idx (output size along gather dim)
    inner_size,      # product of dimensions after gather dimension
    src_batch_stride,
    src_gather_stride,
    idx_batch_stride,
    out_batch_stride,
    out_gather_stride,
    BLOCK_INNER: tl.constexpr,
):
    """2D gather kernel for batched gather: out[b, i, j] = src[b, idx[b, i], j]."""
    pid_batch = tl.program_id(0)
    pid_idx = tl.program_id(1)
    
    if pid_batch >= batch_size or pid_idx >= idx_dim_size:
        return
    
    # Load the index
    idx_offset = pid_batch * idx_batch_stride + pid_idx
    idx = tl.load(idx_ptr + idx_offset)
    
    # Clamp index to valid range (for safety)
    idx = tl.minimum(tl.maximum(idx, 0), src_dim_size - 1)
    
    # Process inner dimension in blocks
    for start in range(0, inner_size, BLOCK_INNER):
        offsets = start + tl.arange(0, BLOCK_INNER)
        mask = offsets < inner_size
        
        # Load from src
        src_offset = pid_batch * src_batch_stride + idx * src_gather_stride + offsets
        vals = tl.load(src_ptr + src_offset, mask=mask, other=0.0)
        
        # Store to output
        out_offset = pid_batch * out_batch_stride + pid_idx * out_gather_stride + offsets
        tl.store(out_ptr + out_offset, vals, mask=mask)


def gather(
    src: Tensor,
    dim: int,
    index: Tensor,
) -> Tensor:
    """Gather values from src along dim using index tensor.
    
    Args:
        src: Source tensor of shape (..., src_dim, ...)
        dim: Dimension along which to gather
        index: Index tensor of shape (..., idx_dim, ...) with int64 indices
        
    Returns:
        Output tensor with same shape as index (with src's trailing dims)
        
    Example:
        >>> src = torch.randn(4, 8, device='cuda')
        >>> idx = torch.tensor([0, 2, 1], device='cuda')
        >>> out = gather(src, 0, idx)  # shape: (3, 8)
    """
    if src.device.type != "cuda":
        raise RuntimeError("gather requires CUDA tensors")
    if index.device != src.device:
        raise RuntimeError("src and index must be on the same device")
    if index.dtype not in (torch.int32, torch.int64):
        raise TypeError("index must be int32 or int64")
    
    # Normalize dim
    ndim = src.ndim
    if dim < 0:
        dim = ndim + dim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim {dim} out of range for tensor with {ndim} dimensions")
    
    src = src.contiguous()
    index = index.contiguous()
    
    # Compute shapes
    src_shape = list(src.shape)
    idx_shape = list(index.shape)
    
    # Output shape: replace src's dim with index's corresponding dim
    # For simplicity, we handle the common case where index is 1D or matches batch dims
    if index.ndim == 1:
        # Simple 1D index case: out[i, ...] = src[idx[i], ...]
        out_shape = [index.shape[0]] + src_shape[dim+1:]
        out = torch.empty(out_shape, dtype=src.dtype, device=src.device)
        
        num_indices = index.shape[0]
        inner_size = 1
        for d in range(dim + 1, ndim):
            inner_size *= src_shape[d]
        
        src_stride = src.stride(dim) if dim < ndim else 1
        out_stride = out.stride(0) if out.ndim > 0 else 1
        
        block_size = min(1024, _next_power_of_2(max(32, inner_size)))
        grid = (num_indices,)
        
        _gather_1d_kernel[grid](
            out, src, index,
            src_stride, out_stride,
            num_indices, inner_size,
            BLOCK_SIZE=block_size,
        )
    else:
        # Batched case: assume index has same batch dims as src
        # out[b, i, j] = src[b, idx[b, i], j]
        batch_size = 1
        for d in range(dim):
            batch_size *= src_shape[d]
        
        src_dim_size = src_shape[dim]
        idx_dim_size = idx_shape[dim] if dim < index.ndim else 1
        
        inner_size = 1
        for d in range(dim + 1, ndim):
            inner_size *= src_shape[d]
        
        out_shape = src_shape[:dim] + [idx_dim_size] + src_shape[dim+1:]
        out = torch.empty(out_shape, dtype=src.dtype, device=src.device)
        
        # Compute strides
        src_batch_stride = src.stride(0) if dim > 0 else 0
        src_gather_stride = src.stride(dim)
        idx_batch_stride = index.stride(0) if dim > 0 and index.ndim > 1 else 0
        out_batch_stride = out.stride(0) if dim > 0 else 0
        out_gather_stride = out.stride(dim) if dim < out.ndim else 1
        
        block_inner = min(256, _next_power_of_2(max(32, inner_size)))
        grid = (batch_size, idx_dim_size)
        
        _gather_2d_kernel[grid](
            out, src, index,
            batch_size, src_dim_size, idx_dim_size, inner_size,
            src_batch_stride, src_gather_stride,
            idx_batch_stride,
            out_batch_stride, out_gather_stride,
            BLOCK_INNER=block_inner,
        )
    
    return out


# -----------------------------------------------------------------------------
# Scatter Add Kernel
# -----------------------------------------------------------------------------

@triton.jit
def _scatter_add_1d_kernel(
    out_ptr,
    src_ptr,
    idx_ptr,
    num_indices,     # number of elements to scatter
    out_size,        # size of output along scatter dimension
    inner_size,      # product of dimensions after scatter dimension
    out_stride,      # stride of output along scatter dimension
    src_stride,      # stride of src along scatter dimension
    BLOCK_INNER: tl.constexpr,
):
    """1D scatter_add kernel: out[idx[i], j] += src[i, j]."""
    pid = tl.program_id(0)
    
    if pid >= num_indices:
        return
    
    # Load the target index
    idx = tl.load(idx_ptr + pid)
    
    # Clamp to valid range
    idx = tl.minimum(tl.maximum(idx, 0), out_size - 1)
    
    # Process inner dimension in blocks
    for start in range(0, inner_size, BLOCK_INNER):
        offsets = start + tl.arange(0, BLOCK_INNER)
        mask = offsets < inner_size
        
        # Load source values
        src_offset = pid * src_stride + offsets
        vals = tl.load(src_ptr + src_offset, mask=mask, other=0.0)
        
        # Atomic add to output
        out_offset = idx * out_stride + offsets
        tl.atomic_add(out_ptr + out_offset, vals, mask=mask)


@triton.jit
def _scatter_add_2d_kernel(
    out_ptr,
    src_ptr,
    idx_ptr,
    batch_size,
    num_indices,     # indices per batch
    out_dim_size,    # size of output along scatter dimension
    inner_size,
    out_batch_stride,
    out_scatter_stride,
    src_batch_stride,
    src_scatter_stride,
    idx_batch_stride,
    BLOCK_INNER: tl.constexpr,
):
    """2D scatter_add kernel: out[b, idx[b, i], j] += src[b, i, j]."""
    pid_batch = tl.program_id(0)
    pid_idx = tl.program_id(1)
    
    if pid_batch >= batch_size or pid_idx >= num_indices:
        return
    
    # Load the target index
    idx_offset = pid_batch * idx_batch_stride + pid_idx
    idx = tl.load(idx_ptr + idx_offset)
    
    # Clamp to valid range
    idx = tl.minimum(tl.maximum(idx, 0), out_dim_size - 1)
    
    # Process inner dimension in blocks
    for start in range(0, inner_size, BLOCK_INNER):
        offsets = start + tl.arange(0, BLOCK_INNER)
        mask = offsets < inner_size
        
        # Load source values
        src_offset = pid_batch * src_batch_stride + pid_idx * src_scatter_stride + offsets
        vals = tl.load(src_ptr + src_offset, mask=mask, other=0.0)
        
        # Atomic add to output
        out_offset = pid_batch * out_batch_stride + idx * out_scatter_stride + offsets
        tl.atomic_add(out_ptr + out_offset, vals, mask=mask)


def scatter_add(
    out: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
) -> Tensor:
    """Scatter add values from src into out along dim using index tensor.
    
    This operation performs: out[idx[i], j] += src[i, j] for all i, j
    
    Args:
        out: Output tensor to scatter into (modified in-place)
        dim: Dimension along which to scatter
        index: Index tensor with int64 indices
        src: Source tensor with values to scatter
        
    Returns:
        The modified output tensor (same as out, modified in-place)
        
    Example:
        >>> out = torch.zeros(4, 8, device='cuda')
        >>> idx = torch.tensor([0, 2, 0], device='cuda')  # Note: 0 appears twice
        >>> src = torch.ones(3, 8, device='cuda')
        >>> scatter_add(out, 0, idx, src)  # out[0] = 2, out[2] = 1
    """
    if out.device.type != "cuda":
        raise RuntimeError("scatter_add requires CUDA tensors")
    if src.device != out.device or index.device != out.device:
        raise RuntimeError("out, src, and index must be on the same device")
    if index.dtype not in (torch.int32, torch.int64):
        raise TypeError("index must be int32 or int64")
    
    # Normalize dim
    ndim = out.ndim
    if dim < 0:
        dim = ndim + dim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim {dim} out of range for tensor with {ndim} dimensions")
    
    out = out.contiguous()
    src = src.contiguous()
    index = index.contiguous()
    
    out_shape = list(out.shape)
    src_shape = list(src.shape)
    idx_shape = list(index.shape)
    
    if index.ndim == 1:
        # Simple 1D index case
        num_indices = index.shape[0]
        out_size = out_shape[dim]
        
        inner_size = 1
        for d in range(dim + 1, ndim):
            inner_size *= out_shape[d]
        
        out_stride = out.stride(dim)
        src_stride = src.stride(dim) if dim < src.ndim else 1
        
        block_inner = min(256, _next_power_of_2(max(32, inner_size)))
        grid = (num_indices,)
        
        _scatter_add_1d_kernel[grid](
            out, src, index,
            num_indices, out_size, inner_size,
            out_stride, src_stride,
            BLOCK_INNER=block_inner,
        )
    else:
        # Batched case
        batch_size = 1
        for d in range(dim):
            batch_size *= out_shape[d]
        
        num_indices = idx_shape[dim] if dim < index.ndim else 1
        out_dim_size = out_shape[dim]
        
        inner_size = 1
        for d in range(dim + 1, ndim):
            inner_size *= out_shape[d]
        
        out_batch_stride = out.stride(0) if dim > 0 else 0
        out_scatter_stride = out.stride(dim)
        src_batch_stride = src.stride(0) if dim > 0 else 0
        src_scatter_stride = src.stride(dim) if dim < src.ndim else 1
        idx_batch_stride = index.stride(0) if dim > 0 and index.ndim > 1 else 0
        
        block_inner = min(256, _next_power_of_2(max(32, inner_size)))
        grid = (batch_size, num_indices)
        
        _scatter_add_2d_kernel[grid](
            out, src, index,
            batch_size, num_indices, out_dim_size, inner_size,
            out_batch_stride, out_scatter_stride,
            src_batch_stride, src_scatter_stride,
            idx_batch_stride,
            BLOCK_INNER=block_inner,
        )
    
    return out


def scatter_add_(
    out: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
) -> Tensor:
    """In-place version of scatter_add. See scatter_add for details."""
    return scatter_add(out, dim, index, src)


# Autograd wrapper for gather with backward support
class _GatherFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src: Tensor, dim: int, index: Tensor) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(index)
        ctx.dim = dim
        ctx.src_shape = src.shape
        return gather(src, dim, index)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        (index,) = ctx.saved_tensors
        dim = ctx.dim
        src_shape = ctx.src_shape
        
        # Backward of gather is scatter_add
        grad_src = torch.zeros(src_shape, dtype=grad_output.dtype, device=grad_output.device)
        scatter_add(grad_src, dim, index, grad_output)
        
        return grad_src, None, None


def gather_with_grad(src: Tensor, dim: int, index: Tensor) -> Tensor:
    """Gather with autograd support.
    
    The backward pass uses scatter_add to accumulate gradients.
    """
    return _GatherFn.apply(src, dim, index)


__all__ = [
    "gather",
    "gather_with_grad",
    "scatter_add",
    "scatter_add_",
]
