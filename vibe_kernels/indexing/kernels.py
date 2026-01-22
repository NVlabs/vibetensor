# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Triton JIT kernels for gather and scatter_add.

These kernels are framework-agnostic and can be compiled to PTX
for use with any tensor framework that supports CUDA.
"""

import triton
import triton.language as tl


@triton.jit
def gather_1d_kernel(
    src_ptr,
    idx_ptr,
    out_ptr,
    src_stride,      # stride of src along the gather dimension
    out_stride,      # stride of out along the gather dimension  
    num_indices,     # number of indices (output size along gather dim)
    inner_size,      # product of dimensions after the gather dimension
    BLOCK_SIZE: tl.constexpr,
):
    """1D gather kernel: out[i, j] = src[idx[i], j] for gather along dim 0.
    
    This kernel gathers rows from src based on indices in idx.
    Each program instance handles one index (one output row).
    """
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
def scatter_add_1d_kernel(
    src_ptr,
    idx_ptr,
    out_ptr,
    num_indices,     # number of elements to scatter
    out_size,        # size of output along scatter dimension
    inner_size,      # product of dimensions after scatter dimension
    out_stride,      # stride of output along scatter dimension
    src_stride,      # stride of src along scatter dimension
    BLOCK_INNER: tl.constexpr,
):
    """1D scatter_add kernel: out[idx[i], j] += src[i, j].
    
    This kernel scatters rows from src into out based on indices,
    using atomic addition for thread-safe accumulation.
    Each program instance handles one source row.
    """
    pid = tl.program_id(0)
    
    if pid >= num_indices:
        return
    
    # Load the target index
    idx = tl.load(idx_ptr + pid)
    
    # Clamp to valid range (safety)
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
def gather_2d_kernel(
    src_ptr,
    idx_ptr,
    out_ptr,
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
    
    # Clamp index to valid range
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


@triton.jit
def scatter_add_2d_kernel(
    src_ptr,
    idx_ptr,
    out_ptr,
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


@triton.jit
def subtract_at_indices_kernel(
    data_ptr,
    idx_ptr,
    value,           # scalar value to subtract
    batch_size,
    dim_size,
    batch_stride,    # stride of data along batch dimension
    elem_stride,     # stride of data along element dimension
):
    """Subtract a scalar value at positions [i, idx[i]] for each row i.
    
    This is a simple kernel for: data[i, idx[i]] -= value for all i.
    Equivalent to: data = data - one_hot(idx) * value
    """
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    # Load the index for this row
    idx = tl.load(idx_ptr + pid)
    
    # Clamp to valid range (safety)
    idx = tl.minimum(tl.maximum(idx, 0), dim_size - 1)
    
    # Compute offset: data[pid, idx]
    offset = pid * batch_stride + idx * elem_stride
    
    # Load current value
    current = tl.load(data_ptr + offset)
    
    # Subtract and store
    tl.store(data_ptr + offset, current - value)


__all__ = [
    "gather_1d_kernel",
    "scatter_add_1d_kernel",
    "gather_2d_kernel",
    "scatter_add_2d_kernel",
    "subtract_at_indices_kernel",
]
