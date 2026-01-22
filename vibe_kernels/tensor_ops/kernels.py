# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Triton JIT kernels for tensor operations like stack/cat.

These kernels are framework-agnostic and can be compiled to PTX
for use with any tensor framework that supports CUDA.
"""

import triton
import triton.language as tl


@triton.jit
def stack_kernel(
    out_ptr,
    src_ptr,          # Pointer to the source tensor for this slice
    out_stride0,      # Stride along dim 0 (the stacked dimension)
    slice_idx,        # Which slice this program is writing
    slice_size,       # Total elements per slice
    BLOCK_SIZE: tl.constexpr,
):
    """Copy one tensor slice into the output at position slice_idx.
    
    This kernel handles one source tensor. Launch one program per slice.
    The caller is responsible for launching multiple kernels for multiple sources.
    """
    pid = tl.program_id(0)
    
    # Each program handles a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < slice_size
    
    # Load from source
    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    
    # Store to output at the correct slice position
    out_offset = slice_idx * out_stride0 + offsets
    tl.store(out_ptr + out_offset, vals, mask=mask)


@triton.jit
def cat_kernel(
    out_ptr,
    src_ptr,
    out_offset,       # Offset in output where this source starts
    src_size,         # Number of elements in this source
    BLOCK_SIZE: tl.constexpr,
):
    """Concatenate one tensor into output at given offset.
    
    Simple copy kernel for concatenation along dim 0.
    """
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < src_size
    
    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offset + offsets, vals, mask=mask)


@triton.jit
def arange_kernel(
    out_ptr,
    start,
    step,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill tensor with arange values: out[i] = start + i * step.
    
    GPU-native implementation without NumPy dependency.
    """
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute values: start + offset * step
    vals = start + offsets.to(tl.float32) * step
    
    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.jit
def add_one_mod_kernel(
    out_ptr,
    inp_ptr,
    mod_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute (x + 1) % mod_val for int64 tensors.
    
    GPU-native implementation for target generation.
    """
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(inp_ptr + offsets, mask=mask, other=0)
    
    # Compute (x + 1) % mod_val
    result = x + 1
    # For mod, use: result - mod_val * (result >= mod_val)
    # Since we're working with int64 and triton doesn't have native mod for int
    # We use the fact that result is in [1, mod_val]
    # result = result - mod_val when result == mod_val
    result = tl.where(result >= mod_val, result - mod_val, result)
    
    tl.store(out_ptr + offsets, result, mask=mask)


__all__ = [
    "stack_kernel",
    "cat_kernel",
    "arange_kernel",
    "add_one_mod_kernel",
]
