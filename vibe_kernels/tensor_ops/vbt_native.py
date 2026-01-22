# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure VibeTensor implementation of tensor operations (stack, cat).

This module provides stack/cat operations using ONLY VibeTensor,
with NO PyTorch dependency at runtime.

Usage:
    import vibetensor.torch as vt
    from vibe_kernels.tensor_ops import vbt_native as tensor_ops
    
    tensors = [vt.randn([4, 8]).cuda() for _ in range(3)]
    stacked = tensor_ops.stack(tensors, dim=0)  # [3, 4, 8]
"""

from __future__ import annotations

import threading
from typing import List, Tuple, Dict

import triton
import triton.language as tl


def _get_vbt_modules():
    """Lazy import VibeTensor modules."""
    import vibetensor.torch as vt
    from vibetensor import _C
    import vibetensor.triton as vt_triton
    return vt, _C, vt_triton


def _next_power_of_2(x: int) -> int:
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


# -----------------------------------------------------------------------------
# Kernel Cache
# -----------------------------------------------------------------------------

_kernel_cache: Dict[str, Tuple[int, int, int]] = {}
_cache_lock = threading.Lock()


def _get_add_one_mod_kernel(block_size: int, device_idx: int) -> Tuple[int, int]:
    """Get compiled add_one_mod kernel handle."""
    vt, _C, vt_triton = _get_vbt_modules()
    
    key = f"add_one_mod_{block_size}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            _, func_h, extra = _kernel_cache[key]
            return func_h, extra
    
    # Load kernel
    import os
    import importlib.util
    kernels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.py")
    spec = importlib.util.spec_from_file_location("tensor_kernels_mod", kernels_path)
    kernels_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels_mod)
    add_one_mod_kernel = kernels_mod.add_one_mod_kernel
    
    # Compile: out_ptr, inp_ptr, mod_val, n_elements
    signature = "*i64,*i64,i64,i32"
    meta = {"BLOCK_SIZE": block_size}
    
    ptx, entry, shmem = vt_triton._compile_to_ptx(
        add_one_mod_kernel,
        signature=signature,
        meta=meta,
        num_warps=4,
    )
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    sig_tokens = [t.strip() for t in signature.split(",")]
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _kernel_cache[key] = (mod_h, func_h, extra_params)
    
    return func_h, extra_params


def _get_arange_kernel(block_size: int, device_idx: int) -> Tuple[int, int]:
    """Get compiled arange kernel handle."""
    vt, _C, vt_triton = _get_vbt_modules()
    
    key = f"arange_{block_size}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            _, func_h, extra = _kernel_cache[key]
            return func_h, extra
    
    # Load kernel
    import os
    import importlib.util
    kernels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.py")
    spec = importlib.util.spec_from_file_location("tensor_kernels_arange", kernels_path)
    kernels_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels_mod)
    arange_kernel = kernels_mod.arange_kernel
    
    # Compile: out_ptr, start, step, n_elements
    signature = "*fp32,fp32,fp32,i32"
    meta = {"BLOCK_SIZE": block_size}
    
    ptx, entry, shmem = vt_triton._compile_to_ptx(
        arange_kernel,
        signature=signature,
        meta=meta,
        num_warps=4,
    )
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    sig_tokens = [t.strip() for t in signature.split(",")]
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _kernel_cache[key] = (mod_h, func_h, extra_params)
    
    return func_h, extra_params


def _get_stack_kernel(block_size: int, device_idx: int) -> Tuple[int, int]:
    """Get compiled stack kernel handle."""
    vt, _C, vt_triton = _get_vbt_modules()
    
    key = f"stack_{block_size}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            _, func_h, extra = _kernel_cache[key]
            return func_h, extra
    
    # Load kernel
    import os
    import importlib.util
    kernels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.py")
    spec = importlib.util.spec_from_file_location("tensor_kernels", kernels_path)
    kernels_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels_mod)
    stack_kernel = kernels_mod.stack_kernel
    
    # Compile
    signature = "*fp32,*fp32,i32,i32,i32"
    meta = {"BLOCK_SIZE": block_size}
    
    ptx, entry, shmem = vt_triton._compile_to_ptx(
        stack_kernel,
        signature=signature,
        meta=meta,
        num_warps=4,
    )
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    sig_tokens = [t.strip() for t in signature.split(",")]
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _kernel_cache[key] = (mod_h, func_h, extra_params)
    
    return func_h, extra_params


def _get_cat_kernel(block_size: int, device_idx: int) -> Tuple[int, int]:
    """Get compiled cat kernel handle."""
    vt, _C, vt_triton = _get_vbt_modules()

    key = f"cat_{block_size}_{device_idx}"
    with _cache_lock:
        if key in _kernel_cache:
            _, func_h, extra = _kernel_cache[key]
            return func_h, extra

    import os
    import importlib.util

    kernels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.py")
    spec = importlib.util.spec_from_file_location("tensor_kernels_cat", kernels_path)
    kernels_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels_mod)
    cat_kernel = kernels_mod.cat_kernel

    signature = "*fp32,*fp32,i32,i32"
    meta = {"BLOCK_SIZE": block_size}

    ptx, entry, shmem = vt_triton._compile_to_ptx(
        cat_kernel,
        signature=signature,
        meta=meta,
        num_warps=4,
    )

    total_params = vt_triton._count_entry_params(ptx, entry)
    sig_tokens = [t.strip() for t in signature.split(",")]
    extra_params = max(0, total_params - len(sig_tokens))

    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)

    with _cache_lock:
        _kernel_cache[key] = (mod_h, func_h, extra_params)

    return func_h, extra_params


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def stack(tensors: List, dim: int = 0):
    """Stack a list of tensors along a new dimension.
    
    Pure VibeTensor implementation - NO PyTorch/NumPy dependency.
    
    Args:
        tensors: List of VibeTensor tensors, all same shape, float32, CUDA
        dim: Dimension to insert (must be 0 for now)
        
    Returns:
        Stacked tensor with shape [len(tensors), *tensor_shape]
        
    Example:
        >>> tensors = [vt.randn([4, 8]).cuda() for _ in range(3)]
        >>> out = stack(tensors, dim=0)  # shape: [3, 4, 8]
    """
    vt, _C, _ = _get_vbt_modules()
    
    if len(tensors) == 0:
        raise ValueError("stack requires at least one tensor")
    
    if dim != 0:
        raise NotImplementedError("stack currently only supports dim=0")
    
    # Get tensor properties from first tensor
    first = tensors[0]
    tensor_sizes = tuple(int(s) for s in first.sizes)
    tensor_dtype = str(first.dtype)
    tensor_device = first.device
    
    if tensor_device[0] != 2:
        raise RuntimeError("stack requires CUDA tensors")
    if tensor_dtype != "float32":
        raise TypeError("stack requires float32 tensors")
    
    device_idx = int(tensor_device[1])
    num_tensors = len(tensors)
    
    # Calculate slice size (total elements per tensor)
    slice_size = 1
    for s in tensor_sizes:
        slice_size *= s
    
    # Output shape: [num_tensors, *tensor_sizes]
    out_shape = [num_tensors] + list(tensor_sizes)
    out_stride0 = slice_size  # Stride along stacked dimension
    
    # Allocate output
    out = _C._cuda_empty(out_shape, tensor_dtype, device_idx)
    
    # Get kernel
    block_size = min(1024, _next_power_of_2(max(32, slice_size)))
    func_h, extra_params = _get_stack_kernel(block_size, device_idx)
    
    # Launch kernel for each tensor
    grid_size = (slice_size + block_size - 1) // block_size
    grid = (grid_size, 1, 1)
    block = (128, 1, 1)  # num_warps=4
    shmem = 0
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    for i, tensor in enumerate(tensors):
        # Validate same shape and device
        t_sizes = tuple(int(s) for s in tensor.sizes)
        if t_sizes != tensor_sizes:
            raise ValueError(f"All tensors must have same shape, got {t_sizes} vs {tensor_sizes}")
        if tensor.device[0] != 2 or int(tensor.device[1]) != device_idx:
            raise ValueError("All tensors must be on same CUDA device")
        
        # Ensure contiguous
        tensor = tensor.contiguous()
        
        # Launch copy kernel for this slice
        args = [out, tensor, out_stride0, i, slice_size]
        args.extend([None] * extra_params)
        
        _C._cuda_launch(func_h, grid, block, shmem, stream, args)
    
    return out


def cat(tensors: List, dim: int = 0):
    """Concatenate tensors along an existing dimension.
    
    Pure VibeTensor implementation - NO PyTorch/NumPy dependency.
    
    Args:
        tensors: List of VibeTensor tensors, same shape except along dim
        dim: Dimension to concatenate along (must be 0 for now)
        
    Returns:
        Concatenated tensor
    """
    vt, _C, _ = _get_vbt_modules()
    
    if len(tensors) == 0:
        raise ValueError("cat requires at least one tensor")
    
    if dim != 0:
        raise NotImplementedError("cat currently only supports dim=0")
    
    first = tensors[0]
    tensor_dtype = str(first.dtype)
    tensor_device = first.device
    
    if tensor_device[0] != 2:
        raise RuntimeError("cat requires CUDA tensors")
    if tensor_dtype != "float32":
        raise TypeError("cat requires float32 tensors")
    
    device_idx = int(tensor_device[1])
    
    # Calculate total size along dim 0
    total_dim0 = sum(int(t.sizes[0]) for t in tensors)
    other_dims = list(int(s) for s in first.sizes[1:])
    
    out_shape = [total_dim0] + other_dims
    out = _C._cuda_empty(out_shape, tensor_dtype, device_idx)
    
    inner_size = 1
    for s in other_dims:
        inner_size *= s
    
    stream = _C._cuda_stream_handle_current_for_device(device_idx)

    offset = 0
    for tensor in tensors:
        t_sizes = tuple(int(s) for s in tensor.sizes)
        if list(t_sizes[1:]) != other_dims:
            raise ValueError("cat: all tensors must match in non-concatenated dims")
        if tensor.device[0] != 2 or int(tensor.device[1]) != device_idx:
            raise ValueError("cat: all tensors must be on same CUDA device")
        if str(tensor.dtype) != tensor_dtype:
            raise TypeError("cat: all tensors must have same dtype")

        tensor = tensor.contiguous()
        src = tensor.reshape([-1])
        src_size = int(tensor.sizes[0]) * inner_size

        block_size = min(1024, _next_power_of_2(max(32, src_size)))
        func_h, extra_params = _get_cat_kernel(block_size, device_idx)

        grid_size = (src_size + block_size - 1) // block_size
        grid = (grid_size, 1, 1)
        block_dim = (128, 1, 1)

        args = [out, src, int(offset), int(src_size)]
        args.extend([None] * extra_params)

        _C._cuda_launch(func_h, grid, block_dim, 0, stream, args)
        offset += src_size
    
    return out


def arange(start, end=None, step=1.0, device_idx: int = 0):
    """Create a tensor with values from start to end with given step.
    
    Pure VibeTensor implementation - NO NumPy dependency.
    
    Args:
        start: Start value (or end if end is None)
        end: End value (exclusive)
        step: Step size (default 1.0)
        device_idx: CUDA device index (default 0)
        
    Returns:
        1D float32 tensor on CUDA
        
    Example:
        >>> out = arange(0, 10, 2)  # [0, 2, 4, 6, 8]
        >>> out = arange(5)  # [0, 1, 2, 3, 4]
    """
    vt, _C, _ = _get_vbt_modules()
    
    # Handle arange(n) case
    if end is None:
        start, end = 0.0, float(start)
    else:
        start, end = float(start), float(end)
    step = float(step)
    
    if step == 0:
        raise ValueError("step cannot be zero")
    
    # Calculate number of elements
    if step > 0:
        n_elements = max(0, int((end - start + step - 1e-9) // step))
    else:
        n_elements = max(0, int((start - end - step - 1e-9) // (-step)))
    
    if n_elements == 0:
        # Return empty tensor
        return _C._cuda_empty([0], "float32", device_idx)
    
    # Allocate output
    out = _C._cuda_empty([n_elements], "float32", device_idx)
    
    # Get kernel
    block_size = min(1024, _next_power_of_2(max(32, n_elements)))
    func_h, extra_params = _get_arange_kernel(block_size, device_idx)
    
    # Launch kernel
    grid_size = (n_elements + block_size - 1) // block_size
    grid = (grid_size, 1, 1)
    block = (128, 1, 1)  # num_warps=4
    shmem = 0
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    # Args: out_ptr, start, step, n_elements
    args = [out, start, step, n_elements]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block, shmem, stream, args)
    
    return out


def add_one_mod(inp, mod_val: int, device_idx: int = 0):
    """Compute (x + 1) % mod_val for int64 tensors.
    
    Pure VibeTensor implementation - NO NumPy dependency.
    
    Args:
        inp: Input int64 tensor on CUDA
        mod_val: Modulo value
        device_idx: CUDA device index (default 0)
        
    Returns:
        Output int64 tensor with (inp + 1) % mod_val
    """
    vt, _C, _ = _get_vbt_modules()
    
    inp_sizes = tuple(int(s) for s in inp.sizes)
    inp_dtype = str(inp.dtype)
    inp_device = inp.device
    
    if inp_device[0] != 2:
        raise RuntimeError("add_one_mod requires CUDA tensors")
    if inp_dtype != "int64":
        raise TypeError("add_one_mod requires int64 tensors")
    
    device_idx = int(inp_device[1])
    
    # Flatten for kernel
    n_elements = 1
    for s in inp_sizes:
        n_elements *= s
    
    # Allocate output (same shape as input)
    out = _C._cuda_empty(list(inp_sizes), inp_dtype, device_idx)
    
    # Contiguous input
    inp_flat = inp.reshape([-1])
    
    # Get kernel
    block_size = min(1024, _next_power_of_2(max(32, n_elements)))
    func_h, extra_params = _get_add_one_mod_kernel(block_size, device_idx)
    
    # Launch kernel
    grid_size = (n_elements + block_size - 1) // block_size
    grid = (grid_size, 1, 1)
    block = (128, 1, 1)  # num_warps=4
    shmem = 0
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    # Args: out_ptr, inp_ptr, mod_val, n_elements
    args = [out, inp_flat, mod_val, n_elements]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block, shmem, stream, args)
    
    return out


__all__ = [
    "stack",
    "cat",
    "arange",
    "add_one_mod",
]
