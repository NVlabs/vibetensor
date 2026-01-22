# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure VibeTensor implementation of gather and scatter_add kernels.

This module provides gather/scatter_add operations using ONLY VibeTensor,
with NO PyTorch dependency. It uses VibeTensor's internal CUDA launch
mechanism to run Triton-compiled PTX kernels.

Usage:
    import vibetensor.torch as vt
    from vibe_kernels.indexing import vbt_native as idx_ops
    
    src = vt.randn((4, 8), dtype="float32", device="cuda")
    idx = vt.tensor([0, 2, 1], dtype="int64", device="cuda")
    out = idx_ops.gather(src, 0, idx)
"""

from __future__ import annotations

import threading
from typing import Tuple, Dict, Any


def _next_power_of_2(x: int) -> int:
    """Round up to the next power of 2."""
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


def _get_vbt_modules():
    """Lazy import VibeTensor modules."""
    import vibetensor.torch as vt
    from vibetensor import _C
    import vibetensor.triton as vt_triton
    return vt, _C, vt_triton


def _compute_contiguous_strides(sizes: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute row-major (C-contiguous) strides for given sizes."""
    if len(sizes) == 0:
        return ()
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    return tuple(strides)


# -----------------------------------------------------------------------------
# Kernel Cache
# -----------------------------------------------------------------------------

_kernel_cache: Dict[str, Tuple[int, int, int]] = {}  # key -> (mod_handle, func_handle, extra_params)
_cache_lock = threading.Lock()


def _compile_to_ptx_entry(vt_triton, kernel, *, signature: str, meta: Dict[str, Any], num_warps: int) -> Tuple[str, str]:
    compiled = vt_triton._compile_to_ptx(  # type: ignore[attr-defined]
        kernel,
        signature=signature,
        meta=meta,
        num_warps=num_warps,
    )
    if isinstance(compiled, tuple) and len(compiled) >= 2:
        return compiled[0], compiled[1]
    raise RuntimeError("vibetensor.triton._compile_to_ptx returned an unexpected value")


def _get_gather_kernel(block_size: int, device_idx: int) -> Tuple[int, int]:
    """Get function handle and extra param count for gather kernel."""
    vt, _C, vt_triton = _get_vbt_modules()
    
    key = f"gather_{block_size}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            _, func_h, extra = _kernel_cache[key]
            return func_h, extra
    
    # Compile kernel
    from .kernels import gather_1d_kernel
    
    signature = "*fp32,*i64,*fp32,i32,i32,i32,i32"
    meta = {"BLOCK_SIZE": block_size}
    
    ptx, entry = _compile_to_ptx_entry(
        vt_triton, gather_1d_kernel, signature=signature, meta=meta, num_warps=4
    )
    
    # Count extra params (Triton 3.5+ scratch pointers)
    total_params = vt_triton._count_entry_params(ptx, entry)
    sig_tokens = [t.strip() for t in signature.split(",")]
    extra_params = max(0, total_params - len(sig_tokens))
    
    # Load module
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _kernel_cache[key] = (mod_h, func_h, extra_params)
    
    return func_h, extra_params


def _get_gather_2d_kernel(block_inner: int, device_idx: int) -> Tuple[int, int]:
    """Get function handle and extra param count for 2D gather kernel."""
    vt, _C, vt_triton = _get_vbt_modules()
    
    key = f"gather2d_{block_inner}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            _, func_h, extra = _kernel_cache[key]
            return func_h, extra
    
    # Compile kernel - use absolute import for dynamic loading
    import os
    import importlib.util
    kernels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.py")
    spec = importlib.util.spec_from_file_location("indexing_kernels", kernels_path)
    kernels_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels_mod)
    gather_2d_kernel = kernels_mod.gather_2d_kernel
    
    # batch_size, src_dim_size, idx_dim_size, inner_size,
    # src_batch_stride, src_gather_stride, idx_batch_stride,
    # out_batch_stride, out_gather_stride
    signature = "*fp32,*i64,*fp32,i32,i32,i32,i32,i32,i32,i32,i32,i32"
    meta = {"BLOCK_INNER": block_inner}
    
    compiled = vt_triton._compile_to_ptx(  # type: ignore[attr-defined]
        gather_2d_kernel,
        signature=signature,
        meta=meta,
        num_warps=4,
    )
    if not (isinstance(compiled, tuple) and len(compiled) >= 2):
        raise RuntimeError("vibetensor.triton._compile_to_ptx returned an unexpected value")
    ptx, entry = compiled[0], compiled[1]
    
    # Count extra params
    total_params = vt_triton._count_entry_params(ptx, entry)
    sig_tokens = [t.strip() for t in signature.split(",")]
    extra_params = max(0, total_params - len(sig_tokens))
    
    # Load module
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _kernel_cache[key] = (mod_h, func_h, extra_params)
    
    return func_h, extra_params


def _get_scatter_kernel(block_inner: int, device_idx: int) -> Tuple[int, int]:
    """Get function handle and extra param count for scatter_add kernel."""
    vt, _C, vt_triton = _get_vbt_modules()
    
    key = f"scatter_{block_inner}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            _, func_h, extra = _kernel_cache[key]
            return func_h, extra
    
    # Compile kernel
    from .kernels import scatter_add_1d_kernel
    
    signature = "*fp32,*i64,*fp32,i32,i32,i32,i32,i32"
    meta = {"BLOCK_INNER": block_inner}
    
    ptx, entry = _compile_to_ptx_entry(
        vt_triton, scatter_add_1d_kernel, signature=signature, meta=meta, num_warps=4
    )
    
    # Count extra params
    total_params = vt_triton._count_entry_params(ptx, entry)
    sig_tokens = [t.strip() for t in signature.split(",")]
    extra_params = max(0, total_params - len(sig_tokens))
    
    # Load module
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _kernel_cache[key] = (mod_h, func_h, extra_params)
    
    return func_h, extra_params


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def gather(src, dim: int, index):
    """Gather values from src along dim using index tensor.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    
    Args:
        src: Source tensor (VibeTensor CUDA tensor, float32)
        dim: Dimension along which to gather (must be 0)
        index: Index tensor (VibeTensor CUDA tensor, int64)
        
    Returns:
        Output tensor (VibeTensor CUDA tensor)
        
    Example:
        >>> import vibetensor.torch as vt
        >>> from vibe_kernels.indexing import vbt_native as idx_ops
        >>> 
        >>> src = vt.cuda.to_device(np.arange(32, dtype=np.float32).reshape(4, 8))
        >>> idx = vt.cuda.to_device(np.array([1, 3, 0], dtype=np.int64))
        >>> out = idx_ops.gather(src, 0, idx)  # shape: (3, 8)
    """
    vt, _C, _ = _get_vbt_modules()
    
    # Get tensor properties
    src_sizes = tuple(int(s) for s in src.sizes)
    src_strides = tuple(int(s) for s in src.strides)
    src_dtype = str(src.dtype)
    src_device = src.device
    
    idx_sizes = tuple(int(s) for s in index.sizes)
    idx_dtype = str(index.dtype)
    
    # Validate
    if src_device[0] != 2:  # kDLCUDA = 2
        raise RuntimeError("gather requires CUDA tensors")
    if index.device[0] != 2:
        raise RuntimeError("index must be on CUDA")
    if src_dtype != "float32":
        raise TypeError("src must be float32 (for now)")
    if idx_dtype != "int64":
        raise TypeError("index must be int64")
    
    ndim = len(src_sizes)
    if dim < 0:
        dim = ndim + dim
    if dim != 0:
        raise NotImplementedError("gather currently only supports dim=0")
    if len(idx_sizes) != 1:
        raise NotImplementedError("gather currently only supports 1D index")
    
    device_idx = int(src_device[1])
    
    # Compute dimensions
    num_indices = idx_sizes[0]
    inner_size = 1
    for d in range(dim + 1, ndim):
        inner_size *= src_sizes[d]
    
    src_stride = src_strides[dim] if dim < ndim else 1
    out_shape = [num_indices] + list(src_sizes[dim+1:])
    out_strides = _compute_contiguous_strides(tuple(out_shape))
    out_stride = out_strides[0] if len(out_strides) > 0 else 1
    
    # Allocate output
    out = _C._cuda_empty(out_shape, src_dtype, device_idx)
    
    # Get kernel handle
    block_size = min(1024, _next_power_of_2(max(32, inner_size)))
    func_handle, extra_params = _get_gather_kernel(block_size, device_idx)
    
    # Launch kernel
    grid = (num_indices, 1, 1)
    block = (128, 1, 1)  # num_warps=4 -> 128 threads
    shmem = 0
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    # Args: src_ptr, idx_ptr, out_ptr, src_stride, out_stride, num_indices, inner_size
    args = [src, index, out, src_stride, out_stride, num_indices, inner_size]
    args.extend([None] * extra_params)  # Triton 3.5+ scratch pointers
    
    _C._cuda_launch(func_handle, grid, block, shmem, stream, args)
    
    return out


def gather_dim1(src, index):
    """Gather values from src along dim=1 using 1D index tensor.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    
    This is a simplified gather for the common case:
        out[i] = src[i, index[i]]
    
    Args:
        src: Source tensor [batch, dim], float32, CUDA
        index: Index tensor [batch], int64, CUDA
        
    Returns:
        Output tensor [batch], float32, CUDA
    """
    vt, _C, _ = _get_vbt_modules()
    
    # Get tensor properties
    src_sizes = tuple(int(s) for s in src.sizes)
    src_strides = tuple(int(s) for s in src.strides)
    src_dtype = str(src.dtype)
    src_device = src.device
    
    idx_sizes = tuple(int(s) for s in index.sizes)
    idx_dtype = str(index.dtype)
    
    # Validate
    if src_device[0] != 2:
        raise RuntimeError("gather_dim1 requires CUDA tensors")
    if index.device[0] != 2:
        raise RuntimeError("index must be on CUDA")
    if src_dtype != "float32":
        raise TypeError("src must be float32")
    if idx_dtype != "int64":
        raise TypeError("index must be int64")
    if len(src_sizes) != 2:
        raise ValueError("src must be 2D [batch, dim]")
    if len(idx_sizes) != 1:
        raise ValueError("index must be 1D [batch]")
    
    device_idx = int(src_device[1])
    
    batch_size = src_sizes[0]
    src_dim_size = src_sizes[1]
    idx_dim_size = 1  # We're selecting 1 element per row
    inner_size = 1    # No inner dimensions
    
    # Strides
    src_batch_stride = src_strides[0]
    src_gather_stride = src_strides[1]  # stride along dim=1
    idx_batch_stride = 1  # index is 1D, stride is 1
    
    # Output: [batch, 1] but we'll squeeze to [batch]
    out = _C._cuda_empty([batch_size, 1], src_dtype, device_idx)
    out_batch_stride = 1
    out_gather_stride = 1
    
    # Get kernel handle
    block_inner = 32  # inner_size is 1, so small block
    func_handle, extra_params = _get_gather_2d_kernel(block_inner, device_idx)
    
    # Launch kernel
    grid = (batch_size, idx_dim_size, 1)
    block = (128, 1, 1)
    shmem = 0
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    # Args: src_ptr, idx_ptr, out_ptr, batch_size, src_dim_size, idx_dim_size, inner_size,
    #       src_batch_stride, src_gather_stride, idx_batch_stride, out_batch_stride, out_gather_stride
    args = [src, index, out,
            batch_size, src_dim_size, idx_dim_size, inner_size,
            src_batch_stride, src_gather_stride, idx_batch_stride,
            out_batch_stride, out_gather_stride]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_handle, grid, block, shmem, stream, args)
    
    # Squeeze to [batch]
    return out.reshape([batch_size])


def _get_scatter_2d_kernel(block_inner: int, device_idx: int) -> Tuple[int, int]:
    """Get function handle and extra param count for 2D scatter_add kernel."""
    vt, _C, vt_triton = _get_vbt_modules()
    
    key = f"scatter2d_{block_inner}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            _, func_h, extra = _kernel_cache[key]
            return func_h, extra
    
    # Compile kernel
    import os
    import importlib.util
    kernels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.py")
    spec = importlib.util.spec_from_file_location("indexing_kernels2", kernels_path)
    kernels_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels_mod)
    scatter_add_2d_kernel = kernels_mod.scatter_add_2d_kernel
    
    # batch_size, num_indices, out_dim_size, inner_size,
    # out_batch_stride, out_scatter_stride, src_batch_stride, src_scatter_stride, idx_batch_stride
    signature = "*fp32,*i64,*fp32,i32,i32,i32,i32,i32,i32,i32,i32,i32"
    meta = {"BLOCK_INNER": block_inner}
    
    ptx, entry, shmem = vt_triton._compile_to_ptx(
        scatter_add_2d_kernel,
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


def scatter_add_dim1(out, index, value):
    """Scatter add a scalar value at positions [i, index[i]] for each row.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    
    This is a simplified scatter for the common case:
        out[i, index[i]] += value
    
    Args:
        out: Output tensor [batch, dim], float32, CUDA (modified in-place)
        index: Index tensor [batch], int64, CUDA
        value: Scalar value to add (float)
        
    Returns:
        Modified output tensor
    """
    vt, _C, _ = _get_vbt_modules()
    
    out_sizes = tuple(int(s) for s in out.sizes)
    out_strides = tuple(int(s) for s in out.strides)
    out_dtype = str(out.dtype)
    out_device = out.device
    
    idx_sizes = tuple(int(s) for s in index.sizes)
    idx_dtype = str(index.dtype)
    
    if out_device[0] != 2:
        raise RuntimeError("scatter_add_dim1 requires CUDA tensors")
    if index.device[0] != 2:
        raise RuntimeError("index must be on CUDA")
    if out_dtype != "float32":
        raise TypeError("out must be float32")
    if idx_dtype != "int64":
        raise TypeError("index must be int64")
    if len(out_sizes) != 2:
        raise ValueError("out must be 2D [batch, dim]")
    if len(idx_sizes) != 1:
        raise ValueError("index must be 1D [batch]")
    
    device_idx = int(out_device[1])
    
    batch_size = out_sizes[0]
    out_dim_size = out_sizes[1]
    num_indices = 1  # One index per row
    inner_size = 1   # Scalar value
    
    # Create source tensor with the value
    src = _C._cuda_empty([batch_size, 1], out_dtype, device_idx)
    src.fill_(value)
    
    # Strides
    out_batch_stride = out_strides[0]
    out_scatter_stride = out_strides[1]
    src_batch_stride = 1
    src_scatter_stride = 1
    idx_batch_stride = 1
    
    # Reshape index to [batch, 1] for kernel
    index_2d = index.reshape([batch_size, 1])
    
    # Get kernel handle
    block_inner = 32
    func_handle, extra_params = _get_scatter_2d_kernel(block_inner, device_idx)
    
    # Launch kernel
    grid = (batch_size, num_indices, 1)
    block = (128, 1, 1)
    shmem = 0
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    # Args: src_ptr, idx_ptr, out_ptr, batch_size, num_indices, out_dim_size, inner_size,
    #       out_batch_stride, out_scatter_stride, src_batch_stride, src_scatter_stride, idx_batch_stride
    args = [src, index_2d, out,
            batch_size, num_indices, out_dim_size, inner_size,
            out_batch_stride, out_scatter_stride, src_batch_stride, src_scatter_stride, idx_batch_stride]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_handle, grid, block, shmem, stream, args)
    
    return out


def scatter_add(out, dim: int, index, src):
    """Scatter add values from src into out using index tensor.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    
    out[idx[i], j] += src[i, j] for all i, j
    
    Args:
        out: Output tensor (VibeTensor CUDA tensor, modified in-place)
        dim: Dimension along which to scatter (must be 0)
        index: Index tensor (VibeTensor CUDA tensor, int64)
        src: Source tensor (VibeTensor CUDA tensor, float32)
        
    Returns:
        The modified output tensor
        
    Example:
        >>> import vibetensor.torch as vt
        >>> from vibe_kernels.indexing import vbt_native as idx_ops
        >>> 
        >>> out = _C._cuda_zeros([4, 8], "float32", 0)
        >>> idx = vt.cuda.to_device(np.array([0, 2, 0], dtype=np.int64))
        >>> src = vt.cuda.to_device(np.ones((3, 8), dtype=np.float32))
        >>> idx_ops.scatter_add(out, 0, idx, src)  # out[0] = 2, out[2] = 1
    """
    vt, _C, _ = _get_vbt_modules()
    
    # Get tensor properties
    out_sizes = tuple(int(s) for s in out.sizes)
    out_strides = tuple(int(s) for s in out.strides)
    out_dtype = str(out.dtype)
    out_device = out.device
    
    src_sizes = tuple(int(s) for s in src.sizes)
    src_strides = tuple(int(s) for s in src.strides)
    
    idx_sizes = tuple(int(s) for s in index.sizes)
    idx_dtype = str(index.dtype)
    
    # Validate
    if out_device[0] != 2:
        raise RuntimeError("scatter_add requires CUDA tensors")
    if src.device[0] != 2 or index.device[0] != 2:
        raise RuntimeError("out, src, and index must be on CUDA")
    if out_dtype != "float32" or str(src.dtype) != "float32":
        raise TypeError("tensors must be float32 (for now)")
    if idx_dtype != "int64":
        raise TypeError("index must be int64")
    
    ndim = len(out_sizes)
    if dim < 0:
        dim = ndim + dim
    if dim != 0:
        raise NotImplementedError("scatter_add currently only supports dim=0")
    if len(idx_sizes) != 1:
        raise NotImplementedError("scatter_add currently only supports 1D index")
    
    device_idx = int(out_device[1])
    
    # Compute dimensions
    num_indices = idx_sizes[0]
    out_size = out_sizes[dim]
    inner_size = 1
    for d in range(dim + 1, ndim):
        inner_size *= out_sizes[d]
    
    out_stride = out_strides[dim]
    src_stride = src_strides[dim] if dim < len(src_strides) else 1
    
    # Get kernel handle
    block_inner = min(256, _next_power_of_2(max(32, inner_size)))
    func_handle, extra_params = _get_scatter_kernel(block_inner, device_idx)
    
    # Launch kernel
    grid = (num_indices, 1, 1)
    block = (128, 1, 1)  # num_warps=4 -> 128 threads
    shmem = 0
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    # Args: src_ptr, idx_ptr, out_ptr, num_indices, out_size, inner_size, out_stride, src_stride
    args = [src, index, out, num_indices, out_size, inner_size, out_stride, src_stride]
    args.extend([None] * extra_params)  # Triton 3.5+ scratch pointers
    
    _C._cuda_launch(func_handle, grid, block, shmem, stream, args)
    
    return out


def scatter_add_(out, dim: int, index, src):
    """In-place scatter_add. See scatter_add for details."""
    return scatter_add(out, dim, index, src)


def argmax(x, dim: int = -1):
    """Compute argmax along the specified dimension.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    Currently implemented via CPU transfer (can optimize later).
    
    Args:
        x: Input tensor, float32, CUDA
        dim: Dimension to find argmax
        
    Returns:
        Output tensor with argmax indices (int64)
    """
    vt, _C, _ = _get_vbt_modules()
    import numpy as np
    
    # Get tensor properties
    x_sizes = tuple(int(s) for s in x.sizes)
    x_dtype = str(x.dtype)
    x_device = x.device
    
    # Validate
    if x_device[0] != 2:
        raise RuntimeError("argmax requires CUDA tensors")
    
    ndim = len(x_sizes)
    if dim < 0:
        dim = ndim + dim
    
    device_idx = int(x_device[1])
    
    # For now, do argmax on CPU (can optimize later with a Triton kernel)
    x_np = _C._cuda_d2h_copy_numpy_sync(x)
    result_np = np.argmax(x_np, axis=dim).astype(np.int64)
    
    # Transfer back to CUDA
    return _C._cuda_h2d_alloc_copy(result_np, 'int64', device_idx)


def _get_subtract_at_indices_kernel(device_idx: int) -> Tuple[int, int]:
    """Get function handle for subtract_at_indices kernel."""
    vt, _C, vt_triton = _get_vbt_modules()
    
    key = f"subtract_at_idx_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            _, func_h, extra = _kernel_cache[key]
            return func_h, extra
    
    # Compile kernel
    import os
    import importlib.util
    kernels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.py")
    spec = importlib.util.spec_from_file_location("indexing_kernels3", kernels_path)
    kernels_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels_mod)
    subtract_at_indices_kernel = kernels_mod.subtract_at_indices_kernel
    
    # data_ptr, idx_ptr, value, batch_size, dim_size, batch_stride, elem_stride
    signature = "*fp32,*i64,fp32,i32,i32,i32,i32"
    meta = {}
    
    ptx, entry, shmem = vt_triton._compile_to_ptx(
        subtract_at_indices_kernel,
        signature=signature,
        meta=meta,
        num_warps=1,
    )
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    sig_tokens = [t.strip() for t in signature.split(",")]
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _kernel_cache[key] = (mod_h, func_h, extra_params)
    
    return func_h, extra_params


def subtract_at_indices(data, index, value: float):
    """Subtract a scalar value at positions [i, index[i]] for each row i.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    
    This performs: data[i, index[i]] -= value for all i
    Equivalent to: data = data - one_hot(index) * value
    
    Args:
        data: Data tensor [batch, dim], float32, CUDA (modified in-place)
        index: Index tensor [batch], int64, CUDA
        value: Scalar value to subtract (float)
        
    Returns:
        Modified data tensor
    """
    vt, _C, _ = _get_vbt_modules()
    
    data_sizes = tuple(int(s) for s in data.sizes)
    data_strides = tuple(int(s) for s in data.strides)
    data_dtype = str(data.dtype)
    data_device = data.device
    
    idx_sizes = tuple(int(s) for s in index.sizes)
    idx_dtype = str(index.dtype)
    
    if data_device[0] != 2:
        raise RuntimeError("subtract_at_indices requires CUDA tensors")
    if index.device[0] != 2:
        raise RuntimeError("index must be on CUDA")
    if data_dtype != "float32":
        raise TypeError("data must be float32")
    if idx_dtype != "int64":
        raise TypeError("index must be int64")
    if len(data_sizes) != 2:
        raise ValueError("data must be 2D [batch, dim]")
    if len(idx_sizes) != 1:
        raise ValueError("index must be 1D [batch]")
    
    device_idx = int(data_device[1])
    
    batch_size = data_sizes[0]
    dim_size = data_sizes[1]
    batch_stride = data_strides[0]
    elem_stride = data_strides[1]
    
    # Get kernel handle
    func_handle, extra_params = _get_subtract_at_indices_kernel(device_idx)
    
    # Launch kernel - one thread block per row
    grid = (batch_size, 1, 1)
    block = (32, 1, 1)  # num_warps=1 -> 32 threads
    shmem = 0
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    # Args: data_ptr, idx_ptr, value, batch_size, dim_size, batch_stride, elem_stride
    args = [data, index, value, batch_size, dim_size, batch_stride, elem_stride]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_handle, grid, block, shmem, stream, args)
    
    return data


__all__ = [
    "gather",
    "gather_dim1",
    "scatter_add",
    "scatter_add_",
    "scatter_add_dim1",
    "subtract_at_indices",
    "argmax",
]
