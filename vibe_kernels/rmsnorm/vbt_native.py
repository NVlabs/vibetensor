# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure VibeTensor implementation of RMSNorm using Triton kernels.

This module wraps RMSNorm kernels for use with VibeTensor tensors,
with NO PyTorch dependency at runtime.

Usage:
    import vibetensor.torch as vt
    from vibe_kernels.rmsnorm import vbt_native as rmsnorm_ops
    
    x = vt.cuda.to_device(np.random.randn(4, 256).astype(np.float32))
    gamma = vt.cuda.to_device(np.ones(256).astype(np.float32))
    out = rmsnorm_ops.rmsnorm(x, gamma)
"""

from __future__ import annotations

import threading
from typing import Tuple, Dict, Optional

import triton
import triton.language as tl


def _get_vbt_modules():
    """Lazy import VibeTensor modules."""
    import vibetensor.torch as vt
    from vibetensor import _C
    import vibetensor.triton as vt_triton
    return vt, _C, vt_triton


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


# -----------------------------------------------------------------------------
# RMSNorm Forward Kernel
# -----------------------------------------------------------------------------

@triton.jit
def _rmsnorm_fwd_kernel(
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
):
    """RMSNorm forward kernel.
    
    Each program handles one row of the input tensor.
    """
    row_id = tl.program_id(axis=0)
    in_row = input_ptr + row_id * input_stride
    out_row = output_ptr + row_id * output_stride

    # First pass: compute sum of squares
    sum_sq = 0.0
    for chunk in tl.static_range(0, CHUNKS):
        col_start = chunk * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(in_row + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute inverse RMS
    mean_sq = sum_sq / n_cols
    inv_rms = tl.math.rsqrt(mean_sq + eps)

    # Second pass: normalize and apply gamma
    for chunk in tl.static_range(0, CHUNKS):
        col_start = chunk * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(in_row + cols, mask=mask, other=0.0).to(tl.float32)
        norm = x * inv_rms
        if HAS_GAMMA:
            gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            norm = norm * gamma
        tl.store(out_row + cols, norm, mask=mask)

    # Store inv_rms for backward
    tl.store(inv_rms_ptr + row_id, inv_rms)


# -----------------------------------------------------------------------------
# RMSNorm Backward Kernel
# -----------------------------------------------------------------------------

@triton.jit
def _rmsnorm_bwd_kernel(
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
):
    """RMSNorm backward kernel.
    
    Computes gradients w.r.t. input and gamma.
    """
    row_id = tl.program_id(axis=0)
    in_row = input_ptr + row_id * input_stride
    grad_out_row = grad_output_ptr + row_id * grad_output_stride
    grad_in_row = grad_input_ptr + row_id * grad_input_stride

    inv_rms = tl.load(inv_rms_ptr + row_id).to(tl.float32)
    
    # First pass: compute dot product
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
            # Accumulate grad_gamma (element-wise)
            grad_gamma_contrib = dy * normed
            tl.atomic_add(grad_gamma_ptr + cols, grad_gamma_contrib, mask=mask)
        else:
            dy_scaled = dy
        dot += tl.sum(dy_scaled * normed, axis=0)

    dot_mean = dot / n_cols

    # Second pass: compute grad_input
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
        tl.store(grad_in_row + cols, dx, mask=mask)


# -----------------------------------------------------------------------------
# Kernel Cache
# -----------------------------------------------------------------------------

_kernel_cache: Dict[str, Tuple[int, int, int, int]] = {}
_cache_lock = threading.Lock()


def _get_rmsnorm_fwd_kernel(
    has_gamma: bool,
    block_size: int,
    chunks: int,
    num_warps: int,
    device_idx: int,
) -> Tuple[int, int, int, int]:
    """Get compiled forward kernel handle."""
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    key = f"rmsnorm_fwd_{has_gamma}_{block_size}_{chunks}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            return _kernel_cache[key]
    
    # Signature: output_ptr, output_stride, inv_rms_ptr, input_ptr, input_stride, gamma_ptr, n_cols, eps
    signature = "*fp32,i32,*fp32,*fp32,i32,*fp32,i32,fp32"
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    meta = {
        "HAS_GAMMA": 1 if has_gamma else 0,
        "BLOCK_SIZE": block_size,
        "CHUNKS": chunks,
    }
    
    arg_names = _rmsnorm_fwd_kernel.arg_names
    params = _rmsnorm_fwd_kernel.params
    sig_map = {}
    idx = 0
    for i, p in enumerate(params):
        name = arg_names[i]
        is_constexpr = bool(getattr(p, "is_constexpr", False))
        if is_constexpr:
            sig_map[name] = "constexpr"
        else:
            sig_map[name] = sig_tokens[idx]
            idx += 1
    
    target = driver.active.get_current_target()
    src = ASTSource(_rmsnorm_fwd_kernel, sig_map, meta, {})
    compiled = triton.compile(src, target=target, options={"num_warps": num_warps})
    
    asm = getattr(compiled, "asm", {})
    if "ptx" not in asm:
        raise RuntimeError("Triton compile produced no PTX")
    ptx_val = asm["ptx"]
    ptx = ptx_val if isinstance(ptx_val, bytes) else str(ptx_val).encode("utf-8")
    
    entry = getattr(compiled, "name", None)
    if not isinstance(entry, str):
        md = getattr(compiled, "metadata", None)
        if hasattr(md, "name"):
            entry = md.name
    if not isinstance(entry, str) or not entry:
        entry = "_rmsnorm_fwd_kernel"
    
    md = getattr(compiled, "metadata", None)
    shared_mem = 0
    if hasattr(md, "shared"):
        shared_mem = int(md.shared)
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _kernel_cache[key] = (mod_h, func_h, extra_params, shared_mem)
    
    return mod_h, func_h, extra_params, shared_mem


def _get_rmsnorm_bwd_kernel(
    has_gamma: bool,
    block_size: int,
    chunks: int,
    num_warps: int,
    device_idx: int,
) -> Tuple[int, int, int, int]:
    """Get compiled backward kernel handle."""
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    key = f"rmsnorm_bwd_{has_gamma}_{block_size}_{chunks}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            return _kernel_cache[key]
    
    # Signature: grad_input_ptr, grad_input_stride, grad_gamma_ptr, grad_output_ptr, grad_output_stride,
    #            input_ptr, input_stride, inv_rms_ptr, gamma_ptr, n_cols
    signature = "*fp32,i32,*fp32,*fp32,i32,*fp32,i32,*fp32,*fp32,i32"
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    meta = {
        "HAS_GAMMA": 1 if has_gamma else 0,
        "BLOCK_SIZE": block_size,
        "CHUNKS": chunks,
    }
    
    arg_names = _rmsnorm_bwd_kernel.arg_names
    params = _rmsnorm_bwd_kernel.params
    sig_map = {}
    idx = 0
    for i, p in enumerate(params):
        name = arg_names[i]
        is_constexpr = bool(getattr(p, "is_constexpr", False))
        if is_constexpr:
            sig_map[name] = "constexpr"
        else:
            sig_map[name] = sig_tokens[idx]
            idx += 1
    
    target = driver.active.get_current_target()
    src = ASTSource(_rmsnorm_bwd_kernel, sig_map, meta, {})
    compiled = triton.compile(src, target=target, options={"num_warps": num_warps})
    
    asm = getattr(compiled, "asm", {})
    if "ptx" not in asm:
        raise RuntimeError("Triton compile produced no PTX")
    ptx_val = asm["ptx"]
    ptx = ptx_val if isinstance(ptx_val, bytes) else str(ptx_val).encode("utf-8")
    
    entry = getattr(compiled, "name", None)
    if not isinstance(entry, str):
        md = getattr(compiled, "metadata", None)
        if hasattr(md, "name"):
            entry = md.name
    if not isinstance(entry, str) or not entry:
        entry = "_rmsnorm_bwd_kernel"
    
    md = getattr(compiled, "metadata", None)
    shared_mem = 0
    if hasattr(md, "shared"):
        shared_mem = int(md.shared)
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _kernel_cache[key] = (mod_h, func_h, extra_params, shared_mem)
    
    return mod_h, func_h, extra_params, shared_mem


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def rmsnorm(
    x,
    gamma=None,
    eps: float = 1e-6,
):
    """RMSNorm forward pass.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    
    Args:
        x: Input tensor [..., hidden_size], float32, CUDA
        gamma: Optional weight tensor [hidden_size], float32, CUDA
        eps: Epsilon for numerical stability
        
    Returns:
        Tuple of (output, inv_rms) where inv_rms is needed for backward
    """
    vt, _C, _ = _get_vbt_modules()
    
    x_sizes = tuple(int(s) for s in x.sizes)
    x_strides = tuple(int(s) for s in x.strides)
    x_dtype = str(x.dtype)
    x_device = x.device
    
    if x_device[0] != 2:
        raise RuntimeError("rmsnorm requires CUDA tensors")
    if x_dtype != "float32":
        raise TypeError("rmsnorm requires float32 (for now)")
    
    device_idx = int(x_device[1])
    hidden = x_sizes[-1]
    rows = 1
    for s in x_sizes[:-1]:
        rows *= s
    
    has_gamma = gamma is not None
    
    # Select block size
    block = min(256, _next_power_of_2(hidden))
    chunks = (hidden + block - 1) // block
    num_warps = _pick_num_warps(block)
    
    # Allocate outputs
    out = _C._cuda_empty(list(x_sizes), x_dtype, device_idx)
    inv_rms = _C._cuda_empty([rows], x_dtype, device_idx)
    
    # Get kernel
    _, func_h, extra_params, shmem = _get_rmsnorm_fwd_kernel(
        has_gamma, block, chunks, num_warps, device_idx
    )
    
    # Compute strides for 2D view
    input_stride = x_strides[-2] if len(x_sizes) >= 2 else hidden
    output_stride = hidden  # Output is contiguous
    
    grid = (rows, 1, 1)
    block_dim = (num_warps * 32, 1, 1)
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    # Build args
    gamma_tensor = gamma if has_gamma else x  # Dummy if no gamma
    args = [out, output_stride, inv_rms, x, input_stride, gamma_tensor, hidden, eps]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block_dim, shmem, stream, args)
    
    return out, inv_rms


def rmsnorm_backward(
    grad_output,
    x,
    inv_rms,
    gamma=None,
):
    """RMSNorm backward pass.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    
    Args:
        grad_output: Gradient from upstream [..., hidden_size], float32
        x: Original input tensor [..., hidden_size], float32
        inv_rms: Inverse RMS from forward [rows], float32
        gamma: Optional weight tensor [hidden_size], float32
        
    Returns:
        Tuple of (grad_input, grad_gamma) where grad_gamma is None if gamma was None
    """
    vt, _C, _ = _get_vbt_modules()
    
    x_sizes = tuple(int(s) for s in x.sizes)
    x_strides = tuple(int(s) for s in x.strides)
    grad_sizes = tuple(int(s) for s in grad_output.sizes)
    grad_strides = tuple(int(s) for s in grad_output.strides)
    x_dtype = str(x.dtype)
    x_device = x.device
    
    if x_device[0] != 2:
        raise RuntimeError("rmsnorm_backward requires CUDA tensors")
    if x_dtype != "float32":
        raise TypeError("rmsnorm_backward requires float32 (for now)")
    
    device_idx = int(x_device[1])
    hidden = x_sizes[-1]
    rows = 1
    for s in x_sizes[:-1]:
        rows *= s
    
    has_gamma = gamma is not None
    
    # Select block size (same as forward)
    block = min(256, _next_power_of_2(hidden))
    chunks = (hidden + block - 1) // block
    num_warps = _pick_num_warps(block)
    
    # Allocate outputs
    grad_input = _C._cuda_empty(list(x_sizes), x_dtype, device_idx)
    if has_gamma:
        # Zero-initialize grad_gamma for atomic adds
        grad_gamma = _C._cuda_zeros([hidden], x_dtype, device_idx)
    else:
        grad_gamma = None
    
    # Get kernel
    _, func_h, extra_params, shmem = _get_rmsnorm_bwd_kernel(
        has_gamma, block, chunks, num_warps, device_idx
    )
    
    # Compute strides
    input_stride = x_strides[-2] if len(x_sizes) >= 2 else hidden
    grad_input_stride = hidden
    grad_output_stride = grad_strides[-2] if len(grad_sizes) >= 2 else hidden
    
    grid = (rows, 1, 1)
    block_dim = (num_warps * 32, 1, 1)
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    # Build args
    gamma_tensor = gamma if has_gamma else x  # Dummy if no gamma
    grad_gamma_tensor = grad_gamma if has_gamma else grad_input  # Dummy if no gamma
    args = [
        grad_input, grad_input_stride, grad_gamma_tensor,
        grad_output, grad_output_stride,
        x, input_stride, inv_rms, gamma_tensor, hidden
    ]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block_dim, shmem, stream, args)
    
    return grad_input, grad_gamma


__all__ = [
    "rmsnorm",
    "rmsnorm_backward",
]
