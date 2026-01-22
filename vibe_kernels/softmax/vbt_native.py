# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure VibeTensor implementation of softmax/log_softmax using Triton kernels.

This module wraps the kernel_factory Triton softmax kernels for use
with VibeTensor tensors, with NO PyTorch dependency at runtime.

Usage:
    import vibetensor.torch as vt
    from vibe_kernels.softmax import vbt_native as softmax_ops
    
    x = vt.cuda.to_device(np.random.randn(4, 8).astype(np.float32))
    y = softmax_ops.softmax(x, dim=-1)
    log_y = softmax_ops.log_softmax(x, dim=-1)
"""

from __future__ import annotations

import threading
from typing import Tuple, Dict

import triton
import triton.language as tl


def _get_vbt_modules():
    """Lazy import VibeTensor modules."""
    import vibetensor.torch as vt
    from vibetensor import _C
    import vibetensor.triton as vt_triton
    return vt, _C, vt_triton


# -----------------------------------------------------------------------------
# Kernel Definition
# -----------------------------------------------------------------------------

@triton.jit
def softmax_kernel(
    out_ptr,
    logits_ptr,
    rows,
    cols,
    stride_logits,
    stride_out,
    OUTPUT_LOG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Softmax kernel that operates on float32 input.
    
    Each program handles one row of the input tensor.
    """
    row = tl.program_id(0)
    if row >= rows:
        return

    row_offset_logits = row * stride_logits
    row_offset_out = row * stride_out
    offsets = tl.arange(0, BLOCK_SIZE)

    # First pass: find max
    max_val = tl.full([1], -float("inf"), dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(logits_ptr + row_offset_logits + idx, mask=mask, other=-float("inf"))
        vals = vals.to(tl.float32)
        block_max = tl.max(tl.where(mask, vals, -float("inf")), axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Second pass: compute sum of exp
    sum_exp = tl.zeros([1], dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(logits_ptr + row_offset_logits + idx, mask=mask, other=-float("inf"))
        vals = vals.to(tl.float32)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0), axis=0)

    sum_exp = tl.maximum(sum_exp, 1e-20)
    logsumexp = tl.log(sum_exp) + max_val

    # Third pass: compute output
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(logits_ptr + row_offset_logits + idx, mask=mask, other=-float("inf"))
        vals = vals.to(tl.float32)
        if OUTPUT_LOG:
            out = vals - logsumexp
        else:
            out = tl.exp(vals - logsumexp)
        out = out.to(tl.float32)
        tl.store(out_ptr + row_offset_out + idx, out, mask=mask)


# -----------------------------------------------------------------------------
# Kernel Cache
# -----------------------------------------------------------------------------

# key -> (mod_handle, func_handle, extra_params, shared_mem)
_kernel_cache: Dict[str, Tuple[int, int, int, int]] = {}
_cache_lock = threading.Lock()


def _get_softmax_kernel(output_log: bool, block_size: int, device_idx: int) -> Tuple[int, int, int, int]:
    """Get compiled kernel handle for softmax.
    
    Returns:
        (mod_handle, func_handle, extra_params, shared_mem_bytes)
    """
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    key = f"softmax_{output_log}_{block_size}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            return _kernel_cache[key]
    
    # Build signature
    signature = "*fp32,*fp32,i32,i32,i32,i32"
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    # Constexpr values
    meta = {
        "OUTPUT_LOG": 1 if output_log else 0,
        "BLOCK_SIZE": block_size,
    }
    
    # Build signature map
    arg_names = softmax_kernel.arg_names
    params = softmax_kernel.params
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
    
    # Compile
    target = driver.active.get_current_target()
    src = ASTSource(softmax_kernel, sig_map, meta, {})
    compiled = triton.compile(src, target=target, options={"num_warps": 4})
    
    # Extract PTX
    asm = getattr(compiled, "asm", {})
    if "ptx" not in asm:
        raise RuntimeError("Triton compile produced no PTX")
    ptx_val = asm["ptx"]
    ptx = ptx_val if isinstance(ptx_val, bytes) else str(ptx_val).encode("utf-8")
    
    # Get entry name
    entry = getattr(compiled, "name", None)
    if not isinstance(entry, str):
        md = getattr(compiled, "metadata", None)
        if hasattr(md, "name"):
            entry = md.name
    if not isinstance(entry, str) or not entry:
        entry = "softmax_kernel"
    
    # Get shared memory requirement
    md = getattr(compiled, "metadata", None)
    shared_mem = 0
    if hasattr(md, "shared"):
        shared_mem = int(md.shared)
    
    # Count extra params
    total_params = vt_triton._count_entry_params(ptx, entry)
    extra_params = max(0, total_params - len(sig_tokens))
    
    # Load module
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _kernel_cache[key] = (mod_h, func_h, extra_params, shared_mem)
    
    return mod_h, func_h, extra_params, shared_mem


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def softmax(x, dim: int = -1):
    """Compute softmax along the specified dimension.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    
    Args:
        x: Input tensor, float32, CUDA
        dim: Dimension to apply softmax (must be -1 or last dim)
        
    Returns:
        Output tensor with softmax applied
    """
    vt, _C, _ = _get_vbt_modules()
    
    # Get tensor properties
    x_sizes = tuple(int(s) for s in x.sizes)
    x_strides = tuple(int(s) for s in x.strides)
    x_dtype = str(x.dtype)
    x_device = x.device
    
    # Validate
    if x_device[0] != 2:  # kDLCUDA = 2
        raise RuntimeError("softmax requires CUDA tensors")
    if x_dtype != "float32":
        raise TypeError("softmax requires float32 (for now)")
    
    ndim = len(x_sizes)
    if dim < 0:
        dim = ndim + dim
    if dim != ndim - 1:
        raise NotImplementedError("softmax currently only supports dim=-1 (last dimension)")
    
    device_idx = int(x_device[1])
    
    # Compute dimensions
    cols = x_sizes[-1] if ndim > 0 else 1
    rows = 1
    for i in range(ndim - 1):
        rows *= x_sizes[i]
    
    # Handle edge cases
    if rows == 0 or cols == 0:
        return _C._cuda_empty(list(x_sizes), x_dtype, device_idx)
    
    # Allocate output
    out = _C._cuda_empty(list(x_sizes), x_dtype, device_idx)
    
    # Get kernel
    block_size = min(1024, max(128, (cols + 127) // 128 * 128))  # Round up to multiple of 128
    block_size = min(block_size, 1024)
    _, func_h, extra_params, shmem = _get_softmax_kernel(False, block_size, device_idx)
    
    # Strides for 2D view
    stride_logits = x_strides[-2] if ndim >= 2 else cols
    stride_out = cols  # Output is contiguous
    
    # Launch
    grid = (rows, 1, 1)
    block = (128, 1, 1)  # num_warps=4
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    args = [out, x, rows, cols, stride_logits, stride_out]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block, shmem, stream, args)
    
    return out


def log_softmax(x, dim: int = -1):
    """Compute log softmax along the specified dimension.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    
    Args:
        x: Input tensor, float32, CUDA
        dim: Dimension to apply softmax (must be -1 or last dim)
        
    Returns:
        Output tensor with log_softmax applied
    """
    vt, _C, _ = _get_vbt_modules()
    
    # Get tensor properties
    x_sizes = tuple(int(s) for s in x.sizes)
    x_strides = tuple(int(s) for s in x.strides)
    x_dtype = str(x.dtype)
    x_device = x.device
    
    # Validate
    if x_device[0] != 2:  # kDLCUDA = 2
        raise RuntimeError("log_softmax requires CUDA tensors")
    if x_dtype != "float32":
        raise TypeError("log_softmax requires float32 (for now)")
    
    ndim = len(x_sizes)
    if dim < 0:
        dim = ndim + dim
    if dim != ndim - 1:
        raise NotImplementedError("log_softmax currently only supports dim=-1 (last dimension)")
    
    device_idx = int(x_device[1])
    
    # Compute dimensions
    cols = x_sizes[-1] if ndim > 0 else 1
    rows = 1
    for i in range(ndim - 1):
        rows *= x_sizes[i]
    
    # Handle edge cases
    if rows == 0 or cols == 0:
        return _C._cuda_empty(list(x_sizes), x_dtype, device_idx)
    
    # Allocate output
    out = _C._cuda_empty(list(x_sizes), x_dtype, device_idx)
    
    # Get kernel
    block_size = min(1024, max(128, (cols + 127) // 128 * 128))
    block_size = min(block_size, 1024)
    _, func_h, extra_params, shmem = _get_softmax_kernel(True, block_size, device_idx)
    
    # Strides for 2D view
    stride_logits = x_strides[-2] if ndim >= 2 else cols
    stride_out = cols  # Output is contiguous
    
    # Launch
    grid = (rows, 1, 1)
    block = (128, 1, 1)  # num_warps=4
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    args = [out, x, rows, cols, stride_logits, stride_out]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block, shmem, stream, args)
    
    return out


__all__ = [
    "softmax",
    "log_softmax",
]
