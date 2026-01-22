# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure VibeTensor implementation of cross_entropy loss using Triton kernels.

This module wraps the kernel_factory Triton loss kernels for use
with VibeTensor tensors, with NO PyTorch dependency at runtime.

Usage:
    import vibetensor.torch as vt
    from vibe_kernels.loss import vbt_native as loss_ops
    
    logits = vt.cuda.to_device(np.random.randn(4, 10).astype(np.float32))
    targets = vt.cuda.to_device(np.array([0, 1, 2, 3], dtype=np.int64))
    loss = loss_ops.cross_entropy(logits, targets)
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


# -----------------------------------------------------------------------------
# Kernel Definition
# -----------------------------------------------------------------------------

@triton.jit
def cross_entropy_fwd_kernel(
    loss_ptr,       # Output: per-row loss values
    logits_ptr,     # Input: (rows, cols) logits
    targets_ptr,    # Input: (rows,) target indices
    rows,           # Number of rows
    cols,           # Vocabulary size
    stride_logits,  # Stride along rows
    IGNORE_INDEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Cross entropy forward kernel.
    
    Computes: loss[row] = logsumexp - logits[row, targets[row]]
    """
    row = tl.program_id(0)
    if row >= rows:
        return

    row_offset = row * stride_logits
    offsets = tl.arange(0, BLOCK_SIZE)

    target = tl.load(targets_ptr + row)
    is_active = target != IGNORE_INDEX

    # First pass: find max
    max_val = tl.full([1], -float("inf"), dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(logits_ptr + row_offset + idx, mask=mask, other=-float("inf"))
        vals = vals.to(tl.float32)
        block_max = tl.max(tl.where(mask, vals, -float("inf")), axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Second pass: compute sum of exp
    sum_exp = tl.zeros([1], dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(logits_ptr + row_offset + idx, mask=mask, other=-float("inf"))
        vals = vals.to(tl.float32)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0), axis=0)

    sum_exp = tl.maximum(sum_exp, 1e-20)
    logsumexp_val = tl.log(sum_exp) + max_val
    
    # Compute loss = logsumexp - logits[target]
    target_logit = tl.load(
        logits_ptr + row_offset + target, mask=is_active, other=0.0
    ).to(tl.float32)
    
    loss_val = tl.where(is_active, logsumexp_val - target_logit, 0.0)
    
    # Store using scalar indexing (single element)
    out_idx = row + tl.arange(0, 1)
    tl.store(loss_ptr + out_idx, loss_val.to(tl.float32), mask=out_idx < rows)


# -----------------------------------------------------------------------------
# Kernel Cache
# -----------------------------------------------------------------------------

_kernel_cache: Dict[str, Tuple[int, int, int, int]] = {}
_cache_lock = threading.Lock()


def _get_cross_entropy_kernel(block_size: int, ignore_index: int, device_idx: int) -> Tuple[int, int, int, int]:
    """Get compiled kernel handle for cross_entropy."""
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    key = f"cross_entropy_{block_size}_{ignore_index}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            return _kernel_cache[key]
    
    # Build signature
    signature = "*fp32,*fp32,*i64,i32,i32,i32"
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    # Constexpr values
    meta = {
        "IGNORE_INDEX": ignore_index,
        "BLOCK_SIZE": block_size,
    }
    
    # Build signature map
    arg_names = cross_entropy_fwd_kernel.arg_names
    params = cross_entropy_fwd_kernel.params
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
    src = ASTSource(cross_entropy_fwd_kernel, sig_map, meta, {})
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
        entry = "cross_entropy_fwd_kernel"
    
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

def cross_entropy(logits, targets, ignore_index: int = -100, reduction: str = "mean"):
    """Compute cross entropy loss.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    
    Args:
        logits: Input logits tensor of shape (batch, num_classes), float32, CUDA
        targets: Target indices tensor of shape (batch,), int64, CUDA
        ignore_index: Target value to ignore in loss computation
        reduction: "mean" or "sum" or "none"
        
    Returns:
        Scalar loss (for mean/sum) or per-sample losses (for none)
    """
    vt, _C, _ = _get_vbt_modules()
    
    # Get tensor properties
    logits_sizes = tuple(int(s) for s in logits.sizes)
    logits_strides = tuple(int(s) for s in logits.strides)
    logits_dtype = str(logits.dtype)
    logits_device = logits.device
    
    targets_sizes = tuple(int(s) for s in targets.sizes)
    targets_dtype = str(targets.dtype)
    
    # Validate
    if logits_device[0] != 2:
        raise RuntimeError("cross_entropy requires CUDA tensors")
    if targets.device[0] != 2:
        raise RuntimeError("targets must be on CUDA")
    if logits_dtype != "float32":
        raise TypeError("logits must be float32 (for now)")
    if targets_dtype != "int64":
        raise TypeError("targets must be int64")
    if len(logits_sizes) != 2:
        raise ValueError("logits must be 2D (batch, num_classes)")
    if len(targets_sizes) != 1:
        raise ValueError("targets must be 1D (batch,)")
    
    device_idx = int(logits_device[1])
    rows, cols = logits_sizes
    
    if rows != targets_sizes[0]:
        raise ValueError(f"Batch size mismatch: logits has {rows}, targets has {targets_sizes[0]}")
    
    # Handle edge cases
    if rows == 0:
        if reduction == "none":
            return _C._cuda_empty([0], "float32", device_idx)
        else:
            # Return scalar 0
            zero = _C._cuda_empty([1], "float32", device_idx)
            # Zero-fill
            import numpy as np
            zero_np = np.zeros(1, dtype=np.float32)
            return _C._cuda_h2d_alloc_copy(zero_np, 'float32', device_idx)
    
    # Allocate output (per-row losses)
    losses = _C._cuda_empty([rows], "float32", device_idx)
    
    # Get kernel
    block_size = min(1024, max(128, (cols + 127) // 128 * 128))
    _, func_h, extra_params, shmem = _get_cross_entropy_kernel(block_size, ignore_index, device_idx)
    
    # Stride
    stride_logits = logits_strides[0]
    
    # Launch
    grid = (rows, 1, 1)
    block = (128, 1, 1)
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    args = [losses, logits, targets, rows, cols, stride_logits]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block, shmem, stream, args)
    
    # Reduce
    if reduction == "none":
        return losses
    else:
        # Sum or mean - need to copy to CPU and compute
        # For now, do it on CPU (can optimize later with reduction kernel)
        import numpy as np
        losses_np = _C._cuda_d2h_copy_numpy_sync(losses)
        
        if reduction == "sum":
            result = np.sum(losses_np)
        else:  # mean
            targets_np = _C._cuda_d2h_copy_numpy_sync(targets)
            valid = int((targets_np != int(ignore_index)).sum())
            if valid == 0:
                result = np.float32(0.0)
            else:
                result = np.sum(losses_np) / valid
        
        # Return as CUDA tensor
        result_np = np.array([result], dtype=np.float32)
        return _C._cuda_h2d_alloc_copy(result_np, 'float32', device_idx)


# -----------------------------------------------------------------------------
# Cross Entropy with Cache (for backward)
# -----------------------------------------------------------------------------

@triton.jit
def cross_entropy_fwd_with_lse_kernel(
    loss_ptr,
    lse_ptr,
    logits_ptr,
    targets_ptr,
    rows,
    cols,
    stride_logits,
    IGNORE_INDEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Cross entropy forward kernel that saves LSE for backward."""
    row = tl.program_id(0)
    if row >= rows:
        return

    row_offset = row * stride_logits
    offsets = tl.arange(0, BLOCK_SIZE)

    target = tl.load(targets_ptr + row)
    is_active = target != IGNORE_INDEX

    max_val = tl.full([1], -float("inf"), dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(logits_ptr + row_offset + idx, mask=mask, other=-float("inf"))
        vals = vals.to(tl.float32)
        block_max = tl.max(tl.where(mask, vals, -float("inf")), axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = tl.zeros([1], dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(logits_ptr + row_offset + idx, mask=mask, other=-float("inf"))
        vals = vals.to(tl.float32)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0), axis=0)

    sum_exp = tl.maximum(sum_exp, 1e-20)
    logsumexp_val = tl.log(sum_exp) + max_val
    
    target_logit = tl.load(
        logits_ptr + row_offset + target, mask=is_active, other=0.0
    ).to(tl.float32)
    
    loss_val = tl.where(is_active, logsumexp_val - target_logit, 0.0)
    
    out_idx = row + tl.arange(0, 1)
    tl.store(loss_ptr + out_idx, loss_val.to(tl.float32), mask=out_idx < rows)
    tl.store(lse_ptr + out_idx, logsumexp_val.to(tl.float32), mask=out_idx < rows)


# -----------------------------------------------------------------------------
# Cross Entropy Backward Kernel
# -----------------------------------------------------------------------------

@triton.jit
def cross_entropy_bwd_kernel(
    grad_ptr,
    logits_ptr,
    targets_ptr,
    dloss_ptr,
    lse_ptr,
    rows,
    cols,
    stride_logits,
    IGNORE_INDEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Cross entropy backward kernel.
    
    Computes: grad[i] = (softmax[i] - one_hot[i]) * dloss
    """
    row = tl.program_id(0)
    if row >= rows:
        return

    row_offset = row * stride_logits
    offsets = tl.arange(0, BLOCK_SIZE)

    target = tl.load(targets_ptr + row)
    is_active = target != IGNORE_INDEX
    
    if not is_active:
        # Zero gradient for ignored indices
        for start in range(0, cols, BLOCK_SIZE):
            idx = start + offsets
            mask = idx < cols
            tl.store(grad_ptr + row_offset + idx, tl.zeros([BLOCK_SIZE], dtype=tl.float32), mask=mask)
        return

    lse = tl.load(lse_ptr + row)
    scale = tl.load(dloss_ptr + row)

    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        logits = tl.load(logits_ptr + row_offset + idx, mask=mask, other=-float("inf"))
        logits = logits.to(tl.float32)
        probs = tl.exp(logits - lse)
        grad = probs * scale

        grad = tl.where(idx == target, grad - scale, grad)

        tl.store(grad_ptr + row_offset + idx, grad, mask=mask)


# -----------------------------------------------------------------------------
# Backward Kernel Cache
# -----------------------------------------------------------------------------

_fwd_lse_cache: Dict[str, Tuple[int, int, int, int]] = {}
_bwd_cache: Dict[str, Tuple[int, int, int, int]] = {}


def _get_ce_fwd_lse_kernel(block_size: int, ignore_index: int, device_idx: int) -> Tuple[int, int, int, int]:
    """Get compiled forward kernel with LSE."""
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    key = f"ce_fwd_lse_{block_size}_{ignore_index}_{device_idx}"
    
    with _cache_lock:
        if key in _fwd_lse_cache:
            return _fwd_lse_cache[key]
    
    signature = "*fp32,*fp32,*fp32,*i64,i32,i32,i32"
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    meta = {
        "IGNORE_INDEX": ignore_index,
        "BLOCK_SIZE": block_size,
    }
    
    arg_names = cross_entropy_fwd_with_lse_kernel.arg_names
    params = cross_entropy_fwd_with_lse_kernel.params
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
    src = ASTSource(cross_entropy_fwd_with_lse_kernel, sig_map, meta, {})
    compiled = triton.compile(src, target=target, options={"num_warps": 4})
    
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
        entry = "cross_entropy_fwd_with_lse_kernel"
    
    md = getattr(compiled, "metadata", None)
    shared_mem = 0
    if hasattr(md, "shared"):
        shared_mem = int(md.shared)
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _fwd_lse_cache[key] = (mod_h, func_h, extra_params, shared_mem)
    
    return mod_h, func_h, extra_params, shared_mem


def _get_ce_bwd_kernel(block_size: int, ignore_index: int, device_idx: int) -> Tuple[int, int, int, int]:
    """Get compiled backward kernel."""
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    key = f"ce_bwd_{block_size}_{ignore_index}_{device_idx}"
    
    with _cache_lock:
        if key in _bwd_cache:
            return _bwd_cache[key]
    
    signature = "*fp32,*fp32,*i64,*fp32,*fp32,i32,i32,i32"
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    meta = {
        "IGNORE_INDEX": ignore_index,
        "BLOCK_SIZE": block_size,
    }
    
    arg_names = cross_entropy_bwd_kernel.arg_names
    params = cross_entropy_bwd_kernel.params
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
    src = ASTSource(cross_entropy_bwd_kernel, sig_map, meta, {})
    compiled = triton.compile(src, target=target, options={"num_warps": 4})
    
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
        entry = "cross_entropy_bwd_kernel"
    
    md = getattr(compiled, "metadata", None)
    shared_mem = 0
    if hasattr(md, "shared"):
        shared_mem = int(md.shared)
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _bwd_cache[key] = (mod_h, func_h, extra_params, shared_mem)
    
    return mod_h, func_h, extra_params, shared_mem


# -----------------------------------------------------------------------------
# Public API: Forward with Cache + Backward
# -----------------------------------------------------------------------------

def cross_entropy_with_cache(logits, targets, ignore_index: int = -100):
    """Compute cross entropy loss with saved state for backward.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    
    Args:
        logits: [batch, num_classes] float32 CUDA
        targets: [batch] int64 CUDA
        ignore_index: Target value to ignore
        
    Returns:
        (loss, cache): scalar loss and cache dict for backward
    """
    vt, _C, _ = _get_vbt_modules()
    
    logits_sizes = tuple(int(s) for s in logits.sizes)
    logits_strides = tuple(int(s) for s in logits.strides)
    logits_dtype = str(logits.dtype)
    logits_device = logits.device
    
    targets_sizes = tuple(int(s) for s in targets.sizes)
    targets_dtype = str(targets.dtype)
    
    if logits_device[0] != 2:
        raise RuntimeError("cross_entropy requires CUDA tensors")
    if logits_dtype != "float32":
        raise TypeError("logits must be float32")
    if targets_dtype != "int64":
        raise TypeError("targets must be int64")
    
    device_idx = int(logits_device[1])
    rows, cols = logits_sizes
    
    # Allocate outputs
    losses = _C._cuda_empty([rows], "float32", device_idx)
    lse = _C._cuda_empty([rows], "float32", device_idx)
    
    # Get kernel
    block_size = min(1024, max(128, (cols + 127) // 128 * 128))
    _, func_h, extra_params, shmem = _get_ce_fwd_lse_kernel(block_size, ignore_index, device_idx)
    
    stride_logits = logits_strides[0]
    
    grid = (rows, 1, 1)
    block = (128, 1, 1)
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    args = [losses, lse, logits, targets, rows, cols, stride_logits]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block, shmem, stream, args)
    
    # Compute mean loss on CPU (valid_count must be based on targets, not loss values).
    import numpy as np
    losses_np = _C._cuda_d2h_copy_numpy_sync(losses)
    targets_np = _C._cuda_d2h_copy_numpy_sync(targets)
    valid_count = int((targets_np != int(ignore_index)).sum())
    if valid_count == 0:
        loss_val = 0.0
    else:
        loss_val = float(np.sum(losses_np) / valid_count)
    
    cache = {
        'logits': logits,
        'targets': targets,
        'lse': lse,
        'rows': rows,
        'cols': cols,
        'stride_logits': stride_logits,
        'ignore_index': ignore_index,
        'valid_count': int(valid_count),
    }
    
    return loss_val, cache


def cross_entropy_backward(cache, grad_out=1.0):
    """Compute cross entropy backward using Triton kernel.
    
    Args:
        cache: Cache dict from cross_entropy_with_cache
        grad_out: Upstream gradient (scalar, default 1.0)
        
    Returns:
        grad_logits: [batch, num_classes] gradient
    """
    vt, _C, _ = _get_vbt_modules()
    import numpy as np
    
    logits = cache['logits']
    targets = cache['targets']
    lse = cache['lse']
    rows = cache['rows']
    cols = cache['cols']
    stride_logits = cache['stride_logits']
    ignore_index = cache['ignore_index']
    valid_count = cache['valid_count']
    
    device_idx = int(logits.device[1])
    
    # Allocate gradient output (use zeros to ensure clean state)
    grad_logits = _C._cuda_zeros([rows, cols], "float32", device_idx)
    
    # Compute per-row gradient scale: grad_out / valid_count
    if valid_count == 0:
        scale = 0.0
    else:
        scale = float(grad_out) / float(valid_count)
    
    # Create dloss tensor (scale for each row)
    dloss_np = np.full(rows, scale, dtype=np.float32)
    dloss = _C._cuda_h2d_alloc_copy(dloss_np, 'float32', device_idx)
    
    # Get backward kernel
    block_size = min(1024, max(128, (cols + 127) // 128 * 128))
    _, func_h, extra_params, shmem = _get_ce_bwd_kernel(block_size, ignore_index, device_idx)
    
    grid = (rows, 1, 1)
    block = (128, 1, 1)
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    args = [grad_logits, logits, targets, dloss, lse, rows, cols, stride_logits]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block, shmem, stream, args)
    
    return grad_logits


__all__ = [
    "cross_entropy",
    "cross_entropy_with_cache",
    "cross_entropy_backward",
]
