# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure VibeTensor implementation of Embedding lookup using Triton kernels.

This module provides token embedding lookup for VibeTensor tensors,
with NO PyTorch dependency at runtime.

Usage:
    import vibetensor.torch as vt
    from vibe_kernels.embedding import vbt_native as embed_ops
    
    # weight: [vocab_size, embed_dim]
    # token_ids: [batch, seqlen]
    output = embed_ops.embedding_forward(weight, token_ids)
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

import triton
import triton.language as tl
import numpy as np


def _get_vbt_modules():
    """Lazy import VibeTensor modules."""
    import vibetensor.torch as vt
    from vibetensor import _C
    import vibetensor.triton as vt_triton
    return vt, _C, vt_triton


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _pick_num_warps(block_size: int) -> int:
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    if block_size >= 1024:
        return 4
    if block_size >= 256:
        return 2
    return 1


# -----------------------------------------------------------------------------
# Embedding Forward Kernel
# -----------------------------------------------------------------------------

@triton.jit
def _embedding_fwd_kernel(
    output_ptr,
    output_stride,
    embedding_ptr,
    embedding_stride,
    token_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple embedding lookup kernel.
    
    Each row in output corresponds to one token.
    """
    row_id = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    token = tl.load(token_ptr + row_id)
    emb_row = embedding_ptr + token * embedding_stride
    x = tl.load(emb_row + col_offsets, mask=mask, other=0.0)

    out_row = output_ptr + row_id * output_stride
    tl.store(out_row + col_offsets, x, mask=mask)


@triton.jit
def _embedding_bwd_kernel(
    grad_weight_ptr,
    grad_weight_stride,
    grad_output_ptr,
    grad_output_stride,
    token_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Embedding backward kernel - accumulates gradients into weight table.
    
    Uses atomic_add since multiple tokens may map to same embedding.
    """
    row_id = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    token = tl.load(token_ptr + row_id)
    
    grad_row = grad_output_ptr + row_id * grad_output_stride
    dy = tl.load(grad_row + col_offsets, mask=mask, other=0.0)

    grad_weight_row = grad_weight_ptr + token * grad_weight_stride
    tl.atomic_add(grad_weight_row + col_offsets, dy, mask=mask)


# -----------------------------------------------------------------------------
# Kernel Cache
# -----------------------------------------------------------------------------

_kernel_cache: Dict[str, tuple] = {}
_cache_lock = threading.Lock()


def _get_embedding_fwd_kernel(
    block_size: int,
    num_warps: int,
    device_idx: int,
) -> Tuple[int, int, int, int]:
    """Get compiled forward kernel handle."""
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    key = f"embedding_fwd_{block_size}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            return _kernel_cache[key]
    
    signature = "*fp32,i32,*fp32,i32,*i64,i32"  # Note: *i64 for int64 tokens
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    meta = {"BLOCK_SIZE": block_size}
    
    arg_names = _embedding_fwd_kernel.arg_names
    params = _embedding_fwd_kernel.params
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
    src = ASTSource(_embedding_fwd_kernel, sig_map, meta, {})
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
        entry = "_embedding_fwd_kernel"
    
    md = getattr(compiled, "metadata", None)
    shared_mem = 0
    if hasattr(md, "shared"):
        shared_mem = int(md.shared)
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    result = (mod_h, func_h, extra_params, shared_mem)
    
    with _cache_lock:
        _kernel_cache[key] = result
    
    return result


def _get_embedding_bwd_kernel(
    block_size: int,
    num_warps: int,
    device_idx: int,
) -> Tuple[int, int, int, int]:
    """Get compiled backward kernel handle."""
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    key = f"embedding_bwd_{block_size}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            return _kernel_cache[key]
    
    signature = "*fp32,i32,*fp32,i32,*i64,i32"  # Note: *i64 for int64 tokens
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    meta = {"BLOCK_SIZE": block_size}
    
    arg_names = _embedding_bwd_kernel.arg_names
    params = _embedding_bwd_kernel.params
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
    src = ASTSource(_embedding_bwd_kernel, sig_map, meta, {})
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
        entry = "_embedding_bwd_kernel"
    
    md = getattr(compiled, "metadata", None)
    shared_mem = 0
    if hasattr(md, "shared"):
        shared_mem = int(md.shared)
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    result = (mod_h, func_h, extra_params, shared_mem)
    
    with _cache_lock:
        _kernel_cache[key] = result
    
    return result


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def embedding_forward(
    weight,
    token_ids,
):
    """Look up embeddings for token IDs.
    
    Pure VibeTensor implementation using Triton kernel - NO PyTorch dependency.
    
    Args:
        weight: Embedding table [vocab_size, embed_dim], float32, CUDA
        token_ids: Token indices [batch, seqlen] or [n_tokens], int32, CUDA
        
    Returns:
        Embedded tokens [*token_ids.shape, embed_dim]
    """
    vt, _C, vt_triton = _get_vbt_modules()
    
    # Get tensor properties
    w_sizes = tuple(int(s) for s in weight.sizes)
    w_strides = tuple(int(s) for s in weight.strides)
    w_dtype = str(weight.dtype)
    w_device = weight.device
    
    if w_device[0] != 2:
        raise RuntimeError("embedding_forward requires CUDA tensors")
    if w_dtype != "float32":
        raise TypeError("embedding_forward requires float32 weight (for now)")
    
    device_idx = int(w_device[1])
    
    vocab_size, embed_dim = w_sizes
    embedding_stride = w_strides[0]
    
    # Flatten token_ids
    token_sizes = tuple(int(s) for s in token_ids.sizes)
    tokens_flat = token_ids.contiguous().reshape([-1])
    n_tokens = 1
    for s in token_sizes:
        n_tokens *= s
    
    # Allocate output
    out = _C._cuda_empty([n_tokens, embed_dim], w_dtype, device_idx)
    output_stride = embed_dim
    
    # Kernel config
    block_size = _next_power_of_2(embed_dim)
    num_warps = _pick_num_warps(block_size)
    
    # Compile kernel
    mod_h, func_h, extra_params, shared_mem = _get_embedding_fwd_kernel(
        block_size, num_warps, device_idx
    )
    
    # Launch
    grid = (n_tokens, 1, 1)
    block_dim = (num_warps * 32, 1, 1)
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    args = [out, output_stride, weight, embedding_stride, tokens_flat, embed_dim]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block_dim, shared_mem, stream, args)
    
    # Reshape output to match token_ids shape + embed_dim
    out_shape = list(token_sizes) + [embed_dim]
    return out.reshape(out_shape)


def embedding_backward(
    grad_output,
    token_ids,
    vocab_size: int,
):
    """Compute gradient for embedding weight.
    
    Pure VibeTensor implementation using Triton kernel - NO PyTorch dependency.
    
    Args:
        grad_output: Gradient from upstream [*token_ids.shape, embed_dim], float32, CUDA
        token_ids: Token indices [batch, seqlen] or [n_tokens], int32, CUDA
        vocab_size: Size of vocabulary (to create grad_weight)
        
    Returns:
        grad_weight: Gradient for embedding table [vocab_size, embed_dim]
    """
    vt, _C, vt_triton = _get_vbt_modules()
    
    # Get tensor properties
    g_sizes = tuple(int(s) for s in grad_output.sizes)
    g_dtype = str(grad_output.dtype)
    g_device = grad_output.device
    
    if g_device[0] != 2:
        raise RuntimeError("embedding_backward requires CUDA tensors")
    
    device_idx = int(g_device[1])
    
    embed_dim = g_sizes[-1]
    
    # Flatten grad_output and token_ids
    token_sizes = tuple(int(s) for s in token_ids.sizes)
    tokens_flat = token_ids.contiguous().reshape([-1])
    n_tokens = 1
    for s in token_sizes:
        n_tokens *= s
    
    grad_flat = grad_output.contiguous().reshape([n_tokens, embed_dim])
    grad_output_stride = embed_dim
    
    # Allocate grad_weight (initialized to zero) - use _cuda_zeros for proper zeroing
    grad_weight = _C._cuda_zeros([vocab_size, embed_dim], "float32", device_idx)
    grad_weight_stride = embed_dim
    
    # Kernel config
    block_size = _next_power_of_2(embed_dim)
    num_warps = _pick_num_warps(block_size)
    
    # Compile kernel
    mod_h, func_h, extra_params, shared_mem = _get_embedding_bwd_kernel(
        block_size, num_warps, device_idx
    )
    
    # Launch
    grid = (n_tokens, 1, 1)
    block_dim = (num_warps * 32, 1, 1)
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    args = [grad_weight, grad_weight_stride, grad_flat, grad_output_stride, tokens_flat, embed_dim]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block_dim, shared_mem, stream, args)
    
    return grad_weight


__all__ = [
    "embedding_forward",
    "embedding_backward",
]
