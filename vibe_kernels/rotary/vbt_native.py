# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure VibeTensor implementation of Rotary Position Embeddings using Triton kernels.

This module wraps RoPE kernels for use with VibeTensor tensors,
with NO PyTorch dependency at runtime.

Usage:
    import vibetensor.torch as vt
    from vibe_kernels.rotary import vbt_native as rope_ops
    
    # q, k: [batch, heads, seqlen, head_dim]
    # cos, sin: [max_seq, head_dim // 2]
    q_rot, k_rot = rope_ops.apply_rotary_embedding(q, k, cos, sin)
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


def _pick_num_warps(block: int) -> int:
    if block >= 8192:
        return 16
    if block >= 2048:
        return 8
    if block >= 1024:
        return 4
    if block >= 512:
        return 4
    if block >= 256:
        return 2
    return 1


# -----------------------------------------------------------------------------
# Position Generation Kernel (GPU-native, no NumPy)
# -----------------------------------------------------------------------------

@triton.jit
def _generate_positions_kernel(
    out_ptr,
    seqlen,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Generate position indices: out[i] = i % seqlen"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Position = index % seqlen
    positions = offsets % seqlen
    tl.store(out_ptr + offsets, positions.to(tl.int32), mask=mask)


# -----------------------------------------------------------------------------
# Rotary Kernel (copied from triton_impl.py)
# -----------------------------------------------------------------------------

# KERNEL_PLACEHOLDER - will be copied via yank

@triton.jit
def _apply_rotary(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    rows,
    q_row_stride,
    k_row_stride,
    cos_stride,
    head_dim_half,
    Q_IS_FP16: tl.constexpr,
    Q_IS_BF16: tl.constexpr,
    K_IS_FP16: tl.constexpr,
    K_IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_offsets = pid * ROWS_PER_CTA + tl.arange(0, ROWS_PER_CTA)
    row_mask = row_offsets < rows

    col = tl.arange(0, BLOCK_SIZE)
    col_mask = col < head_dim_half
    mask_2d = row_mask[:, None] & col_mask[None, :]

    pos = tl.load(pos_ptr + row_offsets, mask=row_mask, other=0).to(tl.int32)

    cos_base = cos_ptr + pos[:, None] * cos_stride
    sin_base = sin_ptr + pos[:, None] * cos_stride
    cos = tl.load(cos_base + col[None, :], mask=mask_2d, other=0.0).to(tl.float32)
    sin = tl.load(sin_base + col[None, :], mask=mask_2d, other=0.0).to(tl.float32)

    q_base = q_ptr + row_offsets[:, None] * q_row_stride
    k_base = k_ptr + row_offsets[:, None] * k_row_stride

    q1 = tl.load(q_base + col[None, :], mask=mask_2d, other=0.0).to(tl.float32)
    q2 = tl.load(q_base + head_dim_half + col[None, :], mask=mask_2d, other=0.0).to(
        tl.float32
    )
    k1 = tl.load(k_base + col[None, :], mask=mask_2d, other=0.0).to(tl.float32)
    k2 = tl.load(k_base + head_dim_half + col[None, :], mask=mask_2d, other=0.0).to(
        tl.float32
    )

    q_out1 = tl.math.fma(q2, sin, q1 * cos)
    q_out2 = tl.math.fma(q2, cos, (-q1) * sin)
    k_out1 = tl.math.fma(k2, sin, k1 * cos)
    k_out2 = tl.math.fma(k2, cos, (-k1) * sin)

    if Q_IS_FP16:
        q_out1 = q_out1.to(tl.float16)
        q_out2 = q_out2.to(tl.float16)
    elif Q_IS_BF16:
        q_out1 = q_out1.to(tl.bfloat16)
        q_out2 = q_out2.to(tl.bfloat16)

    if K_IS_FP16:
        k_out1 = k_out1.to(tl.float16)
        k_out2 = k_out2.to(tl.float16)
    elif K_IS_BF16:
        k_out1 = k_out1.to(tl.bfloat16)
        k_out2 = k_out2.to(tl.bfloat16)

    tl.store(q_base + col[None, :], q_out1, mask=mask_2d)
    tl.store(q_base + head_dim_half + col[None, :], q_out2, mask=mask_2d)
    tl.store(k_base + col[None, :], k_out1, mask=mask_2d)
    tl.store(k_base + head_dim_half + col[None, :], k_out2, mask=mask_2d)


# -----------------------------------------------------------------------------
# Kernel Cache
# -----------------------------------------------------------------------------

_kernel_cache: Dict[str, tuple] = {}
_cache_lock = threading.Lock()


def _get_rotary_kernel(
    q_is_fp16: bool,
    q_is_bf16: bool,
    k_is_fp16: bool,
    k_is_bf16: bool,
    block_size: int,
    rows_per_cta: int,
    device_idx: int = 0,
) -> tuple:
    """Compile and cache the rotary kernel."""
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    cache_key = f"rotary_{q_is_fp16}_{q_is_bf16}_{k_is_fp16}_{k_is_bf16}_{block_size}_{rows_per_cta}_{device_idx}"
    
    with _cache_lock:
        if cache_key in _kernel_cache:
            return _kernel_cache[cache_key]
    
    # Signature for non-constexpr params: q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr, rows, q_row_stride, k_row_stride, cos_stride, head_dim_half
    signature = "*fp32,*fp32,*fp32,*fp32,*i32,i32,i32,i32,i32,i32"
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    meta = {
        "Q_IS_FP16": 1 if q_is_fp16 else 0,
        "Q_IS_BF16": 1 if q_is_bf16 else 0,
        "K_IS_FP16": 1 if k_is_fp16 else 0,
        "K_IS_BF16": 1 if k_is_bf16 else 0,
        "BLOCK_SIZE": block_size,
        "ROWS_PER_CTA": rows_per_cta,
    }
    
    arg_names = _apply_rotary.arg_names
    params = _apply_rotary.params
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
    
    num_warps = _pick_num_warps(block_size * rows_per_cta)
    
    target = driver.active.get_current_target()
    src = ASTSource(_apply_rotary, sig_map, meta, {})
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
        entry = "_apply_rotary"
    
    md = getattr(compiled, "metadata", None)
    shared_mem = 0
    if hasattr(md, "shared"):
        shared_mem = int(md.shared)
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    result = (mod_h, func_h, extra_params, shared_mem, num_warps)
    
    with _cache_lock:
        _kernel_cache[cache_key] = result
    
    return result


def _create_rope_positions_gpu(n_elements: int, seqlen: int, device_idx: int = 0):
    """Create RoPE position indices on GPU without NumPy.
    
    Generates positions[i] = i % seqlen for i in [0, n_elements).
    """
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    # Allocate output
    out = _C._cuda_empty([n_elements], "int32", device_idx)
    
    # Compile kernel
    block_size = 256
    signature = "*i32,i32,i32"
    sig_tokens = [tok.strip() for tok in signature.split(",")]
    
    meta = {"BLOCK_SIZE": block_size}
    
    arg_names = _generate_positions_kernel.arg_names
    params = _generate_positions_kernel.params
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
    src = ASTSource(_generate_positions_kernel, sig_map, meta, {})
    compiled = triton.compile(src, target=target, options={"num_warps": 4})
    
    asm = getattr(compiled, "asm", {})
    ptx_val = asm.get("ptx", b"")
    ptx = ptx_val if isinstance(ptx_val, bytes) else str(ptx_val).encode("utf-8")
    
    entry = getattr(compiled, "name", "_generate_positions_kernel")
    if not isinstance(entry, str):
        md = getattr(compiled, "metadata", None)
        if hasattr(md, "name"):
            entry = md.name
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    # Launch kernel
    grid = ((n_elements + block_size - 1) // block_size, 1, 1)
    block_dim = (128, 1, 1)
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    args = [out, seqlen, n_elements]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block_dim, 0, stream, args)
    
    return out


# -----------------------------------------------------------------------------
# Public API  
# -----------------------------------------------------------------------------

def apply_rotary_embedding(
    q,
    k, 
    cos,
    sin,
    positions=None,
):
    """Apply rotary position embeddings to query/key projections.
    
    Pure VibeTensor implementation using Triton kernel - NO PyTorch dependency.
    
    Args:
        q, k: Query/key tensors of shape (batch, heads, seqlen, head_dim), float32, CUDA
        cos, sin: Precomputed cosine/sine tables (max_position, head_dim // 2), float32
        positions: Optional position indices (batch, seqlen), int32
        
    Returns:
        Tuple (q_rot, k_rot) with rotary embeddings applied (in-place on cloned tensors)
    """
    vt, _C, vt_triton = _get_vbt_modules()
    
    # Get tensor properties
    q_sizes = tuple(int(s) for s in q.sizes)
    q_strides = tuple(int(s) for s in q.strides)
    q_dtype = str(q.dtype)
    q_device = q.device
    
    if q_device[0] != 2:
        raise RuntimeError("apply_rotary_embedding requires CUDA tensors")
    if q_dtype != "float32":
        raise TypeError("apply_rotary_embedding requires float32 (for now)")
    
    device_idx = int(q_device[1])
    
    # Parse dimensions
    if len(q_sizes) == 4:
        batch, heads, seqlen, head_dim = q_sizes
    else:
        raise ValueError("q must be 4D tensor [batch, heads, seqlen, head_dim]")
    
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    head_dim_half = head_dim // 2
    
    # Flatten q, k for kernel
    rows = batch * heads * seqlen
    
    # Get cos/sin properties
    cos_sizes = tuple(int(s) for s in cos.sizes)
    cos_strides = tuple(int(s) for s in cos.strides)
    cos_stride = cos_strides[0] if len(cos_strides) > 0 else head_dim_half
    
    # Create position indices if not provided (GPU-native, no NumPy)
    if positions is None:
        # Default: positions = [0, 1, 2, ..., seqlen-1] repeated for each batch*heads
        # GPU-native generation: positions[i] = i % seqlen
        pos = _create_rope_positions_gpu(rows, seqlen, device_idx)
    else:
        pos = positions.flatten() if len(positions.sizes) > 1 else positions
    
    # Clone inputs since kernel modifies in-place, reshape to [rows, head_dim]
    q_work = q.contiguous().clone().reshape([rows, head_dim])
    k_work = k.contiguous().clone().reshape([rows, head_dim])
    
    # Kernel config
    block_size = _next_power_of_2(head_dim_half)
    rows_per_cta = 4  # Process 4 rows per CTA
    
    # Compile kernel
    mod_h, func_h, extra_params, shared_mem, num_warps = _get_rotary_kernel(
        q_is_fp16=False,
        q_is_bf16=False,
        k_is_fp16=False,
        k_is_bf16=False,
        block_size=block_size,
        rows_per_cta=rows_per_cta,
        device_idx=device_idx,
    )
    
    # Calculate grid
    num_ctas = (rows + rows_per_cta - 1) // rows_per_cta
    grid = (num_ctas, 1, 1)
    block_dim = (num_warps * 32, 1, 1)
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    # Build kernel args
    args = [
        q_work,
        k_work,
        cos,
        sin,
        pos,
        rows,
        head_dim,  # q_row_stride (elements per row)
        head_dim,  # k_row_stride
        cos_stride,
        head_dim_half,
    ]
    args.extend([None] * extra_params)
    
    # Launch kernel
    _C._cuda_launch(func_h, grid, block_dim, shared_mem, stream, args)
    
    # Reshape back to original shape
    q_rot = q_work.reshape(list(q_sizes))
    k_rot = k_work.reshape(list(q_sizes))
    
    return q_rot, k_rot


def apply_rotary_embedding_backward(
    dq,
    dk,
    cos,
    sin,
    positions=None,
):
    """Apply inverse rotary embedding for backward pass (gradient computation).
    
    RoPE forward rotates by angle θ. For gradients, we need to rotate back by -θ.
    This is achieved by negating the sin values.
    
    Args:
        dq, dk: Gradient tensors of shape (batch, heads, seqlen, head_dim), float32, CUDA
        cos, sin: Precomputed cosine/sine tables (max_position, head_dim // 2), float32
        positions: Optional position indices (batch, seqlen), int32
        
    Returns:
        Tuple (dq_pre, dk_pre) - gradients w.r.t. pre-RoPE q and k
    """
    vt, _C, _ = _get_vbt_modules()
    
    # Create -sin for inverse rotation
    neg_one = vt.full([1], -1.0).cuda()
    neg_sin = sin * neg_one
    
    # Apply inverse rotation (rotation by -θ)
    return apply_rotary_embedding(dq, dk, cos, neg_sin, positions)


__all__ = [
    "apply_rotary_embedding",
    "apply_rotary_embedding_backward",
]
