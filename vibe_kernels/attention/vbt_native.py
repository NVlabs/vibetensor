# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure VibeTensor implementation of Flash Attention using Triton kernels.

This module wraps attention kernels for use with VibeTensor tensors,
with NO PyTorch dependency at runtime.

Usage:
    import vibetensor.torch as vt
    from vibe_kernels.attention import vbt_native as attn_ops
    
    # q, k, v: [batch, heads, seqlen, head_dim]
    q = vt.cuda.to_device(np.random.randn(2, 8, 128, 64).astype(np.float32))
    k = vt.cuda.to_device(np.random.randn(2, 8, 128, 64).astype(np.float32))
    v = vt.cuda.to_device(np.random.randn(2, 8, 128, 64).astype(np.float32))
    out = attn_ops.attention(q, k, v, causal=True)
"""

from __future__ import annotations

import math
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
# Attention Forward Kernel
# -----------------------------------------------------------------------------

@triton.jit
def _attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Flash attention forward kernel.
    
    Each program computes one (batch, head, block_m) tile of output.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Compute base pointers
    q_base = Q_ptr + off_z * stride_qz + off_h * stride_qh
    k_base = K_ptr + off_z * stride_kz + off_h * stride_kh
    v_base = V_ptr + off_z * stride_vz + off_h * stride_vh
    o_base = O_ptr + off_z * stride_oz + off_h * stride_oh
    
    # Offsets within block
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    
    # Load Q block [BLOCK_M, HEAD_DIM]
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Initialize accumulators
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Scale factor
    qk_scale = sm_scale * 1.44269504  # sm_scale / log(2)
    
    # Determine iteration range based on causal mask
    if IS_CAUSAL:
        hi = min((start_m + 1) * BLOCK_M, N_CTX)
    else:
        hi = N_CTX
    
    # Iterate over K, V blocks
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K block [BLOCK_N, HEAD_DIM]
        k_offs_n = start_n + offs_n
        k_ptrs = k_base + k_offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k_mask = k_offs_n[:, None] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        # Load V block [BLOCK_N, HEAD_DIM]  
        v_ptrs = v_base + k_offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        
        # Compute QK^T [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k), allow_tf32=False)
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= k_offs_n[None, :]
            qk = tl.where(causal_mask, qk, -float("inf"))
        
        # Apply out-of-bounds mask
        qk = tl.where(k_offs_n[None, :] < N_CTX, qk, -float("inf"))
        
        # Compute softmax (online algorithm)
        m_ij = tl.max(qk, 1) * qk_scale
        m_ij = tl.maximum(m_ij, m_i)
        
        # Compute exp2(qk - m_ij) - MUST use exp2 with 1/log(2) scaling!
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        
        # Update accumulators with rescaling - use exp2 for consistency
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        
        # Accumulate
        l_ij = tl.sum(p, 1)
        l_i = l_i + l_ij
        acc = acc + tl.dot(p.to(q.dtype), v, allow_tf32=False)
        
        m_i = m_ij
    
    # Finalize: divide by sum
    acc = acc / l_i[:, None]
    
    # Store output [BLOCK_M, HEAD_DIM]
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    o_mask = offs_m[:, None] < N_CTX
    tl.store(o_ptrs, acc.to(q.dtype), mask=o_mask)


# -----------------------------------------------------------------------------
# Kernel Cache
# -----------------------------------------------------------------------------

_kernel_cache: Dict[str, Tuple[int, int, int, int]] = {}
_cache_lock = threading.Lock()


def _get_attention_kernel(
    head_dim: int,
    block_m: int,
    block_n: int,
    is_causal: bool,
    device_idx: int,
) -> Tuple[int, int, int, int]:
    """Get compiled kernel handle for attention.
    
    Uses Triton's warmup() with MockTensor to get the optimized JIT-compiled kernel
    without requiring GPU memory allocation during compilation.
    
    Returns:
        (mod_handle, func_handle, extra_params, shared_mem_bytes)
    """
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.runtime.jit import MockTensor
    
    key = f"attn_fwd_{head_dim}_{block_m}_{block_n}_{is_causal}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            return _kernel_cache[key]
    
    # Use warmup() with MockTensor to get optimized kernel without GPU allocation
    # Use non-specializable values (3, 5) to prevent constexpr specialization
    warmup_z, warmup_h, warmup_n_ctx = 3, 5, block_m
    
    # Create MockTensors for warmup (no GPU allocation needed)
    mock_q = MockTensor(tl.float32, [warmup_z, warmup_h, warmup_n_ctx, head_dim])
    mock_k = MockTensor(tl.float32, [warmup_z, warmup_h, warmup_n_ctx, head_dim])
    mock_v = MockTensor(tl.float32, [warmup_z, warmup_h, warmup_n_ctx, head_dim])
    mock_o = MockTensor(tl.float32, [warmup_z, warmup_h, warmup_n_ctx, head_dim])
    
    # Compute strides for contiguous layout [Z, H, N_CTX, HEAD_DIM]
    stride_k = 1
    stride_m = head_dim
    stride_h = warmup_n_ctx * head_dim
    stride_z = warmup_h * warmup_n_ctx * head_dim
    
    # Warmup returns the JIT-compiled kernel with optimized CUBIN
    compiled = _attn_fwd_kernel.warmup(
        mock_q, mock_k, mock_v, mock_o,
        1.0 / (head_dim ** 0.5),  # sm_scale
        stride_z, stride_h, stride_m, stride_k,  # Q strides
        stride_z, stride_h, stride_m, stride_k,  # K strides
        stride_z, stride_h, stride_m, stride_k,  # V strides
        stride_z, stride_h, stride_m, stride_k,  # O strides
        warmup_z, warmup_h, warmup_n_ctx,
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=head_dim,
        IS_CAUSAL=1 if is_causal else 0,
        num_warps=4, num_stages=2,
        grid=(1,)
    )
    
    # Get CUBIN (preferred) or PTX
    asm = compiled.asm
    if "cubin" in asm:
        binary = asm["cubin"]
    elif "ptx" in asm:
        binary = asm["ptx"]
        if isinstance(binary, str):
            binary = binary.encode("utf-8")
    else:
        raise RuntimeError("Triton compile produced no CUBIN or PTX")
    
    # Get kernel name and shared memory
    entry = compiled.name
    shared_mem = int(compiled.metadata.shared)
    
    # Count extra params (Triton may add internal params)
    ptx_for_count = asm.get("ptx", b"")
    if isinstance(ptx_for_count, str):
        ptx_for_count = ptx_for_count.encode("utf-8")
    
    # Count non-constexpr params from compiled signature
    sig = compiled.src.signature
    expected_params = sum(1 for v in sig.values() if v != "constexpr")
    total_params = vt_triton._count_entry_params(ptx_for_count, entry)
    extra_params = max(0, total_params - expected_params)
    
    # Load module
    mod_h = _C._cuda_module_load_ptx(binary)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    # Set max dynamic shared memory if kernel requires > 48KB
    if shared_mem > 49152:
        _C._cuda_func_set_attribute(func_h, 8, shared_mem)  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    
    with _cache_lock:
        _kernel_cache[key] = (mod_h, func_h, extra_params, shared_mem)
    
    return mod_h, func_h, extra_params, shared_mem


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def attention(
    q,
    k,
    v,
    causal: bool = False,
    sm_scale: Optional[float] = None,
):
    """Flash attention forward pass.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    
    Args:
        q: Query tensor [batch, heads, seqlen, head_dim], float32, CUDA
        k: Key tensor [batch, heads, seqlen, head_dim], float32, CUDA
        v: Value tensor [batch, heads, seqlen, head_dim], float32, CUDA
        causal: Whether to apply causal masking
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))
        
    Returns:
        Output tensor [batch, heads, seqlen, head_dim]
    """
    vt, _C, _ = _get_vbt_modules()
    
    # Get tensor properties
    q_sizes = tuple(int(s) for s in q.sizes)
    q_strides = tuple(int(s) for s in q.strides)
    k_sizes = tuple(int(s) for s in k.sizes)
    k_strides = tuple(int(s) for s in k.strides)
    v_sizes = tuple(int(s) for s in v.sizes)
    v_strides = tuple(int(s) for s in v.strides)
    
    q_dtype = str(q.dtype)
    q_device = q.device
    
    # Validate inputs
    if q_device[0] != 2:  # kDLCUDA = 2
        raise RuntimeError("attention requires CUDA tensors")
    if q_dtype != "float32":
        raise TypeError("attention requires float32 (for now)")
    if len(q_sizes) != 4:
        raise ValueError("attention requires 4D tensors [batch, heads, seqlen, head_dim]")
    if q_sizes != k_sizes or q_sizes != v_sizes:
        raise ValueError("q, k, v must have same shape")
    
    Z, H, N_CTX, HEAD_DIM = q_sizes
    
    if HEAD_DIM not in {16, 32, 64, 128}:
        raise ValueError(f"head_dim must be 16, 32, 64, or 128, got {HEAD_DIM}")
    
    device_idx = int(q_device[1])
    
    # Default scale
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    # Select block sizes based on head_dim and seqlen
    # Use smaller blocks to keep shared memory under 48KB
    if HEAD_DIM <= 32:
        BLOCK_M, BLOCK_N = 64, 32
    elif HEAD_DIM <= 64:
        BLOCK_M, BLOCK_N = 32, 32
    else:
        BLOCK_M, BLOCK_N = 32, 16
    
    # Ensure blocks don't exceed seqlen
    BLOCK_M = min(BLOCK_M, N_CTX)
    BLOCK_N = min(BLOCK_N, N_CTX)
    
    # Allocate output
    out = _C._cuda_empty(list(q_sizes), q_dtype, device_idx)
    out_strides = tuple(int(s) for s in out.strides)
    
    # Get kernel
    _, func_h, extra_params, shmem = _get_attention_kernel(
        HEAD_DIM, BLOCK_M, BLOCK_N, causal, device_idx
    )
    
    # Compute grid
    grid_m = (N_CTX + BLOCK_M - 1) // BLOCK_M
    grid = (grid_m, Z * H, 1)
    block = (128, 1, 1)  # num_warps=4
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    # Build args with proper types (kernel expects i32 for strides and scalars)
    # Note: stride_*k (innermost) are constexprs when = 1, so we don't pass them
    args = [
        q, k, v, out,
        _C._cuda_arg_f32(sm_scale),
        _C._cuda_arg_i32(q_strides[0]), _C._cuda_arg_i32(q_strides[1]),
        _C._cuda_arg_i32(q_strides[2]),  # stride_qk is constexpr
        _C._cuda_arg_i32(k_strides[0]), _C._cuda_arg_i32(k_strides[1]),
        _C._cuda_arg_i32(k_strides[2]),  # stride_kk is constexpr
        _C._cuda_arg_i32(v_strides[0]), _C._cuda_arg_i32(v_strides[1]),
        _C._cuda_arg_i32(v_strides[2]),  # stride_vk is constexpr
        _C._cuda_arg_i32(out_strides[0]), _C._cuda_arg_i32(out_strides[1]),
        _C._cuda_arg_i32(out_strides[2]),  # stride_ok is constexpr
        _C._cuda_arg_i32(Z), _C._cuda_arg_i32(H), _C._cuda_arg_i32(N_CTX),
    ]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block, shmem, stream, args)
    
    return out


def scaled_dot_product_attention(
    q,
    k,
    v,
    is_causal: bool = False,
    scale: Optional[float] = None,
):
    """Alias for attention() with PyTorch-like signature."""
    return attention(q, k, v, causal=is_causal, sm_scale=scale)


# -----------------------------------------------------------------------------
# Attention Forward with LSE (for backward)
# -----------------------------------------------------------------------------

@triton.jit
def _attn_fwd_with_lse_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_mz, stride_mh,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Flash attention forward kernel that also saves LSE for backward."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    q_base = Q_ptr + off_z * stride_qz + off_h * stride_qh
    k_base = K_ptr + off_z * stride_kz + off_h * stride_kh
    v_base = V_ptr + off_z * stride_vz + off_h * stride_vh
    o_base = O_ptr + off_z * stride_oz + off_h * stride_oh
    m_base = M_ptr + off_z * stride_mz + off_h * stride_mh
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    qk_scale = sm_scale * 1.44269504
    
    if IS_CAUSAL:
        hi = min((start_m + 1) * BLOCK_M, N_CTX)
    else:
        hi = N_CTX
    
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_offs_n = start_n + offs_n
        k_ptrs = k_base + k_offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k_mask = k_offs_n[:, None] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        v_ptrs = v_base + k_offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        
        qk = tl.dot(q, tl.trans(k), allow_tf32=False)
        
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= k_offs_n[None, :]
            qk = tl.where(causal_mask, qk, -float("inf"))
        
        qk = tl.where(k_offs_n[None, :] < N_CTX, qk, -float("inf"))
        
        m_ij = tl.max(qk, 1) * qk_scale
        m_ij = tl.maximum(m_ij, m_i)
        
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        
        l_ij = tl.sum(p, 1)
        l_i = l_i + l_ij
        acc = acc + tl.dot(p.to(q.dtype), v, allow_tf32=False)
        
        m_i = m_ij
    
    acc = acc / l_i[:, None]
    
    # Store LSE in log2 scale (m_i + log2(l_i)) for backward
    lse = m_i + tl.math.log2(l_i)
    m_ptrs = m_base + offs_m
    tl.store(m_ptrs, lse, mask=offs_m < N_CTX)
    
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    o_mask = offs_m[:, None] < N_CTX
    tl.store(o_ptrs, acc.to(q.dtype), mask=o_mask)


# -----------------------------------------------------------------------------
# Attention Backward Kernels (Simple 4-stride design matching kernel.py)
# -----------------------------------------------------------------------------

@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Compute delta = sum(o * do) for each position.
    
    Assumes contiguous layout [B*H, N_CTX, HEAD_DIM].
    """
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    
    # Direct pointer math assuming contiguous [B*H, N_CTX, HEAD_DIM]
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def _attn_bwd_dkdv(
    dk, dv,
    Q, k, v, sm_scale, DO, M, D,
    stride_tok, stride_d,
    H, N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_n, start_m, num_steps,
    MASK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Helper kernel for computing dk, dv."""
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    
    for _ in range(num_steps):
        qT = tl.load(qT_ptrs)
        offs_m_cur = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m_cur)
        
        qkT = tl.dot(k, qT, allow_tf32=False)
        pT = tl.math.exp2(qkT - m[None, :])
        
        if MASK:
            mask = offs_m_cur[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        
        do = tl.load(do_ptrs)
        dv += tl.dot(pT.to(DTYPE), do, allow_tf32=False)
        
        Di = tl.load(D + offs_m_cur)
        dpT = tl.dot(v, tl.trans(do), allow_tf32=False).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dk += tl.dot(dsT.to(DTYPE), tl.trans(qT), allow_tf32=False)
        
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    
    return dk, dv


@triton.jit
def _attn_bwd_dq(
    dq, q, K, V, do, m, D,
    stride_tok, stride_d,
    H, N_CTX,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_m, start_n, num_steps,
    MASK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Helper kernel for computing dq."""
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    
    Di = tl.load(D + offs_m)
    
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    
    for _ in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        
        qk = tl.dot(q, kT, allow_tf32=False)
        p = tl.math.exp2(qk - m)
        
        if MASK:
            offs_n_cur = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n_cur[None, :]
            p = tl.where(mask, p, 0.0)
        
        dp = tl.dot(do, vT, allow_tf32=False).to(tl.float32)
        ds = p * (dp - Di[:, None])
        dq += tl.dot(ds.to(DTYPE), tl.trans(kT), allow_tf32=False)
        
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    
    return dq


@triton.jit
def _attn_bwd(
    Q, K, V, sm_scale, DO, DQ, DK, DV, M, D,
    stride_z, stride_h, stride_tok, stride_d,
    H, N_CTX,
    CAUSAL: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INPUT_IS_BF16: tl.constexpr,
):
    """Main attention backward kernel matching kernel.py design.
    
    Uses only 4 strides and pointer adjustment for all tensors.
    """
    LN2: tl.constexpr = 0.6931471824645996
    if INPUT_IS_BF16:
        DTYPE: tl.constexpr = tl.bfloat16
    else:
        DTYPE: tl.constexpr = tl.float32
    
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)
    
    # Adjust all pointers by batch-head offset
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    
    offs_k = tl.arange(0, HEAD_DIM)
    
    # Compute dk, dv
    start_n = pid * BLOCK_N1
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    
    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk, dv, Q, k, v, sm_scale, DO, M, D,
            stride_tok, stride_d, H, N_CTX,
            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,
            start_n, start_m, num_steps, True, DTYPE,
        )
        start_m += num_steps * MASK_BLOCK_M1
        num_steps = (N_CTX - start_m) // BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk, dv, Q, k, v, sm_scale, DO, M, D,
            stride_tok, stride_d, H, N_CTX,
            BLOCK_M1, BLOCK_N1, HEAD_DIM,
            start_n, start_m, num_steps, False, DTYPE,
        )
    else:
        start_m = 0
        num_steps = N_CTX // BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk, dv, Q, k, v, sm_scale, DO, M, D,
            stride_tok, stride_d, H, N_CTX,
            BLOCK_M1, BLOCK_N1, HEAD_DIM,
            start_n, start_m, num_steps, False, DTYPE,
        )
    
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)
    
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)
    
    # Compute dq - iterate backwards like original (end_n -> 0)
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    
    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    m = tl.load(M + offs_m)
    m = m[:, None]
    
    if CAUSAL:
        # First pass: masked region near diagonal (iterate backwards from end_n)
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(
            dq, q, K, V, do, m, D,
            stride_tok, stride_d, H, N_CTX,
            BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,
            start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps, True, DTYPE,
        )
        # Second pass: unmasked region (continue backwards from 0)
        end_n -= num_steps * MASK_BLOCK_N2
        num_steps = end_n // BLOCK_N2
        dq = _attn_bwd_dq(
            dq, q, K, V, do, m, D,
            stride_tok, stride_d, H, N_CTX,
            BLOCK_M2, BLOCK_N2, HEAD_DIM,
            start_m, end_n - num_steps * BLOCK_N2, num_steps, False, DTYPE,
        )
    else:
        num_steps = N_CTX // BLOCK_N2
        dq = _attn_bwd_dq(
            dq, q, K, V, do, m, D,
            stride_tok, stride_d, H, N_CTX,
            BLOCK_M2, BLOCK_N2, HEAD_DIM,
            start_m, 0, num_steps, False, DTYPE,
        )
    
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2  # Final scaling to convert from log2 to natural log scale
    tl.store(dq_ptrs, dq)


# -----------------------------------------------------------------------------
# Backward Kernel Cache and Compilation
# -----------------------------------------------------------------------------

_bwd_kernel_cache: Dict[str, Tuple[int, int, int, int]] = {}
_bwd_preprocess_cache: Dict[str, Tuple[int, int, int, int]] = {}
_fwd_lse_cache: Dict[str, Tuple[int, int, int, int]] = {}


def _get_attn_fwd_lse_kernel(head_dim, block_m, block_n, is_causal, device_idx):
    """Get compiled forward kernel with LSE."""
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    key = f"attn_fwd_lse_{head_dim}_{block_m}_{block_n}_{is_causal}_{device_idx}"
    
    with _cache_lock:
        if key in _fwd_lse_cache:
            return _fwd_lse_cache[key]
    
    # 5 pointers + sm_scale + 18 strides + Z,H,N_CTX = 5+1+18+3 = 27 non-constexpr
    signature = "*fp32,*fp32,*fp32,*fp32,*fp32,fp32," + ",".join(["i32"] * 21)
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    meta = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "HEAD_DIM": head_dim,
        "IS_CAUSAL": 1 if is_causal else 0,
    }
    
    arg_names = _attn_fwd_with_lse_kernel.arg_names
    params = _attn_fwd_with_lse_kernel.params
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
    src = ASTSource(_attn_fwd_with_lse_kernel, sig_map, meta, {})
    compiled = triton.compile(src, target=target, options={"num_warps": 4, "num_stages": 2})
    
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
        entry = "_attn_fwd_with_lse_kernel"
    
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


def _get_bwd_preprocess_kernel(head_dim, block_m, device_idx):
    """Get compiled backward preprocess kernel."""
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    key = f"attn_bwd_pre_{head_dim}_{block_m}_{device_idx}"
    
    with _cache_lock:
        if key in _bwd_preprocess_cache:
            return _bwd_preprocess_cache[key]
    
    # 3 pointers + Z,H,N_CTX = 6 non-constexpr (no strides in new design)
    signature = "*fp32,*fp32,*fp32,i32,i32,i32"
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    meta = {
        "BLOCK_M": block_m,
        "HEAD_DIM": head_dim,
    }
    
    arg_names = _attn_bwd_preprocess.arg_names
    params = _attn_bwd_preprocess.params
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
    src = ASTSource(_attn_bwd_preprocess, sig_map, meta, {})
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
        entry = "_attn_bwd_preprocess"
    
    md = getattr(compiled, "metadata", None)
    shared_mem = 0
    if hasattr(md, "shared"):
        shared_mem = int(md.shared)
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _bwd_preprocess_cache[key] = (mod_h, func_h, extra_params, shared_mem)
    
    return mod_h, func_h, extra_params, shared_mem


def _get_bwd_kernel(head_dim, block_m1, block_n1, block_m2, block_n2, is_causal, device_idx):
    """Get compiled backward kernel."""
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    blk_slice_factor = 2 if int(block_m1) >= 32 else 1
    key = f"attn_bwd_{head_dim}_{block_m1}_{block_n1}_{block_m2}_{block_n2}_{blk_slice_factor}_{is_causal}_{device_idx}"
    
    with _cache_lock:
        if key in _bwd_kernel_cache:
            return _bwd_kernel_cache[key]
    
    # 10 pointers + sm_scale + 4 strides + H,N_CTX = 10+1+4+2 = 17 non-constexpr
    signature = "*fp32,*fp32,*fp32,fp32,*fp32,*fp32,*fp32,*fp32,*fp32,*fp32,i32,i32,i32,i32,i32,i32"
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    meta = {
        "CAUSAL": 1 if is_causal else 0,
        "BLOCK_M1": block_m1,
        "BLOCK_N1": block_n1,
        "BLOCK_M2": block_m2,
        "BLOCK_N2": block_n2,
        "BLK_SLICE_FACTOR": blk_slice_factor,
        "HEAD_DIM": head_dim,
        "INPUT_IS_BF16": 0,  # We only support float32 for now
    }
    
    arg_names = _attn_bwd.arg_names
    params = _attn_bwd.params
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
    src = ASTSource(_attn_bwd, sig_map, meta, {})
    compiled = triton.compile(src, target=target, options={"num_warps": 4, "num_stages": 5})
    
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
        entry = "_attn_bwd"
    
    md = getattr(compiled, "metadata", None)
    shared_mem = 0
    if hasattr(md, "shared"):
        shared_mem = int(md.shared)
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _bwd_kernel_cache[key] = (mod_h, func_h, extra_params, shared_mem)
    
    return mod_h, func_h, extra_params, shared_mem


# -----------------------------------------------------------------------------
# Public API: Forward with LSE + Backward
# -----------------------------------------------------------------------------

def attention_with_lse(
    q, k, v,
    causal: bool = False,
    sm_scale: Optional[float] = None,
):
    """Flash attention forward that also returns LSE for backward.
    
    Args:
        q, k, v: [batch, heads, seqlen, head_dim]
        causal: Whether to use causal masking
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))
        
    Returns:
        (out, lse): output tensor and log-sum-exp for backward
    """
    vt, _C, _ = _get_vbt_modules()
    
    q_sizes = tuple(int(s) for s in q.sizes)
    q_strides = tuple(int(s) for s in q.strides)
    k_strides = tuple(int(s) for s in k.strides)
    v_strides = tuple(int(s) for s in v.strides)
    
    q_dtype = str(q.dtype)
    q_device = q.device
    
    if q_device[0] != 2:
        raise RuntimeError("attention requires CUDA tensors")
    if q_dtype != "float32":
        raise TypeError("attention requires float32")
    if len(q_sizes) != 4:
        raise ValueError("attention requires 4D tensors [B, H, S, D]")
    
    Z, H, N_CTX, HEAD_DIM = q_sizes
    
    if HEAD_DIM not in {16, 32, 64, 128}:
        raise ValueError(f"head_dim must be 16, 32, 64, or 128, got {HEAD_DIM}")
    
    device_idx = int(q_device[1])
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    if HEAD_DIM <= 32:
        BLOCK_M, BLOCK_N = 64, 32
    elif HEAD_DIM <= 64:
        BLOCK_M, BLOCK_N = 32, 32
    else:
        BLOCK_M, BLOCK_N = 32, 16
    
    BLOCK_M = min(BLOCK_M, N_CTX)
    BLOCK_N = min(BLOCK_N, N_CTX)
    
    out = _C._cuda_empty(list(q_sizes), q_dtype, device_idx)
    lse = _C._cuda_empty([Z, H, N_CTX], q_dtype, device_idx)
    
    out_strides = tuple(int(s) for s in out.strides)
    lse_strides = (H * N_CTX, N_CTX)
    
    _, func_h, extra_params, shmem = _get_attn_fwd_lse_kernel(
        HEAD_DIM, BLOCK_M, BLOCK_N, causal, device_idx
    )
    
    grid_m = (N_CTX + BLOCK_M - 1) // BLOCK_M
    grid = (grid_m, Z * H, 1)
    block = (128, 1, 1)
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    args = [
        q, k, v, out, lse,
        sm_scale,
        q_strides[0], q_strides[1], q_strides[2], q_strides[3],
        k_strides[0], k_strides[1], k_strides[2], k_strides[3],
        v_strides[0], v_strides[1], v_strides[2], v_strides[3],
        out_strides[0], out_strides[1], out_strides[2], out_strides[3],
        lse_strides[0], lse_strides[1],
        Z, H, N_CTX,
    ]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block, shmem, stream, args)
    
    return out, lse


def attention_backward(
    grad_out, q, k, v, out, lse,
    causal: bool = False,
    sm_scale: Optional[float] = None,
):
    """Flash attention backward pass.
    
    Args:
        grad_out: [B, H, S, D] gradient from upstream
        q, k, v: [B, H, S, D] saved inputs
        out: [B, H, S, D] saved forward output
        lse: [B, H, S] log-sum-exp from forward
        causal: Whether causal masking was used
        sm_scale: Softmax scale (must match forward)
        
    Returns:
        (dq, dk, dv): gradients w.r.t. q, k, v
    """
    vt, _C, _ = _get_vbt_modules()
    
    q_sizes = tuple(int(s) for s in q.sizes)
    q_strides = tuple(int(s) for s in q.strides)
    
    Z, H, N_CTX, HEAD_DIM = q_sizes
    device_idx = int(q.device[1])
    q_dtype = str(q.dtype)
    
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    # Allocate outputs
    dq = _C._cuda_empty(list(q_sizes), q_dtype, device_idx)
    dk = _C._cuda_empty(list(q_sizes), q_dtype, device_idx)
    dv = _C._cuda_empty(list(q_sizes), q_dtype, device_idx)
    delta = _C._cuda_empty([Z, H, N_CTX], q_dtype, device_idx)
    
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    # Step 1: Preprocess - compute delta = sum(o * do)
    # Find best PRE_BLOCK that divides N_CTX
    for candidate in (128, 64, 32, 16):
        if N_CTX % candidate == 0:
            PRE_BLOCK = candidate
            break
    else:
        PRE_BLOCK = 16
    
    _, pre_func, pre_extra, pre_shmem = _get_bwd_preprocess_kernel(HEAD_DIM, PRE_BLOCK, device_idx)
    
    # Configure shared memory if needed
    if pre_shmem > 49152:
        _C._cuda_func_set_attribute(pre_func, 8, pre_shmem)
    
    pre_grid = (N_CTX // PRE_BLOCK, Z * H, 1)
    pre_args = [
        out, grad_out, delta,
        _C._cuda_arg_i32(Z), _C._cuda_arg_i32(H), _C._cuda_arg_i32(N_CTX)
    ]
    pre_args.extend([None] * pre_extra)
    
    _C._cuda_launch(pre_func, pre_grid, (128, 1, 1), pre_shmem, stream, pre_args)
    
    # Step 2: Main backward kernel
    # Use block sizes based on HEAD_DIM to stay under 48KB shared memory limit
    # Shared mem ~ (BLOCK_N1 + BLOCK_M2) * HEAD_DIM * sizeof(float) * num_buffers
    def _pick_block(target):
        for cand in (target, target // 2, target // 4, target // 8):
            if cand <= 0:
                continue
            if N_CTX % cand == 0:
                return cand
        return max(16, min(target, N_CTX))
    
    # Adjust block sizes based on HEAD_DIM to stay under shared memory limit
    if HEAD_DIM <= 32:
        # Small head: can use larger blocks
        max_block = 64
    elif HEAD_DIM <= 64:
        # Medium head: use smaller blocks  
        max_block = 32
    else:
        # Large head: use minimum blocks
        max_block = 32
    
    BLOCK_N1 = _pick_block(max_block)
    BLOCK_M1 = min(32, BLOCK_N1)
    BLOCK_M2 = _pick_block(max_block)
    BLOCK_N2 = min(32, BLOCK_M2)
    
    # Scale K for backward (RCP_LN2 = 1/ln(2)) - THIS IS CRITICAL!
    # The original kernel.py does: arg_k = k * (sm_scale * RCP_LN2)
    RCP_LN2 = 1.4426950408889634
    scale_factor = sm_scale * RCP_LN2
    # Create scaled copy of k
    k_scaled = _C._cuda_empty(list(k.sizes), str(k.dtype), device_idx)
    # Simple element-wise multiply: k_scaled = k * scale_factor
    # We need to use VibeTensor ops for this
    import vibetensor.torch as vt
    k_scaled = k * vt.full([1], scale_factor).cuda()
    
    _, bwd_func, bwd_extra, bwd_shmem = _get_bwd_kernel(
        HEAD_DIM, BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, causal, device_idx
    )
    
    # Configure shared memory if needed
    if bwd_shmem > 49152:
        _C._cuda_func_set_attribute(bwd_func, 8, bwd_shmem)
    
    bwd_grid = (N_CTX // BLOCK_N1, 1, Z * H)
    
    # Only pass 4 strides (stride_z, stride_h, stride_tok, stride_d)
    bwd_args = [
        q, k_scaled, v,
        _C._cuda_arg_f32(sm_scale),
        grad_out, dq, dk, dv, lse, delta,
        _C._cuda_arg_i32(q_strides[0]), _C._cuda_arg_i32(q_strides[1]),
        _C._cuda_arg_i32(q_strides[2]), _C._cuda_arg_i32(q_strides[3]),
        _C._cuda_arg_i32(H), _C._cuda_arg_i32(N_CTX),
    ]
    bwd_args.extend([None] * bwd_extra)
    
    _C._cuda_launch(bwd_func, bwd_grid, (128, 1, 1), bwd_shmem, stream, bwd_args)
    
    return dq, dk, dv


__all__ = [
    "attention",
    "attention_with_lse",
    "attention_backward",
    "scaled_dot_product_attention",
]
