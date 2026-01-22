# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure VibeTensor implementation of GEMM using optimized Triton kernels.

This module wraps the kernel_factory Triton GEMM kernels for use
with VibeTensor tensors, with NO PyTorch dependency at runtime.

Usage:
    import vibetensor.torch as vt
    from vibe_kernels.gemm import vbt_native as gemm_ops
    
    a = vt.cuda.to_device(np.random.randn(4, 8).astype(np.float32))
    b = vt.cuda.to_device(np.random.randn(8, 16).astype(np.float32))
    c = gemm_ops.matmul(a, b)  # shape: (4, 16)
"""

from __future__ import annotations

import threading
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class GEMMConfig:
    """GEMM tile configuration."""
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int = 2


def _get_vbt_modules():
    """Lazy import VibeTensor modules."""
    import vibetensor.torch as vt
    from vibetensor import _C
    import vibetensor.triton as vt_triton
    return vt, _C, vt_triton


# -----------------------------------------------------------------------------
# Kernel Cache
# -----------------------------------------------------------------------------

# key -> (mod_handle, func_handle, extra_params, shared_mem)
_kernel_cache: Dict[str, Tuple[int, int, int, int]] = {}
_cache_lock = threading.Lock()


# Default configurations tuned for different problem sizes
# Note: Larger block sizes require more shared memory, so we prioritize
# smaller configs that work on most GPUs (48KB shared memory limit)
DEFAULT_CONFIGS = [
    # Small blocks with minimal shared memory (most compatible)
    GEMMConfig(32, 32, 32, num_warps=4, num_stages=2),
    GEMMConfig(64, 64, 32, num_warps=4, num_stages=2),
    GEMMConfig(64, 64, 64, num_warps=4, num_stages=2),
    # Larger blocks (require more shared memory)
    GEMMConfig(128, 64, 64, num_warps=4, num_stages=2),
    GEMMConfig(64, 128, 64, num_warps=4, num_stages=2),
    # Very large blocks (may exceed shared memory on some GPUs)
    # GEMMConfig(128, 128, 64, num_warps=8, num_stages=3),
]

# Maximum shared memory we should try to use (48KB is safe on most GPUs)
_MAX_SHARED_MEM = 48 * 1024


def _select_config(M: int, N: int, K: int) -> GEMMConfig:
    """Select optimal tiling configuration for given problem size.
    
    We prefer configs that:
    1. Have block sizes that fit the matrix (to reduce wasted computation)
    2. Don't require excessive shared memory
    """
    best_cfg = None
    best_score = None
    
    for cfg in DEFAULT_CONFIGS:
        # Skip if block sizes are much larger than matrix
        # (This would waste computation on padding)
        if cfg.block_m > M * 2 or cfg.block_n > N * 2:
            continue
            
        tiles_m = (M + cfg.block_m - 1) // cfg.block_m
        tiles_n = (N + cfg.block_n - 1) // cfg.block_n
        tile_count = tiles_m * tiles_n
        
        # Prefer larger coverage (fewer tiles) but not too large
        coverage = cfg.block_m * cfg.block_n
        
        # Score: prefer fewer tiles and smaller remainder
        remainder_k = K % cfg.block_k
        
        # Lower score is better: fewer tiles, smaller remainder, reasonable coverage
        score = (tile_count, remainder_k, -coverage)
        
        if best_score is None or score < best_score:
            best_cfg = cfg
            best_score = score
    
    if best_cfg is None:
        # Fallback to smallest config
        best_cfg = GEMMConfig(32, 32, 32, num_warps=4, num_stages=2)
    
    return best_cfg


def _get_gemm_kernel(cfg: GEMMConfig, device_idx: int) -> Tuple[int, int, int, int]:
    """Get compiled kernel handle for GEMM with given config.
    
    Returns:
        (mod_handle, func_handle, extra_params, shared_mem_bytes)
    """
    vt, _C, vt_triton = _get_vbt_modules()
    
    key = f"gemm_{cfg.block_m}_{cfg.block_n}_{cfg.block_k}_{cfg.num_warps}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            return _kernel_cache[key]
    
    # Import kernel and Triton compiler
    import os
    import importlib.util
    kf_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("gemm_kernels", os.path.join(kf_path, "gemm", "kernels.py"))
    kernels_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels_mod)
    gemm_kernel = kernels_mod.gemm_kernel
    import triton
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    # Build signature: pointers + scalars + constexprs
    # a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn
    signature = "*fp32,*fp32,*fp32,i32,i32,i32,i32,i32,i32,i32,i32,i32"
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    # Constexpr values
    meta = {
        "BLOCK_M": cfg.block_m,
        "BLOCK_N": cfg.block_n,
        "BLOCK_K": cfg.block_k,
    }
    
    # Build signature map for Triton ASTSource
    arg_names = gemm_kernel.arg_names
    params = gemm_kernel.params
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
    
    # Compile using Triton to get both PTX and metadata (including shared memory)
    target = driver.active.get_current_target()
    src = ASTSource(gemm_kernel, sig_map, meta, {})
    
    opt_dict = {
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
    
    compiled = triton.compile(src, target=target, options=opt_dict)
    
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
        entry = "gemm_kernel"
    
    # Get shared memory requirement from metadata
    md = getattr(compiled, "metadata", None)
    shared_mem = 0
    if hasattr(md, "shared"):
        shared_mem = int(md.shared)
    
    # Count extra params (Triton 3.5+ scratch pointers)
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

def matmul(a, b, out=None):
    """Matrix multiplication C = A @ B.
    
    Uses optimized Triton GEMM kernels compiled and launched via
    VibeTensor's PTX mechanism.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    
    Args:
        a: Input tensor A of shape (M, K), float32, CUDA
        b: Input tensor B of shape (K, N), float32, CUDA  
        out: Optional output tensor of shape (M, N)
        
    Returns:
        Output tensor C of shape (M, N)
    """
    vt, _C, _ = _get_vbt_modules()
    
    # Get tensor properties
    a_sizes = tuple(int(s) for s in a.sizes)
    a_strides = tuple(int(s) for s in a.strides)
    a_dtype = str(a.dtype)
    a_device = a.device
    
    b_sizes = tuple(int(s) for s in b.sizes)
    b_strides = tuple(int(s) for s in b.strides)
    b_dtype = str(b.dtype)
    
    # Validate
    if a_device[0] != 2:  # kDLCUDA = 2
        raise RuntimeError("matmul requires CUDA tensors")
    if b.device[0] != 2:
        raise RuntimeError("b must be on CUDA")
    if a_dtype != "float32" or b_dtype != "float32":
        raise TypeError("tensors must be float32 (for now)")
    if len(a_sizes) != 2 or len(b_sizes) != 2:
        raise ValueError("matmul requires 2D tensors")
    
    M, K = a_sizes
    K2, N = b_sizes
    
    if K != K2:
        raise ValueError(f"Inner dimensions must match: A is {a_sizes}, B is {b_sizes}")
    
    device_idx = int(a_device[1])
    
    # Allocate output
    if out is None:
        c = _C._cuda_empty([M, N], a_dtype, device_idx)
    else:
        c = out
        c_sizes = tuple(int(s) for s in c.sizes)
        if c_sizes != (M, N):
            raise ValueError(f"Output shape mismatch: expected ({M}, {N}), got {c_sizes}")
    
    # Handle edge cases
    if M == 0 or N == 0:
        return c
    if K == 0:
        # Zero-fill output
        c = _C._cuda_zeros([M, N], a_dtype, device_idx)
        return c
    
    # Select optimal config
    cfg = _select_config(M, N, K)
    
    # Get kernel handle and shared memory requirement
    _, func_handle, extra_params, shmem = _get_gemm_kernel(cfg, device_idx)
    
    # Compute grid
    import triton
    grid_m = triton.cdiv(M, cfg.block_m)
    grid_n = triton.cdiv(N, cfg.block_n)
    grid = (grid_m * grid_n, 1, 1)
    block = (cfg.num_warps * 32, 1, 1)
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    # Get strides
    stride_am, stride_ak = a_strides
    stride_bk, stride_bn = b_strides
    c_strides = tuple(int(s) for s in c.strides)
    stride_cm, stride_cn = c_strides
    
    # Args: a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn
    args = [a, b, c, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn]
    args.extend([None] * extra_params)  # Triton 3.5+ scratch pointers
    
    _C._cuda_launch(func_handle, grid, block, shmem, stream, args)
    
    return c


def mm(a, b, out=None):
    """Alias for matmul."""
    return matmul(a, b, out=out)


# -----------------------------------------------------------------------------
# GEMM Backward Kernels
# -----------------------------------------------------------------------------

import triton
import triton.language as tl


@triton.jit
def _dgrad_kernel(
    grad_out_ptr, weight_ptr, grad_input_ptr,
    M, N, K,
    stride_go_m, stride_go_n,
    stride_w_n, stride_w_k,
    stride_gi_m, stride_gi_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute grad_input = grad_out @ weight.T
    
    grad_out: [M, N]
    weight: [N, K] (stored as [K, N] transposed access pattern)
    grad_input: [M, K]
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_N)
    pid_m = pid // num_pid_k
    pid_k = pid % num_pid_k
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_k = offs_k < K
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_K)
    num_tiles_n = tl.cdiv(N, BLOCK_K)
    
    for n in range(0, num_tiles_n):
        n_start = n * BLOCK_K
        n_offsets = n_start + offs_n
        mask_n = n_offsets < N
        
        go_ptrs = grad_out_ptr + (offs_m[:, None] * stride_go_m + n_offsets[None, :] * stride_go_n)
        w_ptrs = weight_ptr + (n_offsets[:, None] * stride_w_n + offs_k[None, :] * stride_w_k)
        
        go = tl.load(go_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(go, w, out_dtype=tl.float32, allow_tf32=False)
    
    out_tile = acc.to(tl.float32)
    out_ptrs = grad_input_ptr + (offs_m[:, None] * stride_gi_m + offs_k[None, :] * stride_gi_k)
    tl.store(out_ptrs, out_tile, mask=mask_m[:, None] & mask_k[None, :])


@triton.jit
def _wgrad_kernel(
    input_ptr, grad_out_ptr, grad_weight_ptr,
    M, K, N,
    stride_a_m, stride_a_k,
    stride_go_m, stride_go_n,
    stride_gw_k, stride_gw_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute grad_weight = input.T @ grad_out
    
    input: [M, K]
    grad_out: [M, N]
    grad_weight: [K, N]
    """
    pid = tl.program_id(axis=0)
    num_pid_k = tl.cdiv(K, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_k = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_k = pid_k * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_k = offs_k < K
    mask_n = offs_n < N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_m = tl.arange(0, BLOCK_K)
    num_tiles_m = tl.cdiv(M, BLOCK_K)
    
    for m in range(0, num_tiles_m):
        m_start = m * BLOCK_K
        m_offsets = m_start + offs_m
        mask_m = m_offsets < M
        
        a_ptrs = input_ptr + (m_offsets[None, :] * stride_a_m + offs_k[:, None] * stride_a_k)
        go_ptrs = grad_out_ptr + (m_offsets[:, None] * stride_go_m + offs_n[None, :] * stride_go_n)
        
        a_tile = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
        go_tile = tl.load(go_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a_tile, go_tile, out_dtype=tl.float32, allow_tf32=False)
    
    out_tile = acc.to(tl.float32)
    out_ptrs = grad_weight_ptr + (offs_k[:, None] * stride_gw_k + offs_n[None, :] * stride_gw_n)
    tl.store(out_ptrs, out_tile, mask=mask_k[:, None] & mask_n[None, :])


# -----------------------------------------------------------------------------
# Backward Kernel Cache
# -----------------------------------------------------------------------------

_dgrad_cache: Dict[str, Tuple[int, int, int, int]] = {}
_wgrad_cache: Dict[str, Tuple[int, int, int, int]] = {}


def _get_dgrad_kernel_handle(cfg: GEMMConfig, device_idx: int) -> Tuple[int, int, int, int]:
    """Get compiled dgrad kernel."""
    vt, _C, vt_triton = _get_vbt_modules()
    
    key = f"dgrad_{cfg.block_m}_{cfg.block_n}_{cfg.block_k}_{cfg.num_warps}_{device_idx}"
    
    with _cache_lock:
        if key in _dgrad_cache:
            return _dgrad_cache[key]
    
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    signature = "*fp32,*fp32,*fp32,i32,i32,i32,i32,i32,i32,i32,i32,i32"
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    meta = {
        "BLOCK_M": cfg.block_m,
        "BLOCK_N": cfg.block_n,
        "BLOCK_K": cfg.block_k,
    }
    
    arg_names = _dgrad_kernel.arg_names
    params = _dgrad_kernel.params
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
    src = ASTSource(_dgrad_kernel, sig_map, meta, {})
    compiled = triton.compile(src, target=target, options={"num_warps": cfg.num_warps, "num_stages": cfg.num_stages})
    
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
        entry = "_dgrad_kernel"
    
    md = getattr(compiled, "metadata", None)
    shared_mem = 0
    if hasattr(md, "shared"):
        shared_mem = int(md.shared)
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _dgrad_cache[key] = (mod_h, func_h, extra_params, shared_mem)
    
    return mod_h, func_h, extra_params, shared_mem


def _get_wgrad_kernel_handle(cfg: GEMMConfig, device_idx: int) -> Tuple[int, int, int, int]:
    """Get compiled wgrad kernel."""
    vt, _C, vt_triton = _get_vbt_modules()
    
    key = f"wgrad_{cfg.block_m}_{cfg.block_n}_{cfg.block_k}_{cfg.num_warps}_{device_idx}"
    
    with _cache_lock:
        if key in _wgrad_cache:
            return _wgrad_cache[key]
    
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    signature = "*fp32,*fp32,*fp32,i32,i32,i32,i32,i32,i32,i32,i32,i32"
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    meta = {
        "BLOCK_M": cfg.block_m,
        "BLOCK_N": cfg.block_n,
        "BLOCK_K": cfg.block_k,
    }
    
    arg_names = _wgrad_kernel.arg_names
    params = _wgrad_kernel.params
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
    src = ASTSource(_wgrad_kernel, sig_map, meta, {})
    compiled = triton.compile(src, target=target, options={"num_warps": cfg.num_warps, "num_stages": cfg.num_stages})
    
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
        entry = "_wgrad_kernel"
    
    md = getattr(compiled, "metadata", None)
    shared_mem = 0
    if hasattr(md, "shared"):
        shared_mem = int(md.shared)
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _wgrad_cache[key] = (mod_h, func_h, extra_params, shared_mem)
    
    return mod_h, func_h, extra_params, shared_mem


# -----------------------------------------------------------------------------
# Public Backward API
# -----------------------------------------------------------------------------

def matmul_backward(
    grad_output,
    a,
    b,
    compute_grad_a: bool = True,
    compute_grad_b: bool = True,
):
    """Compute gradients for C = A @ B using optimized Triton kernels.
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    
    Args:
        grad_output: [M, N] gradient from upstream
        a: [M, K] input tensor A
        b: [K, N] input tensor B
        compute_grad_a: Whether to compute grad_a
        compute_grad_b: Whether to compute grad_b
        
    Returns:
        (grad_a, grad_b): gradients w.r.t. a and b (None if not computed)
    """
    vt, _C, _ = _get_vbt_modules()
    
    # Get tensor properties
    go_sizes = tuple(int(s) for s in grad_output.sizes)
    go_strides = tuple(int(s) for s in grad_output.strides)
    a_sizes = tuple(int(s) for s in a.sizes)
    a_strides = tuple(int(s) for s in a.strides)
    b_sizes = tuple(int(s) for s in b.sizes)
    b_strides = tuple(int(s) for s in b.strides)
    
    go_device = grad_output.device
    go_dtype = str(grad_output.dtype)
    
    if go_device[0] != 2:
        raise RuntimeError("matmul_backward requires CUDA tensors")
    if go_dtype != "float32":
        raise TypeError("matmul_backward requires float32")
    
    M, N = go_sizes
    K = a_sizes[1]
    device_idx = int(go_device[1])
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    grad_a = None
    grad_b = None
    
    # Compute grad_a = grad_output @ b.T
    if compute_grad_a and K > 0 and N > 0:
        grad_a = _C._cuda_zeros([M, K], go_dtype, device_idx)
        ga_strides = tuple(int(s) for s in grad_a.strides)
        
        # Use conservative config to avoid shared memory issues
        cfg = GEMMConfig(64, 64, 32, num_warps=4, num_stages=2)
        _, func_h, extra_params, shmem = _get_dgrad_kernel_handle(cfg, device_idx)
        
        grid_m = (M + cfg.block_m - 1) // cfg.block_m
        grid_k = (K + cfg.block_n - 1) // cfg.block_n
        grid = (grid_m * grid_k, 1, 1)
        block = (cfg.num_warps * 32, 1, 1)
        
        # b is [K, N], we need to access it as [N, K] for the multiplication
        # grad_a[M,K] = grad_output[M,N] @ b.T[N,K]
        # For transposed access: swap strides so b.T[n,k] = b[k,n]
        args = [
            grad_output, b, grad_a,
            M, N, K,
            go_strides[0], go_strides[1],
            b_strides[1], b_strides[0],  # SWAPPED: stride_w_n=b_stride_n, stride_w_k=b_stride_k
            ga_strides[0], ga_strides[1],
        ]
        args.extend([None] * extra_params)
        
        _C._cuda_launch(func_h, grid, block, shmem, stream, args)
    
    # Compute grad_b = a.T @ grad_output
    if compute_grad_b and M > 0 and K > 0:
        grad_b = _C._cuda_zeros([K, N], go_dtype, device_idx)
        gb_strides = tuple(int(s) for s in grad_b.strides)
        
        # Use conservative config to avoid shared memory issues
        cfg = GEMMConfig(64, 64, 32, num_warps=4, num_stages=2)
        _, func_h, extra_params, shmem = _get_wgrad_kernel_handle(cfg, device_idx)
        
        grid_k = (K + cfg.block_m - 1) // cfg.block_m
        grid_n = (N + cfg.block_n - 1) // cfg.block_n
        grid = (grid_k * grid_n, 1, 1)
        block = (cfg.num_warps * 32, 1, 1)
        
        # grad_b[K,N] = a.T[K,M] @ grad_output[M,N]
        args = [
            a, grad_output, grad_b,
            M, K, N,
            a_strides[0], a_strides[1],
            go_strides[0], go_strides[1],
            gb_strides[0], gb_strides[1],
        ]
        args.extend([None] * extra_params)
        
        _C._cuda_launch(func_h, grid, block, shmem, stream, args)
    
    return grad_a, grad_b


__all__ = [
    "matmul",
    "mm",
    "matmul_backward",
]
