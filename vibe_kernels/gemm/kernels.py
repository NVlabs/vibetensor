# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Triton JIT kernels for GEMM.

These kernels are framework-agnostic and can be compiled to PTX
for use with any tensor framework that supports CUDA.

The kernels match the performance-tuned implementations from triton_impl.py
but use explicit constexpr parameters instead of closures.
"""

import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Classic GEMM kernel: C = A @ B.
    
    This kernel computes matrix multiplication with explicit block tiling.
    Each program instance computes one (BLOCK_M, BLOCK_N) output tile.
    
    Args:
        a_ptr: Pointer to A matrix (M, K)
        b_ptr: Pointer to B matrix (K, N)
        c_ptr: Pointer to output C matrix (M, N)
        M, N, K: Matrix dimensions
        stride_am, stride_ak: Strides for A
        stride_bk, stride_bn: Strides for B
        stride_cm, stride_cn: Strides for C
        BLOCK_M, BLOCK_N, BLOCK_K: Tile sizes (constexpr)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k * BLOCK_K
        k_offsets = k_start + offs_k
        k_mask = k_offsets < K

        # Load A tile
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        
        # Load B tile
        b_ptrs = b_ptr + (k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        
        # Accumulate
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # Store output
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def gemm_bias_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    bias_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Classic GEMM kernel with bias: C = A @ B + bias.
    
    Same as gemm_kernel but with an optional bias vector.
    If bias_stride == 0, bias is ignored.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k * BLOCK_K
        k_offsets = k_start + offs_k
        k_mask = k_offsets < K

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        
        b_ptrs = b_ptr + (k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # Add bias if present
    if bias_stride != 0:
        bias_vals = tl.load(bias_ptr + offs_n * bias_stride, mask=mask_n, other=0.0)
        out_tile = acc + bias_vals[None, :]
    else:
        out_tile = acc

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, out_tile, mask=mask_m[:, None] & mask_n[None, :])


__all__ = [
    "gemm_kernel",
    "gemm_bias_kernel",
]
