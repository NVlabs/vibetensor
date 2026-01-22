# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import torch  # type: ignore[import]
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

from vibe_kernels.common import alloc
from vibe_kernels.common.tensor_types import TensorLike
from vibe_kernels.gemm.kernel import triton_gemm

Tensor = TensorLike


# --- AdamW Kernels ---


@triton.jit
def _adamw_update_kernel(
    params_ptr,
    grads_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    step_size,
    beta1,
    one_minus_beta1,
    beta2,
    one_minus_beta2,
    inv_bias_correction2,
    eps,
    decay,
    maximize_sign,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    params = tl.load(params_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grads = tl.load(grads_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grads = grads * maximize_sign
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    exp_avg = beta1 * exp_avg + one_minus_beta1 * grads
    exp_avg_sq = beta2 * exp_avg_sq + one_minus_beta2 * grads * grads

    denom = tl.sqrt(exp_avg_sq * inv_bias_correction2) + eps
    param = params * (1.0 - decay) - step_size * (exp_avg / denom)

    tl.store(params_ptr + offsets, param, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)


def pick_block_size(n_elements: int) -> int:
    if n_elements >= 1 << 18:
        return 1024
    if n_elements >= 1 << 14:
        return 512
    return 256


def num_warps(block_size: int) -> int:
    if block_size >= 1024:
        return 8
    if block_size >= 512:
        return 4
    return 2


# --- Muon Kernels ---


@dataclass(frozen=True)
class MatmulTransposeConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


_CLASSIC_CONFIGS: List[MatmulTransposeConfig] = [
    MatmulTransposeConfig(128, 128, 32, num_warps=8, num_stages=2),
    MatmulTransposeConfig(64, 64, 32, num_warps=4, num_stages=3),
    MatmulTransposeConfig(128, 64, 32, num_warps=4, num_stages=4),
    MatmulTransposeConfig(64, 128, 32, num_warps=4, num_stages=4),
]
_TORCH_TO_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}
_TMA_MIN_DIM = 8192


def _supports_hopper_tma() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9 and hasattr(tl, "make_tensor_descriptor")


def _should_use_tma(matrix: Tensor) -> bool:
    return (
        matrix.dtype in (torch.float16, torch.bfloat16)
        and min(matrix.size(0), matrix.size(1)) >= _TMA_MIN_DIM
        and matrix.size(1) % 128 == 0
        and _supports_hopper_tma()
    )


def _select_config(
    configs: List[MatmulTransposeConfig], M: int, K: int
) -> MatmulTransposeConfig:
    best_cfg: Optional[MatmulTransposeConfig] = None
    best_score: Optional[tuple[int, int, int]] = None
    for cfg in configs:
        tiles_m = (M + cfg.block_m - 1) // cfg.block_m
        tiles_n = (M + cfg.block_n - 1) // cfg.block_n
        tile_count = tiles_m * tiles_n
        remainder_k = K % cfg.block_k
        coverage = cfg.block_m * cfg.block_n
        score = (tile_count, remainder_k, -coverage)
        if best_score is None or score < best_score:
            best_cfg = cfg
            best_score = score
    if best_cfg is None:
        raise RuntimeError("No matmul_transpose tiling configuration available")
    return best_cfg


@lru_cache(maxsize=None)
def _get_classic_kernel(cfg: MatmulTransposeConfig, out_dtype: torch.dtype):
    block_m = tl.constexpr(cfg.block_m)
    block_n = tl.constexpr(cfg.block_n)
    block_k = tl.constexpr(cfg.block_k)
    num_warps = cfg.num_warps
    num_stages = cfg.num_stages
    tl_dtype = _TORCH_TO_TL_DTYPE[out_dtype]

    @triton.jit
    def kernel(
        x_ptr,
        y_ptr,
        M,
        K,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        if pid_m > pid_n:
            return

        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        mask_m = offs_m < M
        mask_n = offs_n < M

        acc = tl.zeros((block_m, block_n), dtype=tl.float32)
        offs_k = tl.arange(0, block_k)
        for k in range(0, tl.cdiv(K, block_k)):
            k_start = k * block_k
            k_offsets = k_start + offs_k
            k_mask = k_offsets < K

            a_ptrs = (
                x_ptr + offs_m[:, None] * stride_xm + k_offsets[None, :] * stride_xn
            )
            b_ptrs = (
                x_ptr + offs_n[:, None] * stride_xm + k_offsets[None, :] * stride_xn
            )

            a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[:, None] & k_mask[None, :], other=0.0)
            acc += tl.dot(a, tl.trans(b))

        out_tile = acc.to(tl_dtype)
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, out_tile, mask=mask_m[:, None] & mask_n[None, :])

        if pid_m < pid_n:
            y_tp_ptrs = (
                y_ptr + offs_n[:, None] * stride_ym + offs_m[None, :] * stride_yn
            )
            tl.store(
                y_tp_ptrs, tl.trans(out_tile), mask=mask_n[:, None] & mask_m[None, :]
            )

    return kernel


def matmul_transpose_assign(d_in: Tensor, d_out: Tensor) -> None:
    if not d_in.is_cuda or not d_out.is_cuda:
        raise ValueError("matmul_transpose_assign expects CUDA tensors")
    if d_in.device != d_out.device:
        raise ValueError("Input tensors must be on the same device")
    if d_in.dtype != d_out.dtype:
        raise ValueError("Input tensors must share the same dtype")
    if d_in.ndim != 2 or d_out.ndim != 2:
        raise ValueError("matmul_transpose_assign expects 2D tensors")
    if d_out.shape[0] != d_out.shape[1] or d_out.shape[0] != d_in.shape[0]:
        raise ValueError(
            "Output must be square with leading dimension equal to input rows"
        )

    d_in = d_in.contiguous()
    d_out = d_out.contiguous()

    M, K = d_in.shape
    if _should_use_tma(d_in):
        xt = d_in.transpose(-1, -2).contiguous()
        triton_gemm(d_in, xt, out=d_out)
        return

    cfg = _select_config(_CLASSIC_CONFIGS, M, K)
    kernel = _get_classic_kernel(cfg, d_in.dtype)
    grid = (triton.cdiv(M, cfg.block_m), triton.cdiv(M, cfg.block_n))
    kernel[grid](
        d_in,
        d_out,
        M,
        K,
        d_in.stride(0),
        d_in.stride(1),
        d_out.stride(0),
        d_out.stride(1),
    )


def matmul_transpose(d_in: Tensor) -> Tensor:
    if d_in.ndim != 2:
        raise ValueError("matmul_transpose expects a 2D tensor")
    if d_in.device.type != "cuda":
        raise ValueError("matmul_transpose expects a CUDA tensor")
    M = d_in.size(0)
    d_out = alloc.empty((M, M), like=d_in, dtype=d_in.dtype)
    matmul_transpose_assign(d_in, d_out)
    return d_out


def fast_newton_schulz(G: Tensor, steps: int = 5) -> Tensor:
    if G.ndim < 2:
        raise ValueError("Muon Newton-Schulz expects ndims >= 2")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transpose_result = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transpose_result = True

    buf1 = alloc.empty((X.size(-2), X.size(-2)), like=X, dtype=X.dtype)
    buf2 = alloc.empty_like(buf1)

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        matmul_transpose_assign(X, buf1)
        matmul_transpose_assign(buf1, buf2)
        B = b * buf1 + c * buf2
        X = a * X + torch.matmul(B, X)

    if transpose_result:
        X = X.mT
    return X.to(G.dtype)
