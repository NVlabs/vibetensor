# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# pyright: reportMissingImports=false, reportOptionalMemberAccess=false

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast, Optional, Sequence, Tuple

try:  # pragma: no cover - handled by runtime checks
    import torch  # type: ignore[import]
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - handled by runtime checks
    import triton  # type: ignore[import]
    import triton.language as tl  # type: ignore[import]
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

from vibe_kernels.common import alloc
from vibe_kernels.common.tensor_types import TensorLike

TorchTensor = TensorLike


@dataclass(frozen=True)
class GEMMTiling:
    """Tile configuration used when launching the Triton GEMM kernel."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    group_m: int = 8
    instruction_shape: Optional[Tuple[int, int]] = None
    use_tma: bool = False
    warp_specialize: bool = False


def is_triton_available() -> bool:
    """Return True if Triton bindings are importable."""

    return triton is not None and tl is not None


def make_default_gemm_configs(*, use_wgmma: bool = True) -> Sequence[GEMMTiling]:
    """Return a list of tiling strategies tuned for Hopper and Ampere GPUs."""

    configs = [
        GEMMTiling(128, 128, 64, num_warps=8, num_stages=3),
        GEMMTiling(128, 64, 64, num_warps=4, num_stages=3),
        GEMMTiling(64, 128, 64, num_warps=4, num_stages=3),
        GEMMTiling(64, 64, 64, num_warps=4, num_stages=2),
    ]
    if use_wgmma:
        configs.extend(
            [
                GEMMTiling(
                    128, 128, 64, num_warps=4, num_stages=2, instruction_shape=(64, 64)
                ),
                GEMMTiling(
                    128, 256, 64, num_warps=8, num_stages=2, instruction_shape=(64, 128)
                ),
            ]
        )
    return configs


_TMA_ALLOCATOR_INITIALIZED = False
_TMA_MIN_DIM = 4096


def _supports_hopper_tma() -> bool:
    if torch is None or not is_triton_available():
        return False
    if not hasattr(torch, "cuda") or not torch.cuda.is_available():  # type: ignore[attr-defined]
        return False
    try:
        major, _ = torch.cuda.get_device_capability()  # type: ignore[attr-defined]
    except (RuntimeError, AttributeError):
        return False
    return major >= 9 and hasattr(tl, "make_tensor_descriptor")


def _should_use_tma(dtype: Any) -> bool:
    if torch is None:
        return False
    return dtype in (torch.float16, torch.bfloat16) and _supports_hopper_tma()


def _ensure_tma_allocator() -> None:
    global _TMA_ALLOCATOR_INITIALIZED
    if not _TMA_ALLOCATOR_INITIALIZED:

        def _allocator(size: int, alignment: int, stream: Optional[int]) -> Any:
            # Triton expects a CUDA int8 buffer.
            if torch is None:
                raise RuntimeError("PyTorch is required for Triton TMA allocator wrapper")
            dev = torch.device(f"cuda:{torch.cuda.current_device()}")  # type: ignore[attr-defined]
            return torch.empty((int(size),), device=dev, dtype=torch.int8)

        triton.set_allocator(_allocator)  # type: ignore[attr-defined]
        _TMA_ALLOCATOR_INITIALIZED = True


if is_triton_available():  # pragma: no cover - depends on optional runtime
    assert tl is not None and triton is not None
    tl = cast(Any, tl)
    triton = cast(Any, triton)
    _TORCH_TO_TL_DTYPE = {
        torch.float16: tl.float16,  # type: ignore[attr-defined]
        torch.bfloat16: tl.bfloat16,  # type: ignore[attr-defined]
        torch.float32: tl.float32,  # type: ignore[attr-defined]
    }

    def make_hopper_tma_configs() -> Sequence[GEMMTiling]:
        configs: list[GEMMTiling] = []
        for block_n in (128, 256):
            for block_k in (64, 128 if block_n == 128 else 64):
                for warp_specialize in (True, False):
                    num_warps = 4 if block_n == 128 else 8
                    num_stages = 3 if block_k == 64 else 2
                    configs.append(
                        GEMMTiling(
                            128,
                            block_n,
                            block_k,
                            num_warps=num_warps,
                            num_stages=num_stages,
                            group_m=8,
                            use_tma=True,
                            warp_specialize=warp_specialize,
                        )
                    )
        return configs

    @lru_cache(maxsize=None)
    def _get_tma_kernel(cfg: GEMMTiling, out_dtype: Any) -> Any:
        block_m = int(cfg.block_m)
        block_n = int(cfg.block_n)
        block_k = int(cfg.block_k)
        group_size_m = int(cfg.group_m)
        warp_specialize = bool(cfg.warp_specialize)

        block_m_const = tl.constexpr(block_m)  # type: ignore[attr-defined]
        block_n_const = tl.constexpr(block_n)  # type: ignore[attr-defined]
        block_k_const = tl.constexpr(block_k)  # type: ignore[attr-defined]
        group_size_m_const = tl.constexpr(group_size_m)  # type: ignore[attr-defined]
        warp_specialize_const = tl.constexpr(int(warp_specialize))  # type: ignore[attr-defined]
        out_dtype_const = tl.constexpr(out_dtype)  # type: ignore[attr-defined]

        @triton.jit  # type: ignore[attr-defined]
        def _kernel(
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
        ):
            pid = tl.program_id(axis=0)  # type: ignore[attr-defined]
            num_pid_m = tl.cdiv(M, block_m_const)  # type: ignore[attr-defined]
            num_pid_n = tl.cdiv(N, block_n_const)  # type: ignore[attr-defined]
            num_pid_in_group = group_size_m_const * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * group_size_m_const
            group_size_m_eff = min(num_pid_m - first_pid_m, group_size_m_const)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m_eff)
            pid_n = (pid % num_pid_in_group) // group_size_m_eff

            offs_cm = pid_m * block_m_const
            offs_cn = pid_n * block_n_const
            offs_bn = offs_cn

            k_tiles = tl.cdiv(K, block_k_const)  # type: ignore[attr-defined]

            a_desc = tl.make_tensor_descriptor(  # type: ignore[attr-defined]
                a_ptr,
                shape=[M, K],
                strides=[stride_am, stride_ak],
                block_shape=[block_m_const, block_k_const],
            )
            b_desc = tl.make_tensor_descriptor(  # type: ignore[attr-defined]
                b_ptr,
                shape=[K, N],
                strides=[stride_bk, stride_bn],
                block_shape=[block_k_const, block_n_const],
            )
            c_desc = tl.make_tensor_descriptor(  # type: ignore[attr-defined]
                c_ptr,
                shape=[M, N],
                strides=[stride_cm, stride_cn],
                block_shape=[block_m_const, block_n_const],
            )

            accumulator = tl.zeros((block_m_const, block_n_const), dtype=tl.float32)  # type: ignore[attr-defined]

            for k in tl.range(k_tiles, warp_specialize=warp_specialize_const):  # type: ignore[attr-defined]
                offs_k = k * block_k_const
                a_tile = a_desc.load([offs_cm, offs_k])  # type: ignore[attr-defined]
                b_tile = b_desc.load([offs_k, offs_bn])  # type: ignore[attr-defined]
                accumulator = tl.dot(  # type: ignore[attr-defined]
                    a_tile,
                    b_tile,
                    accumulator,
                    out_dtype=tl.float32,  # type: ignore[attr-defined]
                    allow_tf32=False,
                )

            if bias_stride != 0:
                bias_offsets = offs_cn + tl.arange(0, block_n_const)  # type: ignore[attr-defined]
                bias_mask = bias_offsets < N
                bias_vals = tl.load(  # type: ignore[attr-defined]
                    bias_ptr + bias_offsets * bias_stride,
                    mask=bias_mask,
                    other=0.0,
                )
                out_tile = accumulator.to(out_dtype_const) + bias_vals[None, :].to(
                    out_dtype_const
                )
            else:
                out_tile = accumulator.to(out_dtype_const)

            c_desc.store([offs_cm, offs_cn], out_tile)  # type: ignore[attr-defined]

        return cast(Any, _kernel)

    _gemm_kernel_tma = cast(Any, _get_tma_kernel)

    @lru_cache(maxsize=None)
    def _get_tma_dgrad_kernel(cfg: GEMMTiling, out_dtype: Any) -> Any:
        block_m = int(cfg.block_m)
        block_n = int(cfg.block_n)
        block_k = int(cfg.block_k)
        group_size_m = int(cfg.group_m)
        warp_specialize = int(bool(cfg.warp_specialize))

        block_m_const = tl.constexpr(block_m)  # type: ignore[attr-defined]
        block_n_const = tl.constexpr(block_n)  # type: ignore[attr-defined]
        block_k_const = tl.constexpr(block_k)  # type: ignore[attr-defined]
        group_size_m_const = tl.constexpr(group_size_m)  # type: ignore[attr-defined]
        warp_specialize_const = tl.constexpr(warp_specialize)  # type: ignore[attr-defined]
        out_dtype_const = tl.constexpr(out_dtype)  # type: ignore[attr-defined]

        @triton.jit  # type: ignore[attr-defined]
        def _kernel(
            grad_out_ptr,
            weight_ptr,
            grad_input_ptr,
            M,
            N,
            K,
            stride_go_m,
            stride_go_n,
            stride_wk,
            stride_wn,
            stride_gi_m,
            stride_gi_k,
        ):
            pid = tl.program_id(axis=0)  # type: ignore[attr-defined]
            num_pid_m = tl.cdiv(M, block_m_const)  # type: ignore[attr-defined]
            num_pid_k = tl.cdiv(K, block_n_const)  # type: ignore[attr-defined]
            num_pid_in_group = group_size_m_const * num_pid_k
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * group_size_m_const
            group_size_m_eff = min(num_pid_m - first_pid_m, group_size_m_const)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m_eff)
            pid_k = (pid % num_pid_in_group) // group_size_m_eff

            offs_m = pid_m * block_m_const
            offs_k = pid_k * block_n_const

            k_tiles = tl.cdiv(N, block_k_const)  # type: ignore[attr-defined]

            grad_out_desc = tl.make_tensor_descriptor(  # type: ignore[attr-defined]
                grad_out_ptr,
                shape=[M, N],
                strides=[stride_go_m, stride_go_n],
                block_shape=[block_m_const, block_k_const],
            )
            weight_desc = tl.make_tensor_descriptor(  # type: ignore[attr-defined]
                weight_ptr,
                shape=[K, N],
                strides=[stride_wk, stride_wn],
                block_shape=[block_n_const, block_k_const],
            )
            grad_input_desc = tl.make_tensor_descriptor(  # type: ignore[attr-defined]
                grad_input_ptr,
                shape=[M, K],
                strides=[stride_gi_m, stride_gi_k],
                block_shape=[block_m_const, block_n_const],
            )

            accumulator = tl.zeros((block_m_const, block_n_const), dtype=tl.float32)  # type: ignore[attr-defined]

            for nk in tl.range(k_tiles, warp_specialize=warp_specialize_const):  # type: ignore[attr-defined]
                offs_n = nk * block_k_const
                go_tile = grad_out_desc.load((offs_m, offs_n))  # type: ignore[attr-defined]
                w_tile = weight_desc.load((offs_k, offs_n))  # type: ignore[attr-defined]
                accumulator = tl.dot(  # type: ignore[attr-defined]
                    go_tile,
                    tl.trans(w_tile),
                    accumulator,
                    out_dtype=tl.float32,
                    allow_tf32=False,
                )

            grad_input_desc.store((offs_m, offs_k), accumulator.to(out_dtype_const))  # type: ignore[attr-defined]

        return cast(Any, _kernel)

    _dgrad_kernel_tma = cast(Any, _get_tma_dgrad_kernel)

    @lru_cache(maxsize=None)
    def _get_tma_wgrad_kernel(cfg: GEMMTiling, out_dtype: Any) -> Any:
        block_m = int(cfg.block_m)
        block_n = int(cfg.block_n)
        block_k = int(cfg.block_k)
        group_size_m = int(cfg.group_m)
        warp_specialize = int(bool(cfg.warp_specialize))

        block_m_const = tl.constexpr(block_m)  # type: ignore[attr-defined]
        block_n_const = tl.constexpr(block_n)  # type: ignore[attr-defined]
        block_k_const = tl.constexpr(block_k)  # type: ignore[attr-defined]
        group_size_m_const = tl.constexpr(group_size_m)  # type: ignore[attr-defined]
        warp_specialize_const = tl.constexpr(warp_specialize)  # type: ignore[attr-defined]
        out_dtype_const = tl.constexpr(out_dtype)  # type: ignore[attr-defined]

        @triton.jit  # type: ignore[attr-defined]
        def _kernel(
            a_ptr,
            grad_out_ptr,
            grad_weight_ptr,
            M,
            K,
            N,
            stride_am,
            stride_ak,
            stride_go_m,
            stride_go_n,
            stride_gw_k,
            stride_gw_n,
        ):
            pid = tl.program_id(axis=0)  # type: ignore[attr-defined]
            num_pid_k = tl.cdiv(K, block_m_const)  # type: ignore[attr-defined]
            num_pid_n = tl.cdiv(N, block_n_const)  # type: ignore[attr-defined]
            num_pid_in_group = group_size_m_const * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_k = group_id * group_size_m_const
            group_size_k_eff = min(num_pid_k - first_pid_k, group_size_m_const)
            pid_k = first_pid_k + ((pid % num_pid_in_group) % group_size_k_eff)
            pid_n = (pid % num_pid_in_group) // group_size_k_eff

            offs_k = pid_k * block_m_const
            offs_n = pid_n * block_n_const

            m_tiles = tl.cdiv(M, block_k_const)  # type: ignore[attr-defined]

            a_desc = tl.make_tensor_descriptor(  # type: ignore[attr-defined]
                a_ptr,
                shape=[M, K],
                strides=[stride_am, stride_ak],
                block_shape=[block_k_const, block_m_const],
            )
            grad_out_desc = tl.make_tensor_descriptor(  # type: ignore[attr-defined]
                grad_out_ptr,
                shape=[M, N],
                strides=[stride_go_m, stride_go_n],
                block_shape=[block_k_const, block_n_const],
            )
            grad_weight_desc = tl.make_tensor_descriptor(  # type: ignore[attr-defined]
                grad_weight_ptr,
                shape=[K, N],
                strides=[stride_gw_k, stride_gw_n],
                block_shape=[block_m_const, block_n_const],
            )

            accumulator = tl.zeros((block_m_const, block_n_const), dtype=tl.float32)  # type: ignore[attr-defined]

            for mt in tl.range(m_tiles, warp_specialize=warp_specialize_const):  # type: ignore[attr-defined]
                offs_m = mt * block_k_const
                a_tile = a_desc.load((offs_m, offs_k))  # type: ignore[attr-defined]
                go_tile = grad_out_desc.load((offs_m, offs_n))  # type: ignore[attr-defined]
                accumulator = tl.dot(  # type: ignore[attr-defined]
                    tl.trans(a_tile),
                    go_tile,
                    accumulator,
                    out_dtype=tl.float32,
                    allow_tf32=False,
                )

            grad_weight_desc.store((offs_k, offs_n), accumulator.to(out_dtype_const))  # type: ignore[attr-defined]

        return cast(Any, _kernel)

    _wgrad_kernel_tma = cast(Any, _get_tma_wgrad_kernel)

    @lru_cache(maxsize=None)
    def _get_classic_kernel(cfg: GEMMTiling, out_dtype: Any, allow_tf32: bool) -> Any:
        block_m = int(cfg.block_m)
        block_n = int(cfg.block_n)
        block_k = int(cfg.block_k)

        block_m_const = tl.constexpr(block_m)  # type: ignore[attr-defined]
        block_n_const = tl.constexpr(block_n)  # type: ignore[attr-defined]
        block_k_const = tl.constexpr(block_k)  # type: ignore[attr-defined]
        out_dtype_const = tl.constexpr(out_dtype)  # type: ignore[attr-defined]
        allow_tf32_const = tl.constexpr(int(allow_tf32))  # type: ignore[attr-defined]

        @triton.jit  # type: ignore[attr-defined]
        def _kernel(
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
        ):
            pid = tl.program_id(axis=0)  # type: ignore[attr-defined]
            num_pid_m = tl.cdiv(M, block_m_const)  # type: ignore[attr-defined]
            num_pid_n = tl.cdiv(N, block_n_const)  # type: ignore[attr-defined]
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            offs_m = pid_m * block_m_const + tl.arange(0, block_m_const)  # type: ignore[attr-defined]
            offs_n = pid_n * block_n_const + tl.arange(0, block_n_const)  # type: ignore[attr-defined]
            mask_m = offs_m < M
            mask_n = offs_n < N

            acc = tl.zeros((block_m_const, block_n_const), dtype=tl.float32)  # type: ignore[attr-defined]
            offs_k = tl.arange(0, block_k_const)  # type: ignore[attr-defined]
            for k in range(0, tl.cdiv(K, block_k_const)):  # type: ignore[attr-defined]
                k_start = k * block_k_const
                k_offsets = k_start + offs_k
                k_mask = k_offsets < K

                a_ptrs = a_ptr + (
                    offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
                )
                b_ptrs = b_ptr + (
                    k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn
                )

                a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)  # type: ignore[attr-defined]
                b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)  # type: ignore[attr-defined]
                acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=allow_tf32_const)  # type: ignore[attr-defined]

            if bias_stride != 0:
                bias_vals = tl.load(bias_ptr + offs_n * bias_stride, mask=mask_n, other=0.0)  # type: ignore[attr-defined]
                out_tile = acc.to(out_dtype_const) + bias_vals[None, :].to(
                    out_dtype_const
                )
            else:
                out_tile = acc.to(out_dtype_const)

            c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
            tl.store(c_ptrs, out_tile, mask=mask_m[:, None] & mask_n[None, :])  # type: ignore[attr-defined]

        return cast(Any, _kernel)

    @lru_cache(maxsize=None)
    def _get_dgrad_kernel(cfg: GEMMTiling, out_dtype: Any, allow_tf32: bool) -> Any:
        block_m = int(cfg.block_m)
        block_n = int(cfg.block_n)
        block_k = int(cfg.block_k)

        block_m_const = tl.constexpr(block_m)  # type: ignore[attr-defined]
        block_n_const = tl.constexpr(block_n)  # type: ignore[attr-defined]
        block_k_const = tl.constexpr(block_k)  # type: ignore[attr-defined]
        out_dtype_const = tl.constexpr(out_dtype)  # type: ignore[attr-defined]
        allow_tf32_const = tl.constexpr(int(allow_tf32))  # type: ignore[attr-defined]

        @triton.jit  # type: ignore[attr-defined]
        def _kernel(
            grad_out_ptr,
            weight_ptr,
            grad_input_ptr,
            M,
            N,
            K,
            stride_go_m,
            stride_go_n,
            stride_w_n,
            stride_w_k,
            stride_gi_m,
            stride_gi_k,
        ):
            pid = tl.program_id(axis=0)  # type: ignore[attr-defined]
            num_pid_m = tl.cdiv(M, block_m_const)  # type: ignore[attr-defined]
            num_pid_k = tl.cdiv(K, block_n_const)  # type: ignore[attr-defined]
            pid_m = pid // num_pid_k
            pid_k = pid % num_pid_k

            offs_m = pid_m * block_m_const + tl.arange(0, block_m_const)  # type: ignore[attr-defined]
            offs_k = pid_k * block_n_const + tl.arange(0, block_n_const)  # type: ignore[attr-defined]
            mask_m = offs_m < M
            mask_k = offs_k < K

            acc = tl.zeros((block_m_const, block_n_const), dtype=tl.float32)  # type: ignore[attr-defined]
            offs_n = tl.arange(0, block_k_const)  # type: ignore[attr-defined]
            num_tiles_n = tl.cdiv(N, block_k_const)  # type: ignore[attr-defined]

            for n in range(0, num_tiles_n):
                n_start = n * block_k_const
                n_offsets = n_start + offs_n
                mask_n = n_offsets < N

                go_ptrs = grad_out_ptr + (
                    offs_m[:, None] * stride_go_m + n_offsets[None, :] * stride_go_n
                )
                w_ptrs = weight_ptr + (
                    n_offsets[:, None] * stride_w_n + offs_k[None, :] * stride_w_k
                )

                go = tl.load(
                    go_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
                )  # type: ignore[attr-defined]
                w = tl.load(
                    w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0
                )  # type: ignore[attr-defined]
                acc += tl.dot(
                    go, w, out_dtype=tl.float32, allow_tf32=allow_tf32_const
                )  # type: ignore[attr-defined]

            out_tile = acc.to(out_dtype_const)
            out_ptrs = grad_input_ptr + (
                offs_m[:, None] * stride_gi_m + offs_k[None, :] * stride_gi_k
            )
            tl.store(
                out_ptrs, out_tile, mask=mask_m[:, None] & mask_k[None, :]
            )  # type: ignore[attr-defined]

        return cast(Any, _kernel)

    @lru_cache(maxsize=None)
    def _get_wgrad_kernel(cfg: GEMMTiling, out_dtype: Any, allow_tf32: bool) -> Any:
        block_m = int(cfg.block_m)
        block_n = int(cfg.block_n)
        block_k = int(cfg.block_k)

        block_m_const = tl.constexpr(block_m)  # type: ignore[attr-defined]
        block_n_const = tl.constexpr(block_n)  # type: ignore[attr-defined]
        block_k_const = tl.constexpr(block_k)  # type: ignore[attr-defined]
        out_dtype_const = tl.constexpr(out_dtype)  # type: ignore[attr-defined]
        allow_tf32_const = tl.constexpr(int(allow_tf32))  # type: ignore[attr-defined]

        @triton.jit  # type: ignore[attr-defined]
        def _kernel(
            a_ptr,
            grad_out_ptr,
            grad_weight_ptr,
            M,
            K,
            N,
            stride_am,
            stride_ak,
            stride_go_m,
            stride_go_n,
            stride_gw_k,
            stride_gw_n,
        ):
            pid = tl.program_id(axis=0)  # type: ignore[attr-defined]
            num_pid_k = tl.cdiv(K, block_m_const)  # type: ignore[attr-defined]
            num_pid_n = tl.cdiv(N, block_n_const)  # type: ignore[attr-defined]
            pid_k = pid // num_pid_n
            pid_n = pid % num_pid_n

            offs_k = pid_k * block_m_const + tl.arange(0, block_m_const)  # type: ignore[attr-defined]
            offs_n = pid_n * block_n_const + tl.arange(0, block_n_const)  # type: ignore[attr-defined]
            mask_k = offs_k < K
            mask_n = offs_n < N

            acc = tl.zeros((block_m_const, block_n_const), dtype=tl.float32)  # type: ignore[attr-defined]
            offs_m = tl.arange(0, block_k_const)  # type: ignore[attr-defined]
            num_tiles_m = tl.cdiv(M, block_k_const)  # type: ignore[attr-defined]

            for m in range(0, num_tiles_m):
                m_start = m * block_k_const
                m_offsets = m_start + offs_m
                mask_m = m_offsets < M

                a_ptrs = a_ptr + (
                    m_offsets[None, :] * stride_am + offs_k[:, None] * stride_ak
                )
                go_ptrs = grad_out_ptr + (
                    m_offsets[:, None] * stride_go_m + offs_n[None, :] * stride_go_n
                )

                a_tile = tl.load(
                    a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0
                )  # type: ignore[attr-defined]
                go_tile = tl.load(
                    go_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
                )  # type: ignore[attr-defined]
                acc += tl.dot(
                    a_tile, go_tile, out_dtype=tl.float32, allow_tf32=allow_tf32_const
                )  # type: ignore[attr-defined]

            out_tile = acc.to(out_dtype_const)
            out_ptrs = grad_weight_ptr + (
                offs_k[:, None] * stride_gw_k + offs_n[None, :] * stride_gw_n
            )
            tl.store(
                out_ptrs, out_tile, mask=mask_k[:, None] & mask_n[None, :]
            )  # type: ignore[attr-defined]

        return cast(Any, _kernel)

else:  # pragma: no cover
    _TORCH_TO_TL_DTYPE = {}
    _gemm_kernel_tma = cast(Any, None)
    _dgrad_kernel_tma = cast(Any, None)
    _wgrad_kernel_tma = cast(Any, None)
    _get_classic_kernel = cast(Any, None)
    _get_dgrad_kernel = cast(Any, None)
    _get_wgrad_kernel = cast(Any, None)


def _select_tiling(configs: Sequence[GEMMTiling], M: int, N: int, K: int) -> GEMMTiling:
    """Pick a tile configuration heuristically optimal for the given problem size."""

    best_cfg: Optional[GEMMTiling] = None
    best_score: Optional[Tuple[int, int, int]] = None
    for cfg in configs:
        tiles_m = (M + cfg.block_m - 1) // cfg.block_m
        tiles_n = (N + cfg.block_n - 1) // cfg.block_n
        tile_count = tiles_m * tiles_n
        remainder_k = K % cfg.block_k
        coverage = cfg.block_m * cfg.block_n
        score = (tile_count, remainder_k, -coverage)
        if best_score is None or score < best_score:
            best_cfg = cfg
            best_score = score
    if best_cfg is None:  # pragma: no cover - defensive
        raise RuntimeError("No GEMM tiling configuration available")
    return best_cfg


def _normalize_tensor(t: TorchTensor) -> TorchTensor:
    if torch is None:  # pragma: no cover - sanity guard
        raise RuntimeError("PyTorch is required for Triton GEMM support")
    if not t.is_cuda:
        raise ValueError("Triton GEMM expects CUDA tensors")
    if t.dim() != 2:
        raise ValueError("Triton GEMM only supports 2D matrices")
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def triton_gemm(
    a: TorchTensor,
    b: TorchTensor,
    *,
    bias: Optional[TorchTensor] = None,
    out: Optional[TorchTensor] = None,
    configs: Optional[Sequence[GEMMTiling]] = None,
    allow_tf32: bool = False,
) -> TorchTensor:
    """Compute a matrix multiplication using Triton when available."""

    if torch is None:
        raise RuntimeError("PyTorch is required for triton_gemm")
    if not is_triton_available():
        raise RuntimeError("Triton is required for triton_gemm but is not installed")

    a = _normalize_tensor(a)
    b = _normalize_tensor(b)
    if a.dtype != b.dtype:
        raise TypeError("Input dtypes must match")

    if a.size(1) != b.size(0):
        raise ValueError("Inner dimensions must align for GEMM")

    M, K = a.shape
    _, N = b.shape

    if out is None:
        result = alloc.empty((M, N), like=a, dtype=a.dtype)
    else:
        if out.shape != (M, N):
            raise ValueError("Provided output tensor has wrong shape")
        if out.dtype != a.dtype:
            raise TypeError("Output tensor dtype must match inputs")
        if not out.is_contiguous():
            raise ValueError("Output tensor must be contiguous")
        result = out

    if bias is not None:
        if bias.device != a.device:
            raise ValueError("Bias tensor must be on the same device as inputs")
        if bias.dim() != 1 or bias.shape[0] != N:
            raise ValueError("Bias must be a 1D vector matching the output columns")
        if not bias.is_contiguous():
            bias = bias.contiguous()

    if M == 0 or N == 0:
        result.zero_()
        return result
    if K == 0:
        result.zero_()
        if bias is not None:
            result += bias.view(1, -1)
        return result

    tl_dtype = _TORCH_TO_TL_DTYPE.get(result.dtype) if is_triton_available() else None
    if tl_dtype is None:
        raise TypeError(f"Unsupported dtype for Triton GEMM: {result.dtype}")

    allow_tf32 = bool(allow_tf32 and result.dtype == torch.float32)

    stride_am, stride_ak = [int(s) for s in a.stride()]
    stride_bk, stride_bn = [int(s) for s in b.stride()]
    stride_cm, stride_cn = [int(s) for s in result.stride()]
    bias_stride = int(bias.stride(0)) if bias is not None else 0

    if (
        _should_use_tma(result.dtype)
        and _gemm_kernel_tma is not None
        and min(M, N) >= _TMA_MIN_DIM
        and K >= _TMA_MIN_DIM
    ):
        _ensure_tma_allocator()
        bias_ptr = bias if bias is not None else result
        tma_configs = tuple(make_hopper_tma_configs())
        selected_tma = _select_tiling(tma_configs, M, N, K)
        grid = (
            triton.cdiv(M, selected_tma.block_m) * triton.cdiv(N, selected_tma.block_n),  # type: ignore[attr-defined]
        )
        kernel_tma = _get_tma_kernel(selected_tma, tl_dtype)
        kernel_tma[grid](
            a,
            b,
            result,
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
            num_warps=int(selected_tma.num_warps),
            num_stages=int(selected_tma.num_stages),
        )
        return result

    configs = tuple(configs or make_default_gemm_configs())
    selected_cfg = _select_tiling(configs, M, N, K)

    grid = (
        triton.cdiv(M, selected_cfg.block_m) * triton.cdiv(N, selected_cfg.block_n),  # type: ignore[attr-defined]
    )
    bias_ptr = bias if bias is not None else result

    kernel = _get_classic_kernel(selected_cfg, tl_dtype, allow_tf32)
    kernel[grid](
        a,
        b,
        result,
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
        num_warps=int(selected_cfg.num_warps),
        num_stages=int(selected_cfg.num_stages),
    )

    return result


def triton_gemm_backward(
    grad_output: TorchTensor,
    a: TorchTensor,
    b: TorchTensor,
    *,
    compute_grad_a: bool = True,
    compute_grad_b: bool = True,
    compute_grad_bias: bool = False,
    configs: Optional[Sequence[GEMMTiling]] = None,
    allow_tf32: bool = False,
) -> Tuple[Optional[TorchTensor], Optional[TorchTensor], Optional[TorchTensor]]:
    """Compute gradients for C = a @ b + bias using Triton kernels."""

    if torch is None:
        raise RuntimeError("PyTorch is required for triton_gemm_backward")
    if not is_triton_available():
        raise RuntimeError(
            "Triton is required for triton_gemm_backward but is not installed"
        )

    grad_output = _normalize_tensor(grad_output)
    a = _normalize_tensor(a)
    b = _normalize_tensor(b)

    if grad_output.dtype != a.dtype or grad_output.dtype != b.dtype:
        raise TypeError("All tensors must share the same dtype")

    M, N = grad_output.shape
    if a.shape[0] != M or b.shape[1] != N:
        raise ValueError("Shape mismatch between grad_output and inputs")
    K = a.shape[1]
    if b.shape[0] != K:
        raise ValueError("Input matrix shapes are incompatible")

    configs_tuple = tuple(configs or make_default_gemm_configs())
    allow_tf32_flag = bool(allow_tf32 and grad_output.dtype == torch.float32)
    tma_configs: Optional[Sequence[GEMMTiling]] = None

    grad_a: Optional[TorchTensor] = alloc.zeros_like(a) if compute_grad_a else None
    grad_b: Optional[TorchTensor] = alloc.zeros_like(b) if compute_grad_b else None
    grad_bias: Optional[TorchTensor] = None

    if compute_grad_a and K > 0 and N > 0:
        tl_dtype = _TORCH_TO_TL_DTYPE.get(a.dtype) if is_triton_available() else None
        if tl_dtype is None:
            raise TypeError(f"Unsupported dtype for Triton GEMM backward: {a.dtype}")
        assert grad_a is not None  # for type checkers
        stride_go_m, stride_go_n = [int(s) for s in grad_output.stride()]
        stride_wk, stride_wn = [int(s) for s in b.stride()]
        stride_gi_m, stride_gi_k = [int(s) for s in grad_a.stride()]
        used_tma = False
        if (
            _dgrad_kernel_tma is not None
            and _should_use_tma(a.dtype)
            and min(M, K) >= _TMA_MIN_DIM
            and N >= _TMA_MIN_DIM
        ):
            tma_configs = tma_configs or tuple(make_hopper_tma_configs())
            cfg_tma = _select_tiling(tma_configs, M, K, N)
            grid = (
                triton.cdiv(M, cfg_tma.block_m) * triton.cdiv(K, cfg_tma.block_n),  # type: ignore[attr-defined]
            )
            _ensure_tma_allocator()
            kernel_tma = _get_tma_dgrad_kernel(cfg_tma, tl_dtype)
            kernel_tma[grid](
                grad_output,
                b,
                grad_a,
                M,
                N,
                K,
                stride_go_m,
                stride_go_n,
                stride_wk,
                stride_wn,
                stride_gi_m,
                stride_gi_k,
                num_warps=int(cfg_tma.num_warps),
                num_stages=int(cfg_tma.num_stages),
            )
            used_tma = True
        if not used_tma:
            cfg = _select_tiling(configs_tuple, M, K, N)
            grid = (
                triton.cdiv(M, cfg.block_m) * triton.cdiv(K, cfg.block_n),  # type: ignore[attr-defined]
            )
            kernel = _get_dgrad_kernel(cfg, tl_dtype, allow_tf32_flag)
            kernel[grid](
                grad_output,
                b,
                grad_a,
                M,
                N,
                K,
                stride_go_m,
                stride_go_n,
                stride_wn,
                stride_wk,
                stride_gi_m,
                stride_gi_k,
                num_warps=int(cfg.num_warps),
                num_stages=int(cfg.num_stages),
            )

    if compute_grad_b and M > 0 and K > 0:
        tl_dtype = _TORCH_TO_TL_DTYPE.get(b.dtype) if is_triton_available() else None
        if tl_dtype is None:
            raise TypeError(f"Unsupported dtype for Triton GEMM backward: {b.dtype}")
        assert grad_b is not None  # for type checkers
        stride_am, stride_ak = [int(s) for s in a.stride()]
        stride_go_m, stride_go_n = [int(s) for s in grad_output.stride()]
        stride_gw_k, stride_gw_n = [int(s) for s in grad_b.stride()]
        used_tma = False
        if (
            _wgrad_kernel_tma is not None
            and _should_use_tma(b.dtype)
            and min(K, N) >= _TMA_MIN_DIM
            and M >= _TMA_MIN_DIM
        ):
            tma_configs = tma_configs or tuple(make_hopper_tma_configs())
            cfg_tma = _select_tiling(tma_configs, K, N, M)
            grid = (
                triton.cdiv(K, cfg_tma.block_m) * triton.cdiv(N, cfg_tma.block_n),  # type: ignore[attr-defined]
            )
            _ensure_tma_allocator()
            kernel_tma = _get_tma_wgrad_kernel(cfg_tma, tl_dtype)
            kernel_tma[grid](
                a,
                grad_output,
                grad_b,
                M,
                K,
                N,
                stride_am,
                stride_ak,
                stride_go_m,
                stride_go_n,
                stride_gw_k,
                stride_gw_n,
                num_warps=int(cfg_tma.num_warps),
                num_stages=int(cfg_tma.num_stages),
            )
            used_tma = True
        if not used_tma:
            cfg = _select_tiling(configs_tuple, K, N, M)
            grid = (
                triton.cdiv(K, cfg.block_m) * triton.cdiv(N, cfg.block_n),  # type: ignore[attr-defined]
            )
            kernel = _get_wgrad_kernel(cfg, tl_dtype, allow_tf32_flag)
            kernel[grid](
                a,
                grad_output,
                grad_b,
                M,
                K,
                N,
                stride_am,
                stride_ak,
                stride_go_m,
                stride_go_n,
                stride_gw_k,
                stride_gw_n,
                num_warps=int(cfg.num_warps),
                num_stages=int(cfg.num_stages),
            )

    if compute_grad_bias:
        grad_bias = grad_output.sum(dim=0)

    return grad_a, grad_b, grad_bias
