# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# pyright: reportMissingImports=false, reportOptionalMemberAccess=false, reportGeneralTypeIssues=false

from functools import partial
from typing import Any, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from vibe_kernels.common import alloc
from vibe_kernels.common.tensor_types import TensorLike
from vibe_kernels.gemm.impl.cutedsl_gemm import copy_utils, utils
from vibe_kernels.rmsnorm.impl.cutedsl_rmsnorm.reduce import row_reduce
from vibe_kernels.rmsnorm.impl.cutedsl_rmsnorm.reduction_base import ReductionBase


_DTYPE_TO_CUTLASS: dict[str, cutlass.Numeric] = {
    "float16": cutlass.Float16,
    "torch.float16": cutlass.Float16,
    "bfloat16": cutlass.BFloat16,
    "torch.bfloat16": cutlass.BFloat16,
    "float32": cutlass.Float32,
    "torch.float32": cutlass.Float32,
}


def _dtype_key(x: Any) -> str:
    return str(getattr(x, "dtype", x))


def _is_cuda_tensor_like(x: Any) -> bool:
    is_cuda = getattr(x, "is_cuda", None)
    if isinstance(is_cuda, bool):
        return bool(is_cuda)
    dev = getattr(x, "device", None)
    return isinstance(dev, (tuple, list)) and len(dev) >= 2 and int(dev[0]) in (2, 13)


def _device_index(x: Any) -> int:
    dev = getattr(x, "device", None)
    if isinstance(dev, (tuple, list)) and len(dev) >= 2:
        return int(dev[1])
    idx = getattr(dev, "index", None)
    if idx is None:
        return 0
    return int(idx)


def _cute_numeric_for(x: Any) -> cutlass.Numeric:
    key = _dtype_key(x)
    out = _DTYPE_TO_CUTLASS.get(key)
    if out is None:
        raise TypeError(f"unsupported dtype for CuTeDSL LayerNorm: {key}")
    return out


def _cu_stream_handle(stream: int | None) -> cuda.CUstream:
    return cuda.CUstream(int(stream or 0))


class LayerNorm(ReductionBase):
    def __init__(self, dtype: cutlass.Numeric, N: int):
        super().__init__(dtype, N, stage=2)  # 2 stages for mean and var
        self.reload_from = None if N <= 16384 else "smem"
        self.delay_w_load = False

    def _calculate_threads_per_row(self):
        N = self.N
        if N <= 64:
            return 8
        elif N <= 128:
            return 16
        elif N <= 3072:
            return 32
        elif N <= 6144:
            return 64
        elif N <= 16384:
            return 128
        else:
            return 256

    def _set_cluster_n(self):
        N = self.N
        if cutlass.const_expr(self.dtype.width == 16):
            cluster_n = (
                1
                if N <= 16 * 1024
                else (
                    2
                    if N <= 32 * 1024
                    else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16))
                )
            )
        else:  # fp32
            cluster_n = (
                1
                if N <= 32 * 1024
                else (
                    2
                    if N <= 64 * 1024
                    else (4 if N <= 128 * 1024 else (8 if N <= 256 * 1024 else 16))
                )
            )
        self.cluster_n = cluster_n

    def _smem_size_in_bytes(self, tiler_mn, num_warps):
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn))
            + self.stage
            * num_warps
            * self.cluster_n
            * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mB: Optional[cute.Tensor],
        mO: cute.Tensor,
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        stream: cuda.CUstream,
        eps: cutlass.Float32 = 1e-6,
    ):
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype
        self._set_cluster_n()
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW = cute.make_tensor(mW.iterator, mW_expanded_layout)
        if cutlass.const_expr(mB is not None):
            mB_expanded_layout = cute.prepend(
                mB.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mB = cute.make_tensor(mB.iterator, mB_expanded_layout)
        if cutlass.const_expr(mRstd is not None):
            mRstd_expanded_layout = cute.append(
                mRstd.layout, cute.make_layout((self.N,), stride=(0,))
            )
            mRstd = cute.make_tensor(mRstd.iterator, mRstd_expanded_layout)
        if cutlass.const_expr(mMean is not None):
            mMean_expanded_layout = cute.append(
                mMean.layout, cute.make_layout((self.N,), stride=(0,))
            )
            mMean = cute.make_tensor(mMean.iterator, mMean_expanded_layout)
        delay_w_load = mB is not None
        self.kernel(
            mX,
            mW,
            mB,
            mO,
            mRstd,
            mMean,
            eps,
            tv_layout,
            tiler_mn,
            self.reload_from,
            delay_w_load,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=(
                [1, self.cluster_n, 1]
                if cutlass.const_expr(self.cluster_n > 1)
                else None
            ),
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mB: Optional[cute.Tensor],
        mO: cute.Tensor,
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        eps: cute.Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
        reload_from: cutlass.Constexpr = None,
        delay_w_load: cutlass.Constexpr = False,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        # We use domain_offset_i64 to deal with tensors larger than 2^31 elements
        mX, mO = [
            utils.domain_offset_i64((bidx * tiler_mn[0], 0), mT) for mT in (mX, mO)
        ]
        gX, gO = [cute.local_tile(mT, tiler_mn, (0, cluster_y)) for mT in (mX, mO)]
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y))
        gB = (
            cute.local_tile(mB, tiler_mn, (0, cluster_y))
            if cutlass.const_expr(mB is not None)
            else None
        )
        gRstd = (
            cute.local_tile(mRstd, tiler_mn, (bidx, cluster_y))
            if cutlass.const_expr(mRstd is not None)
            else None
        )
        gMean = (
            cute.local_tile(mMean, tiler_mn, (bidx, cluster_y))
            if cutlass.const_expr(mMean is not None)
            else None
        )

        # declare the atoms which will be used later for memory copy
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=128
        )
        copy_atom_load_X_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=128
        )
        copy_atom_load_W = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mW.element_type, num_bits_per_copy=128
        )
        copy_atom_store_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mO.element_type, num_bits_per_copy=128
        )

        thr_copy_X = cute.make_tiled_copy(
            copy_atom_load_X_async, tv_layout, tiler_mn
        ).get_slice(tidx)
        thr_copy_W = cute.make_tiled_copy(
            copy_atom_load_W, tv_layout, tiler_mn
        ).get_slice(tidx)
        thr_copy_O = cute.make_tiled_copy(
            copy_atom_store_O, tv_layout, tiler_mn
        ).get_slice(tidx)

        tWgW = thr_copy_W.partition_S(gW)
        tWgB = (
            thr_copy_W.partition_S(gB) if cutlass.const_expr(mB is not None) else None
        )
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_O.partition_D(gO)
        tXrRstd = (
            thr_copy_O.partition_D(gRstd)
            if cutlass.const_expr(mRstd is not None)
            else None
        )
        tXrMean = (
            thr_copy_O.partition_D(gMean)
            if cutlass.const_expr(mMean is not None)
            else None
        )
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)
        if cutlass.const_expr(mB is not None):
            tWrB = cute.make_fragment_like(tWgB)
            tXrB = thr_copy_X.retile(tWrB)
        else:
            tWrB = None
            tXrB = None
        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        tXpX = utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
        row = tXcX[0][0]
        if row < shape[0]:
            cute.copy(copy_atom_load_X_async, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()

        tWpW = utils.predicate_k(thr_copy_W.partition_S(cX), limit=shape[1])
        if cutlass.const_expr(not delay_w_load):
            cute.copy(copy_atom_load_W, tWgW, tWrW, pred=tWpW)
            if cutlass.const_expr(mB is not None):
                cute.copy(copy_atom_load_W, tWgB, tWrB, pred=tWpW)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        threads_per_row = tv_layout.shape[0][0]
        sum_x = row_reduce(
            x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr + 0 if cutlass.const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
            hook_fn=(
                cute.arch.cluster_wait
                if cutlass.const_expr(self.cluster_n > 1)
                else None
            ),
        )
        mean = sum_x / shape[1]
        if cutlass.const_expr(reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(cute.Float32)
        elif cutlass.const_expr(reload_from == "gmem"):
            cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)
            x = tXrX.load().to(cute.Float32)

        sum_sq_x_sub_mean = row_reduce(
            (x - mean) * (x - mean),
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 1],
            mbar_ptr + 1 if cutlass.const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
        )
        rstd = cute.math.rsqrt(sum_sq_x_sub_mean / shape[1] + eps, fastmath=True)
        if cutlass.const_expr(mRstd is not None):
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrRstd[0] = rstd
        if cutlass.const_expr(mMean is not None):
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrMean[0] = mean
        if cutlass.const_expr(delay_w_load):
            cute.copy(copy_atom_load_W, tWgW, tWrW, pred=tWpW)
            if cutlass.const_expr(mB is not None):
                cute.copy(copy_atom_load_W, tWgB, tWrB, pred=tWpW)
        if cutlass.const_expr(reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(cute.Float32)
        elif cutlass.const_expr(reload_from == "gmem"):
            cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)
            x = tXrX.load().to(cute.Float32)
        x_hat = (x - mean) * rstd
        w = tXrW.load().to(cute.Float32)
        y = x_hat * w
        if cutlass.const_expr(mB is not None):
            b = tXrB.load().to(tXrO.element_type)
            y = y.to(tXrO.element_type) + b
        else:
            y = y.to(tXrO.element_type)
        tXrO.store(y)
        tOpO = utils.predicate_k(thr_copy_O.partition_S(cX), limit=shape[1])
        if row < shape[0]:
            cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tOpO)


def layernorm(
    x: TensorLike,
    weight: TensorLike,
    bias: Optional[TensorLike] = None,
    eps: float = 1e-6,
    return_rstd: bool = False,
    return_mean: bool = False,
    *,
    stream: int | None = None,
) -> Any:
    """LayerNorm forward pass.

    Args:
        x: Input tensor of shape (M, N)
        weight: Weight tensor of shape (N,)
        eps: Small value for numerical stability
        return_rstd: Whether to return the reciprocal standard deviation
        return_mean: Whether to return the mean

    Returns:
        Normalized output tensor of same shape as x
        If return_rstd is True, also returns rstd tensor of shape (M,)
        If return_mean is True, also returns mean tensor of shape (M,)
    """
    x_sizes = getattr(x, "shape", None) or getattr(x, "sizes", None)
    w_sizes = getattr(weight, "shape", None) or getattr(weight, "sizes", None)
    if not isinstance(x_sizes, (tuple, list)) or len(x_sizes) != 2:
        raise TypeError("LayerNorm: x must be rank-2 tensor-like")
    if not isinstance(w_sizes, (tuple, list)) or len(w_sizes) != 1:
        raise TypeError("LayerNorm: weight must be rank-1 tensor-like")
    if int(x_sizes[-1]) != int(w_sizes[0]):
        raise ValueError("LayerNorm: last dim of x must match weight dim")
    if not (_is_cuda_tensor_like(x) and _is_cuda_tensor_like(weight)):
        raise ValueError("LayerNorm: tensors must be on CUDA device")
    _ = _cute_numeric_for(x)
    if _dtype_key(weight) not in ("float32", "torch.float32"):
        raise TypeError("LayerNorm: weight must be float32")
    if bias is not None:
        b_sizes = getattr(bias, "shape", None) or getattr(bias, "sizes", None)
        if not isinstance(b_sizes, (tuple, list)) or len(b_sizes) != 1 or int(b_sizes[0]) != int(x_sizes[-1]):
            raise ValueError("LayerNorm: bias must be 1D and match hidden dim")
        if not _is_cuda_tensor_like(bias):
            raise ValueError("LayerNorm: bias must be on CUDA device")
        _ = _cute_numeric_for(bias)
    M, N = int(x_sizes[0]), int(x_sizes[1])
    out = alloc.empty_like(x)
    rstd = alloc.empty((M,), like=x, dtype="float32") if return_rstd else None
    mean = alloc.empty((M,), like=x, dtype="float32") if return_mean else None
    dtype = _cute_numeric_for(x)

    def convert_2d(t: Any) -> cute.Tensor:
        return from_dlpack(t.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

    def convert_1d(t: Any, *, assumed_align: int, divisibility: int) -> cute.Tensor:
        return (
            from_dlpack(t.detach(), assumed_align=assumed_align)
            .mark_layout_dynamic(leading_dim=0)
            .mark_compact_shape_dynamic(mode=0, stride_order=(0,), divisibility=divisibility)
        )

    x_tensor, out_tensor = [
        convert_2d(t)
        for t in (x, out)
    ]
    weight_tensor = convert_1d(
        weight,
        assumed_align=16,
        divisibility=128 // cutlass.Float32.width,
    )
    if bias is not None:
        bias_dtype = _cute_numeric_for(bias)
        bias_tensor = convert_1d(
            bias,
            assumed_align=16,
            divisibility=128 // bias_dtype.width,
        )
    else:
        bias_tensor = None
    rstd_tensor = (
        from_dlpack(rstd.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if rstd is not None
        else None
    )
    mean_tensor = (
        from_dlpack(mean.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if mean is not None
        else None
    )
    current_stream = _cu_stream_handle(stream)
    compile_key = (
        dtype,
        N,
        bias_tensor.element_type if bias_tensor is not None else None,
        rstd is not None,
        mean is not None,
    )
    if compile_key not in layernorm.compile_cache:
        rmsnorm_op = LayerNorm(dtype, N)
        layernorm.compile_cache[compile_key] = cute.compile(
            rmsnorm_op,
            x_tensor,
            weight_tensor,
            bias_tensor,
            out_tensor,
            rstd_tensor,
            mean_tensor,
            current_stream,
        )
    layernorm.compile_cache[compile_key](
        x_tensor,
        weight_tensor,
        bias_tensor,
        out_tensor,
        rstd_tensor,
        mean_tensor,
        current_stream,
        eps,
    )
    return (
        (out, rstd, mean)
        if return_mean and return_rstd
        else (
            (out, rstd)
            if return_rstd and not return_mean
            else ((out, mean) if return_mean and not return_rstd else (out))
        )
    )


layernorm.compile_cache = {}


class LayerNormBackward(ReductionBase):
    def __init__(self, dtype: cutlass.Numeric, N: int):
        super().__init__(dtype, N, stage=2, reduction_dtype=Float32)
        self.reload_wdy = None if N <= 16 * 1024 else "smem"
        if self.N > 128 * 1024 and self.dtype.width >= 32:
            raise ValueError(
                "LayerNormBackward does not support N > 128k with dtype >= 32 bits"
            )

    def _get_num_threads(self):
        return 128 if self.N <= 4096 else 256

    def _calculate_threads_per_row(self):
        N = self.N
        return (
            8
            if N <= 64
            else (
                16
                if N <= 128
                else (
                    32
                    if N <= 256
                    else (64 if N <= 512 else (128 if N <= 4096 else 256))
                )
            )
        )

    def _set_cluster_n(self):
        self.cluster_n = 1

    def _smem_size_in_bytes(self, tiler_mn, num_warps, do_dtype=None):
        if do_dtype is None:
            do_dtype = self.dtype
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)) * 2
            + cute.size_in_bytes(do_dtype, cute.make_layout(tiler_mn)) * 2
            + self.stage
            * num_warps
            * self.cluster_n
            * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8) * 2
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mdY: cute.Tensor,
        mMean: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: Optional[cute.Tensor],
        mdB: Optional[cute.Tensor],
        sm_count: Int32,
        stream: cuda.CUstream,
    ):
        semistatic_shape = (*mX.shape[:-1], self.N)
        new_stride = lambda t: (
            cute.assume(t.stride[0], divby=128 // t.element_type.width),
            t.stride[1],
        )
        mX, mdY, mdX = [
            cute.make_tensor(
                t.iterator, cute.make_layout(semistatic_shape, stride=new_stride(t))
            )
            for t in (mX, mdY, mdX)
        ]
        self._set_cluster_n()
        largest_dtype_width = cutlass.const_expr(
            max(mX.element_type.width, mdY.element_type.width, mdX.element_type.width)
        )
        tiler_mn, tv_layout = self._get_tv_layout(
            num_copy_bits=128 // largest_dtype_width * mX.element_type.width
        )
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW = cute.make_tensor(mW.iterator, mW_expanded_layout)

        num_blocks = sm_count
        self.kernel(
            mX,
            mW,
            mdY,
            mMean,
            mRstd,
            mdX,
            mdW,
            mdB,
            tv_layout,
            tiler_mn,
        ).launch(
            grid=[num_blocks, self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=None,
            smem=self._smem_size_in_bytes(
                tiler_mn, num_warps, do_dtype=mdY.element_type
            ),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mdY: cute.Tensor,
        mMean: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: Optional[cute.Tensor],
        mdB: Optional[cute.Tensor],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_start, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()

        shape = mX.shape
        M, N = shape[0], shape[1]
        is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * self.cluster_n)

        idX = cute.make_identity_tensor(shape)

        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_ordered_layout(
            (tiler_mn[0], tiler_mn[1], 2), order=(1, 0, 2)
        )
        sX = smem.allocate_tensor(mX.element_type, smem_layout, byte_alignment=16)
        sdY = smem.allocate_tensor(mdY.element_type, smem_layout, byte_alignment=16)
        reduction_buffer, _ = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout, is_persistent=False
        )

        num_copy_elems = tv_layout.shape[1][0]
        copy_atom = copy_utils.get_copy_atom(
            mX.element_type, num_copy_elems, is_async=False
        )
        thr_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn).get_slice(tidx)
        copy = partial(copy_utils.copy, num_copy_elems=num_copy_elems)

        gX, gdY, gdX, cX = [
            cute.local_tile(mT, tiler_mn, (None, 0)) for mT in (mX, mdY, mdX, idX)
        ]
        gW = cute.local_tile(mW, tiler_mn, (0, 0))
        gdW, gdB = [
            (
                cute.local_tile(mT, (1, tiler_mn[1]), (bidx_start, 0))
                if cutlass.const_expr(mT is not None)
                else None
            )
            for mT in (mdW, mdB)
        ]

        tXgX = thr_copy.partition_S(gX)
        tXsX = thr_copy.partition_D(sX)
        tXgdY = thr_copy.partition_S(gdY)
        tXsdY = thr_copy.partition_D(sdY)
        tXgdX = thr_copy.partition_D(gdX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None, None]

        tXrX, tXrdY, tXrdX = [
            cute.make_fragment_like(thr[None, None, None, 0])
            for thr in (tXgX, tXgdY, tXgdX)
        ]

        tXpX = (
            utils.predicate_k(thr_copy.partition_S(cX[None, None, 0]), limit=shape[1])
            if not is_even_N
            else None
        )

        tXgdW, tXrdW = None, None
        tXgdB, tXrdB = None, None
        if cutlass.const_expr(mdW is not None):
            tXgdW = thr_copy.partition_S(gdW)
            tXrdW = cute.make_fragment_like(tXgdW, Float32)
        if cutlass.const_expr(mdB is not None):
            tXgdB = thr_copy.partition_S(gdB)
            tXrdB = cute.make_fragment_like(tXgdB, Float32)

        tXrW = cute.make_fragment_like(thr_copy.partition_S(gW))
        if not is_even_N:
            tXrW.fill(0.0)
        copy(thr_copy.partition_S(gW), tXrW, pred=tXpX)

        if cutlass.const_expr(mdW is not None):
            tXrdW.fill(0.0)
        if cutlass.const_expr(mdB is not None):
            tXrdB.fill(0.0)

        row = tXcX[None, None, None, bidx_start][0][0]
        if row < M:
            tXgX_cur = utils.coord_offset_i64(bidx_start, tXgX, dim=3)[
                None, None, None, 0
            ]
            tXgdY_cur = utils.coord_offset_i64(bidx_start, tXgdY, dim=3)[
                None, None, None, 0
            ]
            copy(tXgX_cur, tXsX[None, None, None, 0], pred=tXpX, is_async=True)
            copy(tXgdY_cur, tXsdY[None, None, None, 0], pred=tXpX, is_async=True)
        elif tiler_mn[0] > 1:
            utils.fill_oob(
                tXsX[None, None, None, 0], None, fill_value=mX.element_type.zero
            )
            utils.fill_oob(
                tXsdY[None, None, None, 0], None, fill_value=mdY.element_type.zero
            )
        cute.arch.cp_async_commit_group()

        threads_per_row = tv_layout.shape[0][0]
        stage = Int32(0)

        for bidx in cutlass.range(bidx_start, cute.ceil_div(M, tiler_mn[0]), gdim):
            row = tXcX[None, None, None, bidx][0][0]
            if row + gdim * tiler_mn[0] < M:
                tXgX_cur = utils.coord_offset_i64(bidx + gdim, tXgX, dim=3)[
                    None, None, None, 0
                ]
                tXgdY_cur = utils.coord_offset_i64(bidx + gdim, tXgdY, dim=3)[
                    None, None, None, 0
                ]
                copy(
                    tXgX_cur,
                    tXsX[None, None, None, stage ^ 1],
                    pred=tXpX,
                    is_async=True,
                )
                copy(
                    tXgdY_cur,
                    tXsdY[None, None, None, stage ^ 1],
                    pred=tXpX,
                    is_async=True,
                )
            elif tiler_mn[0] > 1:
                utils.fill_oob(
                    tXsX[None, None, None, stage ^ 1],
                    None,
                    fill_value=mX.element_type.zero,
                )
                utils.fill_oob(
                    tXsdY[None, None, None, stage ^ 1],
                    None,
                    fill_value=mdY.element_type.zero,
                )
            cute.arch.cp_async_commit_group()

            mean = cutlass.Float.zero
            rstd = cutlass.Float.zero
            if row < M or tiler_mn[0] == 1:
                mean = mMean[row]
                rstd = mRstd[row]

            cute.arch.cp_async_wait_group(1)
            cute.autovec_copy(tXsX[None, None, None, stage], tXrX)
            x = tXrX.load().to(cute.Float32)
            cute.autovec_copy(tXsdY[None, None, None, stage], tXrdY)
            dy = tXrdY.load().to(cute.Float32)

            x_hat = (x - mean) * rstd
            wdy = dy
            if cutlass.const_expr(mW is not None):
                wdy *= tXrW.load().to(Float32)

            sum_xhat_wdy = row_reduce(
                x_hat * wdy,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, stage],
                None,
                init_val=0.0,
            )
            sum_wdy = row_reduce(
                wdy,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, stage],
                None,
                init_val=0.0,
            )
            mean_xhat_wdy = sum_xhat_wdy / shape[1]
            mean_wdy = sum_wdy / shape[1]

            dx = (wdy - mean_wdy - x_hat * mean_xhat_wdy) * rstd
            tXrdX.store(dx.to(tXrdX.element_type))
            if row < M or tiler_mn[0] == 1:
                tXgdX_cur = utils.coord_offset_i64(bidx, tXgdX, dim=3)[
                    None, None, None, 0
                ]
                copy(tXrdX, tXgdX_cur, pred=tXpX)

            if cutlass.const_expr(mdW is not None):
                tXrdW.store(tXrdW.load() + dy * x_hat)
            if cutlass.const_expr(mdB is not None):
                tXrdB.store(tXrdB.load() + dy)

            stage ^= 1

        if cutlass.const_expr(tiler_mn[0] > 1):
            if cutlass.const_expr(mdW is not None):
                sdW = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdW = thr_copy.partition_D(sdW)
                cute.arch.barrier()
                row = tXcX[None, None, None, 0][0][0]
                if row > 0:
                    cute.autovec_copy(tXrdW, tXsdW)
                cute.arch.barrier()
                if row == 0:
                    for i in cutlass.range_constexpr(
                        1, cutlass.const_expr(tiler_mn[0])
                    ):
                        tXrdW_other = cute.make_fragment_like(tXrdW)
                        tXsdW_other = cute.make_tensor(
                            tXsdW.iterator + i * sdW.stride[0], tXsdW.layout
                        )
                        cute.autovec_copy(tXsdW_other, tXrdW_other)
                        tXrdW.store(tXrdW.load() + tXrdW_other.load())
                    copy(tXrdW, tXgdW, pred=tXpX)
                cute.arch.barrier()
            if cutlass.const_expr(mdB is not None):
                sdB = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdB = thr_copy.partition_D(sdB)
                cute.arch.barrier()
                row = tXcX[None, None, None, 0][0][0]
                if row > 0:
                    cute.autovec_copy(tXrdB, tXsdB)
                cute.arch.barrier()
                if row == 0:
                    for i in cutlass.range_constexpr(
                        1, cutlass.const_expr(tiler_mn[0])
                    ):
                        tXrdB_other = cute.make_fragment_like(tXrdB)
                        tXsdB_other = cute.make_tensor(
                            tXsdB.iterator + i * sdB.stride[0], tXsdB.layout
                        )
                        cute.autovec_copy(tXsdB_other, tXrdB_other)
                        tXrdB.store(tXrdB.load() + tXrdB_other.load())
                    copy(tXrdB, tXgdB, pred=tXpX)
                cute.arch.barrier()
        else:
            if cutlass.const_expr(mdW is not None):
                copy(tXrdW, tXgdW, pred=tXpX)
            if cutlass.const_expr(mdB is not None):
                copy(tXrdB, tXgdB, pred=tXpX)


def _get_sm_count(N: int, device_index: int) -> int:
    sm_count_multiple = (
        16
        if N <= 256
        else (8 if N <= 1024 else (4 if N <= 2048 else (2 if N <= 4096 else 1)))
    )
    err, sm_count = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        int(device_index),
    )
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            "cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) failed: "
            f"{cuda.cuGetErrorString(err)[1]}"
        )
    if N <= 8192:
        sm_count *= sm_count_multiple
    elif N <= 16384:
        sm_count //= 2
    else:
        sm_count *= 2
    return sm_count


def _layernorm_bwd(
    x: TensorLike,
    weight: TensorLike,
    dout: TensorLike,
    mean: TensorLike,
    rstd: TensorLike,
    dx: TensorLike,
    dw_partial: Optional[TensorLike] = None,
    db_partial: Optional[TensorLike] = None,
    sm_count: Optional[int] = None,
    *,
    stream: int | None = None,
) -> None:
    x_sizes = getattr(x, "shape", None) or getattr(x, "sizes", None)
    dy_sizes = getattr(dout, "shape", None) or getattr(dout, "sizes", None)
    w_sizes = getattr(weight, "shape", None) or getattr(weight, "sizes", None)
    m_sizes = getattr(mean, "shape", None) or getattr(mean, "sizes", None)
    r_sizes = getattr(rstd, "shape", None) or getattr(rstd, "sizes", None)
    if not isinstance(x_sizes, (tuple, list)) or len(x_sizes) != 2:
        raise TypeError("LayerNorm backward: x must be rank-2")
    if not isinstance(dy_sizes, (tuple, list)) or tuple(int(s) for s in dy_sizes) != tuple(
        int(s) for s in x_sizes
    ):
        raise ValueError("LayerNorm backward: dout must match x shape")
    if not isinstance(w_sizes, (tuple, list)) or len(w_sizes) != 1 or int(w_sizes[0]) != int(x_sizes[1]):
        raise ValueError("LayerNorm backward: weight must be 1D with size N")
    if not isinstance(m_sizes, (tuple, list)) or len(m_sizes) != 1 or int(m_sizes[0]) != int(x_sizes[0]):
        raise ValueError("LayerNorm backward: mean must be 1D with size M")
    if not isinstance(r_sizes, (tuple, list)) or len(r_sizes) != 1 or int(r_sizes[0]) != int(x_sizes[0]):
        raise ValueError("LayerNorm backward: rstd must be 1D with size M")
    if not (_is_cuda_tensor_like(x) and _is_cuda_tensor_like(dout) and _is_cuda_tensor_like(weight)):
        raise ValueError("LayerNorm backward: x/dout/weight must be CUDA tensors")
    if not (_is_cuda_tensor_like(mean) and _is_cuda_tensor_like(rstd)):
        raise ValueError("LayerNorm backward: mean/rstd must be CUDA tensors")

    convert_to_tv = lambda t: (
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
    )
    x_tensor = convert_to_tv(x)
    dout_tensor = convert_to_tv(dout)
    dx_tensor = convert_to_tv(dx)

    weight_tensor = (
        from_dlpack(weight.detach(), assumed_align=16)
        .mark_layout_dynamic(leading_dim=0)
        .mark_compact_shape_dynamic(mode=0, stride_order=(0,), divisibility=128 // cutlass.Float32.width)
    )
    mean_tensor = from_dlpack(mean.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )
    rstd_tensor = from_dlpack(rstd.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )

    dw_partial_tensor = (
        from_dlpack(dw_partial.detach(), assumed_align=16).mark_compact_shape_dynamic(mode=0)
        if dw_partial is not None
        else None
    )
    db_partial_tensor = (
        from_dlpack(db_partial.detach(), assumed_align=16).mark_compact_shape_dynamic(mode=0)
        if db_partial is not None
        else None
    )

    if sm_count is None:
        sm_count = (
            int((getattr(dw_partial, "shape", None) or getattr(dw_partial, "sizes", None))[0])
            if dw_partial is not None
            else (
                int((getattr(db_partial, "shape", None) or getattr(db_partial, "sizes", None))[0])
                if db_partial is not None
                else _get_sm_count(int(x_sizes[1]), _device_index(x))
            )
        )

    current_stream = _cu_stream_handle(stream)

    compile_key = (
        int(x_sizes[1]),
        x_tensor.element_type,
        _dtype_key(db_partial) if db_partial is not None else None,
    )
    if compile_key not in _layernorm_bwd.compile_cache:
        ln_bw = LayerNormBackward(x_tensor.element_type, int(x_sizes[1]))
        _layernorm_bwd.compile_cache[compile_key] = cute.compile(
            ln_bw,
            x_tensor,
            weight_tensor,
            dout_tensor,
            mean_tensor,
            rstd_tensor,
            dx_tensor,
            dw_partial_tensor,
            db_partial_tensor,
            sm_count,
            current_stream,
        )

    _layernorm_bwd.compile_cache[compile_key](
        x_tensor,
        weight_tensor,
        dout_tensor,
        mean_tensor,
        rstd_tensor,
        dx_tensor,
        dw_partial_tensor,
        db_partial_tensor,
        sm_count,
        current_stream,
    )


_layernorm_bwd.compile_cache = {}


def layernorm_bwd(
    x: TensorLike,
    weight: TensorLike,
    grad_out: TensorLike,
    mean: TensorLike,
    rstd: TensorLike,
    *,
    has_bias: bool,
    stream: int | None = None,
) -> Tuple[Any, Any, Optional[Any]]:
    x_sizes = getattr(x, "shape", None) or getattr(x, "sizes", None)
    if not isinstance(x_sizes, (tuple, list)) or len(x_sizes) != 2:
        raise TypeError("layernorm_bwd: x must be rank-2")
    N = int(x_sizes[1])
    dx = alloc.empty_like(x)
    sm_count = _get_sm_count(N, _device_index(x))
    dw_partial = alloc.empty((sm_count, N), like=x, dtype="float32")
    db_partial = (
        alloc.empty((sm_count, N), like=x, dtype="float32")
        if has_bias
        else None
    )

    _layernorm_bwd(
        x,
        weight,
        grad_out,
        mean,
        rstd,
        dx,
        dw_partial,
        db_partial,
        sm_count,
        stream=stream,
    )

    dw = dw_partial.sum(dim=0).to(weight.dtype)
    db = (
        db_partial.sum(dim=0).to(weight.dtype)
        if has_bias and db_partial is not None
        else None
    )
    return dx, dw, db


__all__ = [
    "LayerNorm",
    "layernorm",
    "layernorm_bwd",
]
