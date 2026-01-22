# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# pyright: reportMissingImports=false, reportOptionalMemberAccess=false, reportGeneralTypeIssues=false

import math
from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack
from vibe_kernels.common import alloc
from vibe_kernels.common.tensor_types import TensorLike

from vibe_kernels.gemm.impl.cutedsl_gemm import utils
from vibe_kernels.gemm.impl.cutedsl_gemm.cute_dsl_utils import torch2cute_dtype_map
from vibe_kernels.rmsnorm.impl.cutedsl_rmsnorm.reduce import (
    online_softmax_reduce,
    row_reduce,
)
from vibe_kernels.rmsnorm.impl.cutedsl_rmsnorm.reduction_base import ReductionBase

Tensor = TensorLike


class Softmax(ReductionBase):
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        *,
        online_softmax: bool = True,
    ):
        super().__init__(
            dtype,
            N,
            stage=1 if online_softmax else 2,
            reduction_dtype=cutlass.Int64 if online_softmax else cutlass.Float32,
        )
        self.online_softmax = online_softmax

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
                    if N <= 3072
                    else (64 if N <= 6144 else (128 if N <= 16384 else 256))
                )
            )
        )

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

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype
        self._set_cluster_n()
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        self.kernel(mX, mO, tv_layout, tiler_mn).launch(
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
        mO: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        mX, mO = [
            utils.domain_offset_i64((bidx * tiler_mn[0], 0), mT) for mT in (mX, mO)
        ]
        gX, gO = [cute.local_tile(mT, tiler_mn, (0, cluster_y)) for mT in (mX, mO)]
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=128
        )
        copy_atom_store_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gO.element_type, num_bits_per_copy=128
        )

        thr_copy_X = cute.make_tiled_copy(
            copy_atom_load_X, tv_layout, tiler_mn
        ).get_slice(tidx)
        thr_copy_O = cute.make_tiled_copy(
            copy_atom_store_O, tv_layout, tiler_mn
        ).get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_O.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if cutlass.const_expr(not is_even_N)
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        if cutlass.const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)

        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)
        threads_per_row = tv_layout.shape[0][0]

        if cutlass.const_expr(self.online_softmax):
            max_x, denom, exp_x = online_softmax_reduce(
                x,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                hook_fn=(
                    cute.arch.cluster_wait
                    if cutlass.const_expr(self.cluster_n > 1)
                    else None
                ),
                return_exp_x=True,
            )
        else:
            max_x = row_reduce(
                x,
                cute.ReductionOp.MAX,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr + 0 if cutlass.const_expr(self.cluster_n > 1) else None,
                init_val=-Float32.inf,
                hook_fn=(
                    cute.arch.cluster_wait
                    if cutlass.const_expr(self.cluster_n > 1)
                    else None
                ),
            )
            log2_e = math.log2(math.e)
            exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=True)
            denom = row_reduce(
                exp_x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 1],
                mbar_ptr + 1 if cutlass.const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
            )

        y = exp_x * cute.arch.rcp_approx(denom)

        tXrO.store(y.to(tXrO.element_type))
        tOpO = (
            utils.predicate_k(thr_copy_O.partition_S(cX), limit=shape[1])
            if cutlass.const_expr(not is_even_N)
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tOpO)


class SoftmaxBackward(ReductionBase):
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        super().__init__(dtype, N, stage=1, reduction_dtype=cutlass.Float32)

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
                    if N <= 3072
                    else (64 if N <= 6144 else (128 if N <= 8192 else 256))
                )
            )
        )

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
                if N <= 16 * 1024
                else (
                    2
                    if N <= 32 * 1024
                    else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16))
                )
            )
        self.cluster_n = cluster_n

    def _get_num_threads(self):
        return 128 if self.N <= 8192 else 256

    def _smem_size_in_bytes(self, tiler_mn, num_warps):
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)) * 2
            + self.stage
            * num_warps
            * self.cluster_n
            * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(
        self,
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mdY.element_type == self.dtype
        assert mY.element_type == self.dtype
        assert mdX.element_type == self.dtype
        self._set_cluster_n()
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        self.kernel(mdY, mY, mdX, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(mdY.shape[0], tiler_mn[0]), self.cluster_n, 1],
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
        mdY: cute.Tensor,
        mY: cute.Tensor,
        mdX: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        shape = mdY.shape
        idX = cute.make_identity_tensor(shape)
        mdY, mY, mdX = [
            utils.domain_offset_i64((bidx * tiler_mn[0], 0), mT)
            for mT in (mdY, mY, mdX)
        ]
        gdY, gY, gdX = [
            cute.local_tile(mT, tiler_mn, (0, cluster_y)) for mT in (mdY, mY, mdX)
        ]
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        smem = cutlass.utils.SmemAllocator()
        sdY = smem.allocate_tensor(
            mdY.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        sY = smem.allocate_tensor(
            mY.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mdY.element_type, num_bits_per_copy=128
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gdX.element_type, num_bits_per_copy=128
        )

        thr_copy_load = cute.make_tiled_copy(
            copy_atom_load, tv_layout, tiler_mn
        ).get_slice(tidx)
        thr_copy_store = cute.make_tiled_copy(
            copy_atom_store, tv_layout, tiler_mn
        ).get_slice(tidx)

        tdYgdY = thr_copy_load.partition_S(gdY)
        tdYsdY = thr_copy_load.partition_D(sdY)
        tYgY = thr_copy_load.partition_S(gY)
        tYsY = thr_copy_load.partition_D(sY)
        tdXgdX = thr_copy_store.partition_D(gdX)
        tXcX = thr_copy_load.partition_S(cX)[(0, None), None, None]

        tdYrdY, tYrY, tdXrdX = [
            cute.make_fragment_like(thr) for thr in (tdYgdY, tYgY, tdXgdX)
        ]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tdYpdY = (
            utils.predicate_k(thr_copy_load.partition_S(cX), limit=shape[1])
            if cutlass.const_expr(not is_even_N)
            else None
        )

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load, tdYgdY, tdYsdY, pred=tdYpdY)
            cute.copy(copy_atom_load, tYgY, tYsY, pred=tdYpdY)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(tdYsdY, tdYrdY)
        cute.autovec_copy(tYsY, tYrY)
        dy = tdYrdY.load().to(Float32)
        y = tYrY.load().to(Float32)

        threads_per_row = tv_layout.shape[0][0]
        dot = row_reduce(
            dy * y,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr if cutlass.const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
            hook_fn=(
                cute.arch.cluster_wait
                if cutlass.const_expr(self.cluster_n > 1)
                else None
            ),
        )

        dx = y * (dy - dot)
        tdXrdX.store(dx.to(tdXrdX.element_type))
        tdXpdX = (
            utils.predicate_k(thr_copy_store.partition_S(cX), limit=shape[1])
            if cutlass.const_expr(not is_even_N)
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_store, tdXrdX, tdXgdX, pred=tdXpdX)


class LogSoftmax(ReductionBase):
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        *,
        online_softmax: bool = True,
    ):
        super().__init__(
            dtype,
            N,
            stage=1 if online_softmax else 2,
            reduction_dtype=cutlass.Int64 if online_softmax else cutlass.Float32,
        )
        self.online_softmax = online_softmax

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
                    if N <= 3072
                    else (64 if N <= 6144 else (128 if N <= 16384 else 256))
                )
            )
        )

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

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype
        self._set_cluster_n()
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        self.kernel(mX, mO, tv_layout, tiler_mn).launch(
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
        mO: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        mX, mO = [
            utils.domain_offset_i64((bidx * tiler_mn[0], 0), mT) for mT in (mX, mO)
        ]
        gX, gO = [cute.local_tile(mT, tiler_mn, (0, cluster_y)) for mT in (mX, mO)]
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=128
        )
        copy_atom_store_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gO.element_type, num_bits_per_copy=128
        )

        thr_copy_X = cute.make_tiled_copy(
            copy_atom_load_X, tv_layout, tiler_mn
        ).get_slice(tidx)
        thr_copy_O = cute.make_tiled_copy(
            copy_atom_store_O, tv_layout, tiler_mn
        ).get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_O.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if cutlass.const_expr(not is_even_N)
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        if cutlass.const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)

        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)
        threads_per_row = tv_layout.shape[0][0]

        if cutlass.const_expr(self.online_softmax):
            max_x, denom, _ = online_softmax_reduce(
                x,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                hook_fn=(
                    cute.arch.cluster_wait
                    if cutlass.const_expr(self.cluster_n > 1)
                    else None
                ),
                return_exp_x=False,
            )
        else:
            max_x = row_reduce(
                x,
                cute.ReductionOp.MAX,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr + 0 if cutlass.const_expr(self.cluster_n > 1) else None,
                init_val=-Float32.inf,
                hook_fn=(
                    cute.arch.cluster_wait
                    if cutlass.const_expr(self.cluster_n > 1)
                    else None
                ),
            )
            log2_e = math.log2(math.e)
            exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=True)
            denom = row_reduce(
                exp_x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 1],
                mbar_ptr + 1 if cutlass.const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
            )

        logsumexp = cute.math.log(denom, fastmath=True) + max_x
        out_val = x - logsumexp

        tXrO.store(out_val.to(tXrO.element_type))
        tOpO = (
            utils.predicate_k(thr_copy_O.partition_S(cX), limit=shape[1])
            if cutlass.const_expr(not is_even_N)
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tOpO)


class LogSoftmaxBackward(ReductionBase):
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        super().__init__(dtype, N, stage=1, reduction_dtype=cutlass.Float32)

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
                    if N <= 3072
                    else (64 if N <= 6144 else (128 if N <= 8192 else 256))
                )
            )
        )

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
                if N <= 16 * 1024
                else (
                    2
                    if N <= 32 * 1024
                    else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16))
                )
            )
        self.cluster_n = cluster_n

    def _get_num_threads(self):
        return 128 if self.N <= 8192 else 256

    def _smem_size_in_bytes(self, tiler_mn, num_warps):
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)) * 2
            + self.stage
            * num_warps
            * self.cluster_n
            * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(
        self,
        mdY: cute.Tensor,
        mLogY: cute.Tensor,
        mdX: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mdY.element_type == self.dtype
        assert mLogY.element_type == self.dtype
        assert mdX.element_type == self.dtype
        self._set_cluster_n()
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        self.kernel(mdY, mLogY, mdX, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(mdY.shape[0], tiler_mn[0]), self.cluster_n, 1],
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
        mdY: cute.Tensor,
        mLogY: cute.Tensor,
        mdX: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        shape = mdY.shape
        idX = cute.make_identity_tensor(shape)
        mdY, mLogY, mdX = [
            utils.domain_offset_i64((bidx * tiler_mn[0], 0), mT)
            for mT in (mdY, mLogY, mdX)
        ]
        gdY, gLogY, gdX = [
            cute.local_tile(mT, tiler_mn, (0, cluster_y)) for mT in (mdY, mLogY, mdX)
        ]
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        smem = cutlass.utils.SmemAllocator()
        sdY = smem.allocate_tensor(
            mdY.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        sLogY = smem.allocate_tensor(
            mLogY.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout
        )

        copy_atom_load = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mdY.element_type, num_bits_per_copy=128
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gdX.element_type, num_bits_per_copy=128
        )

        thr_copy_load = cute.make_tiled_copy(
            copy_atom_load, tv_layout, tiler_mn
        ).get_slice(tidx)
        thr_copy_store = cute.make_tiled_copy(
            copy_atom_store, tv_layout, tiler_mn
        ).get_slice(tidx)

        tdYgdY = thr_copy_load.partition_S(gdY)
        tdYsdY = thr_copy_load.partition_D(sdY)
        tLogYg = thr_copy_load.partition_S(gLogY)
        tLogYs = thr_copy_load.partition_D(sLogY)
        tdXgdX = thr_copy_store.partition_D(gdX)
        tXcX = thr_copy_load.partition_S(cX)[(0, None), None, None]

        tdYrdY, tLogY, tdXrdX = [
            cute.make_fragment_like(thr) for thr in (tdYgdY, tLogYg, tdXgdX)
        ]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tdYpdY = (
            utils.predicate_k(thr_copy_load.partition_S(cX), limit=shape[1])
            if cutlass.const_expr(not is_even_N)
            else None
        )

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load, tdYgdY, tdYsdY, pred=tdYpdY)
            cute.copy(copy_atom_load, tLogYg, tLogYs, pred=tdYpdY)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(tdYsdY, tdYrdY)
        cute.autovec_copy(tLogYs, tLogY)
        dy = tdYrdY.load().to(Float32)
        log_y = tLogY.load().to(Float32)
        exp_log_y = cute.math.exp(log_y, fastmath=True)

        threads_per_row = tv_layout.shape[0][0]
        sum_dy = row_reduce(
            dy,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr if cutlass.const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
            hook_fn=(
                cute.arch.cluster_wait
                if cutlass.const_expr(self.cluster_n > 1)
                else None
            ),
        )

        dx = dy - exp_log_y * sum_dy
        tdXrdX.store(dx.to(tdXrdX.element_type))
        tdXpdX = (
            utils.predicate_k(thr_copy_store.partition_S(cX), limit=shape[1])
            if cutlass.const_expr(not is_even_N)
            else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_store, tdXrdX, tdXgdX, pred=tdXpdX)


def _convert_tensor(tensor: Tensor) -> cute.Tensor:
    return from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1)
    )


def _alignment_for_dtype(tensor: Tensor) -> int:
    return max(128 // (tensor.element_size() * 8), 1)


def _should_use_online(cols: int) -> bool:
    return True


def _maybe_pad_forward(x: Tensor, fill: float) -> Tuple[Tensor, int, Optional[int]]:
    cols = x.size(1)
    align = _alignment_for_dtype(x)
    if cols % align == 0:
        return x, cols, None
    padded_cols = ((cols + align - 1) // align) * align
    padded = x.new_full((x.size(0), padded_cols), fill)
    padded[:, :cols] = x
    return padded, cols, padded_cols


def _maybe_pad_backward(
    data: Tensor, saved: Tensor, fill: float
) -> Tuple[Tensor, Tensor, int]:
    cols = saved.size(1)
    align = _alignment_for_dtype(saved)
    if cols % align == 0:
        return data, saved, cols
    padded_cols = ((cols + align - 1) // align) * align
    padded_data = data.new_zeros((data.size(0), padded_cols))
    padded_saved = saved.new_full((saved.size(0), padded_cols), fill)
    padded_data[:, :cols] = data
    padded_saved[:, :cols] = saved
    return padded_data, padded_saved, cols


def _trim_output(out: Tensor, cols: int) -> Tensor:
    return out if out.size(1) == cols else out[:, :cols]


def _softmax_forward_impl(x: Tensor, out: Tensor) -> None:
    dtype = torch2cute_dtype_map[x.dtype]
    convert = _convert_tensor
    x_tensor, out_tensor = [convert(tensor) for tensor in (x, out)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    use_online = _should_use_online(x.size(1))
    compile_key = (dtype, x.size(1), use_online)
    if compile_key not in _softmax_forward_impl.compile_cache:
        op = Softmax(dtype, x.size(1), online_softmax=use_online)
        _softmax_forward_impl.compile_cache[compile_key] = cute.compile(
            op, x_tensor, out_tensor, current_stream
        )
    _softmax_forward_impl.compile_cache[compile_key](
        x_tensor, out_tensor, current_stream
    )


_softmax_forward_impl.compile_cache = {}


def _log_softmax_forward_impl(x: Tensor, out: Tensor) -> None:
    dtype = torch2cute_dtype_map[x.dtype]
    convert = _convert_tensor
    x_tensor, out_tensor = [convert(tensor) for tensor in (x, out)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    use_online = _should_use_online(x.size(1))
    compile_key = (dtype, x.size(1), use_online)
    if compile_key not in _log_softmax_forward_impl.compile_cache:
        op = LogSoftmax(dtype, x.size(1), online_softmax=use_online)
        _log_softmax_forward_impl.compile_cache[compile_key] = cute.compile(
            op, x_tensor, out_tensor, current_stream
        )
    _log_softmax_forward_impl.compile_cache[compile_key](
        x_tensor, out_tensor, current_stream
    )


_log_softmax_forward_impl.compile_cache = {}


def _softmax_backward_impl(dy: Tensor, y: Tensor, dx: Tensor) -> None:
    dtype = torch2cute_dtype_map[dy.dtype]
    convert = _convert_tensor
    dy_tensor, y_tensor, dx_tensor = [convert(tensor) for tensor in (dy, y, dx)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, dy.size(1))
    if compile_key not in _softmax_backward_impl.compile_cache:
        op = SoftmaxBackward(dtype, dy.size(1))
        _softmax_backward_impl.compile_cache[compile_key] = cute.compile(
            op, dy_tensor, y_tensor, dx_tensor, current_stream
        )
    _softmax_backward_impl.compile_cache[compile_key](
        dy_tensor, y_tensor, dx_tensor, current_stream
    )


_softmax_backward_impl.compile_cache = {}


def _log_softmax_backward_impl(dy: Tensor, log_y: Tensor, dx: Tensor) -> None:
    dtype = torch2cute_dtype_map[dy.dtype]
    convert = _convert_tensor
    dy_tensor, log_y_tensor, dx_tensor = [convert(tensor) for tensor in (dy, log_y, dx)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, dy.size(1))
    if compile_key not in _log_softmax_backward_impl.compile_cache:
        op = LogSoftmaxBackward(dtype, dy.size(1))
        _log_softmax_backward_impl.compile_cache[compile_key] = cute.compile(
            op, dy_tensor, log_y_tensor, dx_tensor, current_stream
        )
    _log_softmax_backward_impl.compile_cache[compile_key](
        dy_tensor, log_y_tensor, dx_tensor, current_stream
    )


_log_softmax_backward_impl.compile_cache = {}


@torch.library.custom_op("kernel_factory::_cutedsl_softmax_fwd", mutates_args={"out"})
def _softmax_fwd(x: Tensor, out: Tensor) -> None:
    _softmax_forward_impl(x, out)


@torch.library.custom_op(
    "kernel_factory::_cutedsl_log_softmax_fwd", mutates_args={"out"}
)
def _log_softmax_fwd(x: Tensor, out: Tensor) -> None:
    _log_softmax_forward_impl(x, out)


@torch.library.custom_op("kernel_factory::_cutedsl_softmax_bwd", mutates_args={"dx"})
def _softmax_bwd(dy: Tensor, y: Tensor, dx: Tensor) -> None:
    _softmax_backward_impl(dy, y, dx)


@torch.library.custom_op(
    "kernel_factory::_cutedsl_log_softmax_bwd", mutates_args={"dx"}
)
def _log_softmax_bwd(dy: Tensor, log_y: Tensor, dx: Tensor) -> None:
    _log_softmax_backward_impl(dy, log_y, dx)


def softmax_forward(x: Tensor) -> Tensor:
    padded_x, cols, _ = _maybe_pad_forward(x, fill=-float("inf"))
    out = alloc.empty_like(padded_x)
    _softmax_fwd(padded_x, out)
    return _trim_output(out, cols)


def softmax_backward(dy: Tensor, y: Tensor) -> Tensor:
    dy_pad, y_pad, cols = _maybe_pad_backward(dy, y, fill=0.0)
    dx_pad = alloc.empty_like(dy_pad)
    _softmax_bwd(dy_pad, y_pad, dx_pad)
    return _trim_output(dx_pad, cols)


def log_softmax_forward(x: Tensor) -> Tensor:
    padded_x, cols, _ = _maybe_pad_forward(x, fill=-float("inf"))
    out = alloc.empty_like(padded_x)
    _log_softmax_fwd(padded_x, out)
    return _trim_output(out, cols)


def log_softmax_backward(dy: Tensor, log_y: Tensor) -> Tensor:
    dy_pad, log_y_pad, cols = _maybe_pad_backward(dy, log_y, fill=-float("inf"))
    dx_pad = alloc.empty_like(dy_pad)
    _log_softmax_bwd(dy_pad, log_y_pad, dx_pad)
    return _trim_output(dx_pad, cols)


class _CutedslSoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor) -> Tensor:  # type: ignore[override]
        if logits.device.type != "cuda":  # pragma: no cover
            raise RuntimeError("softmax CuTeDSL kernel requires CUDA tensors")
        if logits.ndim == 0:
            raise ValueError("softmax input must have at least one dimension")
        cols = logits.shape[-1]
        rows = logits.numel() // cols if cols > 0 else 0
        logits_2d = logits.contiguous().view(rows, cols)
        result = softmax_forward(logits_2d)
        ctx.save_for_backward(result)
        ctx.shape = logits.shape
        ctx.rows = rows
        ctx.cols = cols
        return result.view(logits.shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:  # type: ignore[override]
        if ctx.rows == 0:
            return (grad_output,)
        (probs,) = ctx.saved_tensors
        grad = grad_output.contiguous().view(ctx.rows, ctx.cols).to(probs.dtype)
        probs_2d = probs.view(ctx.rows, ctx.cols)
        dx = softmax_backward(grad, probs_2d)
        return (dx.view(ctx.shape),)


class _CutedslLogSoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor) -> Tensor:  # type: ignore[override]
        if logits.device.type != "cuda":  # pragma: no cover
            raise RuntimeError("log_softmax CuTeDSL kernel requires CUDA tensors")
        if logits.ndim == 0:
            raise ValueError("log_softmax input must have at least one dimension")
        cols = logits.shape[-1]
        rows = logits.numel() // cols if cols > 0 else 0
        logits_2d = logits.contiguous().view(rows, cols)
        result = log_softmax_forward(logits_2d)
        ctx.save_for_backward(result)
        ctx.shape = logits.shape
        ctx.rows = rows
        ctx.cols = cols
        return result.view(logits.shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:  # type: ignore[override]
        if ctx.rows == 0:
            return (grad_output,)
        (log_probs,) = ctx.saved_tensors
        grad = grad_output.contiguous().view(ctx.rows, ctx.cols).to(log_probs.dtype)
        log_probs_2d = log_probs.view(ctx.rows, ctx.cols)
        dx = log_softmax_backward(grad, log_probs_2d)
        return (dx.view(ctx.shape),)


def softmax(x: Tensor) -> Tensor:
    return _CutedslSoftmaxFn.apply(x)


def log_softmax(x: Tensor) -> Tensor:
    return _CutedslLogSoftmaxFn.apply(x)


__all__ = [
    "softmax",
    "softmax_forward",
    "softmax_backward",
    "log_softmax",
    "log_softmax_forward",
    "log_softmax_backward",
]
