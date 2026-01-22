# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Literal, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Boolean, const_expr, Float32, Int32
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


class CrossEntropy(ReductionBase):
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        online_softmax: bool = True,
    ):
        super().__init__(
            dtype,
            N,
            stage=2,
            reduction_dtype=cutlass.Int64 if online_softmax else Float32,
        )
        self.online_softmax = online_softmax
        self.reload_from = None if N <= 16_384 or online_softmax else "smem"

    def _preferred_copy_bits(self) -> int:
        elem_bits = self.dtype.width
        max_vec = max(1, 128 // elem_bits)
        for vec in range(max_vec, 0, -1):
            bits = elem_bits * vec
            if bits in (32, 64, 128) and self.N % vec == 0:
                return bits
        return elem_bits

    def _calculate_threads_per_row(self) -> int:
        N = self.N
        if N <= 64:
            return 8
        if N <= 128:
            return 16
        if N <= 3_072:
            return 32
        if N <= 4_096:
            return 128
        return 256

    def _set_cluster_n(self) -> None:
        N = self.N
        if const_expr(self.dtype.width == 16):
            if N <= 16 * 1024:
                cluster_n = 1
            elif N <= 32 * 1024:
                cluster_n = 2
            elif N <= 64 * 1024:
                cluster_n = 4
            elif N <= 128 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16
        else:  # fp32
            if N <= 16 * 1024:
                cluster_n = 1
            elif N <= 64 * 1024:
                cluster_n = 2
            elif N <= 128 * 1024:
                cluster_n = 4
            elif N <= 256 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16
        self.cluster_n = cluster_n

    def _get_num_threads(self) -> int:
        if self.N >= 4_096:
            return 256
        return super()._get_num_threads()

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mTarget: cute.Tensor,
        mTargetLogit: Optional[cute.Tensor],
        mLoss: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mdX: Optional[cute.Tensor],
        ignore_index: Int32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        if const_expr(mTargetLogit is None):
            mTargetLogit = mX
        self._set_cluster_n()
        num_copy_bits = self._preferred_copy_bits()
        tiler_mn, tv_layout = self._get_tv_layout(num_copy_bits=num_copy_bits)
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        self.kernel(
            mX,
            mTarget,
            mTargetLogit,
            mLoss,
            mLSE,
            mdX,
            ignore_index,
            tv_layout,
            tiler_mn,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=(
                [1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None
            ),
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mTarget: cute.Tensor,
        mTargetLogit: cute.Tensor,
        mLoss: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mdX: Optional[cute.Tensor],
        ignore_index: Int32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        mX_off = utils.domain_offset_i64((bidx * tiler_mn[0], 0), mX)
        gX = cute.local_tile(mX_off, tiler_mn, (0, cluster_y))
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

        num_copy_elems_X = tv_layout.shape[1][0]
        num_copy_bits_X = mX.element_type.width * num_copy_elems_X
        use_cp_async_fwd = num_copy_bits_X in (32, 64, 128)
        copy_atom_load_X = cute.make_copy_atom(
            (
                cute.nvgpu.cpasync.CopyG2SOp()
                if use_cp_async_fwd
                else cute.nvgpu.CopyUniversalOp()
            ),
            gX.element_type,
            num_bits_per_copy=num_copy_bits_X,
        )
        thr_copy_X = cute.make_tiled_copy(
            copy_atom_load_X, tv_layout, tiler_mn
        ).get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_fragment_like(tXgX)

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        row = tXcX[0][0]
        target = Int32.zero
        if row < shape[0]:
            target = Int32(mTarget[row])

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if const_expr(not is_even_N)
            else None
        )
        if row < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXsX, pred=tXpX)
        if const_expr(use_cp_async_fwd):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        if const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        target_logit = Float32.zero
        should_ignore = Boolean(target == ignore_index)
        if row < shape[0] and tXcX[0][1] == 0 and not should_ignore:
            if const_expr(cute.rank(mTargetLogit.shape) == 2):
                mTargetLogit_off = utils.domain_offset_i64((row, 0), mTargetLogit)
                target_logit = Float32(mTargetLogit_off[0, target])
            else:
                target_logit = Float32(mTargetLogit[row])

        threads_per_row = tv_layout.shape[0][0]
        if const_expr(not self.online_softmax):
            max_x = row_reduce(
                x,
                cute.ReductionOp.MAX,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
                init_val=-Float32.inf,
                hook_fn=(
                    cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None
                ),
            )
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                x = tXrX.load().to(Float32)
            log2_e = math.log2(math.e)
            exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=False)
            denom = row_reduce(
                exp_x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 1],
                mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
            )
        else:
            max_x, denom, exp_x = online_softmax_reduce(
                x,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                hook_fn=(
                    cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None
                ),
                return_exp_x=const_expr(mdX is not None),
            )

        if (
            tXcX[0][1] == 0
            and row < shape[0]
            and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
        ):
            lse = max_x + cute.math.log(denom, fastmath=True)
            loss_val = (lse - target_logit) if not should_ignore else Float32.zero
            mLoss[row] = mLoss.element_type(loss_val)
            if const_expr(mLSE is not None):
                mLSE[row] = lse

        if const_expr(mdX is not None):
            denom_inv = (
                cute.arch.rcp_approx(denom)
                if not (denom == 0.0 or denom != denom or should_ignore)
                else Float32.zero
            )
            probs = exp_x * denom_inv
            mdX_off = utils.domain_offset_i64((bidx * tiler_mn[0], 0), mdX)
            gdX = cute.local_tile(mdX_off, tiler_mn, (0, cluster_y))
            copy_atom_store = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                mdX.element_type,
                num_bits_per_copy=num_copy_bits_X,
            )
            thr_copy_dX = cute.make_tiled_copy(
                copy_atom_store, tv_layout, tiler_mn
            ).get_slice(tidx)
            tXgdX = thr_copy_dX.partition_D(gdX)
            tXrdX = cute.make_fragment_like(tXgdX)
            tXcFull = thr_copy_X.partition_S(cX)

            probs_frag = cute.make_fragment_like(tXrX, Float32)
            probs_frag.store(probs)
            if not should_ignore:
                for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                    probs_frag[i] = (
                        probs_frag[i]
                        if tXcFull[i][1] != target
                        else probs_frag[i] - 1.0
                    )
            tXrdX.store(probs_frag.load().to(tXrdX.element_type))
            tXpdX = (
                utils.predicate_k(thr_copy_dX.partition_S(cX), limit=shape[1])
                if not is_even_N
                else None
            )
            if row < shape[0]:
                cute.copy(copy_atom_store, tXrdX, tXgdX, pred=tXpdX)


class CrossEntropyBackward:
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        self.dtype = dtype
        self.N = N
        self.vecsize = 128 // dtype.width

    def _preferred_copy_bits(self) -> int:
        elem_bits = self.dtype.width
        max_vec = max(1, 128 // elem_bits)
        for vec in range(max_vec, 0, -1):
            bits = elem_bits * vec
            if bits in (32, 64, 128) and self.N % vec == 0:
                return bits
        return elem_bits

    def _calculate_threads_per_row(self) -> int:
        N = min(self.N, 16_384)
        if N <= 64:
            return 8
        if N <= 128:
            return 16
        if N <= 3_072:
            return 32
        if N <= 6_144:
            return 64
        if N <= 16_384:
            return 128
        return 256

    def _get_tv_layout(self, num_copy_bits=128):
        vecsize = num_copy_bits // self.dtype.width
        assert (
            self.N % vecsize == 0
        ), f"Input N {self.N} is not divisible by vector size {vecsize}"
        N = min(self.N, 16_384)
        num_threads = 128 if N <= 16_384 else 256
        threads_per_row = self._calculate_threads_per_row()
        cols_per_block = num_threads // threads_per_row
        num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row)
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
        tv_layout = cute.make_layout(
            ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * threads_per_row),
            ),
        )
        return tiler_mn, tv_layout

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mTarget: cute.Tensor,
        mDLoss: cute.Tensor,
        mdX: cute.Tensor,
        mLSE: cute.Tensor,
        ignore_index: Int32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mdX.element_type == self.dtype
        num_copy_bits = self._preferred_copy_bits()
        tiler_mn, tv_layout = self._get_tv_layout(num_copy_bits=num_copy_bits)
        num_threads = cute.size(tv_layout, mode=[0])
        mDLoss, mTarget, mLSE = [
            cute.make_tensor(
                X.iterator,
                cute.append(X.layout, cute.make_layout((self.N,), stride=(0,))),
            )
            for X in (mDLoss, mTarget, mLSE)
        ]
        smem_size = cute.size_in_bytes(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
        )
        self.kernel(
            mX,
            mTarget,
            mDLoss,
            mdX,
            mLSE,
            ignore_index,
            mX.shape,
            tv_layout,
            tiler_mn,
        ).launch(
            grid=[
                cute.ceil_div(mX.shape[0], tiler_mn[0]),
                cute.ceil_div(mX.shape[1], tiler_mn[1]),
                1,
            ],
            block=[num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mTarget: cute.Tensor,
        mDLoss: cute.Tensor,
        mdX: cute.Tensor,
        mLSE: cute.Tensor,
        ignore_index: Int32,
        shape: cute.Shape,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        idX = cute.make_identity_tensor(shape)
        mX, mdX = [
            utils.domain_offset_i64((bidx * tiler_mn[0], 0), mT) for mT in (mX, mdX)
        ]
        gX, gdX = [cute.local_tile(mT, tiler_mn, (0, bidy)) for mT in (mX, mdX)]
        cX = cute.local_tile(idX, tiler_mn, (bidx, bidy))

        num_copy_elems_X = tv_layout.shape[1][0]
        num_copy_bits_X = mX.element_type.width * num_copy_elems_X
        use_cp_async_backward = num_copy_bits_X in (32, 64, 128)
        copy_atom_load_X = cute.make_copy_atom(
            (
                cute.nvgpu.cpasync.CopyG2SOp()
                if use_cp_async_backward
                else cute.nvgpu.CopyUniversalOp()
            ),
            gX.element_type,
            num_bits_per_copy=num_copy_bits_X,
        )
        copy_atom_store_dX = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            gdX.element_type,
            num_bits_per_copy=num_copy_bits_X,
        )
        thr_copy_X = cute.make_tiled_copy(
            copy_atom_load_X, tv_layout, tiler_mn
        ).get_slice(tidx)
        thr_copy_dX = cute.make_tiled_copy(
            copy_atom_store_dX, tv_layout, tiler_mn
        ).get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_S(sX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]
        tXcFull = thr_copy_X.partition_S(cX)
        tXgdX = thr_copy_dX.partition_D(gdX)
        tXrX, tXrdX = [cute.make_fragment_like(thr) for thr in (tXgX, tXgdX)]

        is_even_N = const_expr(shape[1] % tiler_mn[1] == 0)
        row = tXcX[0][0]
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        if row < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXsX, pred=tXpX)
        if const_expr(use_cp_async_backward):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        if const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        target = Int32.zero
        dloss = Float32.zero
        lse = Float32.zero
        if row < shape[0]:
            target = Int32(mTarget[row])
            should_ignore = Boolean(target == ignore_index)
            dloss = Float32(mDLoss[row]) if not should_ignore else Float32.zero
            lse = Float32(mLSE[row])
        else:
            should_ignore = Boolean(True)

        log2_e = math.log2(math.e)
        probs = cute.math.exp2(x * log2_e - (lse * log2_e), fastmath=True)
        prob_shifted = probs - 1.0
        mask = cute.make_fragment_like(tXrX, cutlass.Boolean)
        for i in cutlass.range(cute.size(tXcFull), unroll_full=True):
            mask[i] = tXcFull[i][1] == target
        grad = cute.where(mask.load(), prob_shifted, probs)
        grad = grad * dloss

        tXrdX.store(grad.to(tXrdX.element_type))
        tXpdX = (
            utils.predicate_k(thr_copy_dX.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        if row < shape[0]:
            cute.copy(copy_atom_store_dX, tXrdX, tXgdX, pred=tXpdX)


@torch.library.custom_op(
    "kernel_factory::_cutedsl_cross_entropy_fwd_out",
    mutates_args={"loss", "lse", "dx"},
)
def _cross_entropy_fwd_out(
    x: Tensor,
    target: Tensor,
    target_logit: Optional[Tensor],
    loss: Tensor,
    lse: Optional[Tensor],
    dx: Optional[Tensor],
    ignore_index: int = -100,
) -> None:
    assert x.dim() == 2, "Input must be 2D"
    assert target.dim() == 1, "Target must be 1D"
    assert x.shape[0] == target.shape[0], "Batch dimensions must match"
    assert x.is_cuda and target.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert target.dtype in [torch.int32, torch.int64]
    if target_logit is not None:
        assert target_logit.shape[0] == x.shape[0]
        assert target_logit.is_cuda
        assert target_logit.dtype in [torch.float16, torch.bfloat16, torch.float32]
    if dx is not None:
        assert dx.shape == x.shape
        assert dx.is_cuda
        assert dx.dtype == x.dtype
    N = x.size(1)
    dtype = torch2cute_dtype_map[x.dtype]

    def convert(t: Tensor) -> cute.Tensor:
        return from_dlpack(t.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

    x_tensor = convert(x)
    loss_tensor = from_dlpack(loss.detach(), assumed_align=4).mark_layout_dynamic()
    lse_tensor = (
        from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic()
        if lse is not None
        else None
    )
    target_tensor = from_dlpack(target.detach(), assumed_align=8).mark_layout_dynamic()
    target_logit_tensor = (
        from_dlpack(target_logit.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=target_logit.ndim - 1
        )
        if target_logit is not None
        else None
    )
    dx_tensor = convert(dx) if dx is not None else None
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (
        dtype,
        N,
        target_logit.dtype if target_logit is not None else None,
        lse.dtype if lse is not None else None,
        dx is not None,
        loss.stride(),
        lse.stride() if lse is not None else None,
        target.stride(),
        target_logit.stride(-1) if target_logit is not None else None,
    )
    if compile_key not in _cross_entropy_fwd_out.compile_cache:
        cross_entropy_op = CrossEntropy(dtype, N, online_softmax=dx is None)
        _cross_entropy_fwd_out.compile_cache[compile_key] = cute.compile(
            cross_entropy_op,
            x_tensor,
            target_tensor,
            target_logit_tensor,
            loss_tensor,
            lse_tensor,
            dx_tensor,
            Int32(ignore_index),
            stream,
        )
    _cross_entropy_fwd_out.compile_cache[compile_key](
        x_tensor,
        target_tensor,
        target_logit_tensor,
        loss_tensor,
        lse_tensor,
        dx_tensor,
        Int32(ignore_index),
        stream,
    )


_cross_entropy_fwd_out.compile_cache = {}


def cross_entropy_forward(
    x: TensorLike,
    target: TensorLike,
    target_logit: Optional[TensorLike] = None,
    *,
    ignore_index: int = -100,
    return_lse: bool = False,
    return_dx: bool = False,
    inplace_backward: bool = False,
) -> tuple[TensorLike, Optional[TensorLike], Optional[TensorLike]]:
    M = x.size(0)
    loss = alloc.empty((M,), like=x, dtype="float32")
    lse = alloc.empty((M,), like=x, dtype="float32") if return_lse else None
    dx = (
        x
        if inplace_backward and return_dx
        else alloc.empty_like(x) if return_dx else None
    )
    _cross_entropy_fwd_out(x, target, target_logit, loss, lse, dx, ignore_index)
    return loss, lse, dx


def cross_entropy_backward(
    x: TensorLike,
    target: TensorLike,
    dloss: TensorLike,
    lse: TensorLike,
    dx: TensorLike,
    *,
    ignore_index: int = -100,
) -> None:
    N = x.size(1)
    dtype = torch2cute_dtype_map[x.dtype]

    def convert(t: TensorLike) -> cute.Tensor:
        return from_dlpack(t.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

    x_tensor = convert(x)
    dx_tensor = convert(dx)
    dloss_tensor = from_dlpack(dloss.detach(), assumed_align=4).mark_layout_dynamic()
    lse_tensor = from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic()
    target_tensor = from_dlpack(target.detach(), assumed_align=8).mark_layout_dynamic()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (
        dtype,
        N,
        target.dtype,
        dloss.stride(),
        lse.stride(),
        target.stride(),
    )
    if compile_key not in cross_entropy_backward.compile_cache:
        op = CrossEntropyBackward(dtype, N)
        cross_entropy_backward.compile_cache[compile_key] = cute.compile(
            op,
            x_tensor,
            target_tensor,
            dloss_tensor,
            dx_tensor,
            lse_tensor,
            Int32(ignore_index),
            stream,
        )
    cross_entropy_backward.compile_cache[compile_key](
        x_tensor,
        target_tensor,
        dloss_tensor,
        dx_tensor,
        lse_tensor,
        Int32(ignore_index),
        stream,
    )


cross_entropy_backward.compile_cache = {}


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: TensorLike,
        target: TensorLike,
        lse_partial: Optional[TensorLike] = None,
        ignore_index: int = -100,
        inplace_backward: bool = False,
    ):
        if lse_partial is None:
            loss, lse = cross_entropy_forward(
                x,
                target,
                ignore_index=ignore_index,
                return_lse=True,
                return_dx=False,
            )
        else:
            loss, lse = cross_entropy_forward(
                lse_partial,
                target,
                target_logit=x,
                ignore_index=ignore_index,
                return_lse=True,
                return_dx=False,
            )
        ctx.save_for_backward(x, target, lse)
        ctx.ignore_index = ignore_index
        ctx.inplace_backward = inplace_backward
        return loss

    @staticmethod
    def backward(ctx, dloss: TensorLike):
        x, target, lse = ctx.saved_tensors
        if ctx.inplace_backward and not torch.compiler.is_compiling():
            dx = x
        else:
            dx = alloc.empty_like(x)
        cross_entropy_backward(
            x,
            target,
            dloss,
            lse,
            dx,
            ignore_index=ctx.ignore_index,
        )
        return dx, None, None, None, None


def cross_entropy(
    x: TensorLike,
    target: TensorLike,
    lse_partial: Optional[TensorLike] = None,
    *,
    ignore_index: int = -100,
    reduction: Literal["none", "mean", "sum"] = "mean",
    inplace_backward: bool = False,
):
    loss = CrossEntropyFunction.apply(
        x, target, lse_partial, ignore_index, inplace_backward
    )
    if reduction == "mean":
        return loss.sum() / (target != ignore_index).sum().float()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(
        f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', or 'sum'"
    )
