# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute


class ReductionBase:
    """Minimal port of quack.reduction_base without quack dependencies."""

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        stage: int,
        reduction_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    ) -> None:
        self.dtype = dtype
        self.N = N
        self.stage = stage
        self.reduction_dtype = reduction_dtype

    def _calculate_threads_per_row(self) -> int:
        raise NotImplementedError()

    def _set_cluster_n(self) -> None:
        self.cluster_n = 1

    def _get_num_threads(self) -> int:
        return 128 if self.N <= 16384 else 256

    def _get_tv_layout(
        self, num_copy_bits: int = 128
    ) -> Tuple[cute.Shape, cute.Layout]:
        vecsize = num_copy_bits // self.dtype.width
        assert (
            self.N % vecsize == 0
        ), f"Input N {self.N} is not divisible by vector size {vecsize}"
        num_threads = self._get_num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0

        threads_per_row = self._calculate_threads_per_row()
        num_blocks_N = cute.ceil_div(
            self.N // vecsize, threads_per_row * self.cluster_n
        )
        cols_per_block = num_threads // threads_per_row
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
        tv_layout = cute.make_layout(
            ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * threads_per_row),
            ),
        )
        return tiler_mn, tv_layout

    def _smem_size_in_bytes(self, tiler_mn, num_warps):
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn))
            + self.stage
            * num_warps
            * self.cluster_n
            * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8)
        )

    def _get_reduction_buffer_layout(self, tv_layout: cute.Layout, cluster_n: int):
        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        warps_per_row = max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
        return cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, cluster_n), self.stage),
            order=(1, 0, 2),
        )

    def _allocate_reduction_buffer_and_mbar(
        self,
        smem: cutlass.utils.SmemAllocator,
        tv_layout: cute.Layout,
        is_persistent: bool = False,
    ) -> Tuple[cute.Tensor, Optional[cute.Pointer]]:
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype,
            self._get_reduction_buffer_layout(tv_layout, self.cluster_n),
            byte_alignment=4,
        )
        if cutlass.const_expr(self.cluster_n > 1):
            mbar_ptr = smem.allocate_array(
                cutlass.Int64,
                num_elems=self.stage if not is_persistent else self.stage * 2,
            )
        else:
            mbar_ptr = None
        return reduction_buffer, mbar_ptr

    @cute.jit
    def _initialize_cluster(
        self,
        tidx: cutlass.Int32,
        mbar_ptr: cute.Pointer,
        num_warps: int,
        is_persistent: bool = False,
    ) -> None:
        if cutlass.const_expr(self.cluster_n > 1):
            if tidx < self.stage:
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
                if cutlass.const_expr(is_persistent):
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.stage + tidx, num_warps * self.cluster_n
                    )
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
