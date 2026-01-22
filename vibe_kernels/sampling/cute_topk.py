# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Standalone CuTe DSL Top-K Kernel
# Based on QuACK implementation but self-contained
# Copyright note: Adapted from QuACK by Wentao Guo, Mayank Mishra, Tri Dao

import math
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass.cute.runtime import from_dlpack


# ============================================================================
# Utility Functions
# ============================================================================


def torch2cute_dtype_map_fn(dtype):
    """Map torch dtype to cutlass dtype."""
    mapping = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    return mapping[dtype]


@cute.jit
def fmin(a, b):
    """Float minimum using NVVM intrinsic."""
    from cutlass._mlir.dialects import nvvm
    from cutlass.cutlass_dsl import T

    return cutlass.Float32(
        nvvm.fmin(
            T.f32(),
            cutlass.Float32(a).ir_value(),
            cutlass.Float32(b).ir_value(),
        )
    )


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    """Create predicates for out-of-bounds checking."""
    tApA = cute.make_fragment(
        cute.make_layout(
            (
                cute.size(tAcA, mode=[0, 1]),
                cute.size(tAcA, mode=[1]),
                cute.size(tAcA, mode=[2]),
            ),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(
                tAcA[(0, rest_v), 0, rest_k][1], limit
            )
    return tApA


@cute.jit
def fill_oob(tXsX: cute.Tensor, tXpX, fill_value) -> None:
    """Fill out-of-bounds values."""
    tXrX_fill = cute.make_fragment_like(tXsX[(None, 0), None, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
            if const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])


from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def domain_offset_i64(
    coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None
) -> cute.Tensor:
    """Apply offset to tensor for large indices (> 2^31)."""
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(flat_stride)
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    assert isinstance(tensor.iterator, cute.Pointer)
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


# ============================================================================
# Sorting Network Functions
# ============================================================================


@cute.jit
def compare_and_swap(arr: cute.Tensor, i: int, j: int, ascending: bool = True) -> None:
    """Compare and swap elements at indices i and j."""
    min_fn = min if cutlass.const_expr(arr.element_type != cutlass.Float32) else fmin
    max_fn = (
        max
        if cutlass.const_expr(arr.element_type != cutlass.Float32)
        else cute.arch.fmax
    )
    if cutlass.const_expr(ascending):
        arr[i], arr[j] = min_fn(arr[i], arr[j]), max_fn(arr[i], arr[j])
    else:
        arr[i], arr[j] = max_fn(arr[i], arr[j]), min_fn(arr[i], arr[j])


# Optimal sorting networks for small sizes
@cute.jit
def optimal_sort_2(arr: cute.Tensor, start: int, ascending: bool):
    compare_and_swap(arr, start + 0, start + 1, ascending)


@cute.jit
def optimal_sort_4(arr: cute.Tensor, start: int, ascending: bool):
    compare_and_swap(arr, start + 0, start + 2, ascending)
    compare_and_swap(arr, start + 1, start + 3, ascending)
    compare_and_swap(arr, start + 0, start + 1, ascending)
    compare_and_swap(arr, start + 2, start + 3, ascending)
    compare_and_swap(arr, start + 1, start + 2, ascending)


@cute.jit
def optimal_sort_8(arr: cute.Tensor, start: int, ascending: bool):
    # Layer 1
    compare_and_swap(arr, start + 0, start + 2, ascending)
    compare_and_swap(arr, start + 1, start + 3, ascending)
    compare_and_swap(arr, start + 4, start + 6, ascending)
    compare_and_swap(arr, start + 5, start + 7, ascending)
    # Layer 2
    compare_and_swap(arr, start + 0, start + 4, ascending)
    compare_and_swap(arr, start + 1, start + 5, ascending)
    compare_and_swap(arr, start + 2, start + 6, ascending)
    compare_and_swap(arr, start + 3, start + 7, ascending)
    # Layer 3
    compare_and_swap(arr, start + 0, start + 1, ascending)
    compare_and_swap(arr, start + 2, start + 3, ascending)
    compare_and_swap(arr, start + 4, start + 5, ascending)
    compare_and_swap(arr, start + 6, start + 7, ascending)
    # Layer 4
    compare_and_swap(arr, start + 2, start + 4, ascending)
    compare_and_swap(arr, start + 3, start + 5, ascending)
    # Layer 5
    compare_and_swap(arr, start + 1, start + 4, ascending)
    compare_and_swap(arr, start + 3, start + 6, ascending)
    # Layer 6
    compare_and_swap(arr, start + 1, start + 2, ascending)
    compare_and_swap(arr, start + 3, start + 4, ascending)
    compare_and_swap(arr, start + 5, start + 6, ascending)


@cute.jit
def optimal_sort_16(arr: cute.Tensor, start: int, ascending: bool):
    """Optimal 16-element sorting network (60 comparators, depth 10)."""
    s = start
    # Layer 1
    for i, j in [(0, 13), (1, 12), (2, 15), (3, 14), (4, 8), (5, 6), (7, 11), (9, 10)]:
        compare_and_swap(arr, s + i, s + j, ascending)
    # Layer 2
    for i, j in [(0, 5), (1, 7), (2, 9), (3, 4), (6, 13), (8, 14), (10, 15), (11, 12)]:
        compare_and_swap(arr, s + i, s + j, ascending)
    # Layer 3
    for i, j in [(0, 1), (2, 3), (4, 5), (6, 8), (7, 9), (10, 11), (12, 13), (14, 15)]:
        compare_and_swap(arr, s + i, s + j, ascending)
    # Layer 4
    for i, j in [(0, 2), (1, 3), (4, 10), (5, 11), (6, 7), (8, 9), (12, 14), (13, 15)]:
        compare_and_swap(arr, s + i, s + j, ascending)
    # Layer 5
    for i, j in [(1, 2), (3, 12), (4, 6), (5, 7), (8, 10), (9, 11), (13, 14)]:
        compare_and_swap(arr, s + i, s + j, ascending)
    # Layer 6
    for i, j in [(1, 4), (2, 6), (5, 8), (7, 10), (9, 13), (11, 14)]:
        compare_and_swap(arr, s + i, s + j, ascending)
    # Layer 7
    for i, j in [(2, 4), (3, 6), (9, 12), (11, 13)]:
        compare_and_swap(arr, s + i, s + j, ascending)
    # Layer 8
    for i, j in [(3, 5), (6, 8), (7, 9), (10, 12)]:
        compare_and_swap(arr, s + i, s + j, ascending)
    # Layer 9
    for i, j in [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]:
        compare_and_swap(arr, s + i, s + j, ascending)
    # Layer 10
    for i, j in [(6, 7), (8, 9)]:
        compare_and_swap(arr, s + i, s + j, ascending)


@cute.jit
def optimal_sort(arr: cute.Tensor, n: int, start: int, ascending: bool):
    """Dispatch to optimal sorting network based on size."""
    if cutlass.const_expr(n == 2):
        optimal_sort_2(arr, start, ascending)
    elif cutlass.const_expr(n == 4):
        optimal_sort_4(arr, start, ascending)
    elif cutlass.const_expr(n == 8):
        optimal_sort_8(arr, start, ascending)
    elif cutlass.const_expr(n == 16):
        optimal_sort_16(arr, start, ascending)
    # Add more sizes as needed


# ============================================================================
# Bitonic Sort Implementation
# ============================================================================


@cute.jit
def bitonic_merge(arr: cute.Tensor, n: int, start: int, ascending: bool = True) -> None:
    """Merge a bitonic sequence into a sorted sequence."""
    if cutlass.const_expr(n > 1):
        num_levels = int(math.log2(n))
        assert n == 2**num_levels, "n must be a power of 2"
        for level in cutlass.range_constexpr(num_levels):
            length = n >> level
            step = length // 2
            for i in cutlass.range(n // length, unroll_full=True):
                start_i = start + i * length
                for j in cutlass.range(step, unroll_full=True):
                    compare_and_swap(arr, start_i + j, start_i + j + step, ascending)


@cute.jit
def bitonic_sort(
    arr: cute.Tensor, n: int = None, start: int = 0, ascending: bool = True
) -> None:
    """Bitonic sort for arrays of size <= 128."""
    if cutlass.const_expr(n is None):
        n = cute.size(arr.shape)
    assert n <= 128
    if cutlass.const_expr(n > 1):
        if cutlass.const_expr(n in [2, 4, 8, 16]):
            optimal_sort(arr, n, start, ascending)
        else:
            assert n % 2 == 0
            bitonic_sort(arr, n // 2, start, True)
            bitonic_sort(arr, n // 2, start + n // 2, False)
            bitonic_merge(arr, n, start, ascending)


@cute.jit
def bitonic_topk_merge(
    arr0: cute.Tensor,
    arr1: cute.Tensor,
    k: int = None,
    start0: int = 0,
    start1: int = 0,
    ascending: bool = False,
) -> None:
    """Merge two sorted top-k sequences."""
    if cutlass.const_expr(k is None):
        k = cute.size(arr0.shape)
    if cutlass.const_expr(arr0.element_type == cutlass.Float32):
        minmax_fn = fmin if ascending else cute.arch.fmax
    else:
        minmax_fn = min if ascending else max

    for i in cutlass.range(k, unroll_full=True):
        arr0[start0 + i] = minmax_fn(arr0[start0 + i], arr1[start1 + k - 1 - i])

    bitonic_merge(arr0, k, start0, ascending)


@cute.jit
def bitonic_topk(
    arr: cute.Tensor, k: int, ascending: bool = False, warp_width: int = 32
) -> cute.Tensor:
    """
    Bitonic top-k selection.

    Args:
        arr: Input array (flat)
        k: Number of top elements (must be power of 2, <= 128)
        ascending: Sort in ascending order (False = top-k largest)
        warp_width: Warp size for cross-thread reduction

    Returns:
        Fragment containing top-k values
    """
    assert arr.element_type in [cutlass.Float32, cutlass.Int32]
    n = cute.size(arr.shape)
    assert k == 1 << int(math.log2(k)), "k must be a power of 2"
    assert n % k == 0, "n must be divisible by k"

    # Initialize top-k with first k elements
    topk_vals = cute.make_fragment(k, arr.element_type)
    for v in cutlass.range(k, unroll_full=True):
        topk_vals[v] = arr[v]
    bitonic_sort(topk_vals, k, 0, ascending)

    # Process remaining chunks
    other_vals = cute.make_fragment(k, arr.element_type)
    for i in cutlass.range(1, n // k, unroll_full=True):
        for v in cutlass.range(k, unroll_full=True):
            other_vals[v] = arr[i * k + v]
        bitonic_sort(other_vals, k, 0, ascending)
        bitonic_topk_merge(topk_vals, other_vals, k, 0, 0, ascending)

    # Cross-thread reduction using warp shuffles
    for i in cutlass.range(int(math.log2(warp_width)), unroll_full=True):
        other_vals = cute.make_fragment(k, arr.element_type)
        for v in cutlass.range(k, unroll_full=True):
            other_vals[v] = cute.arch.shuffle_sync_bfly(topk_vals[v], offset=1 << i)
        bitonic_topk_merge(topk_vals, other_vals, k, 0, 0, ascending)

    return topk_vals


# ============================================================================
# Top-K Kernel
# ============================================================================


class TopK:
    """CuTe DSL Top-K kernel."""

    def __init__(self, dtype: Type[cutlass.Numeric], N: int, k: int):
        self.dtype = dtype
        self.N = N
        self.vecsize = 128 // dtype.width
        self.k = k
        assert N == 2 ** int(math.log2(N)), "N must be a power of 2"
        assert k == 2 ** int(math.log2(k)), "k must be a power of 2"
        assert k <= 128, "k must be <= 128"
        assert N <= 4096, "N must be <= 4096"

    def _calculate_threads_per_row(self):
        """Calculate how many threads work on each row."""
        N = self.N
        # Ensure num_elems_per_thread >= k and each thread handles <= 64 elements
        num_threads_per_row = max(min(N // self.k, 32, N // 64), 1)
        return num_threads_per_row

    def _get_tv_layout(self):
        """Get thread-value layout for tiling."""
        N = self.N
        vecsize = self.vecsize
        num_threads = 128 if N <= 16384 else 256
        threads_per_row = self._calculate_threads_per_row()
        cols_per_block = num_threads // threads_per_row
        num_blocks_N = cute.ceil_div(min(N, 16384) // vecsize, threads_per_row)
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
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mValues.element_type == self.dtype
        assert mIndices.element_type == cutlass.Int32
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        self.kernel(mX, mValues, mIndices, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), 1, 1],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mValues: cute.Tensor,
        mIndices: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)

        # Slice for this CTA
        mX = domain_offset_i64((bidx * tiler_mn[0], 0), mX)
        gX = cute.local_tile(mX, tiler_mn, (0, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        # Setup copy atom
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=128
        )
        thr_copy_X = cute.make_tiled_copy(
            copy_atom_load_X, tv_layout, tiler_mn
        ).get_slice(tidx)
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # Allocate fragment for gmem->rmem
        tXrX = cute.make_fragment_like(tXgX)

        is_even_N = const_expr(shape[1] == tiler_mn[1])
        tXpX = (
            predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )

        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)

        # Convert to FP32 for sorting
        tXrX_f32 = cute.make_fragment(tXrX.shape, cutlass.Float32)
        tXrX_f32.store(tXrX.load().to(cutlass.Float32))

        # Encode indices into the bottom bits of values (packed key trick)
        log_N = int(math.log2(self.N))
        idx_mask = (1 << log_N) - 1
        vecsize = cutlass.const_expr(tv_layout.shape[1][0])
        tXrX_u32 = cute.recast_tensor(tXrX_f32, cutlass.Uint32)

        for i in cutlass.range(cute.size(tXrX_u32), unroll_full=True):
            col_idx = cutlass.Uint32(tXcX[i // vecsize][1] + i % vecsize)
            # Invert bits for positive values for stable tie-breaking
            encoded_idx = ~col_idx if tXrX_f32[i] >= 0 else col_idx
            encoded_idx = encoded_idx & idx_mask
            # Pack: clear bottom log_N bits and insert encoded index
            tXrX_u32[i] = (tXrX_u32[i] & ~idx_mask) | encoded_idx

        # Fill OOB with -inf
        if const_expr(not is_even_N):
            fill_oob(tXrX_f32, tXpX, -tXrX_f32.element_type.inf)

        # Run bitonic top-k
        threads_per_row = tv_layout.shape[0][0]
        topk_vals = bitonic_topk(tXrX_f32, self.k, warp_width=threads_per_row)

        # Extract indices and clean values
        topk_vals_u32 = cute.recast_tensor(topk_vals, cutlass.Uint32)
        topk_indices = cute.make_fragment(self.k, cutlass.Int32)
        for i in cutlass.range(self.k):
            encoded_idx = topk_vals_u32[i] & idx_mask
            topk_vals_u32[i] = topk_vals_u32[i] & ~idx_mask  # Clear bottom bits
            # Restore original index
            col_idx = ~encoded_idx if topk_vals[i] >= 0 else encoded_idx
            topk_indices[i] = cutlass.Int32(col_idx & idx_mask)

        # Convert to output type
        topk_vals_out = cute.make_fragment_like(topk_vals, mValues.element_type)
        topk_vals_out.store(topk_vals.load().to(mValues.element_type))

        row = tXcX[0][0]
        # Only 1st thread in row writes output
        if row < shape[0] and tXcX[0][1] == 0:
            # Vectorized write
            elems_per_store = const_expr(math.gcd(vecsize, self.k))
            mValues_store = cute.tiled_divide(mValues[row, None], (elems_per_store,))
            mIndices_store = cute.tiled_divide(mIndices[row, None], (elems_per_store,))
            topk_vals_out_store = cute.tiled_divide(topk_vals_out, (elems_per_store,))
            topk_indices_store = cute.tiled_divide(topk_indices, (elems_per_store,))
            for i in cutlass.range(
                cute.size(topk_vals_out_store.shape, [1]), unroll_full=True
            ):
                cute.autovec_copy(topk_vals_out_store[None, i], mValues_store[None, i])
                cute.autovec_copy(topk_indices_store[None, i], mIndices_store[None, i])


# ============================================================================
# PyTorch Interface
# ============================================================================


@torch.library.custom_op("cute_topk::_topk_fwd", mutates_args={"values", "indices"})
def _topk_fwd(
    x: torch.Tensor, k: int, values: torch.Tensor, indices: torch.Tensor
) -> None:
    """Top-k forward pass using CuTe DSL."""
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Tensor must be on CUDA device"
    assert x.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ], "Unsupported dtype"
    assert k > 0 and k <= x.shape[1], "k must be positive and <= N"

    N = x.size(1)
    dtype = torch2cute_dtype_map_fn(x.dtype)

    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )

    x_tensor, values_tensor, indices_tensor = [
        convert_from_dlpack(tensor) for tensor in (x, values, indices)
    ]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, N, k)

    if compile_key not in _topk_fwd.compile_cache:
        topk_op = TopK(dtype, N, k)
        _topk_fwd.compile_cache[compile_key] = cute.compile(
            topk_op, x_tensor, values_tensor, indices_tensor, current_stream
        )
    _topk_fwd.compile_cache[compile_key](
        x_tensor, values_tensor, indices_tensor, current_stream
    )


_topk_fwd.compile_cache = {}


def topk(x: torch.Tensor, k: int):
    """
    Top-k operation using CuTe DSL.

    Args:
        x: Input tensor of shape (M, N) - N and k must be power of 2
        k: Number of top elements to return (must be <= 128)

    Returns:
        Tuple of (values, indices) each of shape (M, k)
    """
    M = x.size(0)
    values = torch.empty((M, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=x.device)
    _topk_fwd(x, k, values, indices)
    return values, indices


if __name__ == "__main__":
    # Test
    torch.manual_seed(42)
    M, N, k = 4096, 1024, 32
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    # Our implementation
    vals, idx = topk(x, k)

    # PyTorch reference
    vals_ref, idx_ref = torch.topk(x, k, dim=-1, largest=True, sorted=True)

    # Check correctness
    print(f"Shape: {vals.shape}")
    print(f"Max diff: {(vals.float() - vals_ref.float()).abs().max().item():.6f}")
    print(
        f"Values match: {torch.allclose(vals.float(), vals_ref.float(), rtol=1e-3, atol=1e-3)}"
    )
    print(f"Indices match: {torch.equal(idx, idx_ref.int())}")
