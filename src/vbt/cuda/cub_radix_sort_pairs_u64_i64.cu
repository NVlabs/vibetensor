// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/cub.h"

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA

#include "cub_radix_sort_pairs_impl.cuh"

#include <stdexcept>

namespace vbt { namespace cuda { namespace cub {

void radix_sort_pairs_u64_i64(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    std::uint64_t* d_keys_inout,
    long long* d_vals_inout,
    int n) {
  if (n < 0) {
    throw std::invalid_argument("cub: radix_sort_pairs_u64_i64: n must be >= 0");
  }

  // Accept null pointers for the empty case.
  if (n == 0) {
    vbt::cuda::cub_detail::CubContext ctx(alloc, stream);
    (void)ctx;
    return;
  }

  if (!d_keys_inout) {
    throw std::invalid_argument(
        "cub: radix_sort_pairs_u64_i64: d_keys_inout must be non-null when n > 0");
  }
  if (!d_vals_inout) {
    throw std::invalid_argument(
        "cub: radix_sort_pairs_u64_i64: d_vals_inout must be non-null when n > 0");
  }

  vbt::cuda::cub_detail::radix_sort_pairs_u64_inplace_impl<long long>(
      alloc,
      stream,
      d_keys_inout,
      d_vals_inout,
      n,
      "cub::DeviceRadixSort::SortPairs(u64,i64)",
      "cudaMemcpyAsync(radix_sort_pairs_u64_i64 keys)",
      "cudaMemcpyAsync(radix_sort_pairs_u64_i64 vals)",
      "cub: radix_sort_pairs_u64_i64: d_keys_inout and d_vals_inout must not overlap");
}

}}} // namespace vbt::cuda::cub

#endif // VBT_WITH_CUDA
