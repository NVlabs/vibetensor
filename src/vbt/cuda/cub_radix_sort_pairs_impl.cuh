// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if !VBT_WITH_CUDA
#  error "cub_radix_sort_pairs_impl.cuh is CUDA-only"
#endif

#include "cub_detail_common.cuh"
#include "cub_detail_compat.cuh"

#include <cuda_runtime_api.h>

namespace vbt { namespace cuda { namespace cub_detail {

template <class ValueT>
inline void radix_sort_pairs_u64_inplace_impl(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    std::uint64_t* d_keys_inout,
    ValueT* d_vals_inout,
    int n,
    const char* what,
    const char* memcpy_keys_what,
    const char* memcpy_vals_what,
    const char* overlap_msg) {
  vbt::cuda::cub_detail::CubContext ctx(alloc, stream);

  // This wrapper allocates temporary storage and is not capture-safe.
  if (vbt::cuda::streamCaptureStatus(stream) != vbt::cuda::CaptureStatus::None) {
    vbt::cuda::cub_detail::throw_capture_denied(stream.device_index());
  }

  const std::size_t keys_bytes = static_cast<std::size_t>(n) * sizeof(std::uint64_t);
  const std::size_t vals_bytes = static_cast<std::size_t>(n) * sizeof(ValueT);

  vbt::cuda::cub_detail::check_no_overlap_ranges(
      d_keys_inout,
      keys_bytes,
      d_vals_inout,
      vals_bytes,
      overlap_msg);

  // Allocate alternate buffers for the DoubleBuffer API.
  vbt::cuda::cub_detail::StreamDeviceBuffer keys_alt(ctx.alloc, ctx.stream, keys_bytes);
  vbt::cuda::cub_detail::StreamDeviceBuffer vals_alt(ctx.alloc, ctx.stream, vals_bytes);

  auto* d_keys_alt_ptr = static_cast<std::uint64_t*>(keys_alt.data());
  auto* d_vals_alt_ptr = static_cast<ValueT*>(vals_alt.data());

  auto d_keys = vbt::cuda::cub_wrapped::cub::DoubleBuffer<std::uint64_t>(d_keys_inout, d_keys_alt_ptr);
  auto d_vals = vbt::cuda::cub_wrapped::cub::DoubleBuffer<ValueT>(d_vals_inout, d_vals_alt_ptr);

  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  vbt::cuda::cub_detail::call_with_temp_storage_checked(
      alloc,
      stream,
      what,
      [&](void* tmp, std::size_t& temp_bytes) -> cudaError_t {
        return vbt::cuda::cub_wrapped::cub::DeviceRadixSort::SortPairs(
            tmp,
            temp_bytes,
            d_keys,
            d_vals,
            n,
            /*begin_bit=*/0,
            /*end_bit=*/sizeof(std::uint64_t) * 8,
            cu_stream);
      });

  // Ensure results land in the caller-provided inout buffers.
  if (d_keys.Current() != d_keys_inout || d_vals.Current() != d_vals_inout) {
    cudaError_t st = cudaMemcpyAsync(
        d_keys_inout,
        d_keys.Current(),
        keys_bytes,
        cudaMemcpyDeviceToDevice,
        cu_stream);
    vbt::cuda::cub_detail::cudaCheck(st, memcpy_keys_what);

    st = cudaMemcpyAsync(
        d_vals_inout,
        d_vals.Current(),
        vals_bytes,
        cudaMemcpyDeviceToDevice,
        cu_stream);
    vbt::cuda::cub_detail::cudaCheck(st, memcpy_vals_what);
  }
}

}}} // namespace vbt::cuda::cub_detail
