// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/cub.h"

#include <stdexcept>
#include <string>

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

namespace vbt { namespace cuda { namespace cub {

#if !VBT_WITH_CUDA

namespace {
[[noreturn]] void throw_no_cuda(const char* what) {
  throw std::runtime_error(std::string(what) + ": CUDA is not enabled");
}
} // anonymous

void select_indices_from_u8_flags_i64(
    vbt::cuda::Allocator& /*alloc*/, vbt::cuda::Stream /*stream*/, const std::uint8_t* /*d_flags01*/, int /*n*/,
    std::int64_t* /*d_out_indices*/, int* /*d_num_selected_out*/) {
  throw_no_cuda("cub::select_indices_from_u8_flags_i64");
}

void reduce_sum_u8_as_i32(
    vbt::cuda::Allocator& /*alloc*/, vbt::cuda::Stream /*stream*/, const std::uint8_t* /*d_flags01*/, int /*n*/,
    int* /*d_out_sum*/) {
  throw_no_cuda("cub::reduce_sum_u8_as_i32");
}

void reduce_all_contig_sum_f32(
    vbt::cuda::Allocator& /*alloc*/, vbt::cuda::Stream /*stream*/, const float* /*d_in*/, int /*n*/, float* /*d_out*/) {
  throw_no_cuda("cub::reduce_all_contig_sum_f32");
}

void reduce_all_contig_sum_i64(
    vbt::cuda::Allocator& /*alloc*/, vbt::cuda::Stream /*stream*/, const long long* /*d_in*/, int /*n*/, long long* /*d_out*/) {
  throw_no_cuda("cub::reduce_all_contig_sum_i64");
}

void exclusive_scan_i32(
    vbt::cuda::Allocator& /*alloc*/, vbt::cuda::Stream /*stream*/, const int* /*d_in*/, int /*n*/, int* /*d_out*/) {
  throw_no_cuda("cub::exclusive_scan_i32");
}

void radix_sort_pairs_u64_f32(
    vbt::cuda::Allocator& /*alloc*/, vbt::cuda::Stream /*stream*/, std::uint64_t* /*d_keys_inout*/,
    float* /*d_vals_inout*/, int /*n*/) {
  throw_no_cuda("cub::radix_sort_pairs_u64_f32");
}

void radix_sort_pairs_u64_i64(
    vbt::cuda::Allocator& /*alloc*/, vbt::cuda::Stream /*stream*/, std::uint64_t* /*d_keys_inout*/,
    long long* /*d_vals_inout*/, int /*n*/) {
  throw_no_cuda("cub::radix_sort_pairs_u64_i64");
}

void reduce_by_key_sum_u64_f32(
    vbt::cuda::Allocator& /*alloc*/, vbt::cuda::Stream /*stream*/, const std::uint64_t* /*d_keys_in*/,
    const float* /*d_vals_in*/, int /*n*/, std::uint64_t* /*d_unique_keys_out*/, float* /*d_sums_out*/,
    int* /*d_num_runs_out*/) {
  throw_no_cuda("cub::reduce_by_key_sum_u64_f32");
}

void reduce_by_key_sum_u64_i64(
    vbt::cuda::Allocator& /*alloc*/, vbt::cuda::Stream /*stream*/, const std::uint64_t* /*d_keys_in*/,
    const long long* /*d_vals_in*/, int /*n*/, std::uint64_t* /*d_unique_keys_out*/, long long* /*d_sums_out*/,
    int* /*d_num_runs_out*/) {
  throw_no_cuda("cub::reduce_by_key_sum_u64_i64");
}

#if VBT_INTERNAL_TESTS
namespace testonly {

bool temp_storage_bytes0_requires_nonnull(vbt::cuda::Allocator& /*alloc*/, vbt::cuda::Stream /*stream*/) {
  throw_no_cuda("cub::testonly::temp_storage_bytes0_requires_nonnull");
}

} // namespace testonly
#endif

#endif // !VBT_WITH_CUDA

}}} // namespace vbt::cuda::cub
