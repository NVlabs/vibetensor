// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace vbt { namespace cuda {

class Allocator;
class Stream;

}} // namespace vbt::cuda

namespace vbt { namespace cuda { namespace cub {

// Centralized wrappers around CUB/CCCL primitives.
//
// Invariants:
// - Callers must pass an explicit vbt::cuda::Stream.
// - Temp storage allocations must be stream-correct (see wrapper impl).
// - This header must not include CUDA runtime headers nor any CUB/Thrust headers.

void select_indices_from_u8_flags_i64(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const std::uint8_t* d_flags01,
    int n,
    std::int64_t* d_out_indices,
    int* d_num_selected_out);

void reduce_sum_u8_as_i32(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const std::uint8_t* d_flags01,
    int n,
    int* d_out_sum);

void reduce_all_contig_sum_f32(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const float* d_in,
    int n,
    float* d_out);

void reduce_all_contig_sum_i64(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const long long* d_in,
    int n,
    long long* d_out);

// Exclusive prefix sum (init=0, op=plus) for int32.
// Note: d_in and d_out must not overlap; in-place scan is not supported by this wrapper.
void exclusive_scan_i32(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const int* d_in,
    int n,
    int* d_out);

void radix_sort_pairs_u64_f32(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    std::uint64_t* d_keys_inout,
    float* d_vals_inout,
    int n);

void radix_sort_pairs_u64_i64(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    std::uint64_t* d_keys_inout,
    long long* d_vals_inout,
    int n);

void reduce_by_key_sum_u64_f32(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const std::uint64_t* d_keys_in,
    const float* d_vals_in,
    int n,
    std::uint64_t* d_unique_keys_out,
    float* d_sums_out,
    int* d_num_runs_out);

void reduce_by_key_sum_u64_i64(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const std::uint64_t* d_keys_in,
    const long long* d_vals_in,
    int n,
    std::uint64_t* d_unique_keys_out,
    long long* d_sums_out,
    int* d_num_runs_out);

#if VBT_INTERNAL_TESTS
namespace testonly {

// Probe the internal "bytes==0" temp-storage path.
// Returns true iff the wrapper executed the "run" pass with a non-null
// temp-storage pointer.
bool temp_storage_bytes0_requires_nonnull(vbt::cuda::Allocator& alloc, vbt::cuda::Stream stream);

} // namespace testonly
#endif

}}} // namespace vbt::cuda::cub
