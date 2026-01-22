// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/cub.h"

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA

#include "cub_detail_common.cuh"
#include "cub_detail_compat.cuh"

#include <cuda_runtime_api.h>

#include <cuda/std/functional>

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <stdexcept>

namespace vbt { namespace cuda { namespace cub_detail {
__device__ __align__(16) unsigned char g_vbt_cub_dummy_storage[16];

void* cub_dummy_temp_storage_ptr() {
  void* ptr = nullptr;
  cudaError_t st = cudaGetSymbolAddress(&ptr, g_vbt_cub_dummy_storage);
  cudaCheck(st, "cudaGetSymbolAddress(cub dummy)");
  return ptr;
}
}}}

namespace vbt { namespace cuda { namespace cub {

namespace vbt_cub_detail = vbt::cuda::cub_detail;

void select_indices_from_u8_flags_i64(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const std::uint8_t* d_flags01,
    int n,
    std::int64_t* d_out_indices,
    int* d_num_selected_out) {
  if (n < 0) {
    throw std::invalid_argument("cub: select_indices_from_u8_flags_i64: n must be >= 0");
  }

  // Accept null pointers for the empty case.
  if (n == 0) {
    vbt_cub_detail::CubContext ctx(alloc, stream);
    if (d_num_selected_out) {
      cudaError_t st = cudaMemsetAsync(
          d_num_selected_out,
          0,
          sizeof(int),
          reinterpret_cast<cudaStream_t>(ctx.stream.handle()));
      vbt_cub_detail::cudaCheck(st, "cudaMemsetAsync(num_selected=0)");
    }
    return;
  }

  if (!d_flags01) {
    throw std::invalid_argument("cub: select_indices_from_u8_flags_i64: d_flags01 must be non-null when n > 0");
  }
  if (!d_out_indices) {
    throw std::invalid_argument("cub: select_indices_from_u8_flags_i64: d_out_indices must be non-null when n > 0");
  }
  if (!d_num_selected_out) {
    throw std::invalid_argument(
        "cub: select_indices_from_u8_flags_i64: d_num_selected_out must be non-null when n > 0");
  }

  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  vbt_cub_detail::call_with_temp_storage_checked(
      alloc,
      stream,
      "cub::DeviceSelect::Flagged(i64 indices from u8 flags)",
      [&](void* tmp, std::size_t& temp_bytes) -> cudaError_t {
        // Select indices [0..n) where flags01[i] != 0.
        vbt_cub_detail::CountingIterator<std::int64_t> it0(static_cast<std::int64_t>(0));
        return vbt::cuda::cub_wrapped::cub::DeviceSelect::Flagged(
            tmp,
            temp_bytes,
            it0,
            d_flags01,
            d_out_indices,
            d_num_selected_out,
            n,
            cu_stream);
      });
}

void reduce_sum_u8_as_i32(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const std::uint8_t* d_flags01,
    int n,
    int* d_out_sum) {
  if (n < 0) {
    throw std::invalid_argument("cub: reduce_sum_u8_as_i32: n must be >= 0");
  }

  // Accept null pointers for the empty case.
  if (n == 0) {
    vbt_cub_detail::CubContext ctx(alloc, stream);
    if (d_out_sum) {
      cudaError_t st = cudaMemsetAsync(
          d_out_sum,
          0,
          sizeof(int),
          reinterpret_cast<cudaStream_t>(ctx.stream.handle()));
      vbt_cub_detail::cudaCheck(st, "cudaMemsetAsync(sum=0)");
    }
    return;
  }

  if (!d_flags01) {
    throw std::invalid_argument("cub: reduce_sum_u8_as_i32: d_flags01 must be non-null when n > 0");
  }
  if (!d_out_sum) {
    throw std::invalid_argument("cub: reduce_sum_u8_as_i32: d_out_sum must be non-null when n > 0");
  }

  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  vbt_cub_detail::call_with_temp_storage_checked(
      alloc,
      stream,
      "cub::DeviceReduce::Sum(u8 flags -> i32)",
      [&](void* tmp, std::size_t& temp_bytes) -> cudaError_t {
        // DeviceReduce::Sum promotes via OutputIteratorT (int*), preventing u8 overflow.
        return vbt::cuda::cub_wrapped::cub::DeviceReduce::Sum(tmp, temp_bytes, d_flags01, d_out_sum, n, cu_stream);
      });
}

void reduce_all_contig_sum_f32(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const float* d_in,
    int n,
    float* d_out) {
  if (n < 0) {
    throw std::invalid_argument("cub: reduce_all_contig_sum_f32: n must be >= 0");
  }

  // Accept null pointers for the empty case.
  if (n == 0) {
    vbt_cub_detail::CubContext ctx(alloc, stream);
    if (d_out) {
      cudaError_t st = cudaMemsetAsync(
          d_out,
          0,
          sizeof(float),
          reinterpret_cast<cudaStream_t>(ctx.stream.handle()));
      vbt_cub_detail::cudaCheck(st, "cudaMemsetAsync(sum=0)");
    }
    return;
  }

  if (!d_in) {
    throw std::invalid_argument("cub: reduce_all_contig_sum_f32: d_in must be non-null when n > 0");
  }
  if (!d_out) {
    throw std::invalid_argument("cub: reduce_all_contig_sum_f32: d_out must be non-null when n > 0");
  }

  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  vbt_cub_detail::call_with_temp_storage_checked(
      alloc,
      stream,
      "cub::DeviceReduce::Sum(f32)",
      [&](void* tmp, std::size_t& temp_bytes) -> cudaError_t {
        return vbt::cuda::cub_wrapped::cub::DeviceReduce::Sum(
            tmp,
            temp_bytes,
            d_in,
            d_out,
            n,
            cu_stream);
      });
}

void reduce_all_contig_sum_i64(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const long long* d_in,
    int n,
    long long* d_out) {
  if (n < 0) {
    throw std::invalid_argument("cub: reduce_all_contig_sum_i64: n must be >= 0");
  }

  // Accept null pointers for the empty case.
  if (n == 0) {
    vbt_cub_detail::CubContext ctx(alloc, stream);
    if (d_out) {
      cudaError_t st = cudaMemsetAsync(
          d_out,
          0,
          sizeof(long long),
          reinterpret_cast<cudaStream_t>(ctx.stream.handle()));
      vbt_cub_detail::cudaCheck(st, "cudaMemsetAsync(sum=0)");
    }
    return;
  }

  if (!d_in) {
    throw std::invalid_argument("cub: reduce_all_contig_sum_i64: d_in must be non-null when n > 0");
  }
  if (!d_out) {
    throw std::invalid_argument("cub: reduce_all_contig_sum_i64: d_out must be non-null when n > 0");
  }

  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  vbt_cub_detail::call_with_temp_storage_checked(
      alloc,
      stream,
      "cub::DeviceReduce::Sum(i64)",
      [&](void* tmp, std::size_t& temp_bytes) -> cudaError_t {
        return vbt::cuda::cub_wrapped::cub::DeviceReduce::Sum(
            tmp,
            temp_bytes,
            d_in,
            d_out,
            n,
            cu_stream);
      });
}

namespace {

__global__ void exclusive_scan_i32_seed_kernel(const int* last_in, const int* last_out, int* seed) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *seed = (*last_out) + (*last_in);
  }
}

inline int exclusive_scan_i32_max_items() noexcept {
  constexpr int kDefault = (INT_MAX / 2) + 1;
#if VBT_INTERNAL_TESTS
  const char* env = std::getenv("VBT_INTERNAL_CUDA_CUB_SCAN_MAX_ITEMS");
  if (!env || !*env) return kDefault;
  char* end = nullptr;
  errno = 0;
  long v = std::strtol(env, &end, 10);
  if (errno != 0 || end == env || (end && *end != '\0') || v <= 0 || v > kDefault) {
    return kDefault;
  }
  return static_cast<int>(v);
#else
  return kDefault;
#endif
}

} // namespace

void exclusive_scan_i32(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const int* d_in,
    int n,
    int* d_out) {
  if (n < 0) {
    throw std::invalid_argument("cub: exclusive_scan_i32: n must be >= 0");
  }

  // Accept null pointers for the empty case.
  if (n == 0) {
    vbt_cub_detail::CubContext ctx(alloc, stream);
    (void)ctx;
    return;
  }

  if (!d_in) {
    throw std::invalid_argument("cub: exclusive_scan_i32: d_in must be non-null when n > 0");
  }
  if (!d_out) {
    throw std::invalid_argument("cub: exclusive_scan_i32: d_out must be non-null when n > 0");
  }

  vbt_cub_detail::CubContext ctx(alloc, stream);

  // This wrapper allocates temporary storage and is not capture-safe.
  if (vbt::cuda::streamCaptureStatus(stream) != vbt::cuda::CaptureStatus::None) {
    vbt_cub_detail::throw_capture_denied(stream.device_index());
  }

  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(int);
  vbt_cub_detail::check_no_overlap_ranges(
      d_in,
      bytes,
      d_out,
      bytes,
      "cub: exclusive_scan_i32: d_in and d_out must not overlap");

  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  const int max_items = exclusive_scan_i32_max_items();
  if (max_items <= 0) {
    throw std::runtime_error("cub: exclusive_scan_i32: max_items must be > 0");
  }

  int offset = 0;
  int chunk = (n < max_items) ? n : max_items;

  vbt_cub_detail::call_with_temp_storage_checked(
      alloc,
      stream,
      "cub::DeviceScan::ExclusiveScan(i32)",
      [&](void* tmp, std::size_t& temp_bytes) -> cudaError_t {
        return vbt::cuda::cub_wrapped::cub::DeviceScan::ExclusiveScan(
            tmp,
            temp_bytes,
            d_in,
            d_out,
            ::cuda::std::plus<int>(),
            /*init_value=*/0,
            chunk,
            cu_stream);
      });

  offset += chunk;

  while (offset < n) {
    int remaining = n - offset;
    int this_chunk = (remaining < max_items) ? remaining : max_items;

    vbt_cub_detail::StreamDeviceBuffer seed_buf(ctx.alloc, ctx.stream, sizeof(int));
    auto* d_seed = static_cast<int*>(seed_buf.data());

    exclusive_scan_i32_seed_kernel<<<1, 1, 0, cu_stream>>>(
        d_in + (offset - 1),
        d_out + (offset - 1),
        d_seed);
    vbt_cub_detail::cudaCheck(cudaGetLastError(), "exclusive_scan_i32 seed kernel");

    auto init = vbt::cuda::cub_wrapped::cub::FutureValue<int>(d_seed);

    vbt_cub_detail::call_with_temp_storage_checked(
        alloc,
        stream,
        "cub::DeviceScan::ExclusiveScan(i32,chunked)",
        [&](void* tmp, std::size_t& temp_bytes) -> cudaError_t {
          return vbt::cuda::cub_wrapped::cub::DeviceScan::ExclusiveScan(
              tmp,
              temp_bytes,
              d_in + offset,
              d_out + offset,
              ::cuda::std::plus<int>(),
              init,
              this_chunk,
              cu_stream);
        });

    offset += this_chunk;
  }
}

void reduce_by_key_sum_u64_f32(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const std::uint64_t* d_keys_in,
    const float* d_vals_in,
    int n,
    std::uint64_t* d_unique_keys_out,
    float* d_sums_out,
    int* d_num_runs_out) {
  if (n < 0) {
    throw std::invalid_argument(
        "cub: reduce_by_key_sum_u64_f32: n must be >= 0");
  }

  // Accept null pointers for the empty case.
  if (n == 0) {
    vbt_cub_detail::CubContext ctx(alloc, stream);
    if (d_num_runs_out) {
      cudaError_t st = cudaMemsetAsync(
          d_num_runs_out,
          0,
          sizeof(int),
          reinterpret_cast<cudaStream_t>(ctx.stream.handle()));
      vbt_cub_detail::cudaCheck(st, "cudaMemsetAsync(num_runs=0)");
    }
    return;
  }

  if (!d_keys_in) {
    throw std::invalid_argument(
        "cub: reduce_by_key_sum_u64_f32: d_keys_in must be non-null when n > 0");
  }
  if (!d_vals_in) {
    throw std::invalid_argument(
        "cub: reduce_by_key_sum_u64_f32: d_vals_in must be non-null when n > 0");
  }
  if (!d_unique_keys_out) {
    throw std::invalid_argument(
        "cub: reduce_by_key_sum_u64_f32: d_unique_keys_out must be non-null when n > 0");
  }
  if (!d_sums_out) {
    throw std::invalid_argument(
        "cub: reduce_by_key_sum_u64_f32: d_sums_out must be non-null when n > 0");
  }
  if (!d_num_runs_out) {
    throw std::invalid_argument(
        "cub: reduce_by_key_sum_u64_f32: d_num_runs_out must be non-null when n > 0");
  }

  const std::size_t keys_bytes =
      static_cast<std::size_t>(n) * sizeof(std::uint64_t);
  const std::size_t vals_bytes =
      static_cast<std::size_t>(n) * sizeof(float);

  vbt_cub_detail::check_no_overlap_ranges(
      d_keys_in,
      keys_bytes,
      d_unique_keys_out,
      keys_bytes,
      "cub: reduce_by_key_sum_u64_f32: keys input/output must not overlap");

  vbt_cub_detail::check_no_overlap_ranges(
      d_vals_in,
      vals_bytes,
      d_sums_out,
      vals_bytes,
      "cub: reduce_by_key_sum_u64_f32: values input/output must not overlap");

  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  vbt_cub_detail::call_with_temp_storage_checked(
      alloc,
      stream,
      "cub::DeviceReduce::ReduceByKey(u64,f32,sum)",
      [&](void* tmp, std::size_t& temp_bytes) -> cudaError_t {
        return vbt::cuda::cub_wrapped::cub::DeviceReduce::ReduceByKey(
            tmp,
            temp_bytes,
            d_keys_in,
            d_unique_keys_out,
            d_vals_in,
            d_sums_out,
            d_num_runs_out,
            ::cuda::std::plus<float>(),
            n,
            cu_stream);
      });
}

void reduce_by_key_sum_u64_i64(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const std::uint64_t* d_keys_in,
    const long long* d_vals_in,
    int n,
    std::uint64_t* d_unique_keys_out,
    long long* d_sums_out,
    int* d_num_runs_out) {
  if (n < 0) {
    throw std::invalid_argument(
        "cub: reduce_by_key_sum_u64_i64: n must be >= 0");
  }

  // Accept null pointers for the empty case.
  if (n == 0) {
    vbt_cub_detail::CubContext ctx(alloc, stream);
    if (d_num_runs_out) {
      cudaError_t st = cudaMemsetAsync(
          d_num_runs_out,
          0,
          sizeof(int),
          reinterpret_cast<cudaStream_t>(ctx.stream.handle()));
      vbt_cub_detail::cudaCheck(st, "cudaMemsetAsync(num_runs=0)");
    }
    return;
  }

  if (!d_keys_in) {
    throw std::invalid_argument(
        "cub: reduce_by_key_sum_u64_i64: d_keys_in must be non-null when n > 0");
  }
  if (!d_vals_in) {
    throw std::invalid_argument(
        "cub: reduce_by_key_sum_u64_i64: d_vals_in must be non-null when n > 0");
  }
  if (!d_unique_keys_out) {
    throw std::invalid_argument(
        "cub: reduce_by_key_sum_u64_i64: d_unique_keys_out must be non-null when n > 0");
  }
  if (!d_sums_out) {
    throw std::invalid_argument(
        "cub: reduce_by_key_sum_u64_i64: d_sums_out must be non-null when n > 0");
  }
  if (!d_num_runs_out) {
    throw std::invalid_argument(
        "cub: reduce_by_key_sum_u64_i64: d_num_runs_out must be non-null when n > 0");
  }

  const std::size_t keys_bytes =
      static_cast<std::size_t>(n) * sizeof(std::uint64_t);
  const std::size_t vals_bytes =
      static_cast<std::size_t>(n) * sizeof(long long);

  vbt_cub_detail::check_no_overlap_ranges(
      d_keys_in,
      keys_bytes,
      d_unique_keys_out,
      keys_bytes,
      "cub: reduce_by_key_sum_u64_i64: keys input/output must not overlap");

  vbt_cub_detail::check_no_overlap_ranges(
      d_vals_in,
      vals_bytes,
      d_sums_out,
      vals_bytes,
      "cub: reduce_by_key_sum_u64_i64: values input/output must not overlap");

  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  vbt_cub_detail::call_with_temp_storage_checked(
      alloc,
      stream,
      "cub::DeviceReduce::ReduceByKey(u64,i64,sum)",
      [&](void* tmp, std::size_t& temp_bytes) -> cudaError_t {
        return vbt::cuda::cub_wrapped::cub::DeviceReduce::ReduceByKey(
            tmp,
            temp_bytes,
            d_keys_in,
            d_unique_keys_out,
            d_vals_in,
            d_sums_out,
            d_num_runs_out,
            ::cuda::std::plus<long long>(),
            n,
            cu_stream);
      });
}

#if VBT_INTERNAL_TESTS
namespace testonly {

bool temp_storage_bytes0_requires_nonnull(vbt::cuda::Allocator& alloc, vbt::cuda::Stream stream) {
  bool saw_nonnull = false;
  vbt_cub_detail::call_with_temp_storage_checked(
      alloc,
      stream,
      "cub bytes==0 temp storage probe",
      [&](void* tmp, std::size_t& temp_bytes) -> cudaError_t {
        if (tmp == nullptr) {
          temp_bytes = 0;
          return cudaSuccess;
        }
        saw_nonnull = (tmp != nullptr);
        return cudaSuccess;
      });
  return saw_nonnull;
}

} // namespace testonly
#endif

}}} // namespace vbt::cuda::cub

#endif // VBT_WITH_CUDA
