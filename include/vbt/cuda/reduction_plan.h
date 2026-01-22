// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "vbt/core/tensor_iterator/core.h"
#include "vbt/cuda/reduction_env.h"

namespace vbt {
namespace cuda {
namespace reduction {

inline constexpr std::int32_t kCudaReducePlanMaxDims = vbt::core::kTensorIterMaxRank;

struct CudaReducePlanDevice {
  std::int64_t out_numel{0};
  std::int64_t slice_len{0};

  std::int32_t iter_ndim{0};
  std::int32_t kept_ndim{0};
  std::int32_t red_ndim{0};
  std::int32_t reserved0{0};

  std::int64_t kept_sizes[kCudaReducePlanMaxDims]{};
  std::int64_t kept_in_strides[kCudaReducePlanMaxDims]{};
  std::int64_t kept_out_strides[kCudaReducePlanMaxDims]{};

  std::int64_t red_linear_stride{0};
  std::int64_t reserved_i64[7]{};
};

static_assert(std::is_trivially_copyable_v<CudaReducePlanDevice>,
              "CudaReducePlanDevice must be trivially copyable for ABI stability");
static_assert(std::is_standard_layout_v<CudaReducePlanDevice>,
              "CudaReducePlanDevice must have standard layout");
static_assert(alignof(CudaReducePlanDevice) == alignof(std::int64_t),
              "CudaReducePlanDevice must be 8-byte aligned");
static_assert(sizeof(CudaReducePlanDevice) == 1632,
              "CudaReducePlanDevice size must match ABI contract");
static_assert(offsetof(CudaReducePlanDevice, out_numel) == 0,
              "CudaReducePlanDevice::out_numel must be first");
static_assert(offsetof(CudaReducePlanDevice, slice_len) == 8,
              "CudaReducePlanDevice::slice_len offset mismatch");
static_assert(offsetof(CudaReducePlanDevice, iter_ndim) == 16,
              "CudaReducePlanDevice::iter_ndim offset mismatch");
static_assert(offsetof(CudaReducePlanDevice, kept_sizes) == 32,
              "CudaReducePlanDevice::kept_sizes offset mismatch");
static_assert(offsetof(CudaReducePlanDevice, kept_in_strides) == 32 + 8 * kCudaReducePlanMaxDims,
              "CudaReducePlanDevice::kept_in_strides offset mismatch");
static_assert(offsetof(CudaReducePlanDevice, kept_out_strides) == 32 + 16 * kCudaReducePlanMaxDims,
              "CudaReducePlanDevice::kept_out_strides offset mismatch");
static_assert(offsetof(CudaReducePlanDevice, red_linear_stride) == 32 + 24 * kCudaReducePlanMaxDims,
              "CudaReducePlanDevice::red_linear_stride offset mismatch");

struct CudaReducePlanBuildResult {
  CudaReducePlanDevice        plan{};
  CudaReduceIneligibleReason  ineligible_reason{CudaReduceIneligibleReason::None};
};

// Build a CUDA reduction plan with no allocations.
//
// Preconditions (validated; failures return Overflow):
// - out_meta.ndim == in_meta.ndim in [0, kTensorIterMaxRank]
// - reduce_dims is non-empty, strictly increasing, and each dim in [0, iter_ndim)
// - Intended for non-empty reductions (out_numel > 0 && slice_len > 0);
//   empty cases are handled by the reduction dispatcher.
CudaReducePlanBuildResult build_cuda_reduce_plan_noalloc(
    const vbt::core::DeviceStrideMeta& out_meta,
    const vbt::core::DeviceStrideMeta& in_meta,
    std::span<const int64_t> reduce_dims,
    std::int64_t out_numel,
    std::int64_t slice_len) noexcept;

}  // namespace reduction
}  // namespace cuda
}  // namespace vbt
