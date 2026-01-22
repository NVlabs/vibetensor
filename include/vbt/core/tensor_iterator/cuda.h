// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "vbt/core/tensor_iterator/core.h"

namespace vbt {
namespace core {

struct DeviceStrideMeta;

// Practical CUDA dim limit for TI-backed metadata and kernels.
// Must not exceed kTensorIterMaxRank or the plugin C ABI limits.
inline constexpr std::int32_t kTensorIterCudaMaxNdim = 25;
static_assert(kTensorIterCudaMaxNdim > 0,
              "TI CUDA: kTensorIterCudaMaxNdim must be positive");
static_assert(kTensorIterCudaMaxNdim <= kTensorIterMaxRank,
              "TI CUDA: kTensorIterCudaMaxNdim must not exceed kTensorIterMaxRank");

// Device-only helper; used by TI-backed CUDA kernels.
#ifdef __CUDACC__
__device__ __forceinline__ std::int64_t compute_offset_elems(std::int64_t li,
                                                             const DeviceStrideMeta& m) {
  if (m.ndim == 0) {
    return 0;
  }

  std::int64_t off = 0;
  // Iterate over TI iteration dims in order; this matches the host-side
  // helper used in tests and ensures row-major linear indexing when
  // DeviceStrideMeta::sizes/strides describe the TI iteration space.
  for (std::int64_t d = 0; d < m.ndim; ++d) {
    std::int64_t size_d = (m.sizes[d] == 0 ? 1 : m.sizes[d]);
    std::int64_t idx_d  = (size_d == 1) ? 0 : (li % size_d);
    li = (size_d == 1) ? li : (li / size_d);
    off += idx_d * m.strides[d];  // may be zero or negative
  }
  return off;
}
#endif

} // namespace core
} // namespace vbt
