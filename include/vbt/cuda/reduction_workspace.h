// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include <vbt/core/checked_math.h>

namespace vbt {
namespace cuda {
namespace reduction {

inline constexpr std::size_t kCudaReductionK2MultiAlign = 256;

static_assert((kCudaReductionK2MultiAlign & (kCudaReductionK2MultiAlign - 1)) == 0,
              "kCudaReductionK2MultiAlign must be a power of two");
static_assert(kCudaReductionK2MultiAlign <=
              static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()),
              "kCudaReductionK2MultiAlign must fit in int64_t");

struct K2MultiWorkspaceLayout {
  std::size_t partials_bytes{0};
  std::size_t sema_off{0};
  std::size_t semaphores_bytes{0};
  std::size_t total_bytes{0};
};

namespace detail {

[[nodiscard]] inline bool align_up_256_i64(std::int64_t x, std::int64_t& out) noexcept {
  if (x < 0) return false;

  // kCudaReductionK2MultiAlign is a power-of-two.
  constexpr std::int64_t kAlign = static_cast<std::int64_t>(kCudaReductionK2MultiAlign);
  constexpr std::int64_t kMask = kAlign - 1;

  std::int64_t tmp = 0;
  if (!vbt::core::checked_add_i64(x, kMask, tmp)) return false;

  out = tmp & ~kMask;
  return true;
}

} // namespace detail

// Returns false on overflow/invalid inputs.
[[nodiscard]] inline bool compute_k2multi_workspace_layout(
    std::int64_t out_numel,
    std::uint32_t ctas_per_output,
    std::size_t itemsize,
    K2MultiWorkspaceLayout* out) noexcept {
  if (!out) return false;
  *out = {};

  if (out_numel < 0) return false;
  if (ctas_per_output < 2) return false;
  if (itemsize == 0) return false;

  if (itemsize > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) return false;

  const std::int64_t ctas_i64 = static_cast<std::int64_t>(ctas_per_output);
  const std::int64_t itemsize_i64 = static_cast<std::int64_t>(itemsize);

  std::int64_t partial_elems = 0;
  if (!vbt::core::checked_mul_i64(out_numel, ctas_i64, partial_elems)) return false;

  std::int64_t partials_bytes_i64 = 0;
  if (!vbt::core::checked_mul_i64(partial_elems, itemsize_i64, partials_bytes_i64)) return false;

  std::int64_t sema_off_i64 = 0;
  if (!detail::align_up_256_i64(partials_bytes_i64, sema_off_i64)) return false;

  std::int64_t semaphores_bytes_i64 = 0;
  if (!vbt::core::checked_mul_i64(
          out_numel,
          static_cast<std::int64_t>(sizeof(std::uint32_t)),
          semaphores_bytes_i64)) {
    return false;
  }

  std::int64_t sema_end_i64 = 0;
  if (!vbt::core::checked_add_i64(sema_off_i64, semaphores_bytes_i64, sema_end_i64)) return false;

  std::int64_t total_i64 = 0;
  if (!detail::align_up_256_i64(sema_end_i64, total_i64)) return false;

  out->partials_bytes = static_cast<std::size_t>(partials_bytes_i64);
  out->sema_off = static_cast<std::size_t>(sema_off_i64);
  out->semaphores_bytes = static_cast<std::size_t>(semaphores_bytes_i64);
  out->total_bytes = static_cast<std::size_t>(total_i64);
  return true;
}

} // namespace reduction
} // namespace cuda
} // namespace vbt
