// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/contiguity.h"

#include <algorithm>

namespace vbt {
namespace core {

bool compute_non_overlapping_and_dense(const std::vector<int64_t>& sizes,
                                       const std::vector<int64_t>& strides) noexcept {
  const auto dim = sizes.size();
  if (dim == 0) {
    // Scalar: treat as contiguous/dense
    return true;
  }
  if (dim == 1) {
    return sizes[0] < 2 || strides[0] == 1;
  }
  // Build permutation of dims and sort by stride, placing size<2 dims at end
  std::vector<int64_t> perm(dim);
  for (size_t i = 0; i < dim; ++i) perm[i] = static_cast<int64_t>(i);
  std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
    if (sizes[a] < 2) {
      return false;
    } else if (sizes[b] < 2) {
      return true;
    }
    return strides[a] < strides[b];
  });
  int64_t require_stride = 1;
  for (size_t i = 0; i < dim; ++i) {
    const auto size_i = sizes[perm[i]];
    if (size_i < 2) {
      return true;
    }
    if (strides[perm[i]] != require_stride) {
      return false;
    }
    require_stride *= size_i;
  }
  return true;
}

} // namespace core
} // namespace vbt
