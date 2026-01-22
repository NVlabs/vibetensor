// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

namespace vbt {
namespace core {

[[nodiscard]] inline bool checked_add_i64(int64_t a, int64_t b, int64_t& out) noexcept {
  // Detect overflow before computing
  if (b > 0 && a > (std::numeric_limits<int64_t>::max() - b)) return false;
  if (b < 0 && a < (std::numeric_limits<int64_t>::min() - b)) return false;
  out = static_cast<int64_t>(a + b);
  return true;
}

[[nodiscard]] inline bool checked_mul_i64(int64_t a, int64_t b, int64_t& out) noexcept {
  // Fast paths
  if (a == 0 || b == 0) { out = 0; return true; }
  if (a == 1) { out = b; return true; }
  if (b == 1) { out = a; return true; }
  if (a == -1) {
    if (b == std::numeric_limits<int64_t>::min()) return false; // overflow
    out = -b; return true;
  }
  if (b == -1) {
    if (a == std::numeric_limits<int64_t>::min()) return false; // overflow
    out = -a; return true;
  }
  // General bound check: |a| <= max/|b|
  auto max = std::numeric_limits<int64_t>::max();
  if (a > 0) {
    if (b > 0) {
      if (a > max / b) return false;
    } else { // b < 0
      if (b < std::numeric_limits<int64_t>::min() / a) return false;
    }
  } else { // a < 0
    if (b > 0) {
      if (a < std::numeric_limits<int64_t>::min() / b) return false;
    } else { // b < 0
      if (a != 0 && b < max / a) return false;
    }
  }
  out = static_cast<int64_t>(a * b);
  return true;
}

[[nodiscard]] inline bool accumulate_span_terms(int64_t stride, int64_t size, int64_t& min_acc, int64_t& max_acc) noexcept {
  const int64_t d = size > 0 ? (size - 1) : 0;
  if (d == 0) return true; // no-op
  int64_t term = 0;
  if (!checked_mul_i64(stride, d, term)) return false;
  if (stride >= 0) {
    int64_t tmp = max_acc;
    if (!checked_add_i64(tmp, term, tmp)) return false;
    max_acc = tmp;
  } else {
    int64_t tmp = min_acc;
    if (!checked_add_i64(tmp, term, tmp)) return false;
    min_acc = tmp;
  }
  return true;
}

} // namespace core
} // namespace vbt
