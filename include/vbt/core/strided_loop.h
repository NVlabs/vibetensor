// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <limits>

#include "vbt/core/tensor.h"
#include "vbt/core/checked_math.h"

namespace vbt {
namespace core {

// Local helper for absolute value with INT64_MIN guard
inline bool checked_abs_i64_hdr(int64_t x, int64_t& out) {
  if (x == std::numeric_limits<int64_t>::min()) return false;
  out = x >= 0 ? x : -x;
  return true;
}

struct LoopSpec {
  int64_t ndim{0};
  std::vector<int64_t> sizes;       // iteration sizes per permuted dim
  std::vector<int64_t> step_bytes;  // positive byte step per permuted dim (abs stride * itemsize)
  std::uintptr_t p0{0};             // starting pointer (lower corner), as integer
};

// Build a unary iteration spec for an arbitrary strided tensor. The returned
// permutation orders dimensions by increasing absolute byte step (ignores size==1).
LoopSpec build_unary_spec(const TensorImpl& t, std::vector<int64_t>& perm_out);

// Binary: iterate dst and src in lockstep using dst's permutation
LoopSpec build_binary_spec(const TensorImpl& dst, const TensorImpl& src, std::vector<int64_t>& perm_out);

// Unary inplace iteration over dst using positive step_bytes and pre-shifted p0
// The callback receives the current element pointer within dst.
template <class Fn>
inline void for_each_1out_inplace(const TensorImpl& dst, Fn fn) {
  std::vector<int64_t> perm;
  LoopSpec spd = build_unary_spec(dst, perm);
  if (dst.numel() == 0) return;
  auto* pd = reinterpret_cast<std::uint8_t*>(spd.p0);
  // Fast-path for scalar/all-size==1 tensors (exactly one logical element)
  if (spd.ndim == 0) {
    fn(pd);
    return;
  }
  if (spd.ndim == 1) {
    for (int64_t i = 0; i < spd.sizes[0]; ++i) {
      fn(pd);
      pd += static_cast<std::ptrdiff_t>(spd.step_bytes[0]);
    }
    return;
  }
  std::vector<int64_t> idx(static_cast<std::size_t>(spd.ndim), 0);
  while (true) {
    fn(pd);
    int d = static_cast<int>(spd.ndim - 1);
    pd += static_cast<std::ptrdiff_t>(spd.step_bytes[static_cast<std::size_t>(d)]);
    ++idx[static_cast<std::size_t>(d)];
    while (d >= 0 && idx[static_cast<std::size_t>(d)] >= spd.sizes[static_cast<std::size_t>(d)]) {
      int64_t prod_d = 0;
      if (!checked_mul_i64(spd.step_bytes[static_cast<std::size_t>(d)], spd.sizes[static_cast<std::size_t>(d)], prod_d)) {
        throw std::overflow_error("strided_loop: overflow during pointer carry (unary)");
      }
      pd -= static_cast<std::ptrdiff_t>(prod_d);
      idx[static_cast<std::size_t>(d)] = 0;
      --d;
      if (d < 0) return;
      pd += static_cast<std::ptrdiff_t>(spd.step_bytes[static_cast<std::size_t>(d)]);
      ++idx[static_cast<std::size_t>(d)];
    }
  }
}

template <class Fn>
inline void for_each_1out_1in(const TensorImpl& dst, const TensorImpl& src, Fn fn) {
  std::vector<int64_t> perm;
  LoopSpec spd = build_binary_spec(dst, src, perm);
  if (dst.numel() == 0) return;
  // Compute src per-dimension step bytes in original dim order (abs(stride*itemsize))
  const auto& dsizes = dst.sizes();
  const auto& sstrides = src.strides();
  const int64_t item_b = static_cast<int64_t>(src.itemsize());
  std::vector<int64_t> src_step_bytes_orig(sstrides.size(), 0);
  for (std::size_t i = 0; i < sstrides.size(); ++i) {
    int64_t tmp = 0, abs_st = 0;
    if (!checked_mul_i64(sstrides[i], item_b, tmp) || !checked_abs_i64_hdr(tmp, abs_st)) {
      throw std::overflow_error("strided_loop: overflow computing src stride bytes");
    }
    src_step_bytes_orig[i] = abs_st;
  }
  // Map src step bytes into dst’s perm order, skipping size==1 dims to match spd.ndim
  std::vector<int64_t> src_steps;
  src_steps.reserve(static_cast<std::size_t>(spd.ndim));
  for (std::size_t k = 0; k < perm.size(); ++k) {
    const std::size_t i = static_cast<std::size_t>(perm[k]);
    if (dsizes[i] == 1) continue;
    src_steps.push_back(src_step_bytes_orig[i]);
  }
  auto* pd = reinterpret_cast<std::uint8_t*>(spd.p0);
  // compute src p0 similarly
  std::vector<int64_t> perm_tmp;
  LoopSpec sps_full = build_unary_spec(src, perm_tmp);
  auto* ps = reinterpret_cast<std::uint8_t*>(sps_full.p0);
  // Fast-path for scalar/all-size==1 tensors (exactly one logical element)
  if (spd.ndim == 0) {
    fn(pd, ps);
    return;
  }
  if (spd.ndim == 1) {
    for (int64_t i = 0; i < spd.sizes[0]; ++i) {
      fn(pd, ps);
      pd += static_cast<std::ptrdiff_t>(spd.step_bytes[0]);
      ps += static_cast<std::ptrdiff_t>(src_steps[0]);
    }
    return;
  }
  std::vector<int64_t> idx(static_cast<std::size_t>(spd.ndim), 0);
  while (true) {
    fn(pd, ps);
    // increment last dim
    int d = static_cast<int>(spd.ndim - 1);
    pd += static_cast<std::ptrdiff_t>(spd.step_bytes[static_cast<std::size_t>(d)]);
    ps += static_cast<std::ptrdiff_t>(src_steps[static_cast<std::size_t>(d)]);
    ++idx[static_cast<std::size_t>(d)];
    // Optional: add debug-only assertions for pointer bounds here
    // e.g., VBT_DEBUG_ONLY(assert(in_bounds(pd) && in_bounds(ps));)
    while (d >= 0 && idx[static_cast<std::size_t>(d)] >= spd.sizes[static_cast<std::size_t>(d)]) {
      // reset this dim
      int64_t prod_d = 0, prod_s = 0;
      if (!checked_mul_i64(spd.step_bytes[static_cast<std::size_t>(d)], spd.sizes[static_cast<std::size_t>(d)], prod_d) ||
          !checked_mul_i64(src_steps[static_cast<std::size_t>(d)], spd.sizes[static_cast<std::size_t>(d)], prod_s)) {
        throw std::overflow_error("strided_loop: overflow during pointer carry (binary)");
      }
      pd -= static_cast<std::ptrdiff_t>(prod_d);
      ps -= static_cast<std::ptrdiff_t>(prod_s);
      idx[static_cast<std::size_t>(d)] = 0;
      --d;
      if (d < 0) return;
      pd += static_cast<std::ptrdiff_t>(spd.step_bytes[static_cast<std::size_t>(d)]);
      ps += static_cast<std::ptrdiff_t>(src_steps[static_cast<std::size_t>(d)]);
      ++idx[static_cast<std::size_t>(d)];
    }
  }
}

} // namespace core
} // namespace vbt
