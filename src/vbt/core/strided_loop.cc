// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/strided_loop.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include "vbt/core/checked_math.h"

namespace vbt {
namespace core {

static inline bool checked_abs_i64(int64_t x, int64_t& out) {
  if (x == std::numeric_limits<int64_t>::min()) return false;
  out = x >= 0 ? x : -x;
  return true;
}

LoopSpec build_unary_spec(const TensorImpl& t, std::vector<int64_t>& perm_out) {
  LoopSpec sp;
  const auto& sizes = t.sizes();
  const auto& strides = t.strides();
  const std::size_t ndim = sizes.size();
  sp.ndim = static_cast<int64_t>(ndim);
  if (ndim == 0 || t.numel() == 0) {
    sp.p0 = reinterpret_cast<std::uintptr_t>(t.data());
    return sp;
  }

  // Itemsize as i64
  const std::size_t item_b_sz = t.itemsize();
  const int64_t item_b = static_cast<int64_t>(item_b_sz);

  // Compute signed stride bytes and positive step bytes
  std::vector<int64_t> signed_stride_bytes(ndim, 0);
  std::vector<int64_t> step_bytes(ndim, 0);
  for (std::size_t i = 0; i < ndim; ++i) {
    int64_t tmp = 0;
    if (!checked_mul_i64(strides[i], item_b, tmp)) {
      throw std::overflow_error("strided_loop: overflow computing stride bytes");
    }
    signed_stride_bytes[i] = tmp;
    int64_t abs_st = 0;
    if (!checked_abs_i64(tmp, abs_st)) {
      throw std::overflow_error("strided_loop: overflow computing absolute stride bytes");
    }
    step_bytes[i] = abs_st;
  }

  // Compute starting pointer p0 = data + sum_{i, stride[i]<0} ( (sizes[i]-1) * signed_stride_bytes[i])
  auto* base = static_cast<std::uint8_t*>(t.data());
  std::uint8_t* p = base;
  for (std::size_t i = 0; i < ndim; ++i) {
    if (sizes[i] <= 0) continue;
    if (strides[i] < 0) {
      const int64_t d = sizes[i] - 1;
      int64_t delta = 0;
      if (!checked_mul_i64(d, signed_stride_bytes[i], delta)) {
        throw std::overflow_error("strided_loop: overflow computing start pointer");
      }
      p += static_cast<std::ptrdiff_t>(delta);
    }
  }

  // Build permutation by increasing step_bytes, ignoring size==1 dims
  perm_out.clear();
  perm_out.reserve(ndim);
  for (std::size_t i = 0; i < ndim; ++i) perm_out.push_back(static_cast<int64_t>(i));
  std::stable_sort(perm_out.begin(), perm_out.end(), [&](int64_t a, int64_t b){
    const auto ia = static_cast<std::size_t>(a);
    const auto ib = static_cast<std::size_t>(b);
    const int64_t sa = (sizes[ia] == 1 ? std::numeric_limits<int64_t>::max() : step_bytes[ia]);
    const int64_t sb = (sizes[ib] == 1 ? std::numeric_limits<int64_t>::max() : step_bytes[ib]);
    return sa < sb;
  });
  // Remove size==1 dims from the front of perm (they were set to max above and will end up at the end);
  // Build spec vectors in permuted order for dims with size>1
  for (auto d : perm_out) {
    const std::size_t i = static_cast<std::size_t>(d);
    if (sizes[i] == 1) continue;
    sp.sizes.push_back(sizes[i]);
    sp.step_bytes.push_back(step_bytes[i]);
  }
  sp.ndim = static_cast<int64_t>(sp.sizes.size());
  sp.p0 = reinterpret_cast<std::uintptr_t>(p);
  return sp;
}
LoopSpec build_binary_spec(const TensorImpl& dst, const TensorImpl& src, std::vector<int64_t>& perm_out) {
  if (dst.sizes() != src.sizes()) {
    throw std::invalid_argument("strided_loop: size mismatch in binary spec");
  }
  // Reuse unary builder for dst; perm_out encodes dst's order
  LoopSpec spd = build_unary_spec(dst, perm_out);
  // Validate src has the same ndim and sizes; nothing else to do here
  (void)src;
  return spd;
}

} // namespace core
} // namespace vbt
