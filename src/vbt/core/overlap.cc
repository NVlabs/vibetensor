// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/overlap.h"

#include <algorithm>
#include <stdexcept>
#include <limits>

#include "vbt/core/checked_math.h"

namespace vbt {
namespace core {

namespace {

bool compute_span_bytes(const TensorImpl& t, int64_t& L, int64_t& U) noexcept {
  L = 0; U = 0;
  const auto& sizes = t.sizes();
  const auto& strides = t.strides();
  if (sizes.size() != strides.size()) return false;

  int64_t min_elem_off = 0, max_elem_off = 0;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    const int64_t n = sizes[i];
    const int64_t d = n > 0 ? (n - 1) : 0;
    if (d == 0) continue;
    const int64_t st = strides[i];
    if (st == std::numeric_limits<int64_t>::min()) return false;
    int64_t term = 0;
    if (!checked_mul_i64(st, d, term)) return false;
    if (st >= 0) {
      int64_t tmp = max_elem_off;
      if (!checked_add_i64(tmp, term, tmp)) return false;
      max_elem_off = tmp;
    } else {
      int64_t tmp = min_elem_off;
      if (!checked_add_i64(tmp, term, tmp)) return false;
      min_elem_off = tmp;
    }
  }
  int64_t max_plus_one = 0;
  if (!checked_add_i64(max_elem_off, 1, max_plus_one)) return false;

  const std::size_t item_b_sz = t.itemsize();
  if (item_b_sz > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) return false;
  const int64_t item_b = static_cast<int64_t>(item_b_sz);

  int64_t base_b = 0;
  if (!checked_mul_i64(t.storage_offset(), item_b, base_b)) return false;

  int64_t lower_mul = 0;
  if (!checked_mul_i64(min_elem_off, item_b, lower_mul)) return false;
  if (!checked_add_i64(base_b, lower_mul, L)) return false;

  int64_t upper_mul = 0;
  if (!checked_mul_i64(max_plus_one, item_b, upper_mul)) return false;
  if (!checked_add_i64(base_b, upper_mul, U)) return false;

  return true;
}

} // namespace

MemOverlap has_internal_overlap(const TensorImpl& t) noexcept {
  // Parity with ATen: if NO&D, return No; otherwise check zero strides; else TooHard
  if (t.is_non_overlapping_and_dense_or_false()) {
    return MemOverlap::No;
  }
  const auto& sizes = t.sizes();
  const auto& strides = t.strides();
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    // size>1 and stride==0 implies definite internal overlap
    if (sizes[i] > 1 && strides[i] == 0) {
      return MemOverlap::Yes;
    }
  }
  return MemOverlap::TooHard;
}

MemOverlapStatus get_overlap_status(const TensorImpl& a, const TensorImpl& b) noexcept {
  if (&a == &b) return MemOverlapStatus::Full;
  if (a.numel() == 0 || b.numel() == 0) return MemOverlapStatus::No;

  // Storage equality via base pointer equality.
  // If the base pointers differ, the tensors cannot overlap, even if either
  // tensor is strided/non-dense.
  // Invariant: distinct Storage objects never alias unless their base pointers
  // compare equal.
  const auto* a_base = base_ptr_of(a);
  const auto* b_base = base_ptr_of(b);
  if (a_base && b_base && a_base != b_base) {
    return MemOverlapStatus::No;
  }

  // If the tensors share storage (or we failed to compute base pointers), fall
  // back to the conservative NO&D-based overlap check.
  if (!a.is_non_overlapping_and_dense_or_false() || !b.is_non_overlapping_and_dense_or_false()) {
    return MemOverlapStatus::TooHard;
  }
  if (a_base && b_base && a_base == b_base) {
    const auto a_begin = static_cast<const char*>(a.data());
    const auto a_end = a_begin + a.numel() * static_cast<std::ptrdiff_t>(a.itemsize());
    const auto b_begin = static_cast<const char*>(b.data());
    const auto b_end = b_begin + b.numel() * static_cast<std::ptrdiff_t>(b.itemsize());

    if (a_begin == b_begin && a_end == b_end) {
      return (a.strides() == b.strides()) ? MemOverlapStatus::Full : MemOverlapStatus::Partial;
    }
    if (a_begin < b_end && b_begin < a_end) {
      return MemOverlapStatus::Partial;
    }
  }
  return MemOverlapStatus::No;
}

void assert_no_internal_overlap(const TensorImpl& t) {
  // Exact message from ATen
  if (has_internal_overlap(t) == MemOverlap::Yes) {
    throw std::invalid_argument(
      "unsupported operation: more than one element of the written-to tensor "
      "refers to a single memory location. Please clone() the tensor before "
      "performing the operation.");
  }
}

void assert_no_partial_overlap(const TensorImpl& a, const TensorImpl& b) {
  if (get_overlap_status(a, b) == MemOverlapStatus::Partial) {
    throw std::invalid_argument(
      "unsupported operation: some elements of the input tensor and "
      "the written-to tensor refer to a single memory location. "
      "Please clone() the tensor before performing the operation.");
  }
}

void assert_no_overlap(const TensorImpl& a, const TensorImpl& b) {
  const auto lap = get_overlap_status(a, b);
  if (lap == MemOverlapStatus::Partial || lap == MemOverlapStatus::Full) {
    throw std::invalid_argument(
      "unsupported operation: some elements of the input tensor and "
      "the written-to tensor refer to a single memory location. "
      "Please clone() the tensor before performing the operation.");
  }
}

} // namespace core
} // namespace vbt
