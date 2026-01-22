// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/tensor.h"

#include <stdexcept>
#include <limits>

#include "vbt/core/checked_math.h"

namespace vbt {
namespace core {

TensorImpl TensorImpl::as_strided(const std::vector<int64_t>& new_sizes,
                                  const std::vector<int64_t>& new_strides,
                                  int64_t new_storage_offset) const {
  // Basic rank/size guards
  if (new_sizes.size() != new_strides.size()) {
    throw std::invalid_argument("as_strided: sizes/strides length mismatch");
  }
  for (auto s : new_sizes) {
    if (s < 0) throw std::invalid_argument("as_strided: negative size");
  }
  if (new_storage_offset < 0) throw std::invalid_argument("as_strided: negative storage_offset");
  for (auto st : new_strides) {
    if (st == std::numeric_limits<int64_t>::min()) {
      throw std::invalid_argument("as_strided: INT64_MIN stride is invalid");
    }
  }

  // Empty tensors: no storage coverage is required.
  for (auto s : new_sizes) {
    if (s == 0) {
      return TensorImpl{storage_, new_sizes, new_strides, new_storage_offset,
                        dtype_, device_, version_, flags_};
    }
  }

  // Compute min/max element coverage in elements using checked math
  int64_t min_elem_off = 0, max_elem_off = 0;
  for (std::size_t i = 0; i < new_sizes.size(); ++i) {
    const int64_t n = new_sizes[i];
    const int64_t d = n > 0 ? (n - 1) : 0;
    if (d == 0) continue;
    const int64_t st = new_strides[i];
    int64_t term = 0;
    if (!checked_mul_i64(st, d, term)) {
      throw std::overflow_error("as_strided: stride*extent overflow");
    }
    if (st >= 0) {
      int64_t tmp = max_elem_off;
      if (!checked_add_i64(tmp, term, tmp)) throw std::overflow_error("as_strided: max accumulation overflow");
      max_elem_off = tmp;
    } else {
      int64_t tmp = min_elem_off;
      if (!checked_add_i64(tmp, term, tmp)) throw std::overflow_error("as_strided: min accumulation overflow");
      min_elem_off = tmp;
    }
  }
  // max_plus_one
  int64_t max_plus_one = 0;
  if (!checked_add_i64(max_elem_off, 1, max_plus_one)) {
    throw std::overflow_error("as_strided: max+1 overflow");
  }

  // Convert to bytes with bounds; first, ensure itemsize and nbytes fit in int64
  const std::size_t item_b_sz = itemsize();
  if (item_b_sz > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error("as_strided: itemsize too large");
  }
  const int64_t item_b = static_cast<int64_t>(item_b_sz);
  const std::size_t nb_sz = storage_ ? storage_->nbytes() : 0;
  if (nb_sz > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error("as_strided: storage nbytes too large");
  }
  const int64_t nbytes_i64 = static_cast<int64_t>(nb_sz);

  int64_t base_bytes = 0;
  if (!checked_mul_i64(new_storage_offset, item_b, base_bytes)) throw std::overflow_error("as_strided: base byte offset overflow");

  int64_t lower_mul = 0;
  if (!checked_mul_i64(min_elem_off, item_b, lower_mul)) throw std::overflow_error("as_strided: lower mul overflow");
  int64_t lower_bytes = 0;
  if (!checked_add_i64(base_bytes, lower_mul, lower_bytes)) throw std::overflow_error("as_strided: lower add overflow");

  int64_t upper_excl_bytes_term = 0;
  if (!checked_mul_i64(max_plus_one, item_b, upper_excl_bytes_term)) throw std::overflow_error("as_strided: upper mul overflow");
  int64_t upper_excl_bytes = 0;
  if (!checked_add_i64(base_bytes, upper_excl_bytes_term, upper_excl_bytes)) throw std::overflow_error("as_strided: upper add overflow");

  if (lower_bytes < 0) {
    throw std::out_of_range("as_strided: lower bound underflow");
  }
  if (upper_excl_bytes > nbytes_i64) {
    throw std::out_of_range("as_strided: upper bound overflow");
  }

  // Construct view sharing version counter
  return TensorImpl{storage_, new_sizes, new_strides, new_storage_offset,
                    dtype_, device_, version_, flags_};
}

TensorImpl TensorImpl::as_strided_dtype_(
    const std::vector<int64_t>& new_sizes,
    const std::vector<int64_t>& new_strides,
    int64_t new_storage_offset,
    ScalarType new_dtype) const {
  // Basic rank/size guards
  if (new_sizes.size() != new_strides.size()) {
    throw std::invalid_argument("as_strided_dtype_: sizes/strides length mismatch");
  }
  for (auto s : new_sizes) {
    if (s < 0) throw std::invalid_argument("as_strided_dtype_: negative size");
  }
  if (new_storage_offset < 0) {
    throw std::invalid_argument("as_strided_dtype_: negative storage_offset");
  }
  for (auto st : new_strides) {
    if (st == std::numeric_limits<int64_t>::min()) {
      throw std::invalid_argument("as_strided_dtype_: INT64_MIN stride is invalid");
    }
  }

  // Preserve flags, but clear conj bit if new dtype is not complex.
  std::uint8_t out_flags = flags_;
  if (!(new_dtype == ScalarType::Complex64 || new_dtype == ScalarType::Complex128)) {
    out_flags &= static_cast<std::uint8_t>(~kConj);
  }

  // Empty tensors: no storage coverage is required.
  for (auto s : new_sizes) {
    if (s == 0) {
      return TensorImpl{storage_, new_sizes, new_strides, new_storage_offset,
                        new_dtype, device_, version_, out_flags};
    }
  }

  // Compute min/max element coverage in elements using checked math
  int64_t min_elem_off = 0, max_elem_off = 0;
  for (std::size_t i = 0; i < new_sizes.size(); ++i) {
    const int64_t n = new_sizes[i];
    const int64_t d = n > 0 ? (n - 1) : 0;
    if (d == 0) continue;
    const int64_t st = new_strides[i];
    int64_t term = 0;
    if (!checked_mul_i64(st, d, term)) {
      throw std::overflow_error("as_strided_dtype_: stride*extent overflow");
    }
    if (st >= 0) {
      int64_t tmp = max_elem_off;
      if (!checked_add_i64(tmp, term, tmp)) {
        throw std::overflow_error("as_strided_dtype_: max accumulation overflow");
      }
      max_elem_off = tmp;
    } else {
      int64_t tmp = min_elem_off;
      if (!checked_add_i64(tmp, term, tmp)) {
        throw std::overflow_error("as_strided_dtype_: min accumulation overflow");
      }
      min_elem_off = tmp;
    }
  }

  // max_plus_one
  int64_t max_plus_one = 0;
  if (!checked_add_i64(max_elem_off, 1, max_plus_one)) {
    throw std::overflow_error("as_strided_dtype_: max+1 overflow");
  }

  // Convert to bytes with bounds; first, ensure itemsize and nbytes fit in int64.
  const std::size_t item_b_sz = vbt::core::itemsize(new_dtype);
  if (item_b_sz > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error("as_strided_dtype_: itemsize too large");
  }
  const int64_t item_b = static_cast<int64_t>(item_b_sz);

  const std::size_t nb_sz = storage_ ? storage_->nbytes() : 0;
  if (nb_sz > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error("as_strided_dtype_: storage nbytes too large");
  }
  const int64_t nbytes_i64 = static_cast<int64_t>(nb_sz);

  int64_t base_bytes = 0;
  if (!checked_mul_i64(new_storage_offset, item_b, base_bytes)) {
    throw std::overflow_error("as_strided_dtype_: base byte offset overflow");
  }

  int64_t lower_mul = 0;
  if (!checked_mul_i64(min_elem_off, item_b, lower_mul)) {
    throw std::overflow_error("as_strided_dtype_: lower mul overflow");
  }
  int64_t lower_bytes = 0;
  if (!checked_add_i64(base_bytes, lower_mul, lower_bytes)) {
    throw std::overflow_error("as_strided_dtype_: lower add overflow");
  }

  int64_t upper_excl_bytes_term = 0;
  if (!checked_mul_i64(max_plus_one, item_b, upper_excl_bytes_term)) {
    throw std::overflow_error("as_strided_dtype_: upper mul overflow");
  }
  int64_t upper_excl_bytes = 0;
  if (!checked_add_i64(base_bytes, upper_excl_bytes_term, upper_excl_bytes)) {
    throw std::overflow_error("as_strided_dtype_: upper add overflow");
  }

  if (lower_bytes < 0) {
    throw std::out_of_range("as_strided_dtype_: lower bound underflow");
  }
  if (upper_excl_bytes > nbytes_i64) {
    throw std::out_of_range("as_strided_dtype_: upper bound overflow");
  }

  return TensorImpl{storage_, new_sizes, new_strides, new_storage_offset,
                    new_dtype, device_, version_, out_flags};
}

} // namespace core
} // namespace vbt
