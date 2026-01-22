// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/indexing.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "vbt/core/broadcast.h"
#include "vbt/core/checked_math.h"
#include "vbt/core/indexing/index_errors.h"
#include "vbt/core/overlap.h"
#include "vbt/core/strided_loop.h"
#include "vbt/core/view_ops.h"
#include "vbt/core/write_guard.h"

namespace vbt {
namespace core {
namespace indexing {

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::checked_add_i64;
using vbt::core::checked_mul_i64;
using vbt::core::infer_broadcast_shape;

std::int64_t count_specified_dims(const IndexSpec& spec, std::int64_t /*self_dim*/) {
  std::int64_t specified = 0;
  for (const auto& it : spec.items) {
    switch (it.kind) {
      case IndexKind::Integer:
      case IndexKind::Slice:
        ++specified;
        break;
      case IndexKind::Tensor: {
        // Integer tensor index consumes 1 dim; bool mask consumes mask.dim().
        if (it.tensor.dtype() == ScalarType::Bool) {
          specified += static_cast<std::int64_t>(it.tensor.sizes().size());
        } else {
          ++specified;
        }
        break;
      }
      case IndexKind::Boolean:
      case IndexKind::None:
      case IndexKind::Ellipsis:
        // Do not consume a dimension.
        break;
    }
  }
  return specified;
}

IndexSpec expand_ellipsis_and_validate(const IndexSpec& raw, std::int64_t self_dim) {
  const auto n = static_cast<std::int64_t>(raw.items.size());

  // Validate ellipsis count and compute specified dims.
  std::int64_t ellipsis_pos = -1;
  for (std::int64_t i = 0; i < n; ++i) {
    const auto& it = raw.items[static_cast<std::size_t>(i)];
    if (it.kind == IndexKind::Ellipsis) {
      if (ellipsis_pos >= 0) {
        throw std::invalid_argument(errors::kErrMultipleEllipsis);
      }
      ellipsis_pos = i;
    }
  }

  const std::int64_t specified = count_specified_dims(raw, self_dim);
  if (specified > self_dim) {
    throw std::invalid_argument(
        std::string(errors::kErrTooManyIndices) + " " +
        std::to_string(self_dim));
  }

  const std::int64_t missing = self_dim - specified;

  IndexSpec result;
  result.items.reserve(static_cast<std::size_t>(n + std::max<std::int64_t>(missing, 0)));

  if (ellipsis_pos < 0) {
    // No explicit ellipsis: treat as if ellipsis appeared at the end.
    for (const auto& it : raw.items) {
      result.items.push_back(it);
    }
    for (std::int64_t i = 0; i < missing; ++i) {
      result.items.emplace_back(Slice{});  // full-range slice(None)
    }
    return result;
  }

  // Explicit ellipsis: expand in place.
  for (std::int64_t i = 0; i < ellipsis_pos; ++i) {
    result.items.push_back(raw.items[static_cast<std::size_t>(i)]);
  }
  for (std::int64_t i = 0; i < missing; ++i) {
    result.items.emplace_back(Slice{});  // full-range slice(None)
  }
  for (std::int64_t i = ellipsis_pos + 1; i < n; ++i) {
    result.items.push_back(raw.items[static_cast<std::size_t>(i)]);
  }

  return result;
}

namespace {

inline std::int64_t clamp_i64(std::int64_t x, std::int64_t lo, std::int64_t hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

} // anonymous namespace

NormalizedSlice normalize_slice(const Slice& s, std::int64_t dim_size) {
  if (dim_size < 0) {
    throw std::invalid_argument("normalize_slice: negative dim size");
  }

  std::int64_t step = s.step.has_value() ? *s.step : 1;
  if (step == 0) {
    throw std::invalid_argument("slice step cannot be zero");
  }

  NormalizedSlice ns{};
  ns.step = step;

  // Degenerate dimension: any slice is empty.
  if (dim_size == 0) {
    ns.length = 0;
    ns.start = (step > 0) ? 0 : -1;
    return ns;
  }

  const std::int64_t D = dim_size;

  if (step > 0) {
    // Positive step semantics.
    std::int64_t start = s.start.has_value() ? *s.start : 0;
    if (start < 0) start += D;
    start = clamp_i64(start, 0, D);

    std::int64_t stop = s.stop.has_value() ? *s.stop : D;
    if (stop < 0) stop += D;
    stop = clamp_i64(stop, 0, D);

    if (stop <= start) {
      ns.start = start;
      ns.length = 0;
      return ns;
    }

    const std::int64_t diff = stop - start;  // > 0
    const std::int64_t q = diff / step;
    const std::int64_t r = diff % step;
    const std::int64_t len = q + (r != 0 ? 1 : 0);

    ns.start = start;
    ns.length = len;
    return ns;
  }

  // Negative step semantics.
  std::int64_t start = 0;
  if (s.start.has_value()) {
    start = *s.start;
    if (start < 0) start += D;
  } else {
    start = D - 1;
  }
  start = clamp_i64(start, -1, D - 1);

  std::int64_t stop = 0;
  if (s.stop.has_value()) {
    stop = *s.stop;
    if (stop < 0) stop += D;
  } else {
    stop = -1;
  }
  stop = clamp_i64(stop, -1, D - 1);

  if (start <= stop) {
    ns.start = start;
    ns.length = 0;
    return ns;
  }

  const std::int64_t diff = start - stop - 1;  // >= 0
  const std::int64_t step_abs = -step;         // > 0
  const std::int64_t q = diff / step_abs;
  const std::int64_t len = q + 1;

  ns.start = start;
  ns.length = len;
  return ns;
}

TensorImpl basic_index(const TensorImpl& self, const IndexSpec& spec) {
  TensorImpl result = self;
  std::int64_t dim = 0;  // logical dim index into result

  const auto& self_sizes = self.sizes();
  if (self_sizes.empty()) {
    // 0-d special case: only () and None-only indices are allowed.
    if (spec.items.empty()) {
      return self;
    }
    const bool all_none = std::all_of(
        spec.items.begin(), spec.items.end(),
        [](const TensorIndex& it) { return it.kind == IndexKind::None; });
    if (all_none) {
      for (const auto& it : spec.items) {
        (void)it;
        result = vbt::core::unsqueeze(result, dim);
        ++dim;
      }
      return result;
    }
    throw std::out_of_range(errors::kErrInvalidZeroDim);
  }

  for (const auto& it : spec.items) {
    switch (it.kind) {
      case IndexKind::None: {
        result = vbt::core::unsqueeze(result, dim);
        ++dim;
        break;
      }

      case IndexKind::Integer: {
        // Delegate integer bounds/error handling to select().
        result = vbt::core::select(result, dim, it.integer);
        // select removes this dimension → dim stays the same.
        break;
      }

      case IndexKind::Slice: {
        const auto& sizes = result.sizes();
        const auto& strides = result.strides();
        const std::int64_t D = sizes.at(static_cast<std::size_t>(dim));

        NormalizedSlice ns = normalize_slice(it.slice, D);

        auto new_sizes = sizes;
        auto new_strides = strides;

        if (ns.length == 0) {
          new_sizes[static_cast<std::size_t>(dim)] = 0;
          new_strides[static_cast<std::size_t>(dim)] = 1;  // arbitrary when size==0
          result = result.as_strided(new_sizes, new_strides, result.storage_offset());
          ++dim;
          break;
        }

        const std::int64_t stride_d = strides.at(static_cast<std::size_t>(dim));

        std::int64_t new_stride = 0;
        if (!checked_mul_i64(stride_d, ns.step, new_stride)) {
          throw std::overflow_error("basic_index: stride*step overflow");
        }

        std::int64_t offset_add = 0;
        if (!checked_mul_i64(ns.start, stride_d, offset_add)) {
          throw std::overflow_error("basic_index: start*stride overflow");
        }

        std::int64_t new_offset = 0;
        if (!checked_add_i64(result.storage_offset(), offset_add, new_offset)) {
          throw std::overflow_error("basic_index: storage_offset+delta overflow");
        }

        new_sizes[static_cast<std::size_t>(dim)] = ns.length;
        new_strides[static_cast<std::size_t>(dim)] = new_stride;

        result = result.as_strided(new_sizes, new_strides, new_offset);
        ++dim;
        break;
      }

      default:
        // Ellipsis, Boolean, Tensor should never appear post-ellipsis
        // for a basic-only spec. Treat as programmer error.
        throw std::logic_error("basic_index: unexpected index kind in basic path");
    }
  }

  return result;
}

namespace {

TensorImpl broadcast_to_impl(const TensorImpl& src,
                             std::span<const std::int64_t> target_sizes) {
  const auto& src_sizes = src.sizes();
  std::vector<std::int64_t> src_shape(src_sizes.begin(), src_sizes.end());
  std::vector<std::int64_t> tgt_shape(target_sizes.begin(), target_sizes.end());

  // Validate broadcastability using existing helpers, but coerce
  // any failure into a uniform "shape mismatch" error.
  std::vector<std::int64_t> out_shape;
  try {
    out_shape = infer_broadcast_shape(
        std::span<const std::int64_t>(src_shape),
        std::span<const std::int64_t>(tgt_shape));
  } catch (const std::invalid_argument&) {
    throw std::invalid_argument("shape mismatch");
  }

  if (out_shape != tgt_shape) {
    throw std::invalid_argument("shape mismatch");
  }

  // Now apply unsqueeze + expand to build a view.
  TensorImpl result = src;
  const std::int64_t in_rank = static_cast<std::int64_t>(src_shape.size());
  const std::int64_t out_rank = static_cast<std::int64_t>(tgt_shape.size());

  if (out_rank < in_rank) {
    // Defensive; should have been caught by infer_broadcast_shape.
    throw std::invalid_argument("shape mismatch");
  }

  if (out_rank > in_rank) {
    const std::int64_t n_unsq = out_rank - in_rank;
    for (std::int64_t i = 0; i < n_unsq; ++i) {
      result = vbt::core::unsqueeze(result, 0);
    }
  }

  try {
    result = vbt::core::expand(result, tgt_shape);
  } catch (const std::invalid_argument&) {
    throw std::invalid_argument("shape mismatch");
  }

  return result;
}

void copy_into(TensorImpl& dst, const TensorImpl& src) {
  using vbt::core::for_each_1out_1in;

  if (dst.dtype() != src.dtype() ||
      dst.device().type != src.device().type ||
      dst.device().index != src.device().index) {
    throw std::invalid_argument("index assignment: dtype/device mismatch");
  }

  if (dst.numel() != src.numel()) {
    throw std::invalid_argument("shape mismatch");
  }

  const std::int64_t item_b = static_cast<std::int64_t>(dst.itemsize());

  for_each_1out_1in(dst, src, [item_b](std::uint8_t* pd, const std::uint8_t* ps) {
    std::memcpy(pd, ps, static_cast<std::size_t>(item_b));
  });
}

} // anonymous namespace

TensorImpl broadcast_to(const TensorImpl& src,
                        std::span<const std::int64_t> target_sizes) {
  return broadcast_to_impl(src, target_sizes);
}

void basic_index_put(TensorImpl& self,
                     const IndexSpec& spec,
                     const TensorImpl& value) {
  if (self.device().type != kDLCPU) {
    throw std::invalid_argument("basic_index_put: CPU tensors only");
  }

  TensorImpl dst = basic_index(self, spec);  // basic-only spec; view

  const auto& dst_sizes = dst.sizes();
  TensorImpl value_b = broadcast_to_impl(value, dst_sizes);

  vbt::core::check_writable(self);

  if (self.storage().get() == value_b.storage().get()) {
    vbt::core::assert_no_partial_overlap(self, value_b);
  }

  copy_into(dst, value_b);
  if (dst.numel() > 0) {
    self.bump_version();
  }
}

} // namespace indexing
} // namespace core
} // namespace vbt
