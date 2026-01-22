// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/broadcast.h"

#include <stdexcept>

namespace vbt {
namespace core {

std::string shape_to_string(std::span<const std::int64_t> sizes) {
  std::string s;
  s.push_back('(');
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    if (i != 0) {
      s.append(", ");
    }
    s.append(std::to_string(sizes[i]));
  }
  s.push_back(')');
  return s;
}

std::vector<std::int64_t> infer_broadcast_shape(
    std::span<const std::int64_t> a,
    std::span<const std::int64_t> b) {
  const std::size_t na = a.size();
  const std::size_t nb = b.size();
  const std::size_t n  = na > nb ? na : nb;

  std::vector<std::int64_t> rev;
  rev.reserve(n);

  for (std::size_t i = 0; i < n; ++i) {
    const std::int64_t sa = (i < na) ? a[na - 1 - i] : 1;
    const std::int64_t sb = (i < nb) ? b[nb - 1 - i] : 1;

    if (sa < 0 || sb < 0) {
      throw std::invalid_argument("infer_broadcast_shape: negative dimension not supported");
    }

    std::int64_t out_dim = 0;
    if (sa == sb) {
      out_dim = sa;
    } else if (sa == 1) {
      out_dim = sb;
    } else if (sb == 1) {
      out_dim = sa;
    } else {
      // Shapes are not broadcastable at this dimension.
      throw std::invalid_argument(
          "infer_broadcast_shape: shapes " + shape_to_string(a) +
          " and " + shape_to_string(b) +
          " are not broadcastable; operands must have the same shape "
          "or be broadcastable under PyTorch-style rules");
    }

    // Zero-sized dimensions participate normally; the above rules already
    // encode the {0,0}/{0,1}/{1,0} vs {0,K>1} behavior:
    // - {0,0} -> equal branch
    // - {0,1}/{1,0} -> broadcast-from-1 branch
    // - {0,K>1}/{K>1,0} -> fall into the error branch.
    rev.push_back(out_dim);
  }

  // Reverse to restore the original left-to-right order.
  std::vector<std::int64_t> result;
  result.reserve(rev.size());
  for (std::size_t i = 0; i < rev.size(); ++i) {
    result.push_back(rev[rev.size() - 1 - i]);
  }
  return result;
}

std::vector<std::int64_t> infer_broadcast_shape_nary(
    std::span<const std::vector<std::int64_t>> shapes) {
  if (shapes.empty()) {
    return {};
  }

  std::vector<std::int64_t> result = shapes[0];
  for (std::size_t i = 1; i < shapes.size(); ++i) {
    result = infer_broadcast_shape(std::span<const std::int64_t>(result),
                                   std::span<const std::int64_t>(shapes[i]));
  }
  return result;
}

} // namespace core
} // namespace vbt
