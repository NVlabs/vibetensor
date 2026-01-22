// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vbt {
namespace core {

// Format a shape like (2, 3, 4) for error messages.
std::string shape_to_string(std::span<const std::int64_t> sizes);

// Compute the broadcasted shape of two operand shapes using
// PyTorch-style right-aligned broadcasting rules.
//
// For each dimension from the right:
// - Accept if sizes are equal.
// - Accept if one of the sizes is 1 (broadcast).
// - Reject otherwise.
//
// Zero-sized dimensions participate in broadcasting:
// - {0, 0}, {0, 1}, {1, 0} -> 0
// - {0, K>1} or {K>1, 0} -> error.
std::vector<std::int64_t> infer_broadcast_shape(
    std::span<const std::int64_t> a,
    std::span<const std::int64_t> b);

// Compute a common broadcasted shape for N operand shapes by folding
// pairwise via infer_broadcast_shape.
std::vector<std::int64_t> infer_broadcast_shape_nary(
    std::span<const std::vector<std::int64_t>> shapes);

} // namespace core
} // namespace vbt
