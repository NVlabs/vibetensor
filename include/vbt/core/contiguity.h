// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace vbt {
namespace core {

// Compute PyTorch-style non-overlapping-and-dense (NO&D) for strided tensors.
// Ported from c10::_compute_non_overlapping_and_dense (c10/core/Contiguity.h:273-307).
// - Returns true for zero-size tensors and size-1 dimensions.
// - For 1D: true if size < 2 or stride == 1.
// - Negative strides will not satisfy the required positive stride progression
//   (except when numel is 0 or 1 as covered by size<2 cases).
bool compute_non_overlapping_and_dense(const std::vector<int64_t>& sizes,
                                       const std::vector<int64_t>& strides) noexcept;

} // namespace core
} // namespace vbt
