// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>

#include "vbt/core/tensor.h"

namespace vbt {
namespace core {

enum class MemOverlap { No, Yes, TooHard };

enum class MemOverlapStatus { Full, Partial, No, TooHard };

// Expose storage base pointer helper (mirrors TU-local impl in overlap.cc)
// For non-empty tensors, data() points at base + itemsize * storage_offset.
// Returns nullptr when data() is null.
inline const std::uint8_t* base_ptr_of(const TensorImpl& t) noexcept {
  auto* p = static_cast<const std::uint8_t*>(t.data());
  if (!p) return nullptr;
  return p - (t.itemsize() * static_cast<std::size_t>(t.storage_offset()));
}

MemOverlap has_internal_overlap(const TensorImpl& t) noexcept;
MemOverlapStatus get_overlap_status(const TensorImpl& a, const TensorImpl& b) noexcept;

void assert_no_internal_overlap(const TensorImpl& t);
void assert_no_partial_overlap(const TensorImpl& a, const TensorImpl& b);
void assert_no_overlap(const TensorImpl& a, const TensorImpl& b);

} // namespace core
} // namespace vbt
