// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "vbt/interop/safetensors/safetensors.h"

namespace vbt {
namespace interop {
namespace safetensors {
namespace detail {

struct PreparedTensor {
  const TensorEntry* entry = nullptr;
  std::size_t begin = 0;
  std::size_t end = 0;
};

struct PreparedSerialization {
  // Padded to a multiple of 8 bytes with spaces.
  std::vector<std::byte> header_bytes;

  // In serialization order.
  std::vector<PreparedTensor> tensors;

  // Expected number of data bytes computed from dtype/shape.
  std::size_t expected_data_bytes = 0;

  // Sum of TensorEntry::data.size() in serialization order.
  std::size_t actual_data_bytes = 0;
};

[[nodiscard]] PreparedSerialization prepare_serialization(
    std::span<const TensorEntry> tensors,
    const std::optional<std::vector<std::pair<std::string, std::string>>>& user_metadata,
    SerializeOptions opts);

} // namespace detail
} // namespace safetensors
} // namespace interop
} // namespace vbt
