// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vbt {
namespace interop {
namespace safetensors {

enum class DType : std::uint8_t {
  BOOL,
  F4,
  F6_E2M3,
  F6_E3M2,
  U8,
  I8,
  F8_E5M2,
  F8_E4M3,
  F8_E8M0,
  I16,
  U16,
  F16,
  BF16,
  I32,
  U32,
  F32,
  C64,
  F64,
  I64,
  U64,
};

[[nodiscard]] inline constexpr std::size_t dtype_bits(DType dt) {
  switch (dt) {
    case DType::F4: return 4;
    case DType::F6_E3M2: return 6;
    case DType::F6_E2M3: return 6;
    case DType::BOOL: return 8;
    case DType::U8: return 8;
    case DType::I8: return 8;
    case DType::F8_E5M2: return 8;
    case DType::F8_E4M3: return 8;
    case DType::F8_E8M0: return 8;
    case DType::I16: return 16;
    case DType::U16: return 16;
    case DType::I32: return 32;
    case DType::U32: return 32;
    case DType::I64: return 64;
    case DType::U64: return 64;
    case DType::F16: return 16;
    case DType::BF16: return 16;
    case DType::F32: return 32;
    case DType::F64: return 64;
    case DType::C64: return 64;
  }
  return 0;
}

[[nodiscard]] inline constexpr std::size_t dtype_bytes_ceil(DType dt) {
  const std::size_t bits = dtype_bits(dt);
  return (bits + 7) / 8;
}

struct TensorInfo {
  DType dtype;
  std::vector<std::size_t> shape;
  // Offsets are relative to the start of the *data buffer* (bytes after header).
  std::pair<std::size_t, std::size_t> data_offsets;
};

struct TensorEntry {
  std::string name;
  TensorInfo info;

  // Optional data payload, used by `serialize(...)`. For entries originating from
  // `read_metadata(...)` / `SafeTensorsView`, this span is empty and the actual
  // bytes live in the file buffer.
  std::span<const std::byte> data;
};

struct TensorView {
  DType dtype;
  std::vector<std::size_t> shape;
  std::pair<std::size_t, std::size_t> data_offsets;
  std::span<const std::byte> data;
};

struct Metadata {
  // Optional __metadata__ map. Store as sorted vector to avoid hash-flooding.
  std::optional<std::vector<std::pair<std::string, std::string>>> user_metadata;

  // Canonical list sorted by (begin,end) and then name.
  std::vector<TensorEntry> tensors_by_offset;

  // Sorted vector of (name, index_into_tensors_by_offset) for O(log n) lookup.
  std::vector<std::pair<std::string, std::size_t>> name_index;

  std::size_t data_len_bytes = 0;
};

} // namespace safetensors
} // namespace interop
} // namespace vbt
