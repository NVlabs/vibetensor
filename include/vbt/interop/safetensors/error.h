// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace vbt {
namespace interop {
namespace safetensors {

enum class ErrorCode : std::uint8_t {
  // Explicit numbering to keep numeric codes stable for future bindings.
  // Policy: append-only.

  // Framing/header
  HeaderTooSmall = 0,
  HeaderTooLarge = 1,
  InvalidHeaderLength = 2,
  InvalidHeaderUtf8 = 3,
  InvalidHeaderStart = 4,
  InvalidHeaderDeserialization = 5,

  // Validation
  InvalidOffset = 6,
  TensorInvalidInfo = 7,
  MetadataIncompleteBuffer = 8,
  ValidationOverflow = 9,
  MisalignedSlice = 10,

  // Lookup
  TensorNotFound = 11,

  // I/O
  IoError = 12,
  FileTooLarge = 13,
  NotRegularFile = 14,

  // VBT mapping
  UnsupportedDtypeForVbt = 15,
};

namespace detail {
inline constexpr std::size_t kMaxErrorWhatBytes = 512;
inline constexpr std::size_t kMaxTensorNameBytes = 256;

inline std::string truncate_bytes(std::string s, std::size_t max) {
  if (s.size() <= max) return s;
  s.resize(max);
  return s;
}
} // namespace detail

class SafeTensorsError : public std::runtime_error {
 public:
  SafeTensorsError(ErrorCode code, std::string message, std::string_view tensor_name = {})
      : std::runtime_error(detail::truncate_bytes(std::move(message), detail::kMaxErrorWhatBytes)),
        code_(code),
        tensor_name_(std::string(tensor_name.substr(0, detail::kMaxTensorNameBytes))) {}

  ErrorCode code() const noexcept { return code_; }
  std::string_view tensor_name() const noexcept { return tensor_name_; }

 private:
  ErrorCode code_;
  std::string tensor_name_;
};

} // namespace safetensors
} // namespace interop
} // namespace vbt
