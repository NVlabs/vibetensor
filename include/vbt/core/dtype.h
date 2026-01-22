// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

// Reuse DLPack types; do not define our own enums
#include <dlpack/dlpack.h>

namespace vbt {
namespace core {

// Scalar type tag (append-only for ABI stability)
enum class ScalarType : uint8_t {
  Bool,
  Int32,
  Int64,
  Float32,
  Float16,
  BFloat16,
  Float64,
  Complex64,
  Complex128,
  Undefined = 255
};

// ABI pinning: keep ordinals stable (append-only).
static_assert(static_cast<uint8_t>(ScalarType::Bool) == 0);
static_assert(static_cast<uint8_t>(ScalarType::Int32) == 1);
static_assert(static_cast<uint8_t>(ScalarType::Int64) == 2);
static_assert(static_cast<uint8_t>(ScalarType::Float32) == 3);
static_assert(static_cast<uint8_t>(ScalarType::Float16) == 4);
static_assert(static_cast<uint8_t>(ScalarType::BFloat16) == 5);
static_assert(static_cast<uint8_t>(ScalarType::Float64) == 6);
static_assert(static_cast<uint8_t>(ScalarType::Complex64) == 7);
static_assert(static_cast<uint8_t>(ScalarType::Complex128) == 8);
static_assert(static_cast<uint8_t>(ScalarType::Undefined) == 255);

inline constexpr bool is_valid_scalar(ScalarType /*t*/) { return true; }

// Itemsize helper
inline constexpr std::size_t itemsize(ScalarType t) {
  switch (t) {
    case ScalarType::Bool: return 1;
    case ScalarType::Int32: return 4;
    case ScalarType::Int64: return 8;
    case ScalarType::Float32: return 4;
    case ScalarType::Float16: return 2;
    case ScalarType::BFloat16: return 2;
    case ScalarType::Float64: return 8;
    case ScalarType::Complex64: return 8;
    case ScalarType::Complex128: return 16;
  }
  return 0;
}

// Map to DLPack dtype
inline constexpr DLDataType to_dlpack_dtype(ScalarType t) {
  switch (t) {
    case ScalarType::Bool: return DLDataType{static_cast<uint8_t>(kDLBool), 8, 1};
    case ScalarType::Int32: return DLDataType{static_cast<uint8_t>(kDLInt), 32, 1};
    case ScalarType::Int64: return DLDataType{static_cast<uint8_t>(kDLInt), 64, 1};
    case ScalarType::Float32: return DLDataType{static_cast<uint8_t>(kDLFloat), 32, 1};
    case ScalarType::Float16: return DLDataType{static_cast<uint8_t>(kDLFloat), 16, 1};
    case ScalarType::Float64: return DLDataType{static_cast<uint8_t>(kDLFloat), 64, 1};
    case ScalarType::Complex64: return DLDataType{static_cast<uint8_t>(kDLComplex), 64, 1};
    case ScalarType::Complex128: return DLDataType{static_cast<uint8_t>(kDLComplex), 128, 1};
#if VBT_HAS_DLPACK_BF16
    case ScalarType::BFloat16:
      return DLDataType{static_cast<uint8_t>(kDLBfloat), 16, 1};
#endif
  }
  return DLDataType{static_cast<uint8_t>(kDLFloat), 0, 1};
}

inline constexpr std::optional<ScalarType> from_dlpack_dtype(const DLDataType& dt) {
  if (dt.lanes != 1) return std::nullopt;
  if (dt.code == static_cast<uint8_t>(kDLBool) && dt.bits == 8) return ScalarType::Bool;
  if (dt.code == static_cast<uint8_t>(kDLInt) && dt.bits == 32) return ScalarType::Int32;
  if (dt.code == static_cast<uint8_t>(kDLInt) && dt.bits == 64) return ScalarType::Int64;
  if (dt.code == static_cast<uint8_t>(kDLFloat) && dt.bits == 32) return ScalarType::Float32;
  if (dt.code == static_cast<uint8_t>(kDLFloat) && dt.bits == 16) return ScalarType::Float16;
  if (dt.code == static_cast<uint8_t>(kDLFloat) && dt.bits == 64) return ScalarType::Float64;
  if (dt.code == static_cast<uint8_t>(kDLComplex) && dt.bits == 64) return ScalarType::Complex64;
  if (dt.code == static_cast<uint8_t>(kDLComplex) && dt.bits == 128) return ScalarType::Complex128;
#if VBT_HAS_DLPACK_BF16
  if (dt.code == static_cast<uint8_t>(kDLBfloat) && dt.bits == 16) return ScalarType::BFloat16;
#endif
  return std::nullopt;
}

} // namespace core
} // namespace vbt
