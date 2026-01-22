// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>

#include "vbt/core/dtype.h"

namespace vbt {
namespace core {

inline constexpr bool is_complex(ScalarType dt) noexcept {
  return dt == ScalarType::Complex64 || dt == ScalarType::Complex128;
}

inline constexpr ScalarType to_real_value_type(ScalarType dt) noexcept {
  switch (dt) {
    case ScalarType::Complex64: return ScalarType::Float32;
    case ScalarType::Complex128: return ScalarType::Float64;
    default: return dt;
  }
}

inline constexpr ScalarType to_complex_type(ScalarType dt) noexcept {
  switch (dt) {
    case ScalarType::Float32: return ScalarType::Complex64;
    case ScalarType::Float64: return ScalarType::Complex128;
    default: return ScalarType::Undefined;
  }
}

inline constexpr std::size_t required_alignment_bytes(ScalarType dt) noexcept {
  // For the current dtype set, alignment requirements match itemsize.
  return itemsize(dt);
}

inline constexpr bool is_floating_point(ScalarType dt) noexcept {
  switch (dt) {
    case ScalarType::Float16:
    case ScalarType::BFloat16:
    case ScalarType::Float32:
    case ScalarType::Float64: return true;
    default: return false;
  }
}

inline constexpr bool is_integral(ScalarType dt) noexcept {
  return dt == ScalarType::Int32 || dt == ScalarType::Int64;
}

inline constexpr bool is_integral_or_bool(ScalarType dt) noexcept {
  return dt == ScalarType::Bool || is_integral(dt);
}

struct ResultTypeState {
  bool has_bool  = false;
  bool has_int32 = false;
  bool has_int64 = false;
  bool has_f16   = false;
  bool has_bf16  = false;
  bool has_f32   = false;
  bool has_f64   = false;
  bool has_c64   = false;
  bool has_c128  = false;
};

inline void update_result_type_state(ResultTypeState& s, ScalarType dt) {
  switch (dt) {
    case ScalarType::Bool:
      s.has_bool = true;
      break;
    case ScalarType::Int32:
      s.has_int32 = true;
      break;
    case ScalarType::Int64:
      s.has_int64 = true;
      break;
    case ScalarType::Float16:
      s.has_f16 = true;
      break;
    case ScalarType::BFloat16:
      s.has_bf16 = true;
      break;
    case ScalarType::Float32:
      s.has_f32 = true;
      break;
    case ScalarType::Float64:
      s.has_f64 = true;
      break;
    case ScalarType::Complex64:
      s.has_c64 = true;
      break;
    case ScalarType::Complex128:
      s.has_c128 = true;
      break;
    default:
      throw std::logic_error("type_promotion: unsupported dtype");
  }
}

inline ScalarType result_type(const ResultTypeState& s) {
  if (!s.has_bool && !s.has_int32 && !s.has_int64 && !s.has_f16 && !s.has_bf16 &&
      !s.has_f32 && !s.has_f64 && !s.has_c64 && !s.has_c128) {
    throw std::logic_error("type_promotion: result_type called with empty state");
  }

  // Complex dominates real.
  if (s.has_c128) {
    return ScalarType::Complex128;
  }
  if (s.has_c64) {
    // Complex64 + Float64 promotes to Complex128 (preserve higher real type).
    if (s.has_f64) {
      return ScalarType::Complex128;
    }
    return ScalarType::Complex64;
  }

  // Float64 dominates smaller floating/integer types.
  if (s.has_f64) {
    return ScalarType::Float64;
  }

  // Legacy 6-type lattice: {Bool, Int32, Int64, Float16, BFloat16, Float32}.
  const bool any_int_non_bool = s.has_int32 || s.has_int64;
  const bool any_float = s.has_f16 || s.has_bf16 || s.has_f32;

  // Bool-only inputs stay Bool.
  if (s.has_bool && !any_int_non_bool && !any_float) {
    return ScalarType::Bool;
  }

  if (!any_float) {
    // Integer (and possibly bool) inputs only: choose Int64 if present,
    // otherwise Int32.
    if (s.has_int64) {
      return ScalarType::Int64;
    }
    return ScalarType::Int32;
  }

  // At least one floating type present.
  if (s.has_f32) {
    return ScalarType::Float32;
  }
  const bool any_f16 = s.has_f16;
  const bool any_bf16 = s.has_bf16;
  if (any_f16 && any_bf16) {
    // Mixed 16-bit floats upcast to Float32 for safety.
    return ScalarType::Float32;
  }
  if (any_f16) {
    return ScalarType::Float16;
  }
  if (any_bf16) {
    return ScalarType::BFloat16;
  }

  // Fallback – should be unreachable but keep behavior explicit.
  return ScalarType::Float32;
}

inline ScalarType promote_types(ScalarType a, ScalarType b) {
  ResultTypeState s;
  update_result_type_state(s, a);
  update_result_type_state(s, b);
  return result_type(s);
}

inline bool can_cast(ScalarType from, ScalarType to) noexcept {
  if (from == to) {
    return true;
  }

  // Pinned rule: no imaginary drop.
  if (is_complex(from) && !is_complex(to)) {
    return false;
  }

  // Complex <-> complex casts are allowed (including narrowing/widening).
  if (is_complex(from) && is_complex(to)) {
    return true;
  }

  // Bool can be cast to any integer, floating, or complex type.
  if (from == ScalarType::Bool) {
    if (to == ScalarType::Bool) {
      return true;
    }
    if (is_integral(to) || is_floating_point(to) || is_complex(to)) {
      return true;
    }
    return false;
  }

  if (is_integral(from)) {
    // Do not allow narrowing to Bool.
    if (to == ScalarType::Bool) {
      return false;
    }

    if (is_integral(to)) {
      // Int32 -> Int64 is allowed; Int64 -> Int32 is not.
      if (from == ScalarType::Int32 && to == ScalarType::Int64) {
        return true;
      }
      return false;
    }

    // Integer to floating/complex upcast.
    if (is_floating_point(to) || is_complex(to)) {
      return true;
    }

    return false;
  }

  if (is_floating_point(from)) {
    // Float<->float casts are allowed.
    if (is_floating_point(to)) {
      return true;
    }
    // Real -> complex is allowed.
    if (is_complex(to)) {
      return true;
    }
    // Disallow float -> integer/bool.
    return false;
  }

  // Anything else is disallowed by default.
  return false;
}

inline ScalarType opmath_dtype(ScalarType dt) noexcept {
  if (dt == ScalarType::Float16 || dt == ScalarType::BFloat16) {
    return ScalarType::Float32;
  }
  return dt;
}

} // namespace core
} // namespace vbt
