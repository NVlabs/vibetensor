// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <cmath>

#include <node_api.h>

#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/checked_math.h"

namespace vbt::node {

// Parse a dtype string into a ScalarType.
// - When js_val is undefined/null/nullptr, defaults to Float32.
// - On success: returns a value.
// - On failure: sets a JS TypeError and returns std::nullopt.
inline std::optional<vbt::core::ScalarType> ParseDType(napi_env env, napi_value js_val);

// Parse a device string into a Device.
// - On unsupported devices: sets a JS TypeError and returns std::nullopt.
inline std::optional<vbt::core::Device> ParseDeviceCpuOnly(napi_env env, napi_value js_val);

// Parse a JS array of non-negative integers into sizes.
// - On failure: sets a JS TypeError and returns false.
inline bool ParseSizes(napi_env env, napi_value js_array, std::vector<int64_t>* out_sizes);

// Compute bytes = numel * itemsize with overflow protection.
// - On overflow: returns false and does NOT set a JS error.
inline bool NumelBytes(const std::vector<int64_t>& sizes,
                       vbt::core::ScalarType dtype,
                       std::size_t* out_nbytes);

// Error helpers: always set a JS exception and return nullptr.
inline napi_value ThrowTypeError(napi_env env, const char* msg);
inline napi_value ThrowRuntimeError(napi_env env, const char* msg);

// N-API status check used everywhere; on failure it throws a JS Error and
// returns false so callers can early-return.
inline bool CheckNapiOkImpl(napi_env env, napi_status st, const char* context);

#define CHECK_NAPI_OK(env, st, context)                                        \
  do {                                                                         \
    if (!vbt::node::CheckNapiOkImpl((env), (st), (context))) return nullptr;   \
  } while (0)

// ==== Inline definitions =====================================================

namespace detail {

inline std::string GetStringUtf8(napi_env env,
                                 napi_value value,
                                 const char* context,
                                 bool* ok) {
  if (ok) *ok = false;
  if (!value) {
    if (ok) *ok = true;
    return std::string();
  }

  size_t len = 0;
  napi_status st = napi_get_value_string_utf8(env, value, nullptr, 0, &len);
  if (st != napi_ok) {
    // Fallback: generic error; caller is free to add more context.
    napi_throw_error(env, nullptr, context);
    return std::string();
  }

  // Allocate room for the UTF-8 bytes plus the terminating NUL that N-API
  // may write.
  std::string result;
  result.resize(len + 1);
  st = napi_get_value_string_utf8(env, value, result.data(), result.size(), &len);
  if (st != napi_ok) {
    napi_throw_error(env, nullptr, context);
    return std::string();
  }

  // Drop the NUL terminator; the logical string length is `len`.
  result.resize(len);
  if (ok) *ok = true;
  return result;
}

} // namespace detail

inline napi_value ThrowTypeError(napi_env env, const char* msg) {
  napi_throw_type_error(env, nullptr, msg);
  return nullptr;
}

inline napi_value ThrowRuntimeError(napi_env env, const char* msg) {
  napi_throw_error(env, nullptr, msg);
  return nullptr;
}

inline bool CheckNapiOkImpl(napi_env env, napi_status st, const char* context) {
  if (st == napi_ok) return true;

  const napi_extended_error_info* info = nullptr;
  napi_get_last_error_info(env, &info);
  const char* base = info && info->error_message ? info->error_message : "napi call failed";
  std::string msg = std::string(context ? context : "") + ": " + base;
  napi_throw_error(env, nullptr, msg.c_str());
  return false;
}

inline std::optional<vbt::core::ScalarType> ParseDType(napi_env env, napi_value js_val) {
  using vbt::core::ScalarType;

  if (!js_val) return ScalarType::Float32;

  napi_valuetype t;
  napi_status st = napi_typeof(env, js_val, &t);
  if (!CheckNapiOkImpl(env, st, "ParseDType/typeof")) return std::nullopt;

  if (t == napi_undefined || t == napi_null) return ScalarType::Float32;
  if (t != napi_string) {
    ThrowTypeError(env, "dtype must be a string");
    return std::nullopt;
  }

  bool ok = false;
  std::string token = detail::GetStringUtf8(env, js_val, "ParseDType/string", &ok);
  if (!ok) return std::nullopt;  // Error already thrown.

  if (token.empty() || token == "float32") return ScalarType::Float32;
  if (token == "int32") return ScalarType::Int32;
  if (token == "int64") return ScalarType::Int64;
  if (token == "bool") return ScalarType::Bool;

  ThrowTypeError(env,
                 "unsupported dtype: expected one of {float32,int32,int64,bool}");
  return std::nullopt;
}

inline std::optional<vbt::core::Device> ParseDeviceCpuOnly(napi_env env, napi_value js_val) {
  using vbt::core::Device;

  if (!js_val) return Device::cpu();

  napi_valuetype t;
  napi_status st = napi_typeof(env, js_val, &t);
  if (!CheckNapiOkImpl(env, st, "ParseDeviceCpuOnly/typeof")) return std::nullopt;

  if (t == napi_undefined || t == napi_null) return Device::cpu();
  if (t != napi_string) {
    ThrowTypeError(env, "device must be a string like 'cpu'");
    return std::nullopt;
  }

  bool ok = false;
  std::string token = detail::GetStringUtf8(env, js_val, "ParseDeviceCpuOnly/string", &ok);
  if (!ok) return std::nullopt;

  if (token.empty() || token == "cpu" || token == "cpu:0") return Device::cpu();

  ThrowTypeError(env, "zeros: only cpu device supported");
  return std::nullopt;
}

inline bool ParseSizes(napi_env env, napi_value js_array, std::vector<int64_t>* out_sizes) {
  if (!out_sizes) return false;

  bool is_array = false;
  napi_status st = napi_is_array(env, js_array, &is_array);
  if (!CheckNapiOkImpl(env, st, "ParseSizes/is_array")) return false;
  if (!is_array) {
    ThrowTypeError(env, "sizes must be an array of non-negative integers");
    return false;
  }

  uint32_t length = 0;
  st = napi_get_array_length(env, js_array, &length);
  if (!CheckNapiOkImpl(env, st, "ParseSizes/get_array_length")) return false;

  out_sizes->clear();
  out_sizes->reserve(length);

  for (uint32_t i = 0; i < length; ++i) {
    napi_value elem;
    st = napi_get_element(env, js_array, i, &elem);
    if (!CheckNapiOkImpl(env, st, "ParseSizes/get_element")) return false;

    napi_valuetype t;
    st = napi_typeof(env, elem, &t);
    if (!CheckNapiOkImpl(env, st, "ParseSizes/typeof")) return false;
    if (t != napi_number) {
      ThrowTypeError(env, "sizes must contain only numbers");
      return false;
    }

    double dv = 0.0;
    st = napi_get_value_double(env, elem, &dv);
    if (!CheckNapiOkImpl(env, st, "ParseSizes/get_value_double")) return false;

    if (!std::isfinite(dv)) {
      ThrowTypeError(env, "sizes must be finite numbers");
      return false;
    }

    if (dv < 0) {
      ThrowTypeError(env, "sizes must be >= 0");
      return false;
    }

    auto v64 = static_cast<int64_t>(dv);
    if (static_cast<double>(v64) != dv) {
      ThrowTypeError(env, "sizes entries must be integers");
      return false;
    }

    out_sizes->push_back(v64);
  }

  return true;
}

inline bool NumelBytes(const std::vector<int64_t>& sizes,
                       vbt::core::ScalarType dtype,
                       std::size_t* out_nbytes) {
  if (!out_nbytes) return false;

  int64_t n = 1;
  for (int64_t s : sizes) {
    if (s < 0) return false;  // should already be validated
    if (s == 0) {
      n = 0;
      break;
    }
    int64_t tmp = 0;
    if (!vbt::core::checked_mul_i64(n, s, tmp)) {
      return false;
    }
    n = tmp;
  }

  int64_t item_b = static_cast<int64_t>(vbt::core::itemsize(dtype));
  int64_t total = 0;
  if (!vbt::core::checked_mul_i64(n, item_b, total)) {
    return false;
  }

  *out_nbytes = static_cast<std::size_t>(total);
  return true;
}

} // namespace vbt::node
