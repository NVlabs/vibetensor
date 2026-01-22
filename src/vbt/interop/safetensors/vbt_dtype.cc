// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/interop/safetensors/vbt_dtype.h"

#include "vbt/interop/safetensors/error.h"

namespace vbt {
namespace interop {
namespace safetensors {

#if VBT_WITH_SAFETENSORS

std::optional<vbt::core::ScalarType> to_vbt_scalar_type(DType dt) {
  switch (dt) {
    case DType::BOOL: return vbt::core::ScalarType::Bool;
    case DType::I32: return vbt::core::ScalarType::Int32;
    case DType::I64: return vbt::core::ScalarType::Int64;
    case DType::F16: return vbt::core::ScalarType::Float16;
    case DType::BF16: return vbt::core::ScalarType::BFloat16;
    case DType::F32: return vbt::core::ScalarType::Float32;

    // Known safetensors dtypes that VBT does not currently support.
    case DType::F4:
    case DType::F6_E2M3:
    case DType::F6_E3M2:
    case DType::U8:
    case DType::I8:
    case DType::F8_E5M2:
    case DType::F8_E4M3:
    case DType::F8_E8M0:
    case DType::I16:
    case DType::U16:
    case DType::U32:
    case DType::C64:
    case DType::F64:
    case DType::U64: return std::nullopt;
  }
  return std::nullopt;
}

vbt::core::ScalarType require_vbt_scalar_type(DType dt) {
  const std::optional<vbt::core::ScalarType> st = to_vbt_scalar_type(dt);
  if (st.has_value()) return *st;

  throw SafeTensorsError(ErrorCode::UnsupportedDtypeForVbt,
                         "safetensors: dtype is not supported by VBT");
}

#else

std::optional<vbt::core::ScalarType> to_vbt_scalar_type(DType /*dt*/) {
  throw SafeTensorsError(ErrorCode::IoError, "safetensors: support disabled at build time");
}

vbt::core::ScalarType require_vbt_scalar_type(DType /*dt*/) {
  throw SafeTensorsError(ErrorCode::IoError, "safetensors: support disabled at build time");
}

#endif

} // namespace safetensors
} // namespace interop
} // namespace vbt
