// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <optional>
#include <string_view>

#include "vbt/interop/safetensors.h"

using vbt::core::ScalarType;
using vbt::interop::safetensors::DType;
using vbt::interop::safetensors::ErrorCode;
using vbt::interop::safetensors::SafeTensorsError;
using vbt::interop::safetensors::require_vbt_scalar_type;
using vbt::interop::safetensors::to_vbt_scalar_type;

TEST(SafeTensorsVbtDtypeTest, ToVbtScalarTypeSupportedSubset) {
  EXPECT_EQ(to_vbt_scalar_type(DType::BOOL), ScalarType::Bool);
  EXPECT_EQ(to_vbt_scalar_type(DType::I32), ScalarType::Int32);
  EXPECT_EQ(to_vbt_scalar_type(DType::I64), ScalarType::Int64);
  EXPECT_EQ(to_vbt_scalar_type(DType::F16), ScalarType::Float16);
  EXPECT_EQ(to_vbt_scalar_type(DType::BF16), ScalarType::BFloat16);
  EXPECT_EQ(to_vbt_scalar_type(DType::F32), ScalarType::Float32);
}

TEST(SafeTensorsVbtDtypeTest, RequireVbtScalarTypeSupportedSubset) {
  EXPECT_EQ(require_vbt_scalar_type(DType::BOOL), ScalarType::Bool);
  EXPECT_EQ(require_vbt_scalar_type(DType::I32), ScalarType::Int32);
  EXPECT_EQ(require_vbt_scalar_type(DType::I64), ScalarType::Int64);
  EXPECT_EQ(require_vbt_scalar_type(DType::F16), ScalarType::Float16);
  EXPECT_EQ(require_vbt_scalar_type(DType::BF16), ScalarType::BFloat16);
  EXPECT_EQ(require_vbt_scalar_type(DType::F32), ScalarType::Float32);
}

TEST(SafeTensorsVbtDtypeTest, ToVbtScalarTypeUnsupportedReturnsNullopt) {
  EXPECT_FALSE(to_vbt_scalar_type(DType::U8).has_value());
  EXPECT_FALSE(to_vbt_scalar_type(DType::I8).has_value());
  EXPECT_FALSE(to_vbt_scalar_type(DType::U16).has_value());
  EXPECT_FALSE(to_vbt_scalar_type(DType::U32).has_value());
  EXPECT_FALSE(to_vbt_scalar_type(DType::F64).has_value());
  EXPECT_FALSE(to_vbt_scalar_type(DType::C64).has_value());
}

TEST(SafeTensorsVbtDtypeTest, RequireVbtScalarTypeThrowsForUnsupported) {
  try {
    (void)require_vbt_scalar_type(DType::U8);
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::UnsupportedDtypeForVbt);
    EXPECT_EQ(e.tensor_name(), std::string_view{});
  }
}
