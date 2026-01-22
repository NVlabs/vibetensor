// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <dlpack/dlpack.h>

#include <cstddef>
#include <type_traits>

#include "vbt/core/complex.h"
#include "vbt/core/dtype.h"

using vbt::core::Complex64;
using vbt::core::Complex128;
using vbt::core::ScalarType;
using vbt::core::from_dlpack_dtype;
using vbt::core::itemsize;
using vbt::core::to_dlpack_dtype;

static_assert(sizeof(Complex64) == 8, "Complex64 must be 8 bytes");
static_assert(alignof(Complex64) == 8, "Complex64 must be 8-byte aligned");
static_assert(std::is_standard_layout_v<Complex64>, "Complex64 must be standard-layout");
static_assert(std::is_trivially_copyable_v<Complex64>, "Complex64 must be trivially copyable");

static_assert(sizeof(Complex128) == 16, "Complex128 must be 16 bytes");
static_assert(alignof(Complex128) == 16, "Complex128 must be 16-byte aligned");
static_assert(std::is_standard_layout_v<Complex128>, "Complex128 must be standard-layout");
static_assert(std::is_trivially_copyable_v<Complex128>, "Complex128 must be trivially copyable");

TEST(DTypeComplexTest, ItemsizeMapping) {
  EXPECT_EQ(itemsize(ScalarType::Float64), static_cast<std::size_t>(8));
  EXPECT_EQ(itemsize(ScalarType::Complex64), static_cast<std::size_t>(8));
  EXPECT_EQ(itemsize(ScalarType::Complex128), static_cast<std::size_t>(16));
}

TEST(DTypeComplexTest, ToDLPackMapping) {
  DLDataType f64 = to_dlpack_dtype(ScalarType::Float64);
  EXPECT_EQ(f64.code, static_cast<uint8_t>(kDLFloat));
  EXPECT_EQ(f64.bits, 64);
  EXPECT_EQ(f64.lanes, 1);

  DLDataType c64 = to_dlpack_dtype(ScalarType::Complex64);
  EXPECT_EQ(c64.code, static_cast<uint8_t>(kDLComplex));
  EXPECT_EQ(c64.bits, 64);
  EXPECT_EQ(c64.lanes, 1);

  DLDataType c128 = to_dlpack_dtype(ScalarType::Complex128);
  EXPECT_EQ(c128.code, static_cast<uint8_t>(kDLComplex));
  EXPECT_EQ(c128.bits, 128);
  EXPECT_EQ(c128.lanes, 1);
}

TEST(DTypeComplexTest, FromDLPackMappingRoundtrip) {
  DLDataType f64{static_cast<uint8_t>(kDLFloat), 64, 1};
  auto s1 = from_dlpack_dtype(f64);
  ASSERT_TRUE(s1.has_value());
  EXPECT_EQ(s1.value(), ScalarType::Float64);

  DLDataType c64{static_cast<uint8_t>(kDLComplex), 64, 1};
  auto s2 = from_dlpack_dtype(c64);
  ASSERT_TRUE(s2.has_value());
  EXPECT_EQ(s2.value(), ScalarType::Complex64);

  DLDataType c128{static_cast<uint8_t>(kDLComplex), 128, 1};
  auto s3 = from_dlpack_dtype(c128);
  ASSERT_TRUE(s3.has_value());
  EXPECT_EQ(s3.value(), ScalarType::Complex128);

  // lanes must be 1
  DLDataType bad{static_cast<uint8_t>(kDLComplex), 64, 2};
  auto s4 = from_dlpack_dtype(bad);
  EXPECT_FALSE(s4.has_value());
}
