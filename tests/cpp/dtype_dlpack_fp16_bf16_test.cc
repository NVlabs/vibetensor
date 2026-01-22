// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <dlpack/dlpack.h>
#include "vbt/core/dtype.h"

using vbt::core::ScalarType;
using vbt::core::itemsize;
using vbt::core::to_dlpack_dtype;
using vbt::core::from_dlpack_dtype;

TEST(DTypeFp16Bf16Test, ItemsizeMapping) {
  EXPECT_EQ(itemsize(ScalarType::Float16), static_cast<std::size_t>(2));
  EXPECT_EQ(itemsize(ScalarType::BFloat16), static_cast<std::size_t>(2));
}

TEST(DTypeFp16Bf16Test, ToDLPackMapping) {
  DLDataType f16 = to_dlpack_dtype(ScalarType::Float16);
  EXPECT_EQ(f16.code, static_cast<uint8_t>(kDLFloat));
  EXPECT_EQ(f16.bits, 16);
  EXPECT_EQ(f16.lanes, 1);
#if VBT_HAS_DLPACK_BF16
  DLDataType bf16 = to_dlpack_dtype(ScalarType::BFloat16);
  EXPECT_EQ(bf16.code, static_cast<uint8_t>(kDLBfloat));
  EXPECT_EQ(bf16.bits, 16);
  EXPECT_EQ(bf16.lanes, 1);
#endif
}

TEST(DTypeFp16Bf16Test, FromDLPackMapping) {
  DLDataType f16{static_cast<uint8_t>(kDLFloat), 16, 1};
  auto s1 = from_dlpack_dtype(f16);
  ASSERT_TRUE(s1.has_value());
  EXPECT_EQ(s1.value(), ScalarType::Float16);
#if VBT_HAS_DLPACK_BF16
  DLDataType bf16{static_cast<uint8_t>(kDLBfloat), 16, 1};
  auto s2 = from_dlpack_dtype(bf16);
  ASSERT_TRUE(s2.has_value());
  EXPECT_EQ(s2.value(), ScalarType::BFloat16);
#endif
  // lanes must be 1
  DLDataType bad{static_cast<uint8_t>(kDLFloat), 16, 2};
  auto s3 = from_dlpack_dtype(bad);
  EXPECT_FALSE(s3.has_value());
}
