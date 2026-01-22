// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <limits>

#include "vbt/core/checked_math.h"

using vbt::core::checked_add_i64;
using vbt::core::checked_mul_i64;
using vbt::core::accumulate_span_terms;

TEST(CheckedMathTest, AddOverflow) {
  int64_t out = 0;
  EXPECT_FALSE(checked_add_i64(std::numeric_limits<int64_t>::max(), 1, out));
  EXPECT_FALSE(checked_add_i64(std::numeric_limits<int64_t>::min(), -1, out));
  EXPECT_TRUE(checked_add_i64(10, 20, out));
  EXPECT_EQ(out, 30);
}

TEST(CheckedMathTest, MulOverflow) {
  int64_t out = 0;
  EXPECT_FALSE(checked_mul_i64(std::numeric_limits<int64_t>::max(), 2, out));
  EXPECT_FALSE(checked_mul_i64(std::numeric_limits<int64_t>::min(), -1, out));
  EXPECT_TRUE(checked_mul_i64(0, 123, out));
  EXPECT_EQ(out, 0);
  EXPECT_TRUE(checked_mul_i64(7, -3, out));
  EXPECT_EQ(out, -21);
}

TEST(CheckedMathTest, AccumulateSpanTerms) {
  int64_t min_acc = 0, max_acc = 0;
  // stride positive
  ASSERT_TRUE(accumulate_span_terms(4, 5, min_acc, max_acc)); // term = 4*(5-1)=16 -> max += 16
  EXPECT_EQ(min_acc, 0);
  EXPECT_EQ(max_acc, 16);
  // stride negative
  ASSERT_TRUE(accumulate_span_terms(-2, 3, min_acc, max_acc)); // term = -2*(3-1)=-4 -> min += -4
  EXPECT_EQ(min_acc, -4);
  EXPECT_EQ(max_acc, 16);
  // size zero no-op
  ASSERT_TRUE(accumulate_span_terms(123, 0, min_acc, max_acc));
  EXPECT_EQ(min_acc, -4);
  EXPECT_EQ(max_acc, 16);
}
