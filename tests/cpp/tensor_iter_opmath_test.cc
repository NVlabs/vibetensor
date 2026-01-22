// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "vbt/core/tensor_iter.h"
#include "vbt/core/dtype.h"

using vbt::core::ScalarType;
using vbt::core::TensorIterBase;
using vbt::core::testing::TensorIterTestHelper;

TEST(TensorIterOpMathTest, OpMathDtypeLogic) {
  // Helper creates an iterator with common_dtype set
  std::vector<int64_t> shape = {10};
  auto iter_f32 = TensorIterTestHelper::make_iterator_for_shape(shape);
  // Need access to set common_dtype_, but make_iterator_for_shape sets it to Float32.
  EXPECT_EQ(iter_f32.computation_dtype(), ScalarType::Float32);
}
