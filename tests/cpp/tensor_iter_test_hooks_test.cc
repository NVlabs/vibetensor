// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <span>

#include "vbt/core/tensor_iterator/core.h"

#ifndef VBT_TI_ENABLE_TEST_HOOKS
#error "VBT_TI_ENABLE_TEST_HOOKS must be defined for TI tests"
#endif

using vbt::core::testing::TensorIterTestHelper;

TEST(TensorIterTestHooksTest, HelpersAreAvailableAndBasicShapeIsCorrect) {
  std::int64_t s[1] = {1};
  auto it1 = TensorIterTestHelper::make_iterator_for_shape(
      std::span<const std::int64_t>(s, 1));
  auto it2 = TensorIterTestHelper::make_iterator_for_shape_with_dummy_operand(
      std::span<const std::int64_t>(s, 1));

  EXPECT_EQ(it1.ndim(), 1);
  EXPECT_EQ(it2.ndim(), 1);
  EXPECT_EQ(it1.shape().size(), 1u);
  EXPECT_EQ(it2.shape().size(), 1u);
}
