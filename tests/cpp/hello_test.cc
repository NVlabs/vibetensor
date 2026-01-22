// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/hello.h"
#include <gtest/gtest.h>

TEST(Hello, ReturnsConstant) {
  EXPECT_EQ(vbt::HelloString(), std::string{"vibetensor::_C ready"});
}
