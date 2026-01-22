// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "vbt/cuda/device.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/guard.h"

using namespace vbt::cuda;

TEST(CUDAStreams, PriorityRangeParity) {
  if (device_count() == 0) GTEST_SKIP() << "No CUDA device";
  auto pr = priority_range();
  if (pr.first == 0 && pr.second == 0) {
    SUCCEED();
  } else {
    EXPECT_EQ(pr.first, 0);
    EXPECT_LE(pr.second, -1);
  }
}

TEST(CUDAStreams, DefaultCurrentRoundtripAndGuard) {
  if (device_count() == 0) GTEST_SKIP() << "No CUDA device";
  auto def = getDefaultStream();
  EXPECT_EQ(def.id(), 0u);
  EXPECT_EQ(def.handle(), 0u);
  auto cur0 = getCurrentStream();
  EXPECT_TRUE(cur0 == def);
  {
    auto s = getStreamFromPool();
    EXPECT_NE(s.id(), 0u);
    CUDAStreamGuard g(s);
    auto cur = getCurrentStream();
    EXPECT_TRUE(cur == s);
  }
  auto cur1 = getCurrentStream();
  EXPECT_TRUE(cur1 == def);
}
