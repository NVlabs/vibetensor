// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include "vbt/cuda/device.h"
#include "vbt/cuda/event.h"
#include "vbt/cuda/stream.h"

using namespace vbt::cuda;

TEST(CUDAEvents, Lifecycle) {
  if (device_count() == 0) GTEST_SKIP() << "No CUDA device";
  Event e;
  EXPECT_TRUE(e.query());
  auto s1 = getStreamFromPool();
  e.wait(s1);  // no-op when not recorded
  e.record(s1);
  auto s2 = getStreamFromPool();
  e.wait(s2);
  e.synchronize();
  EXPECT_TRUE(e.query());
}

TEST(CUDAEvents, CrossDeviceWaitWorks) {
#if VBT_WITH_CUDA
  if (device_count() < 2) GTEST_SKIP() << "Need >=2 CUDA devices";

  auto s0 = getStreamFromPool(/*priority=*/0, /*device=*/0);
  auto s1 = getStreamFromPool(/*priority=*/0, /*device=*/1);

  Event e;
  e.record(s0);

  // Cross-device waits are used by multi-GPU features (Fabric/manual P2P).
  ASSERT_NO_THROW(e.wait(s1));
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
