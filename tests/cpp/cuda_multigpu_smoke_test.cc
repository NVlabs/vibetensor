// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "vbt/cuda/device.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/event.h"

TEST(CUDAMultiGPU, CrossDeviceEventWaitWorks) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() < 2) GTEST_SKIP() << "Need >=2 CUDA devices";
  auto s0 = vbt::cuda::getStreamFromPool(/*priority=*/0, /*device=*/0);
  vbt::cuda::Event e;
  e.record(s0);
  auto s1 = vbt::cuda::getStreamFromPool(/*priority=*/0, /*device=*/1);

  ASSERT_NO_THROW(e.wait(s1));
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
