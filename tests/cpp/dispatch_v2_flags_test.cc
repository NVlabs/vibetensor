// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/dispatch/dispatcher.h"

TEST(DispatchV2FlagsTest, ModeGuardAcceptsTrueAndRejectsFalse) {
#if VBT_INTERNAL_TESTS
  EXPECT_THROW(
      {
        vbt::dispatch::DispatchV2ModeGuard guard{false};
      },
      std::invalid_argument);

  {
    vbt::dispatch::DispatchV2ModeGuard guard{true};
  }
#else
  GTEST_SKIP() << "internal tests disabled";
#endif
}

TEST(DispatchV2FlagsTest, FabricNoCudaCallsGuardOverridesAndRestores) {
#if VBT_INTERNAL_TESTS
  const bool prev = vbt::dispatch::dispatch_v2_fabric_no_cuda_calls();
  const bool forced = !prev;

  {
    vbt::dispatch::DispatchV2FabricNoCudaCallsGuard guard{forced};
    EXPECT_EQ(vbt::dispatch::dispatch_v2_fabric_no_cuda_calls(), forced);
  }

  EXPECT_EQ(vbt::dispatch::dispatch_v2_fabric_no_cuda_calls(), prev);
#else
  GTEST_SKIP() << "internal tests disabled";
#endif
}
