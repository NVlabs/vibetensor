// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <string>

#include "vbt/dispatch/dispatcher.h"

TEST(DispatchV2SnapshotTest, DefPublishesStateV2) {
  auto& D = vbt::dispatch::Dispatcher::instance();

  const char* op = "test_dispatch_v2::snapshot_def";
  ASSERT_FALSE(D.has(op)) << "test op already registered: " << op;

  D.registerLibrary("test_dispatch_v2");
  vbt::dispatch::OperatorHandle h =
      D.def(std::string(op) + "(Tensor, Tensor) -> Tensor");

#if VBT_WITH_DISPATCH_V2
  const vbt::dispatch::OperatorEntry& entry = h.get();
  const vbt::dispatch::OpDispatchStateV2* st =
      entry.state_v2.load(std::memory_order_acquire);

  ASSERT_NE(st, nullptr);
  EXPECT_EQ(st->fqname, op);
  EXPECT_EQ(st->in_arity, 2);
  EXPECT_EQ(st->device_policy, vbt::dispatch::DevicePolicy::AllSameDevice);
  EXPECT_EQ(st->dispatch_arg_mask, 0u);
  EXPECT_EQ(st->allow_undefined_mask, 0u);
  EXPECT_EQ(st->constraint_kind_by_index[0],
            vbt::dispatch::ConstraintKind::MustMatchDispatchDeviceIfDefined);
  EXPECT_EQ(st->constraint_kind_by_index[1],
            vbt::dispatch::ConstraintKind::MustMatchDispatchDeviceIfDefined);
  EXPECT_EQ(st->constraint_kind_by_index[vbt::dispatch::kV2DevicePolicyMaxArity - 1],
            vbt::dispatch::ConstraintKind::MustMatchDispatchDeviceIfDefined);
  EXPECT_TRUE(st->present_wrappers.empty());
  EXPECT_NE(st->version, 0u);
#else
  GTEST_SKIP() << "dispatch v2 disabled";
#endif
}
