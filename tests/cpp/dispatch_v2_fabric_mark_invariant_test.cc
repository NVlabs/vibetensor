// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <string>

#include "vbt/dispatch/dispatcher.h"

TEST(DispatchV2FabricMarkInvariantTest, EnableSetsPolicy) {
#if !VBT_WITH_DISPATCH_V2
  GTEST_SKIP() << "dispatch v2 disabled";
#else
  auto& D = vbt::dispatch::Dispatcher::instance();

  const std::string fqname = "test_dispatch_v2_fabric_mark::enable_sets_policy";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_dispatch_v2_fabric_mark");
  vbt::dispatch::OperatorHandle h =
      D.def(fqname + "(Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");

  D.mark_fabric_op(fqname, /*is_fabric_op=*/true,
                  /*allow_multi_device_fabric=*/true);

  const vbt::dispatch::OperatorEntry& entry = h.get();
  const vbt::dispatch::OpDispatchStateV2* st =
      entry.state_v2.load(std::memory_order_acquire);

  ASSERT_NE(st, nullptr);
  EXPECT_EQ(st->device_policy, vbt::dispatch::DevicePolicy::Fabric5Arg);
  EXPECT_EQ(st->dispatch_arg_mask, 0u);
  EXPECT_EQ(st->allow_undefined_mask, 0u);
  EXPECT_EQ(st->constraint_kind_by_index[0],
            vbt::dispatch::ConstraintKind::MustMatchDispatchDeviceIfDefined);
  EXPECT_TRUE(st->allow_multi_device_fabric);
#endif
}

TEST(DispatchV2FabricMarkInvariantTest, DisableRestoresDefault) {
#if !VBT_WITH_DISPATCH_V2
  GTEST_SKIP() << "dispatch v2 disabled";
#else
  auto& D = vbt::dispatch::Dispatcher::instance();

  const std::string fqname = "test_dispatch_v2_fabric_mark::disable_restores_default";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_dispatch_v2_fabric_mark");
  vbt::dispatch::OperatorHandle h =
      D.def(fqname + "(Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");

  D.mark_fabric_op(fqname, /*is_fabric_op=*/true,
                  /*allow_multi_device_fabric=*/true);

  const vbt::dispatch::OperatorEntry& entry = h.get();
  const auto* st1 = entry.state_v2.load(std::memory_order_acquire);
  ASSERT_NE(st1, nullptr);
  ASSERT_EQ(st1->device_policy, vbt::dispatch::DevicePolicy::Fabric5Arg);

  // Disable bypass while keeping it marked as a Fabric op.
  D.mark_fabric_op(fqname, /*is_fabric_op=*/true,
                  /*allow_multi_device_fabric=*/false);

  const auto* st2 = entry.state_v2.load(std::memory_order_acquire);
  ASSERT_NE(st2, nullptr);
  EXPECT_EQ(st2->device_policy, vbt::dispatch::DevicePolicy::AllSameDevice);
  EXPECT_EQ(st2->dispatch_arg_mask, 0u);
  EXPECT_EQ(st2->allow_undefined_mask, 0u);
  EXPECT_EQ(st2->constraint_kind_by_index[0],
            vbt::dispatch::ConstraintKind::MustMatchDispatchDeviceIfDefined);
  EXPECT_FALSE(st2->allow_multi_device_fabric);
#endif
}

TEST(DispatchV2FabricMarkInvariantTest, RejectWrongArityEnable) {
  auto& D = vbt::dispatch::Dispatcher::instance();

  const std::string fqname = "test_dispatch_v2_fabric_mark::reject_wrong_arity_enable";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_dispatch_v2_fabric_mark");
  D.def(fqname + "(Tensor, Tensor, Tensor, Tensor) -> Tensor");

  try {
    D.mark_fabric_op(fqname, /*is_fabric_op=*/true,
                    /*allow_multi_device_fabric=*/true);
    FAIL() << "expected mark_fabric_op to reject wrong arity";
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("arity==5"), std::string::npos) << msg;
  }
}

TEST(DispatchV2FabricMarkInvariantTest, RejectNonTensorArgsEnable) {
  auto& D = vbt::dispatch::Dispatcher::instance();

  const std::string fqname =
      "test_dispatch_v2_fabric_mark::reject_non_tensor_args_enable";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_dispatch_v2_fabric_mark");
  D.def(fqname + "(Tensor, Tensor, Tensor, Tensor, Tensor, int) -> Tensor");

  try {
    D.mark_fabric_op(fqname, /*is_fabric_op=*/true,
                    /*allow_multi_device_fabric=*/true);
    FAIL() << "expected mark_fabric_op to reject non-tensor args";
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("has_non_tensor_args==false"), std::string::npos) << msg;
  }
}
