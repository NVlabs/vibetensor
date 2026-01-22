// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>

#include "vbt/dispatch/dispatcher.h"

#if VBT_WITH_DISPATCH_V2

namespace {

std::string make_def_with_n_tensors(const std::string& fqname, std::size_t n) {
  std::string def = fqname;
  def.push_back('(');
  for (std::size_t i = 0; i < n; ++i) {
    if (i != 0) def.append(", ");
    def.append("Tensor");
  }
  def.append(") -> Tensor");
  return def;
}

template <class Fn>
void expect_invalid(Fn&& fn,
                    const std::string& fqname,
                    const std::string& reason_substr) {
  try {
    fn();
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("set_device_policy"), std::string::npos) << msg;
    EXPECT_NE(msg.find(fqname), std::string::npos) << msg;
    EXPECT_NE(msg.find(reason_substr), std::string::npos) << msg;
  }
}

}  // namespace

TEST(DispatchV2DevicePolicySetterTest, RejectsMaskOutOfRange) {
  auto& D = vbt::dispatch::Dispatcher::instance();

  const std::string fqname = "test_device_policy::mask_out_of_range";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_device_policy");
  D.def(fqname + "(Tensor, Tensor) -> Tensor");

  expect_invalid(
      [&] {
        D.set_device_policy(
            fqname,
            vbt::dispatch::DevicePolicy::MaskedSameDevice,
            /*dispatch_arg_mask=*/(static_cast<std::uint64_t>(1) << 2),
            std::span<const vbt::dispatch::DeviceConstraint>{},
            /*allow_undefined_mask=*/0);
      },
      fqname,
      "mask out of range");
}

TEST(DispatchV2DevicePolicySetterTest, RejectsAllowUndefinedMaskOutOfRange) {
  auto& D = vbt::dispatch::Dispatcher::instance();

  const std::string fqname = "test_device_policy::allow_undefined_out_of_range";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_device_policy");
  D.def(fqname + "(Tensor, Tensor) -> Tensor");

  expect_invalid(
      [&] {
        D.set_device_policy(
            fqname,
            vbt::dispatch::DevicePolicy::MaskedSameDevice,
            /*dispatch_arg_mask=*/0,
            std::span<const vbt::dispatch::DeviceConstraint>{},
            /*allow_undefined_mask=*/(static_cast<std::uint64_t>(1) << 2));
      },
      fqname,
      "allow_undefined_mask out of range");
}

TEST(DispatchV2DevicePolicySetterTest, RejectsAllowUndefinedOnMustMatchDefault) {
  auto& D = vbt::dispatch::Dispatcher::instance();

  const std::string fqname = "test_device_policy::allow_undefined_on_must_match";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_device_policy");
  D.def(fqname + "(Tensor) -> Tensor");

  expect_invalid(
      [&] {
        D.set_device_policy(
            fqname,
            vbt::dispatch::DevicePolicy::MaskedSameDevice,
            /*dispatch_arg_mask=*/0,
            std::span<const vbt::dispatch::DeviceConstraint>{},
            /*allow_undefined_mask=*/1);
      },
      fqname,
      "allow_undefined_mask out of range");
}

TEST(DispatchV2DevicePolicySetterTest, AcceptsAllowUndefinedWithDeferToKernel) {
  auto& D = vbt::dispatch::Dispatcher::instance();

  const std::string fqname = "test_device_policy::allow_undefined_defer_to_kernel";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_device_policy");
  vbt::dispatch::OperatorHandle h = D.def(fqname + "(Tensor) -> Tensor");

  const std::array<vbt::dispatch::DeviceConstraint, 1> cs = {
      vbt::dispatch::DeviceConstraint{0, vbt::dispatch::ConstraintKind::DeferToKernel}};

  D.set_device_policy(
      fqname,
      vbt::dispatch::DevicePolicy::MaskedSameDevice,
      /*dispatch_arg_mask=*/0,
      std::span<const vbt::dispatch::DeviceConstraint>{cs},
      /*allow_undefined_mask=*/1);

  const vbt::dispatch::OperatorEntry& entry = h.get();
  EXPECT_EQ(entry.device_policy, vbt::dispatch::DevicePolicy::MaskedSameDevice);
  EXPECT_EQ(entry.allow_undefined_mask, 1u);
  EXPECT_EQ(entry.constraint_kind_by_index[0],
            vbt::dispatch::ConstraintKind::DeferToKernel);
}

TEST(DispatchV2DevicePolicySetterTest, RejectsDuplicateConstraintIndex) {
  auto& D = vbt::dispatch::Dispatcher::instance();

  const std::string fqname = "test_device_policy::duplicate_constraint";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_device_policy");
  D.def(fqname + "(Tensor, Tensor) -> Tensor");

  const std::array<vbt::dispatch::DeviceConstraint, 2> cs = {
      vbt::dispatch::DeviceConstraint{1, vbt::dispatch::ConstraintKind::DeferToKernel},
      vbt::dispatch::DeviceConstraint{1,
                                      vbt::dispatch::ConstraintKind::MustBeCPUScalarInt64_0d}};

  expect_invalid(
      [&] {
        D.set_device_policy(
            fqname,
            vbt::dispatch::DevicePolicy::MaskedSameDevice,
            /*dispatch_arg_mask=*/0,
            std::span<const vbt::dispatch::DeviceConstraint>{cs},
            /*allow_undefined_mask=*/0);
      },
      fqname,
      "duplicate constraint index");
}

TEST(DispatchV2DevicePolicySetterTest, RejectsConstraintIndexOutOfRange) {
  auto& D = vbt::dispatch::Dispatcher::instance();

  const std::string fqname = "test_device_policy::constraint_out_of_range";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_device_policy");
  D.def(fqname + "(Tensor, Tensor) -> Tensor");

  const std::array<vbt::dispatch::DeviceConstraint, 1> cs = {
      vbt::dispatch::DeviceConstraint{2, vbt::dispatch::ConstraintKind::DeferToKernel}};

  expect_invalid(
      [&] {
        D.set_device_policy(
            fqname,
            vbt::dispatch::DevicePolicy::MaskedSameDevice,
            /*dispatch_arg_mask=*/0,
            std::span<const vbt::dispatch::DeviceConstraint>{cs},
            /*allow_undefined_mask=*/0);
      },
      fqname,
      "constraint index out of range");
}

TEST(DispatchV2DevicePolicySetterTest, RejectsInArityGt64Contract) {
  auto& D = vbt::dispatch::Dispatcher::instance();

  const std::string fqname = "test_device_policy::in_arity_gt_64";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_device_policy");
  D.def(make_def_with_n_tensors(fqname, 65));

  expect_invalid(
      [&] {
        D.set_device_policy(
            fqname,
            vbt::dispatch::DevicePolicy::MaskedSameDevice,
            /*dispatch_arg_mask=*/0,
            std::span<const vbt::dispatch::DeviceConstraint>{},
            /*allow_undefined_mask=*/0);
      },
      fqname,
      "in_arity > 64");

  expect_invalid(
      [&] {
        D.set_device_policy(
            fqname,
            vbt::dispatch::DevicePolicy::AllSameDevice,
            /*dispatch_arg_mask=*/1,
            std::span<const vbt::dispatch::DeviceConstraint>{},
            /*allow_undefined_mask=*/0);
      },
      fqname,
      "in_arity > 64");

  const std::array<vbt::dispatch::DeviceConstraint, 1> cs = {
      vbt::dispatch::DeviceConstraint{0, vbt::dispatch::ConstraintKind::DeferToKernel}};

  expect_invalid(
      [&] {
        D.set_device_policy(
            fqname,
            vbt::dispatch::DevicePolicy::AllSameDevice,
            /*dispatch_arg_mask=*/0,
            std::span<const vbt::dispatch::DeviceConstraint>{cs},
            /*allow_undefined_mask=*/0);
      },
      fqname,
      "in_arity > 64");

  expect_invalid(
      [&] {
        D.set_device_policy(
            fqname,
            vbt::dispatch::DevicePolicy::AllSameDevice,
            /*dispatch_arg_mask=*/0,
            std::span<const vbt::dispatch::DeviceConstraint>{},
            /*allow_undefined_mask=*/1);
      },
      fqname,
      "in_arity > 64");
}

TEST(DispatchV2DevicePolicySetterTest, RejectsFabricOp) {
  auto& D = vbt::dispatch::Dispatcher::instance();

  const std::string fqname = "test_device_policy::fabric_reject";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_device_policy");
  D.def(fqname + "(Tensor) -> Tensor");
  D.mark_fabric_op(fqname, /*is_fabric_op=*/true, /*allow_multi_device_fabric=*/false);

  expect_invalid(
      [&] {
        D.set_device_policy(
            fqname,
            vbt::dispatch::DevicePolicy::AllSameDevice,
            /*dispatch_arg_mask=*/0,
            std::span<const vbt::dispatch::DeviceConstraint>{},
            /*allow_undefined_mask=*/0);
      },
      fqname,
      "fabric op");
}

#else

TEST(DispatchV2DevicePolicySetterTest, SkippedWhenDispatchV2Disabled) {
  GTEST_SKIP() << "dispatch v2 disabled";
}

#endif  // VBT_WITH_DISPATCH_V2
