// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/plugin_loader.h"

using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::TensorImpl;
using vbt::dispatch::BoxedStack;
using vbt::dispatch::ConstraintKind;
using vbt::dispatch::DevicePolicy;
using vbt::dispatch::Dispatcher;

namespace {
struct AtomicCommitEnvGuard {
  AtomicCommitEnvGuard() { setenv("VBT_PLUGIN_ATOMIC_COMMIT", "1", 1); }
};
static AtomicCommitEnvGuard g_atomic_commit_env_guard;

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(
      DataPtr(::operator new(nbytes),
              [](void* p) noexcept { ::operator delete(p); }),
      nbytes);
}

static TensorImpl cpu_tensor_f32_1d(const std::vector<float>& values) {
  auto st = make_storage_bytes(values.size() * sizeof(float));
  float* p = static_cast<float*>(st->data());
  for (std::size_t i = 0; i < values.size(); ++i) {
    p[i] = values[i];
  }
  TensorImpl t(st,
               /*sizes=*/{static_cast<std::int64_t>(values.size())},
               /*strides=*/{1},
               /*storage_offset=*/0,
               ScalarType::Float32,
               Device::cpu());
  return t;
}
}  // namespace

#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS

TEST(PluginAtomicCommitSetDevicePolicyTest, OwnedPolicyStagesAndCommits) {
#ifndef PLUGIN_P3_SET_DEVICE_POLICY_PATH
  GTEST_SKIP() << "PLUGIN_P3_SET_DEVICE_POLICY_PATH not provided";
#else
#if !(defined(__linux__) || defined(__APPLE__))
  GTEST_SKIP() << "dlopen/RTLD_NOLOAD not supported on this platform";
#else
  const char* so = PLUGIN_P3_SET_DEVICE_POLICY_PATH;

  // Ensure the plugin is not already loaded in this process.
  (void)dlerror();
  void* pre = dlopen(so, RTLD_NOLOAD | RTLD_LAZY | RTLD_LOCAL);
  if (pre) {
    dlclose(pre);
    GTEST_SKIP() << "p3_set_device_policy plugin already loaded";
  }

  // Enable dispatcher v2 for this test binary.
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  auto& D = Dispatcher::instance();

  const vt_status st = vbt::dispatch::plugin::load_library(so);
  ASSERT_EQ(st, VT_STATUS_OK) << vbt::dispatch::plugin::get_last_error();

  ASSERT_TRUE(D.has("vt::p3_set_device_policy"));

  auto h = D.find("vt::p3_set_device_policy");
  auto& entry = h.get();

  EXPECT_EQ(entry.device_policy, DevicePolicy::MaskedSameDevice);
  EXPECT_EQ(entry.dispatch_arg_mask, 1u);
  EXPECT_EQ(entry.allow_undefined_mask, 2u);
  EXPECT_EQ(entry.constraint_kind_by_index[1], ConstraintKind::DeferToKernel);

  const auto* st2 = entry.state_v2.load(std::memory_order_acquire);
  ASSERT_NE(st2, nullptr);
  EXPECT_EQ(st2->device_policy, DevicePolicy::MaskedSameDevice);
  EXPECT_EQ(st2->dispatch_arg_mask, 1u);
  EXPECT_EQ(st2->allow_undefined_mask, 2u);
  EXPECT_EQ(st2->constraint_kind_by_index[1], ConstraintKind::DeferToKernel);

  // Kernel must be installed and callable.
  TensorImpl a = cpu_tensor_f32_1d({1.0f, 2.0f, 3.0f, 4.0f});
  TensorImpl b = cpu_tensor_f32_1d({10.0f, 20.0f, 30.0f, 40.0f});
  BoxedStack stack{a, b};
  D.callBoxed("vt::p3_set_device_policy", stack);

  ASSERT_EQ(stack.size(), 1u);
  const TensorImpl& out = stack[0];
  ASSERT_EQ(out.device(), Device::cpu());
  ASSERT_EQ(out.dtype(), ScalarType::Float32);

  const float* pout = static_cast<const float*>(out.data());
  ASSERT_NE(pout, nullptr);
  EXPECT_FLOAT_EQ(pout[0], 11.0f);
  EXPECT_FLOAT_EQ(pout[1], 22.0f);
  EXPECT_FLOAT_EQ(pout[2], 33.0f);
  EXPECT_FLOAT_EQ(pout[3], 44.0f);
#endif
#endif
}

TEST(PluginAtomicCommitSetDevicePolicyTest, NonOwnedPolicyRejected) {
#ifndef PLUGIN_SET_DEVICE_POLICY_CORE_REJECT_PATH
  GTEST_SKIP() << "PLUGIN_SET_DEVICE_POLICY_CORE_REJECT_PATH not provided";
#else
#if !(defined(__linux__) || defined(__APPLE__))
  GTEST_SKIP() << "dlopen/RTLD_NOLOAD not supported on this platform";
#else
  const char* so = PLUGIN_SET_DEVICE_POLICY_CORE_REJECT_PATH;

  // Ensure the plugin is not already loaded in this process.
  (void)dlerror();
  void* pre = dlopen(so, RTLD_NOLOAD | RTLD_LAZY | RTLD_LOCAL);
  if (pre) {
    dlclose(pre);
    GTEST_SKIP() << "set_device_policy_core_reject plugin already loaded";
  }

  // Enable dispatcher v2 for this test binary.
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  setenv("VBT_SET_DEVICE_POLICY_CORE_REJECT_MODE", "policy_not_owned", 1);

  const vt_status st = vbt::dispatch::plugin::load_library(so);
  EXPECT_EQ(st, VT_STATUS_UNSUPPORTED)
      << "err=" << vbt::dispatch::plugin::get_last_error();

  const std::string err = vbt::dispatch::plugin::get_last_error();
  EXPECT_NE(err.find("set_device_policy: non-owned op"), std::string::npos)
      << err;
#endif
#endif
}

#else

TEST(PluginAtomicCommitSetDevicePolicyTest, Skipped) {
  GTEST_SKIP() << "requires VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS";
}

#endif  // VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
