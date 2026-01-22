// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <new>

#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/forward.h"
#include "vbt/autograd/wrapper.h"

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;

static StoragePtr make_storage_am(void* /*base*/, std::size_t nbytes) {
  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
  }
  return vbt::core::make_intrusive<Storage>(
      DataPtr(buf, [](void* p) noexcept { ::operator delete(p); }), nbytes);
}

#if VBT_WITH_AUTOGRAD

TEST(AutogradForwardADTest, ForwardGradSlotLifecycle) {
  auto storage = make_storage_am(reinterpret_cast<void*>(0xDEADBEEF),
                                 sizeof(float) * 4);
  TensorImpl primal(storage, {4}, {1}, 0, ScalarType::Float32, Device::cpu());

  // Tangent with the same shape/device/dtype.
  auto t_storage = make_storage_am(reinterpret_cast<void*>(0xCAFEBABE),
                                   sizeof(float) * 4);
  TensorImpl tangent(t_storage, {4}, {1}, 0, ScalarType::Float32, Device::cpu());

  // Open a forward AD level and attach a tangent.
  const std::int64_t level = vbt::autograd::enter_forward_ad_level();
  ASSERT_GE(level, 0);

  EXPECT_FALSE(vbt::autograd::has_forward_grad(primal, level));
  vbt::autograd::set_forward_grad(primal, tangent, level);
  EXPECT_TRUE(vbt::autograd::has_forward_grad(primal, level));
  EXPECT_TRUE(vbt::autograd::has_any_forward_grad(primal));

  // get_forward_grad_copy returns a detached copy with the same shape.
  TensorImpl copy = vbt::autograd::get_forward_grad_copy(primal, level);
  EXPECT_EQ(copy.sizes(), primal.sizes());
  EXPECT_EQ(copy.device(), primal.device());
  EXPECT_EQ(copy.dtype(), ScalarType::Float32);

  // clear_forward_grad drops the tangent but leaves reverse-mode grad intact.
  vbt::autograd::clear_forward_grad(primal);
  EXPECT_FALSE(vbt::autograd::has_forward_grad(primal, level));
  EXPECT_FALSE(vbt::autograd::has_any_forward_grad(primal));

  vbt::autograd::exit_forward_ad_level(level);
}

TEST(AutogradForwardADTest, LevelAndGuardInteraction) {
  // No level: forward-mode is disabled.
  EXPECT_EQ(vbt::autograd::current_forward_ad_level(), -1);
  EXPECT_FALSE(vbt::autograd::is_in_backward());
  EXPECT_FALSE(vbt::autograd::is_forward_ad_enabled());

  const std::int64_t level = vbt::autograd::enter_forward_ad_level();
  ASSERT_GE(level, 0);
  EXPECT_EQ(vbt::autograd::current_forward_ad_level(), level);
  EXPECT_TRUE(vbt::autograd::is_forward_ad_enabled());

  // While BackwardGuard is active, forward-mode is disabled.
  {
    vbt::autograd::BackwardGuard g;
    EXPECT_TRUE(vbt::autograd::is_in_backward());
    EXPECT_FALSE(vbt::autograd::is_forward_ad_enabled());
  }

  EXPECT_FALSE(vbt::autograd::is_in_backward());
  EXPECT_TRUE(vbt::autograd::is_forward_ad_enabled());

  vbt::autograd::exit_forward_ad_level(level);
  EXPECT_EQ(vbt::autograd::current_forward_ad_level(), -1);
  EXPECT_FALSE(vbt::autograd::is_forward_ad_enabled());
}

TEST(AutogradForwardADTest, LevelNestingAndInferenceModeGuards) {
  // Nested levels are rejected.
  const std::int64_t level = vbt::autograd::enter_forward_ad_level();
  ASSERT_GE(level, 0);
  EXPECT_THROW({ vbt::autograd::enter_forward_ad_level(); }, std::runtime_error);
  vbt::autograd::exit_forward_ad_level(level);

  // Entering a level under inference-mode is rejected.
  vbt::autograd::InferenceMode::set_enabled(true);
  EXPECT_THROW({ (void)vbt::autograd::enter_forward_ad_level(); }, std::runtime_error);
  vbt::autograd::InferenceMode::set_enabled(false);
}

#endif  // VBT_WITH_AUTOGRAD
