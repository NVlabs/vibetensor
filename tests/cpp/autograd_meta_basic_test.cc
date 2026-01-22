// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "vbt/core/tensor.h"
#include "vbt/autograd/meta.h"

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;

static StoragePtr make_storage_am(void* base, std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(DataPtr(base, [](void*) noexcept {}), nbytes);
}

#if VBT_WITH_AUTOGRAD
TEST(AutogradMetaBasicTest, ToggleAndPersist) {
  auto storage = make_storage_am(reinterpret_cast<void*>(0xCAFEBABE), 64);
  TensorImpl t(storage, {2}, {1}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_FALSE(vbt::autograd::requires_grad(t));
  EXPECT_EQ(vbt::autograd::get_autograd_meta(t, false), nullptr);

  vbt::autograd::set_requires_grad(t, true);
  EXPECT_TRUE(vbt::autograd::requires_grad(t));
  auto* autograd_meta = vbt::autograd::get_autograd_meta(t, false);
  ASSERT_NE(autograd_meta, nullptr);

  vbt::autograd::set_requires_grad(t, false);
  EXPECT_FALSE(vbt::autograd::requires_grad(t));
  EXPECT_EQ(vbt::autograd::get_autograd_meta(t, false), autograd_meta);
}
#endif
