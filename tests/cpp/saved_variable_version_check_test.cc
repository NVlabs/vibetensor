// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "vbt/autograd/saved_variable.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"

using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;

static StoragePtr make_storage_sv(void* base, std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(DataPtr(base, [](void*) noexcept {}), nbytes);
}

TEST(SavedVariableTest, UninitializedThrows) {
  vbt::autograd::SavedVariable sv;
  try {
    auto t = sv.unpack();
    (void)t;
    FAIL() << "Expected logic_error";
  } catch (const std::logic_error& e) {
    EXPECT_STREQ(e.what(), "SavedVariable: not initialized");
  }
}

TEST(SavedVariableTest, NoThrowWhenVersionMatches) {
  auto storage = make_storage_sv(reinterpret_cast<void*>(0xABCD0000), /*nbytes=*/64);
  TensorImpl t(storage, {2}, {1}, 0, ScalarType::Float32, Device::cpu());
  vbt::autograd::SavedVariable sv(t);
  EXPECT_NO_THROW({ auto u = sv.unpack(); (void)u; });
}

TEST(SavedVariableTest, VersionMismatchThrows) {
  auto storage = make_storage_sv(reinterpret_cast<void*>(0xABCD1000), /*nbytes=*/64);
  TensorImpl t(storage, {2}, {1}, 0, ScalarType::Float32, Device::cpu());
  vbt::autograd::SavedVariable sv(t);
  // Bump version (simulate in-place)
  t.bump_version();
  try {
    (void)sv.unpack();
    FAIL() << "Expected runtime_error";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("in-place modification"), std::string::npos) << msg;
  }
}
