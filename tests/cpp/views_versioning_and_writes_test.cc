// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>

#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/write_guard.h"
#include "vbt/core/intrusive_ptr.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::check_writable;

static vbt::core::StoragePtr make_storage_vw(void* base, std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(DataPtr(base, [](void*) noexcept {}), nbytes);
}

TEST(ViewsVersioningAndWritesTest, VersionSharingAndBump) {
  auto storage = make_storage_vw(reinterpret_cast<void*>(0x5000), 1024);
  TensorImpl base(storage, {2,2}, {2,1}, 0, ScalarType::Float32, Device::cpu());
  auto v = base.as_strided({2,2}, {2,1}, 0);
  EXPECT_EQ(base.version(), 0);
  EXPECT_EQ(v.version(), 0);

  base.bump_version();
  EXPECT_EQ(base.version(), 1);
  EXPECT_EQ(v.version(), 1);

  v.bump_version();
  EXPECT_EQ(base.version(), 2);
  EXPECT_EQ(v.version(), 2);
}

TEST(ViewsVersioningAndWritesTest, CheckWritableAndNegativeStride) {
  auto storage = make_storage_vw(reinterpret_cast<void*>(0x6000), 1024);
  TensorImpl t(storage, {2}, {-1}, 1, ScalarType::Float32, Device::cpu());
  EXPECT_NO_THROW(check_writable(t));

  // Zero-size negative stride OK (no-op)
  TensorImpl tz(storage, {0}, {-1}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_NO_THROW(check_writable(tz));
}

TEST(ViewsVersioningAndWritesTest, InternalOverlapStrideZero) {
  auto storage = make_storage_vw(reinterpret_cast<void*>(0x7000), 1024);
  // size>1 with stride==0 should be rejected
  TensorImpl t(storage, {3}, {0}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_THROW(check_writable(t), std::invalid_argument);
  // size==1 with stride==0 allowed
  TensorImpl t1(storage, {1}, {0}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_NO_THROW(check_writable(t1));
}
