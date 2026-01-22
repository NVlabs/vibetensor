// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <dlpack/dlpack.h>
#include <atomic>
#include <stdexcept>
#include <type_traits>
#include <cstdint>

#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"
#include "vbt/core/intrusive_ptr.h"

using vbt::core::ScalarType;
using vbt::core::itemsize;
using vbt::core::to_dlpack_dtype;
using vbt::core::from_dlpack_dtype;
using vbt::core::Device;
using vbt::core::DataPtr;
using vbt::core::Storage;
using vbt::core::TensorImpl;

TEST(DTypeTest, ItemsizeMapping) {
  EXPECT_EQ(itemsize(ScalarType::Bool), 1u);
  EXPECT_EQ(itemsize(ScalarType::Int32), 4u);
  EXPECT_EQ(itemsize(ScalarType::Int64), 8u);
  EXPECT_EQ(itemsize(ScalarType::Float32), 4u);
}

TEST(DTypeTest, DLPackRoundtrip) {
  auto dt = to_dlpack_dtype(ScalarType::Bool);
  ASSERT_TRUE(from_dlpack_dtype(dt).has_value());
  EXPECT_EQ(from_dlpack_dtype(dt).value(), ScalarType::Bool);

  dt = to_dlpack_dtype(ScalarType::Int32);
  ASSERT_TRUE(from_dlpack_dtype(dt).has_value());
  EXPECT_EQ(from_dlpack_dtype(dt).value(), ScalarType::Int32);

  dt = to_dlpack_dtype(ScalarType::Int64);
  ASSERT_TRUE(from_dlpack_dtype(dt).has_value());
  EXPECT_EQ(from_dlpack_dtype(dt).value(), ScalarType::Int64);

  dt = to_dlpack_dtype(ScalarType::Float32);
  ASSERT_TRUE(from_dlpack_dtype(dt).has_value());
  EXPECT_EQ(from_dlpack_dtype(dt).value(), ScalarType::Float32);

  // Negative and edge cases
  // lanes must be 1
  DLDataType bad{};
  bad.code = kDLFloat; bad.bits = 32; bad.lanes = 2;
  EXPECT_FALSE(from_dlpack_dtype(bad).has_value());
  // legacy bool bit-width not accepted
  bad.code = kDLBool; bad.bits = 1; bad.lanes = 1;
  EXPECT_FALSE(from_dlpack_dtype(bad).has_value());
  // float16 should now map to ScalarType::Float16
  DLDataType f16{static_cast<uint8_t>(kDLFloat), 16, 1};
  ASSERT_TRUE(from_dlpack_dtype(f16).has_value());
  EXPECT_EQ(from_dlpack_dtype(f16).value(), ScalarType::Float16);
}

TEST(DeviceTest, Basics) {
  auto d0 = Device::cpu(0);
  auto d1 = Device::cpu(1);
  EXPECT_NE(d0, d1);
  EXPECT_EQ(d0, Device::cpu(0));
  EXPECT_EQ(d0.to_string(), std::string("cpu:0"));
}

TEST(DataPtrTest, MoveSemanticsAndDeleter) {
  static_assert(!std::is_copy_constructible<DataPtr>::value, "DataPtr must be move-only");

  std::atomic<int> freed{0};
  {
    DataPtr dp((void*)0x1234, [&](void*) noexcept { freed.fetch_add(1); });
    EXPECT_TRUE(dp);
    DataPtr dp2 = std::move(dp);
    EXPECT_FALSE((bool)dp);
    EXPECT_TRUE(dp2);
    dp2.reset();
    EXPECT_EQ(freed.load(), 1);
  }
  // deleter already called
  EXPECT_EQ(freed.load(), 1);

  // Deleter exception must be swallowed
  {
    DataPtr dp((void*)0x2345, [&](void*) { throw std::runtime_error("boom"); });
    EXPECT_NO_THROW(dp.reset());
  }
}

TEST(StorageTest, ReleaseResources) {
  static_assert(noexcept(std::declval<Storage&>().release_resources()), "release_resources must be noexcept");
  static_assert(noexcept(std::declval<const Storage&>().nbytes()), "nbytes must be noexcept");
  static_assert(noexcept(std::declval<const Storage&>().data()), "data must be noexcept");

  // default
  Storage s0;
  EXPECT_EQ(s0.data(), nullptr);
  EXPECT_EQ(s0.nbytes(), 0u);

  std::atomic<int> freed{0};
  DataPtr dp((void*)0x3456, [&](void*) noexcept { freed.fetch_add(1); });
  Storage s(std::move(dp), 128);
  EXPECT_EQ(s.nbytes(), 128u);
  EXPECT_NE(s.data(), nullptr);
  s.release_resources();
  EXPECT_EQ(s.nbytes(), 0u);
  EXPECT_EQ(s.data(), nullptr);
  s.release_resources(); // idempotent
  EXPECT_EQ(freed.load(), 1);
}

TEST(TensorImplTest, PointerMathAndEmpty) {
  // Base pointer and storage
  auto base = reinterpret_cast<void*>(0x1000);
  DataPtr dp(base, [](void*) noexcept {});
  auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), /*nbytes=*/4096);

  // Tensor with sizes [2,3], strides [3,1], offset k=5 elements, dtype float32 (4 bytes)
  TensorImpl t(storage, {2,3}, {3,1}, /*storage_offset=*/5, ScalarType::Float32, Device::cpu());
  ASSERT_EQ(t.numel(), 6);
  auto* expect = reinterpret_cast<std::uint8_t*>(base) + 4 * 5;
  EXPECT_EQ(t.data(), reinterpret_cast<void*>(expect));
  EXPECT_TRUE(t.is_contiguous());
  EXPECT_TRUE(t.is_non_overlapping_and_dense());

  // Empty tensor (one dim zero) always returns nullptr data
  TensorImpl te(storage, {0, 3}, {3,1}, /*storage_offset=*/0, ScalarType::Float32, Device::cpu());
  EXPECT_EQ(te.numel(), 0);
  EXPECT_EQ(te.data(), nullptr);

  // Rank-0 scalar
  TensorImpl ts(storage, {}, {}, /*storage_offset=*/0, ScalarType::Float32, Device::cpu());
  EXPECT_EQ(ts.numel(), 1);
  EXPECT_NE(ts.data(), nullptr);
}
