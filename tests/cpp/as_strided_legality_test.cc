// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <limits>
#include <cstdint>

#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/intrusive_ptr.h"

using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;

static vbt::core::StoragePtr make_storage(void* base, std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(DataPtr(base, [](void*) noexcept {}), nbytes);
}

TEST(AsStridedLegalityTest, MixedSignsWithinBounds) {
  auto base = reinterpret_cast<void*>(0x1000);
  auto storage = make_storage(base, /*nbytes=*/4096);
  TensorImpl base_t(storage, {10, 10}, {10, 1}, /*offset=*/0, ScalarType::Float32, Device::cpu());

  std::vector<int64_t> sz{3, 3};
  std::vector<int64_t> st{-10, 0};
  // storage_offset such that coverage stays in bounds
  auto v = base_t.as_strided(sz, st, /*offset=*/50);
  EXPECT_EQ(v.sizes(), sz);
  EXPECT_EQ(v.strides(), st);
  EXPECT_EQ(v.storage_offset(), 50);
  EXPECT_EQ(v.dtype(), ScalarType::Float32);
  EXPECT_EQ(v.device(), Device::cpu());
}

TEST(AsStridedLegalityTest, ZeroSizeStillEnforcesUpperBoundAndNullData) {
  auto base = reinterpret_cast<void*>(0x2000);
  auto storage = make_storage(base, /*nbytes=*/64); // small storage
  TensorImpl base_t(storage, {4, 4}, {4, 1}, /*offset=*/0, ScalarType::Float32, Device::cpu());

  // Even with size zero, upper bound must not exceed storage
  std::vector<int64_t> sz{0, 1000000};
  std::vector<int64_t> st{1, 1};
  // storage_offset beyond storage should fail regardless of size zero generosity
  EXPECT_THROW(base_t.as_strided(sz, st, /*offset=*/1000), std::out_of_range);

  // A legal zero-size view yields data()==nullptr in result
  auto v = base_t.as_strided({0, 4}, {4, 1}, /*offset=*/0);
  EXPECT_EQ(v.data(), nullptr);
}

TEST(AsStridedLegalityTest, OverflowAndInvalidGuards) {
  auto base = reinterpret_cast<void*>(0x3000);
  auto storage = make_storage(base, /*nbytes=*/4096);
  TensorImpl base_t(storage, {2, 2}, {2, 1}, /*offset=*/0, ScalarType::Float32, Device::cpu());

  // INT64_MIN stride rejection
  EXPECT_THROW(base_t.as_strided({1}, {std::numeric_limits<int64_t>::min()}, 0), std::invalid_argument);

  // Length mismatch
  EXPECT_THROW(base_t.as_strided({1,2}, {1}, 0), std::invalid_argument);

  // Negative size
  EXPECT_THROW(base_t.as_strided({-1}, {1}, 0), std::invalid_argument);

  // Negative storage_offset
  EXPECT_THROW(base_t.as_strided({1}, {1}, -1), std::invalid_argument);

  // Overflow via large stride*extent
  EXPECT_THROW(base_t.as_strided({2}, {std::numeric_limits<int64_t>::max()}, 0), std::overflow_error);
}
