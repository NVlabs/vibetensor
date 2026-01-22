// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/intrusive_ptr.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::Device;
using vbt::core::ScalarType;

static vbt::core::intrusive_ptr<Storage> dummy_storage(std::size_t nbytes=4096) {
  void* base = reinterpret_cast<void*>(0x1000);
  vbt::core::DataPtr dp(base, [](void*) noexcept {});
  return vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
}

TEST(ContiguityNoDTest, BasicTrueCases) {
  auto storage = dummy_storage();
  // Contiguous 2D
  TensorImpl t0(storage, {2,3}, {3,1}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_TRUE(t0.is_non_overlapping_and_dense());
  // Transposed dense: sizes [3,2], strides [1,3]
  TensorImpl t1(storage, {3,2}, {1,3}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_TRUE(t1.is_non_overlapping_and_dense());
  // Size-1 dims are benign
  TensorImpl t2(storage, {2,1,3}, {3,3,1}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_TRUE(t2.is_non_overlapping_and_dense());
  // Zero-size tensors are considered dense
  TensorImpl t3(storage, {0,4}, {4,1}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_TRUE(t3.is_non_overlapping_and_dense());
}

TEST(ContiguityNoDTest, BasicFalseCases) {
  auto storage = dummy_storage();
  // Stride mismatch (non-dense)
  TensorImpl t0(storage, {2,3}, {4,1}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_FALSE(t0.is_non_overlapping_and_dense());
  // Negative stride with more than one element
  TensorImpl t1(storage, {2,3}, {3,-1}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_FALSE(t1.is_non_overlapping_and_dense());
}

TEST(ContiguityNoDTest, OneDimensional) {
  auto storage = dummy_storage();
  // 1D: size < 2 -> true regardless of stride
  TensorImpl t0(storage, {1}, {7}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_TRUE(t0.is_non_overlapping_and_dense());
  // 1D: size >= 2 requires stride==1
  TensorImpl t1(storage, {5}, {1}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_TRUE(t1.is_non_overlapping_and_dense());
  TensorImpl t2(storage, {5}, {2}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_FALSE(t2.is_non_overlapping_and_dense());
}
