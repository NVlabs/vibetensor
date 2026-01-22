// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>

#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/overlap.h"
#include "vbt/core/intrusive_ptr.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::assert_no_overlap;
using vbt::core::assert_no_partial_overlap;

static vbt::core::StoragePtr make_storage_ol(void* base, std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(DataPtr(base, [](void*) noexcept {}), nbytes);
}

TEST(OverlapCrossTensorTest, BasicSymmetryAndAdjacency) {
  auto storage = make_storage_ol(reinterpret_cast<void*>(0x8000), 4096);
  TensorImpl a(storage, {4,1}, {1,1}, 0, ScalarType::Float32, Device::cpu()); // [0,16)
  TensorImpl b(storage, {4,1}, {1,1}, 4, ScalarType::Float32, Device::cpu()); // [16,32)

  // Adjacent: no overlap
  EXPECT_NO_THROW(assert_no_overlap(a,b));
  EXPECT_NO_THROW(assert_no_overlap(b,a));
}

TEST(OverlapCrossTensorTest, PartialAndExact) {
  auto storage = make_storage_ol(reinterpret_cast<void*>(0x9000), 4096);
  // Same span but different shapes -> partial overlap
  TensorImpl a(storage, {4}, {1}, 0, ScalarType::Float32, Device::cpu());
  TensorImpl b(storage, {2,2}, {2,1}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_THROW(assert_no_partial_overlap(a,b), std::invalid_argument);

  // Exact alias
  TensorImpl c(storage, {2,2}, {2,1}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_NO_THROW(assert_no_partial_overlap(b,c));
}

TEST(OverlapCrossTensorTest, DifferentStorageStridedIsNotTooHard) {
  auto storage_a = make_storage_ol(
      reinterpret_cast<void*>(static_cast<std::uintptr_t>(0xA000)), 4096);
  auto storage_b = make_storage_ol(
      reinterpret_cast<void*>(static_cast<std::uintptr_t>(0xB000)), 4096);

  // Strides create a non-dense (NO&D=false) layout that would be TooHard if we
  // only relied on NO&D checks.
  TensorImpl a(storage_a, {2,2}, {3,1}, 0, ScalarType::Float32, Device::cpu());
  TensorImpl b(storage_b, {2,2}, {3,1}, 0, ScalarType::Float32, Device::cpu());

  EXPECT_FALSE(a.is_non_overlapping_and_dense_or_false());
  EXPECT_FALSE(b.is_non_overlapping_and_dense_or_false());

  using vbt::core::MemOverlapStatus;
  EXPECT_EQ(vbt::core::get_overlap_status(a, b), MemOverlapStatus::No);
  EXPECT_EQ(vbt::core::get_overlap_status(b, a), MemOverlapStatus::No);
}
