// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>

#include "vbt/core/strided_loop.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/data_ptr.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::make_intrusive;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::LoopSpec;
using vbt::core::build_unary_spec;

static vbt::core::StoragePtr S(void* base, std::size_t nbytes) {
  return make_intrusive<Storage>(DataPtr(base, [](void*) noexcept {}), nbytes);
}

TEST(StridedLoopTest, OneDNegativeStrideStartPointer) {
  auto st = S(reinterpret_cast<void*>(0x8000), 4096);
  // 1D reversed view: sizes=[6], strides=[-1], storage_offset=5 (float32)
  TensorImpl t(st, {6}, {-1}, 5, ScalarType::Float32, Device::cpu());
  std::vector<int64_t> perm;
  LoopSpec sp = build_unary_spec(t, perm);
  ASSERT_GE(sp.ndim, 1);
  // Expected lower bound L in bytes relative to storage base:
  // L = item_b * (storage_offset + min_elem_off); min_elem_off = (n-1)*(-1) = -(n-1)
  const int64_t item_b = static_cast<int64_t>(t.itemsize());
  const int64_t min_elem_off = -(t.sizes()[0] - 1);
  const int64_t expected_L = item_b * (t.storage_offset() + min_elem_off);
  auto* data = static_cast<const std::uint8_t*>(t.data());
  const auto* base = data - (t.itemsize() * static_cast<std::size_t>(t.storage_offset()));
  const auto* p0 = reinterpret_cast<const std::uint8_t*>(sp.p0);
  EXPECT_EQ(p0, base + expected_L);
}
