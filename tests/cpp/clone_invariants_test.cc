// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <cstring>

#include "vbt/core/tensor.h"
#include "vbt/core/tensor_ops.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;

TEST(CloneInvariantsTest, StridedCloneCopiesData) {
  // Base contiguous [2,3] floats filled 0..5
  const int64_t R = 2, C = 3;
  std::size_t nbytes = static_cast<std::size_t>(R*C) * sizeof(float);
  float* buf = static_cast<float*>(::operator new(nbytes));
  for (int i=0;i<R*C;++i) buf[i] = static_cast<float>(i);
  auto dp = DataPtr(buf, [](void* p) noexcept { ::operator delete(p); });
  auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
  TensorImpl base(storage, {R,C}, {C,1}, 0, ScalarType::Float32, Device::cpu());

  // Make a transposed logical view sizes [3,2], strides [1,3]
  TensorImpl transposed(storage, {C,R}, {1,C}, 0, ScalarType::Float32, Device::cpu());

  // Clone arbitrary strided
  auto cloned = vbt::core::clone_cpu(transposed);
  EXPECT_EQ(cloned.sizes(), (std::vector<int64_t>{C,R}));
  EXPECT_TRUE(cloned.is_contiguous());
  EXPECT_EQ(cloned.storage_offset(), 0);

  // Verify values
  float* out = static_cast<float*>(cloned.data());
  ASSERT_NE(out, nullptr);
  for (int64_t i=0;i<C;++i) {
    for (int64_t j=0;j<R;++j) {
      // transposed(i,j) corresponds to base(j,i)
      float expect = buf[j*C + i];
      float got = out[i*R + j];
      EXPECT_EQ(got, expect);
    }
  }
}

TEST(CloneInvariantsTest, ZeroSizeClone) {
  auto storage = vbt::core::make_intrusive<Storage>(DataPtr(nullptr, [](void*) noexcept {}), 0);
  TensorImpl empty(storage, {0,3}, {3,1}, 0, ScalarType::Float32, Device::cpu());
  auto cloned = vbt::core::clone_cpu(empty);
  EXPECT_EQ(cloned.numel(), 0);
  EXPECT_TRUE(cloned.is_contiguous());
}
