// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;

TEST(CUDATensor, DeviceAndData) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }
  // Non-empty storage
  auto st = vbt::cuda::new_cuda_storage(4 * vbt::core::itemsize(ScalarType::Float32), /*device_index=*/0);
  TensorImpl t(st, {4}, {1}, 0, ScalarType::Float32, Device::cuda(0));
  EXPECT_EQ(t.device(), Device::cuda(0));
  EXPECT_TRUE(t.is_contiguous());
  EXPECT_NE(t.data(), nullptr);

  // Zero-size tensor view must have nullptr data
  auto st_empty = vbt::core::make_intrusive<Storage>(DataPtr(nullptr, nullptr), 0);
  TensorImpl tz(st_empty, {0}, {1}, 0, ScalarType::Float32, Device::cuda(0));
  EXPECT_EQ(tz.data(), nullptr);

  // Zero-size storage helper
  auto stz = vbt::cuda::new_cuda_storage(0, 0);
  EXPECT_EQ(stz->data(), nullptr);
  EXPECT_EQ(stz->nbytes(), 0u);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
