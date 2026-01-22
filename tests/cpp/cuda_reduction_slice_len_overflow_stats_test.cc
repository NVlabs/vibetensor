// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/tensor.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/reduction_env.h"
#include "vbt/cuda/storage.h"

using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::TensorImpl;

extern "C" TensorImpl vbt_cuda_sum_impl(const TensorImpl& self, std::vector<int64_t> dims, bool keepdim);

TEST(CudaReductionSliceLenOverflowStatsTest, SliceLenOverflowWritesOverflowReason) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

#if VBT_INTERNAL_TESTS
  const int dev = 0;
  vbt::cuda::DeviceGuard dg(dev);

  // Construct a tensor with huge sizes but minimal storage.
  // The dispatcher computes slice_len from sizes and should detect overflow
  // before attempting to build a TensorIterator or launch any kernels.
  auto storage = vbt::cuda::new_cuda_storage(sizeof(float), dev);

  const std::int64_t big = (1LL << 62);
  std::vector<std::int64_t> sizes{big, big, 1};
  std::vector<std::int64_t> strides{big, 1, 1};
  TensorImpl self(storage,
                  sizes,
                  strides,
                  /*storage_offset=*/0,
                  ScalarType::Float32,
                  Device::cuda(dev));

  vbt::cuda::reduction::reset_cuda_reduction_last_stats_for_tests();

  try {
    (void)vbt_cuda_sum_impl(self, /*dims=*/{0, 1}, /*keepdim=*/false);
    FAIL() << "Expected slice_len overflow";
  } catch (const std::runtime_error& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("slice_len overflow"), std::string::npos);
  }

  const auto stats = vbt::cuda::reduction::get_cuda_reduction_last_stats_for_tests();
  EXPECT_EQ(stats.ineligible_reason, vbt::cuda::reduction::CudaReduceIneligibleReason::Overflow);
#else
  GTEST_SKIP() << "Built without internal tests";
#endif
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
