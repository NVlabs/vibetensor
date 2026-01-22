// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/cub.h"
#include "vbt/cuda/stream.h"

namespace {

TEST(CudaCubWrappersTest, InvalidStreamDeviceIndexThrows) {
  auto& alloc = vbt::cuda::Allocator::get(0);
  vbt::cuda::Stream bad(vbt::cuda::Stream::UNCHECKED, 0u, static_cast<vbt::cuda::DeviceIndex>(-1));

  EXPECT_THROW(
      vbt::cuda::cub::reduce_sum_u8_as_i32(alloc, bad, /*d_flags01=*/nullptr, /*n=*/0, /*d_out_sum=*/nullptr),
      std::invalid_argument);
}

} // namespace
