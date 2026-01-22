// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

TEST(CudaAllocatorRecordStreamTest, OwnerStreamNoOpAndImmediateReuse) {
#if VBT_WITH_CUDA
  if (device_count() == 0) GTEST_SKIP() << "No CUDA device";
  DeviceIndex dev = 0;
  auto& alloc = Allocator::get(dev);

  Stream S1 = getStreamFromPool(false, dev);
  const std::size_t N = 1 << 15; // 32KiB
  void* p = alloc.raw_alloc(N, S1);
  ASSERT_NE(p, nullptr);

  // record_stream with owner/alloc stream must be a no-op
  alloc.record_stream(p, S1);

  // Free on S1 and immediately reuse on S1
  setCurrentStream(S1);
  alloc.raw_delete(p);

  void* q = alloc.raw_alloc(N, S1);
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(q, p);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
