// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/graphs.h"
#include "vbt/cuda/stream.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace vbt::cuda;

TEST(CUDAGraphDtorFallbackTest, EndsCaptureWhenUserSkipsEnd) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  GraphCounters before = cuda_graphs_counters();

  {
    Stream s = getStreamFromPool(false, dev);
    ASSERT_FALSE(A.debug_device_routing_active());

    {
      CUDAGraph g;
      g.capture_begin(s);
      EXPECT_TRUE(g.is_capturing());
      EXPECT_TRUE(A.debug_device_routing_active());
      // Intentionally omit capture_end(); destructor must perform fallback.
    }

    // After CUDAGraph destruction, allocator routing must be cancelled.
    EXPECT_FALSE(A.debug_device_routing_active());
  }

  GraphCounters after = cuda_graphs_counters();
  EXPECT_EQ(after.captures_started, before.captures_started + 1);
  // No explicit capture_end call took place.
  EXPECT_EQ(after.captures_ended, before.captures_ended);
  EXPECT_EQ(after.end_in_dtor, before.end_in_dtor + 1);
  EXPECT_GE(after.end_in_dtor_errors, before.end_in_dtor_errors);
#endif
}
