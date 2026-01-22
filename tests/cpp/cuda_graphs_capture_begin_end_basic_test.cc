// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/graphs.h"
#include "vbt/cuda/stream.h"

using namespace vbt::cuda;

TEST(CUDAGraphCaptureLifecycleTest, CaptureBeginEndBasic) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;

  GraphCounters before_counters = cuda_graphs_counters();

  Stream s = getStreamFromPool(false, dev);

  {
    CUDAGraph g;
    EXPECT_FALSE(g.is_capturing());

    // Basic lifecycle: begin/end on a non-default stream
    g.capture_begin(s);
    EXPECT_TRUE(g.is_capturing());
    EXPECT_EQ(g.device(), dev);
    EXPECT_EQ(g.capture_stream().id(), s.id());

    g.capture_end();
    EXPECT_FALSE(g.is_capturing());

    GraphCounters mid_counters = cuda_graphs_counters();
    EXPECT_EQ(mid_counters.captures_started, before_counters.captures_started + 1);
    EXPECT_EQ(mid_counters.captures_ended, before_counters.captures_ended + 1);
  }
#endif
}
