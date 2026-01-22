// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "vbt/cuda/graphs.h"
#include "vbt/cuda/stream.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace vbt::cuda;

namespace {
static inline bool has_substr(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}
} // namespace

TEST(CUDAGraphReplayGuardTest, NestedReplayThrowsAndCounts) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  Stream s = getStreamFromPool(false, dev);
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());

  float* d = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d), sizeof(float)), cudaSuccess);

  CUDAGraph g;
  g.capture_begin(s);

  int host_value = 7;
  ASSERT_EQ(cudaMemcpyAsync(d,
                            &host_value,
                            sizeof(int),
                            cudaMemcpyHostToDevice,
                            raw),
            cudaSuccess);

  g.capture_end();
  g.instantiate();

  GraphCounters before = cuda_graphs_counters();

#ifdef VBT_INTERNAL_TESTS
  {
    ReplayGuardTestScope scope;
    try {
      g.replay();
      FAIL() << "Expected nested ReplayGuard exception";
    } catch (const std::runtime_error& e) {
      std::string msg = e.what();
      EXPECT_TRUE(has_substr(msg, std::string(kErrReplayNestedGuard)));
    }
  }

  GraphCounters after = cuda_graphs_counters();
  EXPECT_EQ(after.replay_nesting_errors,
            before.replay_nesting_errors + 1);
  EXPECT_EQ(after.graphs_replayed, before.graphs_replayed);
#else
  (void)before;
#endif

  ASSERT_EQ(cudaFree(d), cudaSuccess);
#endif
}
