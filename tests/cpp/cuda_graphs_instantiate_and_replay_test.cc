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

TEST(CUDAGraphInstantiateAndReplayTest, BasicInstantiateAndReplayCounters) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  Stream s = getStreamFromPool(false, dev);
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());

  GraphCounters before = cuda_graphs_counters();

  float* d = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d), sizeof(float)), cudaSuccess);

  {
    CUDAGraph g;

    // Capture a simple memcpy on the capture stream.
    g.capture_begin(s);

    int host_value = 42;
    ASSERT_EQ(
        cudaMemcpyAsync(d,
                        &host_value,
                        sizeof(int),
                        cudaMemcpyHostToDevice,
                        raw),
        cudaSuccess);

    g.capture_end();
    g.instantiate();

#ifdef VBT_INTERNAL_TESTS
    EXPECT_TRUE(g.debug_has_graph());
    EXPECT_TRUE(g.debug_has_exec());
#endif

    // Replay once on the original capture stream and wait for completion
    // to exercise the basic instantiate + replay path.
    g.replay();
    ASSERT_EQ(cudaStreamSynchronize(raw), cudaSuccess);

    // Wait for replays and host callbacks to finish (second sync is a no-op
    // but keeps the structure similar to the original test).
    ASSERT_EQ(cudaStreamSynchronize(raw), cudaSuccess);

#ifdef VBT_INTERNAL_TESTS
    EXPECT_EQ(g.debug_inflight(), 0);
#endif
  }

  GraphCounters after = cuda_graphs_counters();
  EXPECT_EQ(after.graphs_instantiated, before.graphs_instantiated + 1);
  EXPECT_EQ(after.graphs_replayed, before.graphs_replayed + 1);

  ASSERT_EQ(cudaFree(d), cudaSuccess);
#endif
}
