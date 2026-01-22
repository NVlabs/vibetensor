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

namespace {

Allocator& alloc0() {
  return Allocator::get(0);
}

DeviceStats stats_snapshot() {
  return alloc0().getDeviceStats();
}

} // namespace

TEST(CUDAGraphResetAndDtorPoolLifecycleTest, ResetReleasesImplicitPool) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  Stream s = getStreamFromPool(false, dev);
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());

  DeviceStats stats_before = stats_snapshot();
  GraphCounters counters_before = cuda_graphs_counters();

  float* d = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d), sizeof(float)), cudaSuccess);

  {
    CUDAGraph g;

    // Capture a simple memcpy on the capture stream.
    g.capture_begin(s);

    int host_value = 42;
    ASSERT_EQ(cudaMemcpyAsync(d,
                              &host_value,
                              sizeof(int),
                              cudaMemcpyHostToDevice,
                              raw),
              cudaSuccess);

    g.capture_end();
    g.instantiate();

    MempoolId pool = g.pool();
    EXPECT_TRUE(pool.is_valid());

    // Reset should destroy graph/exec and release the pool.
    g.reset();

    // After reset, the graph's pool handle should be invalid.
    EXPECT_FALSE(g.pool().is_valid());
  }

  ASSERT_EQ(cudaFree(d), cudaSuccess);

  DeviceStats stats_after = stats_snapshot();
  GraphCounters counters_after = cuda_graphs_counters();

  EXPECT_EQ(stats_after.graphs_pools_created,
            stats_before.graphs_pools_created + 1);
  EXPECT_EQ(stats_after.graphs_pools_released,
            stats_before.graphs_pools_released + 1);
  EXPECT_EQ(stats_after.graphs_pools_active,
            stats_before.graphs_pools_active);

  EXPECT_EQ(counters_after.graphs_reset,
            counters_before.graphs_reset + 1);
#endif
}

TEST(CUDAGraphResetAndDtorPoolLifecycleTest, DestructorEventuallyReleasesPool) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  Stream s = getStreamFromPool(false, dev);
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());

  DeviceStats stats_before = stats_snapshot();

  float* d = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d), sizeof(float)), cudaSuccess);

  {
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

    // Trigger a replay and then immediately destroy the graph without
    // explicitly synchronizing the stream. Depending on timing, cleanup may
    // happen either directly in the destructor or be deferred to the host
    // callback, but in all cases the allocator should eventually release the
    // pool and return gauges to their baseline.
    g.replay();
  }

  // Wait for the replay and any deferred host callbacks to complete.
  ASSERT_EQ(cudaStreamSynchronize(raw), cudaSuccess);

  ASSERT_EQ(cudaFree(d), cudaSuccess);

  vbt::cuda::detail::poll_deferred_graph_cleanup();

  DeviceStats stats_after = stats_snapshot();

  EXPECT_EQ(stats_after.graphs_pools_created,
            stats_before.graphs_pools_created + 1);
  EXPECT_EQ(stats_after.graphs_pools_released,
            stats_before.graphs_pools_released + 1);
  EXPECT_EQ(stats_after.graphs_pools_active,
            stats_before.graphs_pools_active);
#endif
}

TEST(CUDAGraphDeferredCleanupTest, DestroyWhileInflightThenPoll) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  Stream s = getStreamFromPool(false, dev);
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());

  DeviceStats stats_before = stats_snapshot();

  float* d = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d), sizeof(float)), cudaSuccess);

  {
    CUDAGraph g;

    g.capture_begin(s);

    int host_value = 13;
    ASSERT_EQ(cudaMemcpyAsync(d,
                              &host_value,
                              sizeof(int),
                              cudaMemcpyHostToDevice,
                              raw),
              cudaSuccess);

    g.capture_end();
    g.instantiate();

    // Replay once; do NOT synchronize before destroying the graph to
    // exercise the inflight-dtor deferred-cleanup path.
    g.replay();
  }

  // Wait for the replay and host callback to complete, then poll for
  // deferred cleanup so that graph/exec/pool resources are released.
  ASSERT_EQ(cudaStreamSynchronize(raw), cudaSuccess);
  vbt::cuda::detail::poll_deferred_graph_cleanup();

  ASSERT_EQ(cudaFree(d), cudaSuccess);

  DeviceStats stats_after = stats_snapshot();

  EXPECT_EQ(stats_after.graphs_pools_created,
            stats_before.graphs_pools_created + 1);
  EXPECT_EQ(stats_after.graphs_pools_released,
            stats_before.graphs_pools_released + 1);
  EXPECT_EQ(stats_after.graphs_pools_active,
            stats_before.graphs_pools_active);
#endif
}
