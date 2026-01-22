// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/graphs.h"

using namespace vbt::cuda;

namespace {

static inline bool has_substr(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}

Allocator& alloc0() {
  return Allocator::get(0);
}

DeviceStats stats_snapshot() {
  return alloc0().getDeviceStats();
}

} // namespace

TEST(CudaGraphsAllocatorPoolReplayBusyTest, BeginAndReplayRespectBusyPredicate) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  auto before = stats_snapshot();

  // Create a pool and simulate a graph retaining it so that it is not
  // immediately GC'd by end/cancel.
  MempoolId id = Allocator::create_pool_id(dev);
  Allocator::retain_pool(dev, id);

  // Mark the pool as in replay.
  Allocator::mark_pool_replay_begin(dev, id);

  // A second begin_allocate_to_pool on the same pool must fail with the
  // canonical busy-pool substring.
  try {
    (void)Allocator::begin_allocate_to_pool(dev, id);
    FAIL() << "Expected busy pool error for begin_allocate_to_pool";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_TRUE(has_substr(msg, "pool is busy with active capture"));
  }

  // A second mark_pool_replay_begin must also see the pool as busy.
  try {
    Allocator::mark_pool_replay_begin(dev, id);
    FAIL() << "Expected busy pool error for mark_pool_replay_begin";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_TRUE(has_substr(msg, "pool is busy with active capture"));
  }

  // End the replay; pool should become available again.
  Allocator::mark_pool_replay_end(dev, id);

  // Now begin_allocate_to_pool should succeed and be able to end cleanly.
  {
    auto guard = Allocator::begin_allocate_to_pool(dev, id);
    EXPECT_TRUE(guard.active());
    guard.end();
  }

  // A fresh replay begin/end pair should also succeed.
  Allocator::mark_pool_replay_begin(dev, id);
  Allocator::mark_pool_replay_end(dev, id);

  // Drop the retained ref so the pool can be erased when quiescent.
  Allocator::release_pool(dev, id);

  auto after = stats_snapshot();
  EXPECT_EQ(after.graphs_pools_created, before.graphs_pools_created + 1);
  EXPECT_EQ(after.graphs_pools_released, before.graphs_pools_released + 1);
  EXPECT_EQ(after.graphs_pools_active, before.graphs_pools_active);
#endif
}
