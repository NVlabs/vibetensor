// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <string>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/graphs.h"

using namespace vbt::cuda;

namespace {

// Helper to get allocator for device 0 (or current when CUDA is disabled).
static Allocator& alloc0() {
  return Allocator::get(0);
}

DeviceStats stats_snapshot() {
  return alloc0().getDeviceStats();
}

} // namespace

TEST(CudaGraphsAllocatorPoolsTest, CreatePoolIdIncrementsCreatedCounter) {
  auto before = stats_snapshot();

  MempoolId id1 = Allocator::create_pool_id(0);
  MempoolId id2 = Allocator::create_pool_id(0);

  EXPECT_TRUE(id1.is_valid());
  EXPECT_TRUE(id2.is_valid());
  EXPECT_EQ(id1.dev, static_cast<DeviceIndex>(0));
  EXPECT_EQ(id2.dev, static_cast<DeviceIndex>(0));
  EXPECT_LT(id1.id, id2.id);

  auto after = stats_snapshot();
  EXPECT_EQ(after.graphs_pools_created, before.graphs_pools_created + 2);
  EXPECT_EQ(after.graphs_pools_active, before.graphs_pools_active);
  EXPECT_EQ(after.graphs_pools_released, before.graphs_pools_released);

  // Release pools so later tests observe a consistent base.
  Allocator::release_pool(0, id1);
  Allocator::release_pool(0, id2);
}

TEST(CudaGraphsAllocatorPoolsTest, RetainReleaseGaugeTransitions) {
  auto before = stats_snapshot();

  MempoolId id = Allocator::create_pool_id(0);
  auto after_create = stats_snapshot();
  EXPECT_EQ(after_create.graphs_pools_created, before.graphs_pools_created + 1);
  EXPECT_EQ(after_create.graphs_pools_active, before.graphs_pools_active);

  Allocator::retain_pool(0, id);
  auto after_first_retain = stats_snapshot();
  EXPECT_EQ(after_first_retain.graphs_pools_active, before.graphs_pools_active + 1);

  // Second retain should not bump active gauge again.
  Allocator::retain_pool(0, id);
  auto after_second_retain = stats_snapshot();
  EXPECT_EQ(after_second_retain.graphs_pools_active, before.graphs_pools_active + 1);

  // First release keeps pool active (refcnt from 2 -> 1).
  Allocator::release_pool(0, id);
  auto after_first_release = stats_snapshot();
  EXPECT_EQ(after_first_release.graphs_pools_active, before.graphs_pools_active + 1);

  // Second release drops refcnt to 0; pool can be erased.
  Allocator::release_pool(0, id);
  auto after_second_release = stats_snapshot();
  EXPECT_EQ(after_second_release.graphs_pools_active, before.graphs_pools_active);
  EXPECT_EQ(after_second_release.graphs_pools_released,
            before.graphs_pools_released + 1);
}

TEST(CudaGraphsAllocatorPoolsTest, OverReleaseIsNoOpAndGaugeNonNegative) {
  auto before = stats_snapshot();

  MempoolId id = Allocator::create_pool_id(0);
  Allocator::release_pool(0, id); // refcnt==0, entry erased, released++
  auto after_first = stats_snapshot();

  // Second release is a no-op; counters do not change.
  Allocator::release_pool(0, id);
  auto after_second = stats_snapshot();

  EXPECT_EQ(after_second.graphs_pools_created, after_first.graphs_pools_created);
  EXPECT_EQ(after_second.graphs_pools_released, after_first.graphs_pools_released);
  // Gauge must never underflow.
  EXPECT_GE(after_second.graphs_pools_active, 0u);
}

TEST(CudaGraphsAllocatorPoolsTest, UnknownIdRetainAndBeginThrow) {
  auto before = stats_snapshot();

  // Create and then fully release a pool; subsequent uses should be unknown.
  MempoolId id = Allocator::create_pool_id(0);
  Allocator::release_pool(0, id);

  auto expect_unknown_substring = [](const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("unknown mempool id"), std::string::npos);
  };

  try {
    Allocator::retain_pool(0, id);
    FAIL() << "retain_pool should throw for unknown id";
  } catch (const std::runtime_error& e) {
    expect_unknown_substring(e);
  }

  try {
    (void)Allocator::begin_allocate_to_pool(0, id);
    FAIL() << "begin_allocate_to_pool should throw for unknown id";
  } catch (const std::runtime_error& e) {
    expect_unknown_substring(e);
  }

  auto after = stats_snapshot();
  // Failure paths must not mutate counters.
  EXPECT_EQ(after.graphs_pools_created, before.graphs_pools_created + 1);
  EXPECT_EQ(after.graphs_pools_released, before.graphs_pools_released + 1);
}

TEST(CudaGraphsAllocatorPoolsTest, EndAndCancelOnUnknownAreNoOps) {
  auto before = stats_snapshot();

  MempoolId bogus{static_cast<DeviceIndex>(0), 0};
  // Should not throw and should not mutate counters.
  Allocator::end_allocate_to_pool(0, bogus);
  Allocator::cancel_allocate_to_pool(0, bogus);

  auto after = stats_snapshot();
  EXPECT_EQ(after.graphs_pools_created, before.graphs_pools_created);
  EXPECT_EQ(after.graphs_pools_released, before.graphs_pools_released);
  EXPECT_EQ(after.graphs_pools_active, before.graphs_pools_active);
}

TEST(CudaGraphsAllocatorPoolsTest, BusyBeginDeniedWithPinnedMessage) {
  auto before = stats_snapshot();

  MempoolId id = Allocator::create_pool_id(0);
  // Simulate a graph retaining the pool so it is not erased on end.
  Allocator::retain_pool(0, id);

  auto guard = Allocator::begin_allocate_to_pool(0, id);

  try {
    (void)Allocator::begin_allocate_to_pool(0, id);
    FAIL() << "Expected busy pool error";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("pool is busy with active capture"), std::string::npos);
  }

  // End capture via guard; pool remains because refcnt==1.
  guard.end();

  auto mid = stats_snapshot();
  EXPECT_EQ(mid.graphs_pools_active, before.graphs_pools_active + 1);

  // Releasing the pool now should drop the gauge and increment released.
  Allocator::release_pool(0, id);
  auto after = stats_snapshot();
  EXPECT_EQ(after.graphs_pools_active, before.graphs_pools_active);
  EXPECT_EQ(after.graphs_pools_released, before.graphs_pools_released + 1);
}

TEST(CudaGraphsAllocatorPoolsTest, GuardDestructorCancelsCapture) {
  auto before = stats_snapshot();

  MempoolId id = Allocator::create_pool_id(0);

  // Begin capture but rely on destructor to cancel.
  {
    auto guard = Allocator::begin_allocate_to_pool(0, id);
    EXPECT_TRUE(guard.active());
  }

  // After guard destruction, active_capture_count must be reset and the pool
  // removed because refcnt==0.
  auto after = stats_snapshot();
  EXPECT_EQ(after.graphs_pools_active, before.graphs_pools_active);
  EXPECT_EQ(after.graphs_pools_released, before.graphs_pools_released + 1);
}

TEST(CudaGraphsAllocatorPoolsTest, GuardMoveSemanticsTransferEngagement) {
  MempoolId id = Allocator::create_pool_id(0);

  Allocator::retain_pool(0, id);

  Allocator::AllocateToPoolGuard g1 = Allocator::begin_allocate_to_pool(0, id);
  EXPECT_TRUE(g1.active());

  Allocator::AllocateToPoolGuard g2(std::move(g1));
  EXPECT_FALSE(g1.active());
  EXPECT_TRUE(g2.active());

  // Move-assign should cancel the previous engagement and adopt the new one.
  Allocator::AllocateToPoolGuard g3;
  g3 = std::move(g2);
  EXPECT_FALSE(g2.active());
  EXPECT_TRUE(g3.active());

  g3.end();
  EXPECT_FALSE(g3.active());

  // Drop refcount and erase pool.
  Allocator::release_pool(0, id);
}
