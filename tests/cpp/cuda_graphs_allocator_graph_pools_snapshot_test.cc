// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/graphs.h"
#include "vbt/cuda/stream.h"

using namespace vbt::cuda;

TEST(CudaGraphsAllocatorSnapshotTest, SnapshotIncludesNewPool) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() <= 0) {
    GTEST_SKIP() << "No CUDA devices";
  }

  DeviceIndex dev = 0;

  // Create a new graph-private pool and ensure it appears in snapshots.
  MempoolId id = Allocator::create_pool_id(dev);

  auto has_id = [&](const std::vector<GraphPoolSnapshot>& snaps) {
    for (const auto& s : snaps) {
      if (s.id.dev == id.dev && s.id.id == id.id) {
        return true;
      }
    }
    return false;
  };

  // All-devices snapshot must include the new pool.
  auto snaps_all = snapshot_graph_pools(std::nullopt);
  EXPECT_TRUE(has_id(snaps_all));

  // Device-only filter (dev, 0) should also include the new pool.
  auto snaps_dev = snapshot_graph_pools(MempoolId{dev, 0});
  EXPECT_TRUE(has_id(snaps_dev));

  // Pool-specific filter (dev, id>0) should return only this pool if present.
  auto snaps_single = snapshot_graph_pools(id);
  ASSERT_FALSE(snaps_single.empty());
  for (const auto& s : snaps_single) {
    EXPECT_EQ(s.id.dev, id.dev);
    EXPECT_EQ(s.id.id, id.id);
  }

  // We intentionally do not assert on segments/bytes here; they are zero
  // until the pool is exercised by allocations.
#endif
}

TEST(CudaGraphsAllocatorSnapshotTest, InvalidDeviceFilterReturnsEmpty) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() <= 0) {
    GTEST_SKIP() << "No CUDA devices";
  }

  // dev < 0 is treated as an invalid filter and returns an empty snapshot.
  auto snaps_neg = snapshot_graph_pools(MempoolId{static_cast<DeviceIndex>(-1), 0});
  EXPECT_TRUE(snaps_neg.empty());

  int n = device_count();
  auto snaps_oor = snapshot_graph_pools(MempoolId{static_cast<DeviceIndex>(n + 5), 0});
  EXPECT_TRUE(snaps_oor.empty());
#endif
}

TEST(CudaGraphsAllocatorSnapshotTest,
     EmptyCacheDoesNotFreeGraphPoolSegments) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
#ifndef VBT_INTERNAL_TESTS
  GTEST_SKIP() << "VBT_INTERNAL_TESTS required for allocator debug helpers";
#else
  if (device_count() <= 0) {
    GTEST_SKIP() << "No CUDA devices";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (A.debug_backend_kind_for_testing() != BackendKind::Native) {
    GTEST_SKIP() << "Requires native backend";
  }

  // Quiesce allocator and start from a clean state.
  A.process_events(-1);
  A.emptyCache();
  A.resetAccumulatedStats();

  // Create a graph-private pool and prewarm it so that pool-tagged
  // segments exist in the allocator.
  MempoolId id = Allocator::create_pool_id(dev);
  Stream s = getStreamFromPool(false, dev);

  const std::size_t kBytes = 1u << 20;  // 1 MiB
  Allocator::prewarm_graph_pool_for_stream(dev, id, s, kBytes, /*min_blocks=*/1);

  // Also create at least one global (non-pooled) idle segment so that
  // emptyCache() has something it is allowed to reclaim.
  void* global_ptr = nullptr;
  try {
    global_ptr = A.raw_alloc(kBytes, s);
  } catch (...) {
    global_ptr = nullptr;
  }
  if (global_ptr) {
    setCurrentStream(s);
    A.raw_delete(global_ptr);
    A.process_events(-1);
  }

  auto segs_before = snapshot_memory_segments(dev);
  auto pools_before = snapshot_graph_pools(id);

  if (pools_before.empty()) {
    GTEST_SKIP() << "Graph pool has no segments; prewarm may have failed";
  }

  std::uint64_t pool_bytes_reserved_before = 0;
  for (const auto& snap : pools_before) {
    pool_bytes_reserved_before += snap.bytes_reserved;
  }

  std::uint64_t global_reserved_before = 0;
  for (const auto& seg : segs_before) {
    if (seg.pool_id == 0) {
      global_reserved_before += seg.bytes_reserved;
    }
  }

  // Run emptyCache(); this should only reclaim global segments.
  A.emptyCache();

  auto segs_after = snapshot_memory_segments(dev);
  auto pools_after = snapshot_graph_pools(id);

  std::uint64_t pool_bytes_reserved_after = 0;
  for (const auto& snap : pools_after) {
    pool_bytes_reserved_after += snap.bytes_reserved;
  }

  std::uint64_t global_reserved_after = 0;
  for (const auto& seg : segs_after) {
    if (seg.pool_id == 0) {
      global_reserved_after += seg.bytes_reserved;
    }
  }

  // Pool segments must be preserved across emptyCache(); only global
  // segments are eligible for reclamation.
  EXPECT_EQ(pool_bytes_reserved_after, pool_bytes_reserved_before);
  EXPECT_LE(global_reserved_after, global_reserved_before);

  // Best-effort cleanup of the pool for subsequent tests.
  Allocator::release_pool(dev, id);
#endif  // VBT_INTERNAL_TESTS
#endif  // VBT_WITH_CUDA
}
