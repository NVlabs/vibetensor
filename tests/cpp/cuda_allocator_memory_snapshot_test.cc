// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

TEST(CudaAllocatorMemorySnapshotTest, NativeSnapshotTracksStatsBounds) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "Built without CUDA";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "CUDA device required";
  }

  DeviceIndex dev = 0;
  Allocator& alloc = Allocator::get(dev);

  // Best-effort: start from a clean cache.
  alloc.emptyCache();

  Stream s = getStreamFromPool(false, dev);

  const std::size_t N1 = 1ull << 20; // 1 MiB
  const std::size_t N2 = 2ull << 20; // 2 MiB

  void* p1 = alloc.raw_alloc(N1, s);
  ASSERT_NE(p1, nullptr);
  void* p2 = alloc.raw_alloc(N2, s);
  ASSERT_NE(p2, nullptr);

  DeviceStats stats_before_free = alloc.getDeviceStats();
  auto snaps = snapshot_memory_segments(dev);

  std::uint64_t snap_reserved = 0;
  std::uint64_t snap_active = 0;
  for (const auto& seg : snaps) {
    EXPECT_EQ(seg.device, dev);
    EXPECT_GE(seg.blocks, 1u);
    snap_reserved += seg.bytes_reserved;
    snap_active += seg.bytes_active;
  }

  EXPECT_LE(snap_reserved, stats_before_free.reserved_bytes_all_current);
  EXPECT_LE(snap_active, stats_before_free.allocated_bytes_all_current);

  // Free allocations but keep cache; reserved may stay >0 while active drops.
  setCurrentStream(s);
  alloc.raw_delete(p1);
  alloc.raw_delete(p2);

  auto snaps_after_free = snapshot_memory_segments(dev);

  std::uint64_t reserved_after_free = 0;
  std::uint64_t active_after_free = 0;
  for (const auto& seg : snaps_after_free) {
    reserved_after_free += seg.bytes_reserved;
    active_after_free += seg.bytes_active;
  }

  EXPECT_EQ(active_after_free, 0u);
  EXPECT_GE(reserved_after_free, snap_reserved);

  // Final cleanup for isolation.
  alloc.emptyCache();
#endif
}

TEST(CudaAllocatorMemorySnapshotTest, CpuOnlyOrNoDeviceReturnsEmpty) {
#if !VBT_WITH_CUDA
  auto snaps = snapshot_memory_segments(std::nullopt);
  EXPECT_TRUE(snaps.empty());
#else
  if (device_count() != 0) {
    GTEST_SKIP() << "Requires no CUDA devices";
  }
  auto snaps_all = snapshot_memory_segments(std::nullopt);
  EXPECT_TRUE(snaps_all.empty());
  auto snaps_dev0 = snapshot_memory_segments(static_cast<DeviceIndex>(0));
  EXPECT_TRUE(snaps_dev0.empty());
#endif
}
