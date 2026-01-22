// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

TEST(CudaAllocatorRoundingTest, RoundSizeAndClassify) {
  struct Case { std::size_t in; std::size_t rounded; PoolKind kind; };
  const std::size_t MiB = 1ull << 20;
  std::vector<Case> cases = {
    {0, 0, PoolKind::Small},
    {1, 512, PoolKind::Small},
    {511, 512, PoolKind::Small},
    {512, 512, PoolKind::Small},
    {513, 1024, PoolKind::Small},
    {MiB - 1, MiB, PoolKind::Small},
    {MiB, MiB, PoolKind::Small},
    {MiB + 1, 2*MiB, PoolKind::Large},
    {2*MiB - 1, 2*MiB, PoolKind::Large},
    {2*MiB, 2*MiB, PoolKind::Large},
    {2*MiB + 1, 4*MiB, PoolKind::Large},
  };
  for (auto c : cases) {
    auto r = round_size(c.in);
    EXPECT_EQ(r, c.rounded) << "in=" << c.in;
    EXPECT_EQ(classify(r), c.kind) << "in=" << c.in;
  }
}

TEST(CudaAllocatorRoundingTest, SameStreamReusePreservesReservedAndPointer) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "Built without CUDA";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "CUDA device required";
  }

  DeviceIndex dev = 0;
  Allocator& alloc = Allocator::get(dev);

  // Ensure a clean cache so reserved-bytes comparisons are meaningful.
  alloc.emptyCache();

  Stream s = getStreamFromPool(false, dev);

  const std::size_t N = 1ull << 20; // 1 MiB

  void* p1 = alloc.raw_alloc(N, s);
  ASSERT_NE(p1, nullptr);

  auto stats_after_first = alloc.getDeviceStats();

  // Same-stream free followed by same-stream alloc should reuse the block
  // without increasing reserved bytes.
  setCurrentStream(s);
  alloc.raw_delete(p1);

  void* p2 = alloc.raw_alloc(N, s);
  ASSERT_NE(p2, nullptr);

  auto stats_after_second = alloc.getDeviceStats();

  EXPECT_EQ(p2, p1);
  EXPECT_EQ(stats_after_second.reserved_bytes_all_current,
            stats_after_first.reserved_bytes_all_current);

  // Cleanup to avoid leaking cached allocations across tests.
  setCurrentStream(s);
  alloc.raw_delete(p2);
#endif
}

TEST(CudaAllocatorRoundingTest, RequestedBytesAccountingAndPeaks) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "Built without CUDA";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "CUDA device required";
  }

  DeviceIndex dev = 0;
  Allocator& alloc = Allocator::get(dev);

  // Best-effort: start from a clean cache so deltas are meaningful.
  alloc.emptyCache();

  Stream s = getStreamFromPool(false, dev);

  const std::size_t N1 = 1ull << 20; // 1 MiB
  const std::size_t N2 = 2ull << 20; // 2 MiB

  DeviceStats before = alloc.getDeviceStats();

  void* p1 = alloc.raw_alloc(N1, s);
  ASSERT_NE(p1, nullptr);
  auto after_first = alloc.getDeviceStats();
  EXPECT_EQ(after_first.requested_bytes_all_current,
            before.requested_bytes_all_current + N1);
  EXPECT_GE(after_first.max_requested_bytes_all,
            after_first.requested_bytes_all_current);

  void* p2 = alloc.raw_alloc(N2, s);
  ASSERT_NE(p2, nullptr);
  auto after_second = alloc.getDeviceStats();
  EXPECT_EQ(after_second.requested_bytes_all_current,
            before.requested_bytes_all_current + N1 + N2);
  EXPECT_GE(after_second.max_requested_bytes_all,
            after_second.requested_bytes_all_current);

  // Free first allocation and confirm requested-bytes drops by N1.
  setCurrentStream(s);
  alloc.raw_delete(p1);
  auto after_free_first = alloc.getDeviceStats();
  EXPECT_EQ(after_free_first.requested_bytes_all_current,
            before.requested_bytes_all_current + N2);
  EXPECT_GE(after_free_first.max_requested_bytes_all,
            after_second.max_requested_bytes_all);

  // Free second allocation and confirm we return to baseline.
  alloc.raw_delete(p2);
  auto after_free_second = alloc.getDeviceStats();
  EXPECT_EQ(after_free_second.requested_bytes_all_current,
            before.requested_bytes_all_current);
  EXPECT_GE(after_free_second.max_requested_bytes_all,
            after_second.max_requested_bytes_all);

  // Verify resetPeakStats clamps requested peaks to the current gauge.
  alloc.resetPeakStats();
  auto after_reset = alloc.getDeviceStats();
  EXPECT_EQ(after_reset.max_requested_bytes_all,
            after_reset.requested_bytes_all_current);
#endif
}
