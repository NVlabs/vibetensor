// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

#ifdef VBT_INTERNAL_TESTS

TEST(CudaAllocatorStatsInvariantsTest, TailGaugeMatchesScanInQuiescentAllocator) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  // Best-effort: start from a clean, quiescent allocator.
  A.process_events(-1);
  A.emptyCache();

  Allocator::DebugTailGaugeSnapshot snap =
      A.debug_tail_gauge_snapshot_for_testing(/*force_scan=*/true);

  EXPECT_EQ(snap.stats_blocks, snap.recomputed_blocks);
  EXPECT_EQ(snap.stats_bytes,  snap.recomputed_bytes);
  EXPECT_TRUE(snap.consistent);
#endif
}

TEST(CudaAllocatorStatsInvariantsTest, InactiveSplitGaugesNotResetByResetAccumulatedStats) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  DeviceStats before = A.getDeviceStats();
  A.resetAccumulatedStats();
  DeviceStats after = A.getDeviceStats();

  EXPECT_EQ(after.inactive_split_blocks_all, before.inactive_split_blocks_all);
  EXPECT_EQ(after.inactive_split_bytes_all,  before.inactive_split_bytes_all);

  EXPECT_EQ(before.fraction_cap_breaches, 0u);
  EXPECT_EQ(before.fraction_cap_misfires, 0u);
  EXPECT_EQ(before.gc_passes, 0u);
  EXPECT_EQ(before.gc_reclaimed_bytes, 0u);

  EXPECT_EQ(after.fraction_cap_breaches, 0u);
  EXPECT_EQ(after.fraction_cap_misfires, 0u);
  EXPECT_EQ(after.gc_passes, 0u);
  EXPECT_EQ(after.gc_reclaimed_bytes, 0u);
#endif
}

TEST(CudaAllocatorStatsInvariantsTest, ToleranceFillsMonotonicAndResettableViaDebugHelpers) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  // Ensure counters start from a clean state.
  A.resetAccumulatedStats();
  DeviceStats st0 = A.getDeviceStats();

  // Use debug_evaluate_candidate_for_testing with tolerance overrides to bump counters.
  auto r1 = A.debug_evaluate_candidate_for_testing(
      /*block_size=*/1024,
      /*is_split_tail=*/false,
      /*req=*/512,
      /*N_override=*/0,
      /*T_override=*/1024);
  (void)r1;

  DeviceStats st1 = A.getDeviceStats();
  EXPECT_GE(st1.tolerance_fills_count, st0.tolerance_fills_count);
  EXPECT_GE(st1.tolerance_fills_bytes, st0.tolerance_fills_bytes);

  auto r2 = A.debug_evaluate_candidate_for_testing(
      /*block_size=*/1024,
      /*is_split_tail=*/false,
      /*req=*/512,
      /*N_override=*/0,
      /*T_override=*/1024);
  (void)r2;

  DeviceStats st2 = A.getDeviceStats();
  EXPECT_GE(st2.tolerance_fills_count, st1.tolerance_fills_count);
  EXPECT_GE(st2.tolerance_fills_bytes, st1.tolerance_fills_bytes);

  // resetAccumulatedStats should zero tolerance_fills_* while leaving gauges intact.
  A.resetAccumulatedStats();
  DeviceStats st3 = A.getDeviceStats();
  EXPECT_EQ(st3.tolerance_fills_count, 0u);
  EXPECT_EQ(st3.tolerance_fills_bytes, 0u);
  EXPECT_EQ(st3.inactive_split_blocks_all, st2.inactive_split_blocks_all);
  EXPECT_EQ(st3.inactive_split_bytes_all,  st2.inactive_split_bytes_all);

  EXPECT_EQ(st0.fraction_cap_breaches, 0u);
  EXPECT_EQ(st0.fraction_cap_misfires, 0u);
  EXPECT_EQ(st0.gc_passes, 0u);
  EXPECT_EQ(st0.gc_reclaimed_bytes, 0u);

  EXPECT_EQ(st3.fraction_cap_breaches, 0u);
  EXPECT_EQ(st3.fraction_cap_misfires, 0u);
  EXPECT_EQ(st3.gc_passes, 0u);
  EXPECT_EQ(st3.gc_reclaimed_bytes, 0u);
#endif
}

TEST(CudaAllocatorStatsInvariantsTest, StatsAndSnapshotsConsistencyHelper) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  // Best-effort: small allocation workload to produce non-trivial stats.
  void* p = nullptr;
  try {
    p = A.raw_alloc(1 << 20);
  } catch (...) {
    p = nullptr;
  }

  Allocator::DebugStatsSnapshotConsistency rep =
      A.debug_stats_snapshot_consistency_for_testing(dev);

  EXPECT_TRUE(rep.stats_vs_segments_ok);
  EXPECT_TRUE(rep.stats_vs_pools_ok);

  if (p) {
    A.raw_delete(p);
  }

  // Final cleanup for isolation.
  A.emptyCache();
#endif
}

#endif  // VBT_INTERNAL_TESTS
