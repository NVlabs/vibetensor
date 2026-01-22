// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

#ifdef VBT_INTERNAL_TESTS

TEST(CudaAllocatorFragmentationFuzzerTest, FragmentationFuzzerIdempotence) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  // Best-effort: start from a clean cache so baseline is simple.
  A.process_events(-1);
  A.emptyCache();

  Allocator::DebugTailGaugeSnapshot tails_before =
      A.debug_tail_gauge_snapshot_for_testing(/*force_scan=*/true);
  auto segs_before = snapshot_memory_segments(dev);

  Allocator::DebugFragmentationConfig cfg;
  cfg.seed = 42;
  cfg.steps = 256;
  cfg.max_block_size = 1ull << 20;

  A.debug_run_fragmentation_fuzzer_for_testing(cfg);
  Allocator::DebugTailGaugeSnapshot tails_after1 =
      A.debug_tail_gauge_snapshot_for_testing(/*force_scan=*/true);
  auto segs_after1 = snapshot_memory_segments(dev);

  A.debug_run_fragmentation_fuzzer_for_testing(cfg);
  Allocator::DebugTailGaugeSnapshot tails_after2 =
      A.debug_tail_gauge_snapshot_for_testing(/*force_scan=*/true);
  auto segs_after2 = snapshot_memory_segments(dev);

  // Tail gauges should return to baseline after each run.
  EXPECT_EQ(tails_after1.stats_blocks, tails_before.stats_blocks);
  EXPECT_EQ(tails_after1.stats_bytes,  tails_before.stats_bytes);
  EXPECT_EQ(tails_after2.stats_blocks, tails_before.stats_blocks);
  EXPECT_EQ(tails_after2.stats_bytes,  tails_before.stats_bytes);

  EXPECT_EQ(tails_after1.recomputed_blocks, tails_before.recomputed_blocks);
  EXPECT_EQ(tails_after1.recomputed_bytes,  tails_before.recomputed_bytes);
  EXPECT_EQ(tails_after2.recomputed_blocks, tails_before.recomputed_blocks);
  EXPECT_EQ(tails_after2.recomputed_bytes,  tails_before.recomputed_bytes);

  // Segment snapshots should also match baseline.
  EXPECT_EQ(segs_after1.size(), segs_before.size());
  EXPECT_EQ(segs_after2.size(), segs_before.size());
  for (std::size_t i = 0; i < segs_before.size(); ++i) {
    const auto& s0 = segs_before[i];
    const auto& s1 = segs_after1[i];
    const auto& s2 = segs_after2[i];
    EXPECT_EQ(s1.device, s0.device);
    EXPECT_EQ(s1.pool_id, s0.pool_id);
    EXPECT_EQ(s1.bytes_reserved, s0.bytes_reserved);
    EXPECT_EQ(s1.bytes_active,  s0.bytes_active);
    EXPECT_EQ(s1.blocks,        s0.blocks);

    EXPECT_EQ(s2.device, s0.device);
    EXPECT_EQ(s2.pool_id, s0.pool_id);
    EXPECT_EQ(s2.bytes_reserved, s0.bytes_reserved);
    EXPECT_EQ(s2.bytes_active,  s0.bytes_active);
    EXPECT_EQ(s2.blocks,        s0.blocks);
  }

  // Final cleanup for isolation.
  A.emptyCache();
#endif
}

#endif  // VBT_INTERNAL_TESTS
