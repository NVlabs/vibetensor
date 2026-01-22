// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

#ifdef VBT_INTERNAL_TESTS

TEST(CudaAllocatorEmptyCacheTest,
     EmptyCacheFreesIdleGlobalSegmentsWithoutTouchingGcStats) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  // Bring allocator into a quiescent state to limit interference
  // from earlier tests.
  A.process_events(-1);
  A.emptyCache();
  A.resetAccumulatedStats();

  void* p = nullptr;
  try {
    p = A.raw_alloc(1 << 20);  // 1 MiB request (rounded by policy)
  } catch (...) {
    p = nullptr;
  }
  if (!p) {
    GTEST_SKIP() << "Allocation failed; skipping emptyCache test";
  }

  // Return the block to the allocator's free lists so it becomes a
  // candidate for emptyCache().
  A.raw_delete(p);
  A.process_events(-1);

  DeviceStats before = A.getDeviceStats();

  // Sanity: the allocator should still know about this pointer prior
  // to emptyCache(); afterwards it should be fully released.
  EXPECT_TRUE(A.owns(p));

  A.emptyCache();

  DeviceStats after = A.getDeviceStats();

  // The pointer should no longer be tracked by the allocator.
  EXPECT_FALSE(A.owns(p));

  // Reserved bytes should strictly decrease when at least one idle
  // global segment is reclaimed.
  EXPECT_LT(after.reserved_bytes_all_current,
            before.reserved_bytes_all_current);

  // emptyCache() frees whole segments and should bump num_device_free
  // by at least one, even if additional candidates exist.
  EXPECT_LT(before.num_device_free, after.num_device_free);

  // A freeing emptyCache() call must not touch GC stats; those are
  // reserved for run_gc_pass_if_eligible().
  EXPECT_EQ(after.gc_passes, before.gc_passes);
  EXPECT_EQ(after.gc_reclaimed_bytes, before.gc_reclaimed_bytes);
#endif
}

#endif  // VBT_INTERNAL_TESTS
