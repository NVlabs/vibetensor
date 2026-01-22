// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

#ifdef VBT_INTERNAL_TESTS

TEST(CudaAllocatorGcBasicTest,
     FractionCapGcFreesIdleGlobalSegmentsAndUpdatesStats) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  // Bring allocator into a quiescent state before the test to reduce
  // interference from prior allocations.
  A.process_events(-1);
  A.emptyCache();
  A.resetAccumulatedStats();

  // Simple allocate + free to ensure at least one idle global segment exists.
  void* p = nullptr;
  try {
    p = A.raw_alloc(1 << 20);  // 1 MiB request (rounded by allocator policy)
  } catch (...) {
    p = nullptr;
  }
  if (!p) {
    GTEST_SKIP() << "Allocation failed; skipping GC basic test";
  }

  A.raw_delete(p);
  A.process_events(-1);

  DeviceStats before = A.getDeviceStats();

  // Choose a GC budget that is at least the current reserved bytes; GC is
  // allowed to reclaim less but must never report more than the target.
  std::size_t gc_target = static_cast<std::size_t>(
      before.reserved_bytes_all_current + (1u << 10));  // +1 KiB slack

  std::size_t reclaimed =
      A.debug_run_gc_pass_for_testing(gc_target, GcReason::FractionCap);

  DeviceStats after = A.getDeviceStats();

  // If GC found no candidates, it may legitimately be a no-op.
  if (reclaimed == 0) {
    EXPECT_EQ(after.gc_passes, before.gc_passes);
    EXPECT_EQ(after.gc_reclaimed_bytes, before.gc_reclaimed_bytes);
    return;
  }

  // Budget contract: planned reclaimed bytes never exceed target.
  EXPECT_LE(reclaimed, gc_target);

  // Stats must advance on a freeing GC pass.
  EXPECT_GT(after.gc_passes, before.gc_passes);

  std::uint64_t freed_bytes =
      after.gc_reclaimed_bytes - before.gc_reclaimed_bytes;
  EXPECT_GE(freed_bytes, reclaimed);

  // Reserved bytes should not increase after GC.
  EXPECT_LE(after.reserved_bytes_all_current,
            before.reserved_bytes_all_current);
#endif
}

#endif  // VBT_INTERNAL_TESTS
