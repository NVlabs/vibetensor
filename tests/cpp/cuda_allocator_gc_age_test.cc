// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

#ifdef VBT_INTERNAL_TESTS

TEST(CudaAllocatorGcAgeTest, GcAgeZeroOnlyAfterBasicWorkload) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  // Simple alloc/free workload to populate allocator state.
  void* p0 = nullptr;
  void* p1 = nullptr;
  try {
    p0 = A.raw_alloc(1 << 20);
  } catch (...) {
    p0 = nullptr;
  }
  if (p0) {
    A.raw_delete(p0);
  }

  try {
    p1 = A.raw_alloc(1 << 18);
  } catch (...) {
    p1 = nullptr;
  }
  if (p1) {
    A.raw_delete(p1);
  }

  // Drain any deferred frees and cached segments.
  A.process_events(-1);
  A.emptyCache();

  auto ptrs = A.debug_tracked_block_ptrs();
  for (void* ptr : ptrs) {
    EXPECT_EQ(A.debug_block_gc_age(ptr), 0u);
  }

  int stack_val = 42;
  EXPECT_EQ(A.debug_block_gc_age(&stack_val), 0u);
  EXPECT_EQ(A.debug_block_gc_age(nullptr), 0u);
#endif
}

#endif  // VBT_INTERNAL_TESTS
