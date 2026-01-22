// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/allocator_async.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/guard.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace vbt::cuda;

TEST(CudaAllocatorFractionAsyncIntegrationTest, EnvSeedsAsyncAndSetterUpdatesLimit) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  // CTest configures this binary with backend=cudamallocasync and an initial
  // per_process_memory_fraction (e.g. 0.5). We verify that the async backend
  // sees the same seed and that runtime setter calls update both the stored
  // fraction and the computed limit.
  DeviceIndex dev = 0;

  Allocator& A = Allocator::get(dev);
  AsyncBackend& B = AsyncBackend::get(dev);

  // Trigger lazy_init_ so limit_bytes_ is populated based on the seeded
  // per_process_fraction_.
  {
    Stream s = getStreamFromPool(false, dev);
    void* p = nullptr;
    try {
      p = A.raw_alloc(1 << 20, s);
    } catch (...) {
      // Best-effort: if allocation fails, we still want fraction semantics.
    }
    if (p) {
      setCurrentStream(s);
      A.raw_delete(p);
    }
  }

  double seeded_frac = B.debug_fraction();
  EXPECT_GT(seeded_frac, 0.0);
  EXPECT_LE(seeded_frac, 1.0);

  // limit_bytes_ should be approximately seeded_frac * totalB.
  std::size_t freeB = 0, totalB = 0;
  {
    DeviceGuard g(dev);
    cudaMemGetInfo(&freeB, &totalB);
  }
  std::size_t limit = B.debug_limit_bytes();
  EXPECT_GT(limit, 0u);
  double expected = seeded_frac * static_cast<double>(totalB);
  // Allow 1% relative tolerance to account for rounding.
  double rel_err = std::fabs(static_cast<double>(limit) - expected) /
                   std::max(expected, 1.0);
  EXPECT_LT(rel_err, 0.01) << "limit_bytes=" << limit << " totalB=" << totalB;

  // Now change the fraction at runtime via the unified setter and confirm
  // both fraction and limit update.
  A.setMemoryFraction(0.25);
  EXPECT_NEAR(B.debug_fraction(), 0.25, 1e-9);

  std::size_t limit_after = B.debug_limit_bytes();
  double expected_after = 0.25 * static_cast<double>(totalB);
  double rel_err_after = std::fabs(static_cast<double>(limit_after) - expected_after) /
                         std::max(expected_after, 1.0);
  EXPECT_LT(rel_err_after, 0.01)
      << "limit_bytes_after=" << limit_after << " totalB=" << totalB;
#endif
}

TEST(CudaAllocatorFractionAsyncIntegrationTest, DeviceStatsNewCountersZero) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  // Best-effort: perform a small allocation to exercise the async backend,
  // but do not depend on it succeeding.
  {
    Stream s = getStreamFromPool(false, dev);
    void* p = nullptr;
    try {
      p = A.raw_alloc(1 << 20, s);
    } catch (...) {
      p = nullptr;
    }
    if (p) {
      setCurrentStream(s);
      A.raw_delete(p);
    }
  }

  DeviceStats st = A.getDeviceStats();

  // Async backend does not track requested-bytes or future diagnostics yet.
  EXPECT_EQ(st.requested_bytes_all_current, 0u);
  EXPECT_EQ(st.max_requested_bytes_all, 0u);
  EXPECT_EQ(st.inactive_split_blocks_all, 0u);
  EXPECT_EQ(st.inactive_split_bytes_all, 0u);
  EXPECT_EQ(st.fraction_cap_breaches, 0u);
  EXPECT_EQ(st.fraction_cap_misfires, 0u);
  EXPECT_EQ(st.gc_passes, 0u);
  EXPECT_EQ(st.gc_reclaimed_bytes, 0u);

#ifdef VBT_INTERNAL_TESTS
  // Even when explicitly invoked, the GC helper must be a no-op for async
  // backends and must not change stats.
  std::size_t before_gc_passes = st.gc_passes;
  std::size_t before_gc_bytes  = st.gc_reclaimed_bytes;

  std::size_t reclaimed =
      A.debug_run_gc_pass_for_testing(1 << 20, GcReason::FractionCap);
  EXPECT_EQ(reclaimed, 0u);

  DeviceStats after = A.getDeviceStats();
  EXPECT_EQ(after.gc_passes, before_gc_passes);
  EXPECT_EQ(after.gc_reclaimed_bytes, before_gc_bytes);
#endif  // VBT_INTERNAL_TESTS
#endif
}
