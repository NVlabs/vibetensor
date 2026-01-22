// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <limits>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

#ifdef VBT_INTERNAL_TESTS

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {

#if VBT_WITH_CUDA
// Hook that reports a fixed total memory and succeeds, used to seed
// Allocator::device_total_bytes_ deterministically for this test.
static cudaError_t MemGetInfoSuccess(size_t* freeB, size_t* totalB) {
  if (freeB) {
    *freeB = 10ull << 30;  // 10 GiB free (unused by allocator)
  }
  if (totalB) {
    *totalB = 24ull << 30;  // 24 GiB total
  }
  return cudaSuccess;
}
#endif  // VBT_WITH_CUDA

}  // namespace

TEST(CudaAllocatorFractionGateTest,
     FractionZeroCapBlocksGrowthAndUpdatesStats) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;

  // Install a success hook so the allocator constructor caches a known
  // total memory value for this device.
  {
    CudaMemGetInfoGuard guard(&MemGetInfoSuccess);
    Allocator& A = Allocator::get(dev);

    // Quiesce allocator state before the test.
    A.process_events(-1);
    A.emptyCache();
    A.resetAccumulatedStats();

    // Hard zero cap: no further native reserved-bytes growth allowed.
    A.setMemoryFraction(0.0);
    EXPECT_EQ(A.debug_current_limit_bytes_for_testing(), 0u);

    bool threw_fraction_cap_oom = false;
    try {
      (void)A.raw_alloc(1 << 20);  // 1 MiB; must not reach cudaMalloc.
    } catch (const std::runtime_error& e) {
      threw_fraction_cap_oom = true;
      std::string msg = e.what();
      EXPECT_NE(msg.find("per-process memory fraction cap"),
                std::string::npos);
    } catch (...) {
      threw_fraction_cap_oom = true;
    }
    EXPECT_TRUE(threw_fraction_cap_oom);

    DeviceStats st = A.getDeviceStats();
    EXPECT_EQ(st.fraction_cap_breaches, 1u);
    EXPECT_EQ(st.fraction_cap_misfires, 1u);
    EXPECT_EQ(st.gc_passes, 0u);
    EXPECT_EQ(st.gc_reclaimed_bytes, 0u);
    EXPECT_EQ(st.num_ooms, 0u);

    // resetAccumulatedStats should zero fraction_cap_* and gc_* counters.
    A.resetAccumulatedStats();
    DeviceStats st_reset = A.getDeviceStats();
    EXPECT_EQ(st_reset.fraction_cap_breaches, 0u);
    EXPECT_EQ(st_reset.fraction_cap_misfires, 0u);
    EXPECT_EQ(st_reset.gc_passes, 0u);
    EXPECT_EQ(st_reset.gc_reclaimed_bytes, 0u);
  }
#endif  // VBT_WITH_CUDA
}

#endif  // VBT_INTERNAL_TESTS
