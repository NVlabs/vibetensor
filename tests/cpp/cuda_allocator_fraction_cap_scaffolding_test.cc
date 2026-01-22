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
// Hook that reports a fixed total memory and succeeds.
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

TEST(CudaAllocatorFractionCapScaffoldingTest,
     InitializationUsesHookAndCurrentLimitSemantics) {
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

    // With total = 24 GiB, fractions in (0,1) should map to floor(T * frac).
    A.setMemoryFraction(0.5);
    std::size_t limit_half = A.debug_current_limit_bytes_for_testing();
    EXPECT_EQ(limit_half, 12ull << 30);  // 0.5 * 24 GiB

    A.setMemoryFraction(0.25);
    std::size_t limit_quarter = A.debug_current_limit_bytes_for_testing();
    EXPECT_EQ(limit_quarter, 6ull << 30);  // 0.25 * 24 GiB

    // Fraction == 0.0 yields a hard zero cap when total bytes are known.
    A.setMemoryFraction(0.0);
    EXPECT_EQ(A.debug_current_limit_bytes_for_testing(), 0u);

    A.setMemoryFraction(1.0);
    EXPECT_EQ(A.debug_current_limit_bytes_for_testing(),
              std::numeric_limits<std::size_t>::max());
  }
#endif
}

#endif  // VBT_INTERNAL_TESTS
