// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <limits>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

#ifdef VBT_INTERNAL_TESTS

TEST(CudaAllocatorCapExceededTest, CapExceededAndProspectiveReservedEdgeCases) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  const std::size_t SZ_MAX = std::numeric_limits<std::size_t>::max();

  // C1: Cap disabled when limit == SIZE_MAX.
  {
    const std::size_t rounded = 1ull << 20;
    const std::size_t reserved = 0;
    const std::size_t limit = SZ_MAX;
    EXPECT_FALSE(A.debug_cap_exceeded_for_testing(rounded, reserved, limit));
  }

  // C2: Hard zero cap: any positive rounded should hit.
  {
    const std::size_t rounded = 1ull << 20;
    const std::size_t reserved = 0;
    const std::size_t limit = 0;
    EXPECT_TRUE(A.debug_cap_exceeded_for_testing(rounded, reserved, limit));
  }

  // C3: No growth at exact limit: rounded == 0, reserved == limit.
  {
    const std::size_t limit = 100ull << 20;
    const std::size_t rounded = 0;
    const std::size_t reserved = limit;
    EXPECT_FALSE(A.debug_cap_exceeded_for_testing(rounded, reserved, limit));
  }

  // C4: Minimal breach: reserved == limit, rounded == 1.
  {
    const std::size_t limit = 1024;
    const std::size_t rounded = 1;
    const std::size_t reserved = limit;
    EXPECT_TRUE(A.debug_cap_exceeded_for_testing(rounded, reserved, limit));
  }

  // C5: Overflow scenario: reserved + rounded would overflow size_t.
  {
    const std::size_t half_plus = SZ_MAX / 2 + 1;
    const std::size_t rounded = half_plus;
    const std::size_t reserved = half_plus;
    const std::size_t limit = SZ_MAX - 10;

    std::size_t prospective =
        A.debug_safe_prospective_reserved_for_testing(rounded, reserved);
    EXPECT_EQ(prospective, SZ_MAX);
    EXPECT_TRUE(A.debug_cap_exceeded_for_testing(rounded, reserved, limit));
  }

  // C6: Another overflow: rounded == SIZE_MAX, reserved == 1, tight limit.
  {
    const std::size_t rounded = SZ_MAX;
    const std::size_t reserved = 1;
    const std::size_t limit = SZ_MAX - 1;

    std::size_t prospective =
        A.debug_safe_prospective_reserved_for_testing(rounded, reserved);
    EXPECT_EQ(prospective, SZ_MAX);
    EXPECT_TRUE(A.debug_cap_exceeded_for_testing(rounded, reserved, limit));
  }

#endif  // VBT_WITH_CUDA
}

#endif  // VBT_INTERNAL_TESTS
