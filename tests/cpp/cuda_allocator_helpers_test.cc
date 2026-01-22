// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

TEST(CudaAllocatorHelpersTest, ShouldSplitHeuristicsBasic) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  // Small-pool: remainder below threshold does not split; at or above does split.
  std::size_t block_small = 4096; // 4 KiB

  // rem = 256 < 512 -> no split.
  EXPECT_FALSE(A.debug_should_split_for_testing(block_small,
                                                /*is_split_tail=*/false,
                                                block_small - 256));

  // rem = 512 -> split.
  EXPECT_TRUE(A.debug_should_split_for_testing(block_small,
                                               /*is_split_tail=*/false,
                                               block_small - 512));

  // Tails are never split.
  EXPECT_FALSE(A.debug_should_split_for_testing(block_small,
                                                /*is_split_tail=*/true,
                                                block_small - 1024));

  // Large-pool: use req > RoundPolicy::kSmallSize (1 MiB).
  std::size_t req_large = (RoundPolicy::kSmallSize << 1); // 2 MiB
  std::size_t block_large1 = req_large + RoundPolicy::kSmallSize;     // rem == 1 MiB
  std::size_t block_large2 = block_large1 + 1;                        // rem == 1 MiB + 1

  EXPECT_FALSE(A.debug_should_split_for_testing(block_large1,
                                                /*is_split_tail=*/false,
                                                req_large));
  EXPECT_TRUE(A.debug_should_split_for_testing(block_large2,
                                               /*is_split_tail=*/false,
                                               req_large));
#endif
}

TEST(CudaAllocatorHelpersTest, EvaluateCandidateDecisionsAndTolerance) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  using DDecision = Allocator::DebugCandidateDecision;

  auto eval = [&](std::size_t block_size,
                  bool        is_split_tail,
                  std::size_t req,
                  std::size_t N,
                  std::size_t T) -> Allocator::DebugCandidateResult {
    A.resetAccumulatedStats();
    return A.debug_evaluate_candidate_for_testing(block_size,
                                                  is_split_tail,
                                                  req,
                                                  N,
                                                  T);
  };

  // Too small: block_size < req => Reject, no tolerance stats.
  {
    auto r = eval(/*block_size=*/1024, /*is_split_tail=*/false,
                  /*req=*/2048, /*N=*/0, /*T=*/0);
    EXPECT_EQ(r.decision, DDecision::Reject);
    EXPECT_FALSE(r.counted_as_tolerance_fill);
    EXPECT_EQ(r.tolerance_waste_bytes, 0u);
  }

  // Exact match: TakeWhole, no tolerance.
  {
    auto r = eval(/*block_size=*/4096, /*is_split_tail=*/false,
                  /*req=*/4096, /*N=*/0, /*T=*/0);
    EXPECT_EQ(r.decision, DDecision::TakeWhole);
    EXPECT_FALSE(r.counted_as_tolerance_fill);
    EXPECT_EQ(r.tolerance_waste_bytes, 0u);
  }

  // General tolerance: non-oversize block, T > 0, 0 < rem <= T.
  {
    std::size_t block_size = 4096;
    std::size_t req = 3072;       // rem = 1024
    std::size_t T = 1024;
    auto r = eval(block_size, /*is_split_tail=*/false, req,
                  /*N=*/0, T);
    EXPECT_EQ(r.decision, DDecision::TakeWhole);
    EXPECT_TRUE(r.counted_as_tolerance_fill);
    EXPECT_EQ(r.tolerance_waste_bytes, block_size - req);
  }

  // Split decision: small-pool block where should_split_unlocked would return true.
  {
    std::size_t block_size = 4096;
    std::size_t req = 2048;       // rem = 2048 >= 512 -> split
    auto r = eval(block_size, /*is_split_tail=*/false, req,
                  /*N=*/0, /*T=*/0);
    EXPECT_EQ(r.decision, DDecision::Split);
    EXPECT_FALSE(r.counted_as_tolerance_fill);
    EXPECT_EQ(r.tolerance_waste_bytes, 0u);
  }

  // Fallback TakeWhole: small rem below tolerance and split disabled.
  {
    std::size_t block_size = 1024;
    std::size_t req = 900;        // rem = 124 (< 512), no tolerance cap
    auto r = eval(block_size, /*is_split_tail=*/false, req,
                  /*N=*/0, /*T=*/0);
    EXPECT_EQ(r.decision, DDecision::TakeWhole);
    EXPECT_FALSE(r.counted_as_tolerance_fill);
    EXPECT_EQ(r.tolerance_waste_bytes, 0u);
  }
#endif
}
