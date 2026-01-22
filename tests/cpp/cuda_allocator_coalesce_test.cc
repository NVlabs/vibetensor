// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

TEST(CudaAllocatorCoalesceTest, GateOffIdentityBehavior) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (A.debug_split_enabled()) {
    GTEST_SKIP() << "Splitting enabled; gate-off behavior not exercised";
  }

  DeviceStats st0 = A.getDeviceStats();

  Allocator::DebugCoalesceScenario s{};
  s.has_left = true;
  s.has_right = true;
  s.left_size = 1024;
  s.self_size = 2048;
  s.right_size = 512;
  s.left_graph_pool_id = 0;
  s.self_graph_pool_id = 0;
  s.right_graph_pool_id = 0;
  s.left_owner_stream = 0;
  s.self_owner_stream = 0;
  s.right_owner_stream = 0;

  Allocator::DebugCoalesceResult r =
      A.debug_coalesce_neighbors_unlocked_for_testing(s);

  EXPECT_EQ(r.head_after.ptr, r.self_before.ptr);
  EXPECT_EQ(r.head_after.size, r.self_before.size);

  EXPECT_EQ(r.left_after.ptr, r.left_before.ptr);
  EXPECT_EQ(r.left_after.size, r.left_before.size);
  EXPECT_EQ(r.right_after.ptr, r.right_before.ptr);
  EXPECT_EQ(r.right_after.size, r.right_before.size);

  EXPECT_EQ(r.inactive_blocks_after, r.inactive_blocks_before);
  EXPECT_EQ(r.inactive_bytes_after, r.inactive_bytes_before);

  DeviceStats st1 = A.getDeviceStats();
  EXPECT_EQ(st1.inactive_split_blocks_all, st0.inactive_split_blocks_all);
  EXPECT_EQ(st1.inactive_split_bytes_all, st0.inactive_split_bytes_all);
#endif
}

TEST(CudaAllocatorCoalesceTest, CenterOnlyNoNeighbors) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (!A.debug_split_enabled()) {
    GTEST_SKIP() << "Block splitting disabled for this configuration";
  }

  DeviceStats st0 = A.getDeviceStats();

  Allocator::DebugCoalesceScenario s{};
  s.has_left = false;
  s.has_right = false;
  s.self_size = 4096;
  s.self_graph_pool_id = 0;
  s.self_owner_stream = 0;

  Allocator::DebugCoalesceResult r =
      A.debug_coalesce_neighbors_unlocked_for_testing(s);

  EXPECT_EQ(r.head_after.ptr, r.self_before.ptr);
  EXPECT_EQ(r.head_after.size, r.self_before.size);

  EXPECT_EQ(r.left_after.ptr, nullptr);
  EXPECT_EQ(r.right_after.ptr, nullptr);

  EXPECT_EQ(r.inactive_blocks_after, r.inactive_blocks_before);
  EXPECT_EQ(r.inactive_bytes_after, r.inactive_bytes_before);

  DeviceStats st1 = A.getDeviceStats();
  EXPECT_EQ(st1.inactive_split_blocks_all, st0.inactive_split_blocks_all);
  EXPECT_EQ(st1.inactive_split_bytes_all, st0.inactive_split_bytes_all);
#endif
}

TEST(CudaAllocatorCoalesceTest, LeftOnlyMergeBasic) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (!A.debug_split_enabled()) {
    GTEST_SKIP() << "Block splitting disabled for this configuration";
  }

  Allocator::DebugCoalesceScenario s{};
  s.has_left = true;
  s.has_right = false;
  s.left_size = 1024;
  s.self_size = 2048;
  s.left_graph_pool_id = 0;
  s.self_graph_pool_id = 0;
  s.left_owner_stream = 0;
  s.self_owner_stream = 0;

  Allocator::DebugCoalesceResult r =
      A.debug_coalesce_neighbors_unlocked_for_testing(s);

  ASSERT_NE(r.left_before.ptr, nullptr);
  ASSERT_NE(r.self_before.ptr, nullptr);

  EXPECT_EQ(r.head_after.ptr, r.left_before.ptr);
  EXPECT_EQ(r.head_after.size,
            r.left_before.size + r.self_before.size);

  // Center block should be gone after merge.
  EXPECT_EQ(r.self_after.ptr, nullptr);
#endif
}

TEST(CudaAllocatorCoalesceTest, RightOnlyMergeBasic) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (!A.debug_split_enabled()) {
    GTEST_SKIP() << "Block splitting disabled for this configuration";
  }

  Allocator::DebugCoalesceScenario s{};
  s.has_left = false;
  s.has_right = true;
  s.self_size = 2048;
  s.right_size = 1024;
  s.self_graph_pool_id = 0;
  s.right_graph_pool_id = 0;
  s.self_owner_stream = 0;
  s.right_owner_stream = 0;

  Allocator::DebugCoalesceResult r =
      A.debug_coalesce_neighbors_unlocked_for_testing(s);

  ASSERT_NE(r.self_before.ptr, nullptr);
  ASSERT_NE(r.right_before.ptr, nullptr);

  EXPECT_EQ(r.head_after.ptr, r.self_before.ptr);
  EXPECT_EQ(r.head_after.size,
            r.self_before.size + r.right_before.size);

  // Right block should be gone after merge.
  EXPECT_EQ(r.right_after.ptr, nullptr);
#endif
}

TEST(CudaAllocatorCoalesceTest, BothSidesMergeBasic) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (!A.debug_split_enabled()) {
    GTEST_SKIP() << "Block splitting disabled for this configuration";
  }

  Allocator::DebugCoalesceScenario s{};
  s.has_left = true;
  s.has_right = true;
  s.left_size = 512;
  s.self_size = 1024;
  s.right_size = 2048;
  s.left_graph_pool_id = 0;
  s.self_graph_pool_id = 0;
  s.right_graph_pool_id = 0;
  s.left_owner_stream = 0;
  s.self_owner_stream = 0;
  s.right_owner_stream = 0;

  Allocator::DebugCoalesceResult r =
      A.debug_coalesce_neighbors_unlocked_for_testing(s);

  ASSERT_NE(r.left_before.ptr, nullptr);
  ASSERT_NE(r.self_before.ptr, nullptr);
  ASSERT_NE(r.right_before.ptr, nullptr);

  EXPECT_EQ(r.head_after.ptr, r.left_before.ptr);
  EXPECT_EQ(r.head_after.size,
            r.left_before.size + r.self_before.size + r.right_before.size);

  EXPECT_EQ(r.self_after.ptr, nullptr);
  EXPECT_EQ(r.right_after.ptr, nullptr);
#endif
}

TEST(CudaAllocatorCoalesceTest, TailGaugeLeftTailMerged) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (!A.debug_split_enabled()) {
    GTEST_SKIP() << "Block splitting disabled for this configuration";
  }

  DeviceStats st0 = A.getDeviceStats();

  Allocator::DebugCoalesceScenario s{};
  s.has_left = true;
  s.has_right = false;
  s.left_size = 1024;
  s.self_size = 2048;
  s.left_graph_pool_id = 0;
  s.self_graph_pool_id = 0;
  s.left_owner_stream = 0;
  s.self_owner_stream = 0;
  s.left_is_split_tail = true;

  Allocator::DebugCoalesceResult r =
      A.debug_coalesce_neighbors_unlocked_for_testing(s);

  ASSERT_NE(r.left_before.ptr, nullptr);

  // Exactly one tail (the left) should have been removed.
  ASSERT_GE(r.inactive_blocks_before, r.inactive_blocks_after);
  EXPECT_EQ(r.inactive_blocks_before - r.inactive_blocks_after, 1u);
  ASSERT_GE(r.inactive_bytes_before, r.inactive_bytes_after);
  EXPECT_EQ(r.inactive_bytes_before - r.inactive_bytes_after,
            r.left_before.size);

  DeviceStats st1 = A.getDeviceStats();
  EXPECT_EQ(st1.inactive_split_blocks_all, st0.inactive_split_blocks_all);
  EXPECT_EQ(st1.inactive_split_bytes_all, st0.inactive_split_bytes_all);
#endif
}

TEST(CudaAllocatorCoalesceTest, TailGaugeCenterTailMergedWithRight) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (!A.debug_split_enabled()) {
    GTEST_SKIP() << "Block splitting disabled for this configuration";
  }

  DeviceStats st0 = A.getDeviceStats();

  Allocator::DebugCoalesceScenario s{};
  s.has_left = false;
  s.has_right = true;
  s.self_size = 2048;
  s.right_size = 1024;
  s.self_graph_pool_id = 0;
  s.right_graph_pool_id = 0;
  s.self_owner_stream = 0;
  s.right_owner_stream = 0;
  s.self_is_split_tail = true;

  Allocator::DebugCoalesceResult r =
      A.debug_coalesce_neighbors_unlocked_for_testing(s);

  ASSERT_NE(r.self_before.ptr, nullptr);

  ASSERT_GE(r.inactive_blocks_before, r.inactive_blocks_after);
  EXPECT_EQ(r.inactive_blocks_before - r.inactive_blocks_after, 1u);
  ASSERT_GE(r.inactive_bytes_before, r.inactive_bytes_after);
  EXPECT_EQ(r.inactive_bytes_before - r.inactive_bytes_after,
            r.self_before.size);

  DeviceStats st1 = A.getDeviceStats();
  EXPECT_EQ(st1.inactive_split_blocks_all, st0.inactive_split_blocks_all);
  EXPECT_EQ(st1.inactive_split_bytes_all, st0.inactive_split_bytes_all);
#endif
}

TEST(CudaAllocatorCoalesceTest, OversizeCenterIsNoOp) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (!A.debug_split_enabled()) {
    GTEST_SKIP() << "Block splitting disabled for this configuration";
  }

  // Configure an oversize threshold below self_size.
  Allocator::DebugCoalesceScenario s{};
  s.has_left = true;
  s.has_right = true;
  s.left_size = 1024;
  s.self_size = 4096;
  s.right_size = 1024;
  s.left_graph_pool_id = 0;
  s.self_graph_pool_id = 0;
  s.right_graph_pool_id = 0;
  s.left_owner_stream = 0;
  s.self_owner_stream = 0;
  s.right_owner_stream = 0;
  s.oversize_threshold_bytes = 2048; // make center oversize, neighbors non-oversize

  Allocator::DebugCoalesceResult r =
      A.debug_coalesce_neighbors_unlocked_for_testing(s);

  EXPECT_EQ(r.head_after.ptr, r.self_before.ptr);
  EXPECT_EQ(r.head_after.size, r.self_before.size);

  EXPECT_EQ(r.left_after.ptr, r.left_before.ptr);
  EXPECT_EQ(r.right_after.ptr, r.right_before.ptr);
}
#endif

TEST(CudaAllocatorCoalesceTest, OversizeNeighborIsIneligible) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (!A.debug_split_enabled()) {
    GTEST_SKIP() << "Block splitting disabled for this configuration";
  }

  Allocator::DebugCoalesceScenario s{};
  s.has_left = false;
  s.has_right = true;
  s.self_size = 1024;   // non-oversize
  s.right_size = 4096;  // oversize
  s.self_graph_pool_id = 0;
  s.right_graph_pool_id = 0;
  s.self_owner_stream = 0;
  s.right_owner_stream = 0;
  s.oversize_threshold_bytes = 2048;

  Allocator::DebugCoalesceResult r =
      A.debug_coalesce_neighbors_unlocked_for_testing(s);

  EXPECT_EQ(r.head_after.ptr, r.self_before.ptr);
  EXPECT_EQ(r.head_after.size, r.self_before.size);

  EXPECT_EQ(r.right_after.ptr, r.right_before.ptr);
  EXPECT_EQ(r.right_after.size, r.right_before.size);
#endif
}

TEST(CudaAllocatorCoalesceTest, AllocatedNeighborDoesNotCoalesce) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (!A.debug_split_enabled()) {
    GTEST_SKIP() << "Block splitting disabled for this configuration";
  }

  Allocator::DebugCoalesceScenario s{};
  s.has_left = true;
  s.has_right = false;
  s.left_size = 1024;
  s.self_size = 2048;
  s.left_graph_pool_id = 0;
  s.self_graph_pool_id = 0;
  s.left_owner_stream = 0;
  s.self_owner_stream = 0;
  s.left_allocated = true;  // ineligible neighbor

  Allocator::DebugCoalesceResult r =
      A.debug_coalesce_neighbors_unlocked_for_testing(s);

  EXPECT_EQ(r.head_after.ptr, r.self_before.ptr);
  EXPECT_EQ(r.head_after.size, r.self_before.size);

  EXPECT_EQ(r.left_after.ptr, r.left_before.ptr);
  EXPECT_EQ(r.left_after.size, r.left_before.size);
#endif
}

#ifndef NDEBUG
TEST(CudaAllocatorCoalesceTest, DeathOnInvalidPreconditions) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (!A.debug_split_enabled()) {
    GTEST_SKIP() << "Block splitting disabled for this configuration";
  }

  // Center still allocated.
  EXPECT_DEATH(
      A.debug_trigger_coalesce_neighbors_assert_for_testing(
          /*mark_center_allocated=*/true,
          /*make_center_eventful=*/false,
          /*insert_center_into_free_indices=*/false,
          /*mismatch_neighbor_graph_pool=*/false),
      "");

  // Center has pending events.
  EXPECT_DEATH(
      A.debug_trigger_coalesce_neighbors_assert_for_testing(
          /*mark_center_allocated=*/false,
          /*make_center_eventful=*/true,
          /*insert_center_into_free_indices=*/false,
          /*mismatch_neighbor_graph_pool=*/false),
      "");

  // Center still in free indices.
  EXPECT_DEATH(
      A.debug_trigger_coalesce_neighbors_assert_for_testing(
          /*mark_center_allocated=*/false,
          /*make_center_eventful=*/false,
          /*insert_center_into_free_indices=*/true,
          /*mismatch_neighbor_graph_pool=*/false),
      "");

  // Neighbor has mismatched graph_pool_id.
  EXPECT_DEATH(
      A.debug_trigger_coalesce_neighbors_assert_for_testing(
          /*mark_center_allocated=*/false,
          /*make_center_eventful=*/false,
          /*insert_center_into_free_indices=*/false,
          /*mismatch_neighbor_graph_pool=*/true),
      "");
#endif
}
#endif
