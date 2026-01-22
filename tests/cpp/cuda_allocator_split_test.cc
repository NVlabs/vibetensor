// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

TEST(CudaAllocatorSplitTest, SmallPoolSplitStructureAndGauges) {
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

  // Baseline gauges before helper.
  DeviceStats st0 = A.getDeviceStats();

  std::size_t block_size = 4096; // 4 KiB
  std::size_t take_size  = 2048; // split in half
  StreamId owner_sid = 0;
  std::uint64_t pool_id = 0;

  Allocator::DebugSplitResult r =
      A.debug_split_block_unlocked_for_testing(block_size,
                                               take_size,
                                               owner_sid,
                                               pool_id);

  // Structural checks.
  EXPECT_EQ(r.front_before.size, block_size);
  EXPECT_EQ(r.front_after.size, take_size);
  ASSERT_NE(r.front_after.ptr, nullptr);
  ASSERT_NE(r.tail_after.ptr, nullptr);

  auto* front_ptr = static_cast<char*>(r.front_after.ptr);
  auto* tail_ptr  = static_cast<char*>(r.tail_after.ptr);
  EXPECT_EQ(tail_ptr, front_ptr + r.front_after.size);

  EXPECT_TRUE(r.front_after.segment_head);
  EXPECT_FALSE(r.tail_after.segment_head);

  EXPECT_FALSE(r.front_after.is_split_tail);
  EXPECT_TRUE(r.tail_after.is_split_tail);

  EXPECT_EQ(r.tail_after.prev_ptr, r.front_after.ptr);
  EXPECT_EQ(r.front_after.next_ptr, r.tail_after.ptr);

  // Free-list membership: only tail should be in free indices.
  EXPECT_FALSE(r.front_in_per_stream_free);
  EXPECT_FALSE(r.front_in_cross_stream_free);
  EXPECT_TRUE(r.tail_in_per_stream_free);
  EXPECT_TRUE(r.tail_in_cross_stream_free);

  // Gauge deltas reflect a single newly created tail.
  std::uint64_t blocks_delta =
      r.inactive_blocks_after - r.inactive_blocks_before;
  std::uint64_t bytes_delta =
      r.inactive_bytes_after - r.inactive_bytes_before;

  EXPECT_EQ(blocks_delta, 1u);
  EXPECT_EQ(bytes_delta, r.tail_after.size);

  // Helper must restore gauges to baseline after cleanup.
  DeviceStats st1 = A.getDeviceStats();
  EXPECT_EQ(st1.inactive_split_blocks_all, st0.inactive_split_blocks_all);
  EXPECT_EQ(st1.inactive_split_bytes_all, st0.inactive_split_bytes_all);
#endif
}

TEST(CudaAllocatorSplitTest, ExactSizeNoSplitIsNoOp) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  DeviceStats st0 = A.getDeviceStats();

  std::size_t block_size = 4096;
  std::size_t take_size  = block_size; // exact-size allocation
  StreamId owner_sid = 0;
  std::uint64_t pool_id = 0;

  Allocator::DebugSplitResult r =
      A.debug_split_block_unlocked_for_testing(block_size,
                                               take_size,
                                               owner_sid,
                                               pool_id);

  EXPECT_EQ(r.front_before.size, block_size);
  EXPECT_EQ(r.front_after.size, block_size);
  EXPECT_EQ(r.tail_after.ptr, nullptr);

  // No free-list entries are created for the front, and there is no tail.
  EXPECT_FALSE(r.front_in_per_stream_free);
  EXPECT_FALSE(r.front_in_cross_stream_free);
  EXPECT_FALSE(r.tail_in_per_stream_free);
  EXPECT_FALSE(r.tail_in_cross_stream_free);

  // Gauges must not change.
  EXPECT_EQ(r.inactive_blocks_after, r.inactive_blocks_before);
  EXPECT_EQ(r.inactive_bytes_after, r.inactive_bytes_before);

  DeviceStats st1 = A.getDeviceStats();
  EXPECT_EQ(st1.inactive_split_blocks_all, st0.inactive_split_blocks_all);
  EXPECT_EQ(st1.inactive_split_bytes_all, st0.inactive_split_bytes_all);
#endif
}

TEST(CudaAllocatorSplitTest, GateOffIdentityBehavior) {
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

  std::size_t block_size = 4096;
  std::size_t take_size  = 2048;
  StreamId owner_sid = 0;
  std::uint64_t pool_id = 0;

  Allocator::DebugSplitResult r =
      A.debug_split_block_unlocked_for_testing(block_size,
                                               take_size,
                                               owner_sid,
                                               pool_id);

  // With split_enabled() == false, helper must behave as identity.
  EXPECT_EQ(r.front_before.size, block_size);
  EXPECT_EQ(r.front_after.size, block_size);
  EXPECT_EQ(r.tail_after.ptr, nullptr);

  EXPECT_FALSE(r.front_in_per_stream_free);
  EXPECT_FALSE(r.front_in_cross_stream_free);
  EXPECT_FALSE(r.tail_in_per_stream_free);
  EXPECT_FALSE(r.tail_in_cross_stream_free);

  EXPECT_EQ(r.inactive_blocks_after, r.inactive_blocks_before);
  EXPECT_EQ(r.inactive_bytes_after, r.inactive_bytes_before);

  DeviceStats st1 = A.getDeviceStats();
  EXPECT_EQ(st1.inactive_split_blocks_all, st0.inactive_split_blocks_all);
  EXPECT_EQ(st1.inactive_split_bytes_all, st0.inactive_split_bytes_all);
#endif
}

TEST(CudaAllocatorSplitTest, GraphPoolIdPreservedOnSplit) {
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

  std::size_t block_size = 4096;
  std::size_t take_size  = 1024;
  StreamId owner_sid = 0;
  std::uint64_t pool_id = 42; // arbitrary non-zero graph pool id

  Allocator::DebugSplitResult r =
      A.debug_split_block_unlocked_for_testing(block_size,
                                               take_size,
                                               owner_sid,
                                               pool_id);

  ASSERT_NE(r.tail_after.ptr, nullptr);

  EXPECT_EQ(r.front_after.graph_pool_id, pool_id);
  EXPECT_EQ(r.tail_after.graph_pool_id, pool_id);
#endif
}

#ifndef NDEBUG
TEST(CudaAllocatorSplitTest, DeathOnInvalidPreconditions) {
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

  // take_size == 0 is invalid.
  EXPECT_DEATH(
      A.debug_trigger_split_block_assert_for_testing(
          /*block_size=*/4096,
          /*take_size=*/0,
          /*mark_as_split_tail=*/false,
          /*make_oversize=*/false,
          /*graph_pool_id=*/0u),
      "");

  // take_size > block_size is invalid.
  EXPECT_DEATH(
      A.debug_trigger_split_block_assert_for_testing(
          /*block_size=*/4096,
          /*take_size=*/8192,
          /*mark_as_split_tail=*/false,
          /*make_oversize=*/false,
          /*graph_pool_id=*/0u),
      "");

  // Split-tail input is invalid.
  EXPECT_DEATH(
      A.debug_trigger_split_block_assert_for_testing(
          /*block_size=*/4096,
          /*take_size=*/2048,
          /*mark_as_split_tail=*/true,
          /*make_oversize=*/false,
          /*graph_pool_id=*/0u),
      "");

  // Oversize block cannot be split when oversize is active.
  EXPECT_DEATH(
      A.debug_trigger_split_block_assert_for_testing(
          /*block_size=*/4096,
          /*take_size=*/2048,
          /*mark_as_split_tail=*/false,
          /*make_oversize=*/true,
          /*graph_pool_id=*/0u),
      "");
#endif
}
#endif
