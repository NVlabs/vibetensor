// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/device.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace vbt::cuda;

TEST(CudaAllocatorCrossStreamTest, CrossStreamFenceOwnerReuse) {
#if VBT_WITH_CUDA
  if (device_count() == 0) GTEST_SKIP() << "No CUDA device";
  DeviceIndex dev = 0;
  auto& alloc = Allocator::get(dev);

  // Two streams on same device
  Stream S1 = getStreamFromPool(false, dev);
  Stream S2 = getStreamFromPool(false, dev);

  const std::size_t N = 1 << 16; // 64KiB
  void* p = alloc.raw_alloc(N, S1);
  ASSERT_NE(p, nullptr);

  // Record cross-stream use on S2
  alloc.record_stream(p, S2);

  // Free on S1 (owner becomes S1). Ensure current stream is S1 for free path.
  setCurrentStream(S1);
  alloc.raw_delete(p);

  // Immediately allocate on S2: must NOT reuse p because of cross-stream fence
  void* q = alloc.raw_alloc(N, S2);
  ASSERT_NE(q, nullptr);
  EXPECT_NE(q, p);

  // Make S2 event ready and process events; then allocation on S1 should reuse p
  S2.synchronize();
  alloc.process_events();

  void* r = alloc.raw_alloc(N, S1);
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(r, p);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaAllocatorCrossStreamTest, CrossStreamFallbackRespectsEnvToggle) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "Built without CUDA";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& alloc = Allocator::get(dev);

  const bool fallback_enabled =
      alloc.debug_cfg_enable_cross_stream_fallback_for_testing();

  // Start from a clean cache so reserved-bytes assertions are meaningful.
  alloc.emptyCache();

  Stream S1 = getStreamFromPool(false, dev);
  Stream S2 = getStreamFromPool(false, dev);

  const std::size_t N = 1 << 16; // 64KiB

  void* p = alloc.raw_alloc(N, S1);
  ASSERT_NE(p, nullptr);

  // Record cross-stream use and free on S1.
  alloc.record_stream(p, S2);
  setCurrentStream(S1);
  alloc.raw_delete(p);

  // Ensure the cross-stream fence has completed and block is in free lists.
  S2.synchronize();
  alloc.process_events();

  auto before = alloc.getDeviceStats();

  void* q = alloc.raw_alloc(N, S2);
  ASSERT_NE(q, nullptr);

  auto after = alloc.getDeviceStats();

  if (fallback_enabled) {
    // With cross-stream fallback enabled, S2 should be able to reuse the
    // freed block without increasing reserved bytes.
    EXPECT_EQ(q, p);
    EXPECT_EQ(after.reserved_bytes_all_current,
              before.reserved_bytes_all_current);
  } else {
    // Without cross-stream fallback, allocation on S2 must not reuse the
    // block owned by S1 and will trigger a new cudaMalloc, increasing
    // reserved bytes.
    EXPECT_NE(q, p);
    EXPECT_GT(after.reserved_bytes_all_current,
              before.reserved_bytes_all_current);
  }

  setCurrentStream(S2);
  alloc.raw_delete(q);
#endif
}
