// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

TEST(CudaAllocatorSkeletonTest, ZeroAndNullBehaviors) {
#if VBT_WITH_CUDA
  if (device_count() == 0) GTEST_SKIP() << "No CUDA device";
  auto& alloc = Allocator::get(0);
  void* p0 = alloc.raw_alloc(0);
  EXPECT_EQ(p0, nullptr);
  alloc.raw_delete(nullptr); // should not crash
  Stream s = getCurrentStream(0);
  void* p1 = alloc.raw_alloc(0, s);
  EXPECT_EQ(p1, nullptr);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaAllocatorSkeletonTest, BasicAllocDeleteEmptyCache) {
#if VBT_WITH_CUDA
  if (device_count() == 0) GTEST_SKIP() << "No CUDA device";
  auto& alloc = Allocator::get(0);
  void* p = alloc.raw_alloc(4096);
  ASSERT_NE(p, nullptr);
  alloc.raw_delete(p);
  EXPECT_EQ(alloc.debug_cached_segments(), 1u);
  alloc.emptyCache();
  EXPECT_EQ(alloc.debug_cached_segments(), 0u);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaAllocatorSkeletonTest, DeviceMismatchEarlyThrow) {
#if VBT_WITH_CUDA
  if (device_count() < 2) GTEST_SKIP() << "Need >=2 CUDA devices";
  auto& alloc0 = Allocator::get(0);
  Stream s1 = getStreamFromPool(false, /*device*/1);
  try {
    (void)alloc0.raw_alloc(1, s1);
    FAIL() << "Expected throw";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("allocator_device="), std::string::npos);
    EXPECT_NE(msg.find("stream_device="), std::string::npos);
    EXPECT_NE(msg.find("nbytes="), std::string::npos);
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaAllocatorSkeletonTest, SingletonThreadSafety) {
#if VBT_WITH_CUDA
  if (device_count() == 0) GTEST_SKIP() << "No CUDA device";
  const void* baseline = static_cast<const void*>(&Allocator::get(0));
  const int N = 8;
  std::vector<std::atomic<const void*>> addrs(N);
  std::vector<std::thread> ths;
  for (int i = 0; i < N; ++i) {
    ths.emplace_back([&, j=i]() {
      auto* ptr = &Allocator::get(0);
      addrs[j].store(static_cast<const void*>(ptr), std::memory_order_release);
    });
  }
  for (auto& t : ths) t.join();
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(baseline, addrs[i].load(std::memory_order_acquire));
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
