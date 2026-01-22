// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/event_pool.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

TEST(CudaEventPoolTest, PrewarmClampedToCap) {
#if VBT_WITH_CUDA
  if (device_count() == 0) GTEST_SKIP() << "No CUDA device";
  DeviceIndex dev = 0;
  EventPool pool(dev, EventPoolConfig{8, 16});
  EXPECT_EQ(pool.size(), 8u);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaEventPoolTest, LifoAndRaiiReturn) {
#if VBT_WITH_CUDA
  if (device_count() == 0) GTEST_SKIP() << "No CUDA device";
  EventPool pool(0, EventPoolConfig{4, 0});
  auto a = pool.get();
  auto b = pool.get();
  void* pa = a.raw();
  void* pb = b.raw();
  // Return b explicitly, a via dtor
  pool.put(std::move(b));
  { PooledEvent tmp = std::move(a); }
  // Next two gets should return a then b
  auto c1 = pool.get();
  EXPECT_EQ(c1.raw(), pa);
  auto c2 = pool.get();
  EXPECT_EQ(c2.raw(), pb);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaEventPoolTest, CrossPoolPutReturnsToOwner) {
#if VBT_WITH_CUDA
  if (device_count() == 0) GTEST_SKIP() << "No CUDA device";
  EventPool A(0, EventPoolConfig{2, 0});
  EventPool B(0, EventPoolConfig{2, 0});
  auto e = A.get();
  std::size_t a0 = A.size();
  std::size_t b0 = B.size();
  B.put(std::move(e));
  EXPECT_EQ(A.size(), a0 + 1);
  EXPECT_EQ(B.size(), b0);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaEventPoolTest, OverCapPutDestroysAndEmptyCache) {
#if VBT_WITH_CUDA
  if (device_count() == 0) GTEST_SKIP() << "No CUDA device";
  constexpr std::size_t cap = 8;
  EventPool pool(0, EventPoolConfig{cap, 0});
  std::vector<PooledEvent> evs;
  for (int i = 0; i < static_cast<int>(cap) + 5; ++i) {
    evs.emplace_back(pool.get());
  }
  // Return all
  for (auto& e : evs) pool.put(std::move(e));
  EXPECT_LE(pool.size(), cap);
  pool.empty_cache();
  EXPECT_EQ(pool.size(), 0u);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
