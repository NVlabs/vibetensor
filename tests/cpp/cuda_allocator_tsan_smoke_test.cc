// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"

using namespace vbt::cuda;

// native CUDA allocator. This test deliberately avoids CUDA Graphs and
// focuses on exercising the allocator's mutex/TSL paths under
// moderate multithreaded alloc/free pressure.
TEST(CudaAllocatorTsanSmokeTest, ConcurrentAllocFreeNoGraphs) {
#if VBT_WITH_CUDA
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& alloc = Allocator::get(0);

  constexpr int kThreads = 8;
  constexpr int kItersPerThread = 128;
  constexpr std::size_t kAllocSize = 1 << 12;  // 4 KiB, small but non-zero.

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::atomic<int> successful_allocs{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&]() {
      // Simple barrier so threads start their loops together, increasing the
      // chance of exercising contended hot paths under TSAN.
      ready.fetch_add(1, std::memory_order_acq_rel);
      while (!start.load(std::memory_order_acquire)) {
        // Busy-wait with yields; iteration count is small so this remains
        // cheap while avoiding tight spins on contended runners.
        std::this_thread::yield();
      }

      for (int i = 0; i < kItersPerThread; ++i) {
        void* p = alloc.raw_alloc(kAllocSize);
        // raw_alloc may return nullptr in OOM scenarios; treat that as a
        // soft failure for this smoke test and just break out.
        if (!p) {
          break;
        }
        successful_allocs.fetch_add(1, std::memory_order_relaxed);
        alloc.raw_delete(p);
      }
    });
  }

  // Wait until all threads have reached the barrier.
  while (ready.load(std::memory_order_acquire) < kThreads) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);

  for (auto& th : threads) {
    th.join();
  }

  // Ensure the workload actually performed some allocations so TSAN covers
  // allocator hot paths instead of trivially exiting on OOM.
  EXPECT_GT(successful_allocs.load(std::memory_order_relaxed), 0);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
