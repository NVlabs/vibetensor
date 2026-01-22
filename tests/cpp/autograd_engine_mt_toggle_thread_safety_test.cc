// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "vbt/autograd/engine_toggles.h"

namespace {

struct MtToggleRestore {
  bool prev;
  MtToggleRestore() : prev(vbt::autograd::is_multithreading_enabled()) {}
  ~MtToggleRestore() { vbt::autograd::set_multithreading_enabled(prev); }
};

} // namespace

// is safe under concurrent reads/writes.
TEST(AutogradToggleThreadSafetyTest, MultithreadingToggleConcurrentReadWrite) {
  MtToggleRestore restore;

  constexpr int kReaderThreads = 8;
  constexpr int kIters = 20000;

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::atomic<int> reads_true{0};
  std::atomic<int> reads_false{0};

  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(kReaderThreads) + 1);

  // One writer thread flips the toggle repeatedly.
  threads.emplace_back([&]() {
    ready.fetch_add(1, std::memory_order_acq_rel);
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }

    for (int i = 0; i < kIters; ++i) {
      vbt::autograd::set_multithreading_enabled((i & 1) == 0);
    }
  });

  // Many reader threads repeatedly observe the toggle.
  for (int t = 0; t < kReaderThreads; ++t) {
    threads.emplace_back([&]() {
      ready.fetch_add(1, std::memory_order_acq_rel);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      for (int i = 0; i < kIters; ++i) {
        const bool v = vbt::autograd::is_multithreading_enabled();
        if (v) {
          reads_true.fetch_add(1, std::memory_order_relaxed);
        } else {
          reads_false.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  while (ready.load(std::memory_order_acquire) < (kReaderThreads + 1)) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);

  for (auto& th : threads) {
    th.join();
  }

  // Ensure we actually executed the loops; exact ratios are intentionally
  // unspecified due to scheduling and relaxed atomics.
  EXPECT_GT(reads_true.load(std::memory_order_relaxed) +
                reads_false.load(std::memory_order_relaxed),
            0);

  // Toggle state is restored to its prior value by MtToggleRestore.
}
