// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

#include "vbt/autograd/concurrent_task_queue.h"

TEST(AutogradConcurrentTaskQueue, PopReturnsFalseOnlyWhenClosedAndEmpty) {
  vbt::autograd::ConcurrentTaskQueue<int> q;
  q.close();

  int out = 0;
  EXPECT_FALSE(q.pop_blocking(&out));
}

TEST(AutogradConcurrentTaskQueue, CloseWakesBlockedPop) {
  vbt::autograd::ConcurrentTaskQueue<int> q;

  std::atomic<bool> ok{true};

  std::thread th([&]() {
    int out = 0;
    ok.store(q.pop_blocking(&out), std::memory_order_release);
  });

  // Ensure the consumer is actually blocked in pop_blocking() before closing.
  while (q._test_waiter_count() == 0) {
    std::this_thread::yield();
  }

  q.close();
  th.join();

  EXPECT_FALSE(ok.load(std::memory_order_acquire));

  int out = 0;
  EXPECT_FALSE(q.pop_blocking(&out));
  EXPECT_FALSE(q.push(123));
}

TEST(AutogradConcurrentTaskQueue, CloseStillAllowsDrainingExistingItems) {
  vbt::autograd::ConcurrentTaskQueue<int> q;

  EXPECT_TRUE(q.push(1));
  EXPECT_TRUE(q.push(2));
  q.close();

  int out = 0;
  EXPECT_TRUE(q.pop_blocking(&out));
  EXPECT_EQ(out, 1);
  EXPECT_TRUE(q.pop_blocking(&out));
  EXPECT_EQ(out, 2);

  EXPECT_FALSE(q.pop_blocking(&out));
}
