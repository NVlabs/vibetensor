// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <thread>
#include <unordered_set>
#include <vector>

#include "vbt/autograd/owner_mailbox.h"

TEST(AutogradOwnerMailbox, ManyProducersSingleConsumer) {
  vbt::autograd::OwnerMailbox<int> mb;

  constexpr int kProducers = 8;
  constexpr int kPerProducer = 2000;
  const std::size_t total = static_cast<std::size_t>(kProducers) * kPerProducer;

  std::vector<std::thread> threads;
  threads.reserve(kProducers);

  for (int p = 0; p < kProducers; ++p) {
    threads.emplace_back([&, p]() {
      for (int i = 0; i < kPerProducer; ++i) {
        mb.push(p * 10000000 + i);
      }
    });
  }

  std::unordered_set<int> seen;
  seen.reserve(total);

  for (std::size_t i = 0; i < total; ++i) {
    const int v = mb.pop_blocking();
    seen.insert(v);
  }

  for (auto& th : threads) {
    th.join();
  }

  EXPECT_EQ(seen.size(), total);
}
