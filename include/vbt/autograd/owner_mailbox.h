// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <utility>

namespace vbt { namespace autograd {

// Many-producer, single-consumer mailbox.
// Unbounded in-memory queue. `push` never blocks (beyond the internal mutex).
//
// Note: this mailbox intentionally has no close/cancel semantics. The intended
// usage is that the owner thread knows exactly how many
// completion messages to expect (tracked via an "outstanding" counter), so
// `pop_blocking()` cannot deadlock as long as workers always push exactly one
// message per scheduled task.
//
// Ordering across producers is unspecified.
template <class T>
class OwnerMailbox {
 public:
  OwnerMailbox() = default;
  OwnerMailbox(const OwnerMailbox&) = delete;
  OwnerMailbox& operator=(const OwnerMailbox&) = delete;

  void push(T&& msg) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      q_.push_back(std::move(msg));
    }
    cv_.notify_one();
  }

  T pop_blocking() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&]() { return !q_.empty(); });

    T out = std::move(q_.front());
    q_.pop_front();
    return out;
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> q_;
};

}} // namespace vbt::autograd
