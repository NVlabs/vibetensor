// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace vbt { namespace autograd {

// A closeable blocking queue.
// - push(T&&) returns false if the queue is closed.
// - pop_blocking(T*) blocks until an item is available, or the queue is closed
//   and empty; it returns false iff closed and empty.
// - close() is idempotent and wakes all waiters.
template <class T>
class ConcurrentTaskQueue {
 public:
  ConcurrentTaskQueue() = default;
  ConcurrentTaskQueue(const ConcurrentTaskQueue&) = delete;
  ConcurrentTaskQueue& operator=(const ConcurrentTaskQueue&) = delete;

  bool push(T&& item) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (closed_) {
        return false;
      }
      q_.push_back(std::move(item));
    }
    cv_.notify_one();
    return true;
  }

  bool pop_blocking(T* out) {
    std::unique_lock<std::mutex> lock(mu_);

    const auto pred = [&]() { return closed_ || !q_.empty(); };

#if VBT_INTERNAL_TESTS
    while (!pred()) {
      waiters_.fetch_add(1, std::memory_order_relaxed);
      cv_.wait(lock);
      waiters_.fetch_sub(1, std::memory_order_relaxed);
    }
#else
    cv_.wait(lock, pred);
#endif

    if (q_.empty()) {
      return false;  // closed and empty
    }

    // Move-construct before popping to keep the queue element intact if the
    // move constructor throws.
    T tmp = std::move(q_.front());
    q_.pop_front();
    lock.unlock();

    *out = std::move(tmp);
    return true;
  }

#if VBT_INTERNAL_TESTS
  std::size_t _test_waiter_count() const noexcept {
    return waiters_.load(std::memory_order_relaxed);
  }
#endif

  void close() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (closed_) {
        return;
      }
      closed_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> q_;
  bool closed_{false};
#if VBT_INTERNAL_TESTS
  std::atomic<std::size_t> waiters_{0};
#endif
};

}} // namespace vbt::autograd
