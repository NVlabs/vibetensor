// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace vbt { namespace cuda {

using DeviceIndex = int16_t;

struct EventPoolConfig {
  std::size_t cap{1024};
  std::size_t prewarm{0};
};

class EventPool; // fwd

// Move-only RAII wrapper for a pooled cudaEvent_t (opaque void* here)
class PooledEvent final {
public:
  PooledEvent() noexcept = default;
  PooledEvent(PooledEvent&& other) noexcept { swap(other); }
  PooledEvent& operator=(PooledEvent&& other) noexcept {
    if (this != &other) swap(other);
    return *this;
  }
  PooledEvent(const PooledEvent&) = delete;
  PooledEvent& operator=(const PooledEvent&) = delete;
  ~PooledEvent() noexcept;

  bool valid() const noexcept { return ev_ != nullptr; }
  DeviceIndex device() const noexcept { return dev_; }
  void* raw() const noexcept { return ev_; }

private:
  friend class EventPool;
  explicit PooledEvent(EventPool* owner, DeviceIndex dev, void* ev) noexcept
      : dev_(dev), ev_(ev), owner_(owner) {}
  void clear_() noexcept { dev_ = -1; ev_ = nullptr; owner_ = nullptr; }
  void swap(PooledEvent& other) noexcept {
    std::swap(dev_, other.dev_);
    std::swap(ev_, other.ev_);
    std::swap(owner_, other.owner_);
  }

  DeviceIndex dev_{-1};
  void*       ev_{nullptr};  // cudaEvent_t
  EventPool*  owner_{nullptr};
};

class EventPool final {
public:
  explicit EventPool(DeviceIndex dev, EventPoolConfig cfg = {});

  // LIFO get; may create a new event if idle cache is empty. Creation uses cudaEventDisableTiming.
  PooledEvent get();

  // Return an event to the pool. If pool is over cap, the event is destroyed.
  void put(PooledEvent&& e) noexcept;

  // Destroy an event regardless of cap (used on rollback of recorded events).
  void destroy(PooledEvent&& e) noexcept;

  // Destroy all idle (cached) events. Does not affect outstanding PooledEvent instances.
  void empty_cache() noexcept;

  std::size_t size() const noexcept;
  EventPoolConfig config() const noexcept { return cfg_; }
  DeviceIndex device() const noexcept { return dev_; }

private:
  friend class PooledEvent;
  void return_event_(void* ev) noexcept; // called from PooledEvent dtor

  DeviceIndex dev_{};
  EventPoolConfig cfg_{};
  struct Inner {
    mutable std::mutex     mu;
    std::vector<void*>     idle; // LIFO (back is top)
  } pool_{};

  void prewarm_ctor_();
};

}} // namespace vbt::cuda
