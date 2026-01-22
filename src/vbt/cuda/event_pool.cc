// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/event_pool.h"
#include "vbt/cuda/guard.h"
//#include "vbt/cuda/stream.h"
#include <stdexcept>

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#  include <cuda_runtime_api.h>
#endif

namespace vbt { namespace cuda {

namespace {

}

PooledEvent::~PooledEvent() noexcept {
#if VBT_WITH_CUDA
  if (ev_ == nullptr) return;
  EventPool* owner = owner_;
  void* ev = ev_;
  DeviceIndex dv = dev_;
  clear_();
  if (owner) {
    owner->return_event_(ev);
  } else {
    DeviceGuard g(dv);
    cudaError_t st = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ev));
    if (st != cudaSuccess) (void)cudaGetLastError();
  }
#else
  // No CUDA: nothing to do
#endif
}

EventPool::EventPool(DeviceIndex dev, EventPoolConfig cfg)
  : dev_(dev), cfg_(cfg) {
  if (cfg_.prewarm > cfg_.cap) cfg_.prewarm = cfg_.cap;
  prewarm_ctor_();
}

PooledEvent EventPool::get() {
#if VBT_WITH_CUDA
  {
    std::lock_guard<std::mutex> lg(pool_.mu);
    if (!pool_.idle.empty()) {
      void* ev = pool_.idle.back();
      pool_.idle.pop_back();
      return PooledEvent(this, dev_, ev);
    }
  }
  // Create a new event outside of the mutex
  DeviceGuard g(dev_);
  cudaEvent_t ev = nullptr;
  cudaError_t st = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  if (st != cudaSuccess) { (void)cudaGetLastError(); return PooledEvent(); }
  return PooledEvent(this, dev_, ev);
#else
  return PooledEvent();
#endif
}

void EventPool::put(PooledEvent&& e) noexcept {
#if VBT_WITH_CUDA
  EventPool* owner = e.owner_;
  void* ev = e.ev_;
  DeviceIndex dv = e.dev_;
  e.clear_();
  if (!ev) return;
  if (owner && owner != this) { owner->return_event_(ev); return; }
  // Fast path: push into idle if under cap
  bool destroy = false;
  {
    std::lock_guard<std::mutex> lg(pool_.mu);
    if (pool_.idle.size() < cfg_.cap) {
      pool_.idle.emplace_back(ev);
      return;
    } else {
      destroy = true;
    }
  }
  if (destroy) {
    DeviceGuard g(dv);
    cudaError_t st = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ev));
    if (st != cudaSuccess) (void)cudaGetLastError();
  }
#else
  (void)e;
#endif
}

void EventPool::destroy(PooledEvent&& e) noexcept {
#if VBT_WITH_CUDA
  void* ev = e.ev_;
  DeviceIndex dv = e.dev_;
  e.clear_();
  if (!ev) return;
  DeviceGuard g(dv);
  cudaError_t st = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ev));
  if (st != cudaSuccess) (void)cudaGetLastError();
#else
  (void)e;
#endif
}

void EventPool::empty_cache() noexcept {
#if VBT_WITH_CUDA
  std::vector<void*> to_destroy;
  {
    std::lock_guard<std::mutex> lg(pool_.mu);
    to_destroy.swap(pool_.idle);
  }
  if (!to_destroy.empty()) {
    DeviceGuard g(dev_);
    for (void* ev : to_destroy) {
      cudaError_t st = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ev));
      if (st != cudaSuccess) (void)cudaGetLastError();
    }
  }
#endif
}

std::size_t EventPool::size() const noexcept {
  std::lock_guard<std::mutex> lg(pool_.mu);
  return pool_.idle.size();
}

void EventPool::return_event_(void* ev) noexcept {
#if VBT_WITH_CUDA
  if (!ev) return;
  bool destroy = false;
  {
    std::lock_guard<std::mutex> lg(pool_.mu);
    if (pool_.idle.size() < cfg_.cap) {
      pool_.idle.emplace_back(ev);
      return;
    } else {
      destroy = true;
    }
  }
  if (destroy) {
    DeviceGuard g(dev_);
    cudaError_t st = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ev));
    if (st != cudaSuccess) (void)cudaGetLastError();
  }
#else
  (void)ev;
#endif
}

void EventPool::prewarm_ctor_() {
#if VBT_WITH_CUDA
  if (cfg_.prewarm == 0) return;
  std::vector<void*> created;
  created.reserve(cfg_.prewarm);
  {
    DeviceGuard g(dev_);
    for (std::size_t i = 0; i < cfg_.prewarm; ++i) {
      cudaEvent_t ev = nullptr;
      cudaError_t st = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
      if (st != cudaSuccess) { (void)cudaGetLastError(); break; }
      created.push_back(ev);
    }
  }
  // Store under lock, collect extras
  std::vector<void*> extras;
  {
    std::lock_guard<std::mutex> lg(pool_.mu);
    for (void* ev : created) {
      if (pool_.idle.size() < cfg_.cap) pool_.idle.emplace_back(ev);
      else extras.emplace_back(ev);
    }
  }
  if (!extras.empty()) {
    DeviceGuard g(dev_);
    for (void* ev : extras) {
      cudaError_t st = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ev));
      if (st != cudaSuccess) (void)cudaGetLastError();
    }
  }
#endif
}

}} // namespace vbt::cuda
