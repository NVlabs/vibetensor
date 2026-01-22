// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/event.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/guard.h"
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
#if VBT_WITH_CUDA
static inline void cudaCheck(cudaError_t st, const char* what) {
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    throw std::runtime_error(std::string(what) + ": " + (msg ? msg : ""));
  }
}
#endif
}

Event::Event(bool enable_timing) noexcept {
#if VBT_WITH_CUDA
  flags_ = enable_timing ? 0u : static_cast<unsigned int>(cudaEventDisableTiming);
#else
  (void)enable_timing;
#endif
}

Event::Event(Event&& other) noexcept {
  flags_ = other.flags_;
  is_created_ = other.is_created_;
  was_recorded_ = other.was_recorded_;
  device_index_ = other.device_index_;
  event_ = other.event_;
  other.is_created_ = false;
  other.was_recorded_ = false;
  other.device_index_ = -1;
  other.event_ = nullptr;
}

Event& Event::operator=(Event&& other) noexcept {
  if (this == &other) return *this;
#if VBT_WITH_CUDA
  if (is_created_ && event_) {
    try { DeviceGuard g(device_index_); cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event_)); } catch (...) {}
  }
#endif
  flags_ = other.flags_;
  is_created_ = other.is_created_;
  was_recorded_ = other.was_recorded_;
  device_index_ = other.device_index_;
  event_ = other.event_;
  other.is_created_ = false;
  other.was_recorded_ = false;
  other.device_index_ = -1;
  other.event_ = nullptr;
  return *this;
}

Event::~Event() noexcept {
#if VBT_WITH_CUDA
  if (is_created_ && event_) {
    try { DeviceGuard g(device_index_); (void)cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event_)); } catch (...) {}
  }
#endif
}

bool Event::query() const noexcept {
#if VBT_WITH_CUDA
  if (!is_created_) return true;
  DeviceGuard g(device_index_);
  cudaError_t st = cudaEventQuery(reinterpret_cast<cudaEvent_t>(event_));
  if (st == cudaSuccess) return true;
  if (st == cudaErrorNotReady) { (void)cudaGetLastError(); return false; }
  return false;
#else
  return true;
#endif
}

void Event::record(const Stream& stream) {
#if VBT_WITH_CUDA
  if (!is_created_) {
    device_index_ = stream.device_index();
    DeviceGuard g(device_index_);
    cudaEvent_t ev = nullptr;
    cudaCheck(cudaEventCreateWithFlags(&ev, flags_), "cudaEventCreateWithFlags");
    event_ = ev;
    is_created_ = true;
  } else {
    // Enforce same device
    if (device_index_ != stream.device_index()) {
      throw std::runtime_error("CUDA event recorded on a different device than it was created");
    }
  }
  DeviceGuard g(device_index_);
  cudaCheck(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event_), reinterpret_cast<cudaStream_t>(stream.handle())), "cudaEventRecord");
  was_recorded_ = true;
#else
  (void)stream;
#endif
}

void Event::wait(const Stream& stream) const {
#if VBT_WITH_CUDA
  if (!is_created_) return; // nothing to wait on
  // Note: cross-device waits are expected in multi-GPU features (Fabric/manual P2P).
  // We always scope the current device to the stream's device for the CUDA call.
  DeviceGuard g(stream.device_index());
  cudaCheck(
      cudaStreamWaitEvent(
          reinterpret_cast<cudaStream_t>(stream.handle()),
          reinterpret_cast<cudaEvent_t>(event_),
          0),
      "cudaStreamWaitEvent");
#else
  (void)stream;
#endif
}

void Event::synchronize() const {
#if VBT_WITH_CUDA
  if (!is_created_) return;
  DeviceGuard g(device_index_);
  cudaCheck(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event_)), "cudaEventSynchronize");
#endif
}

}} // namespace vbt::cuda
