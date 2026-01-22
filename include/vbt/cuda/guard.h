// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#if VBT_INTERNAL_TESTS
#include <atomic>
#endif
#include "vbt/cuda/stream.h"

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#  include <cuda_runtime_api.h>
#endif

namespace vbt { namespace cuda {

using DeviceIndex = int16_t;

#if VBT_INTERNAL_TESTS
namespace detail {
inline std::atomic<std::uint64_t> g_device_guard_ctor_calls{0};
}  // namespace detail

inline std::uint64_t device_guard_ctor_calls_for_tests() noexcept {
  return detail::g_device_guard_ctor_calls.load(std::memory_order_relaxed);
}

inline void reset_device_guard_ctor_calls_for_tests() noexcept {
  detail::g_device_guard_ctor_calls.store(0, std::memory_order_relaxed);
}
#endif  // VBT_INTERNAL_TESTS

// RAII guard that sets the current CUDA device on construction
// and restores the previous device on destruction.
class DeviceGuard final {
 public:
  explicit DeviceGuard(DeviceIndex d) noexcept {
#if VBT_INTERNAL_TESTS
    detail::g_device_guard_ctor_calls.fetch_add(1, std::memory_order_relaxed);
#endif
#if VBT_WITH_CUDA
    int cur = -1; cudaGetDevice(&cur); prev_ = cur;
    if (d >= 0 && cur != d) cudaSetDevice(static_cast<int>(d));
#else
    (void)d;
#endif
  }
  ~DeviceGuard() noexcept {
#if VBT_WITH_CUDA
    if (prev_ >= 0) cudaSetDevice(prev_);
#endif
  }

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
#if VBT_WITH_CUDA
  int prev_{-1};
#endif
};

// RAII guard that sets the current CUDA stream for a device on the current
// thread and restores the previous stream on destruction.
//
// Guard discipline (DC3):
// CUDAStreamGuard assumes the *current CUDA device* already equals the stream's
// device index at construction time. Callers must pin the device first:
//
//   DeviceGuard dg(s.device_index());
//   CUDAStreamGuard sg(s);
class CUDAStreamGuard final {
 public:
  CUDAStreamGuard() = delete;
  explicit CUDAStreamGuard(Stream s) noexcept;
  ~CUDAStreamGuard() noexcept;

  CUDAStreamGuard(const CUDAStreamGuard&) = delete;
  CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;

  void reset_stream(Stream s) noexcept;

  Stream      original_stream() const noexcept { return original_stream_; }
  Stream      current_stream()  const noexcept { return current_stream_; }
  DeviceIndex original_device() const noexcept { return original_device_; }

 private:
  DeviceIndex original_device_{-1};
  Stream      original_stream_{Stream::UNCHECKED, 0u, 0};
  Stream      current_stream_{Stream::UNCHECKED, 0u, 0};
};

}} // namespace vbt::cuda
