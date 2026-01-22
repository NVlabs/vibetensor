// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/device.h"

#if VBT_INTERNAL_TESTS
#include <atomic>
#endif

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#  include <cuda.h>
#  include <absl/base/config.h>
#  ifdef ABSL_HAVE_LEAK_SANITIZER
extern "C" void __lsan_disable();
extern "C" void __lsan_enable();

struct LsanDisableGuard {
  LsanDisableGuard() { __lsan_disable(); }
  ~LsanDisableGuard() { __lsan_enable(); }
};
#  endif
#endif

#if VBT_INTERNAL_TESTS
namespace {
std::atomic<std::uint64_t> g_device_count_calls{0};
}  // namespace
#endif

namespace vbt {
namespace cuda {

int device_count() noexcept {
#if VBT_INTERNAL_TESTS
  g_device_count_calls.fetch_add(1, std::memory_order_relaxed);
#endif
#if VBT_WITH_CUDA
  // Prefer CUDA Driver API to avoid libcudart one-time init leaks under ASAN.
  // The CUDA driver may keep internal allocations alive for the life of the
  // process; guard this call with LeakSanitizer disable/enable when available.
#  ifdef ABSL_HAVE_LEAK_SANITIZER
  LsanDisableGuard lsan_guard;
#  endif

  CUresult r = cuInit(0);
  if (r != CUDA_SUCCESS) {
    return 0;
  }
  int count = 0;
  r = cuDeviceGetCount(&count);
  if (r != CUDA_SUCCESS) return 0;
  return count < 0 ? 0 : count;
#else
  return 0;
#endif
}

#if VBT_INTERNAL_TESTS
std::uint64_t device_count_calls_for_tests() noexcept {
  return g_device_count_calls.load(std::memory_order_relaxed);
}

void reset_device_count_calls_for_tests() noexcept {
  g_device_count_calls.store(0, std::memory_order_relaxed);
}
#endif

} // namespace cuda
} // namespace vbt
