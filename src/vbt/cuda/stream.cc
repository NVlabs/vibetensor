// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/stream.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/guard.h"

#include <array>
#include <atomic>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

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


static inline DeviceIndex current_device_index() noexcept {
  int dev = 0; cudaGetDevice(&dev); return static_cast<DeviceIndex>(dev);
}

static inline void cudaCheck(cudaError_t st, const char* what) {
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    throw std::runtime_error(std::string(what) + ": " + (msg ? msg : ""));
  }
}
#endif

static thread_local std::vector<uintptr_t> tls_current_handles; // per-device current stream handle (0 == default)

static void ensure_tls_capacity() {
#if VBT_WITH_CUDA
  int n = vbt::cuda::device_count();
  if (n < 0) n = 0;
  if (static_cast<int>(tls_current_handles.size()) < n) {
    tls_current_handles.resize(n, 0);
  }
#else
  (void)tls_current_handles;
#endif
}

static inline uintptr_t stream_handle_from_id(uint64_t id) {
  return static_cast<uintptr_t>(id);
}

#if VBT_WITH_CUDA
static std::mutex pool_mu;
static constexpr int kStreamsPerPool = 8;
// Pools indexed by device and bucket (0=default, 1=high)
static std::vector<std::array<std::vector<uintptr_t>, 2>> pool_handles;
static std::vector<std::array<uint32_t, 2>> pool_indices;

static void ensure_pool_initialized(DeviceIndex dev, int bucket) {
  std::lock_guard<std::mutex> lock(pool_mu);
  if (static_cast<size_t>(dev) >= pool_handles.size()) {
    pool_handles.resize(static_cast<size_t>(dev) + 1);
    pool_indices.resize(static_cast<size_t>(dev) + 1);
    // initialize counters
    for (int b = 0; b < 2; ++b) {
      pool_indices[static_cast<size_t>(dev)][b] = 0u;
    }
  }
  // Do not pre-create streams here; creation is on-demand in getStreamFromPool
  auto& vec = pool_handles[static_cast<size_t>(dev)][bucket];
  (void)vec;
}

static uintptr_t create_stream_for_bucket(DeviceIndex dev, int bucket) {
  // Ensure context is initialized for this device before creating streams
  (void)cudaFree(0);
  auto pr = priority_range();
  int p = (bucket == 1 && !(pr.first == 0 && pr.second == 0)) ? pr.second : 0;
  cudaStream_t st = nullptr;
  cudaError_t rc;
  if (p != 0) {
    // Try priority + non-blocking, then priority + default
    rc = cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking, p);
    if (rc != cudaSuccess) {
      st = nullptr;
      rc = cudaStreamCreateWithPriority(&st, cudaStreamDefault, p);
    }
  } else {
    // Try non-blocking first
    rc = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
  }
  if (rc != cudaSuccess) {
    // Legacy default creator
    st = nullptr;
    rc = cudaStreamCreate(&st);
  }
  if (rc != cudaSuccess) {
    // Default flags as another attempt
    st = nullptr;
    rc = cudaStreamCreateWithFlags(&st, cudaStreamDefault);
  }
  if (rc != cudaSuccess || st == nullptr) {
    // Exhausted attempts: fail loudly to avoid returning a default stream
    cudaCheck(rc != cudaSuccess ? rc : cudaErrorUnknown, "cudaStreamCreate");
  }
  return reinterpret_cast<uintptr_t>(st);
}
#endif

} // anonymous

Stream::Stream(Unchecked, uint64_t packed_id, DeviceIndex device) noexcept
    : id_(packed_id), device_index_(device), handle_(stream_handle_from_id(packed_id)) {}

Stream::Stream(int priority, DeviceIndex device) {
  *this = getStreamFromPool(priority, device);
}

std::pair<int,int> priority_range() {
#if VBT_WITH_CUDA
  DeviceGuard g(current_device_index());
  int least = 0, greatest = 0;
  auto st = cudaDeviceGetStreamPriorityRange(&least, &greatest);
  if (st != cudaSuccess) {
    // Conservatively report no priority support
    return {0, 0};
  }
  // CUDA specifies least==0; greatest<=-1 when priorities supported
  if (least == 0 && greatest <= -1) {
    // Clamp to compile-time cap
    int clamped_greatest = - (kMaxCompileTimePriorities - 1);
    if (greatest < clamped_greatest) greatest = clamped_greatest;
    return {least, greatest};
  }
  return {0, 0};
#else
  return {0, 0};
#endif
}

static int clamp_priority(int p) {
  auto [least, greatest] = priority_range();
  if (least == 0 && greatest == 0) return 0;
  if (p > least) p = least;
  if (p < greatest) p = greatest;
  return p;
}

Stream getStreamFromPool(bool high_priority, DeviceIndex device) {
#if VBT_WITH_CUDA
  DeviceIndex dev = device;
  if (dev < 0) dev = current_device_index();
  DeviceGuard g(dev);
  auto pr = priority_range();
  int bucket = (high_priority && !(pr.first == 0 && pr.second == 0)) ? 1 : 0;
  ensure_pool_initialized(dev, bucket);
  uint32_t idx;
  {
    std::lock_guard<std::mutex> lock(pool_mu);
    idx = pool_indices[static_cast<size_t>(dev)][bucket]++;
  }
  uint32_t target = idx % kStreamsPerPool;
  {
    std::lock_guard<std::mutex> lock(pool_mu);
    auto& vec = pool_handles[static_cast<size_t>(dev)][bucket];
    while (vec.size() <= target) {
      vec.push_back(create_stream_for_bucket(dev, bucket));
    }
    auto handle = vec[target];
    return Stream(Stream::UNCHECKED, static_cast<uint64_t>(handle), dev);
  }
#else
  (void)high_priority; (void)device; return Stream(Stream::UNCHECKED, 0u, 0);
#endif
}

Stream getStreamFromPool(int priority, DeviceIndex device) {
#if VBT_WITH_CUDA
  DeviceIndex dev = device;
  if (dev < 0) dev = current_device_index();
  DeviceGuard g(dev);
  int p = clamp_priority(priority);
  int bucket = (p < 0) ? 1 : 0;
  ensure_pool_initialized(dev, bucket);
  uint32_t idx;
  {
    std::lock_guard<std::mutex> lock(pool_mu);
    idx = pool_indices[static_cast<size_t>(dev)][bucket]++;
  }
  uint32_t target = idx % kStreamsPerPool;
  {
    std::lock_guard<std::mutex> lock(pool_mu);
    auto& vec = pool_handles[static_cast<size_t>(dev)][bucket];
    while (vec.size() <= target) {
      vec.push_back(create_stream_for_bucket(dev, bucket));
    }
    auto handle = vec[target];
    return Stream(Stream::UNCHECKED, static_cast<uint64_t>(handle), dev);
  }
#else
  (void)priority; (void)device; return Stream(Stream::UNCHECKED, 0u, 0);
#endif
}

Stream getDefaultStream(DeviceIndex device) {
#if VBT_WITH_CUDA
  DeviceIndex dev = device;
  if (dev < 0) dev = current_device_index();
  return Stream(Stream::UNCHECKED, 0u, dev);
#else
  (void)device; return Stream(Stream::UNCHECKED, 0u, 0);
#endif
}

Stream getCurrentStream(DeviceIndex device) {
#if VBT_WITH_CUDA
  DeviceIndex dev = device;
  if (dev < 0) dev = current_device_index();
  ensure_tls_capacity();
  uintptr_t h = 0;
  if (dev >= 0 && dev < static_cast<DeviceIndex>(tls_current_handles.size())) {
    h = tls_current_handles[dev];
  }
  return Stream(Stream::UNCHECKED, static_cast<uint64_t>(h), dev);
#else
  (void)device; return Stream(Stream::UNCHECKED, 0u, 0);
#endif
}

void setCurrentStream(Stream s) {
#if VBT_WITH_CUDA
  ensure_tls_capacity();
  if (s.device_index_ >= 0 && s.device_index_ < static_cast<DeviceIndex>(tls_current_handles.size())) {
    tls_current_handles[s.device_index_] = s.handle_;
  }
#else
  (void)s;
#endif
}

bool Stream::query() const noexcept {
#if VBT_WITH_CUDA
  DeviceGuard g(device_index_);
  cudaError_t st = cudaStreamQuery(reinterpret_cast<cudaStream_t>(handle_));
  if (st == cudaSuccess) return true;
  if (st == cudaErrorNotReady) {
    // Clear only the expected sticky error
    (void)cudaGetLastError();
    return false;
  }
  // Unexpected errors: do not clear to preserve sticky error
  return false;
#else
  return true;
#endif
}

void Stream::synchronize() const {
#if VBT_WITH_CUDA
  DeviceGuard g(device_index_);
  cudaCheck(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(handle_)), "cudaStreamSynchronize");
#else
  // No-op
#endif
}

int Stream::priority() const {
#if VBT_WITH_CUDA
  DeviceGuard g(device_index_);
  int p = 0; cudaCheck(cudaStreamGetPriority(reinterpret_cast<cudaStream_t>(handle_), &p), "cudaStreamGetPriority");
  return p;
#else
  return 0;
#endif
}

std::string to_string(const Stream& s) {
  std::ostringstream oss;
  oss << "Stream(device=cuda:" << static_cast<int>(s.device_index())
      << ", id=" << s.id() << ", handle=0x" << std::hex << s.handle() << ")";
  return oss.str();
}

}} // namespace vbt::cuda
