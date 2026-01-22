// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vbt/cuda/stream.h"
#include "vbt/cuda/allocator.h"

#if !defined(VBT_WITH_CUDA)
#  define VBT_WITH_CUDA 0
#endif
#if VBT_WITH_CUDA
#  include <cuda_runtime_api.h>
#endif

namespace vbt { namespace cuda {

#if VBT_WITH_CUDA
class AsyncBackend {
 public:
  struct PtrUsage {
    std::unordered_set<StreamId> recorded_streams;
    StreamId creation_stream{0};
    std::size_t size{0};
  };

  static AsyncBackend& get(DeviceIndex dev);

  // Configure mempool knobs (release threshold, reuse flags).
  // per_process_fraction is intentionally ignored; all real fraction changes
  // must go through set_memory_fraction() via Allocator::setMemoryFraction.
  void configure(double per_process_fraction,
                 std::size_t release_threshold_bytes,
                 bool reuse_follow_event_deps,
                 bool reuse_allow_opportunistic,
                 bool reuse_allow_internal_deps);

  // Update the per-device fraction and recompute the per-process limit.
  // Intended to be called only from Allocator::setMemoryFraction and tests.
  void set_memory_fraction(double fraction);

  void* raw_alloc(std::size_t nbytes);
  void* raw_alloc(std::size_t nbytes, Stream s);
  void  raw_delete(void* ptr) noexcept;
  void  record_stream(void* ptr, Stream s) noexcept;
  void  process_events(int /*max_pops*/ = -1) noexcept {} // no-op
  void  emptyCache() noexcept;

  DeviceStats getDeviceStats() const noexcept;
  void resetPeakStats() noexcept;
  void resetAccumulatedStats() noexcept {}
  bool owns(const void* p) const noexcept;
  void* getBaseAllocation(void* p, std::size_t* size) const noexcept;

  cudaError_t memcpyAsync(void* dst, int dstDev, const void* src, int srcDev,
                          std::size_t bytes, Stream s, bool p2p_enabled) noexcept;
  cudaError_t enablePeerAccess(int dev, int peer) noexcept;

#ifdef VBT_INTERNAL_TESTS
  // Debug helpers for tests to inspect current fraction and limit.
  double debug_fraction() const noexcept;
  std::size_t debug_limit_bytes() const noexcept;
#endif
 private:
  explicit AsyncBackend(DeviceIndex dev);
  void lazy_init_();
  void mallocAsync_(void** devPtr, DeviceIndex device, std::size_t size, cudaStream_t stream);
  void free_impl_(void* ptr);
  bool any_stream_capturing_(const PtrUsage& u, StreamId owner) const noexcept;

  DeviceIndex dev_;
  mutable std::mutex mu_;
  bool inited_{false};
  cudaStream_t unifying_stream_{nullptr};
  // accounting
  std::size_t used_bytes_{0};
  std::size_t limit_bytes_{std::numeric_limits<std::size_t>::max()};
  // ptr map
  std::unordered_map<void*, PtrUsage> ptrs_;
  std::vector<void*> deferred_;
  // config
  double per_process_fraction_{1.0};
  std::size_t release_threshold_bytes_{static_cast<std::size_t>(-1)};
  bool reuse_follow_event_deps_{true};
  bool reuse_allow_opportunistic_{true};
  bool reuse_allow_internal_deps_{true};
};
#endif

}} // namespace vbt::cuda
