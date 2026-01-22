// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/storage.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#  include "vbt/cuda/allocator.h"
#endif

namespace vbt { namespace cuda {

namespace {
#if VBT_WITH_CUDA
static std::atomic<std::size_t> g_record_stream_calls{0};

using StreamId = std::uint64_t;
struct ProducerMetadata {
  DeviceIndex device{-1};
  std::unordered_set<StreamId> producer_streams; // stream.handle() values
};

static std::mutex g_producer_mu;
static std::unordered_map<void*, ProducerMetadata> g_producer_meta;

static inline void register_producer_metadata_(void* ptr, DeviceIndex dev) noexcept {
  if (!ptr) return;
  std::lock_guard<std::mutex> lg(g_producer_mu);
  g_producer_meta.emplace(ptr, ProducerMetadata{dev, {}});
}

static inline void unregister_producer_metadata_(void* ptr) noexcept {
  if (!ptr) return;
  std::lock_guard<std::mutex> lg(g_producer_mu);
  g_producer_meta.erase(ptr);
}

#else
static std::size_t g_record_stream_calls = 0;
#endif
} // anonymous

vbt::core::StoragePtr new_cuda_storage(std::size_t nbytes, int device_index) {
#if VBT_WITH_CUDA
  if (nbytes == 0) {
    return vbt::core::make_intrusive<vbt::core::Storage>(vbt::core::DataPtr(nullptr, nullptr), 0);
  }
  // Route through caching allocator. Resolve the concrete allocator/device now and capture it in the deleter.
  Allocator& A = Allocator::get(static_cast<DeviceIndex>(device_index));
  void* ptr = A.raw_alloc(nbytes);
  DeviceIndex resolved = A.device();

  // Register producer metadata for this allocation. New allocations start with
  // a known-empty producer set (zero-producer case).
  register_producer_metadata_(ptr, resolved);

  vbt::core::DataPtr dp(ptr, [resolved](void* p) noexcept {
    if (!p) return;
    unregister_producer_metadata_(p);
    Allocator::get(resolved).raw_delete(p);
  });
  return vbt::core::make_intrusive<vbt::core::Storage>(std::move(dp), nbytes);
#else
  (void)nbytes; (void)device_index;
  return vbt::core::make_intrusive<vbt::core::Storage>(vbt::core::DataPtr(nullptr, nullptr), 0);
#endif
}

void record_stream(const vbt::core::StoragePtr& storage, Stream s) noexcept {
#if VBT_WITH_CUDA
  g_record_stream_calls.fetch_add(1, std::memory_order_relaxed);
  if (!storage) return;
  void* ptr = storage->data();
  if (!ptr) return;

  // Update producer metadata when the Storage is one of our CUDA allocations.
  {
    std::lock_guard<std::mutex> lg(g_producer_mu);
    auto it = g_producer_meta.find(ptr);
    if (it != g_producer_meta.end() && it->second.device == s.device_index()) {
      it->second.producer_streams.insert(static_cast<StreamId>(s.id()));
    }
  }

  Allocator::get(s.device_index()).record_stream(ptr, s);
#else
  (void)storage; (void)s;
#endif
}

bool has_producer_metadata(const vbt::core::StoragePtr& storage) noexcept {
#if VBT_WITH_CUDA
  if (!storage) return false;
  void* ptr = storage->data();
  if (!ptr) return true; // empty/zero-byte storage
  std::lock_guard<std::mutex> lg(g_producer_mu);
  return g_producer_meta.find(ptr) != g_producer_meta.end();
#else
  (void)storage;
  return false;
#endif
}

namespace detail {

std::vector<Stream> producer_streams_snapshot(const vbt::core::StoragePtr& storage) {
#if VBT_WITH_CUDA
  if (!storage) throw std::runtime_error("producer_streams_snapshot: null storage");
  void* ptr = storage->data();
  if (!ptr) return {};

  ProducerMetadata meta;
  {
    std::lock_guard<std::mutex> lg(g_producer_mu);
    auto it = g_producer_meta.find(ptr);
    if (it == g_producer_meta.end()) {
      throw std::runtime_error("producer_streams_snapshot: missing producer metadata");
    }
    meta = ProducerMetadata{it->second.device, it->second.producer_streams};
  }

  std::vector<StreamId> ids;
  ids.reserve(meta.producer_streams.size());
  for (StreamId sid : meta.producer_streams) ids.push_back(sid);
  std::sort(ids.begin(), ids.end());

  std::vector<Stream> out;
  out.reserve(ids.size());
  for (StreamId sid : ids) {
    out.emplace_back(Stream::UNCHECKED, static_cast<std::uint64_t>(sid), meta.device);
  }
  return out;
#else
  (void)storage;
  return {};
#endif
}

} // namespace detail

#if defined(VBT_INTERNAL_TESTS)
void debug_clear_producer_metadata_for_testing() noexcept {
#if VBT_WITH_CUDA
  std::lock_guard<std::mutex> lg(g_producer_mu);
  g_producer_meta.clear();
#endif
}
#endif

std::size_t debug_record_stream_call_count() noexcept {
#if VBT_WITH_CUDA
  return g_record_stream_calls.load(std::memory_order_relaxed);
#else
  return g_record_stream_calls;
#endif
}

void debug_reset_record_stream_call_count() noexcept {
#if VBT_WITH_CUDA
  g_record_stream_calls.store(0, std::memory_order_relaxed);
#else
  g_record_stream_calls = 0;
#endif
}

}} // namespace vbt::cuda
