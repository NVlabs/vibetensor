// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/fabric_lifetime.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_set>

#include "vbt/cuda/device.h"
#include "vbt/cuda/event_pool.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/storage.h"

#ifndef VBT_WITH_CUDA
#  define VBT_WITH_CUDA 0
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1,
              "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace vbt { namespace cuda { namespace fabric {

namespace {

struct ProducerStreamKey {
  DeviceIndex device;
  StreamId    id;
  bool operator==(const ProducerStreamKey&) const = default;
};

struct ProducerStreamKeyHash {
  std::size_t operator()(const ProducerStreamKey& k) const noexcept {
    // Cheap 64-bit mix; StreamId is already pointer-like.
    std::uint64_t x = (static_cast<std::uint64_t>(static_cast<std::uint16_t>(k.device)) << 48) ^
                      static_cast<std::uint64_t>(k.id);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<std::size_t>(x);
  }
};

static std::mutex g_compute_mu;
static std::vector<std::optional<Stream>> g_compute_streams;

static Stream get_or_create_compute_stream_(DeviceIndex dev) {
#if !VBT_WITH_CUDA
  (void)dev;
  throw std::runtime_error("CUDA not built");
#else
  if (dev < 0) {
    throw std::runtime_error("[Fabric] get_fabric_compute_stream: invalid device");
  }
  std::lock_guard<std::mutex> lg(g_compute_mu);
  const std::size_t idx = static_cast<std::size_t>(dev);
  if (g_compute_streams.size() <= idx) {
    g_compute_streams.resize(idx + 1);
  }
  if (!g_compute_streams[idx].has_value()) {
    g_compute_streams[idx] = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev);
  }
  return *g_compute_streams[idx];
#endif
}

static std::mutex g_proxy_mu;
static std::vector<std::optional<Stream>> g_proxy_streams;

static Stream get_or_create_proxy_stream_(DeviceIndex dev) {
#if !VBT_WITH_CUDA
  (void)dev;
  throw std::runtime_error("CUDA not built");
#else
  if (dev < 0) {
    throw std::runtime_error("[Fabric] get_fabric_proxy_stream: invalid device");
  }
  std::lock_guard<std::mutex> lg(g_proxy_mu);
  const std::size_t idx = static_cast<std::size_t>(dev);
  if (g_proxy_streams.size() <= idx) {
    g_proxy_streams.resize(idx + 1);
  }
  if (!g_proxy_streams[idx].has_value()) {
    g_proxy_streams[idx] = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev);
  }
  return *g_proxy_streams[idx];
#endif
}

static std::mutex g_pool_mu;
static std::vector<std::unique_ptr<EventPool>> g_event_pools;

static EventPool& get_or_create_event_pool_(DeviceIndex dev) {
  std::lock_guard<std::mutex> lg(g_pool_mu);
  const std::size_t idx = static_cast<std::size_t>(dev);
  if (g_event_pools.size() <= idx) {
    g_event_pools.resize(idx + 1);
  }
  if (!g_event_pools[idx]) {
    g_event_pools[idx] = std::make_unique<EventPool>(dev);
  }
  return *g_event_pools[idx];
}

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
static std::atomic<std::uint64_t> g_num_producer_events_primary{0};
static std::atomic<std::uint64_t> g_num_producer_events_remote{0};
static std::atomic<std::uint64_t> g_num_event_record_calls{0};
static std::atomic<std::uint64_t> g_num_stream_wait_calls{0};

static std::atomic<int> g_fail_pool_get_on_n{0};
static std::atomic<int> g_fail_record_on_n{0};
static std::atomic<int> g_fail_wait_on_n{0};

static std::atomic<int> g_pool_get_call_index{0};
static std::atomic<int> g_record_call_index{0};
static std::atomic<int> g_wait_call_index{0};

static inline bool should_fail_(std::atomic<int>& which, std::atomic<int>& counter) noexcept {
  const int n = which.load(std::memory_order_relaxed);
  if (n <= 0) return false;
  const int cur = counter.fetch_add(1, std::memory_order_relaxed) + 1;
  return cur == n;
}

#endif

static inline FabricRuntimeError make_metadata_error_(std::string msg) {
  return FabricRuntimeError{/*is_metadata_failure=*/true,
                            /*is_resource_failure=*/false,
                            std::move(msg)};
}

static inline FabricRuntimeError make_resource_error_(std::string msg) {
  return FabricRuntimeError{/*is_metadata_failure=*/false,
                            /*is_resource_failure=*/true,
                            std::move(msg)};
}

} // namespace

Stream get_fabric_compute_stream(DeviceIndex primary_device) {
  return get_or_create_compute_stream_(primary_device);
}

Stream get_fabric_proxy_stream(DeviceIndex remote_device) {
  return get_or_create_proxy_stream_(remote_device);
}

FencePlanBuildResult build_primary_remote_fence_plan(
    const FabricStorageSets& storages,
    DeviceIndex              primary_device,
    DeviceIndex              remote_device) {

  FencePlanBuildResult out;
  out.plan.primary_device = primary_device;
  out.plan.remote_device  = remote_device;

  std::unordered_set<ProducerStreamKey, ProducerStreamKeyHash> seen;

  auto add_from_storage_vec = [&](const std::vector<vbt::core::StoragePtr>& vec,
                                  DeviceIndex dev,
                                  std::vector<ProducerStreamRef>& dst) -> std::optional<FabricRuntimeError> {
    for (const auto& S : vec) {
      if (!S || S->nbytes() == 0) continue;
      if (!vbt::cuda::has_producer_metadata(S)) {
        return make_metadata_error_(
            std::string("[Fabric] missing producer metadata for storage (device=") +
            std::to_string(static_cast<int>(dev)) + ")");
      }

      try {
        vbt::cuda::for_each_producer_stream(S, [&](const vbt::cuda::Stream& s) -> bool {
          if (s.device_index() != dev) {
            // Fail closed on inconsistent metadata.
            throw std::runtime_error("producer stream device mismatch");
          }
          const ProducerStreamKey key{dev, static_cast<StreamId>(s.id())};
          if (seen.insert(key).second) {
            dst.push_back(ProducerStreamRef{dev, s});
          }
          return true;
        });
      } catch (...) {
        return make_metadata_error_(
            std::string("[Fabric] inconsistent producer metadata for storage (device=") +
            std::to_string(static_cast<int>(dev)) + ")");
      }
    }
    return std::nullopt;
  };

  if (storages.primary_storages.empty() && storages.remote_storages.empty()) {
    return out;
  }

  if (auto err = add_from_storage_vec(storages.primary_storages, primary_device, out.plan.primary_producers)) {
    out.kind = FabricFailureKind::kMetadataFailure;
    out.error = *err;
    out.plan.primary_producers.clear();
    out.plan.remote_producers.clear();
    return out;
  }

  if (!storages.remote_storages.empty()) {
    if (remote_device < 0) {
      out.kind = FabricFailureKind::kMetadataFailure;
      out.error = make_metadata_error_("[Fabric] remote storages present but remote_device < 0");
      out.plan.primary_producers.clear();
      out.plan.remote_producers.clear();
      return out;
    }
    if (auto err = add_from_storage_vec(storages.remote_storages, remote_device, out.plan.remote_producers)) {
      out.kind = FabricFailureKind::kMetadataFailure;
      out.error = *err;
      out.plan.primary_producers.clear();
      out.plan.remote_producers.clear();
      return out;
    }
  }

  return out;
}

FenceExecutionResult execute_primary_remote_fence_plan(
    const PrimaryRemoteFencePlan& plan,
    Stream                        compute_stream) {

  FenceExecutionResult out;

#if !VBT_WITH_CUDA
  (void)plan;
  (void)compute_stream;
  out.kind = FabricFailureKind::kResourceFailure;
  out.error = make_resource_error_("[Fabric] CUDA not built");
  return out;
#else
  if (plan.primary_producers.empty() && plan.remote_producers.empty()) {
    return out;
  }

  const DeviceGuard dg_compute(compute_stream.device_index());

  auto run_vec = [&](const std::vector<ProducerStreamRef>& vec, bool is_primary) -> std::optional<FabricRuntimeError> {
    for (const auto& ref : vec) {
      EventPool& pool = get_or_create_event_pool_(ref.device);

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
      if (should_fail_(g_fail_pool_get_on_n, g_pool_get_call_index)) {
        return make_resource_error_("[Fabric] injected EventPool::get failure");
      }
#endif

      PooledEvent e = pool.get();
      if (!e.valid()) {
        return make_resource_error_("[Fabric] failed to acquire pooled event");
      }

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
      if (is_primary) {
        g_num_producer_events_primary.fetch_add(1, std::memory_order_relaxed);
      } else {
        g_num_producer_events_remote.fetch_add(1, std::memory_order_relaxed);
      }
#endif

      {
        DeviceGuard dg_prod(ref.device);

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
        if (should_fail_(g_fail_record_on_n, g_record_call_index)) {
          return make_resource_error_("[Fabric] injected cudaEventRecord failure");
        }
#endif

        cudaError_t st = cudaEventRecord(reinterpret_cast<cudaEvent_t>(e.raw()),
                                         reinterpret_cast<cudaStream_t>(ref.stream.handle()));
        if (st != cudaSuccess) {
          (void)cudaGetLastError();
          return make_resource_error_("[Fabric] cudaEventRecord failed");
        }

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
        g_num_event_record_calls.fetch_add(1, std::memory_order_relaxed);
#endif
      }

      {
        DeviceGuard dg_wait(compute_stream.device_index());

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
        if (should_fail_(g_fail_wait_on_n, g_wait_call_index)) {
          return make_resource_error_("[Fabric] injected cudaStreamWaitEvent failure");
        }
#endif

        cudaError_t st = cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(compute_stream.handle()),
                                             reinterpret_cast<cudaEvent_t>(e.raw()), 0);
        if (st != cudaSuccess) {
          (void)cudaGetLastError();
          return make_resource_error_("[Fabric] cudaStreamWaitEvent failed");
        }

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
        g_num_stream_wait_calls.fetch_add(1, std::memory_order_relaxed);
#endif
      }
    }
    return std::nullopt;
  };

  if (auto err = run_vec(plan.primary_producers, /*is_primary=*/true)) {
    out.kind = FabricFailureKind::kResourceFailure;
    out.error = *err;
    return out;
  }
  if (auto err = run_vec(plan.remote_producers, /*is_primary=*/false)) {
    out.kind = FabricFailureKind::kResourceFailure;
    out.error = *err;
    return out;
  }

  return out;
#endif
}

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS

FabricFenceDebugCounters debug_get_fabric_fence_counters_for_testing() noexcept {
  FabricFenceDebugCounters c;
  c.num_producer_events_primary = g_num_producer_events_primary.load(std::memory_order_relaxed);
  c.num_producer_events_remote  = g_num_producer_events_remote.load(std::memory_order_relaxed);
  c.num_event_record_calls      = g_num_event_record_calls.load(std::memory_order_relaxed);
  c.num_stream_wait_calls       = g_num_stream_wait_calls.load(std::memory_order_relaxed);
  return c;
}

void debug_reset_fabric_fence_counters_for_testing() noexcept {
  g_num_producer_events_primary.store(0, std::memory_order_relaxed);
  g_num_producer_events_remote.store(0, std::memory_order_relaxed);
  g_num_event_record_calls.store(0, std::memory_order_relaxed);
  g_num_stream_wait_calls.store(0, std::memory_order_relaxed);

  g_pool_get_call_index.store(0, std::memory_order_relaxed);
  g_record_call_index.store(0, std::memory_order_relaxed);
  g_wait_call_index.store(0, std::memory_order_relaxed);

  g_fail_pool_get_on_n.store(0, std::memory_order_relaxed);
  g_fail_record_on_n.store(0, std::memory_order_relaxed);
  g_fail_wait_on_n.store(0, std::memory_order_relaxed);
}

void debug_fail_fence_event_pool_get_on_n_for_testing(int n) noexcept {
  g_fail_pool_get_on_n.store(n, std::memory_order_relaxed);
}

void debug_fail_fence_event_record_on_n_for_testing(int n) noexcept {
  g_fail_record_on_n.store(n, std::memory_order_relaxed);
}

void debug_fail_fence_stream_wait_on_n_for_testing(int n) noexcept {
  g_fail_wait_on_n.store(n, std::memory_order_relaxed);
}

#endif

}}} // namespace vbt::cuda::fabric
