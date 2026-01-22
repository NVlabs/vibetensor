// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/allocator_async.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cassert>
#include <thread>
#include <chrono>
#include <limits>
#include <cmath>
#include <random>

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#include <unordered_set>
#include <string>

#if VBT_WITH_CUDA
#  include <cuda_runtime_api.h>
#endif
#include "vbt/cuda/device.h"
#include "vbt/cuda/guard.h"

namespace vbt { namespace cuda {

#ifdef VBT_INTERNAL_TESTS
// Per-thread depth of Allocator::mu_ holdings for this translation unit.
static thread_local int debug_mu_depth_ = 0;

// Global counters for raw_alloc entrypoints in internal tests.
static std::atomic<std::uint64_t> g_debug_raw_alloc_nostream_calls{0};
static std::atomic<std::uint64_t> g_debug_raw_alloc_stream_calls{0};

struct MuLockGuardDebug final {
  std::mutex& mu;
  explicit MuLockGuardDebug(std::mutex& m) : mu(m) {
    mu.lock();
    ++debug_mu_depth_;
  }
  ~MuLockGuardDebug() {
    --debug_mu_depth_;
    mu.unlock();
  }
  MuLockGuardDebug(const MuLockGuardDebug&) = delete;
  MuLockGuardDebug& operator=(const MuLockGuardDebug&) = delete;
};

struct MuUniqueLockDebug final {
  std::mutex* mu{nullptr};
  bool        owns{false};

  MuUniqueLockDebug() = default;
  explicit MuUniqueLockDebug(std::mutex& m) : mu(&m) {
    lock();
  }

  ~MuUniqueLockDebug() {
    if (owns) {
      unlock();
    }
  }

  void lock() {
    if (!mu || owns) return;
    mu->lock();
    ++debug_mu_depth_;
    owns = true;
  }

  void unlock() {
    if (!mu || !owns) return;
    --debug_mu_depth_;
    mu->unlock();
    owns = false;
  }

  bool owns_lock() const noexcept { return owns; }
};

struct DebugMuDepthExitGuard final {
  ~DebugMuDepthExitGuard() {
    assert(debug_mu_depth_ == 0 &&
           "run_gc_pass_if_eligible leaked or overlapped mu_ usage in this thread");
  }
};
#endif

namespace {

struct CaptureTLS {
  bool        active{false};
  DeviceIndex dev{-1};
  MempoolId   id{};
};
static thread_local CaptureTLS s_capture_tls{};

[[noreturn]] void throw_allocator_capture_denied_device(DeviceIndex dev) {
  detail::bump_allocator_capture_denied();
  throw std::runtime_error(std::string(kErrAllocatorCaptureDenied) +
                           " on device " + std::to_string(dev));
}

[[noreturn]] void throw_allocator_capture_denied_stream(DeviceIndex dev, StreamId sid) {
  detail::bump_allocator_capture_denied();
  throw std::runtime_error(std::string(kErrAllocatorCaptureDenied) +
                           " device=" + std::to_string(dev) +
                           " stream=" + std::to_string(sid));
}

#if VBT_WITH_CUDA
static bool is_stream_capturing(DeviceIndex dev, StreamId sid) {
  DeviceGuard dg(dev);
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  cudaError_t rc = cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(sid)), &st);
  if (rc != cudaSuccess) { return false; }
  return st != cudaStreamCaptureStatusNone;
}
#endif

static bool parse_fraction_01(const std::string& t, double& out) {
  if (t.empty()) return false;
  errno = 0;
  char* end = nullptr;
  double x = std::strtod(t.c_str(), &end);
  if (errno != 0 || !end || *end != '\0') return false;
  if (!std::isfinite(x) || x < 0.0 || x > 1.0) return false;
  out = x;
  return true;
}

static inline bool is_oversize_size(std::size_t s, std::size_t M) noexcept {
  return (M != 0 &&
          M != std::numeric_limits<std::size_t>::max() &&
          s >= M);
}

static std::size_t safe_prospective_reserved(std::size_t rounded,
                                             std::size_t reserved) noexcept {
  const auto max_sz = std::numeric_limits<std::size_t>::max();
  if (rounded > max_sz - reserved) {
    // Saturate on unsigned overflow in reserved + rounded.
    return max_sz;
  }
  return reserved + rounded;
}

static bool cap_exceeded(std::size_t rounded,
                         std::size_t reserved,
                         std::size_t limit) noexcept {
  const auto max_sz = std::numeric_limits<std::size_t>::max();
  if (limit == max_sz) {
    // Cap disabled: fraction >= 1.0, non-finite, or unknown total bytes.
    return false;
  }
  const std::size_t prospective = safe_prospective_reserved(rounded, reserved);
  return prospective > limit;
}

[[noreturn]] static void throw_fraction_cap_oom(
    DeviceIndex dev,
    std::size_t requested_bytes,
    std::size_t rounded_bytes,
    std::size_t limit_bytes,
    std::size_t reserved_bytes_after_gc) {
  auto to_mib_u = [](std::size_t b) {
    return static_cast<unsigned long long>(b >> 20);
  };

  std::string msg =
      "CUDA out of memory: per-process memory fraction cap exceeded. "
      "Tried to allocate " + std::to_string(requested_bytes) +
      " bytes (rounded=" + std::to_string(rounded_bytes) + ", " +
      std::to_string(to_mib_u(rounded_bytes)) + " MiB). " +
      "Device " + std::to_string(dev) +
      " has a per-process memory fraction cap of " +
      std::to_string(limit_bytes) + " bytes (" +
      std::to_string(to_mib_u(limit_bytes)) + " MiB), with " +
      std::to_string(reserved_bytes_after_gc) + " bytes (" +
      std::to_string(to_mib_u(reserved_bytes_after_gc)) +
      " MiB) already reserved by VibeTensor.";

  throw std::runtime_error(msg);
}

static std::once_flag s_warn_fraction_once;

// Simple singleton table sized on first use
struct AllocTable {
  std::once_flag init_once;
  std::vector<std::unique_ptr<Allocator>> per_dev;
};

static std::mutex g_alloc_init_mu;

AllocTable& table() {
  static AllocTable t;
  return t;
}
} // anonymous

#if VBT_WITH_CUDA
// Test hookable wrapper for cudaMemGetInfo used by Allocator::Allocator.
// Default points to ::cudaMemGetInfo; tests may override via the debug hook.
using CudaMemGetInfoFn = cudaError_t (*)(size_t*, size_t*);
static CudaMemGetInfoFn s_cudaMemGetInfo = ::cudaMemGetInfo;

#ifdef VBT_INTERNAL_TESTS
// Global counter for allocator cudaMalloc growth calls in internal tests.
static std::atomic<std::uint64_t> g_debug_cudaMalloc_calls{0};

// Central wrapper for cudaMalloc that all allocator growth paths must use.
static cudaError_t cudaMalloc_with_hook(void** ptr, std::size_t size) {
  g_debug_cudaMalloc_calls.fetch_add(1, std::memory_order_relaxed);
  return ::cudaMalloc(ptr, size);
}

std::uint64_t debug_get_cudaMalloc_call_count_for_testing() noexcept {
  return g_debug_cudaMalloc_calls.load(std::memory_order_relaxed);
}

void debug_reset_cudaMalloc_calls_for_testing() noexcept {
  g_debug_cudaMalloc_calls.store(0, std::memory_order_relaxed);
}
#else
// Production builds call straight into ::cudaMalloc.
static cudaError_t cudaMalloc_with_hook(void** ptr, std::size_t size) {
  return ::cudaMalloc(ptr, size);
}
#endif  // VBT_INTERNAL_TESTS
#endif  // VBT_WITH_CUDA


bool Allocator::CmpSizeAddr::operator()(const Block* a, const Block* b) const {
  if (a->size != b->size) return a->size < b->size;
  return a->ptr < b->ptr;
}

Allocator& Allocator::get(DeviceIndex dev) {
  auto& t = table();
  std::call_once(t.init_once, [&t](){
    std::lock_guard<std::mutex> lk(g_alloc_init_mu);
    t.per_dev.resize(static_cast<std::size_t>(std::max(0, device_count())));
  });
  if (dev < 0) {
#if VBT_WITH_CUDA
    int cur = 0; cudaGetDevice(&cur); dev = static_cast<DeviceIndex>(cur);
#else
    dev = 0;
#endif
  }
  auto idx = static_cast<std::size_t>(dev);
  std::lock_guard<std::mutex> lk(g_alloc_init_mu);
  if (idx >= t.per_dev.size()) {
    t.per_dev.resize(idx + 1);
  }
  if (!t.per_dev[idx]) {
    t.per_dev[idx].reset(new Allocator(dev));
  }
  return *t.per_dev[idx];
}

#ifdef VBT_INTERNAL_TESTS
void Allocator::debug_assert_mu_not_held_for_gc() const noexcept {
  assert(debug_mu_depth_ == 0 &&
         "run_gc_pass_if_eligible called while Allocator::mu_ is held in this thread");
}
#endif

#if VBT_WITH_CUDA && defined(VBT_INTERNAL_TESTS)

void debug_set_cudaMemGetInfo_hook_for_testing(CudaMemGetInfoFn fn) noexcept {
  s_cudaMemGetInfo = fn ? fn : ::cudaMemGetInfo;
}

CudaMemGetInfoGuard::CudaMemGetInfoGuard(CudaMemGetInfoFn fn) noexcept {
  prev_ = s_cudaMemGetInfo;
  s_cudaMemGetInfo = fn ? fn : ::cudaMemGetInfo;
}

CudaMemGetInfoGuard::~CudaMemGetInfoGuard() noexcept {
  s_cudaMemGetInfo = prev_ ? prev_ : ::cudaMemGetInfo;
}

#endif

#if VBT_WITH_CUDA

void Allocator::setMemoryFraction(double f) {
  if (f < 0.0 || f > 1.0) {
    throw std::invalid_argument(
        "per_process_memory_fraction must be in [0, 1]");
  }

  memory_fraction_.store(f, std::memory_order_relaxed);

  if (cfg_.backend == BackendKind::Async) {
    AsyncBackend::get(dev_).set_memory_fraction(f);
  }
}

std::size_t Allocator::current_limit_bytes() const noexcept {
  using std::isfinite;
  const auto MAX = std::numeric_limits<std::size_t>::max();
  const std::size_t T = device_total_bytes_;

  double frac = memory_fraction_.load(std::memory_order_relaxed);

  // Cap disabled when we have no reliable total or a "large" / invalid
  if (T == 0 || !isfinite(frac) || frac >= 1.0) {
    return MAX;
  }
  // Hard-zero cap for T > 0 and non-positive fractions.
  if (frac <= 0.0) {
    return 0;
  }

  long double limit = static_cast<long double>(T) *
                      static_cast<long double>(frac);

  if (limit >= static_cast<long double>(MAX)) {
    return MAX;
  }

  return static_cast<std::size_t>(limit);
}

void Allocator::maybe_run_fraction_gate(std::size_t nbytes,
                                        std::size_t rounded) {
  const auto MAX = std::numeric_limits<std::size_t>::max();

  if (cfg_.backend != BackendKind::Native) {
    return;  // Defensive: async should have returned earlier.
  }
  if (rounded == 0) {
    return;  // No growth.
  }

  // 1. Snapshot current cap state.
  const std::size_t limit_before = current_limit_bytes();
  if (limit_before == MAX) {
    return;
  }

  // 2. Initial reserved snapshot & cap check.
  std::size_t reserved_before = 0;
  {
    MuLockGuard lg(mu_);
    reserved_before = static_cast<std::size_t>(
        stats_.reserved_bytes_all_current);
  }

  if (!cap_exceeded(rounded, reserved_before, limit_before)) {
    return;  // Within cap.
  }

  // 3. Record breach before any reclamation.
  {
    MuLockGuard lg(mu_);
    stats_.fraction_cap_breaches += 1;
  }

  // 4. Drain limbo off-lock.
  process_events();

  // 5. Compute overage and GC budget using the original limit.
  std::size_t overage = 0;
  {
    MuLockGuard lg(mu_);
    const std::size_t reserved_for_overage = static_cast<std::size_t>(
        stats_.reserved_bytes_all_current);
    const std::size_t prospective =
        safe_prospective_reserved(rounded, reserved_for_overage);
    if (prospective > limit_before) {
      overage = prospective - limit_before;
    }
  }

  std::size_t gc_target_bytes = 0;
  if (overage > 0) {
    constexpr std::size_t kFractionGcBudgetMultiplier = 2;
    const std::size_t budget_cap =
        (rounded > MAX / kFractionGcBudgetMultiplier)
            ? MAX
            : rounded * kFractionGcBudgetMultiplier;
    gc_target_bytes = (overage < budget_cap) ? overage : budget_cap;
  }

  // 6. Run GC(FractionCap) if we have a budget (off-lock).
  if (gc_target_bytes > 0) {
    (void)run_gc_pass_if_eligible(gc_target_bytes, GcReason::FractionCap);
  }

  // 7. Re-evaluate cap after GC using fresh snapshots.
  std::size_t limit_after_gc = current_limit_bytes();
  if (limit_after_gc == MAX) {
    // Fraction was relaxed or total became unknown.
    return;
  }

  std::size_t reserved_after_gc = 0;
  {
    MuLockGuard lg(mu_);
    reserved_after_gc = static_cast<std::size_t>(
        stats_.reserved_bytes_all_current);
  }

  if (!cap_exceeded(rounded, reserved_after_gc, limit_after_gc)) {
    return;  // GC and/or fraction change made room.
  }

  // 8. Final reclamation via emptyCache (off-lock).
  emptyCache();

  // 9. Final cap check and potential misfire.
  std::size_t limit_after_ec = current_limit_bytes();
  if (limit_after_ec == MAX) {
    // Cap disabled while we were reclaiming.
    return;
  }

  std::size_t reserved_after_ec = 0;
  {
    MuLockGuard lg(mu_);
    reserved_after_ec = static_cast<std::size_t>(
        stats_.reserved_bytes_all_current);
  }

  if (cap_exceeded(rounded, reserved_after_ec, limit_after_ec)) {
    {
      MuLockGuard lg(mu_);
      stats_.fraction_cap_misfires += 1;
    }
    // Fraction-cap OOM: do NOT increment num_ooms.
    throw_fraction_cap_oom(dev_, nbytes, rounded,
                           limit_after_ec, reserved_after_ec);
  }

  // Otherwise: cap satisfied; caller may proceed to cudaMalloc.
}

#endif  // VBT_WITH_CUDA

std::vector<GraphPoolSnapshot>
snapshot_graph_pools(std::optional<MempoolId> filter) {
  std::vector<GraphPoolSnapshot> out;

#if !VBT_WITH_CUDA
  (void)filter;
  return out;
#else
  // CPU-only or no devices: always return empty.
  int n = device_count();
  if (n <= 0) {
    return out;
  }

  // Negative device indices are treated as invalid filters.
  if (filter.has_value() && filter->dev < 0) {
    return out;
  }

  auto visit_device = [&](DeviceIndex dev_index) {
    Allocator& alloc = Allocator::get(dev_index);

    Allocator::MuLockGuard lg(alloc.mu_);

    // Build aggregation map for pools on this device.
    std::unordered_map<std::uint64_t, GraphPoolSnapshot> by_pool;

    std::optional<std::uint64_t> pool_filter_id;
    if (filter.has_value() && filter->dev >= 0 &&
        static_cast<int>(filter->dev) == static_cast<int>(dev_index) &&
        filter->id != 0) {
      pool_filter_id = filter->id;
    }

    for (const auto& kv : alloc.graph_pools_) {
      std::uint64_t pid = kv.first;
      if (pool_filter_id.has_value() && pid != *pool_filter_id) {
        continue;
      }
      GraphPoolSnapshot snap;
      snap.id = MempoolId{dev_index, pid};
      by_pool.emplace(pid, snap);
    }

    // If requesting a specific pool id on this device and it's unknown,
    // do not emit any snapshots for this device.
    if (pool_filter_id.has_value() && by_pool.empty()) {
      return;
    }

    // Aggregate per-segment metrics from tracked blocks.
    for (const auto& kv : alloc.by_ptr_) {
      Allocator::Block* b = kv.second;
      if (!b || !b->segment_head) {
        continue;
      }
      std::uint64_t pid = b->graph_pool_id;
      if (pid == 0) {
        continue;  // default/global pool
      }
      auto it = by_pool.find(pid);
      if (it == by_pool.end()) {
        continue;  // pool filtered out or unknown
      }

      GraphPoolSnapshot& snap = it->second;
      snap.id = MempoolId{dev_index, pid};

      std::uint64_t seg_blocks = 0;
      std::uint64_t seg_reserved = 0;
      std::uint64_t seg_active = 0;

      for (Allocator::Block* cur = b; cur != nullptr; cur = cur->next) {
        ++seg_blocks;
        seg_reserved += static_cast<std::uint64_t>(cur->size);
        if (cur->allocated) {
          seg_active += static_cast<std::uint64_t>(cur->size);
        }
      }

      snap.segments += 1;
      snap.blocks += seg_blocks;
      snap.bytes_reserved += seg_reserved;
      snap.bytes_active += seg_active;
    }

    for (auto& kv : by_pool) {
      out.push_back(std::move(kv.second));
    }
  };

  if (!filter.has_value()) {
    // All devices, all pools.
    for (int i = 0; i < n; ++i) {
      visit_device(static_cast<DeviceIndex>(i));
    }
  } else {
    // Single-device filters (all pools or a specific pool id).
    DeviceIndex dev_index = filter->dev;
    int dev_int = static_cast<int>(dev_index);
    if (dev_int < 0 || dev_int >= n) {
      return out;  // out-of-range device => empty result
    }
    visit_device(dev_index);
  }

  return out;
#endif
}

std::vector<MemorySegmentSnapshot>
snapshot_memory_segments(std::optional<DeviceIndex> device_filter) {
  std::vector<MemorySegmentSnapshot> out;

#if !VBT_WITH_CUDA
  (void)device_filter;
  return out;
#else
  int n = device_count();
  if (n <= 0) {
    // CPU-only or no devices: always return empty and do not touch allocator state.
    return out;
  }

  auto visit_device = [&](DeviceIndex dev_index) {
    Allocator& alloc = Allocator::get(dev_index);

    if (alloc.cfg_.backend == BackendKind::Async) {
      return;
    }

    Allocator::MuLockGuard lg(alloc.mu_);

    for (const auto& kv : alloc.by_ptr_) {
      Allocator::Block* head = kv.second;
      if (!head || !head->segment_head) {
        continue;
      }

      MemorySegmentSnapshot snap;
      snap.device = dev_index;
      snap.pool_id = head->graph_pool_id;
      snap.bytes_reserved = 0;
      snap.bytes_active = 0;
      snap.blocks = 0;

      for (Allocator::Block* cur = head; cur != nullptr; cur = cur->next) {
        ++snap.blocks;
        snap.bytes_reserved += static_cast<std::uint64_t>(cur->size);
        if (cur->allocated) {
          snap.bytes_active += static_cast<std::uint64_t>(cur->size);
        }
        if (cur->next && cur->next->segment_head) {
          break;
        }
      }

      out.push_back(std::move(snap));
    }
  };

  if (device_filter.has_value()) {
    DeviceIndex dev_index = *device_filter;
    int dev_int = static_cast<int>(dev_index);
    if (dev_int < 0 || dev_int >= n) {
      throw std::out_of_range("snapshot_memory_segments: device index out of range");
    }
    visit_device(dev_index);
  } else {
    for (int i = 0; i < n; ++i) {
      visit_device(static_cast<DeviceIndex>(i));
    }
  }

  return out;
#endif
}

MempoolId Allocator::create_pool_id(DeviceIndex dev) {
  return get(dev).create_pool_id_();
}

void Allocator::retain_pool(DeviceIndex dev, MempoolId id) {
  get(dev).retain_pool_(id);
}

void Allocator::release_pool(DeviceIndex dev, MempoolId id) noexcept {
  get(dev).release_pool_(id);
}

Allocator::AllocateToPoolGuard Allocator::begin_allocate_to_pool(DeviceIndex dev, MempoolId id) {
  return get(dev).begin_allocate_to_pool_(id);
}

void Allocator::end_allocate_to_pool(DeviceIndex dev, MempoolId id) noexcept {
  get(dev).end_allocate_to_pool_(id);
}

void Allocator::cancel_allocate_to_pool(DeviceIndex dev, MempoolId id) noexcept {
  get(dev).cancel_allocate_to_pool_(id);
}

void Allocator::mark_pool_replay_begin(DeviceIndex dev, MempoolId id) {
  get(dev).mark_pool_replay_begin_(id);
}

void Allocator::mark_pool_replay_end(DeviceIndex dev, MempoolId id) noexcept {
  get(dev).mark_pool_replay_end_(id);
}

void Allocator::prewarm_graph_pool_for_stream(DeviceIndex dev, MempoolId id,
                                              Stream stream,
                                              std::size_t min_total_bytes,
                                              int min_blocks) {
  get(dev).prewarm_graph_pool_for_stream_(id, stream, min_total_bytes, min_blocks);
}

Allocator::Config Allocator::parse_env_now_() {
  Config cfg;
  const char* env = std::getenv("VBT_CUDA_ALLOC_CONF");
  if (!env || !*env) return cfg;
  auto is_space = [](char c){ return c==' '||c=='\t'||c=='\n'; };
  std::string s(env);
  size_t i = 0;
  auto trim = [&](std::string& t){ size_t a=0; while (a<t.size() && is_space(t[a])) ++a; size_t b=t.size(); while (b> a && is_space(t[b-1])) --b; t = t.substr(a, b-a); };
  auto to_uint = [&](const std::string& t, std::size_t& out)->bool{
    if (t.empty()) return false;
    // Strict: only decimal digits allowed; reject any sign or other chars
    for (unsigned char ch : t) {
      if (!std::isdigit(ch)) return false;
    }
    char* end=nullptr; errno=0; unsigned long long x = std::strtoull(t.c_str(), &end, 10);
    if (errno!=0 || (end && *end!='\0')) return false;
    if (x > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) return false;
    out = static_cast<std::size_t>(x); return true; };
  auto to_bytes = [&](const std::string& t, std::size_t& out)->bool{
    std::string u=t; for (auto& c: u) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    std::size_t mul=1; if (u.size()>=3){ auto suf = u.substr(u.size()-3); if (suf=="kib") { mul=1ull<<10; u.resize(u.size()-3);} else if (suf=="mib") { mul=1ull<<20; u.resize(u.size()-3);} else if (suf=="gib") { mul=1ull<<30; u.resize(u.size()-3);} }
    std::size_t base=0; if (!to_uint(u, base)) return false; if (mul != 0 && base > std::numeric_limits<std::size_t>::max() / mul) return false; out = base * mul; return true; };
  auto to_double01 = [&](const std::string& t, double& out)->bool{ char* end=nullptr; errno=0; double x = std::strtod(t.c_str(), &end); if (errno!=0|| (end && *end!='\0')) return false; if (x<0.0||x>1.0) return false; out = x; return true; };
  auto to_bool = [&](const std::string& t)->bool{ std::string u=t; for (auto& c:u) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return (u=="1"||u=="true"||u=="yes"||u=="on"); };
  static std::unordered_set<std::string> s_warned_unknown;
  while (i < s.size()) {
    while (i < s.size() && (is_space(s[i]) || s[i]==',')) ++i; if (i>=s.size()) break;
    size_t k0=i; while (i<s.size() && s[i] != '=' && s[i] != ',') ++i; if (i>=s.size()|| s[i] != '=') break; std::string key = s.substr(k0, i-k0); ++i;
    size_t v0=i; while (i<s.size() && s[i] != ',') ++i; std::string val = s.substr(v0, i-v0);
    trim(key); trim(val);
    if (key == "max_split_size") { std::size_t v=0; if (to_bytes(val, v)) cfg.max_split_size_bytes = v; }
    else if (key == "max_split_size_mb") { std::size_t v=0; if (to_uint(val, v)) { if (v <= (std::numeric_limits<std::size_t>::max() >> 20)) cfg.max_split_size_bytes = v * (1ull<<20); } }
    else if (key == "garbage_collection_threshold") { double d=0; if (to_double01(val, d)) cfg.garbage_collection_threshold = d; }
    else if (key == "event_pool_cap") { std::size_t v=0; if (to_uint(val, v)) cfg.event_pool_cap = v; }
    else if (key == "event_pool_prewarm") { std::size_t v=0; if (to_uint(val, v)) cfg.event_pool_prewarm = v; }
    else if (key == "process_events_every_frees") { std::size_t v=0; if (to_uint(val, v)) cfg.process_events_every_frees = v; }
    else if (key == "roundup_tolerance_bytes") { std::size_t v=0; if (to_bytes(val, v)) cfg.roundup_tolerance_bytes = v; }
    else if (key == "oom_retry_count") { std::size_t v=0; if (to_uint(val, v)) cfg.oom_retry_count = v; }
    else if (key == "oom_retry_sleep_ms") { std::size_t v=0; if (to_uint(val, v)) cfg.oom_retry_sleep_ms = v; }
    else if (key == "max_non_split_rounding_mb") { std::size_t v=0; if (to_uint(val, v)) cfg.max_non_split_rounding_bytes = v * (1ull<<20); }
    else if (key == "enable_cross_stream_fallback") { cfg.enable_cross_stream_fallback = to_bool(val); }
    else if (key == "roundup_power2_divisions") { std::size_t v=0; if (to_uint(val, v)) cfg.roundup_power2_divisions = v; }
    else if (key == "backend") {
      std::string u = val; for (auto& c:u) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      if (u == "native") cfg.backend = BackendKind::Native;
      else if (u == "cudamallocasync" || u == "cudaMallocAsync") cfg.backend = BackendKind::Async;
    }
    else if (key == "per_process_memory_fraction") {
      double v = 0.0;
      if (parse_fraction_01(val, v)) {
        cfg.per_process_memory_fraction = v;
      } else {
        cfg.per_process_memory_fraction = 1.0;
        auto bad = val;  // capture by value for call_once
        std::call_once(s_warn_fraction_once, [bad]() {
          std::fprintf(stderr,
            "[VBT_CUDA_ALLOC_CONF] warning: invalid per_process_memory_fraction='%s' "
            "(must be in [0,1]); using default 1.0\n",
            bad.c_str());
        });
      }
    }
    else if (key == "release_threshold_bytes") { std::size_t v=0; if (to_bytes(val, v)) cfg.release_threshold_bytes = v; }
    else if (key == "reuse_follow_event_deps") { cfg.reuse_follow_event_deps = to_bool(val); }
    else if (key == "reuse_allow_opportunistic") { cfg.reuse_allow_opportunistic = to_bool(val); }
    else if (key == "reuse_allow_internal_deps") { cfg.reuse_allow_internal_deps = to_bool(val); }
    else if (key == "enable_block_splitting") { cfg.enable_block_splitting = to_bool(val); }
    else {
      static std::mutex s_warned_mu;
      std::lock_guard<std::mutex> lk(s_warned_mu);
      if (!s_warned_unknown.count(key)) {
        s_warned_unknown.insert(key);
        std::fprintf(stderr, "[VBT_CUDA_ALLOC_CONF] warning: unknown key '%s' (ignored)\n", key.c_str());
      }
    }
  }
  // Post-parse normalization/validation
  if (cfg.event_pool_prewarm > cfg.event_pool_cap) cfg.event_pool_prewarm = cfg.event_pool_cap;
  const std::size_t kMaxCadence = 1000000;
  if (cfg.process_events_every_frees > kMaxCadence) cfg.process_events_every_frees = kMaxCadence;
  return cfg;
}

bool Allocator::split_enabled() const noexcept {
  return cfg_.backend == BackendKind::Native && cfg_.enable_block_splitting;
}

Allocator::Allocator(DeviceIndex dev)
  : cfg_(parse_env_now_()),
    dev_(dev),
    mu_(),
    events_(dev, EventPoolConfig{cfg_.event_pool_cap,
                                 cfg_.event_pool_prewarm}) {
  if (cfg_.backend == BackendKind::Async) {
#if VBT_WITH_CUDA
    // configure async backend (fraction is owned by setMemoryFraction)
    AsyncBackend::get(dev_).configure(
      cfg_.per_process_memory_fraction,
      cfg_.release_threshold_bytes,
      cfg_.reuse_follow_event_deps,
      cfg_.reuse_allow_opportunistic,
      cfg_.reuse_allow_internal_deps);
#endif
  }

#if VBT_WITH_CUDA
  // Seed canonical fraction and async backend from env.
  setMemoryFraction(cfg_.per_process_memory_fraction);

  // native backend. Failure leaves device_total_bytes_ == 0 and clears
  // any sticky CUDA error.
  if (cfg_.backend == BackendKind::Native) {
    DeviceGuard dg(dev_);
    size_t freeB = 0;
    size_t totalB = 0;
    cudaError_t st = s_cudaMemGetInfo(&freeB, &totalB);
    if (st == cudaSuccess) {
      device_total_bytes_ = static_cast<std::size_t>(totalB);
    } else {
      device_total_bytes_ = 0;
      (void)cudaGetLastError();
    }
  }
#endif
}

void Allocator::gc_pool_locked(std::uint64_t id_val, GraphPrivatePool& pool) {
  // Assumes mu_ is held.
  if (pool.refcnt == 0 && pool.active_capture_count == 0 && pool.active_replay_count == 0 && pool.prewarm_in_progress == 0) {
    // Reset ownership tags for blocks belonging to this pool back to global.
    for (auto& kv : by_ptr_) {
      Block* b = kv.second;
      if (b != nullptr && b->graph_pool_id == id_val) {
        b->graph_pool_id = 0;
        // Reset gc_age on the segment head now that the segment becomes global.
        Block* head = b;
        while (head->prev) {
          head = head->prev;
        }
        head->gc_age = 0;
      }
    }
    graph_pools_.erase(id_val);
    stats_.graphs_pools_released += 1;
  }
}

MempoolId Allocator::create_pool_id_() {
  MuLockGuard lg(mu_);
  const std::uint64_t id = next_graph_pool_id_++;
  (void)graph_pools_.emplace(id, GraphPrivatePool{});
  stats_.graphs_pools_created += 1;
  return MempoolId{dev_, id};
}

void Allocator::retain_pool_(MempoolId id) {
  if (id.dev != dev_ || id.id == 0) {
    throw std::runtime_error("unknown mempool id");
  }
  MuLockGuard lg(mu_);
  auto it = graph_pools_.find(id.id);
  if (it == graph_pools_.end()) {
    throw std::runtime_error("unknown mempool id");
  }
  auto prev = it->second.refcnt;
  it->second.refcnt = static_cast<std::uint32_t>(prev + 1);
  if (prev == 0) {
    stats_.graphs_pools_active += 1;
  }
}

void Allocator::release_pool_(MempoolId id) noexcept {
  if (id.dev != dev_ || id.id == 0) {
    return;
  }
  MuLockGuard lg(mu_);
  auto it = graph_pools_.find(id.id);
  if (it == graph_pools_.end()) {
    return;
  }
  auto& gp = it->second;
  if (gp.refcnt > 0) {
    --gp.refcnt;
    if (gp.refcnt == 0 && stats_.graphs_pools_active > 0) {
      stats_.graphs_pools_active -= 1;
    }
  }
  gc_pool_locked(id.id, gp);
}

Allocator::AllocateToPoolGuard Allocator::begin_allocate_to_pool_(MempoolId id) {
  if (id.dev != dev_ || id.id == 0) {
    throw std::runtime_error("unknown mempool id");
  }
  {
    MuLockGuard lg(mu_);
    auto it = graph_pools_.find(id.id);
    if (it == graph_pools_.end()) {
      throw std::runtime_error("unknown mempool id");
    }
    auto& gp = it->second;
    if (gp.active_capture_count > 0 || gp.active_replay_count > 0 || gp.prewarm_in_progress > 0) {
      std::string msg = "pool is busy with active capture";
      msg += " (refcnt=" + std::to_string(gp.refcnt) +
             ", active_capture_count=" + std::to_string(gp.active_capture_count) +
             ", active_replay_count=" + std::to_string(gp.active_replay_count) +
             ", prewarm_in_progress=" + std::to_string(gp.prewarm_in_progress) + ")";
      throw std::runtime_error(msg);
    }
    gp.active_capture_count = 1u;
    gp.begins += 1;
  }
  // Set TLS routing for this thread/device and device-scoped flag
  s_capture_tls.active = true;
  s_capture_tls.dev = dev_;
  s_capture_tls.id = id;
  routing_active_flag_.store(true, std::memory_order_relaxed);
  return AllocateToPoolGuard(this, id);
}

void Allocator::end_allocate_to_pool_(MempoolId id) noexcept {
  // Clear TLS first if it matches
  if (s_capture_tls.active && s_capture_tls.dev == dev_ && s_capture_tls.id.dev == id.dev && s_capture_tls.id.id == id.id) {
    s_capture_tls.active = false;
    s_capture_tls.dev = -1;
    s_capture_tls.id = MempoolId{};
  }
  routing_active_flag_.store(false, std::memory_order_relaxed);
  MuLockGuard lg(mu_);
  if (id.dev != dev_ || id.id == 0) {
    return;
  }
  auto it = graph_pools_.find(id.id);
  if (it == graph_pools_.end()) {
    return;
  }
  auto& gp = it->second;
  if (gp.active_capture_count > 0) {
    gp.active_capture_count = 0u;
    gp.ends += 1;
  }
  gc_pool_locked(id.id, gp);
}

void Allocator::cancel_allocate_to_pool_(MempoolId id) noexcept {
  // Clear TLS first if it matches
  if (s_capture_tls.active && s_capture_tls.dev == dev_ && s_capture_tls.id.dev == id.dev && s_capture_tls.id.id == id.id) {
    s_capture_tls.active = false;
    s_capture_tls.dev = -1;
    s_capture_tls.id = MempoolId{};
  }
  routing_active_flag_.store(false, std::memory_order_relaxed);
  MuLockGuard lg(mu_);
  if (id.dev != dev_ || id.id == 0) {
    return;
  }
  auto it = graph_pools_.find(id.id);
  if (it == graph_pools_.end()) {
    return;
  }
  auto& gp = it->second;
  if (gp.active_capture_count > 0) {
    gp.active_capture_count = 0u;
    gp.cancels += 1;
  }
  gc_pool_locked(id.id, gp);
}

void Allocator::mark_pool_replay_begin_(MempoolId id) {
  if (id.dev != dev_ || id.id == 0) {
    throw std::runtime_error("unknown mempool id");
  }
  MuLockGuard lg(mu_);
  auto it = graph_pools_.find(id.id);
  if (it == graph_pools_.end()) {
    throw std::runtime_error("unknown mempool id");
  }
  auto& gp = it->second;
  if (gp.active_capture_count > 0 || gp.active_replay_count > 0 || gp.prewarm_in_progress > 0) {
    std::string msg = "pool is busy with active capture";
    msg += " (refcnt=" + std::to_string(gp.refcnt) +
           ", active_capture_count=" + std::to_string(gp.active_capture_count) +
           ", active_replay_count=" + std::to_string(gp.active_replay_count) +
           ", prewarm_in_progress=" + std::to_string(gp.prewarm_in_progress) + ")";
    throw std::runtime_error(msg);
  }
  gp.active_replay_count = 1u;
}

void Allocator::mark_pool_replay_end_(MempoolId id) noexcept {
  if (id.dev != dev_ || id.id == 0) {
    return;
  }
  MuLockGuard lg(mu_);
  auto it = graph_pools_.find(id.id);
  if (it == graph_pools_.end()) {
#if defined(VBT_INTERNAL_TESTS)
    // In tests we may want to detect mismatched begin/end; silently ignore here.
#endif
    return;
  }
  auto& gp = it->second;
  if (gp.active_replay_count > 0) {
    gp.active_replay_count -= 1u;
  }
  gc_pool_locked(id.id, gp);
}

void Allocator::prewarm_graph_pool_for_stream_(MempoolId id,
                                                Stream stream,
                                                std::size_t min_total_bytes,
                                                int min_blocks) {
#if !VBT_WITH_CUDA
  (void)id; (void)stream; (void)min_total_bytes; (void)min_blocks;
  return;
#else
  if (id.dev != dev_ || id.id == 0) {
    throw std::runtime_error("cuda allocator: prewarm graph pool: unknown mempool id");
  }
  if (min_total_bytes == 0 || min_blocks <= 0) {
    return;
  }
  if (cfg_.backend == BackendKind::Async) {
    // Async backend uses its own capture semantics; no graph-pool pre-warm.
    return;
  }

  DeviceIndex sd = stream.device_index();
  if (sd < 0) sd = dev_;
  if (sd != dev_) {
    throw std::runtime_error("cuda allocator: prewarm graph pool: stream device mismatch");
  }
  if (streamCaptureStatus(stream) != CaptureStatus::None) {
    throw std::runtime_error("cuda allocator: prewarm graph pool: stream is capturing");
  }

  GraphPrivatePool* gp = nullptr;
  {
    MuLockGuard lg(mu_);
    auto it = graph_pools_.find(id.id);
    if (it == graph_pools_.end()) {
      throw std::runtime_error("cuda allocator: prewarm graph pool: unknown mempool id");
    }
    gp = &it->second;
  }

  // Local RAII guard to manage prewarm_in_progress and coordinate with capture/replay.
  class PrewarmEpochGuardLocal {
   public:
    PrewarmEpochGuardLocal(std::mutex& mu, GraphPrivatePool* gp)
        : mu_(mu), gp_(gp), engaged_(false) {
      Allocator::MuLockGuard lk(mu_);
      if (!gp_) {
        return;
      }
      if (gp_->active_capture_count > 0 || gp_->active_replay_count > 0 || gp_->prewarm_in_progress > 0) {
        throw std::runtime_error("pool is busy with active capture (prewarm)");
      }
      ++gp_->prewarm_in_progress;
      engaged_ = true;
    }

    ~PrewarmEpochGuardLocal() noexcept {
      if (!engaged_ || !gp_) return;
      Allocator::MuLockGuard lk(mu_);
      if (gp_->prewarm_in_progress > 0) {
        --gp_->prewarm_in_progress;
      }
    }

   private:
    std::mutex&       mu_;
    GraphPrivatePool* gp_;
    bool              engaged_;
  };

  PrewarmEpochGuardLocal epoch(mu_, gp);

  StreamId sid = stream.id();
  std::size_t have = 0;
  {
    MuLockGuard lg(mu_);
    auto it = per_stream_free_.find(sid);
    if (it != per_stream_free_.end()) {
      for (Block* b : it->second) {
        if (!b) continue;
        if (b->graph_pool_id != 0 && b->graph_pool_id != id.id) {
          continue;
        }
        have += b->size;
      }
    }
  }

  if (have >= min_total_bytes) {
    return;
  }

  std::size_t need = min_total_bytes - have;
  // Use allocator's rounding policy to pick a block size (~1 MiB by default).
  std::size_t block_bytes = round_size(1ull << 20);
  if (block_bytes == 0) {
    block_bytes = 1ull << 20;
  }
  int blocks_needed = static_cast<int>((need + block_bytes - 1) / block_bytes);
  if (blocks_needed < min_blocks) {
    blocks_needed = min_blocks;
  }

  std::vector<void*> new_ptrs;
  new_ptrs.reserve(static_cast<std::size_t>(blocks_needed));

  {
    DeviceGuard dg(dev_);
    for (int i = 0; i < blocks_needed; ++i) {
      void* p = nullptr;
      cudaError_t st = cudaMalloc_with_hook(&p, block_bytes);
      if (st != cudaSuccess) {
        for (void* q : new_ptrs) {
          (void)cudaFree(q);
        }
        throw std::runtime_error("cuda allocator: prewarm graph pool: cudaMalloc failed");
      }
      new_ptrs.push_back(p);
    }
  }

  {
    MuLockGuard lg(mu_);
    for (void* p : new_ptrs) {
      auto* b = new Block();
      b->device = dev_;
      b->alloc_stream = sid;
      b->owner_stream = sid;
      b->ptr = p;
      b->size = block_bytes;
      b->requested_size = block_bytes;
      b->allocated = false;
      b->mapped = false;
      b->prev = nullptr;
      b->next = nullptr;
      b->segment_head = true;
      b->is_split_tail = false;
      b->event_count = 0;
      b->graph_pool_id = id.id;
      by_ptr_[p] = b;
      insert_free_block_unlocked(b, sid);
      stats_.reserved_bytes_all_current += block_bytes;
      if (stats_.reserved_bytes_all_current > stats_.max_reserved_bytes_all) {
        stats_.max_reserved_bytes_all = stats_.reserved_bytes_all_current;
      }
      stats_.num_device_alloc += 1;
    }
  }
#endif
}

Allocator::AllocateToPoolGuard::AllocateToPoolGuard(Allocator* a, MempoolId id) noexcept
: alloc_(a), id_(id), engaged_(a != nullptr && id.is_valid()) {}

Allocator::AllocateToPoolGuard::~AllocateToPoolGuard() noexcept {
  if (engaged_ && alloc_) {
    alloc_->cancel_allocate_to_pool_(id_);
  }
}

Allocator::AllocateToPoolGuard::AllocateToPoolGuard(AllocateToPoolGuard&& other) noexcept
: alloc_(other.alloc_), id_(other.id_), engaged_(other.engaged_) {
  other.alloc_ = nullptr;
  other.id_ = MempoolId{};
  other.engaged_ = false;
}

Allocator::AllocateToPoolGuard& Allocator::AllocateToPoolGuard::operator=(AllocateToPoolGuard&& other) noexcept {
  if (this == &other) return *this;
  if (engaged_ && alloc_) {
    alloc_->cancel_allocate_to_pool_(id_);
  }
  alloc_ = other.alloc_;
  id_ = other.id_;
  engaged_ = other.engaged_;
  other.alloc_ = nullptr;
  other.id_ = MempoolId{};
  other.engaged_ = false;
  return *this;
}

void Allocator::AllocateToPoolGuard::end() noexcept {
  if (engaged_ && alloc_) {
    alloc_->end_allocate_to_pool_(id_);
    engaged_ = false;
  }
}

void Allocator::AllocateToPoolGuard::cancel() noexcept {
  if (engaged_ && alloc_) {
    alloc_->cancel_allocate_to_pool_(id_);
    engaged_ = false;
  }
}

void* Allocator::raw_alloc(std::size_t nbytes) {
#if VBT_WITH_CUDA
#ifdef VBT_INTERNAL_TESTS
  g_debug_raw_alloc_nostream_calls.fetch_add(1, std::memory_order_relaxed);
#endif
  if (cfg_.backend == BackendKind::Async) {
    // maintain capture restriction
    if (currentStreamCaptureStatus(dev_) == CaptureStatus::Active) {
      throw_allocator_capture_denied_device(dev_);
    }
    return AsyncBackend::get(dev_).raw_alloc(nbytes);
  }
  if (nbytes == 0) return nullptr;
  // Capture gating: check current stream capture status
  Stream cur = getCurrentStream(dev_);
  const bool capturing_now = (currentStreamCaptureStatus(dev_) == CaptureStatus::Active);
  const bool tls_ok = (s_capture_tls.active && s_capture_tls.dev == dev_);
  const bool routing_device_active = routing_active_flag_.load(std::memory_order_relaxed);
  const bool routing_active = capturing_now && tls_ok && routing_device_active;
  if (capturing_now && !tls_ok) {
    throw_allocator_capture_denied_device(dev_);
  }
  // Opportunistically reclaim ready blocks only when not actively routing during capture
  if (!routing_active) {
    process_events();
  }

  std::size_t rounded = round_size(nbytes);
  // Try to reuse from the current stream free list first, then cross-stream fallback if enabled
  /* cur already fetched */
  StreamId sid = cur.id();
  std::uint64_t target_pool_id = routing_active ? s_capture_tls.id.id : 0u;
  {
    MuLockGuard lg(mu_);
    if (Block* b = try_take_free_block_unlocked(sid, rounded, target_pool_id)) {
      on_reuse_from_free_list(b, nbytes, sid, target_pool_id);
      return b->ptr;
    }
  }

  // Cross-stream fallback (conditional)
  if (cfg_.enable_cross_stream_fallback) {
    MuLockGuard lg(mu_);
    const bool allow_cross_stream = (target_pool_id == 0u);
    if (allow_cross_stream) {
      if (Block* b = try_take_from_cross_stream_unlocked(rounded, target_pool_id)) {
        on_reuse_from_free_list(b, nbytes, sid, target_pool_id);
        return b->ptr;
      }
    }
  }

  // If capturing with routing, we cannot call cudaMalloc; deny
  if (routing_active) {
    throw_allocator_capture_denied_device(dev_);
  }

  // Native fraction cap gate (growth-only, runs before cudaMalloc).
  maybe_run_fraction_gate(nbytes, rounded);

  DeviceGuard g(dev_);
  void* p = nullptr;
  cudaError_t st = cudaMalloc_with_hook(&p, rounded);
  if (st != cudaSuccess) {
    {
      MuLockGuard lg(mu_);
      stats_.num_alloc_retries += 1;
    }

    // Opportunistically reclaim ready blocks from limbo.
    process_events();

    // Single GC pass targeting this allocation size.
    const std::size_t gc_target_bytes = rounded;
    (void)run_gc_pass_if_eligible(gc_target_bytes, GcReason::Oom);

    // GC-informed retry.
    p = nullptr;
    st = cudaMalloc_with_hook(&p, rounded);

    if (st != cudaSuccess) {
      // Additional configured retries: opportunistic reclaim + emptyCache + optional sleep.
      for (std::size_t attempt = 0;
           st != cudaSuccess && attempt < cfg_.oom_retry_count;
           ++attempt) {
        process_events();
        emptyCache();
        if (cfg_.oom_retry_sleep_ms > 0) {
          std::this_thread::sleep_for(
              std::chrono::milliseconds(cfg_.oom_retry_sleep_ms));
        }
        p = nullptr;
        st = cudaMalloc_with_hook(&p, rounded);
      }
      if (st != cudaSuccess) {
        // Final failure: throw informative OOM (message text unchanged).
        std::size_t alloc_snapshot = 0, reserv_snapshot2 = 0;
        {
          MuLockGuard lg(mu_);
          stats_.num_ooms += 1;
          alloc_snapshot = static_cast<std::size_t>(
              stats_.allocated_bytes_all_current);
          reserv_snapshot2 = static_cast<std::size_t>(
              stats_.reserved_bytes_all_current);
        }
        size_t freeB = 0, totalB = 0;
        DeviceGuard g2(dev_);
        (void)cudaMemGetInfo(&freeB, &totalB);
        std::size_t unalloc_reserved = 0;
        if (reserv_snapshot2 > alloc_snapshot) {
          unalloc_reserved = reserv_snapshot2 - alloc_snapshot;
        }
        auto to_mib_u = [](std::size_t b) {
          return static_cast<unsigned long long>(b >> 20);
        };
        throw std::runtime_error(
            std::string("CUDA out of memory. Tried to allocate ") +
            std::to_string(nbytes) + " bytes (rounded=" +
            std::to_string(rounded) + ", " +
            std::to_string(to_mib_u(rounded)) + " MiB). " +
            "GPU " + std::to_string(dev_) + " has a total capacity of " +
            std::to_string(totalB) + " bytes (" +
            std::to_string(to_mib_u(totalB)) + " MiB), of which " +
            std::to_string(freeB) + " bytes (" +
            std::to_string(to_mib_u(freeB)) +
            " MiB) is free. " +
            "Of the allocated memory " + std::to_string(alloc_snapshot) +
            " bytes (" + std::to_string(to_mib_u(alloc_snapshot)) +
            " MiB) is allocated by VibeTensor, and " +
            std::to_string(unalloc_reserved) + " bytes (" +
            std::to_string(to_mib_u(unalloc_reserved)) +
            " MiB) is reserved by VibeTensor but unallocated. If reserved "
            "memory is high and free memory is low, this may be due to "
            "fragmentation of cached blocks. Consider calling empty_cache() "
            "to release cached segments. private pools: 0.");
      }
    }
  }
  // Create and track the block/segment
  auto* b = new Block();
  b->device = dev_;
  b->alloc_stream = sid;
  b->owner_stream = sid;
  b->ptr = p;
  b->size = rounded;
  b->requested_size = nbytes;
  b->allocated = true;
  b->segment_head = true;
  b->is_split_tail = false;
  b->graph_pool_id = target_pool_id;
  try {
    MuLockGuard lg(mu_);
    by_ptr_[p] = b;
    active_blocks_.insert(b);
    stats_.reserved_bytes_all_current += rounded;
    if (stats_.reserved_bytes_all_current > stats_.max_reserved_bytes_all) stats_.max_reserved_bytes_all = stats_.reserved_bytes_all_current;
    stats_.allocated_bytes_all_current += rounded;
    if (stats_.allocated_bytes_all_current > stats_.max_allocated_bytes_all) stats_.max_allocated_bytes_all = stats_.allocated_bytes_all_current;
    stats_.requested_bytes_all_current += b->requested_size;
    if (stats_.requested_bytes_all_current > stats_.max_requested_bytes_all) stats_.max_requested_bytes_all = stats_.requested_bytes_all_current;
    stats_.num_device_alloc += 1;
  } catch (...) {
    {
      MuLockGuard lg(mu_);
      by_ptr_.erase(p);
    }
    DeviceGuard g2(dev_);
    (void)cudaFree(p);
    delete b;
    throw;
  }
  return p;
#else
  (void)nbytes; return nullptr;
#endif
}

void* Allocator::raw_alloc(std::size_t nbytes, Stream s) {
#if VBT_WITH_CUDA
#ifdef VBT_INTERNAL_TESTS
  g_debug_raw_alloc_stream_calls.fetch_add(1, std::memory_order_relaxed);
#endif
  if (cfg_.backend == BackendKind::Async) {
    DeviceIndex sd = s.device_index();
    if (sd < 0) sd = dev_;
    if (sd != dev_) {
      throw std::runtime_error(std::string("allocator_device=") + std::to_string(dev_) +
                               " stream_device=" + std::to_string(sd) +
                               " nbytes=" + std::to_string(nbytes));
    }
    if (streamCaptureStatus(s) == CaptureStatus::Active) {
      throw_allocator_capture_denied_stream(dev_, s.id());
    }
    return AsyncBackend::get(dev_).raw_alloc(nbytes, s);
  }
  if (nbytes == 0) return nullptr;
  DeviceIndex sd = s.device_index();
  if (sd < 0) sd = dev_;
  if (sd != dev_) {
    throw std::runtime_error(std::string("allocator_device=") + std::to_string(dev_) +
                             " stream_device=" + std::to_string(sd) +
                             " nbytes=" + std::to_string(nbytes));
  }
  const bool capturing_now = (streamCaptureStatus(s) == CaptureStatus::Active);
  const bool tls_ok = (s_capture_tls.active && s_capture_tls.dev == dev_);
  const bool routing_device_active = routing_active_flag_.load(std::memory_order_relaxed);
  const bool routing_active = capturing_now && tls_ok && routing_device_active;
  // Guard: allocations forbidden during CUDA graph capture on this stream unless routing active (TLS)
  if (capturing_now && !tls_ok) {
    throw_allocator_capture_denied_stream(dev_, s.id());
  }
  // Opportunistically reclaim ready blocks only when not routing during capture
  if (!routing_active) {
    process_events();
  }

  std::size_t rounded = round_size(nbytes);
  StreamId sid = s.id();
  std::uint64_t target_pool_id = routing_active ? s_capture_tls.id.id : 0u;
  {
    MuLockGuard lg(mu_);
    if (Block* b = try_take_free_block_unlocked(sid, rounded, target_pool_id)) {
      on_reuse_from_free_list(b, nbytes, sid, target_pool_id);
      return b->ptr;
    }
  }

  // Cross-stream fallback (conditional)
  if (cfg_.enable_cross_stream_fallback) {
    MuLockGuard lg(mu_);
    const bool allow_cross_stream = (target_pool_id == 0u);
    if (allow_cross_stream) {
      if (Block* b = try_take_from_cross_stream_unlocked(rounded, target_pool_id)) {
        on_reuse_from_free_list(b, nbytes, sid, target_pool_id);
        return b->ptr;
      }
    }
  }

  // If capturing with routing, we cannot call cudaMalloc; deny
  if (routing_active) {
    throw_allocator_capture_denied_stream(dev_, s.id());
  }

  // Native fraction cap gate (growth-only, runs before cudaMalloc).
  maybe_run_fraction_gate(nbytes, rounded);

  // Fallback: cudaMalloc a new block
  DeviceGuard g(dev_);
  void* p = nullptr;
  cudaError_t st = cudaMalloc_with_hook(&p, rounded);
  if (st != cudaSuccess) {
    {
      MuLockGuard lg(mu_);
      stats_.num_alloc_retries += 1;
    }

    process_events();
    const std::size_t gc_target_bytes = rounded;
    (void)run_gc_pass_if_eligible(gc_target_bytes, GcReason::Oom);

    p = nullptr;
    st = cudaMalloc_with_hook(&p, rounded);

    if (st != cudaSuccess) {
      // Additional configured retries: opportunistic reclaim + emptyCache + optional sleep.
      for (std::size_t attempt = 0;
           st != cudaSuccess && attempt < cfg_.oom_retry_count;
           ++attempt) {
        process_events();
        emptyCache();
        if (cfg_.oom_retry_sleep_ms > 0) {
          std::this_thread::sleep_for(
              std::chrono::milliseconds(cfg_.oom_retry_sleep_ms));
        }
        p = nullptr;
        st = cudaMalloc_with_hook(&p, rounded);
      }
      if (st != cudaSuccess) {
        std::size_t alloc_snapshot = 0, reserv_snapshot2 = 0;
        {
          MuLockGuard lg(mu_);
          stats_.num_ooms += 1;
          alloc_snapshot = static_cast<std::size_t>(
              stats_.allocated_bytes_all_current);
          reserv_snapshot2 = static_cast<std::size_t>(
              stats_.reserved_bytes_all_current);
        }
        size_t freeB = 0, totalB = 0;
        (void)cudaMemGetInfo(&freeB, &totalB);
        std::size_t unalloc_reserved = 0;
        if (reserv_snapshot2 > alloc_snapshot) {
          unalloc_reserved = reserv_snapshot2 - alloc_snapshot;
        }
        auto to_mib_u = [](std::size_t b) {
          return static_cast<unsigned long long>(b >> 20);
        };
        throw std::runtime_error(
            std::string("CUDA out of memory. Tried to allocate ") +
            std::to_string(nbytes) + " bytes (rounded=" +
            std::to_string(rounded) + ", " +
            std::to_string(to_mib_u(rounded)) + " MiB). " +
            "GPU " + std::to_string(dev_) + " has a total capacity of " +
            std::to_string(totalB) + " bytes (" +
            std::to_string(to_mib_u(totalB)) + " MiB), of which " +
            std::to_string(freeB) + " bytes (" +
            std::to_string(to_mib_u(freeB)) +
            " MiB) is free. " +
            "Of the allocated memory " + std::to_string(alloc_snapshot) +
            " bytes (" + std::to_string(to_mib_u(alloc_snapshot)) +
            " MiB) is allocated by VibeTensor, and " +
            std::to_string(unalloc_reserved) + " bytes (" +
            std::to_string(to_mib_u(unalloc_reserved)) +
            " MiB) is reserved by VibeTensor but unallocated. If reserved "
            "memory is high and free memory is low, this may be due to "
            "fragmentation of cached blocks. Consider calling empty_cache() "
            "to release cached segments. private pools: 0.");
      }
    }
  }
  auto* b = new Block();
  b->device = dev_;
  b->alloc_stream = sid;
  b->owner_stream = sid;
  b->ptr = p;
  b->size = rounded;
  b->requested_size = nbytes;
  b->allocated = true;
  b->segment_head = true;
  b->is_split_tail = false;
  b->graph_pool_id = target_pool_id;
  try {
    MuLockGuard lg(mu_);
    by_ptr_[p] = b;
    active_blocks_.insert(b);
    stats_.reserved_bytes_all_current += rounded;
    if (stats_.reserved_bytes_all_current > stats_.max_reserved_bytes_all) stats_.max_reserved_bytes_all = stats_.reserved_bytes_all_current;
    stats_.allocated_bytes_all_current += rounded;
    if (stats_.allocated_bytes_all_current > stats_.max_allocated_bytes_all) stats_.max_allocated_bytes_all = stats_.allocated_bytes_all_current;
    stats_.requested_bytes_all_current += b->requested_size;
    if (stats_.requested_bytes_all_current > stats_.max_requested_bytes_all) stats_.max_requested_bytes_all = stats_.requested_bytes_all_current;
    stats_.num_device_alloc += 1;
  } catch (...) {
    {
      MuLockGuard lg(mu_);
      by_ptr_.erase(p);
    }
    DeviceGuard g2(dev_);
    (void)cudaFree(p);
    delete b;
    throw;
  }
  return p;
#else
  (void)nbytes; (void)s; return nullptr;
#endif
}

void  Allocator::raw_delete(void* ptr) noexcept {
#if VBT_WITH_CUDA
  if (cfg_.backend == BackendKind::Async) { AsyncBackend::get(dev_).raw_delete(ptr); return; }
  if (!ptr) return;
  // Look up tracking block
  Block* b = nullptr;
  {
    MuLockGuard lg(mu_);
    auto it = by_ptr_.find(ptr);
    if (it == by_ptr_.end()) {
      // Unknown pointer: ignore
      return;
    }
    b = it->second;
  }

  // Capture freeing stream id before locking
  Stream owner_stream = getCurrentStream(dev_);
  StreamId owner_sid_new = owner_stream.id();

  std::vector<StreamId> snapshot;
  bool immediate_reuse = false;
  StreamId prev_owner_sid = 0;
  {
    MuLockGuard lg(mu_);
    prev_owner_sid = b->owner_stream;
    if (!b->allocated) {
      // Double free or already freed; no-op
      return;
    }
    b->allocated = false;
    b->owner_stream = owner_sid_new;
    // Build snapshot excluding owner stream and include previous owner if different
    snapshot.reserve(b->stream_uses.size() + 1);
    for (auto sid : b->stream_uses) if (sid != owner_sid_new) snapshot.push_back(sid);
    if (prev_owner_sid != owner_sid_new) { snapshot.push_back(prev_owner_sid); stats_.num_prev_owner_fences += 1; }
    b->stream_uses.clear();
    // Update stats on free
    if (b->size > 0 && stats_.allocated_bytes_all_current >= b->size) {
      stats_.allocated_bytes_all_current -= b->size;
    }
    if (b->requested_size > 0 &&
        stats_.requested_bytes_all_current >= b->requested_size) {
      stats_.requested_bytes_all_current -= b->requested_size;
    }
    if (snapshot.empty() && b->event_count == 0 && prev_owner_sid == owner_sid_new) {
      // Same-stream free: coalesce and immediate reinsertion
      active_blocks_.erase(b);
      Block* ins = coalesce_neighbors_unlocked(b);
      insert_free_block_unlocked(ins, owner_sid_new);
      // Avoid immediate cross-stream reuse for same-stream free
      auto itc = cross_stream_free_.find(ins);
      if (itc != cross_stream_free_.end()) cross_stream_free_.erase(itc);
      immediate_reuse = true;
    }
  }
  if (immediate_reuse) {
    if (cfg_.process_events_every_frees > 0 && !s_capture_tls.active) {
      auto n = ++free_count_;
      if (n % cfg_.process_events_every_frees == 0) process_events();
    }
    return;
  }

  // If any referenced stream (owner or recorded streams) is capturing, defer
  bool any_capturing = is_stream_capturing(dev_, owner_sid_new);
  if (!any_capturing) {
    for (auto sid : snapshot) { if (is_stream_capturing(dev_, sid)) { any_capturing = true; break; } }
  }
  if (any_capturing) {
    {
      MuLockGuard lg(mu_);
      deferred_.push_back(DeferredFree{b, owner_sid_new, snapshot});
    }
    if (cfg_.process_events_every_frees > 0 && !s_capture_tls.active) {
      auto n = ++free_count_;
      if (n % cfg_.process_events_every_frees == 0) process_events();
    }
    return;
  }

  struct Pending { StreamId sid; PooledEvent ev; };
  std::vector<Pending> pendings;
  pendings.reserve(snapshot.size());
  bool failure = false;
  {
    DeviceGuard dg(dev_);
    for (auto sid : snapshot) {
      PooledEvent e = events_.get();
      if (!e.valid()) { failure = true; break; }
      // Record event on the given stream handle (sid encodes handle())
      cudaError_t st = cudaEventRecord(reinterpret_cast<cudaEvent_t>(e.raw()), reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(sid)));
      if (st != cudaSuccess) { failure = true; break; }
      pendings.push_back(Pending{sid, std::move(e)});
    }
  }
  if (failure) {
    // Roll back any created events (destroy to avoid pooling recorded events)
    for (auto& p : pendings) { events_.destroy(std::move(p.ev)); }
    // Restore state under lock: revert free attempt so caller may retry later
    {
      MuLockGuard lg(mu_);
      b->allocated = true;
      b->owner_stream = prev_owner_sid;
      for (auto sid : snapshot) b->stream_uses.insert(sid);
      // Re-add stats since we reverted
      if (b->size > 0) {
        stats_.allocated_bytes_all_current += b->size;
        if (stats_.allocated_bytes_all_current > stats_.max_allocated_bytes_all) {
          stats_.max_allocated_bytes_all = stats_.allocated_bytes_all_current;
        }
      }
      if (b->requested_size > 0) {
        stats_.requested_bytes_all_current += b->requested_size;
        if (stats_.requested_bytes_all_current > stats_.max_requested_bytes_all) {
          stats_.max_requested_bytes_all = stats_.requested_bytes_all_current;
        }
      }
    }
    if (cfg_.process_events_every_frees > 0 && !s_capture_tls.active) {
      auto n = ++free_count_;
      if (n % cfg_.process_events_every_frees == 0) process_events();
    }
    return;
  }

  {
    MuLockGuard lg(mu_);
    for (auto& p : pendings) {
      b->event_count += 1;
      limbo_[p.sid].push_back(LimboEntry{limbo_token_seq_++, std::move(p.ev), b});
    }
    // Keep block in active_blocks_ until event_count drains to 0 in process_events()
  }
  if (cfg_.process_events_every_frees > 0 && !s_capture_tls.active) {
    auto n = ++free_count_;
    if (n % cfg_.process_events_every_frees == 0) process_events();
  }
#else
  (void)ptr;
#endif
}

void  Allocator::record_stream(void* ptr, Stream s) noexcept {
#if VBT_WITH_CUDA
  if (cfg_.backend == BackendKind::Async) { AsyncBackend::get(dev_).record_stream(ptr, s); return; }
  if (!ptr) return;
  // Only metadata under lock; ignore if not tracked/allocated or devices mismatch
  MuLockGuard lg(mu_);
  auto it = by_ptr_.find(ptr);
  if (it == by_ptr_.end()) return;
  Block* b = it->second;
  if (!b->allocated) return;
  if (s.device_index() != dev_) return; // cross-device no-op
  StreamId sid = s.id();
  if (sid == b->alloc_stream || sid == b->owner_stream) return; // no-op
  b->stream_uses.insert(sid);
#else
  (void)ptr; (void)s;
#endif
}

void  Allocator::process_events(int max_pops) noexcept {
#if VBT_WITH_CUDA
  if (cfg_.backend == BackendKind::Async) { AsyncBackend::get(dev_).process_events(max_pops); return; }
  // First, opportunistically flush deferred frees that are no longer capturing
  std::vector<DeferredFree> cands;
  {
    MuLockGuard lg(mu_);
    cands.assign(deferred_.begin(), deferred_.end());
  }
  for (const auto& df : cands) {
    bool capturing = is_stream_capturing(dev_, df.owner_sid);
    if (!capturing) {
      for (auto sid : df.streams) { if (is_stream_capturing(dev_, sid)) { capturing = true; break; } }
    }
    if (capturing) continue;
    { MuLockGuard lg(mu_); stats_.deferred_flush_attempts += 1; }
    // Record events off-lock for this deferred free
    struct Pending { StreamId sid; PooledEvent ev; };
    std::vector<Pending> pendings; pendings.reserve(df.streams.size());
    bool failure = false;
    {
      DeviceGuard dg(dev_);
      for (auto sid : df.streams) {
        PooledEvent e = events_.get();
        if (!e.valid()) { failure = true; break; }
        cudaError_t st = cudaEventRecord(reinterpret_cast<cudaEvent_t>(e.raw()), reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(sid)));
        if (st != cudaSuccess) { failure = true; break; }
        pendings.push_back(Pending{sid, std::move(e)});
      }
    }
    if (failure) {
      for (auto& p : pendings) { events_.destroy(std::move(p.ev)); }
      continue; // try later
    }
    // Enqueue and remove from deferred_ under lock
    {
      MuLockGuard lg(mu_);
      for (auto it = deferred_.begin(); it != deferred_.end(); ++it) {
        if (it->b == df.b) { deferred_.erase(it); break; }
      }
      for (auto& p : pendings) {
        df.b->event_count += 1;
        limbo_[p.sid].push_back(LimboEntry{limbo_token_seq_++, std::move(p.ev), df.b});
      }
      stats_.deferred_flush_successes += 1;
    }
  }

  // Collect keys to avoid holding lock during iteration
  std::vector<StreamId> keys;
  {
    MuLockGuard lg(mu_);
    keys.reserve(limbo_.size());
    for (auto& kv : limbo_) if (!kv.second.empty()) keys.push_back(kv.first);
  }
  int pops = 0;
  for (auto sid : keys) {
    // Skip queries during capture on this stream
    if (is_stream_capturing(dev_, sid)) continue;
    bool progress = true;
    while (progress) {
      if (max_pops >= 0 && pops >= max_pops) return;
      // Peek head under lock
      uint64_t token = 0; void* ev_raw = nullptr; Block* b = nullptr;
      {
        MuLockGuard lg(mu_);
        auto it = limbo_.find(sid);
        if (it == limbo_.end() || it->second.empty()) { progress = false; break; }
        token = it->second.front().token;
        ev_raw = it->second.front().ev.raw();
        b = it->second.front().b;
      }
      // Off-lock query
      bool ready = false;
      {
        DeviceGuard dg(dev_);
        cudaError_t st = cudaEventQuery(reinterpret_cast<cudaEvent_t>(ev_raw));
        if (st == cudaSuccess) ready = true;
        else if (st == cudaErrorNotReady) { (void)cudaGetLastError(); ready = false; }
        else { ready = false; }
      }
      if (!ready) break; // head-of-line not ready

      // Pop and reclaim under lock
      PooledEvent ev_ret; StreamId owner_sid = 0; Block* to_insert = nullptr;
      {
        MuLockGuard lg(mu_);
        auto it = limbo_.find(sid);
        if (it == limbo_.end() || it->second.empty()) { progress = false; break; }
        auto& dq = it->second;
        if (!(dq.front().token == token && dq.front().b == b)) {
          // Head changed; retry
          continue;
        }
        auto entry = std::move(dq.front());
        dq.pop_front();
        if (dq.empty()) limbo_.erase(it);
        ev_ret = std::move(entry.ev);
        if (b->event_count > 0) --b->event_count;
        if (b->event_count == 0 && !b->allocated) {
          // Coalesce neighbors before reinsertion
          active_blocks_.erase(b);
          Block* merged = coalesce_neighbors_unlocked(b);
          owner_sid = merged->owner_stream;
          insert_free_block_unlocked(merged, owner_sid);
        }
      }
      // Return event off-lock
      events_.put(std::move(ev_ret));
      ++pops;
    }
  }
#else
  (void)max_pops;
#endif
}

#if VBT_WITH_CUDA
std::vector<Allocator::Block*>
Allocator::find_candidate_heads_locked() noexcept {
  std::vector<Block*> heads;
  std::unordered_set<Block*> seen_heads;

  for (auto& kv : per_stream_free_) {
    auto& free_set = kv.second;
    for (Block* b : free_set) {
      if (!b) continue;

      Block* h = b;
      while (h->prev) {
        h = h->prev;
      }
      if (!h->segment_head || h->prev != nullptr) {
        continue;
      }
      if (!seen_heads.insert(h).second) {
        continue;
      }

      bool ok = true;
      for (Block* cur = h; cur != nullptr; cur = cur->next) {
        if (cur->graph_pool_id != 0u || cur->allocated || cur->event_count != 0) {
          ok = false;
          break;
        }
      }
      if (!ok) {
        continue;
      }

      heads.push_back(h);
    }
  }

  return heads;
}

void Allocator::detach_segment_for_gc_locked(
    Block*               head,
    std::vector<void*>&  free_ptrs,
    std::vector<Block*>& delete_blocks,
    std::size_t&         freed_bytes) noexcept {
  if (!head) return;

#ifndef NDEBUG
  assert(head->segment_head && head->prev == nullptr);
  for (Block* cur = head; cur != nullptr; cur = cur->next) {
    assert(cur->graph_pool_id == 0u);
    assert(!cur->allocated);
    assert(cur->event_count == 0);
  }
#endif

  free_ptrs.push_back(head->ptr);

  for (Block* cur = head; cur != nullptr; ) {
    Block* next = cur->next;

    remove_from_free_indices_unlocked(cur);

    if (cur->is_split_tail) {
      if (stats_.inactive_split_blocks_all > 0) {
        stats_.inactive_split_blocks_all -= 1;
      }
      if (stats_.inactive_split_bytes_all >= cur->size) {
        stats_.inactive_split_bytes_all -= cur->size;
      }
      cur->is_split_tail = false;
    }

    auto it = by_ptr_.find(cur->ptr);
    if (it != by_ptr_.end()) {
      by_ptr_.erase(it);
    }

    active_blocks_.erase(cur);

    freed_bytes += cur->size;
    delete_blocks.push_back(cur);

    cur = next;
  }
}
#endif

std::size_t Allocator::run_gc_pass_if_eligible(std::size_t gc_target_bytes,
                                               GcReason    reason) noexcept {
#if !VBT_WITH_CUDA
  (void)gc_target_bytes;
  (void)reason;
  return 0;
#else
#ifdef VBT_INTERNAL_TESTS
  debug_assert_mu_not_held_for_gc();
  DebugMuDepthExitGuard depth_guard;
#endif

  if (cfg_.backend != BackendKind::Native) {
    return 0;
  }
  if (gc_target_bytes == 0) {
    return 0;
  }

  if (reason == GcReason::Oom && cfg_.garbage_collection_threshold <= 0.0) {
    return 0;
  }

  std::vector<void*>  free_ptrs;
  std::vector<Block*> delete_blocks;
  std::size_t         reclaimed_planned = 0;
  std::size_t         freed_bytes       = 0;

  {
    MuLockGuard lg(mu_);

    const std::size_t reserved_snapshot =
        static_cast<std::size_t>(stats_.reserved_bytes_all_current);
    if (reserved_snapshot == 0) {
      return 0;
    }

    std::vector<Block*> heads = find_candidate_heads_locked();
    if (heads.empty()) {
      return 0;
    }

    struct GcCandidateLocal {
      Block*        head{nullptr};
      std::size_t   seg_bytes{0};
      std::uint32_t age{0};
    };

    std::vector<GcCandidateLocal> candidates;
    candidates.reserve(heads.size());

    std::size_t inactive_total = 0;

    for (Block* h : heads) {
      if (!h) continue;
      std::size_t seg_bytes = 0;
      bool        ok        = true;
      for (Block* cur = h; cur != nullptr; cur = cur->next) {
        if (cur->graph_pool_id != 0u || cur->allocated || cur->event_count != 0) {
          ok = false;
          break;
        }
        seg_bytes += cur->size;
      }
      if (!ok || seg_bytes == 0) {
        continue;
      }
      inactive_total += seg_bytes;
      candidates.push_back(GcCandidateLocal{h, seg_bytes, 0});
    }

    if (inactive_total == 0 || candidates.empty()) {
      return 0;
    }

    if (reason == GcReason::Oom &&
        cfg_.garbage_collection_threshold > 0.0) {
      double frac = static_cast<double>(inactive_total) /
                    static_cast<double>(reserved_snapshot);
      if (frac < cfg_.garbage_collection_threshold) {
        return 0;
      }
    }

    std::uint64_t age_sum = 0;
    for (auto& c : candidates) {
      Block* h = c.head;
      if (!h) continue;
      if (h->gc_age != std::numeric_limits<std::uint32_t>::max()) {
        ++h->gc_age;
      }
      c.age = h->gc_age;
      age_sum += static_cast<std::uint64_t>(c.age);
    }

    if (candidates.empty()) {
      return 0;
    }

    const std::uint32_t avg_age = static_cast<std::uint32_t>(
        age_sum / static_cast<std::uint64_t>(candidates.size()));

    std::sort(candidates.begin(), candidates.end(),
              [](const GcCandidateLocal& a, const GcCandidateLocal& b) {
                if (a.age != b.age) return a.age > b.age;
                if (a.seg_bytes != b.seg_bytes) return a.seg_bytes > b.seg_bytes;
                return a.head < b.head;
              });

    const std::size_t MAX    = std::numeric_limits<std::size_t>::max();
    const std::size_t budget = (gc_target_bytes > MAX) ? MAX : gc_target_bytes;

    std::size_t inactive_remaining = inactive_total;
    std::size_t reserved_after     = reserved_snapshot;

    for (const auto& c : candidates) {
      if (!c.head || c.seg_bytes == 0) continue;
      if (c.age < avg_age) continue;
      if (reclaimed_planned >= budget) break;

      const std::size_t seg = c.seg_bytes;
      if (seg > MAX - reclaimed_planned) {
        reclaimed_planned = budget;
      } else {
        std::size_t sum = reclaimed_planned + seg;
        reclaimed_planned = (sum > budget) ? budget : sum;
      }

      if (inactive_remaining > seg) {
        inactive_remaining -= seg;
      } else {
        inactive_remaining = 0;
      }
      if (reserved_after > seg) {
        reserved_after -= seg;
      } else {
        reserved_after = 0;
      }

      detach_segment_for_gc_locked(c.head, free_ptrs, delete_blocks, freed_bytes);

      if (reason == GcReason::Oom &&
          cfg_.garbage_collection_threshold > 0.0) {
        if (reserved_after == 0) {
          break;
        }
        double frac_after = static_cast<double>(inactive_remaining) /
                            static_cast<double>(reserved_after);
        if (frac_after < cfg_.garbage_collection_threshold) {
          break;
        }
      }
    }

    if (free_ptrs.empty() || freed_bytes == 0) {
      return 0;
    }

    if (stats_.reserved_bytes_all_current >= freed_bytes) {
      stats_.reserved_bytes_all_current -= freed_bytes;
    } else {
      stats_.reserved_bytes_all_current = 0;
    }

    stats_.gc_passes          += 1;
    stats_.gc_reclaimed_bytes += freed_bytes;
    stats_.num_device_free    += static_cast<std::uint64_t>(free_ptrs.size());
  }

  {
    DeviceGuard dg(dev_);
    for (void* p : free_ptrs) {
      if (!p) continue;
      (void)cudaFree(p);
    }
  }

  for (Block* b : delete_blocks) {
    delete b;
  }

  return reclaimed_planned;
#endif
}

void  Allocator::emptyCache() noexcept {
#if VBT_WITH_CUDA
  if (cfg_.backend == BackendKind::Async) {
    AsyncBackend::get(dev_).emptyCache();
    return;
  }

  // Trim the EventPool first; this does not touch allocator stats.
  events_.empty_cache();

  std::vector<void*>  free_ptrs;
  std::vector<Block*> delete_blocks;
  std::size_t         freed_bytes = 0;

  {
    MuLockGuard lg(mu_);

    // Discover global, fully idle segment heads using the GC helper.
    std::vector<Block*> heads = find_candidate_heads_locked();
    if (heads.empty()) {
      return;
    }

    // Detach all candidate segments and accumulate freed bytes.
    for (Block* h : heads) {
      if (!h) continue;
      detach_segment_for_gc_locked(h, free_ptrs, delete_blocks, freed_bytes);
    }

    // Saturating decrement of reserved bytes.
    if (freed_bytes > 0) {
      if (stats_.reserved_bytes_all_current >= freed_bytes) {
        stats_.reserved_bytes_all_current -= freed_bytes;
      } else {
        stats_.reserved_bytes_all_current = 0;
      }
    }

    // Count segments freed (one per base pointer).
    stats_.num_device_free += static_cast<std::uint64_t>(free_ptrs.size());

    // IMPORTANT: do not touch gc_passes or gc_reclaimed_bytes here.
  }

  // Off-lock cudaFree and block deletion.
  if (!free_ptrs.empty()) {
    DeviceGuard g(dev_);
    for (void* p : free_ptrs) {
      if (!p) continue;
      (void)cudaFree(p);
    }
  }

  for (Block* b : delete_blocks) {
    delete b;
  }
#endif
}

std::size_t Allocator::debug_cached_segments() const noexcept {
  MuLockGuard lg(mu_);
  std::size_t n = 0;
  for (auto& kv : per_stream_free_) n += kv.second.size();
  return n;
}

DeviceStats Allocator::getDeviceStats() const {
  if (cfg_.backend == BackendKind::Async) {
#if VBT_WITH_CUDA
    return AsyncBackend::get(dev_).getDeviceStats();
#else
    return DeviceStats{};
#endif
  }
  MuLockGuard lg(mu_);
  return stats_;
}

void Allocator::resetPeakStats() noexcept {
  if (cfg_.backend == BackendKind::Async) {
#if VBT_WITH_CUDA
    AsyncBackend::get(dev_).resetPeakStats();
    return;
#endif
  }
  MuLockGuard lg(mu_);
  stats_.max_allocated_bytes_all = stats_.allocated_bytes_all_current;
  stats_.max_reserved_bytes_all = stats_.reserved_bytes_all_current;
  stats_.max_requested_bytes_all = stats_.requested_bytes_all_current;
}

void Allocator::resetAccumulatedStats() noexcept {
  if (cfg_.backend == BackendKind::Async) {
#if VBT_WITH_CUDA
    AsyncBackend::get(dev_).resetAccumulatedStats();
    return;
#endif
  }
  MuLockGuard lg(mu_);
  // Zero accumulated counters; preserve peaks and current gauges
  stats_.num_alloc_retries = 0;
  stats_.num_ooms = 0;
  stats_.num_device_alloc = 0;
  stats_.num_device_free = 0;
  stats_.tolerance_fills_count = 0;
  stats_.tolerance_fills_bytes = 0;
  stats_.deferred_flush_attempts = 0;
  stats_.deferred_flush_successes = 0;
  stats_.num_prev_owner_fences = 0;

  // become non-zero again between resets on native backends.
  stats_.fraction_cap_breaches = 0;
  stats_.fraction_cap_misfires = 0;
  stats_.gc_passes = 0;
  stats_.gc_reclaimed_bytes = 0;
}

bool Allocator::owns(const void* ptr) const noexcept {
#if VBT_WITH_CUDA
  if (cfg_.backend == BackendKind::Async) { return AsyncBackend::get(dev_).owns(ptr); }
  if (!ptr) return false;
  MuLockGuard lg(mu_);
  return by_ptr_.find(const_cast<void*>(ptr)) != by_ptr_.end();
#else
  (void)ptr; return false;
#endif
}

void* Allocator::getBaseAllocation(void* ptr, std::size_t* size) const noexcept {
#if VBT_WITH_CUDA
  if (cfg_.backend == BackendKind::Async) { return AsyncBackend::get(dev_).getBaseAllocation(ptr, size); }
  if (size) *size = 0;
  if (!ptr) return nullptr;
  MuLockGuard lg(mu_);
  auto it = by_ptr_.find(ptr);
  if (it == by_ptr_.end()) return nullptr;
  Block* b = it->second;
  // Walk to segment head
  while (b->prev) b = b->prev;
  void* base = b->ptr;
  // Sum full segment size (coalesced siblings)
  std::size_t seg_size = 0;
  for (Block* cur = b; cur != nullptr; cur = cur->next) {
    seg_size += cur->size;
  }
  if (size) *size = seg_size;
  return base;
#else
  (void)ptr; (void)size; return nullptr;
#endif
}

#ifdef VBT_INTERNAL_TESTS
std::vector<testonly::GraphPoolDebugInfo>
Allocator::debug_graph_pools_for_testing() const {
  std::vector<testonly::GraphPoolDebugInfo> out;
#if !VBT_WITH_CUDA
  return out;
#else
  MuLockGuard lg(mu_);
  for (const auto& kv : graph_pools_) {
    testonly::GraphPoolDebugInfo info;
    info.id = MempoolId{dev_, kv.first};
    info.refcnt = kv.second.refcnt;
    info.active_capture_count = kv.second.active_capture_count;
    info.active_replay_count = kv.second.active_replay_count;
    info.prewarm_in_progress = kv.second.prewarm_in_progress;
    out.push_back(info);
  }
  return out;
#endif
}

std::vector<MemorySegmentSnapshot>
Allocator::debug_allocator_snapshot_for_testing(DeviceIndex dev) const {
#if !VBT_WITH_CUDA
  (void)dev;
  return {};
#else
  if (dev < 0 || dev >= device_count()) {
    std::fprintf(stderr,
                 "[vbt::cuda::Allocator] allocator_debug_snapshot: invalid device %d\n",
                 static_cast<int>(dev));
    std::abort();
  }

  auto snaps = snapshot_memory_segments(dev);
  for (const auto& seg : snaps) {
    if (seg.device != dev) {
      std::fprintf(stderr,
                   "[vbt::cuda::Allocator] allocator_debug_snapshot: segment device mismatch (expected %d, got %d)\n",
                   static_cast<int>(dev),
                   static_cast<int>(seg.device));
      std::abort();
    }
  }
  return snaps;
#endif
}

void Allocator::debug_gc_pool_now_for_testing(MempoolId id) {
#if !VBT_WITH_CUDA
  (void)id;
#else
  if (id.dev != dev_ || id.id == 0) {
    std::fprintf(stderr,
                 "[vbt::cuda::Allocator] allocator_debug_gc_pool_now: illegal state (invalid id dev=%d id=%llu)\n",
                 static_cast<int>(id.dev),
                 static_cast<unsigned long long>(id.id));
    std::abort();
  }

  MuLockGuard lg(mu_);
  auto it = graph_pools_.find(id.id);
  if (it == graph_pools_.end()) {
    std::fprintf(stderr,
                 "[vbt::cuda::Allocator] allocator_debug_gc_pool_now: illegal state (unknown id=%llu)\n",
                 static_cast<unsigned long long>(id.id));
    std::abort();
  }

  auto& gp = it->second;
  if (gp.refcnt != 0 ||
      gp.active_capture_count > 0 ||
      gp.active_replay_count > 0 ||
      gp.prewarm_in_progress > 0) {
    std::fprintf(stderr,
                 "[vbt::cuda::Allocator] allocator_debug_gc_pool_now: illegal state (refcnt=%u, active_capture_count=%u, active_replay_count=%u, prewarm_in_progress=%u)\n",
                 gp.refcnt,
                 gp.active_capture_count,
                 gp.active_replay_count,
                 gp.prewarm_in_progress);
    std::abort();
  }

  gc_pool_locked(id.id, gp);
#endif
}

std::uint64_t Allocator::debug_block_pool_id(void* ptr) const noexcept {
  MuLockGuard lg(mu_);
  auto it = by_ptr_.find(ptr);
  if (it == by_ptr_.end()) return 0u;
  return it->second ? it->second->graph_pool_id : 0u;
}

bool Allocator::debug_tls_routing_active() const noexcept {
  return s_capture_tls.active && s_capture_tls.dev == dev_;
}

bool Allocator::debug_device_routing_active() const noexcept {
  return routing_active_flag_.load(std::memory_order_relaxed);
}

std::size_t Allocator::debug_deferred_size_for_testing() const {
  MuLockGuard lg(mu_);
  return deferred_.size();
}

bool Allocator::debug_is_in_free_list_for_testing(const void* ptr) const noexcept {
  if (!ptr) {
    return false;
  }
  MuLockGuard lg(mu_);
  auto it = by_ptr_.find(const_cast<void*>(ptr));
  if (it == by_ptr_.end() || !it->second) {
    return false;
  }
  Block* b = it->second;
  for (const auto& kv : per_stream_free_) {
    const auto& free_set = kv.second;
    if (free_set.find(b) != free_set.end()) {
      return true;
    }
  }
  return cross_stream_free_.find(b) != cross_stream_free_.end();
}

std::vector<void*> Allocator::debug_tracked_block_ptrs() const {
  MuLockGuard lg(mu_);
  std::vector<void*> out;
  out.reserve(by_ptr_.size());
  for (const auto& kv : by_ptr_) {
    if (kv.first != nullptr && kv.second != nullptr) {
      out.push_back(kv.first);
    }
  }
  return out;
}

std::vector<const char*> Allocator::debug_allocation_sites_for_testing() const {
  MuLockGuard lg(mu_);
  return debug_allocation_sites_;
}

bool Allocator::debug_block_is_split_tail(void* ptr) const noexcept {
  MuLockGuard lg(mu_);
  auto it = by_ptr_.find(ptr);
  if (it == by_ptr_.end() || !it->second) return false;
  return it->second->is_split_tail;
}

std::uint32_t Allocator::debug_block_gc_age(void* ptr) const noexcept {
  MuLockGuard lg(mu_);
  auto it = by_ptr_.find(ptr);
  if (it == by_ptr_.end() || !it->second) {
    return 0u;
  }
  return it->second->gc_age;
}

Allocator::DebugTailGaugeSnapshot
Allocator::debug_tail_gauge_snapshot_for_testing(bool force_scan) const {
  DebugTailGaugeSnapshot out{};

  // Snapshot gauges via public API so async/native backends are handled uniformly.
  DeviceStats stats_view = getDeviceStats();
  out.stats_blocks = stats_view.inactive_split_blocks_all;
  out.stats_bytes  = stats_view.inactive_split_bytes_all;

  // Fast path: when gauges are zero and scan is not forced, skip by_ptr_ traversal.
  if (!force_scan && out.stats_blocks == 0 && out.stats_bytes == 0) {
    out.recomputed_blocks = 0;
    out.recomputed_bytes  = 0;
    out.consistent        = true;
    out.undercount_only   = false;
    return out;
  }

  std::uint64_t blocks = 0;
  std::uint64_t bytes  = 0;

  {
    MuLockGuard lg(mu_);
    for (const auto& kv : by_ptr_) {
      const Block* b = kv.second;
      if (!b || !b->is_split_tail) {
        continue;
      }
      // Only count well-formed, inactive split tails.
      if (b->allocated) continue;
      if (b->event_count != 0) continue;
      if (b->segment_head) continue;
      if (!b->prev) continue;
      if (b->size == 0) continue;
      if (!b->stream_uses.empty()) continue;

      ++blocks;
      bytes += static_cast<std::uint64_t>(b->size);
    }
  }

  out.recomputed_blocks = blocks;
  out.recomputed_bytes  = bytes;

  out.consistent = (out.stats_blocks == out.recomputed_blocks &&
                    out.stats_bytes  == out.recomputed_bytes);
  out.undercount_only = false;
  if (!out.consistent &&
      out.stats_blocks <= out.recomputed_blocks &&
      out.stats_bytes  <= out.recomputed_bytes) {
    out.undercount_only = true;
  }

  return out;
}

Allocator::DebugStatsSnapshotConsistency
Allocator::debug_stats_snapshot_consistency_for_testing(DeviceIndex dev) const {
  DebugStatsSnapshotConsistency out{};

  // Snapshot allocator-wide stats via public API.
  DeviceStats stats_view = getDeviceStats();
  out.native_backend   = (cfg_.backend == BackendKind::Native);
  out.split_enabled    = split_enabled();
  out.stats_reserved   = stats_view.reserved_bytes_all_current;
  out.stats_allocated  = stats_view.allocated_bytes_all_current;

  // Per-segment snapshots for this device (empty on CPU-only or async backends).
  std::vector<MemorySegmentSnapshot> segs = snapshot_memory_segments(dev);

  std::uint64_t segs_reserved = 0;
  std::uint64_t segs_active   = 0;
  for (const auto& s : segs) {
    segs_reserved += s.bytes_reserved;
    segs_active   += s.bytes_active;
  }
  out.segs_reserved = segs_reserved;
  out.segs_active   = segs_active;
  out.stats_vs_segments_ok =
      (segs_reserved <= out.stats_reserved &&
       segs_active   <= out.stats_allocated);

  // Graph-pool snapshots filtered to this device.
  auto pools = snapshot_graph_pools(std::nullopt);
  bool pools_ok = true;
  std::uint64_t pools_reserved_sum = 0;
  std::uint64_t pools_active_sum   = 0;

  for (const auto& p : pools) {
    if (p.id.dev != dev) {
      continue;
    }

    std::uint64_t pool_segs_reserved = 0;
    std::uint64_t pool_segs_active   = 0;
    for (const auto& s : segs) {
      if (s.pool_id == p.id.id) {
        pool_segs_reserved += s.bytes_reserved;
        pool_segs_active   += s.bytes_active;
      }
    }

    if (p.bytes_reserved > pool_segs_reserved ||
        p.bytes_active   > pool_segs_active) {
      pools_ok = false;
    }

    pools_reserved_sum += p.bytes_reserved;
    pools_active_sum   += p.bytes_active;
  }

  if (pools_reserved_sum > out.stats_reserved ||
      pools_active_sum   > out.stats_allocated) {
    pools_ok = false;
  }

  out.stats_vs_pools_ok = pools_ok;
  return out;
}

void Allocator::debug_run_fragmentation_fuzzer_for_testing(
    const DebugFragmentationConfig& cfg) {
  if (cfg.steps == 0 || cfg.max_block_size == 0) {
    return;
  }

  std::mt19937_64 rng(cfg.seed);

  std::vector<void*> live;
  live.reserve(static_cast<std::size_t>(cfg.steps / 2 + 1));

  auto sample_size = [&](std::size_t max_bytes) -> std::size_t {
    if (max_bytes == 0) {
      return 0;
    }
    std::uint64_t r = rng();
    std::size_t span = max_bytes;
    std::size_t n = static_cast<std::size_t>(r % span);
    return n + 1;  // in [1, max_bytes]
  };

  for (std::size_t i = 0; i < cfg.steps; ++i) {
    bool do_alloc = live.empty() || ((rng() & 1ull) != 0ull);
    if (do_alloc) {
      std::size_t nbytes = sample_size(cfg.max_block_size);
      if (nbytes == 0) {
        continue;
      }
      void* p = nullptr;
      try {
        p = raw_alloc(nbytes);
      } catch (...) {
        // Best-effort: try to reclaim and retry once.
        try {
          process_events(-1);
          emptyCache();
          p = raw_alloc(nbytes);
        } catch (...) {
          p = nullptr;
        }
      }
      if (p) {
        live.push_back(p);
      }
    } else {
      std::size_t idx = static_cast<std::size_t>(rng() % live.size());
      void* p = live[idx];
      live[idx] = live.back();
      live.pop_back();
      try {
        raw_delete(p);
      } catch (...) {
        // raw_delete is noexcept for native backend; ignore just in case.
      }
    }
  }

  // Cleanup remaining allocations.
  for (void* p : live) {
    try {
      raw_delete(p);
    } catch (...) {
      // best-effort
    }
  }

  // Quiesce allocator and attempt to restore snapshots/gauges.
  process_events(-1);
  emptyCache();
}

Allocator::DebugInvariantsReport
Allocator::debug_check_invariants_for_testing() const {
  DebugInvariantsReport rep{};

  DebugTailGaugeSnapshot snap =
      debug_tail_gauge_snapshot_for_testing(/*force_scan=*/true);
  rep.tails_snapshot = snap;
  rep.inactive_split_blocks_gauge = snap.stats_blocks;
  rep.inactive_split_bytes_gauge  = snap.stats_bytes;

  bool ok = true;

  {
    MuLockGuard lg(mu_);
    for (const auto& kv : by_ptr_) {
      const Block* b = kv.second;
      if (!b || !b->is_split_tail) {
        continue;
      }

      if (b->allocated ||
          b->event_count != 0 ||
          b->segment_head ||
          !b->prev ||
          b->size == 0 ||
          !b->stream_uses.empty()) {
        ok = false;
        break;
      }

      if (b->prev && b->prev->ptr && b->ptr) {
        auto* prev_ptr = static_cast<const char*>(b->prev->ptr);
        auto* self_ptr = static_cast<const char*>(b->ptr);
        if (self_ptr != prev_ptr + static_cast<std::ptrdiff_t>(b->prev->size)) {
          ok = false;
          break;
        }
      }
    }
  }

  if (ok) {
    if (!snap.consistent && !snap.undercount_only) {
      ok = false;
    }
  }

  rep.ok = ok;
  rep.failed_check = nullptr;
  return rep;
}

bool Allocator::debug_cap_exceeded_for_testing(std::size_t rounded,
                                               std::size_t reserved,
                                               std::size_t limit) const noexcept {
  (void)this;
  return cap_exceeded(rounded, reserved, limit);
}

std::size_t Allocator::debug_safe_prospective_reserved_for_testing(
    std::size_t rounded,
    std::size_t reserved) const noexcept {
  (void)this;
  return safe_prospective_reserved(rounded, reserved);
}

std::size_t Allocator::debug_run_gc_pass_for_testing(std::size_t gc_target_bytes,
                                                     GcReason    reason) noexcept {
  return run_gc_pass_if_eligible(gc_target_bytes, reason);
}

std::uint64_t Allocator::debug_raw_alloc_nostream_calls_for_testing() const noexcept {
  (void)this;
  return g_debug_raw_alloc_nostream_calls.load(std::memory_order_relaxed);
}

std::uint64_t Allocator::debug_raw_alloc_stream_calls_for_testing() const noexcept {
  (void)this;
  return g_debug_raw_alloc_stream_calls.load(std::memory_order_relaxed);
}

void Allocator::debug_reset_raw_alloc_call_counters_for_testing() noexcept {
  g_debug_raw_alloc_nostream_calls.store(0, std::memory_order_relaxed);
  g_debug_raw_alloc_stream_calls.store(0, std::memory_order_relaxed);
}

bool Allocator::debug_is_oversize_size_for_testing(std::size_t sz) const noexcept {
  return is_oversize_size(sz, cfg_.max_split_size_bytes);
}

bool Allocator::debug_is_oversize_request_for_testing(std::size_t req) const noexcept {
  return is_oversize_request(req);
}

bool Allocator::debug_should_split_for_testing(std::size_t block_size,
                                               bool        is_split_tail,
                                               std::size_t req) const noexcept {
  MuLockGuard lg(mu_);
  Block tmp{};
  tmp.size = block_size;
  tmp.is_split_tail = is_split_tail;
  return should_split_unlocked(&tmp, req);
}

Allocator::DebugCandidateResult
Allocator::debug_evaluate_candidate_for_testing(std::size_t block_size,
                                                bool        is_split_tail,
                                                std::size_t req,
                                                std::size_t N_override,
                                                std::size_t T_override) {
  MuLockGuard lg(mu_);

  Block tmp{};
  tmp.size = block_size;
  tmp.is_split_tail = is_split_tail;

  auto before_count = stats_.tolerance_fills_count;
  auto before_bytes = stats_.tolerance_fills_bytes;

  CandidateDecision decision = evaluate_candidate(
      &tmp,
      req,
      cfg_.max_split_size_bytes,
      N_override,
      T_override);

  auto after_count = stats_.tolerance_fills_count;
  auto after_bytes = stats_.tolerance_fills_bytes;

  DebugCandidateResult out{};
  switch (decision) {
    case CandidateDecision::Reject:
      out.decision = DebugCandidateDecision::Reject;
      break;
    case CandidateDecision::TakeWhole:
      out.decision = DebugCandidateDecision::TakeWhole;
      break;
    case CandidateDecision::Split:
      out.decision = DebugCandidateDecision::Split;
      break;
  }

  out.counted_as_tolerance_fill = (after_count != before_count);
  if (after_bytes >= before_bytes) {
    out.tolerance_waste_bytes = static_cast<std::size_t>(after_bytes - before_bytes);
  } else {
    out.tolerance_waste_bytes = 0;
  }

  return out;
}

Allocator::DebugSplitResult
Allocator::debug_split_block_unlocked_for_testing(
    std::size_t   block_size,
    std::size_t   take_size,
    StreamId      owner_sid,
    std::uint64_t graph_pool_id) noexcept {
  DebugSplitResult out{};

  // Metadata-only helper: exercises split_block_unlocked under the allocator
  // mutex without touching device memory.
  MuLockGuard lg(mu_);

  auto blocks_before = stats_.inactive_split_blocks_all;
  auto bytes_before  = stats_.inactive_split_bytes_all;

  char* backing = new char[block_size];
  Block* b = new Block();

  b->device         = dev_;
  b->alloc_stream   = 0;
  b->owner_stream   = owner_sid;
  b->ptr            = backing;
  b->size           = block_size;
  b->requested_size = 0;
  b->allocated      = false;
  b->mapped         = false;
  b->prev           = nullptr;
  b->next           = nullptr;
  b->segment_head   = true;
  b->is_split_tail  = false;
  b->event_count    = 0;
  b->graph_pool_id  = graph_pool_id;

  by_ptr_[backing] = b;

  auto fill_info = [](DebugBlockInfo& dst, const Block* src) {
    if (!src) {
      dst.ptr = nullptr;
      dst.size = 0;
      dst.allocated = false;
      dst.segment_head = false;
      dst.is_split_tail = false;
      dst.prev_ptr = nullptr;
      dst.next_ptr = nullptr;
      dst.graph_pool_id = 0;
      dst.owner_stream = 0;
      return;
    }
    dst.ptr = src->ptr;
    dst.size = src->size;
    dst.allocated = src->allocated;
    dst.segment_head = src->segment_head;
    dst.is_split_tail = src->is_split_tail;
    dst.prev_ptr = src->prev ? src->prev->ptr : nullptr;
    dst.next_ptr = src->next ? src->next->ptr : nullptr;
    dst.graph_pool_id = src->graph_pool_id;
    dst.owner_stream = src->owner_stream;
  };

  fill_info(out.front_before, b);

  Block* ret = split_block_unlocked(b, take_size);
#ifndef NDEBUG
  assert(ret == b);
#endif
  (void)ret; // return value must equal b

  Block* tail = nullptr;
  if (b->next && b->next->is_split_tail) {
    tail = b->next;
  }

  fill_info(out.front_after, b);
  fill_info(out.tail_after, tail);

  auto in_per_stream = [&](Block* x) -> bool {
    if (!x) return false;
    auto it = per_stream_free_.find(x->owner_stream);
    if (it == per_stream_free_.end()) return false;
    return it->second.find(x) != it->second.end();
  };

  auto in_cross_stream = [&](Block* x) -> bool {
    if (!x) return false;
    return cross_stream_free_.find(x) != cross_stream_free_.end();
  };

  out.front_in_per_stream_free   = in_per_stream(b);
  out.front_in_cross_stream_free = in_cross_stream(b);
  out.tail_in_per_stream_free    = in_per_stream(tail);
  out.tail_in_cross_stream_free  = in_cross_stream(tail);

  out.inactive_blocks_before = blocks_before;
  out.inactive_blocks_after  = stats_.inactive_split_blocks_all;
  out.inactive_bytes_before  = bytes_before;
  out.inactive_bytes_after   = stats_.inactive_split_bytes_all;

  // Cleanup: remove synthetic structures and restore gauges to baseline.
  if (tail) {
    remove_from_free_indices_unlocked(tail);
    if (tail->is_split_tail) {
      if (stats_.inactive_split_blocks_all > 0) {
        stats_.inactive_split_blocks_all -= 1;
      }
      if (stats_.inactive_split_bytes_all >= tail->size) {
        stats_.inactive_split_bytes_all -= tail->size;
      }
      tail->is_split_tail = false;
    }
    by_ptr_.erase(tail->ptr);
    delete tail;
  }

  by_ptr_.erase(b->ptr);
  delete b;
  delete[] backing;

#ifndef NDEBUG
  // Invariants: helper leaves gauges as it found them.
  assert(stats_.inactive_split_blocks_all == blocks_before);
  assert(stats_.inactive_split_bytes_all == bytes_before);
#endif

  return out;
}

void Allocator::debug_trigger_split_block_assert_for_testing(
    std::size_t   block_size,
    std::size_t   take_size,
    bool          mark_as_split_tail,
    bool          make_oversize,
    std::uint64_t graph_pool_id) noexcept {
#ifndef NDEBUG
  MuLockGuard lg(mu_);

  char* backing = new char[block_size];
  Block* b = new Block();

  b->device         = dev_;
  b->alloc_stream   = 0;
  b->owner_stream   = 0;
  b->ptr            = backing;
  b->size           = block_size;
  b->requested_size = 0;
  b->allocated      = false;
  b->mapped         = false;
  b->prev           = nullptr;
  b->next           = nullptr;
  b->segment_head   = true;
  b->is_split_tail  = mark_as_split_tail;
  b->event_count    = 0;
  b->graph_pool_id  = graph_pool_id;

  std::size_t prev_M = cfg_.max_split_size_bytes;
  if (make_oversize) {
    // Choose an active oversize threshold strictly below block_size.
    cfg_.max_split_size_bytes = (block_size > 0) ? block_size : 1;
  }

  by_ptr_[backing] = b;

  (void)split_block_unlocked(b, take_size);

  // If we reach here (e.g. in non-death-test or release builds), perform
  // best-effort cleanup and restore configuration. There may or may not be a
  // tail; if present, remove it from indices, fix gauges, and erase it.
  Block* tail = nullptr;
  if (b->next && b->next->is_split_tail) {
    tail = b->next;
  }
  if (tail) {
    remove_from_free_indices_unlocked(tail);
    if (tail->is_split_tail) {
      if (stats_.inactive_split_blocks_all > 0) {
        stats_.inactive_split_blocks_all -= 1;
      }
      if (stats_.inactive_split_bytes_all >= tail->size) {
        stats_.inactive_split_bytes_all -= tail->size;
      }
      tail->is_split_tail = false;
    }
    by_ptr_.erase(tail->ptr);
    delete tail;
  }

  cfg_.max_split_size_bytes = prev_M;
  by_ptr_.erase(b->ptr);
  delete b;
  delete[] backing;
#else
  (void)block_size;
  (void)take_size;
  (void)mark_as_split_tail;
  (void)make_oversize;
  (void)graph_pool_id;
#endif
}

Allocator::DebugCoalesceResult
Allocator::debug_coalesce_neighbors_unlocked_for_testing(
    const DebugCoalesceScenario& scenario) noexcept {
  DebugCoalesceResult out{};

  MuLockGuard lg(mu_);

  auto blocks_gauge0 = stats_.inactive_split_blocks_all;
  auto bytes_gauge0  = stats_.inactive_split_bytes_all;

  std::size_t prev_M = cfg_.max_split_size_bytes;
  if (scenario.oversize_threshold_bytes != 0) {
    cfg_.max_split_size_bytes = scenario.oversize_threshold_bytes;
  }

  std::size_t left_size  = scenario.has_left  ? scenario.left_size  : 0;
  std::size_t self_size  = scenario.self_size;
  std::size_t right_size = scenario.has_right ? scenario.right_size : 0;
  std::size_t total_size = left_size + self_size + right_size;

  char* backing = nullptr;
  if (total_size > 0) {
    backing = new char[total_size];
  }

  Block* left  = nullptr;
  Block* self  = nullptr;
  Block* right = nullptr;

  auto make_block = [&](Block*& blk,
                        std::size_t offset_bytes,
                        std::size_t size_bytes,
                        std::uint64_t graph_pool_id,
                        StreamId owner_stream,
                        bool allocated,
                        int event_count,
                        bool is_split_tail) {
    if (size_bytes == 0) {
      blk = nullptr;
      return;
    }
    blk = new Block();
    blk->device         = dev_;
    blk->alloc_stream   = 0;
    blk->owner_stream   = owner_stream;
    blk->ptr            = backing ? static_cast<void*>(backing + offset_bytes) : nullptr;
    blk->size           = size_bytes;
    blk->requested_size = 0;
    blk->allocated      = allocated;
    blk->mapped         = false;
    blk->prev           = nullptr;
    blk->next           = nullptr;
    blk->segment_head   = false;
    blk->is_split_tail  = is_split_tail;
    blk->event_count    = event_count;
    blk->graph_pool_id  = graph_pool_id;
    blk->stream_uses.clear();
    if (blk->ptr) {
      by_ptr_[blk->ptr] = blk;
    }
  };

  std::size_t offset = 0;
  if (scenario.has_left) {
    make_block(left,
               offset,
               left_size,
               scenario.left_graph_pool_id,
               scenario.left_owner_stream,
               scenario.left_allocated,
               scenario.left_event_count,
               scenario.left_is_split_tail);
    offset += left_size;
  }

  make_block(self,
             offset,
             self_size,
             scenario.self_graph_pool_id,
             scenario.self_owner_stream,
             /*allocated=*/false,
             /*event_count=*/0,
             scenario.self_is_split_tail);
  offset += self_size;

  if (scenario.has_right) {
    make_block(right,
               offset,
               right_size,
               scenario.right_graph_pool_id,
               scenario.right_owner_stream,
               scenario.right_allocated,
               scenario.right_event_count,
               scenario.right_is_split_tail);
  }

  // Wire adjacency and segment_head flags.
  if (left && self) {
    left->next        = self;
    left->segment_head = true;
    self->prev        = left;
  } else if (self) {
    self->segment_head = true;
  }
  if (self && right) {
    self->next  = right;
    right->prev = self;
  }

  auto maybe_insert_free = [&](Block* blk, bool in_free) {
    if (!blk) return;
    if (!in_free) return;
    if (blk->allocated) return;
    if (blk->event_count != 0) return;
    insert_free_block_unlocked(blk, blk->owner_stream);
  };

  maybe_insert_free(left, scenario.left_in_free_indices);
  maybe_insert_free(right, scenario.right_in_free_indices);

  auto bump_tail_gauge = [&](Block* blk) {
    if (!blk || !blk->is_split_tail) return;
    stats_.inactive_split_blocks_all += 1;
    stats_.inactive_split_bytes_all  += blk->size;
  };

  bump_tail_gauge(left);
  bump_tail_gauge(self);
  bump_tail_gauge(right);

  out.inactive_blocks_before = stats_.inactive_split_blocks_all;
  out.inactive_bytes_before  = stats_.inactive_split_bytes_all;

  auto fill_info = [](DebugBlockInfo& dst, const Block* src) {
    if (!src) {
      dst.ptr          = nullptr;
      dst.size         = 0;
      dst.allocated    = false;
      dst.segment_head = false;
      dst.is_split_tail = false;
      dst.prev_ptr     = nullptr;
      dst.next_ptr     = nullptr;
      dst.graph_pool_id = 0;
      dst.owner_stream = 0;
      return;
    }
    dst.ptr          = src->ptr;
    dst.size         = src->size;
    dst.allocated    = src->allocated;
    dst.segment_head = src->segment_head;
    dst.is_split_tail = src->is_split_tail;
    dst.prev_ptr     = src->prev ? src->prev->ptr : nullptr;
    dst.next_ptr     = src->next ? src->next->ptr : nullptr;
    dst.graph_pool_id = src->graph_pool_id;
    dst.owner_stream = src->owner_stream;
  };

  fill_info(out.left_before, left);
  fill_info(out.self_before, self);
  fill_info(out.right_before, right);

  void* left_key  = left  ? left->ptr  : nullptr;
  void* self_key  = self  ? self->ptr  : nullptr;
  void* right_key = right ? right->ptr : nullptr;

  Block* head = self;
  head = coalesce_neighbors_unlocked(head);

  auto find_live_by_ptr = [&](void* key) -> Block* {
    if (!key) return nullptr;
    auto it = by_ptr_.find(key);
    if (it == by_ptr_.end()) return nullptr;
    return it->second;
  };

  Block* left_live  = left_key  ? find_live_by_ptr(left_key)  : nullptr;
  Block* self_live  = self_key  ? find_live_by_ptr(self_key)  : nullptr;
  Block* right_live = right_key ? find_live_by_ptr(right_key) : nullptr;

  fill_info(out.head_after, head);
  fill_info(out.left_after, left_live);
  fill_info(out.self_after, self_live);
  fill_info(out.right_after, right_live);

  out.inactive_blocks_after = stats_.inactive_split_blocks_all;
  out.inactive_bytes_after  = stats_.inactive_split_bytes_all;

  auto consume_tail = [&](Block* blk) {
    if (!blk || !blk->is_split_tail) return;
    if (stats_.inactive_split_blocks_all > 0) {
      stats_.inactive_split_blocks_all -= 1;
    }
    if (stats_.inactive_split_bytes_all >= blk->size) {
      stats_.inactive_split_bytes_all -= blk->size;
    }
    blk->is_split_tail = false;
  };

  auto cleanup_block = [&](Block* blk) {
    if (!blk) return;
    remove_from_free_indices_unlocked(blk);
    consume_tail(blk);
    if (blk->ptr) {
      auto it = by_ptr_.find(blk->ptr);
      if (it != by_ptr_.end() && it->second == blk) {
        by_ptr_.erase(it);
      } else {
        by_ptr_.erase(blk->ptr);
      }
    }
    delete blk;
  };

  cleanup_block(left_live);
  cleanup_block(self_live);
  cleanup_block(right_live);

  cfg_.max_split_size_bytes = prev_M;
  if (backing) {
    delete[] backing;
  }

#ifndef NDEBUG
  assert(stats_.inactive_split_blocks_all == blocks_gauge0);
  assert(stats_.inactive_split_bytes_all  == bytes_gauge0);
#endif

  return out;
}

void Allocator::debug_trigger_coalesce_neighbors_assert_for_testing(
    bool mark_center_allocated,
    bool make_center_eventful,
    bool insert_center_into_free_indices,
    bool mismatch_neighbor_graph_pool) noexcept {
#ifndef NDEBUG
  MuLockGuard lg(mu_);

  // Simple two-block segment: [left]-[center].
  const std::size_t left_size = 1024;
  const std::size_t self_size = 1024;
  const std::size_t total_size = left_size + self_size;

  char* backing = new char[total_size];

  Block* left = new Block();
  Block* self = new Block();

  left->device         = dev_;
  left->alloc_stream   = 0;
  left->owner_stream   = 0;
  left->ptr            = backing;
  left->size           = left_size;
  left->requested_size = 0;
  left->allocated      = false;
  left->mapped         = false;
  left->prev           = nullptr;
  left->next           = self;
  left->segment_head   = true;
  left->is_split_tail  = false;
  left->event_count    = 0;
  left->graph_pool_id  = 1;
  left->stream_uses.clear();

  self->device         = dev_;
  self->alloc_stream   = 0;
  self->owner_stream   = 0;
  self->ptr            = backing + left_size;
  self->size           = self_size;
  self->requested_size = 0;
  self->allocated      = mark_center_allocated;
  self->mapped         = false;
  self->prev           = left;
  self->next           = nullptr;
  self->segment_head   = false;
  self->is_split_tail  = false;
  self->event_count    = make_center_eventful ? 1 : 0;
  self->graph_pool_id  = mismatch_neighbor_graph_pool ? 2 : 1;
  self->stream_uses.clear();

  by_ptr_[left->ptr] = left;
  by_ptr_[self->ptr] = self;

  if (insert_center_into_free_indices && !self->allocated && self->event_count == 0) {
    insert_free_block_unlocked(self, self->owner_stream);
  }

  (void)coalesce_neighbors_unlocked(self);

  // Best-effort cleanup if we reach here (e.g., in release builds).
  remove_from_free_indices_unlocked(self);
  remove_from_free_indices_unlocked(left);
  by_ptr_.erase(self->ptr);
  by_ptr_.erase(left->ptr);
  delete self;
  delete left;
  delete[] backing;
#else
  (void)mark_center_allocated;
  (void)make_center_eventful;
  (void)insert_center_into_free_indices;
  (void)mismatch_neighbor_graph_pool;
#endif
}

void Allocator::debug_note_allocation_transition(Block* b, const char* site) {
  if (!site) return;
  (void)b; // reserved for future pointer-specific tracking
  debug_allocation_sites_.push_back(site);
}
#endif

bool Allocator::is_oversize_block(const Block* b) const noexcept {
#ifndef NDEBUG
  assert(b != nullptr);
#endif
  if (!b) {
    return false;
  }
  return is_oversize_size(b->size, cfg_.max_split_size_bytes);
}

bool Allocator::is_oversize_request(std::size_t req) const noexcept {
  return is_oversize_size(req, cfg_.max_split_size_bytes);
}

bool Allocator::should_split_unlocked(const Block* b, std::size_t req) const noexcept {
  if (!b) return false;
  if (req == 0) return false;
  if (b->size == 0) return false;
  if (req >= b->size) return false;
  if (b->is_split_tail) return false;

  const std::size_t M_cfg = cfg_.max_split_size_bytes;
  const bool oversize_active =
      (M_cfg != 0 && M_cfg != std::numeric_limits<std::size_t>::max());

  if (oversize_active && is_oversize_block(b)) {
    return false;
  }

  const std::size_t rem = b->size - req;
  if (rem == 0) return false;

  PoolKind pool = classify(req);
  if (pool == PoolKind::Small) {
    constexpr std::size_t kMinTailSmall = 512; // bytes
    return rem >= kMinTailSmall;
  }

  // Large pool.
  if (oversize_active && is_oversize_request(req)) {
    return false;
  }
  constexpr std::size_t kMinTailLarge = RoundPolicy::kSmallSize; // 1 MiB
  return rem > kMinTailLarge;
}

Allocator::CandidateDecision Allocator::evaluate_candidate(
    Block* b,
    std::size_t req,
    std::size_t M,
    std::size_t N,
    std::size_t T) noexcept {
#ifndef NDEBUG
  assert(M == cfg_.max_split_size_bytes);
#else
  (void)M;
#endif
  if (!b || req == 0) {
    return CandidateDecision::Reject;
  }

  const std::size_t M_cfg = cfg_.max_split_size_bytes;
  const bool oversize_active =
      (M_cfg != 0 && M_cfg != std::numeric_limits<std::size_t>::max());

  const bool req_oversize   = is_oversize_request(req);
  const bool block_oversize = is_oversize_block(b);

  if (b->size < req) {
    return CandidateDecision::Reject;
  }

  if (b->size == req) {
    return CandidateDecision::TakeWhole;
  }

  if (req_oversize) {
    if (!block_oversize) {
      return CandidateDecision::Reject;
    }
    std::size_t delta = b->size - req; // > 0 here
    if (N == 0) {
      return CandidateDecision::Reject;
    }
    if (delta >= N) {
      return CandidateDecision::Reject;
    }
    stats_.tolerance_fills_count += 1;
    stats_.tolerance_fills_bytes += delta;
    return CandidateDecision::TakeWhole;
  }

  if (oversize_active && block_oversize) {
    return CandidateDecision::Reject;
  }

  std::size_t rem = b->size - req; // > 0 here
  std::size_t tol_cap = oversize_active
                           ? (T < M_cfg ? T : M_cfg)
                           : T;
  if (T > 0 && rem > 0 && rem <= tol_cap) {
    stats_.tolerance_fills_count += 1;
    stats_.tolerance_fills_bytes += rem;
    return CandidateDecision::TakeWhole;
  }

  if (!b->is_split_tail && should_split_unlocked(b, req)) {
    return CandidateDecision::Split;
  }

  // Fallback: reuse larger-than-needed block without counting as a tolerance fill.
  return CandidateDecision::TakeWhole;
}

// ---- Free-list helpers (native backend) ----
Allocator::Block* Allocator::try_take_free_block_unlocked(StreamId sid,
                                                          std::size_t rounded_size,
                                                          std::uint64_t target_pool_id) noexcept {
  auto no_split_impl = [&]() -> Block* {
    auto it = per_stream_free_.find(sid);
    if (it == per_stream_free_.end() || it->second.empty()) return nullptr;

    // Find first block with size >= rounded_size and matching pool predicate.
    Block key{}; key.size = rounded_size; key.ptr = nullptr;
    auto fit = it->second.lower_bound(&key);
    for (auto iter = fit; iter != it->second.end(); ++iter) {
      Block* b = *iter;
      if (target_pool_id == 0) {
        if (b->graph_pool_id != 0) {
          continue;  // global allocations may only reuse global blocks
        }
      } else {
        if (b->graph_pool_id != 0 && b->graph_pool_id != target_pool_id) {
          continue;  // pool allocations may not steal from other pools
        }
      }
      // Remove from free indices (per-stream + cross-stream) and return.
      remove_from_free_indices_unlocked(b);
      return b;
    }
    return nullptr;
  };

  if (!split_enabled()) {
    return no_split_impl();
  }

  // Gate-on: native backend splitting/oversize policy.
  const std::size_t M = cfg_.max_split_size_bytes;
  const std::size_t N = cfg_.max_non_split_rounding_bytes;
  const std::size_t T = cfg_.roundup_tolerance_bytes;

  auto it = per_stream_free_.find(sid);
  if (it == per_stream_free_.end() || it->second.empty()) {
    return nullptr;
  }

  Block key{}; key.size = rounded_size; key.ptr = nullptr;
  auto fit = it->second.lower_bound(&key);
  for (auto iter = fit; iter != it->second.end(); ++iter) {
    Block* b = *iter;

    // splitting is enabled we require exact pool matches for pooled allocations.
    if (target_pool_id == 0) {
      if (b->graph_pool_id != 0) {
        continue;  // global allocations may only reuse global blocks
      }
    } else {
      if (b->graph_pool_id != target_pool_id) {
        continue;  // pool allocations may only reuse blocks from their own pool
      }
    }

    CandidateDecision decision = evaluate_candidate(b, rounded_size, M, N, T);
    if (decision == CandidateDecision::Reject) {
      continue;  // leave b in indices
    }

    // We will reuse this block in some form.
    remove_from_free_indices_unlocked(b);
    if (decision == CandidateDecision::Split) {
      b = split_block_unlocked(b, rounded_size);
    }
    return b;
  }

  return nullptr;
}

Allocator::Block* Allocator::try_take_from_cross_stream_unlocked(std::size_t rounded_size,
                                                                 std::uint64_t target_pool_id) noexcept {
  auto no_split_impl = [&]() -> Block* {
    if (cross_stream_free_.empty()) return nullptr;

    Block key{}; key.size = rounded_size; key.ptr = nullptr;
    auto fit = cross_stream_free_.lower_bound(&key);
    for (auto iter = fit; iter != cross_stream_free_.end(); ++iter) {
      Block* b = *iter;
      if (target_pool_id == 0) {
        if (b->graph_pool_id != 0) {
          continue;
        }
      } else {
        if (b->graph_pool_id != 0 && b->graph_pool_id != target_pool_id) {
          continue;
        }
      }
      remove_from_free_indices_unlocked(b);
      return b;
    }
    return nullptr;
  };

  if (!split_enabled()) {
    return no_split_impl();
  }

  // Gate-on: only global allocations should reach here.
#if defined(VBT_INTERNAL_TESTS)
  assert(target_pool_id == 0 &&
         "try_take_from_cross_stream_unlocked gate-on called with non-zero pool id");
#endif

  if (cross_stream_free_.empty()) {
    return nullptr;
  }

  Block key{}; key.size = rounded_size; key.ptr = nullptr;
  auto fit = cross_stream_free_.lower_bound(&key);

  const std::size_t M = cfg_.max_split_size_bytes;
  const std::size_t N = cfg_.max_non_split_rounding_bytes;
  const std::size_t T = cfg_.roundup_tolerance_bytes;

  for (auto iter = fit; iter != cross_stream_free_.end(); ++iter) {
    Block* b = *iter;

    // Global-only reuse in gate-on: skip pool-owned segments.
    if (b->graph_pool_id != 0) {
      continue;
    }

    CandidateDecision decision = evaluate_candidate(b, rounded_size, M, N, T);
    if (decision == CandidateDecision::Reject) {
      continue;
    }

    remove_from_free_indices_unlocked(b);
    if (decision == CandidateDecision::Split) {
      b = split_block_unlocked(b, rounded_size);
    }
    return b;
  }

  return nullptr;
}

void Allocator::remove_from_free_indices_unlocked(Block* b) noexcept {
  if (!b) return;
  auto it = per_stream_free_.find(b->owner_stream);
  if (it != per_stream_free_.end()) {
    auto sit = it->second.find(b);
    if (sit != it->second.end()) it->second.erase(sit);
    if (it->second.empty()) per_stream_free_.erase(it);
  }
  auto itc = cross_stream_free_.find(b);
  if (itc != cross_stream_free_.end()) cross_stream_free_.erase(itc);
}

void Allocator::insert_free_block_unlocked(Block* b, StreamId sid) noexcept {
  if (!b) return;
  b->owner_stream = sid;
  per_stream_free_[sid].insert(b);
  cross_stream_free_.insert(b);
}

void Allocator::on_reuse_from_free_list(Block* b,
                                        std::size_t nbytes,
                                        StreamId alloc_sid,
                                        std::uint64_t target_pool_id) noexcept {
  if (!b) return;

  // Reset GC age on the segment head before marking it allocated. This treats
  // any reuse as a fresh segment for GC heuristics.
  Block* head = b;
  while (head->prev) {
    head = head->prev;
  }
  head->gc_age = 0;

  // If this block is an inactive split tail, consume gauges before reuse.
  if (split_enabled() && b->is_split_tail) {
#ifndef NDEBUG
    // Gauges should exactly account for all tails.
    assert(stats_.inactive_split_blocks_all > 0);
    assert(stats_.inactive_split_bytes_all >= b->size);
#endif
    if (stats_.inactive_split_blocks_all > 0) {
      stats_.inactive_split_blocks_all -= 1;
    }
    if (stats_.inactive_split_bytes_all >= b->size) {
      stats_.inactive_split_bytes_all -= b->size;
    }
    b->is_split_tail = false;
  }

  b->allocated      = true;
  b->requested_size = nbytes;
  b->alloc_stream   = alloc_sid;
  b->owner_stream   = alloc_sid;
  b->event_count    = 0;
  b->stream_uses.clear();

  if (!split_enabled()) {
    if (target_pool_id == 0) {
      b->graph_pool_id = 0u;
    } else {
      if (b->graph_pool_id == 0u) {
        b->graph_pool_id = target_pool_id;  // promote global block into pool
      } else {
#ifndef NDEBUG
        // By selection rules, b->graph_pool_id must already equal target_pool_id.
        assert(b->graph_pool_id == target_pool_id);
#endif
      }
    }
  } else {
    // individual blocks here; selection logic ensures compatibility.
    if (target_pool_id == 0) {
#ifndef NDEBUG
      assert(b->graph_pool_id == 0u);
#endif
    } else {
#ifndef NDEBUG
      assert(b->graph_pool_id == target_pool_id);
#endif
    }
  }

  active_blocks_.insert(b);

  std::size_t r = b->size;
  stats_.allocated_bytes_all_current += r;
  if (stats_.allocated_bytes_all_current > stats_.max_allocated_bytes_all) {
    stats_.max_allocated_bytes_all = stats_.allocated_bytes_all_current;
  }
  stats_.requested_bytes_all_current += b->requested_size;
  if (stats_.requested_bytes_all_current > stats_.max_requested_bytes_all) {
    stats_.max_requested_bytes_all = stats_.requested_bytes_all_current;
  }

#ifdef VBT_INTERNAL_TESTS
  debug_note_allocation_transition(b, "raw_alloc_reuse");
#endif
}

Allocator::Block* Allocator::split_block_unlocked(Block* b, std::size_t take_size) noexcept {
  if (!split_enabled()) {
    return b;
  }

#ifndef NDEBUG
  // Debug-time precondition checks for native backend when splitting is enabled.
  if (cfg_.backend == BackendKind::Native) {
    assert(b != nullptr);
    assert(!b->allocated);
    assert(b->event_count == 0);
    assert(b->stream_uses.empty());
    assert(!b->is_split_tail);
    assert(take_size > 0);
    assert(take_size <= b->size);
    std::size_t M = cfg_.max_split_size_bytes;
    if (M != 0 && M != std::numeric_limits<std::size_t>::max()) {
      assert(!is_oversize_block(b));
    }
  }
#endif

  if (!b) {
    return b;
  }

  std::size_t size0 = b->size;
  if (take_size == 0 || take_size >= size0) {
    // Exact-size or invalid sizes behave as a no-op in release builds.
    return b;
  }

  std::size_t rem = size0 - take_size;
  if (rem == 0) {
    return b;
  }

  // Allocate and initialize the tail metadata node.
  Block* tail = new Block();

  // Preserve segment and pool invariants.
  tail->device        = b->device;
  tail->owner_stream  = b->owner_stream;
  tail->graph_pool_id = b->graph_pool_id;

  // Derived geometry.
  tail->ptr  = static_cast<char*>(b->ptr) + take_size;
  tail->size = rem;

  // Canonical initial state for a free, never-allocated tail.
  tail->alloc_stream   = 0;
  tail->requested_size = 0;
  tail->allocated      = false;
  tail->mapped         = false;
  tail->prev           = nullptr;
  tail->next           = nullptr;
  tail->segment_head   = false;
  tail->is_split_tail  = true;
  tail->event_count    = 0;
  tail->stream_uses.clear();

  // Rewire adjacency: insert tail directly after b.
  Block* old_next = b->next;
  tail->prev = b;
  tail->next = old_next;
  if (old_next) {
    old_next->prev = tail;
  }
  b->next = tail;

  // Shrink the front block to the requested size.
  b->size = take_size;

  // Register tail in block map and free indices.
  by_ptr_[tail->ptr] = tail;
  insert_free_block_unlocked(tail, tail->owner_stream);

  // Update inactive split gauges for the newly created tail.
  stats_.inactive_split_blocks_all += 1;
  stats_.inactive_split_bytes_all += tail->size;

  // Splitting creates a logically new idle segment chain for GC heuristics.
  Block* seg_head = b;
  while (seg_head->prev) {
    seg_head = seg_head->prev;
  }
  seg_head->gc_age = 0;

  return b;
}

Allocator::Block* Allocator::coalesce_neighbors_unlocked(Block* b) noexcept {
  if (!split_enabled()) {
    return b;
  }

  if (!b) {
    // Debug helpers may exercise nullptr; production never passes this.
    return nullptr;
  }

#ifndef NDEBUG
  // Basic precondition checks for native backend when splitting is enabled.
  if (cfg_.backend == BackendKind::Native) {
    auto it = by_ptr_.find(b->ptr);
    assert(it != by_ptr_.end());
    assert(it->second == b);
    assert(!b->allocated);
    assert(b->event_count == 0);
    assert(b->stream_uses.empty());

    // Center block must not be present in any free index.
    auto itp = per_stream_free_.find(b->owner_stream);
    if (itp != per_stream_free_.end()) {
      assert(itp->second.find(b) == itp->second.end());
    }
    assert(cross_stream_free_.find(b) == cross_stream_free_.end());

    // Validate basic segment invariants and pool uniformity.
    Block* seg_head = b;
    while (seg_head->prev) {
      seg_head = seg_head->prev;
    }
    assert(seg_head->segment_head);
    DeviceIndex   dev   = seg_head->device;
    std::uint64_t pool  = seg_head->graph_pool_id;
    Block*        prev  = nullptr;
    for (Block* cur = seg_head; cur != nullptr; cur = cur->next) {
      if (!cur->prev) {
        assert(cur->segment_head);
      } else {
        assert(!cur->segment_head);
      }
      if (prev) {
        assert(cur->prev == prev);
        assert(static_cast<char*>(cur->ptr) ==
               static_cast<char*>(prev->ptr) + prev->size);
      }
      assert(cur->device == dev);
      assert(cur->graph_pool_id == pool);
      prev = cur;
    }
  }
#endif

  const std::size_t M_cfg = cfg_.max_split_size_bytes;
  const bool oversize_active =
      (M_cfg != 0 && M_cfg != std::numeric_limits<std::size_t>::max());

  // Oversize centers are islands: do not inspect or merge neighbors.
  if (oversize_active && is_oversize_block(b)) {
    return b;
  }

  auto consume_inactive_tail_unlocked = [&](Block* x) noexcept {
    if (!x || !x->is_split_tail) return;
    if (stats_.inactive_split_blocks_all > 0) {
      stats_.inactive_split_blocks_all -= 1;
    }
    if (stats_.inactive_split_bytes_all >= x->size) {
      stats_.inactive_split_bytes_all -= x->size;
    }
    x->is_split_tail = false;
  };

  auto eligible_neighbor = [&](Block* n, Block* anchor,
                               bool oversize_flag) noexcept -> bool {
    if (!n) return false;
    if (n->allocated) return false;
    if (n->event_count != 0) return false;
    if (!n->stream_uses.empty()) return false;
    if (n->graph_pool_id != anchor->graph_pool_id) return false;
    if (oversize_flag && is_oversize_block(n)) return false;

#ifndef NDEBUG
    assert(n->device == anchor->device);
    if (n == anchor->prev) {
      assert(n->next == anchor);
      assert(static_cast<char*>(anchor->ptr) ==
             static_cast<char*>(n->ptr) + n->size);
    } else if (n == anchor->next) {
      assert(n->prev == anchor);
      assert(static_cast<char*>(n->ptr) ==
             static_cast<char*>(anchor->ptr) + anchor->size);
    }
#endif

    return true;
  };

  StreamId original_owner = b->owner_stream;

  Block* head = b;
  const bool head_was_tail = head->is_split_tail;
  bool       merged_left   = false;

  // Step 1: attempt left merge first.
  Block* left = head->prev;
  if (eligible_neighbor(left, head, oversize_active)) {
    // Respect P4: erase left from free indices before mutating it.
    remove_from_free_indices_unlocked(left);

    // Account for tails before adjusting sizes.
    consume_inactive_tail_unlocked(left);
    if (head_was_tail) {
      consume_inactive_tail_unlocked(head);
    }

    // Grow left to cover head and rewire adjacency.
    left->size += head->size;
    left->next  = head->next;
    if (head->next) {
      head->next->prev = left;
    }

    by_ptr_.erase(head->ptr);
    delete head;

    head        = left;
    merged_left = true;
  }

  // If the merged head has become oversize, stop before considering right.
  if (oversize_active && is_oversize_block(head)) {
    head->segment_head = (head->prev == nullptr);
    head->allocated    = false;
    head->event_count  = 0;
    head->stream_uses.clear();
    head->owner_stream = original_owner;

    auto it2 = by_ptr_.find(head->ptr);
    if (it2 == by_ptr_.end()) {
      by_ptr_[head->ptr] = head;
    } else {
      it2->second = head;
    }

    return head;
  }

  // Step 2: attempt right merge against the (possibly updated) head.
  Block* right = head->next;
  if (eligible_neighbor(right, head, oversize_active)) {
    remove_from_free_indices_unlocked(right);

    consume_inactive_tail_unlocked(right);
    if (!merged_left && head_was_tail) {
      // Center tail becomes a non-tail head when only merging right.
      consume_inactive_tail_unlocked(head);
    }

    head->size += right->size;
    head->next  = right->next;
    if (right->next) {
      right->next->prev = head;
    }

    by_ptr_.erase(right->ptr);
    delete right;
  }

  // Normalize head and ensure tracking invariants.
  head->segment_head = (head->prev == nullptr);
  head->allocated    = false;
  head->event_count  = 0;
  head->stream_uses.clear();
  head->owner_stream = original_owner;

  auto it2 = by_ptr_.find(head->ptr);
  if (it2 == by_ptr_.end()) {
    by_ptr_[head->ptr] = head;
  } else {
    it2->second = head;
  }

  // Coalescing yields a fresh idle segment head for GC purposes.
  head->gc_age = 0;

  return head;
}

#if VBT_WITH_CUDA
cudaError_t Allocator::memcpyAsync(void* dst, int dstDev, const void* src, int srcDev,
                                   std::size_t bytes, Stream s, bool p2p_enabled) noexcept {
#if VBT_WITH_CUDA
  if (cfg_.backend == BackendKind::Async) {
    return AsyncBackend::get(dev_).memcpyAsync(dst, dstDev, src, srcDev, bytes, s, p2p_enabled);
  }
  if (bytes == 0) return cudaSuccess;
  if (!dst || !src) return cudaErrorInvalidValue;
  auto is_host = [](int d){ return d < 0; };
  if (is_host(dstDev) && is_host(srcDev)) return cudaErrorInvalidValue;
#ifndef NDEBUG
  // Best-effort: ensure the current thread is not holding the allocator mutex.
  // Do not assert on global contention from other threads.
  if (mu_.try_lock()) {
    mu_.unlock();
  }
#endif
#ifdef VBT_PARANOID_RUNTIME_CHECKS
  // If locks appear held, route to default memcpy to avoid potential deadlocks
  if (!mu_.try_lock()) {
    DeviceGuard dg(static_cast<DeviceIndex>(dstDev));
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, reinterpret_cast<cudaStream_t>(s.handle()));
  } else {
    mu_.unlock();
  }
#endif
  auto guard_dev_for = [&](cudaMemcpyKind kind)->DeviceIndex{
    switch (kind) {
      case cudaMemcpyHostToDevice: return static_cast<DeviceIndex>(dstDev);
      case cudaMemcpyDeviceToHost: return static_cast<DeviceIndex>(srcDev);
      case cudaMemcpyDeviceToDevice:
      case cudaMemcpyDefault: return static_cast<DeviceIndex>(dstDev);
      default: return static_cast<DeviceIndex>(dstDev);
    }
  };
  auto do_async = [&](cudaMemcpyKind kind)->cudaError_t{
    DeviceGuard dg(guard_dev_for(kind));
    return cudaMemcpyAsync(dst, src, bytes, kind, reinterpret_cast<cudaStream_t>(s.handle()));
  };
  auto do_peer = [&]()->cudaError_t{
    DeviceGuard dg(static_cast<DeviceIndex>(dstDev));
    return cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, bytes, reinterpret_cast<cudaStream_t>(s.handle()));
  };
  if (is_host(srcDev) && !is_host(dstDev)) {
    return do_async(cudaMemcpyHostToDevice);
  } else if (!is_host(srcDev) && is_host(dstDev)) {
    return do_async(cudaMemcpyDeviceToHost);
  } else if (!is_host(srcDev) && !is_host(dstDev)) {
    if (dstDev == srcDev) {
      return do_async(cudaMemcpyDeviceToDevice);
    }
    auto fallback_codes = {
      cudaErrorInvalidValue,
      cudaErrorInvalidDevice,
      cudaErrorInvalidResourceHandle,
      cudaErrorPeerAccessUnsupported,
      cudaErrorPeerAccessNotEnabled,
      cudaErrorNotSupported,
      cudaErrorInvalidMemcpyDirection
    };
    auto is_fallback = [&](cudaError_t st){ for (auto c : fallback_codes) if (st == c) return true; return false; };
    // Compute provenance without taking allocator locks for CUDA calls
    bool src_owned = Allocator::get(static_cast<DeviceIndex>(srcDev)).owns(src);
    bool dst_owned = Allocator::get(static_cast<DeviceIndex>(dstDev)).owns(dst);
    if (p2p_enabled) {
      cudaError_t st = do_peer();
      if (is_fallback(st)) { (void)cudaGetLastError(); return do_async(cudaMemcpyDefault); }
      return st;
    } else {
      // If both owned, default first; else peer first
      if (src_owned && dst_owned) {
        cudaError_t st = do_async(cudaMemcpyDefault);
        if (is_fallback(st)) { (void)cudaGetLastError(); return do_peer(); }
        return st;
      } else {
        cudaError_t st = do_peer();
        if (is_fallback(st)) { (void)cudaGetLastError(); return do_async(cudaMemcpyDefault); }
        return st;
      }
    }
  }
  return do_async(cudaMemcpyDefault);
#else
  (void)dst; (void)dstDev; (void)src; (void)srcDev; (void)bytes; (void)s; (void)p2p_enabled; return cudaSuccess;
#endif
}
#endif

#if VBT_WITH_CUDA
cudaError_t Allocator::enablePeerAccess(int dev, int peer) noexcept {
#if VBT_WITH_CUDA
  // Route through async backend if selected for this device
  Allocator& a = Allocator::get(static_cast<DeviceIndex>(dev));
  if (a.cfg_.backend == BackendKind::Async) {
    return AsyncBackend::get(a.dev_).enablePeerAccess(dev, peer);
  }
  int can = 0;
  DeviceGuard dg(static_cast<DeviceIndex>(dev));
  cudaError_t st = cudaDeviceCanAccessPeer(&can, dev, peer);
  if (st != cudaSuccess) { (void)cudaGetLastError(); can = 0; }
  if (!can) return cudaSuccess;
  st = cudaDeviceEnablePeerAccess(peer, 0);
  if (st == cudaErrorPeerAccessAlreadyEnabled) { (void)cudaGetLastError(); return cudaSuccess; }
  return st;
#else
  (void)dev; (void)peer; return cudaSuccess;
#endif
}

#ifdef VBT_INTERNAL_TESTS
namespace testonly {

std::vector<GraphPoolDebugInfo> debug_graph_pools(DeviceIndex dev) {
#if !VBT_WITH_CUDA
  (void)dev;
  return {};
#else
  return Allocator::get(dev).debug_graph_pools_for_testing();
#endif
}

void allocator_debug_gc_pool_now(DeviceIndex dev, MempoolId id) {
#if !VBT_WITH_CUDA
  (void)dev;
  (void)id;
#else
  Allocator::get(dev).debug_gc_pool_now_for_testing(id);
#endif
}

std::vector<MemorySegmentSnapshot> allocator_debug_snapshot(DeviceIndex dev) {
#if !VBT_WITH_CUDA
  (void)dev;
  return {};
#else
  return Allocator::get(dev).debug_allocator_snapshot_for_testing(dev);
#endif
}

} // namespace testonly
#endif // VBT_INTERNAL_TESTS

#endif


}} // namespace vbt::cuda
