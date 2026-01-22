// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>
#include <limits>
#include <vector>

#include "vbt/cuda/allocator.h"   // Allocator, DeviceStats, MemorySegmentSnapshot, GraphPoolSnapshot
#include "vbt/cuda/graphs.h"     // GraphCounters, cuda_graphs_counters, detail::poll_deferred_graph_cleanup

namespace vbt { namespace cuda { namespace testonly {

using Dev = DeviceIndex;

inline Allocator& A(Dev d = 0) {
  return Allocator::get(d);
}

// ---- Core snapshot helpers -------------------------------------------------

inline DeviceStats R(Dev d = 0) {
  return A(d).getDeviceStats();
}

inline std::vector<MemorySegmentSnapshot> S(Dev d = 0) {
  return snapshot_memory_segments(d);
}

inline std::vector<GraphPoolSnapshot> P(Dev d = 0) {
  std::vector<GraphPoolSnapshot> out;
  for (const auto& gp : snapshot_graph_pools(std::nullopt)) {
    if (gp.id.dev == d) {
      out.push_back(gp);
    }
  }
  return out;
}

struct CombinedSnapshot {
  Dev                                dev;
  DeviceStats                        stats;    // R(d)
  std::vector<MemorySegmentSnapshot> segments; // S(d)
  std::vector<GraphPoolSnapshot>     pools;    // P(d)
  GraphCounters                      graphs;   // cuda_graphs_counters()
};

inline CombinedSnapshot take_snapshot(Dev d = 0) {
  CombinedSnapshot cs;
  cs.dev      = d;
  cs.stats    = R(d);
  cs.segments = S(d);
  cs.pools    = P(d);
  cs.graphs   = cuda_graphs_counters();
  return cs;
}

inline std::uint64_t sum_segment_reserved(const CombinedSnapshot& cs) {
  std::uint64_t acc = 0;
  for (const auto& s : cs.segments) {
    acc += s.bytes_reserved;
  }
  return acc;
}

inline std::uint64_t sum_segment_active(const CombinedSnapshot& cs) {
  std::uint64_t acc = 0;
  for (const auto& s : cs.segments) {
    acc += s.bytes_active;
  }
  return acc;
}

inline std::uint64_t pool_reserved_bytes(const CombinedSnapshot& cs,
                                         std::uint64_t pool_id) {
  std::uint64_t acc = 0;
  for (const auto& s : cs.segments) {
    if (s.pool_id == pool_id) {
      acc += s.bytes_reserved;
    }
  }
  return acc;
}

inline std::uint64_t pool_active_bytes(const CombinedSnapshot& cs,
                                       std::uint64_t pool_id) {
  std::uint64_t acc = 0;
  for (const auto& s : cs.segments) {
    if (s.pool_id == pool_id) {
      acc += s.bytes_active;
    }
  }
  return acc;
}

inline std::uint64_t global_reserved_bytes(const CombinedSnapshot& cs) {
  return pool_reserved_bytes(cs, /*pool_id=*/0);
}

inline std::uint64_t global_active_bytes(const CombinedSnapshot& cs) {
  return pool_active_bytes(cs, /*pool_id=*/0);
}

// ---- Backend detection ------------------------------------------------------

inline bool has_native_backend(Dev d = 0) {
#if defined(VBT_INTERNAL_TESTS)
  return A(d).debug_backend_kind_for_testing() == BackendKind::Native;
#else
  (void)d;
  return false;
#endif
}

inline bool has_async_backend(Dev d = 0) {
#if defined(VBT_INTERNAL_TESTS)
  return A(d).debug_backend_kind_for_testing() == BackendKind::Async;
#else
  (void)d;
  return false;
#endif
}

// ---- Memory fraction guard --------------------------------------------------

struct MemoryFractionGuard {
  Dev    dev;
  double old_fraction;

  explicit MemoryFractionGuard(Dev d, double new_fraction)
      : dev(d), old_fraction(0.0) {
#if VBT_WITH_CUDA
    old_fraction = A(d).getMemoryFraction();
    A(d).setMemoryFraction(new_fraction);
#else
    (void)new_fraction;
#endif
  }

  ~MemoryFractionGuard() {
#if VBT_WITH_CUDA
    A(dev).setMemoryFraction(old_fraction);
#endif
  }
};

// ---- Quiescing helpers ------------------------------------------------------

inline void quiesce_allocator_for_setup(Dev d) {
#if VBT_WITH_CUDA
  Allocator& alloc = A(d);
  alloc.process_events(-1);
  alloc.emptyCache();
  alloc.resetAccumulatedStats();
  alloc.resetPeakStats();
#if defined(VBT_INTERNAL_TESTS)
  debug_reset_cudaMalloc_calls_for_testing();
#endif
#else
  (void)d;
#endif
}

inline void drain_allocator_for_snapshots(Dev d) {
#if VBT_WITH_CUDA
  A(d).process_events(-1);
#else
  (void)d;
#endif
}

inline void quiesce_graphs_for_snapshots(Dev d) {
#if VBT_WITH_CUDA
  (void)d;  // device selection is implicit in graphs/allocator internals
  vbt::cuda::detail::poll_deferred_graph_cleanup();
  drain_allocator_for_snapshots(d);
#else
  (void)d;
#endif
}

// ---- Heavy-allocation sizing helper ----------------------------------------

inline std::size_t pick_heavy_allocation_size(Dev d) {
#if !VBT_WITH_CUDA
  (void)d;
  return 0;  // CPU-only builds skip heavy tests.
#else
  Allocator& alloc = A(d);
#if defined(VBT_INTERNAL_TESTS)
  if (alloc.debug_backend_kind_for_testing() != BackendKind::Native) {
    return 0;  // Async or unknown: heavy native tests will skip.
  }
  std::size_t limit = alloc.debug_current_limit_bytes_for_testing();
#else
  std::size_t limit = 0;
#endif
  if (limit == 0 ||
      limit == std::numeric_limits<std::size_t>::max() ||
      limit < (8ull << 20)) {
    // Unknown cap or too small; let tests skip.
    return 0;
  }
  std::size_t raw = limit / 8;                     // conservative heavy size
  if (raw > (256ull << 20)) raw = 256ull << 20;    // cap at 256 MiB
  std::size_t rounded = round_size(raw);
  if (rounded == 0) {
    rounded = raw;
  }
  return rounded;
#endif
}

}}} // namespace vbt::cuda::testonly
