// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/fabric_state.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>

#include "vbt/cuda/device.h"
#include "vbt/cuda/allocator.h"
#include "vbt/cuda/fabric_events.h"
#include "vbt/cuda/fabric_addmul_decision.h"

#ifndef VBT_WITH_CUDA
#  define VBT_WITH_CUDA 0
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1,
              "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#  include <cuda_runtime_api.h>
#endif

namespace vbt { namespace cuda { namespace fabric {

namespace {

constexpr const char* kFabricNoCudaMessage =
    "[Fabric] Built without CUDA or no CUDA devices are available; Fabric is disabled";
constexpr const char* kFabricUvaInvariantMessage =
    "[Fabric] UVA invariant violated on this platform; Fabric is disabled";

struct FabricGlobals {
  FabricState        state{};
  std::once_flag     init_flag{};
  std::atomic<bool>  initialized{false};
};

static FabricEventsMode parse_events_mode_from_env() noexcept {
  const char* raw = std::getenv("VBT_FABRIC_EVENTS_MODE");
  if (!raw || *raw == '\0') {
    return FabricEventsMode::kOff;
  }

  // Trim leading whitespace.
  while (*raw != '\0' && std::isspace(static_cast<unsigned char>(*raw))) {
    ++raw;
  }

  auto to_lower = [](unsigned char c) noexcept {
    return static_cast<char>(std::tolower(c));
  };

  auto equals_ci = [&](const char* s, const char* lit) noexcept {
    while (*s != '\0' && *lit != '\0') {
      if (to_lower(static_cast<unsigned char>(*s)) != *lit) {
        return false;
      }
      ++s;
      ++lit;
    }
    if (*lit != '\0') {
      return false;
    }
    // Accept either exact match or trailing whitespace after the literal.
    return *s == '\0' || std::isspace(static_cast<unsigned char>(*s));
  };

  if (equals_ci(raw, "basic") || equals_ci(raw, "1")) {
    return FabricEventsMode::kBasic;
  }
  if (equals_ci(raw, "off") || equals_ci(raw, "0")) {
    return FabricEventsMode::kOff;
  }

  return FabricEventsMode::kOff;
}

static void reset_stats_struct(FabricStats& stats) noexcept {
  stats.mesh_builds.store(0, std::memory_order_relaxed);
  stats.p2p_pairs_enabled.store(0, std::memory_order_relaxed);
  stats.p2p_pairs_failed.store(0, std::memory_order_relaxed);

  stats.fabric_ops_attempted.store(0, std::memory_order_relaxed);
  stats.fabric_ops_hit.store(0, std::memory_order_relaxed);
  stats.fabric_ops_fallback.store(0, std::memory_order_relaxed);

  stats.remote_bytes_read.store(0, std::memory_order_relaxed);
  stats.remote_bytes_written.store(0, std::memory_order_relaxed);

  stats.reasons.no_p2p.store(0, std::memory_order_relaxed);
  stats.reasons.requires_grad.store(0, std::memory_order_relaxed);
  stats.reasons.in_backward.store(0, std::memory_order_relaxed);
  stats.reasons.small_tensor.store(0, std::memory_order_relaxed);

  stats.inflight_ops_current.store(0, std::memory_order_relaxed);
  stats.inflight_ops_peak.store(0, std::memory_order_relaxed);

  stats.event_queue_len_peak.store(0, std::memory_order_relaxed);
  stats.event_dropped_total.store(0, std::memory_order_relaxed);
  stats.event_failures_total.store(0, std::memory_order_relaxed);

  stats.mode_enable_calls.store(0, std::memory_order_relaxed);
  stats.mode_disable_calls.store(0, std::memory_order_relaxed);
  stats.mode_set_failures.store(0, std::memory_order_relaxed);

  for (int i = 0; i < FabricStats::kMaxFabricDevices; ++i) {
    stats.ops_as_primary[i].store(0, std::memory_order_relaxed);
    stats.ops_as_remote[i].store(0, std::memory_order_relaxed);
    stats.remote_bytes_read_by_device[i].store(0, std::memory_order_relaxed);
    stats.remote_bytes_written_by_device[i].store(0, std::memory_order_relaxed);
  }
}

void reset_state_struct(FabricState& state) noexcept {
  state.topology = FabricTopology{};
  state.config.mode.store(FabricMode::Disabled, std::memory_order_relaxed);
  state.config.events_mode_raw.store(0, std::memory_order_relaxed);
  state.config.event_lifetime_enabled.store(false, std::memory_order_relaxed);
  state.init_stats = FabricInitStats{};
  reset_stats_struct(state.stats);
  state.next_op_id.store(0, std::memory_order_relaxed);
  state.uva_ok = false;
  state.init_status = FabricInitStatus::Uninitialized;
  state.disable_reason.clear();
}

FabricGlobals& globals() {
  static FabricGlobals g;
  return g;
}

static void enable_all_p2p_pairs_best_effort(FabricState& state) noexcept {
#if !VBT_WITH_CUDA
  (void)state;
#else
  FabricTopology& topo = state.topology;
  const int n = topo.device_count;
  if (n < 2) return;

  // Defensive: require matrices are well-formed.
  if (topo.can_access_peer.size() != static_cast<std::size_t>(n) ||
      topo.p2p_enabled.size() != static_cast<std::size_t>(n)) {
    return;
  }

  // Internal tests may install synthetic topologies whose device_count exceeds
  // the runtime device count. Clamp the enablement loop to the runtime view so
  // we never call CUDA APIs with an invalid device ordinal.
  int runtime_dc = 0;
  try {
    runtime_dc = vbt::cuda::device_count();
  } catch (...) {
    runtime_dc = n;
  }
  if (runtime_dc < 0) runtime_dc = 0;

  const int limit = std::min(n, runtime_dc);
  if (limit < 2) return;

  for (int i = 0; i < limit; ++i) {
    if (topo.can_access_peer[i].size() != static_cast<std::size_t>(n) ||
        topo.p2p_enabled[i].size() != static_cast<std::size_t>(n)) {
      return;
    }
  }

  for (int i = 0; i < limit; ++i) {
    for (int j = i + 1; j < limit; ++j) {
      if (!topo.can_access_peer[i][j] || !topo.can_access_peer[j][i]) {
        continue;
      }

      // Idempotent: don't double-count already-enabled pairs.
      if (topo.p2p_enabled[i][j] && topo.p2p_enabled[j][i]) {
        continue;
      }

      cudaError_t st01 = vbt::cuda::Allocator::enablePeerAccess(i, j);
      cudaError_t st10 = vbt::cuda::Allocator::enablePeerAccess(j, i);
      if (st01 == cudaSuccess && st10 == cudaSuccess) {
        topo.p2p_enabled[i][j] = true;
        topo.p2p_enabled[j][i] = true;
        state.stats.p2p_pairs_enabled.fetch_add(1, std::memory_order_relaxed);
      } else {
        state.stats.p2p_pairs_failed.fetch_add(1, std::memory_order_relaxed);
        (void)cudaGetLastError();
      }
    }
  }
#endif
}

}  // namespace

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
namespace {
FabricTestHooks& hooks_singleton() {
  static FabricTestHooks hooks;
  return hooks;
}
}  // namespace

FabricTestHooks& fabric_test_hooks() noexcept {
  return hooks_singleton();
}

// Forward declaration; implemented below.
static void init_fabric_state_impl(FabricState& state) noexcept;

void reset_fabric_state_for_tests() {
  FabricGlobals& g = globals();
  reset_state_struct(g.state);
  g.initialized.store(false, std::memory_order_release);
  // Reinitialize the once_flag so that the next fabric_state() call will
  // rerun initialization with any updated test hooks. This uses placement
  // new to reconstruct the flag in place; safe because std::once_flag has a
  // trivial destructor.
  new (&g.init_flag) std::once_flag();
}

void reset_fabric_stats_for_tests() {
  FabricGlobals& g = globals();
  reset_stats_struct(g.state.stats);
}

bool run_uva_self_test(
    const FabricTopology& topo,
    FabricInitStats* stats,
    std::string* disable_reason) noexcept {
  if (stats) {
    ++stats->uva_self_test_attempts;
  }

  if (topo.device_count <= 1) {
    // Vacuously OK for 0/1 GPUs; the higher-level gate will still disable
    // Fabric ops when fewer than 2 devices are present.
    return true;
  }

#if !VBT_WITH_CUDA
  (void)disable_reason;
  // CPU-only build: treat UVA as unavailable but do not attempt to call any
  // disabled.
  if (stats) {
    ++stats->uva_self_test_failures;
  }
  if (disable_reason && disable_reason->empty()) {
    *disable_reason = kFabricUvaInvariantMessage;
  }
  return false;
#else
  // Test hook: allow forcing UVA success/failure without calling CUDA APIs.
  if (auto& hooks = fabric_test_hooks(); hooks.forced_uva_ok.has_value()) {
    const bool ok = *hooks.forced_uva_ok;
    if (!ok && stats) {
      ++stats->uva_self_test_failures;
    }
    if (!ok && disable_reason && disable_reason->empty()) {
      *disable_reason = kFabricUvaInvariantMessage;
    }
    return ok;
  }

  try {
    const int n = topo.device_count;
    if (topo.clique_id.size() != static_cast<std::size_t>(n) ||
        topo.clique_size.empty()) {
      // Topology invariants are not satisfied; treat as a UVA failure.
      if (stats) {
        ++stats->uva_self_test_failures;
      }
      if (disable_reason && disable_reason->empty()) {
        *disable_reason = kFabricUvaInvariantMessage;
      }
      return false;
    }

    // Examine only devices that belong to cliques of size >= 2.
    for (int d = 0; d < n; ++d) {
      const int cid = topo.clique_id[d];
      if (cid < 0 || cid >= static_cast<int>(topo.clique_size.size())) {
        continue;
      }
      if (topo.clique_size[cid] < 2) {
        continue;
      }
      int attr = 0;
      cudaError_t st = cudaDeviceGetAttribute(&attr, cudaDevAttrUnifiedAddressing, d);
      if (st != cudaSuccess || attr == 0) {
        if (stats) {
          ++stats->uva_self_test_failures;
        }
        if (disable_reason && disable_reason->empty()) {
          *disable_reason = kFabricUvaInvariantMessage;
        }
        return false;
      }
    }

    return true;
  } catch (...) {
    if (stats) {
      ++stats->uva_self_test_failures;
    }
    if (disable_reason && disable_reason->empty()) {
      *disable_reason = kFabricUvaInvariantMessage;
    }
    return false;
  }
#endif  // VBT_WITH_CUDA
}

#endif  // VBT_INTERNAL_TESTS

static void init_fabric_state_impl(FabricState& state) noexcept {
  // Start from a clean slate.
  reset_state_struct(state);

  // Initialize events mode from env unless it has been set explicitly already.
  if (!fabric_events_mode_was_explicitly_set()) {
    const FabricEventsMode env_mode = parse_events_mode_from_env();
    set_fabric_events_mode(env_mode);
  }
  state.config.events_mode_raw.store(
      static_cast<std::uint8_t>(get_fabric_events_mode()),
      std::memory_order_relaxed);

#if !VBT_WITH_CUDA
  state.topology.device_count = 0;
  state.uva_ok = false;
  state.init_status = FabricInitStatus::NoCuda;
  state.disable_reason = kFabricNoCudaMessage;
  return;
#else
  std::string disable_reason;
  try {
    build_fabric_topology_from_runtime(state.topology, &state.init_stats,
                                       &disable_reason);
  } catch (const std::exception& e) {
    state.topology = FabricTopology{};
    state.uva_ok = false;
    state.init_status = FabricInitStatus::CudaError;
    state.disable_reason = disable_reason.empty() ? e.what() : disable_reason;
    return;
  } catch (...) {
    state.topology = FabricTopology{};
    state.uva_ok = false;
    state.init_status = FabricInitStatus::CudaError;
    if (disable_reason.empty()) {
      state.disable_reason =
          "[Fabric] CUDA error during fabric initialization; Fabric is disabled";
    } else {
      state.disable_reason = disable_reason;
    }
    return;
  }

  if (state.topology.device_count == 0) {
    state.uva_ok = false;
    state.init_status = FabricInitStatus::NoCuda;
    state.disable_reason = kFabricNoCudaMessage;
    return;
  }

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
  bool ok = run_uva_self_test(state.topology, &state.init_stats, &disable_reason);
#else
  // In production builds, perform a minimal UVA self-test guarded by CUDA
  // Unified Addressing attributes.
  if (state.init_stats.uva_self_test_attempts == 0) {
    ++state.init_stats.uva_self_test_attempts;
  }

  bool ok = true;
  if (state.topology.device_count > 1) {
    const int n = state.topology.device_count;
    for (int d = 0; d < n; ++d) {
      const int cid = d < static_cast<int>(state.topology.clique_id.size())
                          ? state.topology.clique_id[d]
                          : -1;
      if (cid < 0 || cid >= static_cast<int>(state.topology.clique_size.size())) {
        continue;
      }
      if (state.topology.clique_size[cid] < 2) {
        continue;
      }
      int attr = 0;
      cudaError_t st = cudaDeviceGetAttribute(&attr, cudaDevAttrUnifiedAddressing, d);
      if (st != cudaSuccess || attr == 0) {
        ok = false;
        ++state.init_stats.uva_self_test_failures;
        if (disable_reason.empty()) {
          disable_reason = kFabricUvaInvariantMessage;
        }
        break;
      }
    }
  }
#endif  // VBT_INTERNAL_TESTS

  state.uva_ok = ok;
  if (ok) {
    state.init_status = FabricInitStatus::Ok;
    state.disable_reason.clear();
    enable_all_p2p_pairs_best_effort(state);
  } else {
    state.init_status = FabricInitStatus::UvaFailed;
    if (disable_reason.empty()) {
      state.disable_reason = kFabricUvaInvariantMessage;
    } else {
      state.disable_reason = disable_reason;
    }
  }
#endif  // VBT_WITH_CUDA
}

FabricState& fabric_state() noexcept {
  FabricGlobals& g = globals();
  std::call_once(g.init_flag, [&]() {
    init_fabric_state_impl(g.state);
    g.initialized.store(true, std::memory_order_release);
  });
  return g.state;
}

FabricState* try_get_fabric_state_if_initialized() noexcept {
  FabricGlobals& g = globals();
  if (!g.initialized.load(std::memory_order_acquire)) {
    return nullptr;
  }
  return &g.state;
}

bool fabric_enabled_for_ops(const FabricState& fs) noexcept {
  FabricMode mode = fs.config.mode.load(std::memory_order_acquire);
  if (mode == FabricMode::Disabled) return false;

  if (fs.init_status != FabricInitStatus::Ok) return false;
  if (!fs.uva_ok) return false;

  if (fs.topology.device_count < 2) return false;

  const auto& cid = fs.topology.clique_id;
  const auto& csz = fs.topology.clique_size;
  const int n = fs.topology.device_count;
  for (int d = 0; d < n; ++d) {
    const int c = (d < static_cast<int>(cid.size())) ? cid[d] : -1;
    if (c >= 0 && c < static_cast<int>(csz.size()) && csz[c] >= 2) {
      return true;
    }
  }
  return false;
}

bool is_fabric_event_lifetime_enabled() noexcept {
  return fabric_state().config.event_lifetime_enabled.load(std::memory_order_acquire);
}

void set_fabric_event_lifetime_enabled(bool enabled) noexcept {
  FabricState& fs = fabric_state();
  fs.config.event_lifetime_enabled.store(enabled, std::memory_order_release);

  FabricEvent ev;
  ev.kind = FabricEventKind::kEventLifetimeToggled;
  ev.level = FabricEventLevel::kInfo;
  ev.message = enabled ? kFabricEventMsgEventLifetimeEnabled
                       : kFabricEventMsgEventLifetimeDisabled;
  record_fabric_event(std::move(ev));
}

FabricTopologySnapshot fabric_topology_snapshot() {
  const FabricState& fs = fabric_state();
  FabricTopologySnapshot snap;

  snap.device_count = fs.topology.device_count;
  snap.uva_ok = fs.uva_ok;
  snap.mode = fs.config.mode.load(std::memory_order_relaxed);
  snap.init_status = fs.init_status;
  snap.disable_reason = fs.disable_reason;

  const int n = fs.topology.device_count;
  const auto& cid = fs.topology.clique_id;
  const auto& csz = fs.topology.clique_size;

  // Build cliques as dense [0, num_cliques) container of device lists.
  const int num_cliques = static_cast<int>(csz.size());
  std::vector<std::vector<int>> clique_devices(
      num_cliques, std::vector<int>{});

  for (int d = 0; d < n; ++d) {
    if (d >= static_cast<int>(cid.size())) continue;
    int c = cid[d];
    if (c < 0 || c >= num_cliques) continue;
    clique_devices[c].push_back(d);
  }

  for (int c = 0; c < num_cliques; ++c) {
    FabricCliqueSnapshot cs;
    cs.id = c;
    cs.devices = std::move(clique_devices[c]);
    snap.cliques.push_back(std::move(cs));
  }

  return snap;
}

FabricStatsSnapshot fabric_stats_snapshot() {
  const FabricState& fs = fabric_state();
  FabricStatsSnapshot snap;

  snap.mesh_builds = fs.stats.mesh_builds.load(std::memory_order_relaxed);
  snap.p2p_pairs_enabled = fs.stats.p2p_pairs_enabled.load(std::memory_order_relaxed);
  snap.p2p_pairs_failed = fs.stats.p2p_pairs_failed.load(std::memory_order_relaxed);

  snap.fabric_ops_attempted = fs.stats.fabric_ops_attempted.load(std::memory_order_relaxed);
  snap.fabric_ops_hit = fs.stats.fabric_ops_hit.load(std::memory_order_relaxed);
  snap.fabric_ops_fallback = fs.stats.fabric_ops_fallback.load(std::memory_order_relaxed);

  snap.remote_bytes_read = fs.stats.remote_bytes_read.load(std::memory_order_relaxed);
  snap.remote_bytes_written = fs.stats.remote_bytes_written.load(std::memory_order_relaxed);

  snap.inflight_ops_current = fs.stats.inflight_ops_current.load(std::memory_order_relaxed);
  snap.inflight_ops_peak = fs.stats.inflight_ops_peak.load(std::memory_order_relaxed);

  snap.event_queue_len_peak = fs.stats.event_queue_len_peak.load(std::memory_order_relaxed);
  snap.event_dropped_total = fs.stats.event_dropped_total.load(std::memory_order_relaxed);
  snap.event_failures_total = fs.stats.event_failures_total.load(std::memory_order_relaxed);

  snap.mode_enable_calls = fs.stats.mode_enable_calls.load(std::memory_order_relaxed);
  snap.mode_disable_calls = fs.stats.mode_disable_calls.load(std::memory_order_relaxed);
  snap.mode_set_failures = fs.stats.mode_set_failures.load(std::memory_order_relaxed);

  snap.reasons.no_p2p = fs.stats.reasons.no_p2p.load(std::memory_order_relaxed);
  snap.reasons.requires_grad = fs.stats.reasons.requires_grad.load(std::memory_order_relaxed);
  snap.reasons.in_backward = fs.stats.reasons.in_backward.load(std::memory_order_relaxed);
  snap.reasons.small_tensor = fs.stats.reasons.small_tensor.load(std::memory_order_relaxed);

  const int n = fs.topology.device_count;
  const int limit = std::min(n, FabricStats::kMaxFabricDevices);
  snap.per_device.reserve(static_cast<std::size_t>(limit));
  for (int d = 0; d < limit; ++d) {
    FabricPerDeviceStatsSnapshot ds;
    ds.device_index = d;
    ds.ops_as_primary = fs.stats.ops_as_primary[d].load(std::memory_order_relaxed);
    ds.ops_as_remote = fs.stats.ops_as_remote[d].load(std::memory_order_relaxed);
    ds.remote_bytes_read = fs.stats.remote_bytes_read_by_device[d].load(std::memory_order_relaxed);
    ds.remote_bytes_written =
        fs.stats.remote_bytes_written_by_device[d].load(std::memory_order_relaxed);
    snap.per_device.push_back(ds);
  }

  return snap;
}

std::uint64_t compute_fabric_bytes(const FabricAddMulDecision& dec,
                                  std::size_t itemsize) noexcept {
  if (dec.numel <= 0) return 0;
  if (itemsize == 0) return 0;

  const std::uint64_t n = static_cast<std::uint64_t>(dec.numel);
  const std::uint64_t item = static_cast<std::uint64_t>(itemsize);

  if (n > (std::numeric_limits<std::uint64_t>::max() / item)) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return n * item;
}

void record_fabric_decision_stats(FabricStats& stats,
                                 const FabricAddMulDecision& dec,
                                 std::size_t itemsize) noexcept {
  // Same-device ops do not contribute to Fabric decision stats.
  if (dec.other_device < 0) {
    return;
  }

  stats.fabric_ops_attempted.fetch_add(1, std::memory_order_relaxed);

  const int primary = dec.primary_device;
  const int other = dec.other_device;

  if (primary >= 0 && primary < FabricStats::kMaxFabricDevices) {
    stats.ops_as_primary[primary].fetch_add(1, std::memory_order_relaxed);
  }
  if (other >= 0 && other < FabricStats::kMaxFabricDevices) {
    stats.ops_as_remote[other].fetch_add(1, std::memory_order_relaxed);
  }

  if (dec.use_fabric) {
    stats.fabric_ops_hit.fetch_add(1, std::memory_order_relaxed);
  } else if (dec.use_copy_fallback) {
    stats.fabric_ops_fallback.fetch_add(1, std::memory_order_relaxed);
  }

  // NOTE: FabricAddMulFallbackReason currently has no small-tensor reason.
  // stats.reasons.small_tensor is reserved for future heuristics.
  switch (dec.reason) {
    case FabricAddMulFallbackReason::kNotInSameCliqueOrNoP2P:
      stats.reasons.no_p2p.fetch_add(1, std::memory_order_relaxed);
      break;
    case FabricAddMulFallbackReason::kRequiresGrad:
      stats.reasons.requires_grad.fetch_add(1, std::memory_order_relaxed);
      break;
    case FabricAddMulFallbackReason::kInBackward:
      stats.reasons.in_backward.fetch_add(1, std::memory_order_relaxed);
      break;
    default:
      break;
  }

  // Bytes are counted only for executed work (Fabric hit or copy fallback).
  if (!(dec.use_fabric || dec.use_copy_fallback)) {
    return;
  }

  const std::uint64_t bytes = compute_fabric_bytes(dec, itemsize);

  stats.remote_bytes_read.fetch_add(bytes, std::memory_order_relaxed);
  stats.remote_bytes_written.fetch_add(bytes, std::memory_order_relaxed);

  // Attribute bytes to the primary device (the device performing the compute).
  // Remote devices are reflected via ops_as_remote (bytes are not currently role-split).
  if (primary >= 0 && primary < FabricStats::kMaxFabricDevices) {
    stats.remote_bytes_read_by_device[primary].fetch_add(bytes, std::memory_order_relaxed);
    stats.remote_bytes_written_by_device[primary].fetch_add(bytes, std::memory_order_relaxed);
  }
}

}}} // namespace vbt::cuda::fabric
