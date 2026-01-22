// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "vbt/cuda/fabric_topology.h"

namespace vbt { namespace cuda { namespace fabric {

enum class FabricMode : std::uint8_t {
  Disabled   = 0,
  BestEffort = 1,
  DryRun     = 2,
};

// in-place (e.g., eager_mesh, allow_partial) consistent with the parent
// Fabric design.
struct FabricConfig {
  std::atomic<FabricMode> mode{FabricMode::Disabled};

  // Snapshot of the current Fabric events mode as a raw enum value.
  // This mirrors vbt::cuda::fabric::FabricEventsMode (see fabric_events.h) but is
  // stored here as a byte to avoid pulling fabric_events.h into this header.
  std::atomic<std::uint8_t> events_mode_raw{0};

  std::atomic<bool>       event_lifetime_enabled{false};
};

// Initialization status classification for the Fabric subsystem.
enum class FabricInitStatus : std::uint8_t {
  Uninitialized = 0,  // before first fabric_state() call
  Ok,                 // discovery enabled, >= 1 device, UVA self-test passed
  NoCuda,             // no CUDA support or 0 devices
  UvaFailed,          // UVA unsupported or self-test failed
  CudaError,          // unexpected exception during topology or UVA init
};

// Minimal init statistics used to pin "one-time" behavior in tests.
struct FabricInitStats {
  std::uint64_t topology_build_attempts{0};
  std::uint64_t topology_build_failures{0};
  std::uint64_t uva_self_test_attempts{0};
  std::uint64_t uva_self_test_failures{0};
};

// Observability counters for Fabric behavior.
//
// These are best-effort statistics intended for debugging and performance
// tuning; they are not used to make correctness decisions.
struct FabricStatsReasons {
  std::atomic<std::uint64_t> no_p2p{0};
  std::atomic<std::uint64_t> requires_grad{0};
  std::atomic<std::uint64_t> in_backward{0};
  std::atomic<std::uint64_t> small_tensor{0};
};

struct FabricStats {
  // Topology / init (reserved for future use).
  std::atomic<std::uint64_t> mesh_builds{0};
  std::atomic<std::uint64_t> p2p_pairs_enabled{0};
  std::atomic<std::uint64_t> p2p_pairs_failed{0};

  // Op-level behavior.
  std::atomic<std::uint64_t> fabric_ops_attempted{0};
  std::atomic<std::uint64_t> fabric_ops_hit{0};
  std::atomic<std::uint64_t> fabric_ops_fallback{0};

  // Data volume (approximate).
  std::atomic<std::uint64_t> remote_bytes_read{0};
  std::atomic<std::uint64_t> remote_bytes_written{0};

  FabricStatsReasons reasons;

  // Host-call backpressure for cross-device Fabric candidates.
  std::atomic<std::uint64_t> inflight_ops_current{0};
  std::atomic<std::uint64_t> inflight_ops_peak{0};

  // Event ring health.
  std::atomic<std::uint64_t> event_queue_len_peak{0};
  std::atomic<std::uint64_t> event_dropped_total{0};
  std::atomic<std::uint64_t> event_failures_total{0};

  // Mode / gate transitions.
  std::atomic<std::uint64_t> mode_enable_calls{0};
  std::atomic<std::uint64_t> mode_disable_calls{0};
  std::atomic<std::uint64_t> mode_set_failures{0};

  // Per-device aggregates. Only meaningful for device_count <= kMaxFabricDevices.
  static constexpr int kMaxFabricDevices = 64;
  std::array<std::atomic<std::uint64_t>, kMaxFabricDevices> ops_as_primary{};
  std::array<std::atomic<std::uint64_t>, kMaxFabricDevices> ops_as_remote{};
  std::array<std::atomic<std::uint64_t>, kMaxFabricDevices> remote_bytes_read_by_device{};
  std::array<std::atomic<std::uint64_t>, kMaxFabricDevices> remote_bytes_written_by_device{};
};


// Global Fabric state snapshot (process-global singleton owned by
// fabric_state()). In production code this is initialized exactly once and
// thereafter treated as read-mostly. Test-only reset helpers allow carefully
// scoped reinitialization under VBT_INTERNAL_TESTS.
struct FabricState {
  FabricTopology   topology;
  FabricConfig     config;
  FabricInitStats  init_stats;
  FabricStats      stats;

  // Monotonic per-process logical op id used by Fabric diagnostic events.
  std::atomic<std::uint64_t> next_op_id{0};

  bool             uva_ok{false};
  FabricInitStatus init_status{FabricInitStatus::Uninitialized};

  // Non-empty iff init_status != Ok.
  std::string      disable_reason;
};

// Accessor for the process-global Fabric state singleton.
FabricState& fabric_state() noexcept;

// Internal helper used by best-effort observability code to avoid forcing
// Fabric initialization.
//
// Returns nullptr if fabric_state() has not been initialized yet.
FabricState* try_get_fabric_state_if_initialized() noexcept;

// Python helpers but is designed for later Dispatcher/TensorIterator
// integration.
bool fabric_enabled_for_ops(const FabricState& fs) noexcept;

bool is_fabric_event_lifetime_enabled() noexcept;
void set_fabric_event_lifetime_enabled(bool enabled) noexcept;

// Snapshot types used for Python bindings; these are simple value types and
// never call CUDA. They are intended to be trivially serializable to Python.
struct FabricCliqueSnapshot {
  int id{0};
  std::vector<int> devices;  // sorted ascending
};

struct FabricTopologySnapshot {
  int              device_count{0};
  bool             uva_ok{false};
  FabricMode       mode{FabricMode::Disabled};
  FabricInitStatus init_status{FabricInitStatus::Uninitialized};
  std::string      disable_reason;
  std::vector<FabricCliqueSnapshot> cliques;
};

struct FabricStatsReasonsSnapshot {
  std::uint64_t no_p2p{0};
  std::uint64_t requires_grad{0};
  std::uint64_t in_backward{0};
  std::uint64_t small_tensor{0};
};

struct FabricPerDeviceStatsSnapshot {
  // 0-based CUDA device index in the FabricTopology numbering.
  int device_index{0};
  std::uint64_t ops_as_primary{0};
  std::uint64_t ops_as_remote{0};
  // Bytes for which this device acted as the primary (compute) device on a
  // cross-device Fabric candidate. Summing over devices matches the global
  // remote_bytes_* counters.
  std::uint64_t remote_bytes_read{0};
  std::uint64_t remote_bytes_written{0};
};

struct FabricStatsSnapshot {
  std::uint64_t mesh_builds{0};
  std::uint64_t p2p_pairs_enabled{0};
  std::uint64_t p2p_pairs_failed{0};

  std::uint64_t fabric_ops_attempted{0};
  std::uint64_t fabric_ops_hit{0};
  std::uint64_t fabric_ops_fallback{0};

  std::uint64_t remote_bytes_read{0};
  std::uint64_t remote_bytes_written{0};

  std::uint64_t inflight_ops_current{0};
  std::uint64_t inflight_ops_peak{0};

  std::uint64_t event_queue_len_peak{0};
  std::uint64_t event_dropped_total{0};
  std::uint64_t event_failures_total{0};

  std::uint64_t mode_enable_calls{0};
  std::uint64_t mode_disable_calls{0};
  std::uint64_t mode_set_failures{0};

  FabricStatsReasonsSnapshot reasons;
  std::vector<FabricPerDeviceStatsSnapshot> per_device;
};

// Forward declaration (defined in fabric_addmul_decision.h).
struct FabricAddMulDecision;

// Compute bytes associated with a 2-device Fabric add/mul decision.
//
// This is used only for best-effort observability; it must never throw.
std::uint64_t compute_fabric_bytes(const FabricAddMulDecision& dec,
                                  std::size_t itemsize) noexcept;

// Single source of truth for decision-driven Fabric add/mul stats.
//
// - Early-returns if dec.other_device < 0 (same-device op).
// - Callers intentionally exclude kInvalidComputeDevice decisions from stats;
//   those configuration errors surface as hard failures only.
// - Uses relaxed atomics only.
// - Must never throw.
void record_fabric_decision_stats(FabricStats& stats,
                                 const FabricAddMulDecision& dec,
                                 std::size_t itemsize) noexcept;

// RAII helper tracking host-call concurrency for cross-device Fabric add/mul.
struct FabricInflightGuard {
  FabricStats* stats{nullptr};
  bool active{false};

  FabricInflightGuard(FabricStats* s, bool cross_device_inputs) noexcept
      : stats(s), active(s != nullptr && cross_device_inputs) {
    if (!active) return;

    const std::uint64_t cur =
        stats->inflight_ops_current.fetch_add(1, std::memory_order_relaxed) + 1;

    std::uint64_t peak = stats->inflight_ops_peak.load(std::memory_order_relaxed);
    while (cur > peak &&
           !stats->inflight_ops_peak.compare_exchange_weak(
               peak, cur,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
      // Retry until we update the max.
    }
  }

  ~FabricInflightGuard() noexcept {
    if (!active) return;
    stats->inflight_ops_current.fetch_sub(1, std::memory_order_relaxed);
  }

  FabricInflightGuard(const FabricInflightGuard&) = delete;
  FabricInflightGuard& operator=(const FabricInflightGuard&) = delete;
};

// Return a POD snapshot of the current Fabric state for Python.
FabricTopologySnapshot fabric_topology_snapshot();

// Return a POD snapshot of Fabric stats for Python.
FabricStatsSnapshot fabric_stats_snapshot();

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS

// Test hooks for overriding topology discovery and UVA gate behavior in
// unit tests. Production builds must compile out or ignore these hooks.
struct FabricTestHooks {
  // If set, overrides the result of run_uva_self_test for device_count >= 2
  // without calling any CUDA APIs.
  std::optional<bool> forced_uva_ok;

  // If set, called instead of build_fabric_topology_from_runtime to populate
  // topo. Used to synthesize 0/1/2/3-GPU topologies in tests.
  std::function<void(FabricTopology&)> fake_topology_builder;
};

FabricTestHooks& fabric_test_hooks() noexcept;

// Test-only reset helper. Resets the internal FabricState singleton and its
// initialization flag so that the next call to fabric_state() re-runs
// initialization and respects updated hooks. Must only be called in
// single-threaded test contexts.
void reset_fabric_state_for_tests();

// Reset FabricStats counters to zero.
//
// Precondition: no in-flight Fabric ops and no concurrent readers.
void reset_fabric_stats_for_tests();

// Expose UVA self-test helper to tests so that they can pin its behavior
// directly without going through the full fabric_state initialization path.
bool run_uva_self_test(
    const FabricTopology& topo,
    FabricInitStats* stats,
    std::string* disable_reason) noexcept;

#endif  // VBT_INTERNAL_TESTS

}}} // namespace vbt::cuda::fabric
