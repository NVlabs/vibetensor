// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <ostream>

#include "vbt/cuda/allocator.h"   // Allocator, DeviceStats, MemorySegmentSnapshot, GraphPoolSnapshot
#include "vbt/cuda/device.h"     // device_count
#include "vbt/cuda/graphs.h"     // GraphCounters, cuda_graphs_counters
#include "vbt/cuda/stream.h"     // Stream, getStreamFromPool
#include "../cuda_graphs_allocator_test_helpers.h"  // CombinedSnapshot helpers

namespace vbt {
namespace cuda {
namespace testonly {

// Perf scenario identifiers for allocator perf harness.
enum class ScenarioId {
  B1,  // Native eager small-allocation throughput
  B2,  // Native fraction-cap stress
  B3,  // Native graphs pool vs global comparisons
  B4   // Async backend graphs workload
};

// Which binary / implementation is producing the record.
enum class Runner {
  CppNative,
  CppAsync,
};

// Run-mode controls iteration counts.
enum class RunMode {
  Smoke,
  Normal,
  Heavy,
};

// Basic iteration counts resolved from run-mode and overrides.
struct RunCounts {
  int warmup_iters{0};
  int measure_iters{0};
  int repeats{0};
};

struct HostInfo {
  std::string hostname;
  std::string os;
  std::string os_release;
  std::string cpu_model;
  int         cpu_cores_logical{0};
  int         cpu_cores_physical{0};
  double      memory_gb{0.0};
};

struct DeviceInfo {
  int           index{-1};
  std::string   type;                 // "cuda" or "none"
  std::string   name;
  std::uint64_t total_memory_bytes{0};
  std::string   compute_capability;   // e.g. "8.0"
  std::string   driver_version;
  std::string   runtime_version;
  std::string   backend;              // "native" | "async" | "none" | "unknown"
};

struct AllocatorConfig {
  double        per_process_memory_fraction{1.0};
  std::uint64_t max_split_size_bytes{0};
  std::uint64_t max_non_split_rounding_bytes{0};
  std::uint64_t roundup_tolerance_bytes{0};
};

struct AllocatorInfo {
  std::string    backend;   // "native" | "async" | "none" | "unknown"
  AllocatorConfig config{};
  std::string    config_signature;  // v1:<16-hex-digits>
};

struct RunSummaryStats {
  int    warmup_iters{0};
  int    measure_iters{0};
  int    repeats{0};
  int    total_iters{0};
  int    num_replays{0};  // 0 for eager-only scenarios
  double wall_mean_ms{0.0};
  double wall_median_ms{0.0};
  double wall_p95_ms{0.0};
  double wall_min_ms{0.0};
  double wall_max_ms{0.0};
};

struct AllocatorDeltaMetrics {
  std::uint64_t fraction_cap_breaches_delta{0};
  std::uint64_t fraction_cap_misfires_delta{0};
  std::uint64_t gc_passes_delta{0};
  std::uint64_t gc_reclaimed_bytes_delta{0};
};

struct GraphsDeltaMetrics {
  std::uint64_t captures_ended_delta{0};
  std::uint64_t graphs_replayed_delta{0};
  std::uint64_t allocator_capture_denied_delta{0};
};

struct SnapshotMetrics {
  std::uint64_t global_reserved_bytes{0};
  std::uint64_t global_active_bytes{0};
  std::uint64_t pool_reserved_bytes{0};
  std::uint64_t pool_active_bytes{0};
  std::uint64_t total_segments{0};
  std::uint64_t total_pools{0};
};

struct Metrics {
  double        median_ms_per_iter{0.0};
  double        p95_ms_per_iter{0.0};
  std::uint64_t peak_reserved_bytes{0};
  std::uint64_t pool_peak_reserved_bytes{0};
  std::uint64_t peak_allocated_bytes{0};
  std::uint64_t pool_peak_allocated_bytes{0};
  AllocatorDeltaMetrics allocator{};
  GraphsDeltaMetrics    graphs{};
  SnapshotMetrics       snapshot{};
  std::string           skip_reason;   // empty => no skip
  std::string           skip_message;  // optional human-readable message
};

// Scenario configuration resolved from CLI options.
struct PerfConfig {
  ScenarioId   scenario_id{ScenarioId::B1};
  Runner       runner{Runner::CppNative};
  RunMode      run_mode{RunMode::Normal};
  int          device_index{0};
  RunCounts    counts{};     // resolved counts for this run
  int          num_replays{0};
  std::string  notes;        // free-form note string
};

// Aggregate result record (one JSON object per run).
struct PerfResult {
  int                     schema_version{1};
  std::string             component{"m_allocator"};
  std::string             scenario_id;  // "B1".."B4"
  std::string             runner;       // "cpp_native" | "cpp_async"
  std::string             run_mode;     // "smoke" | "normal" | "heavy"
  std::string             timestamp_utc;
  HostInfo                host{};
  DeviceInfo              device{};
  AllocatorInfo           allocator{};
  RunSummaryStats         run{};
  Metrics                 metrics{};
  std::vector<std::string> notes;       // additional notes
};

// Resolve warmup/measure/repeat counts from scenario + run-mode and optional
// user-provided overrides. The overrides struct may have zeros to indicate
// "use the default" for that field.
RunCounts resolve_run_counts(ScenarioId scenario,
                             RunMode run_mode,
                             const RunCounts& overrides);

// Compute a stable config-signature token from allocator backend and
// numeric tuning values. The algorithm mirrors the allocator design intent:
// - Build a canonical "k=v" string over backend + four numeric fields.
// - Hash with a 64-bit FNV-1a and take the lower 16 hex digits.
// - Prefix with "v1:".
std::string compute_config_signature(const AllocatorInfo& info);

// Scenario runners. Each function executes the workload for the given
// scenario and fills a PerfResult with host/device metadata, allocator
// configuration, run counts, and timing + memory metrics. Callers are
// responsible for ensuring that CUDA and the appropriate backend are
// available before invoking these.
PerfResult run_B1(const PerfConfig& cfg);
PerfResult run_B2(const PerfConfig& cfg);
PerfResult run_B3(const PerfConfig& cfg);
PerfResult run_B4(const PerfConfig& cfg);

// Serialize a PerfResult as a single JSON object to the given ostream. The
// JSON schema matches the allocator perf schema (schema_version=1). A trailing
// newline is always emitted for convenience when redirecting to files.
void write_perf_result_json(const PerfResult& result, std::ostream& os);

} // namespace testonly
} // namespace cuda
} // namespace vbt
