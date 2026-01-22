// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "include/cuda_allocator_perf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>

#if !defined(VBT_WITH_CUDA)
#  define VBT_WITH_CUDA 0
#endif

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace vbt {
namespace cuda {
namespace testonly {

namespace {

using Clock = std::chrono::steady_clock;

std::string scenario_to_string(ScenarioId id) {
  switch (id) {
    case ScenarioId::B1: return "B1";
    case ScenarioId::B2: return "B2";
    case ScenarioId::B3: return "B3";
    case ScenarioId::B4: return "B4";
  }
  return "";
}

std::string runner_to_string(Runner r) {
  switch (r) {
    case Runner::CppNative: return "cpp_native";
    case Runner::CppAsync:  return "cpp_async";
  }
  return "";
}

std::string run_mode_to_string(RunMode m) {
  switch (m) {
    case RunMode::Smoke:  return "smoke";
    case RunMode::Normal: return "normal";
    case RunMode::Heavy:  return "heavy";
  }
  return "";
}

// Best-effort hostname lookup.
std::string get_hostname() {
  const char* env = std::getenv("HOSTNAME");
  if (env && *env) {
    return std::string(env);
  }
#if defined(_WIN32)
  return std::string("windows-host");
#else
  return std::string("unknown-host");
#endif
}

HostInfo collect_host_info() {
  HostInfo h;
  h.hostname = get_hostname();
#if defined(_WIN32)
  h.os = "Windows";
#elif defined(__APPLE__)
  h.os = "Darwin";
#else
  h.os = "Linux";
#endif
  h.os_release = "";  // Left empty; script consumers should treat as best-effort.
  h.cpu_model = "";   // Not surfaced without extra platform-specific calls.
  h.cpu_cores_logical = static_cast<int>(std::thread::hardware_concurrency());
  h.cpu_cores_physical = 0;  // Unknown without platform APIs.
  h.memory_gb = 0.0;         // Optional; not filled here.
  return h;
}

// Backend classification based on VBT_CUDA_ALLOC_CONF; mirrors the Python
// helper in tests/py/cuda_graphs/_graph_workload_utils.py.
std::string classify_backend_kind_from_env() {
#if !VBT_WITH_CUDA
  (void)device_count;
  return "none";
#else
  if (device_count() <= 0) {
    return "none";
  }
  const char* conf = std::getenv("VBT_CUDA_ALLOC_CONF");
  if (!conf || *conf == '\0') {
    return "native";
  }
  std::string s(conf);
  // Extract backend= token.
  std::string backend_token;
  for (char& c : s) {
    if (c == ',') c = ' ';
  }
  std::istringstream iss(s);
  std::string tok;
  while (iss >> tok) {
    std::size_t eq = tok.find('=');
    if (eq == std::string::npos) continue;
    std::string key = tok.substr(0, eq);
    std::string val = tok.substr(eq + 1);
    if (key == "backend") {
      backend_token = val;
      break;
    }
  }
  if (backend_token.empty()) {
    return "native";
  }
  // Lowercase
  for (char& c : backend_token) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (backend_token == "native" || backend_token == "cudamalloc") {
    return "native";
  }
  if (backend_token == "cudamallocasync") {
    return "async";
  }
  return "unknown";
#endif
}

AllocatorConfig parse_allocator_numeric_config_from_env() {
  AllocatorConfig cfg;
  cfg.per_process_memory_fraction = 1.0;
  cfg.max_split_size_bytes = 0;
  cfg.max_non_split_rounding_bytes = 0;
  cfg.roundup_tolerance_bytes = 0;

  const char* conf = std::getenv("VBT_CUDA_ALLOC_CONF");
  if (!conf || *conf == '\0') {
    return cfg;
  }
  std::string s(conf);
  for (char& c : s) {
    if (c == ',') c = ' ';
  }
  std::istringstream iss(s);
  std::string tok;
  while (iss >> tok) {
    std::size_t eq = tok.find('=');
    if (eq == std::string::npos) continue;
    std::string key = tok.substr(0, eq);
    std::string val = tok.substr(eq + 1);
    if (key == "per_process_memory_fraction") {
      try {
        cfg.per_process_memory_fraction = std::stod(val);
      } catch (...) {
        // Leave default.
      }
    } else if (key == "max_split_size_bytes") {
      try {
        cfg.max_split_size_bytes = static_cast<std::uint64_t>(std::stoll(val));
      } catch (...) {
      }
    } else if (key == "max_non_split_rounding_bytes") {
      try {
        cfg.max_non_split_rounding_bytes = static_cast<std::uint64_t>(std::stoll(val));
      } catch (...) {
      }
    } else if (key == "roundup_tolerance_bytes") {
      try {
        cfg.roundup_tolerance_bytes = static_cast<std::uint64_t>(std::stoll(val));
      } catch (...) {
      }
    }
  }
  return cfg;
}

// FNV-1a 64-bit hash for config signature. We intentionally keep this
// lightweight; cross-language parity is ensured by using the same algorithm
// in the Python harness.
std::uint64_t fnv1a64(const std::string& s) {
  std::uint64_t hash = 1469598103934665603ull;       // FNV offset basis
  for (unsigned char c : s) {
    hash ^= static_cast<std::uint64_t>(c);
    hash *= 1099511628211ull;                        // FNV prime
  }
  return hash;
}

std::string current_timestamp_utc() {
  using namespace std::chrono;
  auto now = system_clock::now();
  auto secs = time_point_cast<seconds>(now);
  std::time_t t = system_clock::to_time_t(secs);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  auto ms = duration_cast<milliseconds>(now - secs).count();
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
  oss << '.' << std::setw(3) << std::setfill('0') << ms << 'Z';
  return oss.str();
}

DeviceInfo collect_device_info(int device_index, const std::string& backend_kind) {
  DeviceInfo d;
  d.index = device_index;
  d.backend = backend_kind;
#if VBT_WITH_CUDA
  d.type = "cuda";

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device_index) == cudaSuccess) {
    d.name = prop.name;
    d.total_memory_bytes = static_cast<std::uint64_t>(prop.totalGlobalMem);
    std::ostringstream cc;
    cc << prop.major << '.' << prop.minor;
    d.compute_capability = cc.str();
  } else {
    d.name = "unknown";
    d.total_memory_bytes = 0;
    d.compute_capability.clear();
  }

  int driver = 0;
  if (cudaDriverGetVersion(&driver) == cudaSuccess && driver > 0) {
    int major = driver / 1000;
    int minor = (driver % 1000) / 10;
    std::ostringstream dv;
    dv << major << '.' << minor;
    d.driver_version = dv.str();
  }

  int runtime = 0;
  if (cudaRuntimeGetVersion(&runtime) == cudaSuccess && runtime > 0) {
    int major = runtime / 1000;
    int minor = (runtime % 1000) / 10;
    std::ostringstream rv;
    rv << major << '.' << minor;
    d.runtime_version = rv.str();
  }
#else
  (void)device_index;
  d.type = "none";
  d.name = "";
  d.total_memory_bytes = 0;
  d.compute_capability.clear();
  d.driver_version.clear();
  d.runtime_version.clear();
  d.backend = "none";
#endif
  return d;
}

struct TimingSummary {
  double mean_ms{0.0};
  double median_ms{0.0};
  double p95_ms{0.0};
  double min_ms{0.0};
  double max_ms{0.0};
};

TimingSummary summarize_timings_ms(const std::vector<double>& samples) {
  TimingSummary s;
  if (samples.empty()) {
    return s;
  }
  double sum = 0.0;
  s.min_ms = samples.front();
  s.max_ms = samples.front();
  for (double v : samples) {
    sum += v;
    s.min_ms = std::min(s.min_ms, v);
    s.max_ms = std::max(s.max_ms, v);
  }
  s.mean_ms = sum / static_cast<double>(samples.size());

  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());

  auto percentile = [&](double q) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted[0];
    double pos = q * (sorted.size() - 1);
    std::size_t i = static_cast<std::size_t>(pos);
    std::size_t j = std::min(i + 1, sorted.size() - 1);
    double frac = pos - static_cast<double>(i);
    return sorted[i] * (1.0 - frac) + sorted[j] * frac;
  };

  s.median_ms = percentile(0.5);
  s.p95_ms = percentile(0.95);
  return s;
}

SnapshotMetrics compute_snapshot_metrics(const CombinedSnapshot& before,
                                         const CombinedSnapshot& after) {
  SnapshotMetrics m{};
  // Global reserved/active bytes from segment snapshots.
  m.global_reserved_bytes = global_reserved_bytes(after);
  m.global_active_bytes = global_active_bytes(after);

  // Pool metrics: aggregate across all non-zero pool ids.
  std::uint64_t pool_reserved = 0;
  std::uint64_t pool_active = 0;
  for (const auto& seg : after.segments) {
    if (seg.pool_id != 0) {
      pool_reserved += seg.bytes_reserved;
      pool_active += seg.bytes_active;
    }
  }
  m.pool_reserved_bytes = pool_reserved;
  m.pool_active_bytes = pool_active;

  m.total_segments = static_cast<std::uint64_t>(after.segments.size());
  m.total_pools = static_cast<std::uint64_t>(after.pools.size());

  (void)before;  // currently unused, but kept for future delta-based metrics.
  return m;
}

AllocatorDeltaMetrics compute_allocator_deltas(const CombinedSnapshot& before,
                                               const CombinedSnapshot& after) {
  AllocatorDeltaMetrics d{};
  d.fraction_cap_breaches_delta =
      after.stats.fraction_cap_breaches - before.stats.fraction_cap_breaches;
  d.fraction_cap_misfires_delta =
      after.stats.fraction_cap_misfires - before.stats.fraction_cap_misfires;
  d.gc_passes_delta = after.stats.gc_passes - before.stats.gc_passes;
  d.gc_reclaimed_bytes_delta =
      after.stats.gc_reclaimed_bytes - before.stats.gc_reclaimed_bytes;
  return d;
}

GraphsDeltaMetrics compute_graphs_deltas(const CombinedSnapshot& before,
                                         const CombinedSnapshot& after) {
  GraphsDeltaMetrics g{};
  g.captures_ended_delta =
      after.graphs.captures_ended - before.graphs.captures_ended;
  g.graphs_replayed_delta =
      after.graphs.graphs_replayed - before.graphs.graphs_replayed;
  g.allocator_capture_denied_delta =
      after.graphs.allocator_capture_denied -
      before.graphs.allocator_capture_denied;
  return g;
}

#if VBT_WITH_CUDA
// RAII helper that releases a graph-private pool on scope exit. This keeps B3
// and B4 scenarios honest without leaking pools if an exception is thrown.
struct PoolReleaseGuard {
  DeviceIndex dev;
  MempoolId   id;
  bool        active;

  PoolReleaseGuard(DeviceIndex d, MempoolId pid)
      : dev(d), id(pid), active(pid.is_valid()) {}

  ~PoolReleaseGuard() {
    if (active) {
      try {
        Allocator::release_pool(dev, id);
      } catch (...) {
        // best-effort: never throw from destructor
      }
    }
  }

  void dismiss() noexcept { active = false; }
};
#endif

// JSON string escaping for a small subset of characters; sufficient for
// our generated content.
std::string json_escape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (unsigned char c : in) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"':  out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  return out;
}

void emit_string_field(std::ostream& os, const char* key,
                       const std::string& value, bool& first) {
  if (!first) {
    os << ",";
  }
  first = false;
  os << '"' << key << "\":";
  os << '"' << json_escape(value) << '"';
}

void emit_number_field(std::ostream& os, const char* key,
                       std::uint64_t value, bool& first) {
  if (!first) {
    os << ",";
  }
  first = false;
  os << '"' << key << "\":" << value;
}

void emit_double_field(std::ostream& os, const char* key,
                       double value, bool& first) {
  if (!first) {
    os << ",";
  }
  first = false;
  os << '"' << key << "\":" << std::setprecision(6) << std::fixed << value;
}

} // anonymous namespace

RunCounts resolve_run_counts(ScenarioId scenario,
                             RunMode run_mode,
                             const RunCounts& overrides) {
  RunCounts out{};
  if (run_mode == RunMode::Smoke) {
    out.warmup_iters = 1;
    out.measure_iters = 3;
    out.repeats = 1;
    return out;
  }

  int default_warmup = 5;
  int default_measure = 50;
  int default_repeats = 3;

  if (run_mode == RunMode::Heavy) {
    if (scenario == ScenarioId::B2) {
      default_measure = 100;
      default_repeats = 5;
    } else {
      default_measure = 50;
      default_repeats = 3;
    }
  }

  auto pick_or_default = [](int override_val, int def_val) {
    if (override_val > 0) return override_val;
    return def_val;
  };

  out.warmup_iters = pick_or_default(overrides.warmup_iters, default_warmup);
  out.measure_iters = pick_or_default(overrides.measure_iters, default_measure);
  out.repeats = pick_or_default(overrides.repeats, default_repeats);

  if (out.warmup_iters <= 0) out.warmup_iters = 1;
  if (out.measure_iters <= 0) out.measure_iters = 1;
  if (out.repeats <= 0) out.repeats = 1;

  // Guardrail: total_iters must not exceed 10x the normal-mode total
  // for the same scenario.
  const int normal_measure = 50;
  const int normal_repeats = 3;
  const std::uint64_t normal_total_iters =
      static_cast<std::uint64_t>(normal_measure) *
      static_cast<std::uint64_t>(normal_repeats);
  const std::uint64_t max_total_iters = normal_total_iters * 10ull;

  std::uint64_t total_iters = static_cast<std::uint64_t>(out.measure_iters) *
                              static_cast<std::uint64_t>(out.repeats);
  if (total_iters > max_total_iters) {
    throw std::runtime_error("resolve_run_counts: total iters too large");
  }

  return out;
}

std::string compute_config_signature(const AllocatorInfo& info) {
  // Build canonical k=v text over backend + four numeric fields.
  std::ostringstream cfg;
  cfg << std::setprecision(8) << std::fixed;

  // We deliberately sort keys lexicographically.
  struct KV { std::string k; std::string v; };
  std::vector<KV> entries;
  entries.push_back({"backend", info.backend});

  std::ostringstream v1;
  v1 << info.config.per_process_memory_fraction;
  entries.push_back({"per_process_memory_fraction", v1.str()});

  entries.push_back({"max_non_split_rounding_bytes",
                     std::to_string(info.config.max_non_split_rounding_bytes)});
  entries.push_back({"max_split_size_bytes",
                     std::to_string(info.config.max_split_size_bytes)});
  entries.push_back({"roundup_tolerance_bytes",
                     std::to_string(info.config.roundup_tolerance_bytes)});

  std::sort(entries.begin(), entries.end(),
            [](const KV& a, const KV& b) { return a.k < b.k; });

  std::string cfg_text;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (i > 0) cfg_text += ';';
    cfg_text += entries[i].k;
    cfg_text += '=';
    cfg_text += entries[i].v;
  }

  std::uint64_t h = fnv1a64(cfg_text);
  std::ostringstream out;
  out << "v1:" << std::hex << std::nouppercase << std::setfill('0')
      << std::setw(16) << (static_cast<unsigned long long>(h));
  return out.str();
}

// ---- Scenario runners -------------------------------------------------------

PerfResult run_B1(const PerfConfig& cfg) {
  const int dev = cfg.device_index;
  if (dev < 0 || device_count() <= dev) {
    throw std::runtime_error("B1: invalid CUDA device index");
  }

  Allocator& alloc = Allocator::get(dev);
  (void)alloc;

  quiesce_allocator_for_setup(dev);
  quiesce_graphs_for_snapshots(dev);

  CombinedSnapshot before_cs = take_snapshot(dev);

  Stream s = getStreamFromPool(false, dev);
#if VBT_WITH_CUDA
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());
  (void)raw;
#endif

  const std::size_t kBytes = 1u << 20;  // 1 MiB small allocations
  std::vector<double> samples_ms;
  samples_ms.reserve(static_cast<std::size_t>(
      std::max(1, cfg.counts.measure_iters * cfg.counts.repeats)));

  // Warmup iterations
  for (int i = 0; i < cfg.counts.warmup_iters; ++i) {
    void* p = alloc.raw_alloc(kBytes, s);
    if (!p) {
      throw std::runtime_error("B1: raw_alloc returned nullptr");
    }
    alloc.raw_delete(p);
    alloc.process_events(-1);
  }

  // Measured iterations
  for (int r = 0; r < cfg.counts.repeats; ++r) {
    for (int i = 0; i < cfg.counts.measure_iters; ++i) {
      auto t0 = Clock::now();
      void* p = alloc.raw_alloc(kBytes, s);
      if (!p) {
        throw std::runtime_error("B1: raw_alloc returned nullptr");
      }
      alloc.raw_delete(p);
      alloc.process_events(-1);
      auto t1 = Clock::now();
      double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
      samples_ms.push_back(ms);
    }
  }

  drain_allocator_for_snapshots(dev);
  quiesce_graphs_for_snapshots(dev);
  CombinedSnapshot after_cs = take_snapshot(dev);

  PerfResult out{};
  out.scenario_id = scenario_to_string(cfg.scenario_id);
  out.runner = runner_to_string(cfg.runner);
  out.run_mode = run_mode_to_string(cfg.run_mode);
  out.timestamp_utc = current_timestamp_utc();

  out.host = collect_host_info();
  std::string backend_kind = classify_backend_kind_from_env();
  out.device = collect_device_info(dev, backend_kind);

  out.allocator.backend = backend_kind;
  out.allocator.config = parse_allocator_numeric_config_from_env();
  out.allocator.config_signature = compute_config_signature(out.allocator);

  out.run.warmup_iters = cfg.counts.warmup_iters;
  out.run.measure_iters = cfg.counts.measure_iters;
  out.run.repeats = cfg.counts.repeats;
  out.run.total_iters = cfg.counts.measure_iters * cfg.counts.repeats;
  out.run.num_replays = 0;  // eager-only scenario

  TimingSummary ts = summarize_timings_ms(samples_ms);
  out.run.wall_mean_ms = ts.mean_ms;
  out.run.wall_median_ms = ts.median_ms;
  out.run.wall_p95_ms = ts.p95_ms;
  out.run.wall_min_ms = ts.min_ms;
  out.run.wall_max_ms = ts.max_ms;

  out.metrics.median_ms_per_iter = ts.median_ms;
  out.metrics.p95_ms_per_iter = ts.p95_ms;
  out.metrics.peak_reserved_bytes = after_cs.stats.max_reserved_bytes_all;
  out.metrics.peak_allocated_bytes = after_cs.stats.max_allocated_bytes_all;

  // Pool metrics aggregate across all non-zero pools.
  std::uint64_t pool_peak_reserved = 0;
  std::uint64_t pool_peak_allocated = 0;
  for (const auto& seg : after_cs.segments) {
    if (seg.pool_id != 0) {
      pool_peak_reserved += seg.bytes_reserved;
      pool_peak_allocated += seg.bytes_active;
    }
  }
  out.metrics.pool_peak_reserved_bytes = pool_peak_reserved;
  out.metrics.pool_peak_allocated_bytes = pool_peak_allocated;

  out.metrics.snapshot = compute_snapshot_metrics(before_cs, after_cs);
  out.metrics.allocator = compute_allocator_deltas(before_cs, after_cs);
  out.metrics.graphs = compute_graphs_deltas(before_cs, after_cs);

  if (!cfg.notes.empty()) {
    out.notes.push_back(cfg.notes);
  }

  return out;
}

PerfResult run_B2(const PerfConfig& cfg) {
  const int dev = cfg.device_index;
  if (dev < 0 || device_count() <= dev) {
    throw std::runtime_error("B2: invalid CUDA device index");
  }

  Allocator& alloc = Allocator::get(dev);
  (void)alloc;

  quiesce_allocator_for_setup(dev);
  quiesce_graphs_for_snapshots(dev);

  CombinedSnapshot before_cs = take_snapshot(dev);

  Stream s = getStreamFromPool(false, dev);
#if VBT_WITH_CUDA
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());
  (void)raw;
#endif

  // this scenario on the current device (e.g., unknown cap or too
  // small for a meaningful heavy test).
  std::size_t heavy = pick_heavy_allocation_size(dev);
  if (heavy == 0) {
    PerfResult out{};
    out.scenario_id = scenario_to_string(cfg.scenario_id);
    out.runner = runner_to_string(cfg.runner);
    out.run_mode = run_mode_to_string(cfg.run_mode);
    out.timestamp_utc = current_timestamp_utc();

    out.host = collect_host_info();
    std::string backend_kind = classify_backend_kind_from_env();
    out.device = collect_device_info(dev, backend_kind);

    out.allocator.backend = backend_kind;
    out.allocator.config = parse_allocator_numeric_config_from_env();
    out.allocator.config_signature = compute_config_signature(out.allocator);

    out.run.warmup_iters = cfg.counts.warmup_iters;
    out.run.measure_iters = cfg.counts.measure_iters;
    out.run.repeats = cfg.counts.repeats;
    out.run.total_iters = 0;
    out.run.num_replays = 0;

    // No timings or deltas when the scenario is skipped.
    out.metrics.median_ms_per_iter = 0.0;
    out.metrics.p95_ms_per_iter = 0.0;
    out.metrics.peak_reserved_bytes = 0;
    out.metrics.pool_peak_reserved_bytes = 0;
    out.metrics.peak_allocated_bytes = 0;
    out.metrics.pool_peak_allocated_bytes = 0;
    out.metrics.snapshot = SnapshotMetrics{};
    out.metrics.allocator = AllocatorDeltaMetrics{};
    out.metrics.graphs = GraphsDeltaMetrics{};
    out.metrics.skip_reason = "test_utils_unavailable";
    out.metrics.skip_message =
        "pick_heavy_allocation_size returned 0; skipping B2 heavy scenario";

    if (!cfg.notes.empty()) {
      out.notes.push_back(cfg.notes);
    }

    return out;
  }

  // Tight fraction to exercise the fraction gate; best-effort only.
  MemoryFractionGuard frac_guard(dev, 0.5);

  std::vector<double> samples_ms;
  samples_ms.reserve(static_cast<std::size_t>(
      std::max(1, cfg.counts.measure_iters * cfg.counts.repeats)));

  for (int i = 0; i < cfg.counts.warmup_iters; ++i) {
    void* p = alloc.raw_alloc(heavy, s);
    if (!p) {
      throw std::runtime_error("B2: raw_alloc returned nullptr");
    }
    alloc.raw_delete(p);
    alloc.process_events(-1);
  }

  for (int r = 0; r < cfg.counts.repeats; ++r) {
    for (int i = 0; i < cfg.counts.measure_iters; ++i) {
      auto t0 = Clock::now();
      void* p = alloc.raw_alloc(heavy, s);
      if (!p) {
        throw std::runtime_error("B2: raw_alloc returned nullptr");
      }
      alloc.raw_delete(p);
      alloc.process_events(-1);
      auto t1 = Clock::now();
      double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
      samples_ms.push_back(ms);
    }
  }

  drain_allocator_for_snapshots(dev);
  quiesce_graphs_for_snapshots(dev);
  CombinedSnapshot after_cs = take_snapshot(dev);

  PerfResult out{};
  out.scenario_id = scenario_to_string(cfg.scenario_id);
  out.runner = runner_to_string(cfg.runner);
  out.run_mode = run_mode_to_string(cfg.run_mode);
  out.timestamp_utc = current_timestamp_utc();

  out.host = collect_host_info();
  std::string backend_kind = classify_backend_kind_from_env();
  out.device = collect_device_info(dev, backend_kind);

  out.allocator.backend = backend_kind;
  out.allocator.config = parse_allocator_numeric_config_from_env();
  out.allocator.config_signature = compute_config_signature(out.allocator);

  out.run.warmup_iters = cfg.counts.warmup_iters;
  out.run.measure_iters = cfg.counts.measure_iters;
  out.run.repeats = cfg.counts.repeats;
  out.run.total_iters = cfg.counts.measure_iters * cfg.counts.repeats;
  out.run.num_replays = 0;

  TimingSummary ts = summarize_timings_ms(samples_ms);
  out.run.wall_mean_ms = ts.mean_ms;
  out.run.wall_median_ms = ts.median_ms;
  out.run.wall_p95_ms = ts.p95_ms;
  out.run.wall_min_ms = ts.min_ms;
  out.run.wall_max_ms = ts.max_ms;

  out.metrics.median_ms_per_iter = ts.median_ms;
  out.metrics.p95_ms_per_iter = ts.p95_ms;
  out.metrics.peak_reserved_bytes = after_cs.stats.max_reserved_bytes_all;
  out.metrics.peak_allocated_bytes = after_cs.stats.max_allocated_bytes_all;

  std::uint64_t pool_peak_reserved = 0;
  std::uint64_t pool_peak_allocated = 0;
  for (const auto& seg : after_cs.segments) {
    if (seg.pool_id != 0) {
      pool_peak_reserved += seg.bytes_reserved;
      pool_peak_allocated += seg.bytes_active;
    }
  }
  out.metrics.pool_peak_reserved_bytes = pool_peak_reserved;
  out.metrics.pool_peak_allocated_bytes = pool_peak_allocated;

  out.metrics.snapshot = compute_snapshot_metrics(before_cs, after_cs);
  out.metrics.allocator = compute_allocator_deltas(before_cs, after_cs);
  out.metrics.graphs = compute_graphs_deltas(before_cs, after_cs);

  if (!cfg.notes.empty()) {
    out.notes.push_back(cfg.notes);
  }

  return out;
}

PerfResult run_B3(const PerfConfig& cfg) {
  const int dev = cfg.device_index;
  if (dev < 0 || device_count() <= dev) {
    throw std::runtime_error("B3: invalid CUDA device index");
  }

  quiesce_allocator_for_setup(dev);
  quiesce_graphs_for_snapshots(dev);

  CombinedSnapshot before_cs = take_snapshot(dev);

  Stream s = getStreamFromPool(false, dev);
#if VBT_WITH_CUDA
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());
#endif

  std::vector<double> samples_ms;

#if VBT_WITH_CUDA
  float*    d    = nullptr;
  MempoolId pool{};

  samples_ms.reserve(static_cast<std::size_t>(
      std::max(1, cfg.counts.measure_iters * cfg.counts.repeats)));

  try {
    // Create a graph-private pool and run a tiny memcpy workload under
    // CUDA Graphs using that pool. This gives us per-pool metrics in the
    // final snapshots without changing production semantics.
    pool = Allocator::create_pool_id(dev);
    PoolReleaseGuard pool_guard(dev, pool);

    if (cudaMalloc(reinterpret_cast<void**>(&d), sizeof(float)) != cudaSuccess) {
      throw std::runtime_error("B3: cudaMalloc failed");
    }

    int host_value = 42;

    CUDAGraph g;
    g.capture_begin(s, pool);
    if (cudaMemcpyAsync(d,
                        &host_value,
                        sizeof(int),
                        cudaMemcpyHostToDevice,
                        raw) != cudaSuccess) {
      throw std::runtime_error("B3: cudaMemcpyAsync failed during capture");
    }
    g.capture_end();
    g.instantiate();

    // Warmup replays.
    for (int i = 0; i < cfg.counts.warmup_iters; ++i) {
      g.replay();
      cudaStreamSynchronize(raw);
    }

    // Measured replays.
    for (int r = 0; r < cfg.counts.repeats; ++r) {
      for (int i = 0; i < cfg.counts.measure_iters; ++i) {
        auto t0 = Clock::now();
        g.replay();
        cudaStreamSynchronize(raw);
        auto t1 = Clock::now();
        double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
        samples_ms.push_back(ms);
      }
    }

    cudaStreamSynchronize(raw);
    if (cudaFree(d) != cudaSuccess) {
      throw std::runtime_error("B3: cudaFree failed");
    }
    // pool_guard destructor releases the pool here.
  } catch (...) {
    if (d != nullptr) {
      (void)cudaFree(d);
    }
    throw;
  }
#else
  (void)cfg;
#endif

  drain_allocator_for_snapshots(dev);
  quiesce_graphs_for_snapshots(dev);
  CombinedSnapshot after_cs = take_snapshot(dev);

  PerfResult out{};
  out.scenario_id = scenario_to_string(cfg.scenario_id);
  out.runner = runner_to_string(cfg.runner);
  out.run_mode = run_mode_to_string(cfg.run_mode);
  out.timestamp_utc = current_timestamp_utc();

  out.host = collect_host_info();
  std::string backend_kind = classify_backend_kind_from_env();
  out.device = collect_device_info(dev, backend_kind);

  out.allocator.backend = backend_kind;
  out.allocator.config = parse_allocator_numeric_config_from_env();
  out.allocator.config_signature = compute_config_signature(out.allocator);

  out.run.warmup_iters = cfg.counts.warmup_iters;
  out.run.measure_iters = cfg.counts.measure_iters;
  out.run.repeats = cfg.counts.repeats;
  out.run.total_iters = cfg.counts.measure_iters * cfg.counts.repeats;
  out.run.num_replays = 1;  // graph replay workload

  TimingSummary ts = summarize_timings_ms(samples_ms);
  out.run.wall_mean_ms = ts.mean_ms;
  out.run.wall_median_ms = ts.median_ms;
  out.run.wall_p95_ms = ts.p95_ms;
  out.run.wall_min_ms = ts.min_ms;
  out.run.wall_max_ms = ts.max_ms;

  out.metrics.median_ms_per_iter = ts.median_ms;
  out.metrics.p95_ms_per_iter = ts.p95_ms;
  out.metrics.peak_reserved_bytes = after_cs.stats.max_reserved_bytes_all;
  out.metrics.peak_allocated_bytes = after_cs.stats.max_allocated_bytes_all;

  std::uint64_t pool_peak_reserved = 0;
  std::uint64_t pool_peak_allocated = 0;
  for (const auto& seg : after_cs.segments) {
    if (seg.pool_id != 0) {
      pool_peak_reserved += seg.bytes_reserved;
      pool_peak_allocated += seg.bytes_active;
    }
  }
  out.metrics.pool_peak_reserved_bytes = pool_peak_reserved;
  out.metrics.pool_peak_allocated_bytes = pool_peak_allocated;

  out.metrics.snapshot = compute_snapshot_metrics(before_cs, after_cs);
  out.metrics.allocator = compute_allocator_deltas(before_cs, after_cs);
  out.metrics.graphs = compute_graphs_deltas(before_cs, after_cs);

  if (!cfg.notes.empty()) {
    out.notes.push_back(cfg.notes);
  }

  return out;
}

PerfResult run_B4(const PerfConfig& cfg) {
  // B4 mirrors B3 but is intended for the async backend. The workload is the
  // same; the CLI wrapper is responsible for ensuring that the backend is
  // configured as async when invoking this scenario.
  PerfResult res = run_B3(cfg);
  return res;
}

void write_perf_result_json(const PerfResult& r, std::ostream& os) {
  os << '{';
  bool first = true;

  // Top-level scalar fields
  if (!first) os << ',';
  os << "\"schema_version\":" << r.schema_version;
  first = false;

  emit_string_field(os, "component", r.component, first);
  emit_string_field(os, "scenario_id", r.scenario_id, first);
  emit_string_field(os, "runner", r.runner, first);
  emit_string_field(os, "run_mode", r.run_mode, first);
  emit_string_field(os, "timestamp_utc", r.timestamp_utc, first);

  // host
  os << ",\"host\":{";
  bool hfirst = true;
  emit_string_field(os, "hostname", r.host.hostname, hfirst);
  emit_string_field(os, "os", r.host.os, hfirst);
  emit_string_field(os, "os_release", r.host.os_release, hfirst);
  emit_string_field(os, "cpu_model", r.host.cpu_model, hfirst);
  emit_number_field(os, "cpu_cores_logical",
                    static_cast<std::uint64_t>(r.host.cpu_cores_logical), hfirst);
  emit_number_field(os, "cpu_cores_physical",
                    static_cast<std::uint64_t>(r.host.cpu_cores_physical), hfirst);
  {
    bool tmp_first = hfirst;
    emit_double_field(os, "memory_gb", r.host.memory_gb, tmp_first);
  }
  os << '}';

  // device
  os << ",\"device\":{";
  bool dfirst = true;
  emit_number_field(os, "index",
                    static_cast<std::uint64_t>(r.device.index), dfirst);
  emit_string_field(os, "type", r.device.type, dfirst);
  emit_string_field(os, "name", r.device.name, dfirst);
  emit_number_field(os, "total_memory_bytes", r.device.total_memory_bytes, dfirst);
  emit_string_field(os, "compute_capability", r.device.compute_capability, dfirst);
  emit_string_field(os, "driver_version", r.device.driver_version, dfirst);
  emit_string_field(os, "runtime_version", r.device.runtime_version, dfirst);
  emit_string_field(os, "backend", r.device.backend, dfirst);
  os << '}';

  // allocator
  os << ",\"allocator\":{";
  bool afirst = true;
  emit_string_field(os, "backend", r.allocator.backend, afirst);
  os << ",\"config\":{";
  bool cfirst = true;
  emit_double_field(os, "per_process_memory_fraction",
                    r.allocator.config.per_process_memory_fraction, cfirst);
  emit_number_field(os, "max_split_size_bytes",
                    r.allocator.config.max_split_size_bytes, cfirst);
  emit_number_field(os, "max_non_split_rounding_bytes",
                    r.allocator.config.max_non_split_rounding_bytes, cfirst);
  emit_number_field(os, "roundup_tolerance_bytes",
                    r.allocator.config.roundup_tolerance_bytes, cfirst);
  os << '}';
  emit_string_field(os, "config_signature", r.allocator.config_signature, afirst);
  os << '}';

  // run summary
  os << ",\"run\":{";
  bool rfirst = true;
  emit_number_field(os, "warmup_iters",
                    static_cast<std::uint64_t>(r.run.warmup_iters), rfirst);
  emit_number_field(os, "measure_iters",
                    static_cast<std::uint64_t>(r.run.measure_iters), rfirst);
  emit_number_field(os, "repeats",
                    static_cast<std::uint64_t>(r.run.repeats), rfirst);
  emit_number_field(os, "total_iters",
                    static_cast<std::uint64_t>(r.run.total_iters), rfirst);
  emit_number_field(os, "num_replays",
                    static_cast<std::uint64_t>(r.run.num_replays), rfirst);

  {
    os << ",\"wall_time_ms_per_iter_summary\":{";
    bool tfirst = true;
    emit_double_field(os, "mean", r.run.wall_mean_ms, tfirst);
    emit_double_field(os, "median", r.run.wall_median_ms, tfirst);
    emit_double_field(os, "p95", r.run.wall_p95_ms, tfirst);
    emit_double_field(os, "min", r.run.wall_min_ms, tfirst);
    emit_double_field(os, "max", r.run.wall_max_ms, tfirst);
    os << '}';
  }

  os << '}';  // end run

  // metrics
  os << ",\"metrics\":{";
  bool mfirst = true;
  emit_double_field(os, "median_ms_per_iter", r.metrics.median_ms_per_iter, mfirst);
  emit_double_field(os, "p95_ms_per_iter", r.metrics.p95_ms_per_iter, mfirst);
  emit_number_field(os, "peak_reserved_bytes", r.metrics.peak_reserved_bytes, mfirst);
  emit_number_field(os, "pool_peak_reserved_bytes", r.metrics.pool_peak_reserved_bytes, mfirst);
  emit_number_field(os, "peak_allocated_bytes", r.metrics.peak_allocated_bytes, mfirst);
  emit_number_field(os, "pool_peak_allocated_bytes", r.metrics.pool_peak_allocated_bytes, mfirst);

  // allocator deltas
  os << ",\"allocator\":{";
  bool adfirst = true;
  emit_number_field(os, "fraction_cap_breaches_delta",
                    r.metrics.allocator.fraction_cap_breaches_delta, adfirst);
  emit_number_field(os, "fraction_cap_misfires_delta",
                    r.metrics.allocator.fraction_cap_misfires_delta, adfirst);
  emit_number_field(os, "gc_passes_delta",
                    r.metrics.allocator.gc_passes_delta, adfirst);
  emit_number_field(os, "gc_reclaimed_bytes_delta",
                    r.metrics.allocator.gc_reclaimed_bytes_delta, adfirst);
  os << '}';

  // graphs deltas
  os << ",\"graphs\":{";
  bool gfirst = true;
  emit_number_field(os, "captures_ended_delta",
                    r.metrics.graphs.captures_ended_delta, gfirst);
  emit_number_field(os, "graphs_replayed_delta",
                    r.metrics.graphs.graphs_replayed_delta, gfirst);
  emit_number_field(os, "allocator_capture_denied_delta",
                    r.metrics.graphs.allocator_capture_denied_delta, gfirst);
  os << '}';

  // memory snapshot metrics
  os << ",\"memory_snapshot\":{";
  bool sfirst = true;
  emit_number_field(os, "global_reserved_bytes",
                    r.metrics.snapshot.global_reserved_bytes, sfirst);
  emit_number_field(os, "global_active_bytes",
                    r.metrics.snapshot.global_active_bytes, sfirst);
  emit_number_field(os, "pool_reserved_bytes",
                    r.metrics.snapshot.pool_reserved_bytes, sfirst);
  emit_number_field(os, "pool_active_bytes",
                    r.metrics.snapshot.pool_active_bytes, sfirst);
  emit_number_field(os, "total_segments",
                    r.metrics.snapshot.total_segments, sfirst);
  emit_number_field(os, "total_pools",
                    r.metrics.snapshot.total_pools, sfirst);
  os << '}';

  // skip metadata (null when absent)
  os << ",\"skip_reason\":";
  if (!r.metrics.skip_reason.empty()) {
    os << '"' << json_escape(r.metrics.skip_reason) << '"';
  } else {
    os << "null";
  }
  os << ",\"skip_message\":";
  if (!r.metrics.skip_message.empty()) {
    os << '"' << json_escape(r.metrics.skip_message) << '"';
  } else {
    os << "null";
  }

  os << '}';  // end metrics

  // notes array
  os << ",\"notes\":";
  os << '[';
  for (std::size_t i = 0; i < r.notes.size(); ++i) {
    if (i > 0) os << ',';
    os << '"' << json_escape(r.notes[i]) << '"';
  }
  os << ']';

  os << '}';
  os << '\n';
}

} // namespace testonly
} // namespace cuda
} // namespace vbt
