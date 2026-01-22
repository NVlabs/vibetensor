// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <chrono>
#include <stdexcept>

#include "vbt/cuda/fabric_events.h"
#include "vbt/cuda/fabric_state.h"

namespace nb = nanobind;

namespace vbt_py {

using vbt::cuda::fabric::FabricMode;
using vbt::cuda::fabric::FabricInitStatus;
using vbt::cuda::fabric::FabricCliqueSnapshot;
using vbt::cuda::fabric::FabricTopologySnapshot;
using vbt::cuda::fabric::FabricStatsSnapshot;
using vbt::cuda::fabric::FabricPerDeviceStatsSnapshot;
using vbt::cuda::fabric::FabricStatsReasonsSnapshot;
using vbt::cuda::fabric::FabricTopology;
using vbt::cuda::fabric::FabricState;
using vbt::cuda::fabric::FabricEventsMode;
using vbt::cuda::fabric::FabricEventKind;
using vbt::cuda::fabric::FabricEventLevel;
using vbt::cuda::fabric::FabricEvent;
using vbt::cuda::fabric::FabricEventSnapshot;
using vbt::cuda::fabric::get_fabric_events_mode;
using vbt::cuda::fabric::set_fabric_events_mode;
using vbt::cuda::fabric::fabric_events_snapshot;
using vbt::cuda::fabric::fabric_events_wait_for_seq;
using vbt::cuda::fabric::record_fabric_event;
using vbt::cuda::fabric::fabric_state;
using vbt::cuda::fabric::fabric_enabled_for_ops;

void bind_fabric(nb::module_& m) {
  // Enums exposed as underscored types for internal use in vibetensor.fabric.
  nb::enum_<FabricMode>(m, "_FabricMode")
      .value("disabled",    FabricMode::Disabled)
      .value("best_effort", FabricMode::BestEffort)
      .value("dry_run",     FabricMode::DryRun);

  nb::enum_<FabricInitStatus>(m, "_FabricInitStatus")
      .value("uninitialized", FabricInitStatus::Uninitialized)
      .value("ok",            FabricInitStatus::Ok)
      .value("no_cuda",       FabricInitStatus::NoCuda)
      .value("uva_failed",    FabricInitStatus::UvaFailed)
      .value("cuda_error",    FabricInitStatus::CudaError);

  nb::enum_<FabricEventsMode>(m, "_FabricEventsMode")
      .value("off",   FabricEventsMode::kOff)
      .value("basic", FabricEventsMode::kBasic);

  nb::enum_<FabricEventKind>(m, "_FabricEventKind")
      .value("op_enqueue",             FabricEventKind::kOpEnqueue)
      .value("op_complete",            FabricEventKind::kOpComplete)
      .value("op_fallback",            FabricEventKind::kOpFallback)
      .value("op_error",               FabricEventKind::kOpError)
      .value("mode_changed",           FabricEventKind::kModeChanged)
      .value("event_lifetime_toggled", FabricEventKind::kEventLifetimeToggled)
      .value("events_mode_changed",    FabricEventKind::kEventsModeChanged);

  nb::enum_<FabricEventLevel>(m, "_FabricEventLevel")
      .value("debug", FabricEventLevel::kDebug)
      .value("info",  FabricEventLevel::kInfo)
      .value("warn",  FabricEventLevel::kWarn)
      .value("error", FabricEventLevel::kError);

  nb::class_<FabricCliqueSnapshot>(m, "_FabricCliqueSnapshot")
      .def_prop_ro("id", [](const FabricCliqueSnapshot& c) { return c.id; })
      .def_prop_ro("devices", [](const FabricCliqueSnapshot& c) { return c.devices; });

  nb::class_<FabricTopologySnapshot>(m, "_FabricTopologySnapshot")
      .def_prop_ro("device_count", [](const FabricTopologySnapshot& s) { return s.device_count; })
      .def_prop_ro("uva_ok", [](const FabricTopologySnapshot& s) { return s.uva_ok; })
      .def_prop_ro("mode", [](const FabricTopologySnapshot& s) { return s.mode; })
      .def_prop_ro("init_status", [](const FabricTopologySnapshot& s) { return s.init_status; })
      .def_prop_ro("disable_reason", [](const FabricTopologySnapshot& s) { return s.disable_reason; })
      .def_prop_ro("cliques", [](const FabricTopologySnapshot& s) { return s.cliques; });

  nb::class_<FabricStatsReasonsSnapshot>(m, "_FabricStatsReasonsSnapshot")
      .def_prop_ro("no_p2p", [](const FabricStatsReasonsSnapshot& r) { return r.no_p2p; })
      .def_prop_ro("requires_grad", [](const FabricStatsReasonsSnapshot& r) { return r.requires_grad; })
      .def_prop_ro("in_backward", [](const FabricStatsReasonsSnapshot& r) { return r.in_backward; })
      .def_prop_ro("small_tensor", [](const FabricStatsReasonsSnapshot& r) { return r.small_tensor; });

  nb::class_<FabricPerDeviceStatsSnapshot>(m, "_FabricPerDeviceStatsSnapshot")
      .def_prop_ro(
          "device_index", [](const FabricPerDeviceStatsSnapshot& s) { return s.device_index; })
      .def_prop_ro(
          "ops_as_primary", [](const FabricPerDeviceStatsSnapshot& s) { return s.ops_as_primary; })
      .def_prop_ro(
          "ops_as_remote", [](const FabricPerDeviceStatsSnapshot& s) { return s.ops_as_remote; })
      .def_prop_ro(
          "remote_bytes_read", [](const FabricPerDeviceStatsSnapshot& s) { return s.remote_bytes_read; })
      .def_prop_ro(
          "remote_bytes_written", [](const FabricPerDeviceStatsSnapshot& s) { return s.remote_bytes_written; });

  nb::class_<FabricStatsSnapshot>(m, "_FabricStatsSnapshot")
      .def_prop_ro("mesh_builds", [](const FabricStatsSnapshot& s) { return s.mesh_builds; })
      .def_prop_ro("p2p_pairs_enabled", [](const FabricStatsSnapshot& s) { return s.p2p_pairs_enabled; })
      .def_prop_ro("p2p_pairs_failed", [](const FabricStatsSnapshot& s) { return s.p2p_pairs_failed; })
      .def_prop_ro("fabric_ops_attempted", [](const FabricStatsSnapshot& s) { return s.fabric_ops_attempted; })
      .def_prop_ro("fabric_ops_hit", [](const FabricStatsSnapshot& s) { return s.fabric_ops_hit; })
      .def_prop_ro("fabric_ops_fallback", [](const FabricStatsSnapshot& s) { return s.fabric_ops_fallback; })
      .def_prop_ro("remote_bytes_read", [](const FabricStatsSnapshot& s) { return s.remote_bytes_read; })
      .def_prop_ro("remote_bytes_written", [](const FabricStatsSnapshot& s) { return s.remote_bytes_written; })
      .def_prop_ro("inflight_ops_current", [](const FabricStatsSnapshot& s) { return s.inflight_ops_current; })
      .def_prop_ro("inflight_ops_peak", [](const FabricStatsSnapshot& s) { return s.inflight_ops_peak; })
      .def_prop_ro("event_queue_len_peak", [](const FabricStatsSnapshot& s) { return s.event_queue_len_peak; })
      .def_prop_ro("event_dropped_total", [](const FabricStatsSnapshot& s) { return s.event_dropped_total; })
      .def_prop_ro("event_failures_total", [](const FabricStatsSnapshot& s) { return s.event_failures_total; })
      .def_prop_ro("mode_enable_calls", [](const FabricStatsSnapshot& s) { return s.mode_enable_calls; })
      .def_prop_ro("mode_disable_calls", [](const FabricStatsSnapshot& s) { return s.mode_disable_calls; })
      .def_prop_ro("mode_set_failures", [](const FabricStatsSnapshot& s) { return s.mode_set_failures; })
      .def_prop_ro("reasons", [](const FabricStatsSnapshot& s) { return s.reasons; })
      .def_prop_ro("per_device", [](const FabricStatsSnapshot& s) { return s.per_device; });

  nb::class_<FabricEvent>(m, "_FabricEvent")
      .def_prop_ro("seq", [](const FabricEvent& e) { return e.seq; })
      .def_prop_ro("t_ns", [](const FabricEvent& e) { return e.t_ns; })
      .def_prop_ro("primary_device", [](const FabricEvent& e) { return e.primary_device; })
      .def_prop_ro("other_device", [](const FabricEvent& e) { return e.other_device; })
      .def_prop_ro("kind", [](const FabricEvent& e) { return e.kind; })
      .def_prop_ro("level", [](const FabricEvent& e) { return e.level; })
      .def_prop_ro("op_id", [](const FabricEvent& e) { return e.op_id; })
      .def_prop_ro("numel", [](const FabricEvent& e) { return e.numel; })
      .def_prop_ro("bytes", [](const FabricEvent& e) { return e.bytes; })
      .def_prop_ro("reason_raw", [](const FabricEvent& e) { return e.reason_raw; })
      .def_prop_ro("message", [](const FabricEvent& e) -> nb::object {
        if (!e.message) return nb::none();
        return nb::str(e.message);
      });

  nb::class_<FabricEventSnapshot>(m, "_FabricEventSnapshot")
      .def_prop_ro("base_seq", [](const FabricEventSnapshot& s) { return s.base_seq; })
      .def_prop_ro("next_seq", [](const FabricEventSnapshot& s) { return s.next_seq; })
      .def_prop_ro("dropped_total", [](const FabricEventSnapshot& s) { return s.dropped_total; })
      .def_prop_ro("capacity", [](const FabricEventSnapshot& s) { return s.capacity; })
      .def_prop_ro("events", [](const FabricEventSnapshot& s) -> const std::vector<FabricEvent>& {
        return s.events;
      });

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
  // Minimal topology class used only by internal tests to synthesize
  // topologies via _fabric_set_fake_topology_builder_for_tests.
  nb::class_<FabricTopology>(m, "_FabricTopology")
      .def_rw("device_count", &FabricTopology::device_count)
      .def_rw("can_access_peer", &FabricTopology::can_access_peer)
      .def_rw("p2p_enabled", &FabricTopology::p2p_enabled)
      .def_rw("clique_id", &FabricTopology::clique_id)
      .def_rw("clique_size", &FabricTopology::clique_size);
#endif

  m.def("_fabric_topology_snapshot", []() {
    return vbt::cuda::fabric::fabric_topology_snapshot();
  });

  m.def("_fabric_stats_snapshot", []() {
    return vbt::cuda::fabric::fabric_stats_snapshot();
  });

  m.def("_fabric_get_events_mode", []() {
    return get_fabric_events_mode();
  });

  m.def("_fabric_set_events_mode", [](FabricEventsMode mode) {
    set_fabric_events_mode(mode);
  });

  m.def("_fabric_events_snapshot", [](std::uint64_t min_seq, std::size_t max_events) {
    return fabric_events_snapshot(min_seq, max_events);
  });

  m.def("_fabric_events_wait_for_seq",
        [](std::uint64_t target_seq, std::uint64_t timeout_ms) {
          using rep_t = std::chrono::milliseconds::rep;
          const rep_t max_rep = std::chrono::milliseconds::max().count();
          const std::uint64_t max_u64 =
              (max_rep <= 0) ? 0ull : static_cast<std::uint64_t>(max_rep);

          const std::uint64_t clamped_u64 = (timeout_ms > max_u64) ? max_u64 : timeout_ms;
          const rep_t clamped = static_cast<rep_t>(clamped_u64);

          return fabric_events_wait_for_seq(target_seq,
                                            std::chrono::milliseconds(clamped));
        },
        nb::call_guard<nb::gil_scoped_release>());

  m.def("_fabric_get_mode", []() {
    FabricState& fs = fabric_state();
    return fs.config.mode.load(std::memory_order_acquire);
  });

  m.def("_fabric_set_mode", [](FabricMode new_mode) {
    FabricState& fs = fabric_state();  // ensures init has run

    // Best-effort mode transition counters.
    if (new_mode == FabricMode::Disabled) {
      fs.stats.mode_disable_calls.fetch_add(1, std::memory_order_relaxed);
    } else {
      fs.stats.mode_enable_calls.fetch_add(1, std::memory_order_relaxed);
    }

    const FabricMode old_mode = fs.config.mode.load(std::memory_order_acquire);

    auto maybe_emit_mode_changed = [&](FabricMode old_m, FabricMode new_m) noexcept {
      if (old_m == new_m) return;
      FabricEvent ev;
      ev.kind = FabricEventKind::kModeChanged;
      ev.level = FabricEventLevel::kInfo;
      ev.message = vbt::cuda::fabric::kFabricEventMsgModeChanged;
      record_fabric_event(std::move(ev));
    };

    try {
      if (!fs.uva_ok) {
        // Canonical UVA-disabled error; fall back to invariant message when
        // disable_reason is empty.
        std::string msg = fs.disable_reason;
        if (msg.empty()) {
          msg = "[Fabric] UVA invariant violated on this platform; Fabric is disabled";
        }
        throw std::runtime_error(msg);
      }

      if (new_mode == FabricMode::Disabled) {
        fs.config.mode.store(FabricMode::Disabled, std::memory_order_release);
        maybe_emit_mode_changed(old_mode, FabricMode::Disabled);
        return;
      }

      fs.config.mode.store(new_mode, std::memory_order_release);
      const bool ok_for_ops = fabric_enabled_for_ops(fs);
      if (!ok_for_ops) {
        // For 0/1-GPU configurations we allow mode changes even though the
        // global gate remains closed; Fabric will simply never be enabled for
        // ops in these environments.
        if (fs.topology.device_count < 2) {
          maybe_emit_mode_changed(old_mode, new_mode);
          return;
        }
        // For >=2 GPUs with UVA ok but no usable multi-device clique, treat
        // this as a hard failure and roll back mode.
        fs.config.mode.store(old_mode, std::memory_order_release);
        std::string msg = fs.disable_reason;
        if (msg.empty()) {
          msg = "[Fabric] Cannot enable Fabric in current topology/UVA/P2P state";
        }
        throw std::runtime_error(msg);
      }

      maybe_emit_mode_changed(old_mode, new_mode);
    } catch (...) {
      fs.stats.mode_set_failures.fetch_add(1, std::memory_order_relaxed);
      throw;
    }
  });

  m.def("_fabric_is_enabled_for_ops", []() {
    return fabric_enabled_for_ops(fabric_state());
  });

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
  // Test-only helpers; these are not part of the public API surface and are
  // only exercised from tests.
  m.def("_fabric_reset_state_for_tests", &vbt::cuda::fabric::reset_fabric_state_for_tests);
  m.def("_fabric_reset_stats_for_tests", &vbt::cuda::fabric::reset_fabric_stats_for_tests);
  m.def("_fabric_reset_events_for_tests", &vbt::cuda::fabric::reset_fabric_events_for_tests);

  m.def("_fabric_set_forced_uva_ok_for_tests", [](nb::object v) {
    auto& hooks = vbt::cuda::fabric::fabric_test_hooks();
    if (v.is_none()) {
      hooks.forced_uva_ok.reset();
    } else {
      hooks.forced_uva_ok = nb::cast<bool>(v);
    }
  });

  m.def("_fabric_set_fake_topology_builder_for_tests", [](nb::object fn) {
    auto& hooks = vbt::cuda::fabric::fabric_test_hooks();
    if (fn.is_none()) {
      hooks.fake_topology_builder = nullptr;
      return;
    }
    // Store a persistent handle to the Python callable and wrap it in a
    // C++ lambda that can be invoked from C++ initialization paths.
    hooks.fake_topology_builder = [fn = nb::object(fn)](vbt::cuda::fabric::FabricTopology& topo) {
      nb::gil_scoped_acquire gil;
      // Expose the FabricTopology object to Python via a simple struct-like
      // topo.device_count, can_access_peer, clique_id, clique_size directly
      // via attributes.
      fn(nb::cast(&topo));
    };
  });

  m.def("_fabric_clear_test_hooks_for_tests", []() {
    auto& hooks = vbt::cuda::fabric::fabric_test_hooks();
    hooks.forced_uva_ok.reset();
    hooks.fake_topology_builder = nullptr;
  });
#endif  // VBT_INTERNAL_TESTS
}

} // namespace vbt_py
