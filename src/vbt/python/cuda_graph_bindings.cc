// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "vbt/cuda/graphs.h"
#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace nb = nanobind;

namespace vbt_py {

void bind_cuda_graphs(nb::module_& m) {
#if VBT_WITH_CUDA
  using vbt::cuda::Allocator;
  using vbt::cuda::CaptureMode;
  using vbt::cuda::CaptureStatus;
  using vbt::cuda::CUDAGraph;
  using vbt::cuda::DeviceIndex;
  using vbt::cuda::MempoolId;
  using vbt::cuda::GraphCounters;
  using vbt::cuda::cuda_graphs_counters;
  using vbt::cuda::currentStreamCaptureStatus;
  using vbt::cuda::detail::poll_deferred_graph_cleanup;

  // Low-level helper: is the current CUDA stream capturing?
  m.def("_cuda_isCurrentStreamCapturing", []() {
    CaptureStatus st = currentStreamCaptureStatus(static_cast<DeviceIndex>(-1));
    return st == CaptureStatus::Active;
  });

  // Low-level helper: create a new allocator graph pool handle for a device.
  m.def(
      "_graph_pool_handle",
      [](nb::object device) {
        int n = vbt::cuda::device_count();
        if (n == 0) {
          throw std::runtime_error("CUDA unavailable: no devices");
        }

        int dev = -1;
        if (device.is_none()) {
          // Map None → current CUDA device via cudaGetDevice.
          cudaError_t st;
          {
            nb::gil_scoped_release r;
            st = cudaGetDevice(&dev);
          }
          if (st != cudaSuccess) {
            const char* msg = cudaGetErrorString(st);
            std::string m = "failed to query current CUDA device: ";
            m += (msg ? msg : "");
            throw std::runtime_error(m);
          }
        } else {
          dev = nb::cast<int>(device);
          if (dev < 0) {
            throw std::runtime_error("device must be >= 0 or None");
          }
          if (dev >= n) {
            throw std::runtime_error("device index out of range");
          }
        }

        DeviceIndex di = static_cast<DeviceIndex>(dev);
        MempoolId id = Allocator::create_pool_id(di);
        return std::make_tuple(static_cast<int>(id.dev), static_cast<std::uint64_t>(id.id));
      },
      nb::arg("device").none(true) = nb::none());

  // Aggregate CUDA Graphs counters and per-device graph-pool gauges.
  m.def(
      "_cuda_graphs_stats",
      [](nb::object device) {
        poll_deferred_graph_cleanup();
        nb::dict out;

        // Graph counters: always available.
        GraphCounters counters = cuda_graphs_counters();
        nb::dict graphs;
        graphs["captures_started"]      = nb::int_(counters.captures_started);
        graphs["captures_ended"]        = nb::int_(counters.captures_ended);
        graphs["denied_default_stream"] = nb::int_(counters.denied_default_stream);
        graphs["nested_capture_denied"] = nb::int_(counters.nested_capture_denied);
        graphs["end_in_dtor"]           = nb::int_(counters.end_in_dtor);
        graphs["end_in_dtor_errors"]    = nb::int_(counters.end_in_dtor_errors);
        graphs["graphs_instantiated"]   = nb::int_(counters.graphs_instantiated);
        graphs["graphs_replayed"]       = nb::int_(counters.graphs_replayed);
        graphs["replay_nesting_errors"] = nb::int_(counters.replay_nesting_errors);
        graphs["unsupported_capture_mode"]    = nb::int_(counters.unsupported_capture_mode);
        graphs["capture_begin_invalid_state"] = nb::int_(counters.capture_begin_invalid_state);
        graphs["capture_end_invalid_state"]   = nb::int_(counters.capture_end_invalid_state);
        graphs["instantiate_invalid_state"]   = nb::int_(counters.instantiate_invalid_state);
        graphs["instantiate_errors"]          = nb::int_(counters.instantiate_errors);
        graphs["replay_invalid_state"]        = nb::int_(counters.replay_invalid_state);
        graphs["replay_device_mismatch"]      = nb::int_(counters.replay_device_mismatch);
        graphs["replay_errors"]               = nb::int_(counters.replay_errors);
        graphs["graphs_reset"]                = nb::int_(counters.graphs_reset);
        graphs["reset_invalid_state"]         = nb::int_(counters.reset_invalid_state);
        graphs["reset_inflight_denied"]       = nb::int_(counters.reset_inflight_denied);
        graphs["reset_errors"]                = nb::int_(counters.reset_errors);
        graphs["allocator_capture_denied"]    = nb::int_(counters.allocator_capture_denied);
        out["graphs"] = std::move(graphs);

        // Per-device graph-pool gauges; normalize device argument.
        int dev = -1;
        if (!device.is_none()) {
          dev = nb::cast<int>(device);
          if (dev < 0) {
            throw std::runtime_error("device must be >= 0 or None for current device");
          }
        }

        int n = vbt::cuda::device_count();
        nb::dict pools;
        if (n <= 0) {
          int norm = (dev >= 0) ? dev : 0;
          pools["device"]               = nb::int_(norm);
          pools["graphs_pools_created"] = nb::int_(0);
          pools["graphs_pools_active"]  = nb::int_(0);
          pools["graphs_pools_released"] = nb::int_(0);
          out["pools"] = std::move(pools);
          return out;
        }

        if (dev < 0) {
          dev = 0;  // default to device 0 when None
        } else if (dev >= n) {
          throw std::runtime_error("device index out of range");
        }

        DeviceIndex di = static_cast<DeviceIndex>(dev);
        auto stats = Allocator::get(di).getDeviceStats();
        pools["device"]               = nb::int_(dev);
        pools["graphs_pools_created"] = nb::int_(stats.graphs_pools_created);
        pools["graphs_pools_active"]  = nb::int_(stats.graphs_pools_active);
        pools["graphs_pools_released"] = nb::int_(stats.graphs_pools_released);
        out["pools"] = std::move(pools);
        return out;
      },
      nb::arg("device").none(true) = nb::none());

#if defined(VBT_INTERNAL_TESTS)
  // Test-only: directly expose release_pool for allocator/graphs pool lifecycle tests.
  m.def("_graph_pool_release", [](std::tuple<int, std::uint64_t> handle) {
    int dev = std::get<0>(handle);
    std::uint64_t id = std::get<1>(handle);
    MempoolId mp{static_cast<DeviceIndex>(dev), id};
    Allocator::release_pool(mp.dev, mp);
  });
#endif

  // CUDAGraph binding. This is a thin wrapper exposing capture_begin/end,
  // instantiate, replay, and pool.
  auto cls = nb::class_<CUDAGraph>(m, "_CUDAGraph");

  cls.def(nb::init<>());

  cls.def(
      "capture_begin",
      [](CUDAGraph& self,
         nb::handle pool,
         const std::string& capture_error_mode) {
        CaptureMode mode = CaptureMode::ThreadLocal;
        if (capture_error_mode == "thread_local") {
          mode = CaptureMode::ThreadLocal;
        } else if (capture_error_mode == "global" ||
                   capture_error_mode == "relaxed") {
          // Only ThreadLocal mode is supported currently.
          throw std::runtime_error(vbt::cuda::kErrUnsupportedCaptureMode);
        } else {
          throw std::runtime_error(
              "Unknown capture_error_mode. Expected 'thread_local'; "
              "'global' and 'relaxed' are not supported");
        }

        std::optional<MempoolId> mp;
        if (!pool.is_none()) {
          auto handle = nb::cast<std::tuple<int, std::uint64_t>>(pool);
          int dev = std::get<0>(handle);
          std::uint64_t id = std::get<1>(handle);
          MempoolId candidate{static_cast<DeviceIndex>(dev), id};
          mp = candidate;
        }

        // Let C++ pick the capture stream via getCurrentStream(-1).
        self.capture_begin(std::nullopt, mp, mode);
      },
      nb::arg("pool") = nb::none(),
      nb::arg("capture_error_mode") = std::string("thread_local"),
      nb::call_guard<nb::gil_scoped_release>());

  cls.def("capture_end", &CUDAGraph::capture_end,
          nb::call_guard<nb::gil_scoped_release>());

  cls.def("instantiate", &CUDAGraph::instantiate,
          nb::call_guard<nb::gil_scoped_release>());

  cls.def(
      "replay",
      [](CUDAGraph& self) {
        nb::gil_scoped_release r;
        self.replay(std::nullopt);
      },
      nb::call_guard<>());

  cls.def("reset", &CUDAGraph::reset,
          nb::call_guard<nb::gil_scoped_release>());

  cls.def(
      "pool",
      [](const CUDAGraph& self) {
        MempoolId id = self.pool();
        return std::make_tuple(static_cast<int>(id.dev), static_cast<std::uint64_t>(id.id));
      },
      nb::call_guard<nb::gil_scoped_release>());

#ifdef VBT_INTERNAL_TESTS
  // Debug helpers wired only for internal tests; no ABI guarantees.
  cls.def("_debug_inflight", &CUDAGraph::debug_inflight);
  cls.def("_debug_has_graph", &CUDAGraph::debug_has_graph);
  cls.def("_debug_has_exec", &CUDAGraph::debug_has_exec);
#endif

#else
  (void)m;
#endif
}

} // namespace vbt_py
