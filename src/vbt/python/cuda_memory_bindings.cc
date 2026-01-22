// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include <optional>
#include <string>
#include <limits>

#include "vbt/core/tensor.h"
#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/storage.h"

namespace nb = nanobind;

namespace vbt_py {

#if VBT_WITH_CUDA
namespace {

struct ParsedDeviceIndex {
  int  index;        // normalized into [0, device_count-1]
  int  device_count; // snapshot of device_count()
  bool cpu_only;     // true iff device_count == 0
};

ParsedDeviceIndex parse_device_index(std::optional<int> device_arg,
                                     int device_count,
                                     int current_device,
                                     const char* api_name) {
  ParsedDeviceIndex out;
  out.device_count = device_count;
  out.cpu_only = (device_count == 0);

  // Negative indices are always invalid, even when device_count==0.
  if (device_arg.has_value() && *device_arg < 0) {
    throw std::runtime_error(
        std::string(api_name) + ": device must be >= 0 or None for current device");
  }

  if (device_count == 0) {
    // CPU-only: caller must not touch Allocator or CUDA; index is unused.
    out.index = 0;
    return out;
  }

  int idx = device_arg.has_value() ? *device_arg : current_device;
  if (idx < 0 || idx >= device_count) {
    throw std::runtime_error(std::string(api_name) + ": device index out of range");
  }

  out.index = idx;
  return out;
}

ParsedDeviceIndex parse_device_arg(nb::object device, const char* api_name) {
  int n = vbt::cuda::device_count();

  std::optional<int> dev_opt;
  if (!device.is_none()) {
    dev_opt = nb::cast<int>(device);
  }

  int current = 0;
#if VBT_WITH_CUDA
  if (n > 0 && !dev_opt.has_value()) {
    int cur = 0;
    cudaError_t st = cudaGetDevice(&cur);
    if (st != cudaSuccess) {
      (void)cudaGetLastError();
      cur = 0;
    }
    current = cur;
  }
#endif

  return parse_device_index(dev_opt, n, current, api_name);
}

} // anonymous namespace
#endif

void bind_cuda_memory(nb::module_& m) {
#if VBT_WITH_CUDA
  using vbt::cuda::Allocator;
  using vbt::cuda::DeviceIndex;
  using vbt::cuda::DeviceStats;
  using vbt::cuda::GraphPoolSnapshot;
  using vbt::cuda::MempoolId;
  using vbt::cuda::MemorySegmentSnapshot;
  using vbt::cuda::snapshot_graph_pools;
  using vbt::cuda::snapshot_memory_segments;
  using vbt::cuda::Stream;
  using vbt::cuda::detail::poll_deferred_graph_cleanup;

  // Basic device stats (aggregated gauges/peaks)
  m.def("_cuda_getDeviceStats",
        [](nb::object device) -> std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t> {
          int dev = -1;
          if (!device.is_none()) dev = nb::cast<int>(device);
          if (!device.is_none()) {
            if (dev < 0) throw std::runtime_error("device must be >= 0 or None for current device");
            int n = vbt::cuda::device_count();
            if (n == 0) {
              return std::make_tuple(0ull, 0ull, 0ull, 0ull);
            }
            if (dev >= n) throw std::runtime_error("device index out of range");
          } else {
            // device is None: if there are no CUDA devices, return zeros
            int n = vbt::cuda::device_count();
            if (n == 0) {
              return std::make_tuple(0ull, 0ull, 0ull, 0ull);
            }
          }
          auto s = Allocator::get(static_cast<DeviceIndex>(dev)).getDeviceStats();
          return std::make_tuple(
              static_cast<std::uint64_t>(s.allocated_bytes_all_current),
              static_cast<std::uint64_t>(s.reserved_bytes_all_current),
              static_cast<std::uint64_t>(s.max_allocated_bytes_all),
              static_cast<std::uint64_t>(s.max_reserved_bytes_all));
        }, nb::arg("device").none(true) = nb::none());

  // Extended stats tree used by Python memory_stats* helpers.
  m.def(
      "_cuda_memoryStats",
      [](nb::object device) {
        ParsedDeviceIndex parsed = parse_device_arg(device, "_cuda_memoryStats");
        nb::dict out;
        if (parsed.cpu_only) {
          return out;
        }

        DeviceStats st =
            Allocator::get(static_cast<DeviceIndex>(parsed.index)).getDeviceStats();

        nb::dict fam_alloc;
        nb::dict fam_reserved;
        nb::dict fam_requested;

        nb::dict alloc_all;
        alloc_all["current"] = nb::int_(st.allocated_bytes_all_current);
        alloc_all["peak"] = nb::int_(st.max_allocated_bytes_all);
        alloc_all["allocated"] = nb::int_(0);
        alloc_all["freed"] = nb::int_(0);
        fam_alloc["all"] = std::move(alloc_all);

        nb::dict reserved_all;
        reserved_all["current"] = nb::int_(st.reserved_bytes_all_current);
        reserved_all["peak"] = nb::int_(st.max_reserved_bytes_all);
        reserved_all["allocated"] = nb::int_(0);
        reserved_all["freed"] = nb::int_(0);
        fam_reserved["all"] = std::move(reserved_all);

        nb::dict requested_all;
        requested_all["current"] = nb::int_(st.requested_bytes_all_current);
        requested_all["peak"] = nb::int_(st.max_requested_bytes_all);
        requested_all["allocated"] = nb::int_(0);
        requested_all["freed"] = nb::int_(0);
        fam_requested["all"] = std::move(requested_all);

        out["allocated_bytes"] = std::move(fam_alloc);
        out["reserved_bytes"] = std::move(fam_reserved);
        out["requested_bytes"] = std::move(fam_requested);

        // Scalar counters mirroring all DeviceStats fields.
        out["num_alloc_retries"] = nb::int_(st.num_alloc_retries);
        out["num_ooms"] = nb::int_(st.num_ooms);
        out["num_device_alloc"] = nb::int_(st.num_device_alloc);
        out["num_device_free"] = nb::int_(st.num_device_free);
        out["tolerance_fills_count"] = nb::int_(st.tolerance_fills_count);
        out["tolerance_fills_bytes"] = nb::int_(st.tolerance_fills_bytes);
        out["deferred_flush_attempts"] = nb::int_(st.deferred_flush_attempts);
        out["deferred_flush_successes"] = nb::int_(st.deferred_flush_successes);
        out["num_prev_owner_fences"] = nb::int_(st.num_prev_owner_fences);
        out["inactive_split_blocks_all"] = nb::int_(st.inactive_split_blocks_all);
        out["inactive_split_bytes_all"] = nb::int_(st.inactive_split_bytes_all);
        out["fraction_cap_breaches"] = nb::int_(st.fraction_cap_breaches);
        out["fraction_cap_misfires"] = nb::int_(st.fraction_cap_misfires);
        out["gc_passes"] = nb::int_(st.gc_passes);
        out["gc_reclaimed_bytes"] = nb::int_(st.gc_reclaimed_bytes);
        out["graphs_pools_created"] = nb::int_(st.graphs_pools_created);
        out["graphs_pools_released"] = nb::int_(st.graphs_pools_released);
        out["graphs_pools_active"] = nb::int_(st.graphs_pools_active);
        return out;
      },
      nb::arg("device").none(true) = nb::none());

  // Mutators
  m.def("_cuda_emptyCache", [](){ Allocator::get(static_cast<DeviceIndex>(-1)).emptyCache(); });

  // Record stream usage for allocator safety.
  m.def("_cuda_record_stream", [](const vbt::core::TensorImpl& t, nb::int_ stream_handle) {
    auto dev = t.device();
    if (dev.type != kDLCUDA) {
      throw nb::type_error("_cuda_record_stream: expected a CUDA tensor");
    }
    int ndev = vbt::cuda::device_count();
    if (ndev <= 0) {
      throw nb::type_error("_cuda_record_stream: CUDA is not available");
    }
    if (dev.index < 0 || dev.index >= ndev ||
        dev.index > std::numeric_limits<DeviceIndex>::max()) {
      throw nb::value_error("_cuda_record_stream: device index out of range");
    }
    if (PyBool_Check(stream_handle.ptr())) {
      throw nb::type_error("_cuda_record_stream: stream_handle must be an int, not bool");
    }
    unsigned long long h = PyLong_AsUnsignedLongLong(stream_handle.ptr());
    if (PyErr_Occurred()) {
      PyErr_Clear();
      throw nb::value_error(
          "_cuda_record_stream: stream_handle must be a non-negative int that fits in uint64");
    }
    Stream s(Stream::UNCHECKED, static_cast<uint64_t>(h), static_cast<DeviceIndex>(dev.index));
    vbt::cuda::record_stream(t.storage(), s);
  }, nb::arg("tensor"), nb::arg("stream_handle"));

  // Debug-only counter for record_stream integration.
  m.def("_cuda_debug_record_stream_call_count", []() {
    return vbt::cuda::debug_record_stream_call_count();
  });
  m.def("_cuda_debug_reset_record_stream_call_count", []() {
    vbt::cuda::debug_reset_record_stream_call_count();
  });

  m.def("_cuda_resetPeakMemoryStats", [](nb::object device){
    int dev = -1; if (!device.is_none()) dev = nb::cast<int>(device);
    if (!device.is_none()) {
      if (dev < 0) throw std::runtime_error("device must be >= 0 or None for current device");
      int n = vbt::cuda::device_count();
      if (n == 0) return; // no-op
      if (dev >= n) throw std::runtime_error("device index out of range");
    }
    Allocator::get(static_cast<DeviceIndex>(dev)).resetPeakStats();
  }, nb::arg("device").none(true) = nb::none());

  m.def("_cuda_resetAccumulatedMemoryStats", [](nb::object device){
    int dev = -1; if (!device.is_none()) dev = nb::cast<int>(device);
    if (!device.is_none()) {
      if (dev < 0) throw std::runtime_error("device must be >= 0 or None for current device");
      int n = vbt::cuda::device_count();
      if (n == 0) return; // no-op
      if (dev >= n) throw std::runtime_error("device index out of range");
    }
    Allocator::get(static_cast<DeviceIndex>(dev)).resetAccumulatedStats();
  }, nb::arg("device").none(true) = nb::none());

  // Memory fraction APIs (skeleton)
  m.def("_cuda_setMemoryFraction", [](double fraction, nb::object device){
    int dev = -1; if (!device.is_none()) dev = nb::cast<int>(device);
    if (!device.is_none()) {
      if (dev < 0) throw std::runtime_error("device must be >= 0 or None for current device");
      int n = vbt::cuda::device_count();
      if (n == 0) return; // no-op
      if (dev >= n) throw std::runtime_error("device index out of range");
    }
    if (fraction < 0.0) fraction = 0.0; if (fraction > 1.0) fraction = 1.0;
    Allocator::get(static_cast<DeviceIndex>(dev)).setMemoryFraction(fraction);
  }, nb::arg("fraction"), nb::arg("device").none(true) = nb::none());

  m.def("_cuda_getMemoryFraction", [](nb::object device){
    int dev = -1; if (!device.is_none()) dev = nb::cast<int>(device);
    if (!device.is_none()) {
      if (dev < 0) throw std::runtime_error("device must be >= 0 or None for current device");
      int n = vbt::cuda::device_count();
      if (n == 0) return 0.0; // default
      if (dev >= n) throw std::runtime_error("device index out of range");
    }
    return Allocator::get(static_cast<DeviceIndex>(dev)).getMemoryFraction();
  }, nb::arg("device").none(true) = nb::none());

  m.def(
      "_cuda_memorySnapshot",
      [](nb::object device) {
        ParsedDeviceIndex parsed = parse_device_arg(device, "_cuda_memorySnapshot");
        nb::list out;
        if (parsed.cpu_only) {
          return out;
        }

        std::optional<DeviceIndex> filter;
        if (!device.is_none()) {
          filter = static_cast<DeviceIndex>(parsed.index);
        }

        std::vector<MemorySegmentSnapshot> snaps = snapshot_memory_segments(filter);
        for (const auto& s : snaps) {
          nb::dict d;
          d["device"]        = nb::int_(static_cast<int>(s.device));
          d["pool_id"]       = nb::int_(s.pool_id);
          d["bytes_reserved"] = nb::int_(s.bytes_reserved);
          d["bytes_active"]   = nb::int_(s.bytes_active);
          d["blocks"]         = nb::int_(s.blocks);
          out.append(std::move(d));
        }
        return out;
      },
      nb::arg("device").none(true) = nb::none());

  // Snapshot graph-private pools across devices.
  // filter: None -> all devices/pools; (dev, 0) -> all pools on dev;
  // (dev, id>0) -> that pool only.
  m.def(
      "_cuda_graph_pools_snapshot",
      [](nb::object filter) {
        poll_deferred_graph_cleanup();
        std::optional<MempoolId> id_filter;
        if (!filter.is_none()) {
          auto tup = nb::cast<std::pair<int, std::uint64_t>>(filter);
          int dev = tup.first;
          std::uint64_t pid = tup.second;
          id_filter = MempoolId{static_cast<DeviceIndex>(dev), pid};
        }

        std::vector<GraphPoolSnapshot> snaps = snapshot_graph_pools(id_filter);
        nb::list out;
        for (const auto& s : snaps) {
          nb::dict d;
          d["device"]         = nb::int_(static_cast<int>(s.id.dev));
          d["id"]             = nb::int_(s.id.id);
          d["segments"]       = nb::int_(s.segments);
          d["blocks"]         = nb::int_(s.blocks);
          d["bytes_reserved"] = nb::int_(s.bytes_reserved);
          d["bytes_active"]   = nb::int_(s.bytes_active);
          out.append(std::move(d));
        }
        return out;
      },
      nb::arg("filter").none(true) = nb::none());

  // Internal helper: pre-warm allocator blocks for the current CUDA stream.
  // Allocates and immediately frees `count` blocks of size `nbytes` on the
  // current stream of the chosen device. This mirrors the C++ allocator
  // routing tests and is used by the Python CUDA Graphs overlay to ensure
  // graph-private pools can reuse cached blocks during capture.
  m.def(
      "_cuda_allocator_prewarm_current_stream",
      [](std::size_t nbytes, int count, nb::object device) {
        if (nbytes == 0 || count <= 0) {
          return;
        }
        int dev = -1;
        if (!device.is_none()) {
          dev = nb::cast<int>(device);
          if (dev < 0) {
            throw std::runtime_error("device must be >= 0 or None for current device");
          }
          int n = vbt::cuda::device_count();
          if (n == 0) {
            return;  // CPU-only build; nothing to do
          }
          if (dev >= n) {
            throw std::runtime_error("device index out of range");
          }
        }
        DeviceIndex di = static_cast<DeviceIndex>(dev);
        Allocator& A = Allocator::get(di);
        Stream s = vbt::cuda::getCurrentStream(di);
        for (int i = 0; i < count; ++i) {
          void* p = nullptr;
          try {
            p = A.raw_alloc(nbytes, s);
          } catch (...) {
            break;  // best-effort: stop on first failure
          }
          if (!p) {
            break;
          }
          A.raw_delete(p);
        }
      },
      nb::arg("nbytes"),
      nb::arg("count") = 1,
      nb::arg("device").none(true) = nb::none());
#else
  (void)m;
#endif
}

} // namespace vbt_py
