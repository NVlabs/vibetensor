// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include "vbt/cpu/allocator.h"

namespace nb = nanobind;

namespace vbt_py {

void bind_cpu_memory(nb::module_& m) {
  using vbt::cpu::Allocator;
  using vbt::cpu::PinnedHostAllocator;
  m.def("_cpu_getMemoryStats", [](){
    auto s = Allocator::get().getDeviceStats();
    return std::make_tuple(
      static_cast<std::uint64_t>(s.allocated_bytes_all_current),
      static_cast<std::uint64_t>(s.reserved_bytes_all_current),
      static_cast<std::uint64_t>(s.max_allocated_bytes_all),
      static_cast<std::uint64_t>(s.max_reserved_bytes_all)
    );
  });
  m.def("_cpu_emptyCache", [](){ Allocator::get().emptyCache(); });
  m.def("_cpu_resetPeakMemoryStats", [](){ Allocator::get().resetPeakStats(); });

  // Optional pinned-host stats with CUDA parity tuple ordering
  m.def("_cpu_getHostPinnedStats", [](){
    auto s = PinnedHostAllocator::get().getDeviceStats();
    return std::make_tuple(
      static_cast<std::uint64_t>(s.allocated_bytes_all_current),
      static_cast<std::uint64_t>(s.reserved_bytes_all_current),
      static_cast<std::uint64_t>(s.max_allocated_bytes_all),
      static_cast<std::uint64_t>(s.max_reserved_bytes_all)
    );
  });
  m.def("_cpu_emptyPinnedCache", [](){ PinnedHostAllocator::get().emptyCache(); });
  m.def("_cpu_resetPeakHostPinnedStats", [](){ PinnedHostAllocator::get().resetPeakStats(); });
}

} // namespace vbt_py
