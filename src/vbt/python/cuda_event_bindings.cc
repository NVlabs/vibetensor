// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include "vbt/cuda/event.h"
#include "vbt/cuda/stream.h"

namespace nb = nanobind;

namespace vbt_py {

void bind_cuda_events(nb::module_& m) {
#if VBT_WITH_CUDA
  using vbt::cuda::Event;
  using vbt::cuda::Stream;

  auto cls = nb::class_<Event>(m, "_CudaEventBase");
  cls.def(nb::init<bool>(), nb::arg("enable_timing") = false)
     .def("record", [](Event& e, const Stream& s){ nb::gil_scoped_release r; e.record(s); })
     .def("wait",   [](const Event& e, const Stream& s){ nb::gil_scoped_release r; e.wait(s); })
     .def("query",  &Event::query)
     .def("synchronize", [](const Event& e){ nb::gil_scoped_release r; e.synchronize(); })
     .def("is_created", &Event::is_created)
     ;
#else
  (void)m;
#endif
}

} // namespace vbt_py
