// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include "vbt/cuda/stream.h"
#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace nb = nanobind;

namespace vbt_py {

void bind_cuda_streams(nb::module_& m) {
#if VBT_WITH_CUDA
  using vbt::cuda::Stream;
  using vbt::cuda::priority_range;
  using vbt::cuda::getStreamFromPool;
  using vbt::cuda::getCurrentStream;
  using vbt::cuda::setCurrentStream;
  using vbt::cuda::DeviceIndex;

  auto cls = nb::class_<Stream>(m, "_CudaStreamBase");
  cls
     // Constructors: priority-only and (priority, device_index)
     .def(nb::init<int>(), nb::arg("priority") = 0)
     .def(nb::init<int, DeviceIndex>(), nb::arg("priority") = 0, nb::arg("device_index") = static_cast<DeviceIndex>(-1))
     // Methods
     .def("query", &Stream::query)
     .def("synchronize", [](const Stream& s){ nb::gil_scoped_release r; s.synchronize(); })
     .def("priority", &Stream::priority)
     .def_prop_ro("device_index", &Stream::device_index)
     .def_prop_ro("cuda_stream", &Stream::handle)
     // Static helpers
     .def_static("priority_range", [](){ auto pr = priority_range(); return std::make_tuple(pr.first, pr.second); })
     .def_static("current",
                [](DeviceIndex device_index) { return getCurrentStream(device_index); },
                nb::arg("device_index") = static_cast<DeviceIndex>(-1))
     .def_static("set_current", [](const Stream& s){ setCurrentStream(s); }, nb::arg("stream"))
    .def_static("set_current_with_device", [](const Stream& s){
#if VBT_WITH_CUDA
      int dev = static_cast<int>(s.device_index());
      // Best-effort; ignore errors here
      (void)cudaSetDevice(dev);
#endif
      setCurrentStream(s);
    }, nb::arg("stream"))
     // Dunder/util
     .def("__repr__", [](const Stream& s){ return std::string("<vibetensor._C._CudaStreamBase ") + vbt::cuda::to_string(s) + ">"; })
     .def("__eq__", [](const Stream& a, const Stream& b){ return a == b; })
     ;
#else
  (void)m;
#endif
}

} // namespace vbt_py
