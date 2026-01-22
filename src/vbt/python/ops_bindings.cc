// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>

#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/tensor.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/boxed.h"

namespace nb = nanobind;

namespace vbt_py {

static inline vbt::core::TensorImpl make_unit_tensor() {
  using namespace vbt::core;
  void* buf = ::operator new(static_cast<std::size_t>(itemsize(ScalarType::Float32)));
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto storage = make_intrusive<Storage>(std::move(dp), itemsize(ScalarType::Float32));
  return TensorImpl(storage, {}, {}, 0, ScalarType::Float32, Device::cpu());
}

void bind_ops(nb::module_& m) {
  using vbt::core::TensorImpl;
  using vbt::dispatch::Dispatcher;
  using vbt::dispatch::BoxedStack;

  // Keep unit() as a simple helper for tests; dispatcher also registers a CPU kernel.
  m.def("unit", [](){
    // Use dispatcher nullary to stay consistent with registry, but fall back to local impl if missing.
    try {
      BoxedStack s{};
      Dispatcher::instance().callBoxed("vt::unit", s);
      return s[0];
    } catch (...) {
      return make_unit_tensor();
    }
  });

  m.def("relu", [](const TensorImpl& a){
    BoxedStack s{a};
    Dispatcher::instance().callBoxed("vt::relu", s);
    return s[0];
  });

  m.def("add", [](const TensorImpl& a, const TensorImpl& b){
    BoxedStack s{a, b};
    Dispatcher::instance().callBoxed("vt::add", s);
    return s[0];
  });

  m.def("mul", [](const TensorImpl& a, const TensorImpl& b){
    BoxedStack s{a, b};
    Dispatcher::instance().callBoxed("vt::mul", s);
    return s[0];
  });

  m.def("add3", [](const TensorImpl& a, const TensorImpl& b, const TensorImpl& c){
    BoxedStack s{a, b, c};
    Dispatcher::instance().callBoxed("vt::add3", s);
    return s[0];
  });
}

} // namespace vbt_py
