// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/nb_defs.h>

#include <atomic>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>
#include <stdexcept>
#include <utility>

#include <dlpack/dlpack.h>
#include <Python.h>

#include "vbt/interop/dlpack.h"
#include "vbt/core/tensor.h"

namespace nb = nanobind;

namespace vbt_py {

namespace {

#if VBT_INTERNAL_TESTS
std::atomic<int> g_fail_wrap_versioned_alloc{0};
std::atomic<std::int64_t> g_test_deleter_calls{0};
#endif

constexpr const char* kErrComplexDisabled =
    "complex dtypes are disabled; set VBT_ENABLE_COMPLEX=1";

inline bool complex_enabled_from_env() noexcept {
  const char* raw = std::getenv("VBT_ENABLE_COMPLEX");
  return raw && raw[0] == '1' && raw[1] == '\0';
}

inline void throw_if_complex_disabled(const DLDataType& dt) {
  if (dt.code == static_cast<std::uint8_t>(kDLComplex) &&
      !complex_enabled_from_env()) {
    throw nb::type_error(kErrComplexDisabled);
  }
}

// Wrap a DLManagedTensorVersioned into a legacy DLManagedTensor adapter.
static DLManagedTensor* wrap_versioned_as_legacy(DLManagedTensorVersioned* v) {
  if (!v) return nullptr;

#if VBT_INTERNAL_TESTS
  // Deterministic allocation failure injection for tests.
  // When g_fail_wrap_versioned_alloc > 0, the next allocation attempt throws.
  int remaining = g_fail_wrap_versioned_alloc.load(std::memory_order_relaxed);
  while (remaining > 0) {
    if (g_fail_wrap_versioned_alloc.compare_exchange_weak(
            remaining, remaining - 1, std::memory_order_relaxed)) {
      throw std::bad_alloc();
    }
  }
#endif

  // Allocate a new legacy wrapper whose lifetime is tied to the provider deleter.
  DLManagedTensor* legacy = new DLManagedTensor{};
  legacy->dl_tensor = v->dl_tensor; // copy DLTensor by value
  legacy->manager_ctx = v;
  legacy->deleter = [](DLManagedTensor* self) {
    if (!self) return;
    auto* ver = reinterpret_cast<DLManagedTensorVersioned*>(self->manager_ctx);
    if (ver && ver->deleter) {
      try {
        ver->deleter(ver);
      } catch (...) {
      }
    }
    delete self;
  };
  return legacy;
}

static void* get_capsule_pointer_or_throw(PyObject* obj, const char* expected_name) {
  void* p = PyCapsule_GetPointer(obj, expected_name);
  if (!p) {
    // PyCapsule_GetPointer sets a Python error on failure.
    if (PyErr_Occurred()) throw nb::python_error();
    throw nb::value_error("from_dlpack: expected a 'dltensor' capsule");
  }
  return p;
}

static void consume_capsule_or_throw(PyObject* obj, const char* old_name) {
  PyCapsule_Destructor old_destructor = PyCapsule_GetDestructor(obj);

  // 1) rename
  if (PyCapsule_SetName(obj, "used_dltensor") != 0) {
    throw nb::python_error();
  }

  // 2) disable destructor
  if (PyCapsule_SetDestructor(obj, nullptr) != 0) {
    // Roll back best-effort.
    (void)PyCapsule_SetName(obj, old_name);
    (void)PyCapsule_SetDestructor(obj, old_destructor);
    throw nb::python_error();
  }
}

#if VBT_INTERNAL_TESTS
static void dlpack_test_legacy_deleter(DLManagedTensor* self) {
  if (!self) return;
  g_test_deleter_calls.fetch_add(1, std::memory_order_relaxed);
  delete[] self->dl_tensor.shape;
  delete[] self->dl_tensor.strides;
  delete self;
}

static void dlpack_test_versioned_deleter(DLManagedTensorVersioned* self) {
  if (!self) return;
  g_test_deleter_calls.fetch_add(1, std::memory_order_relaxed);
  delete[] self->dl_tensor.shape;
  delete[] self->dl_tensor.strides;
  delete self;
}

// Provider-style capsule destructor: if still unused, call the stored deleter.
static void dlpack_test_capsule_destructor(void* p) noexcept {
  auto* mt = reinterpret_cast<DLManagedTensor*>(p);
  if (!mt) return;
  auto* del = std::exchange(mt->deleter, nullptr);
  if (del) {
    try {
      del(mt);
    } catch (...) {
    }
  }
}

static void dlpack_test_capsule_destructor_versioned(void* p) noexcept {
  auto* mt = reinterpret_cast<DLManagedTensorVersioned*>(p);
  if (!mt) return;
  auto* del = std::exchange(mt->deleter, nullptr);
  if (del) {
    try {
      del(mt);
    } catch (...) {
    }
  }
}
#endif

} // namespace

void bind_dlpack(nb::module_& m) {
  using vbt::core::TensorImpl;

  m.def("_to_dlpack", [](const TensorImpl& t) {
    auto up = vbt::interop::to_dlpack(t);
    DLManagedTensor* raw = up.release();
    return nb::capsule(raw, "dltensor", [](void* /*p*/) noexcept {
      // no-op: per DLPack protocol, the consumer must invoke mt->deleter
    });
  });

  m.def("_from_dlpack", [](nb::capsule cap) {
    const char* name = cap.name();
    if (!name) {
      throw nb::value_error("from_dlpack: expected a 'dltensor' capsule");
    }
    const std::string n{name};

    if (n == "used_dltensor") {
      throw std::runtime_error("from_dlpack: capsule already consumed");
    }
    if (n != "dltensor" && n != "dltensor_versioned") {
      throw nb::value_error("from_dlpack: expected a 'dltensor' capsule");
    }

    PyObject* obj = cap.ptr();
    if (!obj) {
      throw nb::value_error("from_dlpack: expected a 'dltensor' capsule");
    }

    void* cap_ptr = nullptr;
    DLManagedTensor* mt = nullptr;
    std::unique_ptr<DLManagedTensor> wrapper_guard;

    if (n == "dltensor_versioned") {
      cap_ptr = get_capsule_pointer_or_throw(obj, "dltensor_versioned");
      auto* ver = reinterpret_cast<DLManagedTensorVersioned*>(cap_ptr);
      throw_if_complex_disabled(ver->dl_tensor.dtype);
      wrapper_guard.reset(wrap_versioned_as_legacy(ver));
      mt = wrapper_guard.get();
      if (!mt) {
        throw nb::value_error("from_dlpack: expected a 'dltensor' capsule");
      }

      // Consume only after wrapper allocation succeeds.
      consume_capsule_or_throw(obj, "dltensor_versioned");
      wrapper_guard.release();
      return vbt::interop::from_dlpack(mt);
    }

    // Legacy capsule path.
    cap_ptr = get_capsule_pointer_or_throw(obj, "dltensor");
    mt = reinterpret_cast<DLManagedTensor*>(cap_ptr);
    throw_if_complex_disabled(mt->dl_tensor.dtype);

    consume_capsule_or_throw(obj, "dltensor");
    return vbt::interop::from_dlpack(mt);
  });

#if VBT_WITH_CUDA
  m.def("_from_dlpack_cuda_copy", [](nb::capsule cap) {
    const char* name = cap.name();
    if (!name) {
      throw nb::value_error("from_dlpack: expected a 'dltensor' capsule");
    }
    const std::string n{name};

    if (n == "used_dltensor") {
      throw std::runtime_error("from_dlpack: capsule already consumed");
    }
    if (n != "dltensor" && n != "dltensor_versioned") {
      throw nb::value_error("from_dlpack: expected a 'dltensor' capsule");
    }

    PyObject* obj = cap.ptr();
    if (!obj) {
      throw nb::value_error("from_dlpack: expected a 'dltensor' capsule");
    }

    void* cap_ptr = nullptr;

    if (n == "dltensor_versioned") {
      cap_ptr = get_capsule_pointer_or_throw(obj, "dltensor_versioned");
      auto* ver = reinterpret_cast<DLManagedTensorVersioned*>(cap_ptr);
      throw_if_complex_disabled(ver->dl_tensor.dtype);

      // Stage A: reject without consuming so the overlay can fall back.
      const int dev_type = ver->dl_tensor.device.device_type;
      if (dev_type != kDLCUDA && dev_type != kDLCUDAHost &&
          dev_type != kDLCUDAManaged) {
        throw nb::value_error("from_dlpack: unsupported device type for CUDA import");
      }

      std::unique_ptr<DLManagedTensor> wrapper_guard(wrap_versioned_as_legacy(ver));
      DLManagedTensor* mt = wrapper_guard.get();
      if (!mt) {
        throw nb::value_error("from_dlpack: expected a 'dltensor' capsule");
      }

      consume_capsule_or_throw(obj, "dltensor_versioned");
      wrapper_guard.release();
      return vbt::interop::from_dlpack_cuda_copy(mt);
    }

    // Legacy capsule path.
    cap_ptr = get_capsule_pointer_or_throw(obj, "dltensor");
    auto* mt = reinterpret_cast<DLManagedTensor*>(cap_ptr);
    throw_if_complex_disabled(mt->dl_tensor.dtype);

    // Stage A: reject without consuming so the overlay can fall back.
    const int dev_type = mt->dl_tensor.device.device_type;
    if (dev_type != kDLCUDA && dev_type != kDLCUDAHost &&
        dev_type != kDLCUDAManaged) {
      throw nb::value_error("from_dlpack: unsupported device type for CUDA import");
    }

    consume_capsule_or_throw(obj, "dltensor");
    return vbt::interop::from_dlpack_cuda_copy(mt);
  });
#endif

#if VBT_INTERNAL_TESTS
  m.def("_dlpack_test_set_wrap_versioned_alloc_fail_count", [](int n) {
    if (n < 0) n = 0;
    g_fail_wrap_versioned_alloc.store(n, std::memory_order_relaxed);
  });

  m.def("_dlpack_test_reset_deleter_call_count", []() {
    g_test_deleter_calls.store(0, std::memory_order_relaxed);
  });

  m.def("_dlpack_test_get_deleter_call_count", []() {
    return static_cast<long long>(g_test_deleter_calls.load(std::memory_order_relaxed));
  });

  m.def("_dlpack_test_make_legacy_capsule_invalid_dtype", []() {
    auto* mt = new DLManagedTensor{};
    mt->dl_tensor.data = reinterpret_cast<void*>(0x100000);
    mt->dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
    // Invalid dtype encoding: lanes must be 1.
    mt->dl_tensor.dtype =
        DLDataType{static_cast<std::uint8_t>(kDLFloat), 32, 2};
    mt->dl_tensor.ndim = 1;
    mt->dl_tensor.shape = new int64_t[1]{1};
    mt->dl_tensor.strides = nullptr;
    mt->dl_tensor.byte_offset = 0;
    mt->manager_ctx = nullptr;
    mt->deleter = dlpack_test_legacy_deleter;
    return nb::capsule(mt, "dltensor", dlpack_test_capsule_destructor);
  });

  m.def("_dlpack_test_make_versioned_capsule_float32", []() {
    auto* ver = new DLManagedTensorVersioned{};
    ver->version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    ver->manager_ctx = nullptr;
    ver->deleter = dlpack_test_versioned_deleter;
    ver->flags = 0;
    ver->dl_tensor.data = reinterpret_cast<void*>(0x200000);
    ver->dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
    ver->dl_tensor.dtype =
        DLDataType{static_cast<std::uint8_t>(kDLFloat), 32, 1};
    ver->dl_tensor.ndim = 1;
    ver->dl_tensor.shape = new int64_t[1]{1};
    ver->dl_tensor.strides = nullptr;
    ver->dl_tensor.byte_offset = 0;
    return nb::capsule(ver, "dltensor_versioned",
                       dlpack_test_capsule_destructor_versioned);
  });
#endif
}

} // namespace vbt_py
