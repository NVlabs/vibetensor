// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <limits>

#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/checked_math.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/allocator.h"

#if VBT_WITH_CUDA
#include <cuda.h>
#endif

namespace nb = nanobind;

namespace vbt_py {

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;

namespace {

bool unsafe_legacy_int_packing_enabled() {
  const char* v = std::getenv("VBT_CUDA_LAUNCH_UNSAFE_LEGACY_INT_PACKING");
  return v && v[0] == '1' && v[1] == '\0';
}

// Pre-packed argument with explicit byte representation (for correct i32/i64 typing)
struct CudaArgV2 {
  std::vector<unsigned char> bytes;
};

template <typename T>
CudaArgV2 cuda_arg_from_scalar(const T& v) {
  CudaArgV2 out;
  out.bytes.resize(sizeof(T));
  std::memcpy(out.bytes.data(), &v, sizeof(T));
  return out;
}

struct PackedCudaLaunchArgs {
  std::vector<std::vector<unsigned char>> storage;
  std::vector<void*> argv;
};

PackedCudaLaunchArgs pack_cuda_launch_args(const nb::list& args_py, const char* api_name) {
  PackedCudaLaunchArgs out;
  out.storage.reserve(args_py.size());
  out.argv.reserve(args_py.size());

  const bool unsafe_int_packing = unsafe_legacy_int_packing_enabled();

  for (nb::handle h : args_py) {
    if (nb::isinstance<TensorImpl>(h)) {
      // Device pointer from VBT CUDA tensor
      const TensorImpl& t = nb::cast<const TensorImpl&>(h);
      auto dev = t.device();
      if (dev.type != kDLCUDA) throw nb::type_error("expected a CUDA tensor for pointer arg");
      void* ptr = t.data();
      // Store pointer value
      out.storage.emplace_back(sizeof(void*));
      std::memcpy(out.storage.back().data(), &ptr, sizeof(void*));
      out.argv.push_back(out.storage.back().data());
    } else if (nb::isinstance<CudaArgV2>(h)) {
      // Pre-packed argument with correct byte size - just copy the bytes
      const CudaArgV2& arg = nb::cast<const CudaArgV2&>(h);
      out.storage.push_back(arg.bytes);
      out.argv.push_back(out.storage.back().data());
    } else if (nb::isinstance<nb::bool_>(h)) {
      // Pack bool into a wider slot to avoid host OOB reads if legacy callers
      // accidentally use bool for an int-like flag.
      uint64_t v = nb::cast<bool>(h) ? 1ull : 0ull;
      out.storage.emplace_back(sizeof(uint64_t));
      std::memcpy(out.storage.back().data(), &v, sizeof(uint64_t));
      out.argv.push_back(out.storage.back().data());
    } else if (nb::isinstance<nb::int_>(h)) {
      long long v64 = nb::cast<long long>(h);
      if (unsafe_int_packing && v64 >= std::numeric_limits<int32_t>::min() &&
          v64 <= std::numeric_limits<int32_t>::max()) {
        int32_t v = static_cast<int32_t>(v64);
        out.storage.emplace_back(sizeof(int32_t));
        std::memcpy(out.storage.back().data(), &v, sizeof(int32_t));
      } else {
        int64_t v = static_cast<int64_t>(v64);
        out.storage.emplace_back(sizeof(int64_t));
        std::memcpy(out.storage.back().data(), &v, sizeof(int64_t));
      }
      out.argv.push_back(out.storage.back().data());
    } else if (nb::isinstance<nb::float_>(h)) {
      float v = nb::cast<float>(h);
      out.storage.emplace_back(sizeof(float));
      std::memcpy(out.storage.back().data(), &v, sizeof(float));
      out.argv.push_back(out.storage.back().data());
    } else if (h.is_none()) {
      void* ptr = nullptr;
      out.storage.emplace_back(sizeof(void*));
      std::memcpy(out.storage.back().data(), &ptr, sizeof(void*));
      out.argv.push_back(out.storage.back().data());
    } else {
      std::string msg = "unsupported argument type for ";
      msg += api_name;
      throw nb::type_error(msg.c_str());
    }
  }

  return out;
}

CudaArgV2 cuda_arg_from_bytes_like(nb::handle h, const char* api_name) {
  struct Guard {
    Py_buffer view{};
    bool active{false};

    Guard(nb::handle obj, const char* api_name) {
      if (PyObject_GetBuffer(obj.ptr(), &view, PyBUF_CONTIG_RO) != 0) {
        PyErr_Clear();
        std::string msg = api_name;
        msg += ": expected a bytes-like object";
        throw nb::type_error(msg.c_str());
      }
      active = true;
    }

    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;

    ~Guard() {
      if (active) {
        PyBuffer_Release(&view);
      }
    }
  } g(h, api_name);

  if (g.view.len < 0) {
    std::string msg = api_name;
    msg += ": invalid buffer length";
    throw nb::value_error(msg.c_str());
  }

  const std::size_t nbytes = static_cast<std::size_t>(g.view.len);
  if (nbytes > 4096u) {
    std::string msg = api_name;
    msg += ": data must be <= 4096 bytes";
    throw nb::value_error(msg.c_str());
  }

  CudaArgV2 out;
  out.bytes.resize(nbytes);
  if (nbytes) {
    std::memcpy(out.bytes.data(), g.view.buf, nbytes);
  }
  return out;
}

long long parse_py_int64_strict(nb::int_ x, const char* api_name) {
  if (PyBool_Check(x.ptr())) {
    std::string msg = api_name;
    msg += ": expected an int, not bool";
    throw nb::type_error(msg.c_str());
  }
  long long v = PyLong_AsLongLong(x.ptr());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    std::string msg = api_name;
    msg += ": value must fit in int64";
    throw nb::value_error(msg.c_str());
  }
  return v;
}

unsigned long long parse_py_uint64_strict(nb::int_ x, const char* api_name) {
  if (PyBool_Check(x.ptr())) {
    std::string msg = api_name;
    msg += ": expected an int, not bool";
    throw nb::type_error(msg.c_str());
  }
  unsigned long long v = PyLong_AsUnsignedLongLong(x.ptr());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    std::string msg = api_name;
    msg += ": value must be a non-negative int that fits in uint64";
    throw nb::value_error(msg.c_str());
  }
  return v;
}

CudaArgV2 cuda_arg_memref(const TensorImpl& t,
                         int rank,
                         int index_width,
                         bool allow_empty_for_grid0) {
  if (t.device().type != kDLCUDA) {
    throw nb::type_error("_cuda_arg_memref: expected a CUDA tensor");
  }
  if (rank < 0) {
    throw nb::value_error("_cuda_arg_memref: rank must be >= 0");
  }
  if (index_width != 64) {
    throw nb::value_error("_cuda_arg_memref: only index_width=64 is supported");
  }
  if (static_cast<std::size_t>(rank) != t.sizes().size()) {
    throw nb::value_error("_cuda_arg_memref: rank does not match tensor sizes");
  }
  if (t.storage_offset() < 0) {
    throw nb::value_error("_cuda_arg_memref: storage_offset must be >= 0");
  }
  if (!t.storage()) {
    throw nb::value_error("_cuda_arg_memref: tensor has no storage");
  }
  if (t.numel() == 0 && !allow_empty_for_grid0) {
    throw nb::value_error(
        "_cuda_arg_memref: tensor is empty; set allow_empty_for_grid0=True to build a descriptor for grid==0 launches");
  }

  std::size_t sz = 24u + 16u * static_cast<std::size_t>(rank);
  if (sz > 4096u) {
    throw nb::value_error("_cuda_arg_memref: packed descriptor is too large");
  }

  void* base_ptr = t.storage()->data();
  std::uint64_t base_u64 = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(base_ptr));
  std::int64_t offset = static_cast<std::int64_t>(t.storage_offset());

  CudaArgV2 out;
  out.bytes.resize(sz);
  unsigned char* p = out.bytes.data();
  // Layout matches CuTeDSL's MLIR memref descriptor (ranked):
  // allocated (u64), aligned (u64), offset (i64), sizes[R] (i64), strides[R] (i64).
  std::memcpy(p + 0, &base_u64, sizeof(std::uint64_t));
  std::memcpy(p + 8, &base_u64, sizeof(std::uint64_t));
  std::memcpy(p + 16, &offset, sizeof(std::int64_t));

  // sizes
  for (int i = 0; i < rank; ++i) {
    std::int64_t s = static_cast<std::int64_t>(t.sizes()[static_cast<std::size_t>(i)]);
    std::memcpy(p + 24 + 8 * static_cast<std::size_t>(i), &s, sizeof(std::int64_t));
  }
  // strides
  const std::size_t strides_base = 24u + 8u * static_cast<std::size_t>(rank);
  for (int i = 0; i < rank; ++i) {
    std::int64_t st = static_cast<std::int64_t>(t.strides()[static_cast<std::size_t>(i)]);
    std::memcpy(p + strides_base + 8 * static_cast<std::size_t>(i), &st, sizeof(std::int64_t));
  }

  return out;
}

std::vector<std::size_t> parse_expected_param_sizes(nb::handle h, const char* api_name) {
  std::string seq_msg = std::string(api_name) + ": expected_param_sizes must be a sequence of ints";
  PyObject* seq = PySequence_Fast(h.ptr(), seq_msg.c_str());
  if (!seq) {
    throw nb::python_error();
  }
  const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
  if (n > 1024) {
    Py_DECREF(seq);
    std::string msg = api_name;
    msg += ": expected_param_sizes is too long";
    throw nb::value_error(msg.c_str());
  }

  std::vector<std::size_t> out;
  out.reserve(static_cast<std::size_t>(n));
  std::size_t sum = 0;

  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PySequence_Fast_GET_ITEM(seq, i); // borrowed
    if (!item) {
      Py_DECREF(seq);
      std::string msg = api_name;
      msg += ": expected_param_sizes[";
      msg += std::to_string(i);
      msg += "] is null";
      throw nb::type_error(msg.c_str());
    }
    if (PyBool_Check(item)) {
      Py_DECREF(seq);
      std::string msg = api_name;
      msg += ": expected_param_sizes[";
      msg += std::to_string(i);
      msg += "] must be an int, not bool";
      throw nb::type_error(msg.c_str());
    }

    // Accept Python ints, numpy scalar ints, IntEnum, etc via __index__.
    PyObject* idx_obj = PyNumber_Index(item); // new ref
    if (!idx_obj) {
      PyErr_Clear();
      Py_DECREF(seq);
      std::string msg = api_name;
      msg += ": expected_param_sizes[";
      msg += std::to_string(i);
      msg += "] must be an int";
      throw nb::type_error(msg.c_str());
    }
    if (PyBool_Check(idx_obj)) {
      Py_DECREF(idx_obj);
      Py_DECREF(seq);
      std::string msg = api_name;
      msg += ": expected_param_sizes[";
      msg += std::to_string(i);
      msg += "] must be an int, not bool";
      throw nb::type_error(msg.c_str());
    }

    long long v = PyLong_AsLongLong(idx_obj);
    Py_DECREF(idx_obj);
    if (PyErr_Occurred()) {
      PyErr_Clear();
      Py_DECREF(seq);
      std::string msg = api_name;
      msg += ": expected_param_sizes[";
      msg += std::to_string(i);
      msg += "] must fit in int64";
      throw nb::value_error(msg.c_str());
    }
    if (v < 1 || v > 4096) {
      Py_DECREF(seq);
      std::string msg = api_name;
      msg += ": expected_param_sizes[";
      msg += std::to_string(i);
      msg += "] must be in [1, 4096]";
      throw nb::value_error(msg.c_str());
    }
    sum += static_cast<std::size_t>(v);
    if (sum > 4096) {
      Py_DECREF(seq);
      std::string msg = api_name;
      msg += ": expected_param_sizes sum must be <= 4096";
      throw nb::value_error(msg.c_str());
    }
    out.push_back(static_cast<std::size_t>(v));
  }

  Py_DECREF(seq);
  return out;
}

struct PackedCudaLaunchArgsChecked {
  std::vector<std::vector<unsigned char>> storage;
  std::vector<void*> argv;
  std::vector<std::uint8_t> is_v2;
};

PackedCudaLaunchArgsChecked pack_cuda_launch_args_checked(const nb::list& args_py, const char* api_name) {
  PackedCudaLaunchArgsChecked out;
  out.storage.reserve(args_py.size());
  out.argv.reserve(args_py.size());
  out.is_v2.reserve(args_py.size());

  const bool unsafe_int_packing = unsafe_legacy_int_packing_enabled();

  for (nb::handle h : args_py) {
    if (nb::isinstance<TensorImpl>(h)) {
      const TensorImpl& t = nb::cast<const TensorImpl&>(h);
      auto dev = t.device();
      if (dev.type != kDLCUDA) throw nb::type_error("expected a CUDA tensor for pointer arg");
      void* ptr = t.data();
      out.storage.emplace_back(sizeof(void*));
      std::memcpy(out.storage.back().data(), &ptr, sizeof(void*));
      out.argv.push_back(out.storage.back().data());
      out.is_v2.push_back(0);
    } else if (nb::isinstance<CudaArgV2>(h)) {
      const CudaArgV2& arg = nb::cast<const CudaArgV2&>(h);
      out.storage.push_back(arg.bytes);
      out.argv.push_back(out.storage.back().data());
      out.is_v2.push_back(1);
    } else if (nb::isinstance<nb::bool_>(h)) {
      uint64_t v = nb::cast<bool>(h) ? 1ull : 0ull;
      out.storage.emplace_back(sizeof(uint64_t));
      std::memcpy(out.storage.back().data(), &v, sizeof(uint64_t));
      out.argv.push_back(out.storage.back().data());
      out.is_v2.push_back(0);
    } else if (nb::isinstance<nb::int_>(h)) {
      long long v64 = nb::cast<long long>(h);
      if (unsafe_int_packing && v64 >= std::numeric_limits<int32_t>::min() &&
          v64 <= std::numeric_limits<int32_t>::max()) {
        int32_t v = static_cast<int32_t>(v64);
        out.storage.emplace_back(sizeof(int32_t));
        std::memcpy(out.storage.back().data(), &v, sizeof(int32_t));
      } else {
        int64_t v = static_cast<int64_t>(v64);
        out.storage.emplace_back(sizeof(int64_t));
        std::memcpy(out.storage.back().data(), &v, sizeof(int64_t));
      }
      out.argv.push_back(out.storage.back().data());
      out.is_v2.push_back(0);
    } else if (nb::isinstance<nb::float_>(h)) {
      float v = nb::cast<float>(h);
      out.storage.emplace_back(sizeof(float));
      std::memcpy(out.storage.back().data(), &v, sizeof(float));
      out.argv.push_back(out.storage.back().data());
      out.is_v2.push_back(0);
    } else if (h.is_none()) {
      void* ptr = nullptr;
      out.storage.emplace_back(sizeof(void*));
      std::memcpy(out.storage.back().data(), &ptr, sizeof(void*));
      out.argv.push_back(out.storage.back().data());
      out.is_v2.push_back(0);
    } else {
      std::string msg = "unsupported argument type for ";
      msg += api_name;
      throw nb::type_error(msg.c_str());
    }
  }

  return out;
}

} // namespace

void bind_cuda_driver(nb::module_& m) {
  nb::class_<CudaArgV2>(m, "_CudaArgV2")
      .def("__repr__", [](const CudaArgV2& self) {
        return std::string("<_CudaArgV2 nbytes=") + std::to_string(self.bytes.size()) + ">";
      });

  m.def(
      "_cuda_arg_i32",
      [](nb::int_ x) {
        long long v64 = parse_py_int64_strict(x, "_cuda_arg_i32");
        if (v64 < std::numeric_limits<int32_t>::min() || v64 > std::numeric_limits<int32_t>::max()) {
          throw nb::value_error("_cuda_arg_i32: value out of range for i32");
        }
        int32_t v = static_cast<int32_t>(v64);
        return cuda_arg_from_scalar(v);
      },
      nb::arg("x"));

  m.def(
      "_cuda_arg_i64",
      [](nb::int_ x) {
        long long v64 = parse_py_int64_strict(x, "_cuda_arg_i64");
        int64_t v = static_cast<int64_t>(v64);
        return cuda_arg_from_scalar(v);
      },
      nb::arg("x"));

  m.def(
      "_cuda_arg_u8",
      [](nb::int_ x) {
        unsigned long long v64 = parse_py_uint64_strict(x, "_cuda_arg_u8");
        if (v64 > std::numeric_limits<std::uint8_t>::max()) {
          throw nb::value_error("_cuda_arg_u8: value out of range for u8");
        }
        std::uint8_t v = static_cast<std::uint8_t>(v64);
        return cuda_arg_from_scalar(v);
      },
      nb::arg("x"));

  m.def(
      "_cuda_arg_u16",
      [](nb::int_ x) {
        unsigned long long v64 = parse_py_uint64_strict(x, "_cuda_arg_u16");
        if (v64 > std::numeric_limits<std::uint16_t>::max()) {
          throw nb::value_error("_cuda_arg_u16: value out of range for u16");
        }
        std::uint16_t v = static_cast<std::uint16_t>(v64);
        return cuda_arg_from_scalar(v);
      },
      nb::arg("x"));

  m.def(
      "_cuda_arg_u32",
      [](nb::int_ x) {
        unsigned long long v64 = parse_py_uint64_strict(x, "_cuda_arg_u32");
        if (v64 > std::numeric_limits<std::uint32_t>::max()) {
          throw nb::value_error("_cuda_arg_u32: value out of range for u32");
        }
        std::uint32_t v = static_cast<std::uint32_t>(v64);
        return cuda_arg_from_scalar(v);
      },
      nb::arg("x"));

  m.def(
      "_cuda_arg_u64",
      [](nb::int_ x) {
        unsigned long long v64 = parse_py_uint64_strict(x, "_cuda_arg_u64");
        std::uint64_t v = static_cast<std::uint64_t>(v64);
        return cuda_arg_from_scalar(v);
      },
      nb::arg("x"));

  m.def(
      "_cuda_arg_f32",
      [](double x) {
        float v = static_cast<float>(x);
        return cuda_arg_from_scalar(v);
      },
      nb::arg("x"));

  m.def(
      "_cuda_arg_f64",
      [](double x) {
        double v = static_cast<double>(x);
        return cuda_arg_from_scalar(v);
      },
      nb::arg("x"));

  m.def(
      "_cuda_arg_device_ptr",
      [](nb::int_ ptr) {
        unsigned long long v64 = parse_py_uint64_strict(ptr, "_cuda_arg_device_ptr");
        std::uint64_t v = static_cast<std::uint64_t>(v64);
        return cuda_arg_from_scalar(v);
      },
      nb::arg("ptr"));

  m.def(
      "_cuda_arg_bytes",
      [](nb::handle data) { return cuda_arg_from_bytes_like(data, "_cuda_arg_bytes"); },
      nb::arg("data"));

  m.def(
      "_cuda_arg_memref",
      [](const TensorImpl& t, nb::int_ rank_py, nb::int_ index_width_py, bool allow_empty_for_grid0) {
        long long rank_ll = parse_py_int64_strict(rank_py, "_cuda_arg_memref: rank");
        if (rank_ll < std::numeric_limits<int>::min() || rank_ll > std::numeric_limits<int>::max()) {
          throw nb::value_error("_cuda_arg_memref: rank out of range");
        }
        long long iw_ll = parse_py_int64_strict(index_width_py, "_cuda_arg_memref: index_width");
        if (iw_ll < std::numeric_limits<int>::min() || iw_ll > std::numeric_limits<int>::max()) {
          throw nb::value_error("_cuda_arg_memref: index_width out of range");
        }
        return cuda_arg_memref(t,
                               static_cast<int>(rank_ll),
                               static_cast<int>(iw_ll),
                               allow_empty_for_grid0);
      },
      nb::arg("tensor"),
      nb::arg("rank"),
      nb::arg("index_width") = 64,
      nb::arg("allow_empty_for_grid0") = false);

#if VBT_WITH_CUDA
  // Compute capability for a device index (or current device when -1)
  m.def("_cuda_device_cc", [](int device_index) {
    int n = vbt::cuda::device_count();
    if (n <= 0) return std::make_tuple(0, 0);
    int dev = device_index;
    if (dev < 0) {
      // Best-effort: query current device via driver API; fallback to 0
      CUcontext ctx = nullptr;
      CUresult r = cuCtxGetCurrent(&ctx);
      if (r == CUDA_SUCCESS && ctx != nullptr) {
        CUdevice d;
        if (cuCtxGetDevice(&d) == CUDA_SUCCESS) {
          dev = (int)d;
        } else {
          dev = 0;
        }
      } else {
        dev = 0;
      }
    }
    if (dev >= n) dev = 0;
    CUdevice cu_dev;
    if (cuDeviceGet(&cu_dev, dev) != CUDA_SUCCESS) return std::make_tuple(0, 0);
    int major = 0, minor = 0;
    CUresult rr = cuDeviceComputeCapability(&major, &minor, cu_dev);
    if (rr != CUDA_SUCCESS) return std::make_tuple(0, 0);
    return std::make_tuple(major, minor);
  }, nb::arg("device_index") = -1);

  // Driver version (cuDriverGetVersion)
  m.def("_cuda_driver_version", []() {
    (void)cuInit(0);
    int v = 0;
    CUresult r = cuDriverGetVersion(&v);
    if (r != CUDA_SUCCESS) {
      const char* err = nullptr; (void)cuGetErrorString(r, &err);
      std::string msg = "cuDriverGetVersion failed: "; msg += (err ? err : "");
      throw std::runtime_error(msg);
    }
    return v;
  });

  // Function attribute setter (cuFuncSetAttribute)
  m.def("_cuda_func_set_attribute", [](nb::int_ func_handle, nb::int_ attribute, nb::int_ value) {
    (void)cuInit(0);

    unsigned long long fh =
        parse_py_uint64_strict(func_handle, "_cuda_func_set_attribute: func_handle");
    long long attr_ll =
        parse_py_int64_strict(attribute, "_cuda_func_set_attribute: attribute");
    long long val_ll =
        parse_py_int64_strict(value, "_cuda_func_set_attribute: value");

    if (attr_ll < std::numeric_limits<int>::min() ||
        attr_ll > std::numeric_limits<int>::max()) {
      throw nb::value_error("_cuda_func_set_attribute: attribute out of range");
    }
    if (val_ll < std::numeric_limits<int>::min() ||
        val_ll > std::numeric_limits<int>::max()) {
      throw nb::value_error("_cuda_func_set_attribute: value out of range");
    }

    CUfunction fn = (CUfunction)((uintptr_t)fh);
    CUresult r = cuFuncSetAttribute(fn, (CUfunction_attribute)(int)attr_ll, (int)val_ll);
    if (r != CUDA_SUCCESS) {
      const char* err = nullptr; (void)cuGetErrorString(r, &err);
      std::string msg = "cuFuncSetAttribute failed: "; msg += (err ? err : "");
      throw std::runtime_error(msg);
    }
  }, nb::arg("func_handle"), nb::arg("attribute"), nb::arg("value"));

  // Expose selected CUfunction_attribute enum values used by CuTeDSL.
  m.attr("_cuda_func_attribute_preferred_shared_memory_carveout") = nb::int_(static_cast<int>(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT));
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  m.attr("_cuda_func_attribute_non_portable_cluster_size_allowed") = nb::int_(static_cast<int>(CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED));
#else
  m.attr("_cuda_func_attribute_non_portable_cluster_size_allowed") = nb::int_(0);
#endif

  // Load a PTX image into a CUmodule and return an opaque handle as int
  m.def("_cuda_module_load_ptx", [](nb::bytes ptx_bytes) {
    // Ensure driver is initialized
    (void)cuInit(0);
    CUmodule mod = nullptr;
    // PTX data must remain alive during call; nanobind keeps bytes alive through this scope
    const char* data = ptx_bytes.c_str();
    CUresult r = cuModuleLoadData(&mod, (const void*)data);
    if (r != CUDA_SUCCESS) {
      const char* err = nullptr; (void)cuGetErrorString(r, &err);
      std::string msg = "cuModuleLoadData failed: "; msg += (err ? err : "");
      throw std::runtime_error(msg);
    }
    return (nb::int_)( (uintptr_t) mod );
  });

  // Get a CUfunction handle for a kernel name from a module handle
  m.def("_cuda_module_get_function", [](nb::int_ mod_handle, const std::string& func_name) {
    CUmodule mod = (CUmodule)( (uintptr_t) (long long) nb::cast<long long>(mod_handle) );
    CUfunction fn = nullptr;
    CUresult r = cuModuleGetFunction(&fn, mod, func_name.c_str());
    if (r != CUDA_SUCCESS) {
      const char* err = nullptr; (void)cuGetErrorString(r, &err);
      std::string msg = "cuModuleGetFunction failed: "; msg += (err ? err : "");
      throw std::runtime_error(msg);
    }
    return (nb::int_)( (uintptr_t) fn );
  });

  // Unload a CUmodule by handle
  m.def("_cuda_module_unload", [](nb::int_ mod_handle) {
    CUmodule mod = (CUmodule)( (uintptr_t) (long long) nb::cast<long long>(mod_handle) );
    if (!mod) return; // no-op
    CUresult r = cuModuleUnload(mod);
    if (r != CUDA_SUCCESS) {
      const char* err = nullptr; (void)cuGetErrorString(r, &err);
      std::string msg = "cuModuleUnload failed: "; msg += (err ? err : "");
      throw std::runtime_error(msg);
    }
  });

  // Launch a kernel given a CUfunction handle and arguments
  m.def("_cuda_launch", [](nb::int_ func_handle,
                            std::tuple<int,int,int> grid,
                            std::tuple<int,int,int> block,
                            int shmem_bytes,
                            nb::int_ stream_handle,
                            nb::list args_py) {
    CUfunction fn = (CUfunction)( (uintptr_t) (long long) nb::cast<long long>(func_handle) );
    CUstream stream = (CUstream)( (uintptr_t) (long long) nb::cast<long long>(stream_handle) );

    // Short-circuit on zero grid dims
    int gx = std::get<0>(grid), gy = std::get<1>(grid), gz = std::get<2>(grid);
    if (gx == 0 || gy == 0 || gz == 0) return;
    if (gx < 0 || gy < 0 || gz < 0) throw nb::value_error("grid dims must be >= 0");

    int bx = std::get<0>(block), by = std::get<1>(block), bz = std::get<2>(block);
    if (bx <= 0 || by <= 0 || bz <= 0) throw nb::value_error("block dims must be positive");
    if (shmem_bytes < 0) throw nb::value_error("shmem_bytes must be >= 0");

    // Build argv as array of pointers to argument storage
    auto packed = pack_cuda_launch_args(args_py, "_cuda_launch");

    CUresult r = cuLaunchKernel(
      fn,
      (unsigned int)gx, (unsigned int)gy, (unsigned int)gz,
      (unsigned int)bx, (unsigned int)by, (unsigned int)bz,
      (unsigned int)shmem_bytes,
      stream,
      packed.argv.empty() ? nullptr : packed.argv.data(),
      nullptr
    );
    if (r != CUDA_SUCCESS) {
      const char* err = nullptr; (void)cuGetErrorString(r, &err);
      std::string msg = "cuLaunchKernel failed: "; msg += (err ? err : "");
      throw std::runtime_error(msg);
    }
  }, nb::arg("func_handle"), nb::arg("grid"), nb::arg("block"), nb::arg("shmem_bytes"), nb::arg("stream_handle"), nb::arg("args"));

  // ABI-safe launch with explicit parameter size checks.
  m.def(
      "_cuda_launch_checked",
      [](nb::int_ func_handle,
         std::tuple<int, int, int> grid,
         std::tuple<int, int, int> block,
         int shmem_bytes,
         nb::int_ stream_handle,
         nb::list args_py,
         nb::handle expected_param_sizes,
         bool strict) {
        // Short-circuit on zero grid dims (parity with _cuda_launch).
        int gx = std::get<0>(grid), gy = std::get<1>(grid), gz = std::get<2>(grid);
        if (gx == 0 || gy == 0 || gz == 0) return;
        if (gx < 0 || gy < 0 || gz < 0) {
          throw nb::value_error("_cuda_launch_checked: grid dims must be >= 0");
        }

        int bx = std::get<0>(block), by = std::get<1>(block), bz = std::get<2>(block);
        if (bx <= 0 || by <= 0 || bz <= 0) {
          throw nb::value_error("_cuda_launch_checked: block dims must be positive");
        }
        if (shmem_bytes < 0) {
          throw nb::value_error("_cuda_launch_checked: shmem_bytes must be >= 0");
        }

        // Parse expected sizes after the grid==0 short-circuit.
        auto expected = parse_expected_param_sizes(expected_param_sizes, "_cuda_launch_checked");

        // Pack args into a flat list of kernel parameters.
        auto packed = pack_cuda_launch_args_checked(args_py, "_cuda_launch_checked");

        if (packed.argv.size() != expected.size()) {
          std::string msg = "_cuda_launch_checked: expected ";
          msg += std::to_string(expected.size());
          msg += " params but got ";
          msg += std::to_string(packed.argv.size());
          throw nb::value_error(msg.c_str());
        }

        for (std::size_t i = 0; i < expected.size(); ++i) {
          const std::size_t need = expected[i];
          const std::size_t got = packed.storage[i].size();
          if (got < need) {
            std::string msg = "_cuda_launch_checked: param ";
            msg += std::to_string(i);
            msg += " underflow (expected ";
            msg += std::to_string(need);
            msg += " bytes, got ";
            msg += std::to_string(got);
            msg += ")";
            throw nb::value_error(msg.c_str());
          }
          if (strict && packed.is_v2[i] != 0 && got != need) {
            std::string msg = "_cuda_launch_checked: param ";
            msg += std::to_string(i);
            msg += " size mismatch for _CudaArgV2 (expected ";
            msg += std::to_string(need);
            msg += " bytes, got ";
            msg += std::to_string(got);
            msg += ")";
            throw nb::value_error(msg.c_str());
          }
        }

        if (PyBool_Check(func_handle.ptr())) {
          throw nb::type_error("_cuda_launch_checked: func_handle must be an int, not bool");
        }
        if (PyBool_Check(stream_handle.ptr())) {
          throw nb::type_error("_cuda_launch_checked: stream_handle must be an int, not bool");
        }

        CUfunction fn = (CUfunction)((uintptr_t)(long long)nb::cast<long long>(func_handle));
        CUstream stream = (CUstream)((uintptr_t)(long long)nb::cast<long long>(stream_handle));

        CUresult r = cuLaunchKernel(
            fn,
            (unsigned int)gx,
            (unsigned int)gy,
            (unsigned int)gz,
            (unsigned int)bx,
            (unsigned int)by,
            (unsigned int)bz,
            (unsigned int)shmem_bytes,
            stream,
            packed.argv.empty() ? nullptr : packed.argv.data(),
            nullptr);
        if (r != CUDA_SUCCESS) {
          const char* err = nullptr;
          (void)cuGetErrorString(r, &err);
          std::string msg = "cuLaunchKernel failed: ";
          msg += (err ? err : "");
          throw std::runtime_error(msg);
        }
      },
      nb::arg("func_handle"),
      nb::arg("grid"),
      nb::arg("block"),
      nb::arg("shmem_bytes"),
      nb::arg("stream_handle"),
      nb::arg("args"),
      nb::arg("expected_param_sizes"),
      nb::arg("strict") = true);

  // Convenience wrapper: load PTX, get function, launch, unload
  m.def("_cuda_launch_ptx", [](nb::bytes ptx_bytes,
                                const std::string& func_name,
                                std::tuple<int,int,int> grid,
                                std::tuple<int,int,int> block,
                                int shmem_bytes,
                                nb::int_ stream_handle,
                                nb::list args_py) {
    (void)cuInit(0);
    CUmodule mod = nullptr;
    CUresult lr = cuModuleLoadData(&mod, (const void*)ptx_bytes.c_str());
    if (lr != CUDA_SUCCESS) {
      const char* err = nullptr; (void)cuGetErrorString(lr, &err);
      std::string msg = "cuModuleLoadData failed: "; msg += (err ? err : "");
      throw std::runtime_error(msg);
    }
    CUfunction fn = nullptr;
    CUresult fr = cuModuleGetFunction(&fn, mod, func_name.c_str());
    if (fr != CUDA_SUCCESS) {
      cuModuleUnload(mod);
      const char* err = nullptr; (void)cuGetErrorString(fr, &err);
      std::string msg = "cuModuleGetFunction failed: "; msg += (err ? err : "");
      throw std::runtime_error(msg);
    }

    // Pack args using the same legacy marshalling as _cuda_launch.
    CUstream stream = (CUstream)( (uintptr_t) (long long) nb::cast<long long>(stream_handle) );
    int gx = std::get<0>(grid), gy = std::get<1>(grid), gz = std::get<2>(grid);
    if (gx == 0 || gy == 0 || gz == 0) { (void)cuModuleUnload(mod); return; }
    if (gx < 0 || gy < 0 || gz < 0) { cuModuleUnload(mod); throw nb::value_error("grid dims must be >= 0"); }
    int bx = std::get<0>(block), by = std::get<1>(block), bz = std::get<2>(block);
    if (bx <= 0 || by <= 0 || bz <= 0) { cuModuleUnload(mod); throw nb::value_error("block dims must be positive"); }
    if (shmem_bytes < 0) { cuModuleUnload(mod); throw nb::value_error("shmem_bytes must be >= 0"); }

    PackedCudaLaunchArgs packed;
    try {
      packed = pack_cuda_launch_args(args_py, "_cuda_launch_ptx");
    } catch (...) {
      (void)cuModuleUnload(mod);
      throw;
    }

    CUresult r = cuLaunchKernel(
      fn,
      (unsigned int)gx, (unsigned int)gy, (unsigned int)gz,
      (unsigned int)bx, (unsigned int)by, (unsigned int)bz,
      (unsigned int)shmem_bytes,
      stream,
      packed.argv.empty() ? nullptr : packed.argv.data(),
      nullptr
    );
    if (r != CUDA_SUCCESS) {
      const char* err = nullptr; (void)cuGetErrorString(r, &err);
      std::string msg = "cuLaunchKernel failed: "; msg += (err ? err : "");
      cuModuleUnload(mod);
      throw std::runtime_error(msg);
    }

    CUresult ur = cuModuleUnload(mod);
    if (ur != CUDA_SUCCESS) {
      const char* err = nullptr; (void)cuGetErrorString(ur, &err);
      std::string msg = "cuModuleUnload failed: "; msg += (err ? err : "");
      throw std::runtime_error(msg);
    }
  }, nb::arg("ptx"), nb::arg("func_name"), nb::arg("grid"), nb::arg("block"), nb::arg("shmem_bytes"), nb::arg("stream_handle"), nb::arg("args"));

  // Helper: get current stream handle for a specific device
  m.def("_cuda_stream_handle_current_for_device", [](int device_index){
    int dev = device_index;
    if (dev < 0) {
      // current device via runtime API; if fails, default 0
      int d = 0;
      (void)d; // silence unused warning if compiled without runtime
      // Note: we avoid including cuda_runtime_api.h here; stream API doesn't require it
    }
    auto s = vbt::cuda::getCurrentStream((vbt::cuda::DeviceIndex) dev);
    return (nb::int_)( (uintptr_t) s.handle() );
  }, nb::arg("device_index") = 0);

#else
  // Stubs when CUDA is not built; keep attributes present with clear errors
  m.def("_cuda_device_cc", [](int){ return std::make_tuple(0, 0); });
  m.def("_cuda_driver_version", [](){ throw nb::type_error("CUDA is not available or not built for VibeTensor"); });
  m.def("_cuda_func_set_attribute", [](nb::int_, nb::int_, nb::int_){ throw nb::type_error("CUDA is not available or not built for VibeTensor"); });
  m.attr("_cuda_func_attribute_preferred_shared_memory_carveout") = nb::int_(0);
  m.attr("_cuda_func_attribute_non_portable_cluster_size_allowed") = nb::int_(0);
  m.def("_cuda_module_load_ptx", [](nb::bytes){ throw nb::type_error("CUDA is not available or not built for VibeTensor"); });
  m.def("_cuda_module_get_function", [](nb::int_, const std::string&){ throw nb::type_error("CUDA is not available or not built for VibeTensor"); });
  m.def("_cuda_module_unload", [](nb::int_){ /* no-op */ });
  m.def("_cuda_launch", [](nb::int_, std::tuple<int,int,int>, std::tuple<int,int,int>, int, nb::int_, nb::list){ throw nb::type_error("CUDA is not available or not built for VibeTensor"); });
  m.def(
      "_cuda_launch_checked",
      [](nb::int_,
         std::tuple<int, int, int>,
         std::tuple<int, int, int>,
         int,
         nb::int_,
         nb::list,
         nb::handle,
         bool) { throw nb::type_error("CUDA is not available or not built for VibeTensor"); },
      nb::arg("func_handle"),
      nb::arg("grid"),
      nb::arg("block"),
      nb::arg("shmem_bytes"),
      nb::arg("stream_handle"),
      nb::arg("args"),
      nb::arg("expected_param_sizes"),
      nb::arg("strict") = true);
  m.def("_cuda_launch_ptx", [](nb::bytes, const std::string&, std::tuple<int,int,int>, std::tuple<int,int,int>, int, nb::int_, nb::list){ throw nb::type_error("CUDA is not available or not built for VibeTensor"); });
  m.def("_cuda_stream_handle_current_for_device", [](int){ return (nb::int_)0; });
  m.def("_cuda_empty", [](const std::vector<int64_t>&, std::string, int){ throw nb::type_error("CUDA is not available or not built for VibeTensor"); });
#endif
}

} // namespace vbt_py
