// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>
#include <cstring>
#include <cstdlib>

#include <Python.h>

#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/core/complex.h"
#include "vbt/core/device.h"
#include "vbt/core/checked_math.h"
#include "vbt/cpu/storage.h"
#if VBT_WITH_CUDA
#include "vbt/cuda/storage.h"
#include "vbt/cuda/allocator.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/guard.h"
#include <cuda_runtime_api.h>
#endif

namespace nb = nanobind;

namespace vbt_py {

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;

static inline std::vector<int64_t> make_contig_strides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> st(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = (std::ptrdiff_t) sizes.size() - 1; i >= 0; --i) {
    st[(std::size_t) i] = acc;
    int64_t dim = sizes[(std::size_t) i];
    if (dim != 0) {
      int64_t tmp = 0;
      if (!vbt::core::checked_mul_i64(acc, dim, tmp)) throw nb::value_error("numel*itemsize overflow");
      acc = tmp;
    }
  }
  return st;
}

static inline std::size_t numel_bytes(const std::vector<int64_t>& sizes, ScalarType st) {
  int64_t n = 1;
  for (int64_t s : sizes) {
    if (s < 0) throw nb::value_error("size must be >= 0");
    if (s == 0) { n = 0; break; }
    int64_t tmp = 0;
    if (!vbt::core::checked_mul_i64(n, s, tmp)) throw nb::value_error("numel*itemsize overflow");
    n = tmp;
  }
  int64_t item_b = (int64_t) vbt::core::itemsize(st);
  int64_t total = 0;
  if (!vbt::core::checked_mul_i64(n, item_b, total)) throw nb::value_error("numel*itemsize overflow");
  return (std::size_t) total;
}

static inline ScalarType dtype_from_token(const std::string& token) {
  if (token == "float32") return ScalarType::Float32;
  if (token == "float64") return ScalarType::Float64;
  if (token == "complex64" || token == "cfloat") return ScalarType::Complex64;
  if (token == "complex128" || token == "cdouble") return ScalarType::Complex128;
  if (token == "int32") return ScalarType::Int32;
  if (token == "int64") return ScalarType::Int64;
  if (token == "bool") return ScalarType::Bool;
  if (token == "float16") return ScalarType::Float16;
#if VBT_HAS_DLPACK_BF16
  if (token == "bfloat16") return ScalarType::BFloat16;
#endif
  const char* msg =
#if VBT_HAS_DLPACK_BF16
    "unsupported dtype: expected one of {float32,float64,complex64,complex128,float16,bfloat16,int32,int64,bool}";
#else
    "unsupported dtype: expected one of {float32,float64,complex64,complex128,float16,int32,int64,bool}";
#endif
  throw nb::type_error(msg);
}

static constexpr const char* kErrComplexDisabled =
    "complex dtypes are disabled; set VBT_ENABLE_COMPLEX=1";

static inline bool complex_enabled_from_env() noexcept {
  const char* raw = std::getenv("VBT_ENABLE_COMPLEX");
  return raw && raw[0] == '1' && raw[1] == '\0';
}

static inline void throw_if_complex_disabled(ScalarType st) {
  if ((st == ScalarType::Complex64 || st == ScalarType::Complex128) &&
      !complex_enabled_from_env()) {
    throw nb::type_error(kErrComplexDisabled);
  }
}


static inline void get_complex_parts(nb::handle value, double* out_re, double* out_im) {
  if (!out_re || !out_im) return;

  PyObject* obj = value.ptr();
  if (!obj) {
    *out_re = 0.0;
    *out_im = 0.0;
    return;
  }

  Py_complex c = PyComplex_AsCComplex(obj);
  if (PyErr_Occurred()) throw nb::python_error();
  *out_re = c.real;
  *out_im = c.imag;
}

static inline void fill_typed(void* data, int64_t n, ScalarType st, nb::object value) {
  if (n <= 0) return;
  switch (st) {
    case ScalarType::Float32: {
      float v = nb::cast<float>(value);
      float* p = reinterpret_cast<float*>(data);
      for (int64_t i = 0; i < n; ++i) p[i] = v;
      return;
    }
    case ScalarType::Float64: {
      double v = nb::cast<double>(value);
      double* p = reinterpret_cast<double*>(data);
      for (int64_t i = 0; i < n; ++i) p[i] = v;
      return;
    }
    case ScalarType::Int32: {
      int32_t v = 0; try { v = nb::cast<int32_t>(value); } catch (...) { v = (int32_t) nb::cast<double>(value); }
      int32_t* p = reinterpret_cast<int32_t*>(data);
      for (int64_t i = 0; i < n; ++i) p[i] = v;
      return;
    }
    case ScalarType::Int64: {
      int64_t v = 0; try { v = nb::cast<int64_t>(value); } catch (...) { v = (int64_t) nb::cast<double>(value); }
      int64_t* p = reinterpret_cast<int64_t*>(data);
      for (int64_t i = 0; i < n; ++i) p[i] = v;
      return;
    }
    case ScalarType::Bool: {
      bool v = false;
      try {
        v = nb::cast<bool>(value);
      } catch (...) {
        // Fallback: accept numeric scalars and interpret zero as False.
        double dv = nb::cast<double>(value);
        v = (dv != 0.0);
      }
      uint8_t* p = reinterpret_cast<uint8_t*>(data);
      for (int64_t i = 0; i < n; ++i) p[i] = v ? 1u : 0u;
      return;
    }
    case ScalarType::Complex64: {
      double re = 0.0;
      double im = 0.0;
      if (nb::isinstance<nb::bool_>(value) || nb::isinstance<nb::int_>(value) ||
          nb::isinstance<nb::float_>(value)) {
        re = nb::cast<double>(value);
      } else {
        get_complex_parts(value, &re, &im);
      }
      vbt::core::Complex<float> v{static_cast<float>(re), static_cast<float>(im)};
      auto* p = reinterpret_cast<vbt::core::Complex<float>*>(data);
      for (int64_t i = 0; i < n; ++i) p[i] = v;
      return;
    }
    case ScalarType::Complex128: {
      double re = 0.0;
      double im = 0.0;
      if (nb::isinstance<nb::bool_>(value) || nb::isinstance<nb::int_>(value) ||
          nb::isinstance<nb::float_>(value)) {
        re = nb::cast<double>(value);
      } else {
        get_complex_parts(value, &re, &im);
      }
      vbt::core::Complex<double> v{re, im};
      auto* p = reinterpret_cast<vbt::core::Complex<double>*>(data);
      for (int64_t i = 0; i < n; ++i) p[i] = v;
      return;
    }
    case ScalarType::Float16:
    case ScalarType::BFloat16:
      throw nb::type_error(
          "unsupported dtype: expected one of {float32,float64,complex64,complex128,int32,int64,bool}");
    case ScalarType::Undefined:
      break;
  }
  throw nb::type_error("unsupported dtype");
}

void bind_factories(nb::module_& m) {
  m.def("_cpu_empty", [](const std::vector<int64_t>& sizes, std::string dtype_token) {
    ScalarType st = dtype_from_token(dtype_token);
    throw_if_complex_disabled(st);
    std::size_t nbytes = numel_bytes(sizes, st);
    auto storage = vbt::cpu::new_cpu_storage(nbytes, /*pinned=*/false);
    auto strides = make_contig_strides(sizes);
    return TensorImpl(storage, sizes, strides, 0, st, Device::cpu());
  });

  m.def("_cpu_zeros", [](const std::vector<int64_t>& sizes, std::string dtype_token) {
    ScalarType st = dtype_from_token(dtype_token);
    throw_if_complex_disabled(st);
    std::size_t nbytes = numel_bytes(sizes, st);
    auto storage = vbt::cpu::new_cpu_storage(nbytes, /*pinned=*/false);
    if (nbytes > 0) std::memset(storage->data(), 0, nbytes);
    auto strides = make_contig_strides(sizes);
    return TensorImpl(storage, sizes, strides, 0, st, Device::cpu());
  });

  m.def("_cpu_full", [](const std::vector<int64_t>& sizes, std::string dtype_token, nb::object fill) {
    ScalarType st = dtype_from_token(dtype_token);
    throw_if_complex_disabled(st);
    if (!nb::isinstance<nb::bool_>(fill) && !nb::isinstance<nb::int_>(fill) &&
        !nb::isinstance<nb::float_>(fill) && !PyComplex_Check(fill.ptr())) {
      throw nb::type_error("fill_value must be a scalar");
    }
    std::size_t nbytes = numel_bytes(sizes, st);
    auto storage = vbt::cpu::new_cpu_storage(nbytes, /*pinned=*/false);
    int64_t n = 0;
    if (vbt::core::itemsize(st) > 0) {
      int64_t acc = 1;
      for (int64_t s : sizes) {
        if (s == 0) { acc = 0; break; }
        int64_t tmp = 0;
        if (!vbt::core::checked_mul_i64(acc, s, tmp)) throw nb::value_error("numel*itemsize overflow");
        acc = tmp;
      }
      n = acc;
    }
    fill_typed(storage->data(), n, st, fill);
    auto strides = make_contig_strides(sizes);
    return TensorImpl(storage, sizes, strides, 0, st, Device::cpu());
  });

  // CPU copy from NumPy (copy-only path)
  m.def("_cpu_from_numpy_copy", [](const nb::ndarray<nb::ro, nb::c_contig>& arr, std::string dtype_token){
    if (!arr.is_valid()) throw nb::value_error("expected a valid NumPy ndarray");
    ScalarType st = dtype_from_token(dtype_token);
    throw_if_complex_disabled(st);
    std::vector<int64_t> sizes(arr.ndim());
    for (size_t i = 0; i < arr.ndim(); ++i) sizes[i] = (int64_t) arr.shape(i);
    std::size_t expected_nbytes = numel_bytes(sizes, st);
    std::size_t got_nbytes = (std::size_t) arr.nbytes();
    if (expected_nbytes != got_nbytes) {
      throw nb::value_error("_cpu_from_numpy_copy: dtype/shape byte size mismatch");
    }

    // Ensure the ndarray dtype matches the requested dtype token. This prevents
    // silent bit reinterpretation when the dtype differs but has the same
    // itemsize (e.g., int64 <-> float64).
    {
      const auto got_dt = arr.dtype();
      const DLDataType expected_dt = vbt::core::to_dlpack_dtype(st);
      if (got_dt.code != expected_dt.code ||
          got_dt.bits != expected_dt.bits ||
          got_dt.lanes != expected_dt.lanes) {
        throw nb::value_error("_cpu_from_numpy_copy: dtype token does not match ndarray dtype");
      }
    }

    auto storage = vbt::cpu::new_cpu_storage(expected_nbytes, /*pinned=*/false);
    if (expected_nbytes > 0) {
      std::memcpy(storage->data(), arr.data(), expected_nbytes);
    }
    auto strides = make_contig_strides(sizes);
    return TensorImpl(storage, sizes, strides, 0, st, Device::cpu());
  }, nb::arg("array").noconvert(), nb::arg("dtype"));

  // CPU copy from flat Python sequence (no NumPy required)
  m.def("_cpu_from_sequence_copy", [](const nb::sequence& values,
                                     const std::vector<int64_t>& sizes,
                                     std::string dtype_token){
    ScalarType st = dtype_from_token(dtype_token);
    throw_if_complex_disabled(st);

    int64_t n = 1;
    for (int64_t s : sizes) {
      if (s < 0) throw nb::value_error("size must be >= 0");
      if (s == 0) { n = 0; break; }
      int64_t tmp = 0;
      if (!vbt::core::checked_mul_i64(n, s, tmp)) throw nb::value_error("numel*itemsize overflow");
      n = tmp;
    }
    if (static_cast<int64_t>(nb::len(values)) != n) {
      throw nb::value_error("sequence length mismatch");
    }

    std::size_t nbytes = numel_bytes(sizes, st);
    auto storage = vbt::cpu::new_cpu_storage(nbytes, /*pinned=*/false);

    if (n > 0) {
      switch (st) {
        case ScalarType::Float32: {
          float* p = reinterpret_cast<float*>(storage->data());
          int64_t i = 0;
          for (nb::handle h : values) {
            p[i++] = nb::cast<float>(h);
          }
          break;
        }
        case ScalarType::Float64: {
          double* p = reinterpret_cast<double*>(storage->data());
          int64_t i = 0;
          for (nb::handle h : values) {
            p[i++] = nb::cast<double>(h);
          }
          break;
        }
        case ScalarType::Int32: {
          int32_t* p = reinterpret_cast<int32_t*>(storage->data());
          int64_t i = 0;
          for (nb::handle h : values) {
            int32_t v = 0;
            try { v = nb::cast<int32_t>(h); } catch (...) { v = static_cast<int32_t>(nb::cast<double>(h)); }
            p[i++] = v;
          }
          break;
        }
        case ScalarType::Int64: {
          int64_t* p = reinterpret_cast<int64_t*>(storage->data());
          int64_t i = 0;
          for (nb::handle h : values) {
            int64_t v = 0;
            try { v = nb::cast<int64_t>(h); } catch (...) { v = static_cast<int64_t>(nb::cast<double>(h)); }
            p[i++] = v;
          }
          break;
        }
        case ScalarType::Bool: {
          uint8_t* p = reinterpret_cast<uint8_t*>(storage->data());
          int64_t i = 0;
          for (nb::handle h : values) {
            bool v = false;
            try {
              v = nb::cast<bool>(h);
            } catch (...) {
              double dv = nb::cast<double>(h);
              v = (dv != 0.0);
            }
            p[i++] = v ? 1u : 0u;
          }
          break;
        }
        case ScalarType::Complex64: {
          auto* p = reinterpret_cast<vbt::core::Complex<float>*>(storage->data());
          int64_t i = 0;
          for (nb::handle h : values) {
            double re = 0.0;
            double im = 0.0;
            get_complex_parts(h, &re, &im);
            p[i++] = vbt::core::Complex<float>{static_cast<float>(re),
                                               static_cast<float>(im)};
          }
          break;
        }
        case ScalarType::Complex128: {
          auto* p = reinterpret_cast<vbt::core::Complex<double>*>(storage->data());
          int64_t i = 0;
          for (nb::handle h : values) {
            double re = 0.0;
            double im = 0.0;
            get_complex_parts(h, &re, &im);
            p[i++] = vbt::core::Complex<double>{re, im};
          }
          break;
        }
        case ScalarType::Float16:
        case ScalarType::BFloat16:
          throw nb::type_error(
              "unsupported dtype: expected one of {float32,float64,complex64,complex128,int32,int64,bool}");
        case ScalarType::Undefined:
          throw nb::type_error("unsupported dtype");
      }
    }

    auto strides = make_contig_strides(sizes);
    return TensorImpl(storage, sizes, strides, 0, st, Device::cpu());
  }, nb::arg("values"), nb::arg("sizes"), nb::arg("dtype"));

#if VBT_WITH_CUDA
  // CUDA factories
  m.def("_cuda_empty", [](const std::vector<int64_t>& sizes, std::string dtype_token, nb::object device_index_obj){
    ScalarType st = dtype_from_token(dtype_token);
    throw_if_complex_disabled(st);
    int dev = -1;
    if (!device_index_obj.is_none()) dev = nb::cast<int>(device_index_obj);
    if (dev < 0) { cudaGetDevice(&dev); }
    std::size_t nbytes = numel_bytes(sizes, st);
    auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
    auto strides = make_contig_strides(sizes);
    return TensorImpl(storage, sizes, strides, 0, st, Device::cuda(dev));
  }, nb::arg("sizes"), nb::arg("dtype"), nb::arg("device").none(true)=nb::none());

  m.def("_cuda_zeros", [](const std::vector<int64_t>& sizes, std::string dtype_token, nb::object device_index_obj){
    ScalarType st = dtype_from_token(dtype_token);
    throw_if_complex_disabled(st);
    int dev = -1; if (!device_index_obj.is_none()) dev = nb::cast<int>(device_index_obj); if (dev < 0) { cudaGetDevice(&dev); }
    std::size_t nbytes = numel_bytes(sizes, st);
    auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
    {
      vbt::cuda::DeviceGuard guard((vbt::cuda::DeviceIndex)dev);
      if (nbytes > 0) {
        // zero device memory
        cudaError_t stc = cudaMemset(storage->data(), 0, nbytes);
        if (stc != cudaSuccess) {
          const char* msg = cudaGetErrorString(stc);
          std::string m = std::string("_cuda_zeros: cudaMemset failed: ") + (msg?msg:"");
          throw std::runtime_error(m.c_str());
        }
      }
    }
    auto strides = make_contig_strides(sizes);
    return TensorImpl(storage, sizes, strides, 0, st, Device::cuda(dev));
  }, nb::arg("sizes"), nb::arg("dtype"), nb::arg("device").none(true)=nb::none());

  m.def("_cuda_full", [](const std::vector<int64_t>& sizes, std::string dtype_token, nb::object fill, nb::object device_index_obj){
    ScalarType st = dtype_from_token(dtype_token);
    throw_if_complex_disabled(st);
    if (!nb::isinstance<nb::bool_>(fill) && !nb::isinstance<nb::int_>(fill) &&
        !nb::isinstance<nb::float_>(fill) && !PyComplex_Check(fill.ptr())) {
      throw nb::type_error("fill_value must be a scalar");
    }
    int dev = -1; if (!device_index_obj.is_none()) dev = nb::cast<int>(device_index_obj); if (dev < 0) { cudaGetDevice(&dev); }
    std::size_t nbytes = numel_bytes(sizes, st);
    auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
    // Fill via temporary host buffer then H2D; adequate for tests
    int64_t n = 0; if (vbt::core::itemsize(st) > 0) {
      int64_t acc = 1; for (int64_t s : sizes) { if (s == 0) { acc = 0; break; } int64_t tmp=0; if (!vbt::core::checked_mul_i64(acc, s, tmp)) throw nb::value_error("numel*itemsize overflow"); acc = tmp; }
      n = acc;
    }
    if (nbytes > 0) {
      std::vector<unsigned char> host(nbytes);
      fill_typed(host.data(), n, st, fill);
      vbt::cuda::DeviceGuard guard((vbt::cuda::DeviceIndex)dev);
      auto stream = vbt::cuda::getCurrentStream((vbt::cuda::DeviceIndex)dev);
      vbt::cuda::Allocator& alloc = vbt::cuda::Allocator::get((vbt::cuda::DeviceIndex)dev);
      auto stc = alloc.memcpyAsync(storage->data(), dev, host.data(), -1, nbytes, stream, /*p2p_enabled=*/false);
      if (stc != cudaSuccess) {
        const char* msg = cudaGetErrorString(stc);
        std::string m = std::string("_cuda_full: memcpyAsync failed: ") + (msg?msg:"");
        throw std::runtime_error(m.c_str());
      }
      cudaError_t sync_st = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.handle()));
      if (sync_st != cudaSuccess) {
        const char* msg = cudaGetErrorString(sync_st);
        std::string m = std::string("_cuda_full: cudaStreamSynchronize failed: ") + (msg?msg:"");
        throw std::runtime_error(m.c_str());
      }
    }
    auto strides = make_contig_strides(sizes);
    return TensorImpl(storage, sizes, strides, 0, st, Device::cuda(dev));
  }, nb::arg("sizes"), nb::arg("dtype"), nb::arg("fill"), nb::arg("device").none(true)=nb::none());
#endif
}

} // namespace vbt_py
