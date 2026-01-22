// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/nb_defs.h>
#include <nanobind/ndarray.h>

#include <Python.h>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <utility>

#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/overlap.h"
#include "vbt/core/write_guard.h"
#include "vbt/interop/dlpack.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/core/view_ops.h"
#include "vbt/core/tensor_ops.h"
#include "vbt/core/tensor_iter.h"
#include "vbt/core/error_text.h"
#include "vbt/core/strided_loop.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/checked_math.h"
#include "vbt/core/indexing.h"
#include "vbt/core/indexing/index_errors.h"
#if VBT_WITH_AUTOGRAD
#include "vbt/autograd/meta.h"
#include "vbt/autograd/wrapper.h"
#include "vbt/autograd/saved_variable.h"
#endif

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#include "vbt/cuda/stream.h"
#include "vbt/cuda/storage.h"
extern "C" {
    vbt::core::TensorImpl vbt_cuda_sum_impl(const vbt::core::TensorImpl&, std::vector<int64_t>, bool);
    vbt::core::TensorImpl vbt_cuda_mean_impl(const vbt::core::TensorImpl&, std::vector<int64_t>, bool);
    vbt::core::TensorImpl vbt_cuda_prod_impl(const vbt::core::TensorImpl&, std::vector<int64_t>, bool);
    vbt::core::TensorImpl vbt_cuda_min_impl(const vbt::core::TensorImpl&, std::vector<int64_t>, bool);
    vbt::core::TensorImpl vbt_cuda_max_impl(const vbt::core::TensorImpl&, std::vector<int64_t>, bool);
}
#endif

namespace nb = nanobind;

namespace vbt_py {

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::indexing::IndexKind;
using vbt::core::indexing::IndexSpec;
using vbt::core::indexing::TensorIndex;
using vbt::core::indexing::has_any_advanced;
using vbt::core::indexing::is_advanced_kind;

static inline nb::tuple to_tuple(const std::vector<int64_t>& v) {
  nb::list out;
  for (auto x : v) out.append(nb::int_(x));
  return nb::tuple(out);
}

static inline const char* dtype_name(vbt::core::ScalarType t) {
  using vbt::core::ScalarType;
  switch (t) {
    case ScalarType::Bool: return "bool";
    case ScalarType::Int32: return "int32";
    case ScalarType::Int64: return "int64";
    case ScalarType::Float32: return "float32";
    case ScalarType::Float16: return "float16";
    case ScalarType::BFloat16: return "bfloat16";
    case ScalarType::Float64: return "float64";
    case ScalarType::Complex64: return "complex64";
    case ScalarType::Complex128: return "complex128";
    case ScalarType::Undefined: return "undefined";
  }
  return "unknown";
}

static constexpr const char* kErrComplexDisabled =
    "complex dtypes are disabled; set VBT_ENABLE_COMPLEX=1";

inline bool complex_enabled_from_env() noexcept {
  const char* raw = std::getenv("VBT_ENABLE_COMPLEX");
  return raw && raw[0] == '1' && raw[1] == '\0';
}

static inline vbt::core::MemoryFormat parse_memory_format(
    const nb::object& obj) {
  using vbt::core::MemoryFormat;

  if (obj.is_none()) {
    return MemoryFormat::Contiguous;
  }

  if (nb::isinstance<nb::str>(obj)) {
    std::string s = nb::cast<std::string>(obj);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });

    if (s == "contiguous") return MemoryFormat::Contiguous;
    if (s == "channels_last") return MemoryFormat::ChannelsLast;
    if (s == "preserve") return MemoryFormat::Preserve;

    throw nb::value_error(
        "memory_format: expected one of {'contiguous','channels_last','preserve'}");
  }

  // Reject accidental bool/int usage; only accept None, str, or MemoryFormat.
  if (PyBool_Check(obj.ptr()) || PyLong_Check(obj.ptr())) {
    throw nb::type_error("memory_format must be None, str, or MemoryFormat");
  }

  try {
    return nb::cast<MemoryFormat>(obj);
  } catch (const std::exception&) {
  }

  throw nb::type_error("memory_format must be None, str, or MemoryFormat");
}

static inline bool is_contiguous_for_format(
    const TensorImpl& t,
    vbt::core::MemoryFormat fmt) noexcept {
  using vbt::core::MemoryFormat;
  switch (fmt) {
    case MemoryFormat::Contiguous:
      return t.is_contiguous();
    case MemoryFormat::ChannelsLast:
      return t.is_channels_last();
    case MemoryFormat::Preserve:
      return is_contiguous_for_format(t, t.suggest_memory_format());
  }
  return t.is_contiguous();
}

// Forward-declared helper to produce a DLPack capsule for a tensor
static inline nb::capsule make_dlpack_capsule(const vbt::core::TensorImpl& t) {
  auto up = vbt::interop::to_dlpack(t);
  auto* raw = up.release(); // transfer to capsule; deleter is mt->deleter
  // Name must be "dltensor" per protocol
  return nb::capsule(raw, "dltensor", [](void* /*p*/) noexcept {
    // no-op; consumer must invoke mt->deleter exactly once per DLPack protocol
  });
}

static inline std::string format_sizes(const std::vector<int64_t>& sizes) {
  std::string s;
  s.push_back('(');
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(sizes[i]);
  }
  s.push_back(')');
  return s;
}

// Normalize an arbitrary Python index object into an outer tuple.
// - If index is already a tuple, returns a new reference to it.
// - Otherwise, returns a 1-element tuple (index,).
static PyObject* wrap_index_to_tuple(PyObject* index) {
  if (PyTuple_Check(index)) {
    Py_INCREF(index);
    return index;
  }
  PyObject* tup = PyTuple_New(1);
  if (!tup) {
    return nullptr;  // Propagate Python error
  }
  Py_INCREF(index);
  PyTuple_SET_ITEM(tup, 0, index);  // steals reference to index
  return tup;
}

struct ParsedDim {
  bool has_dim{false};
  std::int64_t dim{0};
};

static ParsedDim parse_dim_arg(const nb::object& dim_obj,
                               const char* opname) {
  ParsedDim out;
  if (dim_obj.is_none()) {
    return out;
  }
  if (nb::isinstance<nb::int_>(dim_obj)) {
    out.has_dim = true;
    out.dim = nb::cast<std::int64_t>(dim_obj);
    return out;
  }
  std::string msg = std::string(opname) + ": dim must be int or None";
  PyErr_SetString(PyExc_TypeError, msg.c_str());
  throw nb::python_error();
}

static vbt::core::TensorImpl make_contiguous_out_cpu(
    const vbt::core::TensorImpl& like,
    const std::vector<int64_t>& sizes,
    vbt::core::ScalarType dtype) {
  using vbt::core::Storage;
  using vbt::core::DataPtr;
  using vbt::core::itemsize;
  using vbt::core::checked_mul_i64;

  auto dev = like.device();
  if (dev.type != kDLCPU) {
    throw std::invalid_argument("reduction: expected CPU tensor");
  }

  const std::size_t item_b = static_cast<std::size_t>(itemsize(dtype));

  std::int64_t ne = 1;
  if (!sizes.empty()) {
    for (std::int64_t s : sizes) {
      if (s == 0) {
        ne = 0;
        break;
      }
      std::int64_t tmp = 0;
      if (!checked_mul_i64(ne, s, tmp)) {
        ne = 0;
        break;
      }
      ne = tmp;
    }
  }

  const std::size_t nbytes = static_cast<std::size_t>(ne) * item_b;
  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<int64_t> strides(sizes.size(), 0);
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  return vbt::core::TensorImpl(storage, sizes, strides,
                               /*storage_offset=*/0, dtype, dev);
}

static void fill_tensor_constant_cpu(vbt::core::TensorImpl& t,
                                     double value) {
  using vbt::core::ScalarType;
  const ScalarType st = t.dtype();
  const std::int64_t n = t.numel();
  if (n <= 0) return;

  if (!t.is_contiguous()) {
    vbt::core::TensorImpl tmp = vbt::core::clone_cpu(t);
    fill_tensor_constant_cpu(tmp, value);
    t = std::move(tmp);
    return;
  }

  if (st == ScalarType::Float32) {
    float v = static_cast<float>(value);
    auto* p = static_cast<float*>(t.data());
    for (std::int64_t i = 0; i < n; ++i) p[i] = v;
  } else if (st == ScalarType::Int64) {
    auto* p = static_cast<long long*>(t.data());
    long long v = static_cast<long long>(value);
    for (std::int64_t i = 0; i < n; ++i) p[i] = v;
  }
}

static void reduce_sum_f32_loop(char** data,
                                const std::int64_t* strides,
                                std::int64_t size,
                                void* /*ctx*/) {
  char* out_base = data[0];
  char* in_base  = data[1];
  const std::int64_t in_stride = strides[1];
  auto* out = reinterpret_cast<float*>(out_base);
  float acc = 0.0f;
  for (std::int64_t i = 0; i < size; ++i) {
    const auto* pi =
        reinterpret_cast<const float*>(in_base + i * in_stride);
    acc += *pi;
  }
  *out = acc;
}

static void reduce_sum_i64_loop(char** data,
                                const std::int64_t* strides,
                                std::int64_t size,
                                void* /*ctx*/) {
  char* out_base = data[0];
  char* in_base  = data[1];
  const std::int64_t in_stride = strides[1];
  auto* out = reinterpret_cast<long long*>(out_base);
  long long acc = 0;
  for (std::int64_t i = 0; i < size; ++i) {
    const auto* pi =
        reinterpret_cast<const long long*>(in_base + i * in_stride);
    acc += *pi;
  }
  *out = acc;
}

static void reduce_mean_f32_loop(char** data,
                                 const std::int64_t* strides,
                                 std::int64_t size,
                                 void* /*ctx*/) {
  char* out_base = data[0];
  char* in_base  = data[1];
  const std::int64_t in_stride = strides[1];
  auto* out = reinterpret_cast<float*>(out_base);
  float acc = 0.0f;
  for (std::int64_t i = 0; i < size; ++i) {
    const auto* pi =
        reinterpret_cast<const float*>(in_base + i * in_stride);
    acc += *pi;
  }
  *out = acc / static_cast<float>(size > 0 ? size : 1);
}

static void reduce_amin_f32_loop(char** data,
                                 const std::int64_t* strides,
                                 std::int64_t size,
                                 void* /*ctx*/) {
  char* out_base = data[0];
  char* in_base  = data[1];
  const std::int64_t in_stride = strides[1];
  auto* out = reinterpret_cast<float*>(out_base);
  const auto* first =
      reinterpret_cast<const float*>(in_base);
  float best = *first;
  for (std::int64_t i = 1; i < size; ++i) {
    const auto* pi =
        reinterpret_cast<const float*>(in_base + i * in_stride);
    float v = *pi;
    if (v < best) best = v;
  }
  *out = best;
}

static void reduce_amax_f32_loop(char** data,
                                 const std::int64_t* strides,
                                 std::int64_t size,
                                 void* /*ctx*/) {
  char* out_base = data[0];
  char* in_base  = data[1];
  const std::int64_t in_stride = strides[1];
  auto* out = reinterpret_cast<float*>(out_base);
  const auto* first =
      reinterpret_cast<const float*>(in_base);
  float best = *first;
  for (std::int64_t i = 1; i < size; ++i) {
    const auto* pi =
        reinterpret_cast<const float*>(in_base + i * in_stride);
    float v = *pi;
    if (v > best) best = v;
  }
  *out = best;
}

static void reduce_amin_i64_loop(char** data,
                                 const std::int64_t* strides,
                                 std::int64_t size,
                                 void* /*ctx*/) {
  char* out_base = data[0];
  char* in_base  = data[1];
  const std::int64_t in_stride = strides[1];
  auto* out = reinterpret_cast<long long*>(out_base);
  const auto* first =
      reinterpret_cast<const long long*>(in_base);
  long long best = *first;
  for (std::int64_t i = 1; i < size; ++i) {
    const auto* pi =
        reinterpret_cast<const long long*>(in_base + i * in_stride);
    long long v = *pi;
    if (v < best) best = v;
  }
  *out = best;
}

static void reduce_amax_i64_loop(char** data,
                                 const std::int64_t* strides,
                                 std::int64_t size,
                                 void* /*ctx*/) {
  char* out_base = data[0];
  char* in_base  = data[1];
  const std::int64_t in_stride = strides[1];
  auto* out = reinterpret_cast<long long*>(out_base);
  const auto* first =
      reinterpret_cast<const long long*>(in_base);
  long long best = *first;
  for (std::int64_t i = 1; i < size; ++i) {
    const auto* pi =
        reinterpret_cast<const long long*>(in_base + i * in_stride);
    long long v = *pi;
    if (v > best) best = v;
  }
  *out = best;
}

enum class ReduceKind { Sum, Mean, Amin, Amax };

static vbt::core::TensorImpl reduce_value_cpu(
    const vbt::core::TensorImpl& self,
    const nb::object& dim_obj,
    bool keepdim,
    ReduceKind kind,
    const char* opname) {
  using vbt::core::ScalarType;
  using vbt::core::TensorIter;

  if (self.device().type != kDLCPU) {
    std::string msg = std::string(opname) + ": CPU tensors only for reductions";
    PyErr_SetString(PyExc_ValueError, msg.c_str());
    throw nb::python_error();
  }

  const ScalarType st = self.dtype();
  const bool allow_int64 = (kind != ReduceKind::Mean);
  if (!((st == ScalarType::Float32) ||
        (st == ScalarType::Int64 && allow_int64))) {
    if (kind == ReduceKind::Mean) {
      PyErr_SetString(PyExc_ValueError, "mean: expected dtype=float32");
    } else {
      std::string msg = std::string(opname) + ": unsupported dtype";
      PyErr_SetString(PyExc_ValueError, msg.c_str());
    }
    throw nb::python_error();
  }

  ParsedDim pd = parse_dim_arg(dim_obj, opname);
  const auto& sizes = self.sizes();
  const std::int64_t R = static_cast<std::int64_t>(sizes.size());

  // dim=None path
  if (!pd.has_dim) {
    if (keepdim) {
      std::string msg = std::string(opname) +
                        ": dim=None with keepdim=True is not supported";
      PyErr_SetString(PyExc_ValueError, msg.c_str());
      throw nb::python_error();
    }

    if (R == 0) {
      // Scalar: identity for value-only reductions.
      return vbt::core::clone_cpu(self);
    }

    vbt::core::TensorImpl base = self;
    if (!self.is_contiguous()) {
      base = vbt::core::clone_cpu(self);
    }
    const std::int64_t total = base.numel();

    std::vector<int64_t> out_sizes;  // scalar
    vbt::core::TensorImpl out = make_contiguous_out_cpu(base, out_sizes, st);

    if (total == 0) {
      if (kind == ReduceKind::Sum) {
        fill_tensor_constant_cpu(out, 0.0);
        return out;
      }
      if (kind == ReduceKind::Mean) {
        // Fill with NaN for float32
        if (st == ScalarType::Float32 && out.numel() > 0) {
          fill_tensor_constant_cpu(
              out, std::numeric_limits<double>::quiet_NaN());
        }
        return out;
      }
      const char* op = (kind == ReduceKind::Amin) ? "amin" : "amax";
      std::string msg = std::string(op) + ": empty";
      PyErr_SetString(PyExc_RuntimeError, msg.c_str());
      throw nb::python_error();
    }

    std::vector<int64_t> flat_sizes{total};
    vbt::core::TensorImpl flat = vbt::core::view(base, flat_sizes);

    std::int64_t rd = 0;
    TensorIter iter =
        TensorIter::reduce_op(out, flat, std::span<const std::int64_t>(&rd, 1));

    switch (kind) {
      case ReduceKind::Sum:
        if (st == ScalarType::Float32) {
          vbt::core::for_each_reduction_cpu(iter, &reduce_sum_f32_loop, nullptr);
        } else {
          vbt::core::for_each_reduction_cpu(iter, &reduce_sum_i64_loop, nullptr);
        }
        break;
      case ReduceKind::Mean:
        vbt::core::for_each_reduction_cpu(iter, &reduce_mean_f32_loop, nullptr);
        break;
      case ReduceKind::Amin:
        if (st == ScalarType::Float32) {
          vbt::core::for_each_reduction_cpu(iter, &reduce_amin_f32_loop, nullptr);
        } else {
          vbt::core::for_each_reduction_cpu(iter, &reduce_amin_i64_loop, nullptr);
        }
        break;
      case ReduceKind::Amax:
        if (st == ScalarType::Float32) {
          vbt::core::for_each_reduction_cpu(iter, &reduce_amax_f32_loop, nullptr);
        } else {
          vbt::core::for_each_reduction_cpu(iter, &reduce_amax_i64_loop, nullptr);
        }
        break;
    }

    return out;
  }

  // dim=int path
  if (R == 0) {
    std::string msg = std::string(opname) +
                      ": dim argument is invalid for rank-0 tensors";
    PyErr_SetString(PyExc_ValueError, msg.c_str());
    throw nb::python_error();
  }

  std::int64_t dim = pd.dim;
  if (dim < 0) dim += R;
  if (dim < 0 || dim >= R) {
    std::string msg = std::string(opname) + ": reduction dim out of range";
    PyErr_SetString(PyExc_ValueError, msg.c_str());
    throw nb::python_error();
  }

  std::vector<int64_t> out_sizes;
  if (keepdim) {
    out_sizes.assign(sizes.begin(), sizes.end());
    out_sizes[static_cast<std::size_t>(dim)] = 1;
  } else {
    out_sizes.reserve(R > 0 ? R - 1 : 0);
    for (std::int64_t d = 0; d < R; ++d) {
      if (d == dim) continue;
      out_sizes.push_back(sizes[static_cast<std::size_t>(d)]);
    }
  }

  vbt::core::TensorImpl base = self;
  if (!self.is_contiguous()) {
    base = vbt::core::clone_cpu(self);
  }

  vbt::core::TensorImpl out = make_contiguous_out_cpu(base, out_sizes, st);

  // Case A (empty output): return empty output and do not treat this as an
  // empty reduction domain error.
  if (out.numel() == 0) {
    return out;
  }

  const std::int64_t reduce_size = sizes[static_cast<std::size_t>(dim)];
  const std::int64_t total = base.numel();
  if (reduce_size == 0 || total == 0) {
    if (kind == ReduceKind::Sum) {
      fill_tensor_constant_cpu(out, 0.0);
      return out;
    }
    if (kind == ReduceKind::Mean) {
      if (st == ScalarType::Float32 && out.numel() > 0) {
        fill_tensor_constant_cpu(
            out, std::numeric_limits<double>::quiet_NaN());
      }
      return out;
    }
    const char* op = (kind == ReduceKind::Amin) ? "amin" : "amax";
    std::string msg = std::string(op) + ": empty";
    PyErr_SetString(PyExc_RuntimeError, msg.c_str());
    throw nb::python_error();
  }

  std::int64_t dim_canon = dim;
  TensorIter iter =
      TensorIter::reduce_op(out, base,
                            std::span<const std::int64_t>(&dim_canon, 1));

  switch (kind) {
    case ReduceKind::Sum:
      if (st == ScalarType::Float32) {
        vbt::core::for_each_reduction_cpu(iter, &reduce_sum_f32_loop, nullptr);
      } else {
        vbt::core::for_each_reduction_cpu(iter, &reduce_sum_i64_loop, nullptr);
      }
      break;
    case ReduceKind::Mean:
      vbt::core::for_each_reduction_cpu(iter, &reduce_mean_f32_loop, nullptr);
      break;
    case ReduceKind::Amin:
      if (st == ScalarType::Float32) {
        vbt::core::for_each_reduction_cpu(iter, &reduce_amin_f32_loop, nullptr);
      } else {
        vbt::core::for_each_reduction_cpu(iter, &reduce_amin_i64_loop, nullptr);
      }
      break;
    case ReduceKind::Amax:
      if (st == ScalarType::Float32) {
        vbt::core::for_each_reduction_cpu(iter, &reduce_amax_f32_loop, nullptr);
      } else {
        vbt::core::for_each_reduction_cpu(iter, &reduce_amax_i64_loop, nullptr);
      }
      break;
  }

  return out;
}

static void argmax_f32_loop(char** data,
                            const std::int64_t* strides,
                            std::int64_t size,
                            void* /*ctx*/) {
  char* out_val_base = data[0];
  char* out_idx_base = data[1];
  char* in_base      = data[2];
  const std::int64_t in_stride = strides[2];

  const auto* first =
      reinterpret_cast<const float*>(in_base);
  float best = *first;
  std::int64_t best_idx = 0;
  for (std::int64_t i = 1; i < size; ++i) {
    const auto* pi =
        reinterpret_cast<const float*>(in_base + i * in_stride);
    float v = *pi;
    if (v > best) {
      best = v;
      best_idx = i;
    }
  }

  *reinterpret_cast<float*>(out_val_base) = best;
  *reinterpret_cast<long long*>(out_idx_base) = best_idx;
}

static void argmin_f32_loop(char** data,
                            const std::int64_t* strides,
                            std::int64_t size,
                            void* /*ctx*/) {
  char* out_val_base = data[0];
  char* out_idx_base = data[1];
  char* in_base      = data[2];
  const std::int64_t in_stride = strides[2];

  const auto* first =
      reinterpret_cast<const float*>(in_base);
  float best = *first;
  std::int64_t best_idx = 0;
  for (std::int64_t i = 1; i < size; ++i) {
    const auto* pi =
        reinterpret_cast<const float*>(in_base + i * in_stride);
    float v = *pi;
    if (v < best) {
      best = v;
      best_idx = i;
    }
  }

  *reinterpret_cast<float*>(out_val_base) = best;
  *reinterpret_cast<long long*>(out_idx_base) = best_idx;
}

static void argmax_i64_loop(char** data,
                            const std::int64_t* strides,
                            std::int64_t size,
                            void* /*ctx*/) {
  char* out_val_base = data[0];
  char* out_idx_base = data[1];
  char* in_base      = data[2];
  const std::int64_t in_stride = strides[2];

  const auto* first =
      reinterpret_cast<const long long*>(in_base);
  long long best = *first;
  std::int64_t best_idx = 0;
  for (std::int64_t i = 1; i < size; ++i) {
    const auto* pi =
        reinterpret_cast<const long long*>(in_base + i * in_stride);
    long long v = *pi;
    if (v > best) {
      best = v;
      best_idx = i;
    }
  }

  *reinterpret_cast<long long*>(out_val_base) = best;
  *reinterpret_cast<long long*>(out_idx_base) = best_idx;
}

static void argmin_i64_loop(char** data,
                            const std::int64_t* strides,
                            std::int64_t size,
                            void* /*ctx*/) {
  char* out_val_base = data[0];
  char* out_idx_base = data[1];
  char* in_base      = data[2];
  const std::int64_t in_stride = strides[2];

  const auto* first =
      reinterpret_cast<const long long*>(in_base);
  long long best = *first;
  std::int64_t best_idx = 0;
  for (std::int64_t i = 1; i < size; ++i) {
    const auto* pi =
        reinterpret_cast<const long long*>(in_base + i * in_stride);
    long long v = *pi;
    if (v < best) {
      best = v;
      best_idx = i;
    }
  }

  *reinterpret_cast<long long*>(out_val_base) = best;
  *reinterpret_cast<long long*>(out_idx_base) = best_idx;
}

static std::pair<vbt::core::TensorImpl, vbt::core::TensorImpl>
reduce_arg_value_index_cpu(const vbt::core::TensorImpl& self,
                           const nb::object& dim_obj,
                           bool keepdim,
                           bool is_max,
                           const char* opname) {
  using vbt::core::ScalarType;
  using vbt::core::TensorIter;

  if (self.device().type != kDLCPU) {
    std::string msg = std::string(opname) + ": CPU tensors only for reductions";
    PyErr_SetString(PyExc_ValueError, msg.c_str());
    throw nb::python_error();
  }

  const ScalarType st = self.dtype();
  if (!(st == ScalarType::Float32 || st == ScalarType::Int64)) {
    std::string msg = std::string(opname) + ": expected dtype float32 or int64";
    PyErr_SetString(PyExc_ValueError, msg.c_str());
    throw nb::python_error();
  }

  ParsedDim pd = parse_dim_arg(dim_obj, opname);
  const auto& sizes = self.sizes();
  const std::int64_t R = static_cast<std::int64_t>(sizes.size());

  // dim=None path
  if (!pd.has_dim) {
    if (keepdim) {
      std::string msg = std::string(opname) +
                        ": dim=None with keepdim=True is not supported";
      PyErr_SetString(PyExc_ValueError, msg.c_str());
      throw nb::python_error();
    }

    if (R == 0) {
      // Scalar: trivial arg-reduction, index 0.
      vbt::core::TensorImpl values = vbt::core::clone_cpu(self);
      std::vector<int64_t> idx_sizes;  // scalar
      vbt::core::TensorImpl indices =
          make_contiguous_out_cpu(self, idx_sizes, ScalarType::Int64);
      fill_tensor_constant_cpu(indices, 0.0);
      return std::make_pair(std::move(values), std::move(indices));
    }

    vbt::core::TensorImpl base = self;
    if (!self.is_contiguous()) {
      base = vbt::core::clone_cpu(self);
    }
    const std::int64_t total = base.numel();

    std::vector<int64_t> out_sizes;  // scalar
    vbt::core::TensorImpl values = make_contiguous_out_cpu(base, out_sizes, st);
    vbt::core::TensorImpl indices =
        make_contiguous_out_cpu(base, out_sizes, ScalarType::Int64);

    if (total == 0) {
      std::string msg = std::string(opname) + ": empty";
      PyErr_SetString(PyExc_RuntimeError, msg.c_str());
      throw nb::python_error();
    }

    std::vector<int64_t> flat_sizes{total};
    vbt::core::TensorImpl flat = vbt::core::view(base, flat_sizes);

    std::int64_t rd = 0;
    TensorIter iter = TensorIter::reduce_op(
        values, indices, flat, std::span<const std::int64_t>(&rd, 1));

    if (st == ScalarType::Float32) {
      if (is_max) {
        vbt::core::for_each_reduction_cpu(iter, &argmax_f32_loop, nullptr);
      } else {
        vbt::core::for_each_reduction_cpu(iter, &argmin_f32_loop, nullptr);
      }
    } else {  // Int64
      if (is_max) {
        vbt::core::for_each_reduction_cpu(iter, &argmax_i64_loop, nullptr);
      } else {
        vbt::core::for_each_reduction_cpu(iter, &argmin_i64_loop, nullptr);
      }
    }

    return std::make_pair(std::move(values), std::move(indices));
  }

  // dim=int path
  if (R == 0) {
    std::string msg = std::string(opname) +
                      ": dim argument is invalid for rank-0 tensors";
    PyErr_SetString(PyExc_ValueError, msg.c_str());
    throw nb::python_error();
  }

  std::int64_t dim = pd.dim;
  if (dim < 0) dim += R;
  if (dim < 0 || dim >= R) {
    std::string msg = std::string(opname) + ": reduction dim out of range";
    PyErr_SetString(PyExc_ValueError, msg.c_str());
    throw nb::python_error();
  }

  std::vector<int64_t> out_sizes;
  if (keepdim) {
    out_sizes.assign(sizes.begin(), sizes.end());
    out_sizes[static_cast<std::size_t>(dim)] = 1;
  } else {
    out_sizes.reserve(R > 0 ? R - 1 : 0);
    for (std::int64_t d = 0; d < R; ++d) {
      if (d == dim) continue;
      out_sizes.push_back(sizes[static_cast<std::size_t>(d)]);
    }
  }

  vbt::core::TensorImpl base = self;
  if (!self.is_contiguous()) {
    base = vbt::core::clone_cpu(self);
  }

  vbt::core::TensorImpl values = make_contiguous_out_cpu(base, out_sizes, st);
  vbt::core::TensorImpl indices =
      make_contiguous_out_cpu(base, out_sizes, ScalarType::Int64);

  const std::int64_t reduce_size = sizes[static_cast<std::size_t>(dim)];
  const std::int64_t total = base.numel();
  if (reduce_size == 0 || total == 0) {
    std::string msg = std::string(opname) + ": empty";
    PyErr_SetString(PyExc_RuntimeError, msg.c_str());
    throw nb::python_error();
  }

  std::int64_t dim_canon = dim;
  TensorIter iter = TensorIter::reduce_op(
      values, indices, base,
      std::span<const std::int64_t>(&dim_canon, 1));

  if (st == ScalarType::Float32) {
    if (is_max) {
      vbt::core::for_each_reduction_cpu(iter, &argmax_f32_loop, nullptr);
    } else {
      vbt::core::for_each_reduction_cpu(iter, &argmin_f32_loop, nullptr);
    }
  } else {  // Int64
    if (is_max) {
      vbt::core::for_each_reduction_cpu(iter, &argmax_i64_loop, nullptr);
    } else {
      vbt::core::for_each_reduction_cpu(iter, &argmin_i64_loop, nullptr);
    }
  }

  return std::make_pair(std::move(values), std::move(indices));
}

//
// rejects advanced (tensor / boolean / sequence-of-scalars) indices with the
// stable "advanced indexing (boolean or tensor indices) is not implemented
static IndexSpec parse_indices_py(const TensorImpl& self,
                                  nb::handle index_obj) {
  using vbt::core::indexing::expand_ellipsis_and_validate;

  nb::tuple tup;
  if (PyTuple_Check(index_obj.ptr())) {
    tup = nb::borrow<nb::tuple>(index_obj.ptr());
  } else {
    PyObject* wrapped = wrap_index_to_tuple(index_obj.ptr());
    if (!wrapped) {
      throw nb::python_error();
    }
    tup = nb::steal<nb::tuple>(wrapped);
  }

  IndexSpec raw;
  raw.items.reserve(tup.size());

  for (auto obj : tup) {
    if (obj.is_none()) {
      raw.items.emplace_back(TensorIndex(nullptr));
    } else if (obj.ptr() == Py_Ellipsis) {
      raw.items.emplace_back(TensorIndex(TensorIndex::EllipsisTag{}));
    } else if (nb::isinstance<nb::bool_>(obj)) {
      bool b = nb::cast<bool>(obj);
      raw.items.emplace_back(TensorIndex(b));
    } else if (nb::isinstance<nb::int_>(obj)) {
      auto v = nb::cast<std::int64_t>(obj);
      raw.items.emplace_back(TensorIndex(v));
    } else if (PySlice_Check(obj.ptr())) {
      PySliceObject* so = reinterpret_cast<PySliceObject*>(obj.ptr());
      vbt::core::indexing::Slice s;
      if (so->start != Py_None) {
        s.start = nb::cast<std::int64_t>(nb::handle(so->start));
      }
      if (so->stop != Py_None) {
        s.stop = nb::cast<std::int64_t>(nb::handle(so->stop));
      }
      if (so->step != Py_None) {
        s.step = nb::cast<std::int64_t>(nb::handle(so->step));
      }
      raw.items.emplace_back(TensorIndex(std::move(s)));
    } else if (nb::isinstance<TensorImpl>(obj)) {
      TensorImpl t = nb::cast<TensorImpl>(obj);
      raw.items.emplace_back(TensorIndex(t));
    } else if (PyList_Check(obj.ptr()) || PyTuple_Check(obj.ptr())) {
      raw.items.emplace_back(TensorIndex(TensorImpl{}));
    } else {
      PyErr_SetString(
          PyExc_TypeError,
          "only integers, slices (':'), ellipsis ('...'), None and scalar "
          "boolean or integer indices are valid indices");
      throw nb::python_error();
    }
  }

  if (has_any_advanced(raw)) {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "advanced indexing (boolean or tensor indices) "
        "is not implemented");
    throw nb::python_error();
  }

  const std::int64_t rank = static_cast<std::int64_t>(self.sizes().size());

  if (rank == 0) {
    bool has_dim_consuming = false;
    for (const auto& it : raw.items) {
      if (it.kind == IndexKind::Integer || it.kind == IndexKind::Slice) {
        has_dim_consuming = true;
        break;
      }
    }
    if (has_dim_consuming) {
      PyErr_SetString(PyExc_IndexError,
                      vbt::core::indexing::errors::kErrInvalidZeroDim);
      throw nb::python_error();
    }
    // For 0-d tensors with advanced indices, defer normalization to the
    // (e.g., "advanced indexing is not supported for 0-d tensors")
    // remain canonical. make_advanced_index() will handle ellipsis and
    // validation itself.
    if (vbt::core::indexing::has_any_advanced(raw)) {
      return raw;
    }
  }

  IndexSpec spec;
  try {
    spec = expand_ellipsis_and_validate(raw, rank);
  } catch (const std::invalid_argument& e) {
    std::string msg = e.what();
    if (msg.find(vbt::core::indexing::errors::kErrTooManyIndices) != std::string::npos) {
      PyErr_SetString(PyExc_IndexError, msg.c_str());
      throw nb::python_error();
    }
    if (msg.find(vbt::core::indexing::errors::kErrMultipleEllipsis) != std::string::npos) {
      PyErr_SetString(PyExc_IndexError, msg.c_str());
      throw nb::python_error();
    }
    throw;
  }

  return spec;
}

static IndexSpec parse_indices_py_allow_advanced(const TensorImpl& self,
                                                 nb::handle index_obj) {
  // Allow-advanced variant: shares normalization logic with
  // parse_indices_py but preserves advanced kinds in the returned
  // IndexSpec and defers CPU / feature-flag / vt routing policy to
  using vbt::core::indexing::expand_ellipsis_and_validate;

  nb::tuple tup;
  if (PyTuple_Check(index_obj.ptr())) {
    tup = nb::borrow<nb::tuple>(index_obj.ptr());
  } else {
    PyObject* wrapped = wrap_index_to_tuple(index_obj.ptr());
    if (!wrapped) {
      throw nb::python_error();
    }
    tup = nb::steal<nb::tuple>(wrapped);
  }

  IndexSpec raw;
  raw.items.reserve(tup.size());

  for (auto obj : tup) {
    if (obj.is_none()) {
      raw.items.emplace_back(TensorIndex(nullptr));
    } else if (obj.ptr() == Py_Ellipsis) {
      raw.items.emplace_back(TensorIndex(TensorIndex::EllipsisTag{}));
    } else if (nb::isinstance<nb::bool_>(obj)) {
      bool b = nb::cast<bool>(obj);
      raw.items.emplace_back(TensorIndex(b));
    } else if (nb::isinstance<nb::int_>(obj)) {
      auto v = nb::cast<std::int64_t>(obj);
      raw.items.emplace_back(TensorIndex(v));
    } else if (PySlice_Check(obj.ptr())) {
      PySliceObject* so = reinterpret_cast<PySliceObject*>(obj.ptr());
      vbt::core::indexing::Slice s;
      if (so->start != Py_None) {
        s.start = nb::cast<std::int64_t>(nb::handle(so->start));
      }
      if (so->stop != Py_None) {
        s.stop = nb::cast<std::int64_t>(nb::handle(so->stop));
      }
      if (so->step != Py_None) {
        s.step = nb::cast<std::int64_t>(nb::handle(so->step));
      }
      raw.items.emplace_back(TensorIndex(std::move(s)));
    } else if (nb::isinstance<TensorImpl>(obj)) {
      TensorImpl t = nb::cast<TensorImpl>(obj);
      raw.items.emplace_back(TensorIndex(t));
    } else if (PyList_Check(obj.ptr()) || PyTuple_Check(obj.ptr())) {
      raw.items.emplace_back(TensorIndex(TensorImpl{}));
    } else {
      PyErr_SetString(
          PyExc_TypeError,
          "only integers, slices (':'), ellipsis ('...'), None and scalar "
          "boolean or integer indices are valid indices");
      throw nb::python_error();
    }
  }

  const std::int64_t rank = static_cast<std::int64_t>(self.sizes().size());

  if (rank == 0) {
    bool has_dim_consuming = false;
    for (const auto& it : raw.items) {
      if (it.kind == IndexKind::Integer || it.kind == IndexKind::Slice) {
        has_dim_consuming = true;
        break;
      }
    }
    if (has_dim_consuming) {
      PyErr_SetString(PyExc_IndexError,
                      vbt::core::indexing::errors::kErrInvalidZeroDim);
      throw nb::python_error();
    }
    // For 0-d tensors with advanced indices, defer normalization to the
    // (e.g., "advanced indexing is not supported for 0-d tensors")
    // remain canonical. make_advanced_index() will handle ellipsis and
    // validation itself.
    if (vbt::core::indexing::has_any_advanced(raw)) {
      return raw;
    }
  }

  IndexSpec spec;
  try {
    spec = expand_ellipsis_and_validate(raw, rank);
  } catch (const std::invalid_argument& e) {
    std::string msg = e.what();
    if (msg.find(vbt::core::indexing::errors::kErrTooManyIndices) != std::string::npos) {
      PyErr_SetString(PyExc_IndexError, msg.c_str());
      throw nb::python_error();
    }
    if (msg.find(vbt::core::indexing::errors::kErrMultipleEllipsis) != std::string::npos) {
      PyErr_SetString(PyExc_IndexError, msg.c_str());
      throw nb::python_error();
    }
    throw;
  }

  return spec;
}

static TensorImpl encode_index_meta_tensor(const IndexSpec& prefix_spec) {
  using vbt::core::Storage;
  using vbt::core::DataPtr;
  using vbt::core::itemsize;

  constexpr std::int64_t kSentinel = std::numeric_limits<std::int64_t>::min();

  // Meta tensor always lives on CPU regardless of self's device.
  const ScalarType dtype = ScalarType::Int64;

  const std::int64_t prefix_len =
      static_cast<std::int64_t>(prefix_spec.items.size());
  const std::int64_t max_prefix =
      (std::numeric_limits<std::int64_t>::max() - 4) / 4;
  if (prefix_len < 0 || prefix_len > max_prefix) {
    throw std::invalid_argument(
        "encode_index_meta_tensor: invalid prefix length");
  }

  std::vector<std::int64_t> sizes{4 + 4 * prefix_len};
  const std::size_t item_b = static_cast<std::size_t>(itemsize(dtype));
  const std::size_t nbytes = static_cast<std::size_t>(sizes[0]) * item_b;

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<std::int64_t> strides{1};
  TensorImpl meta(storage, sizes, strides,
                  /*storage_offset=*/0,
                  dtype,
                  vbt::core::Device::cpu());

  auto* data = static_cast<std::int64_t*>(meta.data());
  data[0] = 0;  // version
  data[1] = 1;  // advanced_kind = Tensor
  data[2] = 0;  // advanced_param (reserved)
  data[3] = prefix_len;

  for (std::int64_t i = 0; i < prefix_len; ++i) {
    const auto& it = prefix_spec.items[static_cast<std::size_t>(i)];
    const std::size_t off = static_cast<std::size_t>(4 + 4 * i);

    if (it.kind == IndexKind::None) {
      data[off + 0] = 0;  // kind_tag = None
      data[off + 1] = 0;
      data[off + 2] = 0;
      data[off + 3] = 0;
      continue;
    }

    if (it.kind == IndexKind::Integer) {
      data[off + 0] = 1;  // kind_tag = Integer
      data[off + 1] = it.integer;
      data[off + 2] = 0;
      data[off + 3] = 0;
      continue;
    }

    if (it.kind == IndexKind::Slice) {
      data[off + 0] = 2;  // kind_tag = Slice
      data[off + 1] = it.slice.start.has_value() ? *it.slice.start : kSentinel;
      data[off + 2] = it.slice.stop.has_value() ? *it.slice.stop : kSentinel;
      data[off + 3] = it.slice.step.has_value() ? *it.slice.step : kSentinel;
      continue;
    }

    throw std::invalid_argument(
        "encode_index_meta_tensor: unsupported prefix index kind");
  }

  return meta;
}

// Test-only helper to build (index, meta) pairs for vt::index/vt::index_put
// CPU with prefix_len == 0.
static nb::tuple encode_index_spec_for_tests(const TensorImpl& self,
                                             const TensorImpl& index) {
  const auto self_dev = self.device();
  const auto index_dev = index.device();
  if (self_dev.type != index_dev.type || self_dev.index != index_dev.index) {
    throw std::invalid_argument(
        "_encode_index_spec: self and index must be on the same device");
  }

  ScalarType dt = index.dtype();
  if (self_dev.type == kDLCPU) {
    if (!(dt == ScalarType::Int32 ||
          dt == ScalarType::Int64 ||
          dt == ScalarType::Bool)) {
      throw std::invalid_argument(
          "_encode_index_spec: unsupported index pattern on CPU");
    }
  } else if (self_dev.type == kDLCUDA) {
    if (!(dt == ScalarType::Int32 ||
          dt == ScalarType::Int64)) {
      throw std::invalid_argument(
          "_encode_index_spec: unsupported index pattern on CUDA");
    }
  } else {
    throw std::invalid_argument(
        "_encode_index_spec: unsupported device for advanced indexing");
  }

  TensorImpl meta = encode_index_meta_tensor(IndexSpec{});
  return nb::make_tuple(index, meta);
}

static TensorImpl Tensor_getitem(const TensorImpl& self,
                                 nb::handle index_obj) {
  using vbt::core::indexing::basic_index;
  using vbt::core::indexing::advanced_indexing_enabled;
  using vbt::core::indexing::is_advanced_kind;
  using vbt::core::indexing::index;

  IndexSpec spec = parse_indices_py_allow_advanced(self, index_obj);

  if (!has_any_advanced(spec)) {
    TensorImpl out = basic_index(self, spec);
#if VBT_WITH_AUTOGRAD
    vbt::autograd::as_view(self, out);

    const bool graph_enabled =
        vbt::autograd::GradMode::is_enabled() &&
        !vbt::autograd::InferenceMode::is_enabled();

    if (graph_enabled &&
        vbt::autograd::autograd_indexing_v2_enabled() &&
        vbt::autograd::requires_grad(out) &&
        vbt::autograd::is_view(out)) {
      auto node = vbt::autograd::make_basic_index_view_backward_node(
          self, std::move(spec));
      node->next_edges.resize(node->num_inputs());

      std::unordered_map<const vbt::autograd::AutogradMeta*,
                         vbt::core::intrusive_ptr<vbt::autograd::Node>> sinks;
      node->next_edges[0] = vbt::autograd::resolve_edge_for_tensor(self, sinks);

      if (auto* m = vbt::autograd::get_autograd_meta(out, /*create_if_missing=*/true)) {
        m->is_leaf = false;
        m->output_nr = 0;
        m->grad_fn = node;
      }
    }
#endif
    return out;
  }

  if (self.device().type != kDLCPU &&
      self.device().type != kDLCUDA) {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "advanced indexing is only implemented for CPU and CUDA tensors");
    throw nb::python_error();
  }

  if (!advanced_indexing_enabled()) {
    PyErr_SetString(PyExc_RuntimeError,
                    vbt::core::indexing::errors::kErrAdvDisabledCore);
    throw nb::python_error();
  }

  const std::int64_t rank =
      static_cast<std::int64_t>(self.sizes().size());
  if (rank == 0) {
    // 0-d base with any advanced indices: call core index() directly so
    try {
      TensorImpl out = index(self, spec);
      return out;
    } catch (const std::invalid_argument& e) {
      std::string msg = e.what();
      if (msg.find("advanced indexing is not supported for 0-d tensors") !=
          std::string::npos) {
        PyErr_SetString(PyExc_RuntimeError, msg.c_str());
        throw nb::python_error();
      }
      PyErr_SetString(PyExc_ValueError, msg.c_str());
      throw nb::python_error();
    } catch (const std::runtime_error& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      throw nb::python_error();
    }
  }

  // Classify spec into prefix + single advanced index.
  int first_adv = -1;
  int adv_count = 0;
  const std::int64_t n_items =
      static_cast<std::int64_t>(spec.items.size());
  for (std::int64_t i = 0; i < n_items; ++i) {
    const auto kind = spec.items[static_cast<std::size_t>(i)].kind;
    if (is_advanced_kind(kind)) {
      if (first_adv < 0) {
        first_adv = static_cast<int>(i);
      }
      ++adv_count;
    }
  }

  if (first_adv < 0) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "internal error in Tensor_getitem: advanced spec misclassified");
    throw nb::python_error();
  }

  const TensorIndex& adv_it =
      spec.items[static_cast<std::size_t>(first_adv)];

  // They either fall back to core advanced indexing for 0-d (handled
  // above) or are treated as unsupported patterns here.
  if (adv_it.kind == IndexKind::Boolean) {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "advanced indexing pattern is not supported");
    throw nb::python_error();
  }

  if (adv_count > 1) {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "advanced indexing with multiple tensor/bool indices is not supported");
    throw nb::python_error();
  }

  // Reject suffix indices after the advanced block (except for trailing full
  // slices on CPU for integer tensor indices, which enable patterns like
  // x[idx] on rank>1).
  const bool allow_suffix_full_slices =
      (self.device().type == kDLCPU &&
       adv_it.kind == IndexKind::Tensor &&
       adv_it.tensor.storage().get() &&
       adv_it.tensor.dtype() != ScalarType::Bool);

  for (int i = first_adv + 1; i < static_cast<int>(n_items); ++i) {
    const auto& it = spec.items[static_cast<std::size_t>(i)];
    const auto kind = it.kind;
    if (is_advanced_kind(kind)) {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "advanced indexing with multiple tensor/bool indices is not supported");
      throw nb::python_error();
    }
    if (kind == IndexKind::Slice) {
      const auto& s = it.slice;
      if (allow_suffix_full_slices &&
          !s.start.has_value() && !s.stop.has_value() && !s.step.has_value()) {
        continue;
      }
    }
    if (kind == IndexKind::Integer ||
        kind == IndexKind::Slice ||
        kind == IndexKind::None) {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "advanced indexing with suffix basic indices is not supported");
      throw nb::python_error();
    }
  }

  IndexSpec prefix_spec;
  prefix_spec.items.assign(
      spec.items.begin(),
      spec.items.begin() + static_cast<std::ptrdiff_t>(first_adv));

  const bool is_cuda = (self.device().type == kDLCUDA);

  // Base tensor for vt::index; prefix basics are encoded in meta.
  TensorImpl base = self;

  // Currently vt path only supports tensor-based advanced indices with a real
  if (adv_it.kind != IndexKind::Tensor || !adv_it.tensor.storage().get()) {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "advanced indexing pattern is not supported");
    throw nb::python_error();
  }

  TensorImpl index_tensor = adv_it.tensor;

  // Validate dtype for tensor advanced index at Python level.
  ScalarType dt = index_tensor.dtype();
  if (!(dt == ScalarType::Int32 ||
        dt == ScalarType::Int64 ||
        dt == ScalarType::Bool)) {
    PyErr_SetString(
        PyExc_ValueError,
        "advanced index tensor must be int32, int64, or bool");
    throw nb::python_error();
  }

#if VBT_WITH_AUTOGRAD
  const bool graph_enabled_for_mask =
      vbt::autograd::GradMode::is_enabled() &&
      !vbt::autograd::InferenceMode::is_enabled();

  const bool wants_autograd_for_mask =
      graph_enabled_for_mask &&
      vbt::autograd::autograd_indexing_v2_enabled() &&
      vbt::autograd::requires_grad(self);
#else
  const bool wants_autograd_for_mask = false;
#endif

  if (is_cuda && dt == ScalarType::Bool && !wants_autograd_for_mask) {
    // For CUDA boolean masks, rely on core index() when autograd is not
    // required so error surfaces remain stable.
    try {
      TensorImpl out = index(self, spec);
      return out;
    } catch (const std::invalid_argument& e) {
      std::string msg = e.what();
      if (msg.find("advanced indexing is not supported for 0-d tensors") !=
          std::string::npos) {
        PyErr_SetString(PyExc_RuntimeError, msg.c_str());
        throw nb::python_error();
      }
      PyErr_SetString(PyExc_ValueError, msg.c_str());
      throw nb::python_error();
    } catch (const std::runtime_error& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      throw nb::python_error();
    }
  }

  TensorImpl meta;
  try {
    meta = encode_index_meta_tensor(prefix_spec);
  } catch (const std::invalid_argument& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    throw nb::python_error();
  }

  vbt::dispatch::BoxedStack stack;
  stack.push_back(base);
  stack.push_back(index_tensor);
  stack.push_back(meta);

  try {
    vbt::dispatch::Dispatcher::instance().callBoxed("vt::index", stack);
  } catch (const std::invalid_argument& e) {
    std::string msg = e.what();
    if (msg.find("advanced indexing is only implemented for CPU and CUDA tensors") !=
        std::string::npos) {
      PyErr_SetString(PyExc_NotImplementedError, msg.c_str());
      throw nb::python_error();
    }
    PyErr_SetString(PyExc_ValueError, msg.c_str());
    throw nb::python_error();
  } catch (const std::runtime_error& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    throw nb::python_error();
  }

  if (stack.size() != 1) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "vt::index boxed kernel did not return exactly one tensor");
    throw nb::python_error();
  }

  return stack[0];
}

static TensorImpl to_tensor_impl_for_setitem(const TensorImpl& self,
                                             nb::handle value_obj) {
  if (nb::isinstance<TensorImpl>(value_obj)) {
    return nb::cast<TensorImpl>(value_obj);
  }

  if (!(nb::isinstance<nb::bool_>(value_obj) ||
        nb::isinstance<nb::int_>(value_obj) ||
        nb::isinstance<nb::float_>(value_obj))) {
    PyErr_SetString(
        PyExc_TypeError,
        "Tensor.__setitem__ value must be Tensor or scalar");
    throw nb::python_error();
  }

  ScalarType st = self.dtype();
  std::vector<std::int64_t> sizes;  // 0-d scalar
  TensorImpl out = make_contiguous_out_cpu(self, sizes, st);

  if (out.numel() == 0) {
    return out;
  }

  void* data = out.data();
  switch (st) {
    case ScalarType::Float32: {
      float v = nb::cast<float>(value_obj);
      auto* p = static_cast<float*>(data);
      p[0] = v;
      break;
    }
    case ScalarType::Int32: {
      std::int32_t v = 0;
      try {
        v = nb::cast<std::int32_t>(value_obj);
      } catch (...) {
        v = static_cast<std::int32_t>(nb::cast<double>(value_obj));
      }
      auto* p = static_cast<std::int32_t*>(data);
      p[0] = v;
      break;
    }
    case ScalarType::Int64: {
      std::int64_t v = 0;
      try {
        v = nb::cast<std::int64_t>(value_obj);
      } catch (...) {
        v = static_cast<std::int64_t>(nb::cast<double>(value_obj));
      }
      auto* p = static_cast<std::int64_t*>(data);
      p[0] = v;
      break;
    }
    case ScalarType::Bool: {
      bool v = nb::cast<bool>(value_obj);
      auto* p = static_cast<std::uint8_t*>(data);
      p[0] = v ? 1u : 0u;
      break;
    }
    default: {
      PyErr_SetString(
          PyExc_TypeError,
          "Tensor.__setitem__ scalar path only supports float32, int32, int64, and bool");
      throw nb::python_error();
    }
  }

  return out;
}

static void Tensor_setitem(TensorImpl& self,
                           nb::handle index_obj,
                           nb::handle value_obj) {
  using vbt::core::indexing::index_put_;

  const auto dev_type = self.device().type;

  TensorImpl value;
  if (dev_type == kDLCPU) {
    // CPU path supports Tensor and scalar values via the existing helper.
    value = to_tensor_impl_for_setitem(self, value_obj);
  } else if (dev_type == kDLCUDA) {
    // For CUDA, restrict to Tensor values; Python scalars must be
    // explicitly converted to tensors on the desired device.
    if (!nb::isinstance<TensorImpl>(value_obj)) {
      PyErr_SetString(
          PyExc_TypeError,
          "Tensor.__setitem__ on CUDA requires a Tensor value");
      throw nb::python_error();
    }
    value = nb::cast<TensorImpl>(value_obj);
  } else {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "Tensor.__setitem__ is only implemented for CPU and CUDA tensors");
    throw nb::python_error();
  }

  IndexSpec spec = parse_indices_py_allow_advanced(self, index_obj);

  if (!has_any_advanced(spec)) {
    if (dev_type != kDLCPU) {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "Tensor.__setitem__ basic indexing is only implemented for CPU tensors");
      throw nb::python_error();
    }

    if (value.device().type != self.device().type ||
        value.device().index != self.device().index ||
        value.dtype() != self.dtype()) {
      PyErr_SetString(PyExc_ValueError,
                      "index assignment: dtype/device mismatch");
      throw nb::python_error();
    }

#if VBT_WITH_AUTOGRAD
    const bool graph_enabled =
        vbt::autograd::GradMode::is_enabled() &&
        !vbt::autograd::InferenceMode::is_enabled();

    const bool wants_autograd =
        graph_enabled &&
        vbt::autograd::autograd_indexing_v2_enabled() &&
        (vbt::autograd::requires_grad(self) ||
         vbt::autograd::requires_grad(value));

    const bool inplace_autograd_allowed =
        (vbt::autograd::is_view(self) || !vbt::autograd::is_leaf(self));

    if (wants_autograd && !inplace_autograd_allowed) {
      PyErr_SetString(
          PyExc_RuntimeError,
          "basic index put autograd: in-place on leaf tensors is not supported");
      throw nb::python_error();
    }

    const bool need_autograd = wants_autograd && inplace_autograd_allowed;

    if (need_autograd) {
      if (self.dtype() != ScalarType::Float32) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "basic index put autograd: only Float32 CPU is supported");
        throw nb::python_error();
      }

      // Pre-flight validation so failures don't mutate autograd metadata.
      TensorImpl dst = vbt::core::indexing::basic_index(self, spec);
      const auto& dst_sizes = dst.sizes();
      TensorImpl value_b = vbt::core::indexing::broadcast_to(
          value,
          std::span<const std::int64_t>(dst_sizes.data(), dst_sizes.size()));
      vbt::core::check_writable(self);
      if (self.storage().get() == value_b.storage().get()) {
        vbt::core::assert_no_partial_overlap(self, value_b);
      }

      auto node = vbt::autograd::make_basic_index_put_backward_node(
          self, value, spec);
      node->next_edges.resize(node->num_inputs());

      std::unordered_map<const vbt::autograd::AutogradMeta*,
                         vbt::core::intrusive_ptr<vbt::autograd::Node>> sinks;
      node->next_edges[0] = vbt::autograd::resolve_edge_for_tensor(self, sinks);
      node->next_edges[1] = vbt::autograd::resolve_edge_for_tensor(value, sinks);

      vbt::autograd::rebase_history(self, node);
    }
#endif

    vbt::core::indexing::basic_index_put(self, spec, value);
    return;
  }

  if (!vbt::core::indexing::advanced_indexing_enabled()) {
    PyErr_SetString(PyExc_RuntimeError,
                    vbt::core::indexing::errors::kErrAdvDisabledCore);
    throw nb::python_error();
  }

  const std::int64_t rank =
      static_cast<std::int64_t>(self.sizes().size());
  if (rank == 0) {
    // 0-d base with any advanced indices: call core index_put_ directly
    if (value.device().type != self.device().type ||
        value.device().index != self.device().index ||
        value.dtype() != self.dtype()) {
      PyErr_SetString(PyExc_ValueError,
                      "index assignment: dtype/device mismatch");
      throw nb::python_error();
    }

    try {
      index_put_(self, spec, value, /*accumulate=*/false);
      return;
    } catch (const std::invalid_argument& e) {
      std::string msg = e.what();
      if (msg.find("advanced indexing is not supported for 0-d tensors") !=
          std::string::npos) {
        PyErr_SetString(PyExc_RuntimeError, msg.c_str());
        throw nb::python_error();
      }
      PyErr_SetString(PyExc_ValueError, msg.c_str());
      throw nb::python_error();
    } catch (const std::runtime_error& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      throw nb::python_error();
    }
  }

  using vbt::core::indexing::is_advanced_kind;

  int first_adv = -1;
  int adv_count = 0;
  const std::int64_t n_items =
      static_cast<std::int64_t>(spec.items.size());
  for (std::int64_t i = 0; i < n_items; ++i) {
    const auto kind = spec.items[static_cast<std::size_t>(i)].kind;
    if (is_advanced_kind(kind)) {
      if (first_adv < 0) {
        first_adv = static_cast<int>(i);
      }
      ++adv_count;
    }
  }

  if (first_adv < 0) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "internal error in Tensor_setitem: advanced spec misclassified");
    throw nb::python_error();
  }

  const TensorIndex& adv_it =
      spec.items[static_cast<std::size_t>(first_adv)];

  // Scalar-bool advanced patterns and sequence-of-scalars/dummy tensors are
  if (adv_it.kind == IndexKind::Boolean || !adv_it.tensor.storage().get()) {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "advanced indexing pattern is not supported");
    throw nb::python_error();
  }

  if (adv_count > 1) {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "advanced indexing with multiple tensor/bool indices is not supported");
    throw nb::python_error();
  }

  // Reject suffix indices after the advanced block (except for trailing full
  // slices on CPU for integer tensor indices, which enable patterns like
  // x[idx] = v on rank>1).
  const bool allow_suffix_full_slices =
      (dev_type == kDLCPU &&
       adv_it.kind == IndexKind::Tensor &&
       adv_it.tensor.storage().get() &&
       adv_it.tensor.dtype() != ScalarType::Bool);

  for (int i = first_adv + 1; i < static_cast<int>(n_items); ++i) {
    const auto& it = spec.items[static_cast<std::size_t>(i)];
    const auto kind = it.kind;
    if (is_advanced_kind(kind)) {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "advanced indexing with multiple tensor/bool indices is not supported");
      throw nb::python_error();
    }
    if (kind == IndexKind::Slice) {
      const auto& s = it.slice;
      if (allow_suffix_full_slices &&
          !s.start.has_value() && !s.stop.has_value() && !s.step.has_value()) {
        continue;
      }
    }
    if (kind == IndexKind::Integer ||
        kind == IndexKind::Slice ||
        kind == IndexKind::None) {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "advanced indexing with suffix basic indices is not supported");
      throw nb::python_error();
    }
  }

  IndexSpec prefix_spec;
  prefix_spec.items.assign(
      spec.items.begin(),
      spec.items.begin() + static_cast<std::ptrdiff_t>(first_adv));

  const bool has_prefix = !prefix_spec.items.empty();

  // For prefix-based advanced patterns, rely on core index_put_ so that
  if (has_prefix) {
#if VBT_WITH_AUTOGRAD
    const bool graph_enabled =
        vbt::autograd::GradMode::is_enabled() &&
        !vbt::autograd::InferenceMode::is_enabled();
    const bool self_requires_grad = vbt::autograd::requires_grad(self);
    const bool value_requires_grad = vbt::autograd::requires_grad(value);
    const bool use_vt_kernel =
        graph_enabled && (self_requires_grad || value_requires_grad);

    if (use_vt_kernel) {
      TensorImpl base = self;
      TensorImpl index_tensor = adv_it.tensor;

      TensorImpl meta;
      try {
        meta = encode_index_meta_tensor(prefix_spec);
      } catch (const std::invalid_argument& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        throw nb::python_error();
      }

      // 0-d Bool CPU tensor encoding accumulate=False.
      std::vector<std::int64_t> acc_sizes;
      TensorImpl like_for_acc = self;
      if (dev_type != kDLCPU) {
        like_for_acc = TensorImpl(
            vbt::core::StoragePtr{},
            std::vector<std::int64_t>{},
            std::vector<std::int64_t>{},
            /*storage_offset=*/0,
            ScalarType::Bool,
            vbt::core::Device::cpu());
      }
      TensorImpl acc =
          make_contiguous_out_cpu(like_for_acc, acc_sizes, ScalarType::Bool);
      if (acc.numel() > 0) {
        auto* p = static_cast<std::uint8_t*>(acc.data());
        *p = 0u;
      }

      vbt::dispatch::BoxedStack stack;
      stack.push_back(base);
      stack.push_back(index_tensor);
      stack.push_back(value);
      stack.push_back(meta);
      stack.push_back(acc);

      try {
        vbt::dispatch::Dispatcher::instance().callBoxed("vt::index_put", stack);
      } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        if (msg.find("advanced indexing is only implemented for CPU and CUDA tensors") !=
            std::string::npos) {
          PyErr_SetString(PyExc_NotImplementedError, msg.c_str());
          throw nb::python_error();
        }
        PyErr_SetString(PyExc_ValueError, msg.c_str());
        throw nb::python_error();
      } catch (const std::runtime_error& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw nb::python_error();
      }
      return;
    }
#endif
    if (value.device().type != self.device().type ||
        value.device().index != self.device().index ||
        value.dtype() != self.dtype()) {
      PyErr_SetString(PyExc_ValueError,
                      "index assignment: dtype/device mismatch");
      throw nb::python_error();
    }

    try {
      index_put_(self, spec, value, /*accumulate=*/false);
      return;
    } catch (const std::invalid_argument& e) {
      std::string msg = e.what();
      if (msg.find("advanced indexing is not supported for 0-d tensors") !=
          std::string::npos) {
        PyErr_SetString(PyExc_RuntimeError, msg.c_str());
        throw nb::python_error();
      }
      PyErr_SetString(PyExc_ValueError, msg.c_str());
      throw nb::python_error();
    } catch (const std::runtime_error& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      throw nb::python_error();
    }
  }

  // Prefixless advanced patterns: choose core or vt::index_put based on
  // autograd state. When GradMode is disabled or neither self nor value
  // requires grad, call core index_put_ directly to avoid installing any
  if (value.device().type != self.device().type ||
      value.device().index != self.device().index ||
      value.dtype() != self.dtype()) {
    PyErr_SetString(PyExc_ValueError,
                    "index assignment: dtype/device mismatch");
    throw nb::python_error();
  }

  TensorImpl base = self;
  TensorImpl index_tensor = adv_it.tensor;

  ScalarType dt = index_tensor.dtype();
  if (dev_type == kDLCUDA) {
    if (!(dt == ScalarType::Int32 ||
          dt == ScalarType::Int64)) {
      PyErr_SetString(
          PyExc_ValueError,
          "advanced index tensor must be int32 or int64 on CUDA");
      throw nb::python_error();
    }
  } else {
    if (!(dt == ScalarType::Int32 ||
          dt == ScalarType::Int64 ||
          dt == ScalarType::Bool)) {
      PyErr_SetString(
          PyExc_ValueError,
          "advanced index tensor must be int32, int64, or bool");
      throw nb::python_error();
    }
  }

#if VBT_WITH_AUTOGRAD
  const bool graph_enabled =
      vbt::autograd::GradMode::is_enabled() &&
      !vbt::autograd::InferenceMode::is_enabled();
  const bool self_requires_grad = vbt::autograd::requires_grad(self);
  const bool value_requires_grad = vbt::autograd::requires_grad(value);
  const bool use_vt_kernel =
      graph_enabled && (self_requires_grad || value_requires_grad);
#else
  const bool use_vt_kernel = false;
#endif

  if (!use_vt_kernel) {
    // Direct core advanced indexing path mirrors the prefix-based branch
    // above but without involving vt dispatcher/autograd wrappers.
    try {
      index_put_(self, spec, value, /*accumulate=*/false);
      return;
    } catch (const std::invalid_argument& e) {
      std::string msg = e.what();
      if (msg.find("advanced indexing is not supported for 0-d tensors") !=
          std::string::npos) {
        PyErr_SetString(PyExc_RuntimeError, msg.c_str());
        throw nb::python_error();
      }
      PyErr_SetString(PyExc_ValueError, msg.c_str());
      throw nb::python_error();
    } catch (const std::runtime_error& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      throw nb::python_error();
    }
  }

  TensorImpl meta;
  try {
    meta = encode_index_meta_tensor(IndexSpec{});
  } catch (const std::invalid_argument& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    throw nb::python_error();
  }

  // 0-d Bool CPU tensor encoding accumulate=False.
  std::vector<std::int64_t> acc_sizes;
  TensorImpl like_for_acc = self;
  if (dev_type != kDLCPU) {
    like_for_acc = TensorImpl(
        vbt::core::StoragePtr{},
        std::vector<std::int64_t>{},
        std::vector<std::int64_t>{},
        /*storage_offset=*/0,
        ScalarType::Bool,
        vbt::core::Device::cpu());
  }
  TensorImpl acc =
      make_contiguous_out_cpu(like_for_acc, acc_sizes, ScalarType::Bool);
  if (acc.numel() > 0) {
    auto* p = static_cast<std::uint8_t*>(acc.data());
    *p = 0u;
  }

  vbt::dispatch::BoxedStack stack;
  stack.push_back(base);
  stack.push_back(index_tensor);
  stack.push_back(value);
  stack.push_back(meta);
  stack.push_back(acc);

  try {
    vbt::dispatch::Dispatcher::instance().callBoxed("vt::index_put", stack);
  } catch (const std::invalid_argument& e) {
    std::string msg = e.what();
    if (msg.find("advanced indexing is only implemented for CPU and CUDA tensors") !=
        std::string::npos) {
      PyErr_SetString(PyExc_NotImplementedError, msg.c_str());
      throw nb::python_error();
    }
    PyErr_SetString(PyExc_ValueError, msg.c_str());
    throw nb::python_error();
  } catch (const std::runtime_error& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    throw nb::python_error();
  }
}

static void Tensor_index_put(TensorImpl& self,
                             nb::handle index_obj,
                             const TensorImpl& value,
                             bool accumulate) {
  using vbt::core::indexing::index_put_;

  const auto dev_type = self.device().type;
  if (dev_type != kDLCPU && dev_type != kDLCUDA) {
    PyErr_SetString(
        PyExc_NotImplementedError,
        "Tensor.index_put_ is only implemented for CPU and CUDA tensors");
    throw nb::python_error();
  }

  if (value.device().type != self.device().type ||
      value.device().index != self.device().index ||
      value.dtype() != self.dtype()) {
    PyErr_SetString(PyExc_ValueError,
                    "index_put_: dtype/device mismatch");
    throw nb::python_error();
  }

  IndexSpec spec = parse_indices_py_allow_advanced(self, index_obj);

  if (!has_any_advanced(spec)) {
    if (dev_type != kDLCPU) {
      PyErr_SetString(
          PyExc_NotImplementedError,
          "Tensor.index_put_ basic indexing is only implemented for CPU tensors");
      throw nb::python_error();
    }

    try {
      index_put_(self, spec, value, /*accumulate=*/accumulate);
      return;
    } catch (const std::invalid_argument& e) {
      std::string msg = e.what();
      PyErr_SetString(PyExc_ValueError, msg.c_str());
      throw nb::python_error();
    } catch (const std::runtime_error& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      throw nb::python_error();
    }
  }

  // Advanced path: scalar-bool advanced indices (IndexKind::Boolean)
  // across reads and writes.
  if (has_any_advanced(spec)) {
    const std::int64_t n_items =
        static_cast<std::int64_t>(spec.items.size());
    for (std::int64_t i = 0; i < n_items; ++i) {
      const auto kind = spec.items[static_cast<std::size_t>(i)].kind;
      if (kind == IndexKind::Boolean) {
        PyErr_SetString(
            PyExc_NotImplementedError,
            "advanced indexing pattern is not supported");
        throw nb::python_error();
      }
    }
  }

  if (!vbt::core::indexing::advanced_indexing_enabled()) {
    PyErr_SetString(PyExc_RuntimeError,
                    vbt::core::indexing::errors::kErrAdvDisabledCore);
    throw nb::python_error();
  }


  const std::int64_t rank =
      static_cast<std::int64_t>(self.sizes().size());
  if (rank == 0) {
    // 0-d base with any advanced indices: call core index_put_ directly
    try {
      index_put_(self, spec, value, accumulate);
      return;
    } catch (const std::invalid_argument& e) {
      std::string msg = e.what();
      if (msg.find("advanced indexing is not supported for 0-d tensors") !=
          std::string::npos) {
        PyErr_SetString(PyExc_RuntimeError, msg.c_str());
        throw nb::python_error();
      }
      PyErr_SetString(PyExc_ValueError, msg.c_str());
      throw nb::python_error();
    } catch (const std::runtime_error& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      throw nb::python_error();
    }
  }

#if VBT_WITH_AUTOGRAD
  const bool graph_enabled =
      vbt::autograd::GradMode::is_enabled() &&
      !vbt::autograd::InferenceMode::is_enabled();
  const bool self_requires_grad = vbt::autograd::requires_grad(self);
  const bool value_requires_grad = vbt::autograd::requires_grad(value);
  const bool use_vt_kernel =
      graph_enabled && (self_requires_grad || value_requires_grad);

  if (use_vt_kernel) {
    // Restrict autograd-enabled advanced writes to the vt::index_put representable
    // subset: one tensor index, optional basic prefix, and no suffix basics.
    using vbt::core::indexing::is_advanced_kind;

    int first_adv = -1;
    int adv_count = 0;
    const std::int64_t n_items =
        static_cast<std::int64_t>(spec.items.size());
    for (std::int64_t i = 0; i < n_items; ++i) {
      const auto kind = spec.items[static_cast<std::size_t>(i)].kind;
      if (is_advanced_kind(kind)) {
        if (first_adv < 0) {
          first_adv = static_cast<int>(i);
        }
        ++adv_count;
      }
    }

    if (first_adv < 0 || adv_count != 1) {
      PyErr_SetString(
          PyExc_RuntimeError,
          vbt::core::indexing::errors::kErrIndexPutAutogradUnsupported);
      throw nb::python_error();
    }

    const TensorIndex& adv_it =
        spec.items[static_cast<std::size_t>(first_adv)];
    if (adv_it.kind != IndexKind::Tensor || !adv_it.tensor.storage().get()) {
      PyErr_SetString(
          PyExc_RuntimeError,
          vbt::core::indexing::errors::kErrIndexPutAutogradUnsupported);
      throw nb::python_error();
    }

    const bool allow_suffix_full_slices = (dev_type == kDLCPU);

    // Reject suffix indices after the advanced block for autograd-enabled
    // vt::index_put. Trailing full-range slices are allowed on CPU only
    // (they are equivalent to omitted dimensions); CUDA keeps the stricter
    // no-suffix policy to match core behavior and maintain stable error surfaces.
    for (int i = first_adv + 1; i < static_cast<int>(n_items); ++i) {
      const auto& it = spec.items[static_cast<std::size_t>(i)];
      const auto kind = it.kind;
      if (is_advanced_kind(kind) ||
          kind == IndexKind::Integer ||
          kind == IndexKind::None) {
        PyErr_SetString(
            PyExc_RuntimeError,
            vbt::core::indexing::errors::kErrIndexPutAutogradUnsupported);
        throw nb::python_error();
      }
      if (kind == IndexKind::Slice) {
        const auto& s = it.slice;
        if (allow_suffix_full_slices &&
            !s.start.has_value() && !s.stop.has_value() && !s.step.has_value()) {
          continue;
        }
        PyErr_SetString(
            PyExc_RuntimeError,
            vbt::core::indexing::errors::kErrIndexPutAutogradUnsupported);
        throw nb::python_error();
      }
    }

    const ScalarType dt = adv_it.tensor.dtype();
    if (!(dt == ScalarType::Int32 || dt == ScalarType::Int64)) {
      PyErr_SetString(
          PyExc_RuntimeError,
          vbt::core::indexing::errors::kErrIndexPutAutogradUnsupported);
      throw nb::python_error();
    }

    IndexSpec prefix_spec;
    prefix_spec.items.assign(
        spec.items.begin(),
        spec.items.begin() + static_cast<std::ptrdiff_t>(first_adv));

    TensorImpl meta;
    try {
      meta = encode_index_meta_tensor(prefix_spec);
    } catch (const std::invalid_argument& e) {
      PyErr_SetString(PyExc_ValueError, e.what());
      throw nb::python_error();
    }

    // 0-d Bool CPU tensor encoding accumulate flag.
    std::vector<std::int64_t> acc_sizes;
    TensorImpl like_for_acc = self;
    if (dev_type != kDLCPU) {
      like_for_acc = TensorImpl(
          vbt::core::StoragePtr{},
          std::vector<std::int64_t>{},
          std::vector<std::int64_t>{},
          /*storage_offset=*/0,
          ScalarType::Bool,
          vbt::core::Device::cpu());
    }
    TensorImpl acc_t =
        make_contiguous_out_cpu(like_for_acc, acc_sizes, ScalarType::Bool);
    if (acc_t.numel() > 0) {
      auto* p = static_cast<std::uint8_t*>(acc_t.data());
      *p = accumulate ? 1u : 0u;
    }

    vbt::dispatch::BoxedStack stack;
    stack.push_back(self);
    stack.push_back(adv_it.tensor);
    stack.push_back(value);
    stack.push_back(meta);
    stack.push_back(acc_t);

    try {
      vbt::dispatch::Dispatcher::instance().callBoxed("vt::index_put", stack);
    } catch (const std::invalid_argument& e) {
      PyErr_SetString(PyExc_ValueError, e.what());
      throw nb::python_error();
    } catch (const std::runtime_error& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      throw nb::python_error();
    }

    return;
  }
#endif

  try {
    index_put_(self, spec, value, accumulate);
  } catch (const std::invalid_argument& e) {
    std::string msg = e.what();
    if (msg.find("advanced indexing is not supported for 0-d tensors") !=
        std::string::npos) {
      PyErr_SetString(PyExc_RuntimeError, msg.c_str());
      throw nb::python_error();
    }
    PyErr_SetString(PyExc_ValueError, msg.c_str());
    throw nb::python_error();
  } catch (const std::runtime_error& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    throw nb::python_error();
  }
}

// Helper for CUDA reductions
#if VBT_WITH_CUDA
static vbt::core::TensorImpl dispatch_reduction_cuda(
    const char* opname,
    const vbt::core::TensorImpl& self,
    const nb::object& dim_obj,
    bool keepdim) {
    
    std::vector<int64_t> dims;
    if (!dim_obj.is_none()) {
        if (nb::isinstance<nb::int_>(dim_obj)) {
            dims.push_back(nb::cast<int64_t>(dim_obj));
        } else if (nb::isinstance<nb::tuple>(dim_obj) || nb::isinstance<nb::list>(dim_obj)) {
            for (auto item : nb::cast<nb::sequence>(dim_obj)) {
                dims.push_back(nb::cast<int64_t>(item));
            }
        } else {
            throw nb::type_error("dim must be int, tuple, list, or None");
        }
    }

    if (std::strcmp(opname, "vt::sum") == 0) {
        return ::vbt_cuda_sum_impl(self, dims, keepdim);
    }
    if (std::strcmp(opname, "vt::mean") == 0) {
        return ::vbt_cuda_mean_impl(self, dims, keepdim);
    }
    if (std::strcmp(opname, "vt::min") == 0) {
        return ::vbt_cuda_min_impl(self, dims, keepdim);
    }
    if (std::strcmp(opname, "vt::max") == 0) {
        return ::vbt_cuda_max_impl(self, dims, keepdim);
    }
    if (std::strcmp(opname, "vt::prod") == 0) {
        return ::vbt_cuda_prod_impl(self, dims, keepdim);
    }
    throw std::runtime_error("unknown reduction op");
}
#endif

void bind_tensor(nb::module_& m) {
  using vbt::core::TensorImpl;
  using vbt::core::ScalarType;
  using vbt::core::Device;

  nb::enum_<vbt::core::MemoryFormat>(m, "MemoryFormat")
      .value("contiguous", vbt::core::MemoryFormat::Contiguous)
      .value("channels_last", vbt::core::MemoryFormat::ChannelsLast)
      .value("preserve", vbt::core::MemoryFormat::Preserve);

  nb::class_<TensorImpl>(m, "Tensor", nb::dynamic_attr())
      .def_prop_ro("sizes", [](const TensorImpl& t){ return to_tuple(t.sizes()); })
      .def_prop_ro("strides", [](const TensorImpl& t){ return to_tuple(t.strides()); })
      .def_prop_ro("storage_offset", [](const TensorImpl& t){ return t.storage_offset(); })
      .def_prop_ro("dtype", [](const TensorImpl& t){ return nb::str(dtype_name(t.dtype())); })
      .def_prop_ro("device", [](const TensorImpl& t){ return nb::make_tuple(int(t.device().type), int(t.device().index)); })
      .def("is_contiguous", [](const TensorImpl& self) {
          return self.is_contiguous();
      })
      .def("is_contiguous",
           [](const TensorImpl& self, nb::object memory_format) {
             const auto fmt = parse_memory_format(memory_format);
             return is_contiguous_for_format(self, fmt);
           },
           nb::arg("memory_format").none(true))
      .def("contiguous",
           [](const TensorImpl& self, nb::object memory_format) {
             const auto fmt = parse_memory_format(memory_format);
             if (is_contiguous_for_format(self, fmt)) {
               return self;
             }
             if (fmt == vbt::core::MemoryFormat::ChannelsLast) {
               PyErr_SetString(
                   PyExc_NotImplementedError,
                   "contiguous(memory_format='channels_last'): conversion is not implemented");
               throw nb::python_error();
             }
             if (self.device().type == kDLCUDA) {
#if VBT_WITH_CUDA
               if (!(self.dtype() == ScalarType::Float32 ||
                     self.dtype() == ScalarType::Float64 ||
                     self.dtype() == ScalarType::Int64 ||
                     self.dtype() == ScalarType::Complex64 ||
                     self.dtype() == ScalarType::Complex128)) {
                 PyErr_SetString(PyExc_NotImplementedError,
                                 vbt::core::kCloneCudaDtypeAllowlistMsg);
                 throw nb::python_error();
               }
#else
               PyErr_SetString(PyExc_RuntimeError, "CUDA not built");
               throw nb::python_error();
#endif
             }
             return vbt::core::clone_contiguous_same_device(self);
           },
           nb::arg("memory_format").none(true) = nb::none())
      .def("is_non_overlapping_and_dense", &TensorImpl::is_non_overlapping_and_dense)
      .def("version", &TensorImpl::version)
      .def("as_strided", [](const TensorImpl& self, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, int64_t storage_offset){
          auto out = self.as_strided(sizes, strides, storage_offset);
#if VBT_WITH_AUTOGRAD
          vbt::autograd::as_view(self, out);
#endif
          return out;
      })
      .def("__getitem__", [](const TensorImpl& self, nb::object index){
          return Tensor_getitem(self, index);
      }, nb::arg("index").none(true))
      .def("index", [](const TensorImpl& self, nb::args indices){
          // Pure alias for __getitem__; Tensor_getitem already implements
          // all basic/advanced routing and feature-flag gating.
          return Tensor_getitem(self, indices);
      })
      .def("__setitem__", [](TensorImpl& self,
                              nb::object index,
                              nb::object value){
          Tensor_setitem(self, index, value);
      }, nb::arg("index").none(true), nb::arg("value"))
      .def("index_put_", [](TensorImpl& self,
                              nb::object index,
                              const TensorImpl& value,
                              bool accumulate){
          Tensor_index_put(self, index, value, accumulate);
          return &self;
      }, nb::arg("index").none(true), nb::arg("value"), nb::arg("accumulate") = false, nb::rv_policy::reference)
      .def("select", [](const TensorImpl& self, int64_t dim, int64_t index){
          auto out = vbt::core::select(self, dim, index);
#if VBT_WITH_AUTOGRAD
          vbt::autograd::as_view(self, out);
#endif
          return out;
      })
      .def("narrow", [](const TensorImpl& self, int64_t dim, int64_t start, int64_t length){
          auto out = vbt::core::narrow(self, dim, start, length);
#if VBT_WITH_AUTOGRAD
          vbt::autograd::as_view(self, out);
#endif
          return out;
      })
      .def("squeeze", [](const TensorImpl& self){
          auto out = vbt::core::squeeze(self);
#if VBT_WITH_AUTOGRAD
          vbt::autograd::as_view(self, out);
#endif
          return out;
      })
      .def("squeeze", [](const TensorImpl& self, int64_t dim){
          auto out = vbt::core::squeeze(self, dim);
#if VBT_WITH_AUTOGRAD
          vbt::autograd::as_view(self, out);
#endif
          return out;
      })
      .def("squeeze", [](const TensorImpl& self, const std::vector<int64_t>& dims){ auto out = vbt::core::squeeze(self, dims); 
#if VBT_WITH_AUTOGRAD
          vbt::autograd::as_view(self, out);
#endif
          return out; })
      .def("unsqueeze", [](const TensorImpl& self, int64_t dim){
          auto out = vbt::core::unsqueeze(self, dim);
#if VBT_WITH_AUTOGRAD
          vbt::autograd::as_view(self, out);
#endif
          return out;
      })
      .def("permute", [](const TensorImpl& self, const std::vector<int64_t>& dims){
          auto out = vbt::core::permute(self, dims);
#if VBT_WITH_AUTOGRAD
          vbt::autograd::as_view(self, out);
#endif
          return out;
      })
      .def("transpose", [](const TensorImpl& self, int64_t d0, int64_t d1){
          auto out = vbt::core::transpose(self, d0, d1);
#if VBT_WITH_AUTOGRAD
          vbt::autograd::as_view(self, out);
#endif
          return out;
      })
      .def("expand", [](const TensorImpl& self, const std::vector<int64_t>& sizes){
          auto out = vbt::core::expand(self, sizes);
#if VBT_WITH_AUTOGRAD
          vbt::autograd::as_view(self, out);
#endif
          return out;
      })
      .def("view", [](const TensorImpl& self, const std::vector<int64_t>& sizes){
          auto out = vbt::core::view(self, sizes);
#if VBT_WITH_AUTOGRAD
          vbt::autograd::as_view(self, out);
#endif
          return out;
      })
      .def("reshape", [](const TensorImpl& self, const std::vector<int64_t>& sizes){
          auto out = vbt::core::reshape(self, sizes);
#if VBT_WITH_AUTOGRAD
          if (out.storage().get() == self.storage().get() &&
              out.device().type == self.device().type &&
              out.device().index == self.device().index) {
            vbt::autograd::as_view(self, out);
          }
#endif
          return out;
      })
      .def("view_as_real", [](const TensorImpl& self){
          auto out = vbt::core::view_as_real(self);
#if VBT_WITH_AUTOGRAD
          vbt::autograd::as_view(self, out);
#endif
          return out;
      })
      .def("view_as_complex", [](const TensorImpl& self){
          if (!complex_enabled_from_env()) {
            throw nb::type_error(kErrComplexDisabled);
          }
          auto out = vbt::core::view_as_complex(self);
#if VBT_WITH_AUTOGRAD
          vbt::autograd::as_view(self, out);
#endif
          return out;
      })
      .def("conj", [](const TensorImpl& self){
          auto out = vbt::core::conj(self);
#if VBT_WITH_AUTOGRAD
          vbt::autograd::as_view(self, out);
#endif
          return out;
      })
      .def("resolve_conj", [](const TensorImpl& self){
          return vbt::core::resolve_conj(self);
      })
      .def("clone", [](const TensorImpl& self){
          if (self.device().type == kDLCUDA) {
#if VBT_WITH_CUDA
            if (!(self.dtype() == ScalarType::Float32 ||
                  self.dtype() == ScalarType::Float64 ||
                  self.dtype() == ScalarType::Int64 ||
                  self.dtype() == ScalarType::Complex64 ||
                  self.dtype() == ScalarType::Complex128)) {
              PyErr_SetString(PyExc_NotImplementedError, vbt::core::kCloneCudaDtypeAllowlistMsg); throw nb::python_error();
            }
            return vbt::core::clone_cuda(self);
#else
            PyErr_SetString(PyExc_RuntimeError, "CUDA not built"); throw nb::python_error();
#endif
          }
          // CPU clone supports arbitrary strided inputs in P1
          return vbt::core::clone_cpu(self);
      })
      .def("sum", [](const TensorImpl& self,
                     nb::object dim,
                     bool keepdim) {
          if (self.device().type == kDLCUDA) {
#if VBT_WITH_CUDA
              return dispatch_reduction_cuda("vt::sum", self, dim, keepdim);
#else
              throw std::runtime_error("CUDA not built");
#endif
          }
          return reduce_value_cpu(self, dim, keepdim,
                                  ReduceKind::Sum,
                                  "sum");
      }, nb::arg("dim").none(true)=nb::none(), nb::arg("keepdim")=false)
      .def("mean", [](const TensorImpl& self,
                      nb::object dim,
                      bool keepdim) {
          if (self.device().type == kDLCUDA) {
#if VBT_WITH_CUDA
              return dispatch_reduction_cuda("vt::mean", self, dim, keepdim);
#else
              throw std::runtime_error("CUDA not built");
#endif
          }
          return reduce_value_cpu(self, dim, keepdim,
                                  ReduceKind::Mean,
                                  "mean");
      }, nb::arg("dim").none(true)=nb::none(), nb::arg("keepdim")=false)
      .def("amin", [](const TensorImpl& self,
                      nb::object dim,
                      bool keepdim) {
          if (self.device().type == kDLCUDA) {
#if VBT_WITH_CUDA
              return dispatch_reduction_cuda("vt::min", self, dim, keepdim);
#else
              throw std::runtime_error("CUDA not built");
#endif
          }
          return reduce_value_cpu(self, dim, keepdim,
                                  ReduceKind::Amin,
                                  "amin");
      }, nb::arg("dim").none(true)=nb::none(), nb::arg("keepdim")=false)
      .def("amax", [](const TensorImpl& self,
                      nb::object dim,
                      bool keepdim) {
          if (self.device().type == kDLCUDA) {
#if VBT_WITH_CUDA
              return dispatch_reduction_cuda("vt::max", self, dim, keepdim);
#else
              throw std::runtime_error("CUDA not built");
#endif
          }
          return reduce_value_cpu(self, dim, keepdim,
                                  ReduceKind::Amax,
                                  "amax");
      }, nb::arg("dim").none(true)=nb::none(), nb::arg("keepdim")=false)
      .def("argmax", [](const TensorImpl& self,
                        nb::object dim,
                        bool keepdim) {
          auto pair = reduce_arg_value_index_cpu(self, dim, keepdim,
                                                 /*is_max=*/true,
                                                 "argmax");
          return pair.second;
      }, nb::arg("dim").none(true)=nb::none(), nb::arg("keepdim")=false)
      .def("argmin", [](const TensorImpl& self,
                        nb::object dim,
                        bool keepdim) {
          auto pair = reduce_arg_value_index_cpu(self, dim, keepdim,
                                                 /*is_max=*/false,
                                                 "argmin");
          return pair.second;
      }, nb::arg("dim").none(true)=nb::none(), nb::arg("keepdim")=false)
      .def("max", [](const TensorImpl& self,
                     nb::object dim,
                     bool keepdim) -> nb::object {
          if (self.device().type == kDLCUDA) {
#if VBT_WITH_CUDA
              return nb::cast(dispatch_reduction_cuda("vt::max", self, dim, keepdim));
#else
              throw std::runtime_error("CUDA not built");
#endif
          }
          if (dim.is_none()) {
              return nb::cast(reduce_value_cpu(self, dim, keepdim, ReduceKind::Amax, "max"));
          }
          auto pair = reduce_arg_value_index_cpu(self, dim, keepdim,
                                                 /*is_max=*/true,
                                                 "max");
          return nb::make_tuple(pair.first, pair.second);
      }, nb::arg("dim").none(true)=nb::none(), nb::arg("keepdim")=false)
      .def("min", [](const TensorImpl& self,
                     nb::object dim,
                     bool keepdim) -> nb::object {
          if (self.device().type == kDLCUDA) {
#if VBT_WITH_CUDA
              return nb::cast(dispatch_reduction_cuda("vt::min", self, dim, keepdim));
#else
              throw std::runtime_error("CUDA not built");
#endif
          }
          if (dim.is_none()) {
              return nb::cast(reduce_value_cpu(self, dim, keepdim, ReduceKind::Amin, "min"));
          }
          auto pair = reduce_arg_value_index_cpu(self, dim, keepdim,
                                                 /*is_max=*/false,
                                                 "min");
          return nb::make_tuple(pair.first, pair.second);
      }, nb::arg("dim").none(true)=nb::none(), nb::arg("keepdim")=false)
      .def("prod", [](const TensorImpl& self,
                      nb::object dim,
                      bool keepdim) {
          if (self.device().type == kDLCUDA) {
#if VBT_WITH_CUDA
              return dispatch_reduction_cuda("vt::prod", self, dim, keepdim);
#else
              throw std::runtime_error("CUDA not built");
#endif
          }
           throw std::runtime_error("prod not implemented for CPU");
      }, nb::arg("dim").none(true)=nb::none(), nb::arg("keepdim")=false)
      .def("fill_", [](TensorImpl& self, nb::object /*value*/){
          vbt::core::check_writable(self);
#if VBT_WITH_AUTOGRAD
          if (vbt::autograd::GradMode::is_enabled() && vbt::autograd::requires_grad(self) && (vbt::autograd::is_view(self) || !vbt::autograd::is_leaf(self))) {
            std::vector<vbt::autograd::SavedVariable> snaps; snaps.emplace_back(self);
            auto node = vbt::autograd::build_inplace_backward_node("vt::fill_", snaps);
            vbt::autograd::rebase_history(self, node);
          }
#endif
                    if (!self.is_contiguous()) {
            vbt::core::for_each_1out_inplace(self, [](std::uint8_t*){});
          }
          if (self.numel() > 0) self.bump_version();
          return &self;
      }, nb::rv_policy::reference)
      .def("add_", [](TensorImpl& self, const TensorImpl& other){
          // CPU path: mirror guard precedence with CUDA
          if (self.sizes() != other.sizes() || self.dtype() != other.dtype() || self.device() != other.device()) {
            PyErr_SetString(PyExc_ValueError, "add_: dtype/device/size mismatch"); throw nb::python_error();
          }
          vbt::core::check_writable(self);
          vbt::core::assert_no_partial_overlap(self, other);
#if VBT_WITH_AUTOGRAD
          if (vbt::autograd::GradMode::is_enabled() && vbt::autograd::requires_grad(self) && (vbt::autograd::is_view(self) || !vbt::autograd::is_leaf(self))) {
            std::vector<vbt::autograd::SavedVariable> snaps; snaps.emplace_back(self); snaps.emplace_back(other);
            auto node = vbt::autograd::build_inplace_backward_node("vt::add", snaps);
            vbt::autograd::rebase_history(self, node);
          }
#endif
          if (self.device().type == kDLCUDA) {
#if VBT_WITH_CUDA
            if (self.dtype() == ScalarType::Complex64 || self.dtype() == ScalarType::Complex128) {
              PyErr_SetString(PyExc_NotImplementedError, "add_: complex in-place not supported on CUDA"); throw nb::python_error();
            }
            if (!self.is_non_overlapping_and_dense() || !other.is_contiguous()) {
              PyErr_SetString(PyExc_NotImplementedError, "add_: non-contiguous not supported on CUDA"); throw nb::python_error();
            }
            // Clear any stale CUDA error prior to kernel launch via dispatcher
            (void)cudaGetLastError();
            // Compute out-of-place on CUDA backend
            vbt::dispatch::BoxedStack s{self, other};
            vbt::dispatch::Dispatcher::instance().callBoxed("vt::add", s);
            // Check for config-time launch failure
            cudaError_t lc = cudaGetLastError();
            if (lc != cudaSuccess) {
              const char* msg = cudaGetErrorString(lc);
              std::string m = "add_: kernel launch failed: "; m += (msg ? msg : "");
              PyErr_SetString(PyExc_RuntimeError, m.c_str()); throw nb::python_error();
            }
            const auto& out = s[0];
            const std::size_t nbytes = static_cast<std::size_t>(self.itemsize()) * static_cast<std::size_t>(self.numel());
            auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(self.device().index));
            cudaError_t st = cudaMemcpyAsync(self.data(), out.data(), nbytes, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream.handle()));
            if (st != cudaSuccess) {
              const char* msg = cudaGetErrorString(st);
              std::string m = "add_: cudaMemcpyAsync failed: "; m += (msg ? msg : "");
              PyErr_SetString(PyExc_RuntimeError, m.c_str()); throw nb::python_error();
            }
            if (self.numel() > 0) self.bump_version();
            return &self;
#else
            PyErr_SetString(PyExc_RuntimeError, "CUDA not built"); throw nb::python_error();
#endif
          }
          // CPU path: use Tensor Iterator for Float32/Int64 and preserve the
          // previous fallback behavior for other dtypes.
          if (self.device().type != kDLCUDA) {
            using vbt::core::MemOverlapStatus;

            // For dtypes not handled by TI yet, keep the old behavior.
            if (!(self.dtype() == ScalarType::Float32 ||
                  self.dtype() == ScalarType::Int64)) {
              if (self.is_contiguous() && other.is_contiguous()) {
                if (self.numel() > 0) self.bump_version();
                return &self;
              }
              auto lap = vbt::core::get_overlap_status(self, other);
              if (lap == MemOverlapStatus::TooHard) {
                PyErr_SetString(
                    PyExc_NotImplementedError,
                    "add_: non-overlapping-dense required for alias safety; use out-of-place or clone()");
                throw nb::python_error();
              }
              if (lap == MemOverlapStatus::Full) {
                vbt::core::for_each_1out_inplace(self, [](std::uint8_t*){});
                if (self.numel() > 0) self.bump_version();
                return &self;
              }
              // No overlap
              vbt::core::for_each_1out_1in(
                  self, other,
                  [](std::uint8_t* /*pdst*/, std::uint8_t* /*psrc*/) {});
              if (self.numel() > 0) self.bump_version();
              return &self;
            }

            // TI-backed in-place add for Float32/Int64.
            vbt::core::TensorIterConfig cfg;
            cfg.add_output(
                vbt::core::OptionalTensorImplRef(&self, /*defined=*/true),
                vbt::core::IterOperandRole::ReadWrite);
            cfg.add_input(other);
            cfg.check_mem_overlap(true);
            static const vbt::core::IterAliasInfo kAddInplaceAliases[] = {
                {0, 0, /*is_inplace=*/true, /*is_view=*/false},
            };
            static const vbt::core::IterOpSignature kAddInplaceSignature{
                "vt::add_", kAddInplaceAliases, 1};
            cfg.set_op_signature(&kAddInplaceSignature);

            vbt::core::TensorIter iter = cfg.build();

            if (self.dtype() == ScalarType::Float32) {
              auto loop = [](char** data,
                             const std::int64_t* strides,
                             std::int64_t size,
                             void* /*ctx*/) {
                char* out_base = data[0];
                char* a_base   = data[1];
                const std::int64_t out_stride = strides[0];
                const std::int64_t a_stride   = strides[1];
                for (std::int64_t i = 0; i < size; ++i) {
                  auto* po =
                      reinterpret_cast<float*>(out_base + i * out_stride);
                  const auto* pa =
                      reinterpret_cast<const float*>(a_base + i * a_stride);
                  *po += *pa;
                }
              };
              vbt::core::for_each_cpu(iter, loop, nullptr);
            } else {  // Int64
              auto loop = [](char** data,
                             const std::int64_t* strides,
                             std::int64_t size,
                             void* /*ctx*/) {
                char* out_base = data[0];
                char* a_base   = data[1];
                const std::int64_t out_stride = strides[0];
                const std::int64_t a_stride   = strides[1];
                for (std::int64_t i = 0; i < size; ++i) {
                  auto* po = reinterpret_cast<long long*>(out_base + i * out_stride);
                  const auto* pa =
                      reinterpret_cast<const long long*>(a_base + i * a_stride);
                  *po += *pa;
                }
              };
              vbt::core::for_each_cpu(iter, loop, nullptr);
            }

            if (self.numel() > 0) self.bump_version();
            return &self;
          }
      }, nb::rv_policy::reference)
      .def("mul_", [](TensorImpl& self, const TensorImpl& other){
          // CPU path: mirror guard precedence with CUDA
          if (self.sizes() != other.sizes() || self.dtype() != other.dtype() || self.device() != other.device()) {
            PyErr_SetString(PyExc_ValueError, "mul_: dtype/device/size mismatch"); throw nb::python_error();
          }
          vbt::core::check_writable(self);
          vbt::core::assert_no_partial_overlap(self, other);
#if VBT_WITH_AUTOGRAD
          if (vbt::autograd::GradMode::is_enabled() && vbt::autograd::requires_grad(self) && (vbt::autograd::is_view(self) || !vbt::autograd::is_leaf(self))) {
            std::vector<vbt::autograd::SavedVariable> snaps; snaps.emplace_back(self); snaps.emplace_back(other);
            auto node = vbt::autograd::build_inplace_backward_node("vt::mul", snaps);
            vbt::autograd::rebase_history(self, node);
          }
#endif
          if (self.device().type == kDLCUDA) {
#if VBT_WITH_CUDA
            if (self.dtype() == ScalarType::Complex64 || self.dtype() == ScalarType::Complex128) {
              PyErr_SetString(PyExc_NotImplementedError, "mul_: complex in-place not supported on CUDA"); throw nb::python_error();
            }
            if (!self.is_non_overlapping_and_dense() || !other.is_contiguous()) {
              PyErr_SetString(PyExc_NotImplementedError, "mul_: non-contiguous not supported on CUDA"); throw nb::python_error();
            }
            // Clear any stale CUDA error prior to kernel launch via dispatcher
            (void)cudaGetLastError();
            vbt::dispatch::BoxedStack s{self, other};
            vbt::dispatch::Dispatcher::instance().callBoxed("vt::mul", s);
            cudaError_t lc = cudaGetLastError();
            if (lc != cudaSuccess) {
              const char* msg = cudaGetErrorString(lc);
              std::string m = "mul_: kernel launch failed: "; m += (msg ? msg : "");
              PyErr_SetString(PyExc_RuntimeError, m.c_str()); throw nb::python_error();
            }
            const auto& out = s[0];
            const std::size_t nbytes = static_cast<std::size_t>(self.itemsize()) * static_cast<std::size_t>(self.numel());
            auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(self.device().index));
            cudaError_t st = cudaMemcpyAsync(self.data(), out.data(), nbytes, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream.handle()));
            if (st != cudaSuccess) {
              const char* msg = cudaGetErrorString(st);
              std::string m = "mul_: cudaMemcpyAsync failed: "; m += (msg ? msg : "");
              PyErr_SetString(PyExc_RuntimeError, m.c_str()); throw nb::python_error();
            }
            if (self.numel() > 0) self.bump_version();
            return &self;
#else
            PyErr_SetString(PyExc_RuntimeError, "CUDA not built"); throw nb::python_error();
#endif
          }
          if (self.is_contiguous() && other.is_contiguous()) {
            if (self.numel() > 0) self.bump_version();
            return &self;
          }
          auto lap = vbt::core::get_overlap_status(self, other);
          using vbt::core::MemOverlapStatus;
          if (lap == MemOverlapStatus::TooHard) {
            PyErr_SetString(PyExc_NotImplementedError, "mul_: non-overlapping-dense required for alias safety; use out-of-place or clone()");
            throw nb::python_error();
          }
          if (lap == MemOverlapStatus::Full) {
            vbt::core::for_each_1out_inplace(self, [](std::uint8_t*){});
            if (self.numel() > 0) self.bump_version();
            return &self;
          }
          vbt::core::for_each_1out_1in(self, other, [](std::uint8_t* /*pdst*/, std::uint8_t* /*psrc*/){ });
          if (self.numel() > 0) self.bump_version();
          return &self;
      }, nb::rv_policy::reference)
      .def("relu_", [](TensorImpl& self){
          vbt::core::check_writable(self);
#if VBT_WITH_AUTOGRAD
          if (vbt::autograd::GradMode::is_enabled() && vbt::autograd::requires_grad(self) && (vbt::autograd::is_view(self) || !vbt::autograd::is_leaf(self))) {
            std::vector<vbt::autograd::SavedVariable> snaps; snaps.emplace_back(self);
            auto node = vbt::autograd::build_inplace_backward_node("vt::relu", snaps);
            vbt::autograd::rebase_history(self, node);
          }
#endif
          if (self.device().type == kDLCUDA) {
#if VBT_WITH_CUDA
            if (!self.is_non_overlapping_and_dense()) { PyErr_SetString(PyExc_NotImplementedError, "relu_: non-contiguous not supported on CUDA"); throw nb::python_error(); }
            // Clear any stale CUDA error prior to kernel launch via dispatcher
            (void)cudaGetLastError();
            vbt::dispatch::BoxedStack s{self};
            vbt::dispatch::Dispatcher::instance().callBoxed("vt::relu", s);
            // Check for config-time launch failure
            cudaError_t lc = cudaGetLastError();
            if (lc != cudaSuccess) {
              const char* msg = cudaGetErrorString(lc);
              std::string m = "relu_: kernel launch failed: "; m += (msg ? msg : "");
              PyErr_SetString(PyExc_RuntimeError, m.c_str()); throw nb::python_error();
            }
            const auto& out = s[0];
            const std::size_t nbytes = static_cast<std::size_t>(self.itemsize()) * static_cast<std::size_t>(self.numel());
            auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(self.device().index));
            cudaError_t st = cudaMemcpyAsync(self.data(), out.data(), nbytes, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream.handle()));
            if (st != cudaSuccess) {
              const char* msg = cudaGetErrorString(st);
              std::string m = "relu_: cudaMemcpyAsync failed: "; m += (msg ? msg : "");
              PyErr_SetString(PyExc_RuntimeError, m.c_str()); throw nb::python_error();
            }
            self.bump_version();
            return &self;
#else
            PyErr_SetString(PyExc_RuntimeError, "CUDA not built"); throw nb::python_error();
#endif
          }
          if (self.device().type != kDLCUDA) {
            // For dtypes not handled by TI yet, keep the old behavior.
            if (self.dtype() != ScalarType::Float32) {
              if (!self.is_contiguous()) {
                vbt::core::for_each_1out_inplace(self, [](std::uint8_t*){});
              }
              if (self.numel() > 0) self.bump_version();
              return &self;
            }

            // TI-backed in-place ReLU for float32.
            vbt::core::TensorIterConfig cfg;
            cfg.add_output(
                vbt::core::OptionalTensorImplRef(&self, /*defined=*/true),
                vbt::core::IterOperandRole::ReadWrite);
            cfg.check_mem_overlap(true);
            static const vbt::core::IterOpSignature kReluInplaceSignature{
                "vt::relu_", nullptr, 0};
            cfg.set_op_signature(&kReluInplaceSignature);
            vbt::core::TensorIter iter = cfg.build();

            auto loop = [](char** data,
                           const std::int64_t* strides,
                           std::int64_t size,
                           void* /*ctx*/) {
              char* out_base = data[0];
              const std::int64_t out_stride = strides[0];
              for (std::int64_t i = 0; i < size; ++i) {
                auto* p =
                    reinterpret_cast<float*>(out_base + i * out_stride);
                const float v = *p;
                *p = v > 0.0f ? v : 0.0f;
              }
            };
            vbt::core::for_each_cpu(iter, loop, nullptr);
            if (self.numel() > 0) self.bump_version();
            return &self;
          }
      }, nb::rv_policy::reference)
      .def("__repr__", [](const TensorImpl& self) -> std::string {
          using vbt::core::ScalarType;
          // Build a formatting view: ensure dense-contiguous on CPU for correct layout
          const TensorImpl* tptr = &self;
          std::optional<TensorImpl> tmp;
          if (self.device().type == kDLCPU && !self.is_contiguous()) {
            tmp = vbt::core::clone_cpu(self);
            tptr = &(*tmp);
          }
          const TensorImpl& t = *tptr;

          // CPU formatting path
          if (t.device().type == kDLCPU) {
            std::vector<size_t> shape(t.sizes().size());
            for (size_t i = 0; i < shape.size(); ++i) shape[i] = (size_t) t.sizes()[i];
            nb::object arr_obj;
            switch (t.dtype()) {
              case ScalarType::Float32: arr_obj = nb::ndarray<nb::numpy, const float>(t.data(), shape.size(), shape.data(), nb::handle()).cast(); break;
              case ScalarType::Int32:   arr_obj = nb::ndarray<nb::numpy, const int32_t>(t.data(), shape.size(), shape.data(), nb::handle()).cast(); break;
              case ScalarType::Int64:   arr_obj = nb::ndarray<nb::numpy, const int64_t>(t.data(), shape.size(), shape.data(), nb::handle()).cast(); break;
              case ScalarType::Bool:    arr_obj = nb::ndarray<nb::numpy, const bool>(t.data(), shape.size(), shape.data(), nb::handle()).cast(); break;
              default: {
                // Fallback: show dtype and size only
                std::string s = "tensor(<unsupported cpu repr>, dtype=";
                s += dtype_name(t.dtype());
                s += ")";
                return s;
              }
            }
            // Format using numpy.array2string to avoid the 'array(...)' wrapper
            nb::object np = nb::module_::import_("numpy");
            nb::object body_obj = np.attr("array2string")(arr_obj);
            std::string body = nb::cast<std::string>(body_obj);

            // Suffix rules
            bool is_empty = (t.numel() == 0);
            bool print_size = is_empty && (t.sizes().size() != 1);
            bool suppress_dtype_non_empty = (!is_empty) && (t.dtype() == ScalarType::Float32 || t.dtype() == ScalarType::Int64 || t.dtype() == ScalarType::Bool);
            bool print_dtype = !suppress_dtype_non_empty;
            if (is_empty) {
              // For empty tensors, only print dtype when dtype != default float32
              print_dtype = (t.dtype() != ScalarType::Float32);
            }
            // Device suffix for CPU: include when default device type is CUDA
            std::string default_dev = "cpu";
            try {
              nb::object mod = nb::module_::import_("vibetensor._C");
              default_dev = nb::cast<std::string>(mod.attr("_get_default_device_type")());
            } catch (...) {
              // Fallback to env if binding unavailable
              const char* env = std::getenv("VBT_DEFAULT_DEVICE_TYPE");
              if (env && ((std::strcmp(env, "cuda") == 0) || (std::strcmp(env, "CUDA") == 0))) default_dev = "cuda";
            }
            bool print_device = (default_dev == "cuda");

            std::string out = "tensor(";
            out += body;
            std::vector<std::string> parts;
            if (print_size) {
              parts.push_back(std::string("size=") + format_sizes(t.sizes()));
            }
            if (print_dtype) {
              parts.push_back(std::string("dtype=") + dtype_name(t.dtype()));
            }
            if (print_device) {
              parts.push_back("device='cpu'");
            }
            if (!parts.empty()) {
              out += ", ";
              for (std::size_t i = 0; i < parts.size(); ++i) {
                if (i) out += ", ";
                out += parts[i];
              }
            }
            out += ")";
            return out;
          }
          // CUDA path: copy to host via binding and format using NumPy
          try {
#if VBT_WITH_CUDA
            // std::fprintf(stderr, "[repr] enter CUDA path\n");
            // Zero-size CUDA tensor: avoid array materialization; synthesize a minimal repr
            if (self.numel() == 0) {
              // std::fprintf(stderr, "[repr] zero-size CUDA tensor path\n");
              std::string out = "tensor([]";
              std::vector<std::string> parts;
              if (self.sizes().size() != 1) {
                parts.push_back(std::string("size=") + format_sizes(self.sizes()));
              }
              // For empty tensors, include dtype unless default float32 (match CPU path rule)
              if (self.dtype() != ScalarType::Float32) {
                parts.push_back(std::string("dtype=") + dtype_name(self.dtype()));
              }
              parts.push_back(std::string("device='cuda:") + std::to_string((int)self.device().index) + "'");
              if (!parts.empty()) {
                out += ", ";
                for (std::size_t i = 0; i < parts.size(); ++i) {
                  if (i) out += ", ";
                  out += parts[i];
                }
              }
              out += ")";
              return out;
            }
            // Dense-contiguous requirement already checked in CUDA binding
            nb::object modC = nb::module_::import_("vibetensor._C");
            // Optional repr logging: measure D2H copy time and bytes when enabled
            nb::object np = nb::module_::import_("numpy");
            nb::object arr_obj;
            if (self.numel() == 0) {
              // Zero-size: build an empty NumPy array directly to avoid null data buffers
              nb::list shape_list;
              for (auto s : self.sizes()) shape_list.append(nb::int_(s));
              arr_obj = np.attr("empty")(nb::tuple(shape_list), nb::str(dtype_name(self.dtype())));
            } else {
              arr_obj = modC.attr("_cuda_d2h_copy_numpy_sync")(self);
            }
            // Format using NumPy
            nb::object body_obj = np.attr("array2string")(arr_obj);
            std::string body = nb::cast<std::string>(body_obj);

            bool is_empty = false; // handled above; always false here
            bool print_size = is_empty && (self.sizes().size() != 1);
            bool suppress_dtype_non_empty = (!is_empty) && (self.dtype() == ScalarType::Float32 || self.dtype() == ScalarType::Int64 || self.dtype() == ScalarType::Bool);
            bool print_dtype = !suppress_dtype_non_empty;
            if (is_empty) {
              // For empty tensors, only print dtype when dtype != default float32
              print_dtype = (self.dtype() != ScalarType::Float32);
            }

            std::string out = "tensor(";
            out += body;
            std::vector<std::string> parts;
            if (print_size) {
              parts.push_back(std::string("size=") + format_sizes(self.sizes()));
            }
            if (print_dtype) {
              parts.push_back(std::string("dtype=") + dtype_name(self.dtype()));
            }
            // Always include device suffix for CUDA tensors
            parts.push_back(std::string("device='cuda:") + std::to_string((int)self.device().index) + "'");
            if (!parts.empty()) {
              out += ", ";
              for (std::size_t i = 0; i < parts.size(); ++i) {
                if (i) out += ", ";
                out += parts[i];
              }
            }
            out += ")";
            return out;
#else
            // CUDA not built: fallback placeholder
            std::string s = "tensor(<cuda>, dtype=";
            s += dtype_name(self.dtype());
            s += ", device='cuda:";
            s += std::to_string((int)self.device().index);
            s += "')";
            return s;
#endif
          } catch (const nb::python_error& e) {
            std::string msg = e.what();
            if (msg.find("unsupported dtype for D2H copy") != std::string::npos) {
              PyErr_SetString(PyExc_RuntimeError, "unsupported dtype for D2H copy");
              throw nb::python_error();
            }
            throw;
          }
      })
      .def("numpy", [](nb::object self_obj) -> nb::object {
          const TensorImpl& self = nb::cast<const TensorImpl&>(self_obj);
          if (self.device().type != kDLCPU) {
            PyErr_SetString(PyExc_TypeError,
                "can't convert cuda:0 device type tensor to numpy. "
                "Use Tensor.cpu() to copy the tensor to host memory first.");
            throw nb::python_error();
          }

          vbt::core::TensorImpl t = self;
          nb::object owner = self_obj;

          if (!self.is_contiguous()) {
             t = vbt::core::clone_cpu(self);
             owner = nb::cast(t);
          }

          std::vector<size_t> shape(t.sizes().size());
          for (size_t i = 0; i < shape.size(); ++i) shape[i] = (size_t) t.sizes()[i];

          using vbt::core::ScalarType;
          nb::object arr_obj;
          switch (t.dtype()) {
              case ScalarType::Float32: arr_obj = nb::ndarray<nb::numpy, const float>(t.data(), shape.size(), shape.data(), nb::handle(owner)).cast(); break;
              case ScalarType::Int32:   arr_obj = nb::ndarray<nb::numpy, const int32_t>(t.data(), shape.size(), shape.data(), nb::handle(owner)).cast(); break;
              case ScalarType::Int64:   arr_obj = nb::ndarray<nb::numpy, const int64_t>(t.data(), shape.size(), shape.data(), nb::handle(owner)).cast(); break;
              case ScalarType::Bool:    arr_obj = nb::ndarray<nb::numpy, const bool>(t.data(), shape.size(), shape.data(), nb::handle(owner)).cast(); break;
              default:
                PyErr_SetString(PyExc_TypeError, "Tensor.numpy(): unsupported dtype");
                throw nb::python_error();
          }
          return arr_obj;
      })
      // __dlpack_device__ and __dlpack__ live here to avoid re-opening the class in another TU
      .def("__dlpack_device__", [](const TensorImpl& self){ return nb::make_tuple(int(self.device().type), int(self.device().index)); })
      .def("__dlpack__", [](const TensorImpl& self, nb::kwargs kwargs){
          if (self.device().type == kDLCPU) {
            // CPU must accept only stream None or -1; ignore other kwargs
            if (kwargs.contains("stream")) {
              nb::object s = kwargs["stream"];
              if (!s.is_none()) {
                long long sval = nb::cast<long long>(s);
                if (sval != -1) {
                  PyErr_SetString(PyExc_AssertionError, "stream should be None on cpu.");
                  throw nb::python_error();
                }
              }
            }
            // Ignore max_version, dl_device, copy on CPU
            return make_dlpack_capsule(self);
          } else if (self.device().type == kDLCUDA) {
            // CUDA path: accept optional stream kwarg but ignore in current implementation
            (void)kwargs;
            return make_dlpack_capsule(self);
          } else {
            PyErr_SetString(PyExc_NotImplementedError, "__dlpack__: unsupported device type");
            throw nb::python_error();
          }
      });
  m.def("_encode_index_spec",
        [](const TensorImpl& self, const TensorImpl& index) {
          try {
            return encode_index_spec_for_tests(self, index);
          } catch (const std::invalid_argument& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
            throw nb::python_error();
          }
        },
        "Test-only helper that returns (index, meta) pairs for vt::index/vt::index_put.");

  // Test-only helper: make a CPU DLPack 1D provider capsule with requested dtype name
  m.def("_make_cpu_dlpack_1d_dtype", [](int64_t n, std::string dtype){
      if (n < 0) throw nb::value_error("n must be >= 0");
      uint8_t code = 0; uint8_t bits = 0; size_t item_b = 0;
      if (dtype == "float16") { code = static_cast<uint8_t>(kDLFloat); bits = 16; item_b = 2; }
      else if (dtype == "bfloat16") {
    #if VBT_HAS_DLPACK_BF16
        code = static_cast<uint8_t>(kDLBfloat); bits = 16; item_b = 2;
    #else
        throw nb::value_error("bfloat16 not supported by DLPack headers");
    #endif
      } else if (dtype == "float32") { code = static_cast<uint8_t>(kDLFloat); bits = 32; item_b = 4; }
      else if (dtype == "int64") { code = static_cast<uint8_t>(kDLInt); bits = 64; item_b = 8; }
      else if (dtype == "bool") { code = static_cast<uint8_t>(kDLBool); bits = 8; item_b = 1; }
      else {
        throw nb::value_error("unsupported dtype");
      }
      void* dptr = nullptr;
      std::size_t nbytes = static_cast<std::size_t>(n) * item_b;
      if (nbytes > 0) {
        dptr = ::operator new(nbytes);
        std::memset(dptr, 0, nbytes);
      }
      DLManagedTensor* mt = new DLManagedTensor{};
      mt->manager_ctx = dptr;
      mt->deleter = [](DLManagedTensor* self){
        if (!self) return;
        void* p = self->manager_ctx;
        if (self->dl_tensor.shape) { delete[] self->dl_tensor.shape; self->dl_tensor.shape = nullptr; }
        if (self->dl_tensor.strides) { delete[] self->dl_tensor.strides; self->dl_tensor.strides = nullptr; }
        if (p) { ::operator delete(p); }
        delete self;
      };
      DLTensor& dl = mt->dl_tensor;
      dl.data = dptr;
      dl.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
      dl.ndim = 1;
      dl.dtype = DLDataType{.code = code, .bits = bits, .lanes = 1};
      dl.shape = new int64_t[1]{n};
      dl.strides = new int64_t[1]{1};
      dl.byte_offset = 0;
      return nb::capsule(mt, "dltensor", [](void* p) noexcept {
        auto* m = reinterpret_cast<DLManagedTensor*>(p);
        if (m && m->deleter) m->deleter(m);
      });
  });

#if VBT_WITH_CUDA
  // Test-only helpers for CUDA tensors and DLPack provider capsules
  m.def("_make_cuda_tensor", [](const std::vector<int64_t>& sizes, std::string dtype, double fill){
      using vbt::core::ScalarType;
      ScalarType st;
      if (dtype == "float32") st = ScalarType::Float32;
      else if (dtype == "int64") st = ScalarType::Int64;
      else if (dtype == "bool") st = ScalarType::Bool;
      else throw nb::value_error("unsupported dtype");
      int64_t N = 1; for (auto s : sizes) { if (s == 0) { N = 0; break; } if (N <= std::numeric_limits<int64_t>::max()/s) N *= s; else N = 0; }
      const std::size_t item_b = static_cast<std::size_t>(vbt::core::itemsize(st));
      std::size_t nbytes = item_b * static_cast<std::size_t>(N);
      int dev = 0; (void)cudaGetDevice(&dev);
      auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
      if (N > 0) {
        auto stream = vbt::cuda::getCurrentStream((vbt::cuda::DeviceIndex) dev);
        auto h2d = [&](const void* src) {
          cudaError_t st_c = cudaMemcpyAsync(
              storage->data(),
              src,
              nbytes,
              cudaMemcpyHostToDevice,
              reinterpret_cast<cudaStream_t>(stream.handle()));
          if (st_c != cudaSuccess) {
            const char* msg = cudaGetErrorString(st_c);
            throw std::runtime_error(std::string("_make_cuda_tensor: cudaMemcpyAsync H2D failed: ") + (msg ? msg : ""));
          }
          cudaError_t st_s = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.handle()));
          if (st_s != cudaSuccess) {
            const char* msg = cudaGetErrorString(st_s);
            throw std::runtime_error(std::string("_make_cuda_tensor: cudaStreamSynchronize failed: ") + (msg ? msg : ""));
          }
        };
        if (st == ScalarType::Float32) {
          std::vector<float> host(static_cast<std::size_t>(N), static_cast<float>(fill));
          h2d(host.data());
        } else if (st == ScalarType::Int64) {
          std::vector<long long> host(static_cast<std::size_t>(N), static_cast<long long>(fill));
          h2d(host.data());
        } else if (st == ScalarType::Bool) {
          std::vector<uint8_t> host(static_cast<std::size_t>(N), (fill != 0.0) ? 1 : 0);
          h2d(host.data());
        }
      }
      // Build contiguous strides
      std::vector<int64_t> strides(sizes.size());
      int64_t acc = 1; for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size())-1; i>=0; --i){ strides[static_cast<std::size_t>(i)] = acc; acc *= (sizes[static_cast<std::size_t>(i)]==0?1:sizes[static_cast<std::size_t>(i)]);} 
      return vbt::core::TensorImpl(storage, sizes, strides, 0, st, vbt::core::Device::cuda(dev));
  });

  m.def("_make_cuda_dlpack_1d", [](int64_t n){
      if (n < 0) throw nb::value_error("n must be >= 0");
      // Allocate device buffer
      int dev = 0; (void)cudaGetDevice(&dev);
      void* dptr = nullptr; std::size_t nbytes = static_cast<std::size_t>(n) * sizeof(float);
      if (nbytes > 0) cudaMalloc(&dptr, nbytes);
      // Fill with a simple sequence on host
      if (nbytes > 0) {
        std::vector<float> host(static_cast<std::size_t>(n));
        for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) host[i] = static_cast<float>(i);
        cudaMemcpy(dptr, host.data(), nbytes, cudaMemcpyHostToDevice);
      }
      // Build DLManagedTensor capsule
      DLManagedTensor* mt = new DLManagedTensor{};
      mt->manager_ctx = dptr; // store pointer for deleter
      mt->deleter = [](DLManagedTensor* self){ if (!self) return; 
        // Delete any shape/strides arrays if present
        if (self->dl_tensor.shape) { delete[] self->dl_tensor.shape; self->dl_tensor.shape = nullptr; }
        if (self->dl_tensor.strides) { delete[] self->dl_tensor.strides; self->dl_tensor.strides = nullptr; }
        void* p = self->manager_ctx; if (p) { cudaFree(p); } delete self; };
      DLTensor& dl = mt->dl_tensor;
      dl.data = dptr;
      dl.device = DLDevice{.device_type = kDLCUDA, .device_id = dev};
      dl.ndim = 1;
      dl.dtype = DLDataType{.code = uint8_t(kDLFloat), .bits = 32, .lanes = 1};
      dl.shape = new int64_t[1]{n};
      dl.strides = new int64_t[1]{1};
      dl.byte_offset = 0;
      return nb::capsule(mt, "dltensor", [](void* p) noexcept {
        auto* m = reinterpret_cast<DLManagedTensor*>(p);
        if (m && m->deleter) m->deleter(m);
      });
  });

  // Test-only helper: CUDA DLPack with requested dtype
  m.def("_make_cuda_dlpack_1d_dtype", [](int64_t n, std::string dtype){
      if (n < 0) throw nb::value_error("n must be >= 0");
      uint8_t code = 0; uint8_t bits = 0; size_t item_b = 0;
      if (dtype == "float16") { code = static_cast<uint8_t>(kDLFloat); bits = 16; item_b = 2; }
      else if (dtype == "bfloat16") {
  #if VBT_HAS_DLPACK_BF16
        code = static_cast<uint8_t>(kDLBfloat); bits = 16; item_b = 2;
  #else
        throw nb::value_error("bfloat16 not supported by DLPack headers");
  #endif
      } else if (dtype == "float32") { code = static_cast<uint8_t>(kDLFloat); bits = 32; item_b = 4; }
      else if (dtype == "int64") { code = static_cast<uint8_t>(kDLInt); bits = 64; item_b = 8; }
      else if (dtype == "bool") { code = static_cast<uint8_t>(kDLBool); bits = 8; item_b = 1; }
      else { throw nb::value_error("unsupported dtype"); }
      int dev = 0; (void)cudaGetDevice(&dev);
      void* dptr = nullptr; std::size_t nbytes = static_cast<std::size_t>(n) * item_b;
      if (nbytes > 0) cudaMalloc(&dptr, nbytes);
      DLManagedTensor* mt = new DLManagedTensor{};
      mt->manager_ctx = dptr;
      mt->deleter = [](DLManagedTensor* self){
        if (!self) return;
        // Free any shape/strides arrays first
        if (self->dl_tensor.shape) { delete[] self->dl_tensor.shape; self->dl_tensor.shape = nullptr; }
        if (self->dl_tensor.strides) { delete[] self->dl_tensor.strides; self->dl_tensor.strides = nullptr; }
        void* p = self->manager_ctx; if (p) { cudaFree(p); }
        delete self;
      };
      DLTensor& dl = mt->dl_tensor;
      dl.data = dptr;
      dl.device = DLDevice{.device_type = kDLCUDA, .device_id = dev};
      dl.ndim = 1;
      dl.dtype = DLDataType{.code = code, .bits = bits, .lanes = 1};
      dl.shape = new int64_t[1]{n};
      dl.strides = new int64_t[1]{1};
      dl.byte_offset = 0;
      return nb::capsule(mt, "dltensor", [](void* p) noexcept {
        auto* m = reinterpret_cast<DLManagedTensor*>(p);
        if (m && m->deleter) m->deleter(m);
      });
  });
#endif
}

} // namespace vbt_py
