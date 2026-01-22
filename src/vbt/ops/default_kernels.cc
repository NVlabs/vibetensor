// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/dispatch/registration.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/complex.h"
#include "vbt/core/device.h"
#include "vbt/core/tensor_iter.h"
#include "vbt/core/view_ops.h"
#include "vbt/core/broadcast.h"
#include "vbt/core/checked_math.h"
#include "vbt/core/tensor_iterator/cpu.h"

#include <new>
#include <stdexcept>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <set>
#include <cstring>

// Minimal default CPU kernels for vt namespace.
// Implement basic value semantics for a small dtype set on CPU; throw for unsupported dtype/device.

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::IterAliasInfo;
using vbt::core::IterOpSignature;
using vbt::core::Complex64;
using vbt::core::Complex128;

// Forward decl for error messages used before its definition.
static std::string dtype_to_string(ScalarType t);

static const IterOpSignature kVtReluSignature{"vt::relu", nullptr, 0};
static const IterOpSignature kVtAddSignature{"vt::add", nullptr, 0};
static const IterOpSignature kVtSubSignature{"vt::sub", nullptr, 0};
static const IterOpSignature kVtMulSignature{"vt::mul", nullptr, 0};
static const IterOpSignature kVtDivSignature{"vt::div", nullptr, 0};
static const IterOpSignature kVtAbsSignature{"vt::abs", nullptr, 0};
static const IterOpSignature kVtNegSignature{"vt::neg", nullptr, 0};
static const IterOpSignature kVtReciprocalSignature{"vt::reciprocal", nullptr, 0};
static const IterOpSignature kVtSumSignature{"vt::sum", nullptr, 0};
static const IterOpSignature kVtMeanSignature{"vt::mean", nullptr, 0};
static const IterOpSignature kVtEqSignature{"vt::eq", nullptr, 0};
static const IterOpSignature kVtNeSignature{"vt::ne", nullptr, 0};
static const IterOpSignature kVtLtSignature{"vt::lt", nullptr, 0};
static const IterOpSignature kVtGtSignature{"vt::gt", nullptr, 0};
static const IterOpSignature kVtLeSignature{"vt::le", nullptr, 0};
static const IterOpSignature kVtGeSignature{"vt::ge", nullptr, 0};

// Helper to compute a contiguous strides vector for a given shape.
static std::vector<int64_t> make_contiguous_strides_from_sizes(
    const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i =
           static_cast<std::ptrdiff_t>(sizes.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const int64_t sz = sizes[idx];
    if (sz == 0) {
      // Keep acc unchanged so later dims don't overflow; numel() will be 0.
      continue;
    }
    acc *= sz;
  }
  return strides;
}

static std::int64_t safe_numel_from_sizes(const std::vector<int64_t>& sizes) {
  if (sizes.empty()) {
    return 1;  // scalar
  }
  std::int64_t n = 1;
  for (int64_t s : sizes) {
    if (s == 0) {
      return 0;
    }
    std::int64_t tmp = 0;
    if (!vbt::core::checked_mul_i64(n, s, tmp)) {
      // Treat overflow as zero to mirror TensorImpl::numel semantics.
      return 0;
    }
    n = tmp;
  }
  return n;
}

static TensorImpl make_empty_like_broadcast_cpu(const TensorImpl& like,
                                                const std::vector<int64_t>& sizes,
                                                ScalarType dtype = ScalarType::Undefined) {
  if (dtype == ScalarType::Undefined) {
    dtype = like.dtype();
  }
  const Device     device = like.device();
  const std::size_t item_b = vbt::core::itemsize(dtype);
  const std::int64_t n = safe_numel_from_sizes(sizes);
  const std::size_t nbytes = static_cast<std::size_t>(n) * item_b;

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
  auto strides = make_contiguous_strides_from_sizes(sizes);
  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0, dtype, device);
}

static TensorImpl vbt_default_unit_impl() {
  // Return a scalar float32 CPU tensor with empty sizes/strides and an owned 1-element storage.
  void* buf = ::operator new(static_cast<std::size_t>(vbt::core::itemsize(ScalarType::Float32)));
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), /*nbytes=*/vbt::core::itemsize(ScalarType::Float32));
  return TensorImpl(storage, /*sizes=*/{}, /*strides=*/{}, /*storage_offset=*/0, ScalarType::Float32, Device::cpu());
}

static TensorImpl vbt_default_relu_impl(const TensorImpl& a) {
  if (a.device().type == kDLCPU && a.dtype() == ScalarType::Float32) {
    std::vector<int64_t> out_sizes = a.sizes();
    TensorImpl out = make_empty_like_broadcast_cpu(a, out_sizes);

    vbt::core::TensorIterConfig cfg;
    cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
    cfg.add_input(a);
    cfg.check_mem_overlap(true);
    cfg.set_op_signature(&kVtReluSignature);
    vbt::core::TensorIter iter = cfg.build();

    auto loop = [](char** data,
                   const std::int64_t* strides,
                   std::int64_t size,
                   void* /*ctx*/) {
      char* out_base = data[0];
      char* a_base   = data[1];
      const std::int64_t out_stride = strides[0];
      const std::int64_t a_stride   = strides[1];
      for (std::int64_t i = 0; i < size; ++i) {
        auto* po = reinterpret_cast<float*>(out_base + i * out_stride);
        const auto* pa =
            reinterpret_cast<const float*>(a_base + i * a_stride);
        const float v = *pa;
        *po = v > 0.0f ? v : 0.0f;
      }
    };
    vbt::core::for_each_cpu(iter, loop, nullptr);
    return out;
  }
  if (a.device().type != kDLCPU) {
    throw std::invalid_argument("vt::relu: unsupported device (expected CPU)");
  }
  throw std::invalid_argument("vt::relu: unsupported dtype " + dtype_to_string(a.dtype()));
}

static TensorImpl vbt_default_add_impl(const TensorImpl& a, const TensorImpl& b) {
  // Validate dtype/device basic compatibility.
  if (a.dtype() != b.dtype()) {
    throw std::invalid_argument("vt::add: dtype mismatch");
  }
  if (a.device() != b.device()) {
    throw std::invalid_argument("vt::add: device mismatch");
  }

  if (a.device().type != kDLCPU) {
    throw std::invalid_argument("vt::add: unsupported device (expected CPU)");
  }

  const ScalarType dtype = a.dtype();

  const TensorImpl* a_in = &a;
  const TensorImpl* b_in = &b;
  TensorImpl a_physical;
  TensorImpl b_physical;
  if (a.is_conj()) {
    a_physical = vbt::core::resolve_conj(a);
    a_in = &a_physical;
  }
  if (b.is_conj()) {
    b_physical = vbt::core::resolve_conj(b);
    b_in = &b_physical;
  }

  if (dtype == ScalarType::Float32 || dtype == ScalarType::Float64 ||
      dtype == ScalarType::Int64 || dtype == ScalarType::Complex64 ||
      dtype == ScalarType::Complex128) {
    // Compute broadcasted output shape and allocate a contiguous output tensor.
    std::vector<int64_t> out_sizes;
    try {
      out_sizes = vbt::core::infer_broadcast_shape(
          std::span<const std::int64_t>(a.sizes()),
          std::span<const std::int64_t>(b.sizes()));
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument(
          std::string("vt::add: inputs must have the same shape or be "
                      "broadcastable: ") +
          e.what());
    }
    TensorImpl out = make_empty_like_broadcast_cpu(a, out_sizes);

    vbt::core::TensorIterConfig cfg;
    cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
    cfg.add_input(*a_in);
    cfg.add_input(*b_in);
    cfg.check_mem_overlap(true);
    cfg.set_op_signature(&kVtAddSignature);
    vbt::core::TensorIter iter = cfg.build();

    if (dtype == ScalarType::Float32) {
      auto loop = [](char** data,
                     const std::int64_t* strides,
                     std::int64_t size,
                     void* /*ctx*/) {
        char* out_base = data[0];
        char* a_base   = data[1];
        char* b_base   = data[2];
        const std::int64_t out_stride = strides[0];
        const std::int64_t a_stride   = strides[1];
        const std::int64_t b_stride   = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          auto* po = reinterpret_cast<float*>(out_base + i * out_stride);
          const auto* pa = reinterpret_cast<const float*>(a_base + i * a_stride);
          const auto* pb = reinterpret_cast<const float*>(b_base + i * b_stride);
          *po = *pa + *pb;
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    } else if (dtype == ScalarType::Float64) {
      auto loop = [](char** data,
                     const std::int64_t* strides,
                     std::int64_t size,
                     void* /*ctx*/) {
        char* out_base = data[0];
        char* a_base   = data[1];
        char* b_base   = data[2];
        const std::int64_t out_stride = strides[0];
        const std::int64_t a_stride   = strides[1];
        const std::int64_t b_stride   = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          auto* po = reinterpret_cast<double*>(out_base + i * out_stride);
          const auto* pa = reinterpret_cast<const double*>(a_base + i * a_stride);
          const auto* pb = reinterpret_cast<const double*>(b_base + i * b_stride);
          *po = *pa + *pb;
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    } else if (dtype == ScalarType::Int64) {
      auto loop = [](char** data,
                     const std::int64_t* strides,
                     std::int64_t size,
                     void* /*ctx*/) {
        char* out_base = data[0];
        char* a_base   = data[1];
        char* b_base   = data[2];
        const std::int64_t out_stride = strides[0];
        const std::int64_t a_stride   = strides[1];
        const std::int64_t b_stride   = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          auto* po = reinterpret_cast<long long*>(out_base + i * out_stride);
          const auto* pa =
              reinterpret_cast<const long long*>(a_base + i * a_stride);
          const auto* pb =
              reinterpret_cast<const long long*>(b_base + i * b_stride);
          *po = *pa + *pb;
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    } else if (dtype == ScalarType::Complex64) {
      auto loop = [](char** data,
                     const std::int64_t* strides,
                     std::int64_t size,
                     void* /*ctx*/) {
        char* out_base = data[0];
        char* a_base   = data[1];
        char* b_base   = data[2];
        const std::int64_t out_stride = strides[0];
        const std::int64_t a_stride   = strides[1];
        const std::int64_t b_stride   = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          auto* po = reinterpret_cast<Complex64*>(out_base + i * out_stride);
          const auto* pa = reinterpret_cast<const Complex64*>(a_base + i * a_stride);
          const auto* pb = reinterpret_cast<const Complex64*>(b_base + i * b_stride);
          po->re = pa->re + pb->re;
          po->im = pa->im + pb->im;
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    } else {  // Complex128
      auto loop = [](char** data,
                     const std::int64_t* strides,
                     std::int64_t size,
                     void* /*ctx*/) {
        char* out_base = data[0];
        char* a_base   = data[1];
        char* b_base   = data[2];
        const std::int64_t out_stride = strides[0];
        const std::int64_t a_stride   = strides[1];
        const std::int64_t b_stride   = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          auto* po = reinterpret_cast<Complex128*>(out_base + i * out_stride);
          const auto* pa = reinterpret_cast<const Complex128*>(a_base + i * a_stride);
          const auto* pb = reinterpret_cast<const Complex128*>(b_base + i * b_stride);
          po->re = pa->re + pb->re;
          po->im = pa->im + pb->im;
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    }

    return out;
  }

  throw std::invalid_argument("vt::add: unsupported dtype " + dtype_to_string(dtype));
}

static TensorImpl vbt_default_mul_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.dtype() != b.dtype()) throw std::invalid_argument("vt::mul: dtype mismatch");
  if (a.device() != b.device()) throw std::invalid_argument("vt::mul: device mismatch");
  if (a.device().type != kDLCPU) {
    throw std::invalid_argument("vt::mul: unsupported device (expected CPU)");
  }

  const ScalarType dtype = a.dtype();

  const TensorImpl* a_in = &a;
  const TensorImpl* b_in = &b;
  TensorImpl a_physical;
  TensorImpl b_physical;
  if (a.is_conj()) {
    a_physical = vbt::core::resolve_conj(a);
    a_in = &a_physical;
  }
  if (b.is_conj()) {
    b_physical = vbt::core::resolve_conj(b);
    b_in = &b_physical;
  }

  if (dtype == ScalarType::Float32 || dtype == ScalarType::Float64 ||
      dtype == ScalarType::Int64 || dtype == ScalarType::Complex64 ||
      dtype == ScalarType::Complex128) {
    std::vector<int64_t> out_sizes;
    try {
      out_sizes = vbt::core::infer_broadcast_shape(
          std::span<const std::int64_t>(a.sizes()),
          std::span<const std::int64_t>(b.sizes()));
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument("vt::mul: broadcast error: " + std::string(e.what()));
    }
    TensorImpl out = make_empty_like_broadcast_cpu(a, out_sizes);

    vbt::core::TensorIterConfig cfg;
    cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
    cfg.add_input(*a_in);
    cfg.add_input(*b_in);
    cfg.check_mem_overlap(true);
    cfg.set_op_signature(&kVtMulSignature);
    vbt::core::TensorIter iter = cfg.build();

    if (dtype == ScalarType::Float32) {
      auto loop = [](char** data,
                     const std::int64_t* strides,
                     std::int64_t size,
                     void* /*ctx*/) {
        char* out_base = data[0];
        char* a_base = data[1];
        char* b_base = data[2];
        const std::int64_t so = strides[0], sa = strides[1], sb = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          *reinterpret_cast<float*>(out_base + i * so) =
              *reinterpret_cast<const float*>(a_base + i * sa) *
              *reinterpret_cast<const float*>(b_base + i * sb);
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    } else if (dtype == ScalarType::Float64) {
      auto loop = [](char** data,
                     const std::int64_t* strides,
                     std::int64_t size,
                     void* /*ctx*/) {
        char* out_base = data[0];
        char* a_base = data[1];
        char* b_base = data[2];
        const std::int64_t so = strides[0], sa = strides[1], sb = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          *reinterpret_cast<double*>(out_base + i * so) =
              *reinterpret_cast<const double*>(a_base + i * sa) *
              *reinterpret_cast<const double*>(b_base + i * sb);
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    } else if (dtype == ScalarType::Int64) {
      auto loop = [](char** data,
                     const std::int64_t* strides,
                     std::int64_t size,
                     void* /*ctx*/) {
        char* out_base = data[0];
        char* a_base = data[1];
        char* b_base = data[2];
        const std::int64_t so = strides[0], sa = strides[1], sb = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          *reinterpret_cast<long long*>(out_base + i * so) =
              *reinterpret_cast<const long long*>(a_base + i * sa) *
              *reinterpret_cast<const long long*>(b_base + i * sb);
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    } else if (dtype == ScalarType::Complex64) {
      auto loop = [](char** data,
                     const std::int64_t* strides,
                     std::int64_t size,
                     void* /*ctx*/) {
        char* out_base = data[0];
        char* a_base = data[1];
        char* b_base = data[2];
        const std::int64_t so = strides[0], sa = strides[1], sb = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          auto* po = reinterpret_cast<Complex64*>(out_base + i * so);
          const auto* pa = reinterpret_cast<const Complex64*>(a_base + i * sa);
          const auto* pb = reinterpret_cast<const Complex64*>(b_base + i * sb);
          const float re = pa->re * pb->re - pa->im * pb->im;
          const float im = pa->re * pb->im + pa->im * pb->re;
          po->re = re;
          po->im = im;
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    } else {  // Complex128
      auto loop = [](char** data,
                     const std::int64_t* strides,
                     std::int64_t size,
                     void* /*ctx*/) {
        char* out_base = data[0];
        char* a_base = data[1];
        char* b_base = data[2];
        const std::int64_t so = strides[0], sa = strides[1], sb = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          auto* po = reinterpret_cast<Complex128*>(out_base + i * so);
          const auto* pa = reinterpret_cast<const Complex128*>(a_base + i * sa);
          const auto* pb = reinterpret_cast<const Complex128*>(b_base + i * sb);
          const double re = pa->re * pb->re - pa->im * pb->im;
          const double im = pa->re * pb->im + pa->im * pb->re;
          po->re = re;
          po->im = im;
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    }

    return out;
  }

  throw std::invalid_argument("vt::mul: unsupported dtype " + dtype_to_string(dtype));
}

static TensorImpl vbt_default_sub_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.dtype() != b.dtype()) throw std::invalid_argument("vt::sub: dtype mismatch");
  if (a.device() != b.device()) throw std::invalid_argument("vt::sub: device mismatch");
  if (a.device().type != kDLCPU) {
    throw std::invalid_argument("vt::sub: unsupported device (expected CPU)");
  }

  const ScalarType dtype = a.dtype();
  if (dtype == ScalarType::Float32 || dtype == ScalarType::Int64) {
    std::vector<int64_t> out_sizes;
    try {
      out_sizes = vbt::core::infer_broadcast_shape(
          std::span<const std::int64_t>(a.sizes()),
          std::span<const std::int64_t>(b.sizes()));
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument("vt::sub: broadcast error: " + std::string(e.what()));
    }
    TensorImpl out = make_empty_like_broadcast_cpu(a, out_sizes);

    vbt::core::TensorIterConfig cfg;
    cfg.add_output(vbt::core::OptionalTensorImplRef(&out, true));
    cfg.add_input(a);
    cfg.add_input(b);
    cfg.check_mem_overlap(true);
    cfg.set_op_signature(&kVtSubSignature);
    vbt::core::TensorIter iter = cfg.build();

    if (dtype == ScalarType::Float32) {
      auto loop = [](char** data, const std::int64_t* strides, std::int64_t size, void*) {
        char* out_base = data[0]; char* a_base = data[1]; char* b_base = data[2];
        const std::int64_t so = strides[0], sa = strides[1], sb = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          *reinterpret_cast<float*>(out_base + i * so) =
              *reinterpret_cast<const float*>(a_base + i * sa) -
              *reinterpret_cast<const float*>(b_base + i * sb);
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    } else {
      auto loop = [](char** data, const std::int64_t* strides, std::int64_t size, void*) {
        char* out_base = data[0]; char* a_base = data[1]; char* b_base = data[2];
        const std::int64_t so = strides[0], sa = strides[1], sb = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          *reinterpret_cast<long long*>(out_base + i * so) =
              *reinterpret_cast<const long long*>(a_base + i * sa) -
              *reinterpret_cast<const long long*>(b_base + i * sb);
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    }
    return out;
  }

  throw std::invalid_argument("vt::sub: unsupported dtype " + dtype_to_string(dtype));
}

static TensorImpl vbt_default_div_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.dtype() != b.dtype()) throw std::invalid_argument("vt::div: dtype mismatch");
  if (a.device() != b.device()) throw std::invalid_argument("vt::div: device mismatch");
  if (a.device().type != kDLCPU) {
    throw std::invalid_argument("vt::div: unsupported device (expected CPU)");
  }

  const ScalarType dtype = a.dtype();
  if (dtype == ScalarType::Float32 || dtype == ScalarType::Int64) {
    std::vector<int64_t> out_sizes;
    try {
      out_sizes = vbt::core::infer_broadcast_shape(
          std::span<const std::int64_t>(a.sizes()),
          std::span<const std::int64_t>(b.sizes()));
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument("vt::div: broadcast error: " + std::string(e.what()));
    }
    TensorImpl out = make_empty_like_broadcast_cpu(a, out_sizes);

    vbt::core::TensorIterConfig cfg;
    cfg.add_output(vbt::core::OptionalTensorImplRef(&out, true));
    cfg.add_input(a);
    cfg.add_input(b);
    cfg.check_mem_overlap(true);
    cfg.set_op_signature(&kVtDivSignature);
    vbt::core::TensorIter iter = cfg.build();

    if (dtype == ScalarType::Float32) {
      auto loop = [](char** data, const std::int64_t* strides, std::int64_t size, void*) {
        char* out_base = data[0]; char* a_base = data[1]; char* b_base = data[2];
        const std::int64_t so = strides[0], sa = strides[1], sb = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          *reinterpret_cast<float*>(out_base + i * so) =
              *reinterpret_cast<const float*>(a_base + i * sa) /
              *reinterpret_cast<const float*>(b_base + i * sb);
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    } else {
      // Integer division
      auto loop = [](char** data, const std::int64_t* strides, std::int64_t size, void*) {
        char* out_base = data[0]; char* a_base = data[1]; char* b_base = data[2];
        const std::int64_t so = strides[0], sa = strides[1], sb = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          long long val_b = *reinterpret_cast<const long long*>(b_base + i * sb);
          if (val_b == 0) throw std::runtime_error("vt::div: division by zero");
          *reinterpret_cast<long long*>(out_base + i * so) =
              *reinterpret_cast<const long long*>(a_base + i * sa) / val_b;
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    }
    return out;
  }

  throw std::invalid_argument("vt::div: unsupported dtype " + dtype_to_string(dtype));
}

static TensorImpl vbt_default_abs_impl(const TensorImpl& a) {
  if (a.device().type != kDLCPU) {
    throw std::invalid_argument("vt::abs: unsupported device (expected CPU)");
  }
  const ScalarType dtype = a.dtype();
  if (dtype == ScalarType::Float32 || dtype == ScalarType::Int64) {
    TensorImpl out = make_empty_like_broadcast_cpu(a, a.sizes());

    vbt::core::TensorIterConfig cfg;
    cfg.add_output(vbt::core::OptionalTensorImplRef(&out, true));
    cfg.add_input(a);
    cfg.check_mem_overlap(true);
    cfg.set_op_signature(&kVtAbsSignature);
    vbt::core::TensorIter iter = cfg.build();

    if (a.dtype() == ScalarType::Float32) {
        auto loop = [](char** data, const std::int64_t* strides, std::int64_t size, void*) {
            char* out_base = data[0]; char* a_base = data[1];
            const std::int64_t so = strides[0], sa = strides[1];
            for (std::int64_t i = 0; i < size; ++i) {
                *reinterpret_cast<float*>(out_base + i * so) =
                    std::abs(*reinterpret_cast<const float*>(a_base + i * sa));
            }
        };
        vbt::core::for_each_cpu(iter, loop, nullptr);
    } else {
        auto loop = [](char** data, const std::int64_t* strides, std::int64_t size, void*) {
            char* out_base = data[0]; char* a_base = data[1];
            const std::int64_t so = strides[0], sa = strides[1];
            for (std::int64_t i = 0; i < size; ++i) {
                *reinterpret_cast<long long*>(out_base + i * so) =
                    std::abs(*reinterpret_cast<const long long*>(a_base + i * sa));
            }
        };
        vbt::core::for_each_cpu(iter, loop, nullptr);
    }
    return out;
  }

  throw std::invalid_argument("vt::abs: unsupported dtype " + dtype_to_string(dtype));
}

static TensorImpl vbt_default_neg_impl(const TensorImpl& a) {
  if (a.device().type != kDLCPU) {
    throw std::invalid_argument("vt::neg: unsupported device (expected CPU)");
  }
  const ScalarType dtype = a.dtype();
  if (dtype == ScalarType::Float32 || dtype == ScalarType::Int64) {
    TensorImpl out = make_empty_like_broadcast_cpu(a, a.sizes());

    vbt::core::TensorIterConfig cfg;
    cfg.add_output(vbt::core::OptionalTensorImplRef(&out, true));
    cfg.add_input(a);
    cfg.check_mem_overlap(true);
    cfg.set_op_signature(&kVtNegSignature);
    vbt::core::TensorIter iter = cfg.build();

    if (a.dtype() == ScalarType::Float32) {
        auto loop = [](char** data, const std::int64_t* strides, std::int64_t size, void*) {
            char* out_base = data[0]; char* a_base = data[1];
            const std::int64_t so = strides[0], sa = strides[1];
            for (std::int64_t i = 0; i < size; ++i) {
                *reinterpret_cast<float*>(out_base + i * so) =
                    -(*reinterpret_cast<const float*>(a_base + i * sa));
            }
        };
        vbt::core::for_each_cpu(iter, loop, nullptr);
    } else {
        auto loop = [](char** data, const std::int64_t* strides, std::int64_t size, void*) {
            char* out_base = data[0]; char* a_base = data[1];
            const std::int64_t so = strides[0], sa = strides[1];
            for (std::int64_t i = 0; i < size; ++i) {
                *reinterpret_cast<long long*>(out_base + i * so) =
                    -(*reinterpret_cast<const long long*>(a_base + i * sa));
            }
        };
        vbt::core::for_each_cpu(iter, loop, nullptr);
    }
    return out;
  }

  throw std::invalid_argument("vt::neg: unsupported dtype " + dtype_to_string(dtype));
}

static TensorImpl vbt_default_reciprocal_impl(const TensorImpl& a) {
  if (a.device().type != kDLCPU) {
    throw std::invalid_argument("vt::reciprocal: unsupported device (expected CPU)");
  }
  const ScalarType dtype = a.dtype();
  if (dtype == ScalarType::Float32 || dtype == ScalarType::Int64) {
    TensorImpl out = make_empty_like_broadcast_cpu(a, a.sizes());

    vbt::core::TensorIterConfig cfg;
    cfg.add_output(vbt::core::OptionalTensorImplRef(&out, true));
    cfg.add_input(a);
    cfg.check_mem_overlap(true);
    cfg.set_op_signature(&kVtReciprocalSignature);
    vbt::core::TensorIter iter = cfg.build();

    if (dtype == ScalarType::Float32) {
      auto loop = [](char** data, const std::int64_t* strides, std::int64_t size, void*) {
        char* out_base = data[0]; char* a_base = data[1];
        const std::int64_t so = strides[0], sa = strides[1];
        for (std::int64_t i = 0; i < size; ++i) {
          float val = *reinterpret_cast<const float*>(a_base + i * sa);
          *reinterpret_cast<float*>(out_base + i * so) = 1.0f / val;
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    } else {
       // Int64 reciprocal: 1/x. 0 for |x|>1, 1 for 1, -1 for -1.
       // And handle 0.
      auto loop = [](char** data, const std::int64_t* strides, std::int64_t size, void*) {
        char* out_base = data[0]; char* a_base = data[1];
        const std::int64_t so = strides[0], sa = strides[1];
        for (std::int64_t i = 0; i < size; ++i) {
          long long val = *reinterpret_cast<const long long*>(a_base + i * sa);
          if (val == 0) {
              *reinterpret_cast<long long*>(out_base + i * so) = 0; // Avoid div by zero
          } else {
              *reinterpret_cast<long long*>(out_base + i * so) = 1 / val;
          }
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    }
    return out;
  }

  throw std::invalid_argument("vt::reciprocal: unsupported dtype " + dtype_to_string(dtype));
}

static TensorImpl make_empty_like_reduction(const TensorImpl& self,
                                            const std::vector<int64_t>& dims,
                                            bool keepdim) {
  const auto& in_sizes = self.sizes();
  const int64_t R = static_cast<int64_t>(in_sizes.size());

  // Normalize dims
  std::vector<int64_t> norm_dims;
  norm_dims.reserve(dims.size());
  for (int64_t d : dims) {
    if (d < 0) d += R;
    if (d < 0 || d >= R) {
        throw std::invalid_argument("vt::sum/mean: dim out of range");
    }
    norm_dims.push_back(d);
  }
  std::sort(norm_dims.begin(), norm_dims.end());
  auto last = std::unique(norm_dims.begin(), norm_dims.end());
  norm_dims.erase(last, norm_dims.end());

  auto is_reduced = [&](int64_t d) {
      for (auto rd : norm_dims) if (rd == d) return true;
      return false;
  };

  std::vector<int64_t> out_sizes;
  if (keepdim) {
      out_sizes.assign(in_sizes.begin(), in_sizes.end());
      for (auto rd : norm_dims) out_sizes[static_cast<size_t>(rd)] = 1;
  } else {
      out_sizes.reserve(static_cast<size_t>(R - static_cast<int64_t>(norm_dims.size())));
      for (int64_t i = 0; i < R; ++i) {
          if (!is_reduced(i)) out_sizes.push_back(in_sizes[static_cast<size_t>(i)]);
      }
  }

  return make_empty_like_broadcast_cpu(self, out_sizes);
}

TensorImpl vbt_default_sum_impl(const TensorImpl& self, std::vector<int64_t> dims, bool keepdim) {
  if (self.device().type != kDLCPU) {
    throw std::invalid_argument("vt::sum: unsupported device (expected CPU)");
  }

  const ScalarType dtype = self.dtype();
  if (dtype != ScalarType::Float32 && dtype != ScalarType::Int64) {
    throw std::invalid_argument("vt::sum: unsupported dtype " + dtype_to_string(dtype));
  }

  if (dims.empty()) {
    // Sum over all dimensions
    dims.resize(self.sizes().size());
    std::iota(dims.begin(), dims.end(), 0);
  }

  TensorImpl out = make_empty_like_reduction(self, dims, keepdim);

  std::size_t nbytes = static_cast<std::size_t>(out.numel()) * vbt::core::itemsize(dtype);
  if (out.data()) {
    std::memset(out.data(), 0, nbytes);
  }

  vbt::core::TensorIterConfig cfg;
  cfg.check_mem_overlap(true);
  cfg.is_reduction(true);
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, true), vbt::core::IterOperandRole::ReduceOutput);
  cfg.add_input(self);
  cfg.set_reduce_dims(dims, keepdim);
  cfg.set_op_signature(&kVtSumSignature);
  vbt::core::TensorIter iter = cfg.build();

  if (dtype == ScalarType::Float32) {
      auto loop = [](char** data, const std::int64_t* strides, std::int64_t size, void*) {
          char* out_ptr = data[0];
          char* in_ptr = data[1];
          const int64_t in_stride = strides[1];
          float acc = *reinterpret_cast<float*>(out_ptr);
          for (int64_t i = 0; i < size; ++i) {
              acc += *reinterpret_cast<const float*>(in_ptr + i * in_stride);
          }
          *reinterpret_cast<float*>(out_ptr) = acc;
      };
      vbt::core::for_each_reduction_cpu(iter, loop, nullptr);
  } else {
      auto loop = [](char** data, const std::int64_t* strides, std::int64_t size, void*) {
          char* out_ptr = data[0];
          char* in_ptr = data[1];
          const int64_t in_stride = strides[1];
          long long acc = *reinterpret_cast<long long*>(out_ptr);
          for (int64_t i = 0; i < size; ++i) {
              acc += *reinterpret_cast<const long long*>(in_ptr + i * in_stride);
          }
          *reinterpret_cast<long long*>(out_ptr) = acc;
      };
      vbt::core::for_each_reduction_cpu(iter, loop, nullptr);
  }

  return out;
}

TensorImpl vbt_default_mean_impl(const TensorImpl& self, std::vector<int64_t> dims, bool keepdim) {
    if (self.device().type != kDLCPU) {
        throw std::invalid_argument("vt::mean: unsupported device (expected CPU)");
    }
    ScalarType dtype = self.dtype();
    if (dtype != ScalarType::Float32 && dtype != ScalarType::Int64) {
        throw std::invalid_argument("vt::mean: unsupported dtype " + dtype_to_string(dtype));
    }
    
    if (dims.empty()) {
        dims.resize(self.sizes().size());
        std::iota(dims.begin(), dims.end(), 0);
    }

    TensorImpl out = vbt_default_sum_impl(self, dims, keepdim);

    const auto& in_sizes = self.sizes();
    int64_t R = static_cast<int64_t>(in_sizes.size());
    std::vector<int64_t> norm_dims;
    norm_dims.reserve(dims.size());
    for (int64_t d : dims) {
      if (d < 0) d += R;
      norm_dims.push_back(d);
    }
    std::sort(norm_dims.begin(), norm_dims.end());
    norm_dims.erase(std::unique(norm_dims.begin(), norm_dims.end()), norm_dims.end());

    int64_t count = 1;
    for(auto d : norm_dims) {
        if(d >= 0 && d < R) {
            int64_t dim_size = in_sizes[static_cast<size_t>(d)];
            if (!vbt::core::checked_mul_i64(count, dim_size, count)) {
                throw std::runtime_error("vt::mean: count overflow");
            }
        }
    }

    if (count == 0) {
        if (dtype == ScalarType::Float32) {
            // Fill output with NaNs
            TensorImpl out_nan = make_empty_like_broadcast_cpu(out, out.sizes(), dtype);
            float* d = reinterpret_cast<float*>(out_nan.data());
            int64_t n = out_nan.numel();
            for(int64_t i=0; i<n; ++i) d[i] = std::numeric_limits<float>::quiet_NaN();
            return out_nan;
        }
        return out; // Return 0 for Int64
    }

    int64_t n = out.numel();
    if (dtype == ScalarType::Float32) {
        float* d = reinterpret_cast<float*>(out.data());
        float divisor = static_cast<float>(count);
        for(int64_t i=0; i<n; ++i) d[i] /= divisor;
    } else {
        long long* d = reinterpret_cast<long long*>(out.data());
        long long divisor = count;
        for(int64_t i=0; i<n; ++i) d[i] /= divisor;
    }

    return out;
}

// Helper to stringify dtype for error messages
static std::string dtype_to_string(ScalarType t) {
  switch (t) {
    case ScalarType::Bool: return "Bool";
    case ScalarType::Int32: return "Int32";
    case ScalarType::Int64: return "Int64";
    case ScalarType::Float32: return "Float32";
    case ScalarType::Float16: return "Float16";
    case ScalarType::BFloat16: return "BFloat16";
    case ScalarType::Float64: return "Float64";
    case ScalarType::Complex64: return "Complex64";
    case ScalarType::Complex128: return "Complex128";
    case ScalarType::Undefined: return "Undefined";
    default: return "Unknown";
  }
}

template <typename Op>
static TensorImpl vbt_default_comparison_impl(const TensorImpl& a, const TensorImpl& b, const IterOpSignature* sig, const char* name) {
  if (a.dtype() != b.dtype()) throw std::invalid_argument(std::string(name) + ": dtype mismatch");
  if (a.device() != b.device()) throw std::invalid_argument(std::string(name) + ": device mismatch");
  if (a.device().type != kDLCPU) {
    throw std::invalid_argument(std::string(name) + ": unsupported device (expected CPU)");
  }

  const ScalarType dtype = a.dtype();
  if (dtype == ScalarType::Float32 || dtype == ScalarType::Int64) {
    // ... implementation ...
    std::vector<int64_t> out_sizes;
    try {
      out_sizes = vbt::core::infer_broadcast_shape(
          std::span<const std::int64_t>(a.sizes()),
          std::span<const std::int64_t>(b.sizes()));
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument(std::string(name) + ": broadcast error: " + std::string(e.what()));
    }

    // Create Bool output
    const ScalarType out_dtype = ScalarType::Bool;
    const std::size_t item_b = vbt::core::itemsize(out_dtype);
    const std::int64_t n = safe_numel_from_sizes(out_sizes);
    const std::size_t nbytes = static_cast<std::size_t>(n) * item_b;

    void* buf = nullptr;
    if (nbytes > 0) buf = ::operator new(nbytes);
    DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
    auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
    auto strides = make_contiguous_strides_from_sizes(out_sizes);
    TensorImpl out(storage, out_sizes, strides, 0, out_dtype, Device::cpu());

    vbt::core::TensorIterConfig cfg;
    cfg.check_all_same_dtype(false);
    cfg.add_output(vbt::core::OptionalTensorImplRef(&out, true));
    cfg.add_input(a);
    cfg.add_input(b);
    cfg.check_mem_overlap(true);
    cfg.set_op_signature(sig);
    vbt::core::TensorIter iter = cfg.build();

    if (dtype == ScalarType::Float32) {
      auto loop = [](char** data, const std::int64_t* strides, std::int64_t size, void*) {
        char* out_base = data[0]; char* a_base = data[1]; char* b_base = data[2];
        const std::int64_t so = strides[0], sa = strides[1], sb = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          float va = *reinterpret_cast<const float*>(a_base + i * sa);
          float vb = *reinterpret_cast<const float*>(b_base + i * sb);
          *reinterpret_cast<bool*>(out_base + i * so) = Op{}(va, vb);
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    } else {
      auto loop = [](char** data, const std::int64_t* strides, std::int64_t size, void*) {
        char* out_base = data[0]; char* a_base = data[1]; char* b_base = data[2];
        const std::int64_t so = strides[0], sa = strides[1], sb = strides[2];
        for (std::int64_t i = 0; i < size; ++i) {
          long long va = *reinterpret_cast<const long long*>(a_base + i * sa);
          long long vb = *reinterpret_cast<const long long*>(b_base + i * sb);
          *reinterpret_cast<bool*>(out_base + i * so) = Op{}(va, vb);
        }
      };
      vbt::core::for_each_cpu(iter, loop, nullptr);
    }
    return out;
  }
  throw std::invalid_argument(std::string(name) + ": unsupported dtype " + dtype_to_string(dtype));
}

struct EqOp { template <typename T> bool operator()(T a, T b) const { return a == b; } };
struct NeOp { template <typename T> bool operator()(T a, T b) const { return a != b; } };
struct LtOp { template <typename T> bool operator()(T a, T b) const { return a < b; } };
struct GtOp { template <typename T> bool operator()(T a, T b) const { return a > b; } };
struct LeOp { template <typename T> bool operator()(T a, T b) const { return a <= b; } };
struct GeOp { template <typename T> bool operator()(T a, T b) const { return a >= b; } };

static TensorImpl vbt_default_eq_impl(const TensorImpl& a, const TensorImpl& b) {
  return vbt_default_comparison_impl<EqOp>(a, b, &kVtEqSignature, "vt::eq");
}
static TensorImpl vbt_default_ne_impl(const TensorImpl& a, const TensorImpl& b) {
  return vbt_default_comparison_impl<NeOp>(a, b, &kVtNeSignature, "vt::ne");
}
static TensorImpl vbt_default_lt_impl(const TensorImpl& a, const TensorImpl& b) {
  return vbt_default_comparison_impl<LtOp>(a, b, &kVtLtSignature, "vt::lt");
}
static TensorImpl vbt_default_gt_impl(const TensorImpl& a, const TensorImpl& b) {
  return vbt_default_comparison_impl<GtOp>(a, b, &kVtGtSignature, "vt::gt");
}
static TensorImpl vbt_default_le_impl(const TensorImpl& a, const TensorImpl& b) {
  return vbt_default_comparison_impl<LeOp>(a, b, &kVtLeSignature, "vt::le");
}
static TensorImpl vbt_default_ge_impl(const TensorImpl& a, const TensorImpl& b) {
  return vbt_default_comparison_impl<GeOp>(a, b, &kVtGeSignature, "vt::ge");
}

// Static registration: library + schemas + CPU impls
VBT_LIBRARY(vt)
VBT_OP("vt::unit() -> Tensor")
VBT_IMPL_CPU("vt::unit", &vbt_default_unit_impl)

VBT_OP("vt::relu(Tensor) -> Tensor")
VBT_IMPL_CPU("vt::relu", &vbt_default_relu_impl)

VBT_OP("vt::add(Tensor, Tensor) -> Tensor")
VBT_IMPL_CPU("vt::add", &vbt_default_add_impl)

VBT_OP("vt::mul(Tensor, Tensor) -> Tensor")
VBT_IMPL_CPU("vt::mul", &vbt_default_mul_impl)

VBT_OP("vt::sub(Tensor, Tensor) -> Tensor")
VBT_IMPL_CPU("vt::sub", &vbt_default_sub_impl)

VBT_OP("vt::div(Tensor, Tensor) -> Tensor")
VBT_IMPL_CPU("vt::div", &vbt_default_div_impl)

VBT_OP("vt::abs(Tensor) -> Tensor")
VBT_IMPL_CPU("vt::abs", &vbt_default_abs_impl)

VBT_OP("vt::neg(Tensor) -> Tensor")
VBT_IMPL_CPU("vt::neg", &vbt_default_neg_impl)

VBT_OP("vt::reciprocal(Tensor) -> Tensor")
VBT_IMPL_CPU("vt::reciprocal", &vbt_default_reciprocal_impl)

VBT_OP("vt::sum(Tensor, int[], bool) -> Tensor")
// NOTE: This schema includes non-Tensor arguments (int[], bool). The dispatcher
// currently only supports Tensor inputs, so this op is schema-only for now.
// Python reduction methods do not route through the dispatcher.

VBT_OP("vt::mean(Tensor, int[], bool) -> Tensor")
// NOTE: See vt::sum comment above.

VBT_OP("vt::eq(Tensor, Tensor) -> Tensor")
VBT_IMPL_CPU("vt::eq", &vbt_default_eq_impl)

VBT_OP("vt::ne(Tensor, Tensor) -> Tensor")
VBT_IMPL_CPU("vt::ne", &vbt_default_ne_impl)

VBT_OP("vt::lt(Tensor, Tensor) -> Tensor")
VBT_IMPL_CPU("vt::lt", &vbt_default_lt_impl)

VBT_OP("vt::gt(Tensor, Tensor) -> Tensor")
VBT_IMPL_CPU("vt::gt", &vbt_default_gt_impl)

VBT_OP("vt::le(Tensor, Tensor) -> Tensor")
VBT_IMPL_CPU("vt::le", &vbt_default_le_impl)

VBT_OP("vt::ge(Tensor, Tensor) -> Tensor")
VBT_IMPL_CPU("vt::ge", &vbt_default_ge_impl)

#include "vbt/dispatch/dispatcher.h"

extern "C" void vbt_register_default_kernels() {
  auto& D = vbt::dispatch::Dispatcher::instance();
  if (!D.has("vt::unit")) {
    D.registerLibrary("vt");
    D.def("vt::unit() -> Tensor");
    D.registerCpuKernel("vt::unit", &vbt_default_unit_impl);
  }
  if (!D.has("vt::relu")) {
    D.def("vt::relu(Tensor) -> Tensor");
    D.registerCpuKernel("vt::relu", &vbt_default_relu_impl);
  }
  if (!D.has("vt::add")) {
    D.def("vt::add(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel("vt::add", &vbt_default_add_impl);
  }
  if (!D.has("vt::mul")) {
    D.def("vt::mul(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel("vt::mul", &vbt_default_mul_impl);
  }
  if (!D.has("vt::sub")) {
    D.def("vt::sub(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel("vt::sub", &vbt_default_sub_impl);
  }
  if (!D.has("vt::div")) {
    D.def("vt::div(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel("vt::div", &vbt_default_div_impl);
  }
  if (!D.has("vt::abs")) {
    D.def("vt::abs(Tensor) -> Tensor");
    D.registerCpuKernel("vt::abs", &vbt_default_abs_impl);
  }
  if (!D.has("vt::neg")) {
    D.def("vt::neg(Tensor) -> Tensor");
    D.registerCpuKernel("vt::neg", &vbt_default_neg_impl);
  }
  if (!D.has("vt::reciprocal")) {
    D.def("vt::reciprocal(Tensor) -> Tensor");
    D.registerCpuKernel("vt::reciprocal", &vbt_default_reciprocal_impl);
  }
  // NOTE: Reductions with non-Tensor args are schema-only for now. The dispatcher
  // only supports Tensor inputs, so do not register these as unboxed kernels.
  if (!D.has("vt::sum")) {
    D.def("vt::sum(Tensor, int[], bool) -> Tensor");
  }
  if (!D.has("vt::mean")) {
    D.def("vt::mean(Tensor, int[], bool) -> Tensor");
  }
  if (!D.has("vt::eq")) {
    D.def("vt::eq(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel("vt::eq", &vbt_default_eq_impl);
  }
  if (!D.has("vt::ne")) {
    D.def("vt::ne(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel("vt::ne", &vbt_default_ne_impl);
  }
  if (!D.has("vt::lt")) {
    D.def("vt::lt(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel("vt::lt", &vbt_default_lt_impl);
  }
  if (!D.has("vt::gt")) {
    D.def("vt::gt(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel("vt::gt", &vbt_default_gt_impl);
  }
  if (!D.has("vt::le")) {
    D.def("vt::le(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel("vt::le", &vbt_default_le_impl);
  }
  if (!D.has("vt::ge")) {
    D.def("vt::ge(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel("vt::ge", &vbt_default_ge_impl);
  }

  const char* schema_only[] = {
    "vt::erf(Tensor) -> Tensor",
    "vt::erfc(Tensor) -> Tensor",
    "vt::lgamma(Tensor) -> Tensor",
    "vt::sinc(Tensor) -> Tensor",
    "vt::exp(Tensor) -> Tensor",
    "vt::log(Tensor) -> Tensor",
    "vt::sqrt(Tensor) -> Tensor",
    "vt::rsqrt(Tensor) -> Tensor",
    "vt::sin(Tensor) -> Tensor",
    "vt::cos(Tensor) -> Tensor",
    "vt::tanh(Tensor) -> Tensor",
    "vt::tanh_backward(Tensor, Tensor) -> Tensor",
    "vt::sigmoid(Tensor) -> Tensor",
    "vt::sigmoid_backward(Tensor, Tensor) -> Tensor",
    "vt::expm1(Tensor) -> Tensor",
    "vt::log1p(Tensor) -> Tensor",
    "vt::floor(Tensor) -> Tensor",
    "vt::ceil(Tensor) -> Tensor",
    "vt::trunc(Tensor) -> Tensor",
    "vt::round(Tensor) -> Tensor",
    "vt::frac(Tensor) -> Tensor",
    "vt::sign(Tensor) -> Tensor",
    "vt::sgn(Tensor) -> Tensor",
    "vt::angle(Tensor) -> Tensor",
    "vt::conj_physical(Tensor) -> Tensor",
    "vt::isfinite(Tensor) -> Tensor",
    "vt::isinf(Tensor) -> Tensor",
    "vt::isnan(Tensor) -> Tensor",
    "vt::isneginf(Tensor) -> Tensor",
    "vt::isposinf(Tensor) -> Tensor",
    "vt::logical_not(Tensor) -> Tensor",
    "vt::signbit(Tensor) -> Tensor",
    "vt::nan_to_num(Tensor) -> Tensor",
    "vt::exp2(Tensor) -> Tensor",
    "vt::log2(Tensor) -> Tensor",
    "vt::log10(Tensor) -> Tensor",
    "vt::sinh(Tensor) -> Tensor",
    "vt::cosh(Tensor) -> Tensor",
    "vt::asinh(Tensor) -> Tensor",
    "vt::acosh(Tensor) -> Tensor",
    "vt::atanh(Tensor) -> Tensor",
    "vt::tan(Tensor) -> Tensor",
    "vt::deg2rad(Tensor) -> Tensor",
    "vt::rad2deg(Tensor) -> Tensor",
    "vt::asin(Tensor) -> Tensor",
    "vt::acos(Tensor) -> Tensor",
    "vt::atan(Tensor) -> Tensor",
    "vt::positive(Tensor) -> Tensor",
    "vt::square(Tensor) -> Tensor",
    "vt::pow(Tensor, Tensor) -> Tensor",
    "vt::logaddexp(Tensor, Tensor) -> Tensor",
    "vt::logaddexp2(Tensor, Tensor) -> Tensor",
    "vt::ldexp(Tensor, Tensor) -> Tensor",
    "vt::float_power(Tensor, Tensor) -> Tensor",
    "vt::rsub(Tensor, Tensor) -> Tensor",
    "vt::true_divide(Tensor, Tensor) -> Tensor",
    "vt::clamp(Tensor, Tensor, Tensor) -> Tensor",
    "vt::clamp_min(Tensor, Tensor) -> Tensor",
    "vt::clamp_max(Tensor, Tensor) -> Tensor",
    "vt::relu6(Tensor) -> Tensor",
    "vt::hardtanh(Tensor) -> Tensor",
    "vt::hardsigmoid(Tensor) -> Tensor",
    "vt::silu(Tensor) -> Tensor",
    "vt::silu_backward(Tensor, Tensor) -> Tensor",
    "vt::gelu(Tensor) -> Tensor",
    "vt::mish(Tensor) -> Tensor",
    "vt::selu(Tensor) -> Tensor",
    "vt::softplus(Tensor) -> Tensor",
    "vt::hardshrink(Tensor) -> Tensor",
    "vt::softshrink(Tensor) -> Tensor",
    "vt::celu(Tensor) -> Tensor",
    "vt::elu(Tensor) -> Tensor",
    "vt::threshold(Tensor, Tensor, Tensor) -> Tensor",
    "vt::bitwise_not(Tensor) -> Tensor",
    "vt::clip(Tensor, Tensor, Tensor) -> Tensor",
    "vt::lerp(Tensor, Tensor, Tensor) -> Tensor",
    "vt::where(Tensor, Tensor, Tensor) -> Tensor",
    "vt::masked_fill(Tensor, Tensor, Tensor) -> Tensor",
    "vt::addcmul(Tensor, Tensor, Tensor, Tensor) -> Tensor",
    "vt::addcdiv(Tensor, Tensor, Tensor, Tensor) -> Tensor",
    "vt::logit(Tensor) -> Tensor",
    "vt::logit_backward(Tensor, Tensor) -> Tensor",
    "vt::polygamma(Tensor, Tensor) -> Tensor",
    "vt::fmax(Tensor, Tensor) -> Tensor",
    "vt::fmin(Tensor, Tensor) -> Tensor",
    "vt::maximum(Tensor, Tensor) -> Tensor",
    "vt::minimum(Tensor, Tensor) -> Tensor",
    "vt::huber_loss(Tensor, Tensor, Tensor) -> Tensor",
    "vt::mse_loss(Tensor, Tensor) -> Tensor",
    "vt::smooth_l1_loss(Tensor, Tensor, Tensor) -> Tensor",
    "vt::lshift(Tensor, Tensor) -> Tensor",
    "vt::rshift(Tensor, Tensor) -> Tensor",
    "vt::bitwise_left_shift(Tensor, Tensor) -> Tensor",
    "vt::bitwise_right_shift(Tensor, Tensor) -> Tensor",
    "vt::fmod(Tensor, Tensor) -> Tensor",
    "vt::remainder(Tensor, Tensor) -> Tensor",
    "vt::atan2(Tensor, Tensor) -> Tensor",
    "vt::copysign(Tensor, Tensor) -> Tensor",
    "vt::hypot(Tensor, Tensor) -> Tensor",
    "vt::gcd(Tensor, Tensor) -> Tensor",
    "vt::lcm(Tensor, Tensor) -> Tensor",
    "vt::xlogy(Tensor, Tensor) -> Tensor",
    "vt::xlog1py(Tensor, Tensor) -> Tensor",
    "vt::special_xlog1py(Tensor, Tensor) -> Tensor",
    "vt::nextafter(Tensor, Tensor) -> Tensor",
    "vt::heaviside(Tensor, Tensor) -> Tensor",
    "vt::bitwise_and(Tensor, Tensor) -> Tensor",
    "vt::bitwise_or(Tensor, Tensor) -> Tensor",
    "vt::bitwise_xor(Tensor, Tensor) -> Tensor",
    "vt::logical_and(Tensor, Tensor) -> Tensor",
    "vt::logical_or(Tensor, Tensor) -> Tensor",
    "vt::logical_xor(Tensor, Tensor) -> Tensor",
    "vt::max(Tensor, int[], bool) -> Tensor",
    "vt::min(Tensor, int[], bool) -> Tensor",
    "vt::prod(Tensor, int[], bool) -> Tensor"
  };
  for (const char* s : schema_only) {
    // Extract name
    std::string schema(s);
    auto pos = schema.find('(');
    if (pos != std::string::npos) {
        std::string name = schema.substr(0, pos);
        if (!D.has(name)) {
            D.def(schema);
        }
    }
  }
}
