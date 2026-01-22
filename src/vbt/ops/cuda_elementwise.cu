// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <limits>
#include <mutex>
#include <type_traits>
#include <cmath>
#include <cfloat>

#include "vbt/dispatch/registration.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/dtype.h"
#include "vbt/core/complex.h"
#include "vbt/core/view_ops.h"
#include "vbt/core/device.h"
#include "vbt/core/broadcast.h"
#include "vbt/core/tensor_iter.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/guard.h"

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#include "vbt/core/tensor_iterator/cuda_loops.h"
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#ifdef __has_include
#  if __has_include(<cuda_bf16.h>)
#    include <cuda_bf16.h>
#    define VBT_CUDA_HAS_BF16 1
#  else
#    define VBT_CUDA_HAS_BF16 0
#  endif
#else
#  define VBT_CUDA_HAS_BF16 0
#endif

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::TensorIter;
using vbt::core::TensorIterConfig;
using vbt::core::OptionalTensorImplRef;
using vbt::core::IterOperandRole;
using vbt::core::DeviceStrideMeta;
using vbt::core::compute_offset_elems;

constexpr int VBT_CUDA_BROADCAST_MAX_NDIM =
    static_cast<int>(vbt::core::kTensorIterCudaMaxNdim);
static_assert(VBT_CUDA_BROADCAST_MAX_NDIM ==
              vbt::core::kTensorIterCudaMaxNdim,
              "CUDA broadcast max ndim must match TI CUDA max ndim");

namespace vbt::cuda_detail {

inline bool should_use_int32_index(std::int64_t N) noexcept {
  return N <= static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
}

} // namespace vbt::cuda_detail

using vbt::cuda::DeviceGuard;

// Compute row-major contiguous strides for given sizes
static inline std::vector<int64_t> make_contiguous_strides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> st(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    st[static_cast<std::size_t>(i)] = acc;
    int64_t dim = sizes[static_cast<std::size_t>(i)];
    if (dim == 0) dim = 1;
    acc *= dim;
  }
  return st;
}

static inline void launch_bounds_and_grid(int64_t N, dim3& grid, dim3& block) {
  const int threads = 256;
  block = dim3(threads);
  // Round up; clamp grid.x to a reasonable maximum. Kernels use grid-stride
  // loops so a smaller grid still covers the full range.
  int64_t blocks = (N + threads - 1) / threads;
  if (blocks <= 0) blocks = 1;
  if (blocks > 65535) blocks = 65535;
  grid = dim3(static_cast<unsigned int>(blocks));
}

static inline TensorImpl make_cuda_dense_out(const std::vector<int64_t>& sizes, ScalarType dtype, Device device) {
  int64_t N = 1;
  for (auto s : sizes) {
    if (s == 0) { N = 0; break; }
    if (N <= std::numeric_limits<int64_t>::max() / s) N *= s; else N = 0;
  }
  const auto item_b = static_cast<std::size_t>(vbt::core::itemsize(dtype));
  std::size_t nbytes = item_b * static_cast<std::size_t>(N);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, device.index);
  // Note: stream association is handled at the op level via
  // vbt::cuda::record_stream in vbt_cuda_*_impl; keep this helper
  // stream-agnostic.
  auto strides = make_contiguous_strides(sizes);
  return TensorImpl(storage, sizes, std::move(strides), /*storage_offset=*/0, dtype, device);
}

// Return total number of elements (same as TensorImpl::numel())
static inline int64_t numel_of(const TensorImpl& t) {
  return t.numel();
}

static inline bool is_integral_or_bool(ScalarType st) noexcept {
  return st == ScalarType::Int32 || st == ScalarType::Int64 || st == ScalarType::Bool;
}

static inline void reject_integral_inputs(const TensorImpl& t, const char* name) {
  if (is_integral_or_bool(t.dtype())) {
    throw std::invalid_argument(std::string(name) + ": floating tensor expected");
  }
}

namespace detail {
  template <vbt::core::ScalarType> struct Traits;

  template <> struct Traits<ScalarType::Float32> { using storage_t = float; using compute_t = float; };
  template <> struct Traits<ScalarType::Float16> { using storage_t = __half; using compute_t = float; };
#if VBT_CUDA_HAS_BF16
  template <> struct Traits<ScalarType::BFloat16> { using storage_t = __nv_bfloat16; using compute_t = float; };
#endif
  __device__ inline float load_as_float(const __half* p) { return __half2float(*p); }
  __device__ inline void store_from_float(__half* p, float x) { *p = __float2half_rn(x); }
#if VBT_CUDA_HAS_BF16
  __device__ inline float load_as_float(const __nv_bfloat16* p) { return __bfloat162float(*p); }
  __device__ inline void store_from_float(__nv_bfloat16* p, float x) { *p = __float2bfloat16_rn(x); }
#endif

  // Treat CUDA half / bfloat16 as floating for templated ops.
  template <typename T>
  struct is_fp_like : std::bool_constant<std::is_floating_point_v<T>> {};
  template <> struct is_fp_like<__half> : std::true_type {};
#if VBT_CUDA_HAS_BF16
  template <> struct is_fp_like<__nv_bfloat16> : std::true_type {};
#endif

  template <typename T>
  inline constexpr bool is_fp_like_v = is_fp_like<T>::value;
}

// --- Op Functors ---
template <typename T> struct AddOp { __device__ T operator()(T a, T b) const { return a + b; } };
template <typename T> struct SubOp { __device__ T operator()(T a, T b) const { return a - b; } };
template <typename T> struct MulOp { __device__ T operator()(T a, T b) const { return a * b; } };
template <typename T> struct DivOp { __device__ T operator()(T a, T b) const { return a / b; } };
template <typename T> struct RSubOp { __device__ T operator()(T a, T b) const { return b - a; } };
template <typename T> struct RDivOp { __device__ T operator()(T a, T b) const { return b / a; } };

template <> struct AddOp<vbt::core::Complex<float>> {
  __device__ vbt::core::Complex<float> operator()(vbt::core::Complex<float> a,
                                                  vbt::core::Complex<float> b) const {
    return vbt::core::Complex<float>{a.re + b.re, a.im + b.im};
  }
};

template <> struct AddOp<vbt::core::Complex<double>> {
  __device__ vbt::core::Complex<double> operator()(vbt::core::Complex<double> a,
                                                   vbt::core::Complex<double> b) const {
    return vbt::core::Complex<double>{a.re + b.re, a.im + b.im};
  }
};

template <> struct MulOp<vbt::core::Complex<float>> {
  __device__ vbt::core::Complex<float> operator()(vbt::core::Complex<float> a,
                                                  vbt::core::Complex<float> b) const {
    return vbt::core::Complex<float>{a.re * b.re - a.im * b.im,
                                     a.re * b.im + a.im * b.re};
  }
};

template <> struct MulOp<vbt::core::Complex<double>> {
  __device__ vbt::core::Complex<double> operator()(vbt::core::Complex<double> a,
                                                   vbt::core::Complex<double> b) const {
    return vbt::core::Complex<double>{a.re * b.re - a.im * b.im,
                                      a.re * b.im + a.im * b.re};
  }
};
template <typename T> struct BitwiseAndOp { __device__ T operator()(T a, T b) const { if constexpr (std::is_integral_v<T>) return a & b; else return T(0); } };
template <typename T> struct BitwiseOrOp  { __device__ T operator()(T a, T b) const { if constexpr (std::is_integral_v<T>) return a | b; else return T(0); } };
template <typename T> struct BitwiseXorOp { __device__ T operator()(T a, T b) const { if constexpr (std::is_integral_v<T>) return a ^ b; else return T(0); } };
template <typename T> struct LogicalAndOp { __device__ bool operator()(T a, T b) const { return static_cast<bool>(a) && static_cast<bool>(b); } };
template <typename T> struct LogicalOrOp  { __device__ bool operator()(T a, T b) const { return static_cast<bool>(a) || static_cast<bool>(b); } };
template <typename T> struct LogicalXorOp { __device__ bool operator()(T a, T b) const { return static_cast<bool>(a) != static_cast<bool>(b); } };
template <typename T> struct LShiftOp {
  __device__ T operator()(T a, T b) const {
    if constexpr (std::is_integral_v<T>) {
      constexpr int bits = static_cast<int>(sizeof(T) * 8);
      long long sh = static_cast<long long>(b);
      // torch semantics: invalid shifts yield 0
      if (sh < 0 || sh >= bits) return T(0);

      using U = std::make_unsigned_t<T>;
      U ua = static_cast<U>(a);
      U res = ua << static_cast<int>(sh);
      return static_cast<T>(res);
    } else {
      return T(0);
    }
  }
};

template <typename T> struct RShiftOp {
  __device__ T operator()(T a, T b) const {
    if constexpr (std::is_integral_v<T>) {
      constexpr int bits = static_cast<int>(sizeof(T) * 8);
      long long sh = static_cast<long long>(b);
      // torch semantics: invalid shifts yield 0 for non-negative, -1 for negative signed
      if (sh < 0 || sh >= bits) {
        if constexpr (std::is_signed_v<T>) {
          return a < 0 ? T(-1) : T(0);
        } else {
          return T(0);
        }
      }

      using U = std::make_unsigned_t<T>;
      U ua = static_cast<U>(a);
      U res = ua >> static_cast<int>(sh);
      if constexpr (std::is_signed_v<T>) {
        if (a < 0 && sh > 0) {
          U fill = (~U(0)) << (bits - static_cast<int>(sh));
          res |= fill;
        }
      }
      return static_cast<T>(res);
    } else {
      return T(0);
    }
  }
};
template <typename T> struct MaximumOp {
  __device__ T operator()(T a, T b) const {
    if constexpr (detail::is_fp_like_v<T>) {
      // torch.maximum/minimum propagate NaNs, but use fmax/fmin semantics
      // for signed-zero tie-breaking.
      if constexpr (std::is_same_v<T, __half>) {
        const float af = __half2float(a);
        const float bf = __half2float(b);
        if (::isnan(af)) return a;
        if (::isnan(bf)) return b;
        return __float2half_rn(::fmaxf(af, bf));
#if VBT_CUDA_HAS_BF16
      } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        const float af = __bfloat162float(a);
        const float bf = __bfloat162float(b);
        if (::isnan(af)) return a;
        if (::isnan(bf)) return b;
        return __float2bfloat16_rn(::fmaxf(af, bf));
#endif
      } else if constexpr (std::is_same_v<T, float>) {
        if (::isnan(a)) return a;
        if (::isnan(b)) return b;
        return ::fmaxf(a, b);
      } else {
        if (::isnan(a)) return a;
        if (::isnan(b)) return b;
        return static_cast<T>(::fmax(static_cast<double>(a), static_cast<double>(b)));
      }
    }
    return a > b ? a : b;
  }
};

template <typename T> struct MinimumOp {
  __device__ T operator()(T a, T b) const {
    if constexpr (detail::is_fp_like_v<T>) {
      if constexpr (std::is_same_v<T, __half>) {
        const float af = __half2float(a);
        const float bf = __half2float(b);
        if (::isnan(af)) return a;
        if (::isnan(bf)) return b;
        return __float2half_rn(::fminf(af, bf));
#if VBT_CUDA_HAS_BF16
      } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        const float af = __bfloat162float(a);
        const float bf = __bfloat162float(b);
        if (::isnan(af)) return a;
        if (::isnan(bf)) return b;
        return __float2bfloat16_rn(::fminf(af, bf));
#endif
      } else if constexpr (std::is_same_v<T, float>) {
        if (::isnan(a)) return a;
        if (::isnan(b)) return b;
        return ::fminf(a, b);
      } else {
        if (::isnan(a)) return a;
        if (::isnan(b)) return b;
        return static_cast<T>(::fmin(static_cast<double>(a), static_cast<double>(b)));
      }
    }
    return a < b ? a : b;
  }
};
template <typename T> struct FmodOp {
  __device__ T operator()(T a, T b) const {
    if constexpr (std::is_same_v<T, __half>) {
      float af = __half2float(a);
      float bf = __half2float(b);
      return __float2half_rn(::fmodf(af, bf));
#if VBT_CUDA_HAS_BF16
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      float af = __bfloat162float(a);
      float bf = __bfloat162float(b);
      return __float2bfloat16_rn(::fmodf(af, bf));
#endif
    } else if constexpr (std::is_same_v<T, float>) {
      return ::fmodf(a, b);
    } else if constexpr (std::is_integral_v<T>) {
      if (b == 0) return 0;
      // Guard against INT_MIN % -1 which is UB (division overflow)
      // Use bit manipulation to get min value (avoids constexpr host function call)
      if constexpr (std::is_signed_v<T>) {
        constexpr T t_min = T(1) << (sizeof(T) * 8 - 1);
        if (a == t_min && b == T(-1)) return 0;
      }
      return static_cast<T>(a % b);
    } else {
      return static_cast<T>(::fmod(static_cast<double>(a), static_cast<double>(b)));
    }
  }
};

template <typename T> struct RemainderOp {
  __device__ T operator()(T a, T b) const {
    // torch.remainder semantics: modulus with sign of divisor for non-zero results.
    if constexpr (std::is_same_v<T, __half>) {
      float af = __half2float(a);
      float bf = __half2float(b);
      float r = ::fmodf(af, bf);
      if (r != 0.0f && ((r < 0.0f) != (bf < 0.0f))) r += bf;
      return __float2half_rn(r);
#if VBT_CUDA_HAS_BF16
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      float af = __bfloat162float(a);
      float bf = __bfloat162float(b);
      float r = ::fmodf(af, bf);
      if (r != 0.0f && ((r < 0.0f) != (bf < 0.0f))) r += bf;
      return __float2bfloat16_rn(r);
#endif
    } else if constexpr (std::is_same_v<T, float>) {
      float r = ::fmodf(a, b);
      if (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) r += b;
      return r;
    } else if constexpr (std::is_integral_v<T>) {
      if (b == 0) return 0;
      // Guard against INT_MIN % -1 which is UB (division overflow)
      if constexpr (std::is_signed_v<T>) {
        constexpr T t_min = T(1) << (sizeof(T) * 8 - 1);
        if (a == t_min && b == T(-1)) return 0;
      }
      T r = static_cast<T>(a % b);
      if constexpr (std::is_signed_v<T>) {
        if (r != 0 && ((r < 0) != (b < 0))) r = static_cast<T>(r + b);
      }
      return r;
    } else {
      double ad = static_cast<double>(a);
      double bd = static_cast<double>(b);
      double r = ::fmod(ad, bd);
      if (r != 0.0 && ((r < 0.0) != (bd < 0.0))) r += bd;
      return static_cast<T>(r);
    }
  }
};
template <typename T> struct Atan2Op {
  __device__ T operator()(T a, T b) const {
    if constexpr (std::is_same_v<T, float>) {
      return ::atan2f(a, b);
    } else {
      return static_cast<T>(atan2(static_cast<double>(a), static_cast<double>(b)));
    }
  }
};
template <typename T> struct CopysignOp {
  __device__ T operator()(T a, T b) const {
    if constexpr (std::is_same_v<T, float>) {
      return ::copysignf(a, b);
    } else {
      return static_cast<T>(copysign(static_cast<double>(a), static_cast<double>(b)));
    }
  }
};
template <typename T> struct HypotOp {
  __device__ T operator()(T a, T b) const {
    if constexpr (std::is_same_v<T, float>) {
      return ::hypotf(a, b);
    } else {
      return static_cast<T>(hypot(static_cast<double>(a), static_cast<double>(b)));
    }
  }
};
template <typename T> struct XlogyOp {
  __device__ T operator()(T x, T y) const {
    if (x == T(0)) return T(0);
    if constexpr (std::is_same_v<T, float>) {
      return x * ::logf(y);
    } else {
      return static_cast<T>(static_cast<double>(x) * log(static_cast<double>(y)));
    }
  }
};
template <typename T> struct Xlog1pyOp {
  __device__ T operator()(T x, T y) const {
    if (x == T(0)) return T(0);
    if constexpr (std::is_same_v<T, float>) {
      return x * ::log1pf(y);
    } else {
      return static_cast<T>(static_cast<double>(x) * log1p(static_cast<double>(y)));
    }
  }
};

template <typename T> struct LogaddexpOp {
  __device__ T operator()(T a, T b) const {
    if constexpr (std::is_same_v<T, float>) {
      float m = a > b ? a : b;
      if (::isinf(m)) return m;
      float d = ::fabsf(a - b);
      return m + ::log1pf(::expf(-d));
    } else {
      double x = static_cast<double>(a);
      double y = static_cast<double>(b);
      double m = x > y ? x : y;
      if (isinf(m)) return static_cast<T>(m);
      double d = fabs(x - y);
      return static_cast<T>(m + log1p(exp(-d)));
    }
  }
};

template <typename T> struct Logaddexp2Op {
  __device__ T operator()(T a, T b) const {
    if constexpr (std::is_same_v<T, float>) {
      float m = a > b ? a : b;
      if (::isinf(m)) return m;
      float d = ::fabsf(a - b);
      return m + ::log2f(1.0f + ::exp2f(-d));
    } else {
      double x = static_cast<double>(a);
      double y = static_cast<double>(b);
      double m = x > y ? x : y;
      if (isinf(m)) return static_cast<T>(m);
      double d = fabs(x - y);
      return static_cast<T>(m + log2(1.0 + exp2(-d)));
    }
  }
};

template <typename T> struct LdexpOp {
  __device__ T operator()(T a, T b) const {
    // ldexp uses an integer exponent and performs an exact power-of-two scale.
    // Our dispatcher currently keeps input dtypes identical, so cast the
    // exponent to int to match ldexp semantics. Clamp to avoid UB on overflow.
    auto clamp_exp = [](T x) -> int {
      constexpr T max_i = static_cast<T>(INT_MAX);
      constexpr T min_i = static_cast<T>(INT_MIN);
      if (x != x) return 0; // NaN -> 0
      if (x > max_i) return INT_MAX;
      if (x < min_i) return INT_MIN;
      return static_cast<int>(x);
    };
    if constexpr (std::is_same_v<T, float>) {
      return ::ldexpf(a, clamp_exp(b));
    } else {
      return static_cast<T>(::ldexp(static_cast<double>(a), clamp_exp(b)));
    }
  }
};

template <typename T> struct LdexpROp {
  __device__ T operator()(T a, T b) const {
    auto clamp_exp = [](T x) -> int {
      constexpr T max_i = static_cast<T>(INT_MAX);
      constexpr T min_i = static_cast<T>(INT_MIN);
      if (x != x) return 0;
      if (x > max_i) return INT_MAX;
      if (x < min_i) return INT_MIN;
      return static_cast<int>(x);
    };
    if constexpr (std::is_same_v<T, float>) {
      return ::ldexpf(b, clamp_exp(a));
    } else {
      return static_cast<T>(::ldexp(static_cast<double>(b), clamp_exp(a)));
    }
  }
};

template <typename T> struct FloatPowerOp {
  __device__ T operator()(T a, T b) const {
    if constexpr (std::is_same_v<T, float>) {
      return ::powf(a, b);
    } else {
      return static_cast<T>(pow(static_cast<double>(a), static_cast<double>(b)));
    }
  }
};

template <typename T> struct FloatPowerROp {
  __device__ T operator()(T a, T b) const {
    if constexpr (std::is_same_v<T, float>) {
      return ::powf(b, a);
    } else {
      return static_cast<T>(pow(static_cast<double>(b), static_cast<double>(a)));
    }
  }
};

template <typename T> struct NextAfterOp {
  __device__ T operator()(T a, T b) const {
    if constexpr (std::is_same_v<T, float>) {
        return nextafterf(a, b);
    } else {
        return static_cast<T>(nextafter(static_cast<double>(a), static_cast<double>(b)));
    }
  }
};
template <typename T> struct HeavisideOp {
  __device__ T operator()(T x, T y) const {
    if constexpr (std::is_signed_v<T> || std::is_floating_point_v<T>) {
       if (x < T(0)) return T(0);
    }
    if (x > T(0)) return T(1);
    return y;
  }
};
template <typename T> struct GcdOp {
  __device__ T operator()(T a, T b) const {
    if constexpr (std::is_integral_v<T>) {
      if constexpr (std::is_signed_v<T>) {
        using U = std::make_unsigned_t<T>;
        U ua = static_cast<U>(a);
        U ub = static_cast<U>(b);
        if (a < 0) ua = U(0) - ua;
        if (b < 0) ub = U(0) - ub;
        while (ub != 0) {
          U t = ub;
          ub = ua % ub;
          ua = t;
        }
        return static_cast<T>(ua);
      } else {
        while (b != 0) {
          T t = b;
          b = a % b;
          a = t;
        }
        return a;
      }
    } else {
      return T(0);
    }
  }
};
template <typename T> struct LcmOp {
  __device__ T operator()(T a, T b) const {
    if constexpr (std::is_integral_v<T>) {
      if (a == 0 || b == 0) return 0;
      if constexpr (std::is_signed_v<T>) {
        using U = std::make_unsigned_t<T>;
        U ua = static_cast<U>(a);
        U ub = static_cast<U>(b);
        if (a < 0) ua = U(0) - ua;
        if (b < 0) ub = U(0) - ub;
        U g = GcdOp<U>()(ua, ub);
        return static_cast<T>((ua / g) * ub);
      } else {
        T g = GcdOp<T>()(a, b);
        return (a / g) * b;
      }
    } else {
      return T(0);
    }
  }
};

// Unary functors

template <typename T> struct AbsOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, __half>) {
      float x = __half2float(a);
      return __float2half_rn(::fabsf(x));
#if VBT_CUDA_HAS_BF16
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      float x = __bfloat162float(a);
      return __float2bfloat16_rn(::fabsf(x));
#endif
    } else if constexpr (std::is_floating_point_v<T>) {
      if constexpr (std::is_same_v<T, float>) {
        return ::fabsf(a);
      } else {
        return static_cast<T>(::fabs(static_cast<double>(a)));
      }
    } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      using U = std::make_unsigned_t<T>;
      // Two's-complement abs with wrap for min (matches torch)
      U ua = static_cast<U>(a);
      if (a < 0) {
        return static_cast<T>(U(0) - ua);
      }
      return a;
    } else {
      // Unsigned/bool: identity
      return a;
    }
  }
};
template <typename T> struct NegOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      using U = std::make_unsigned_t<T>;
      return static_cast<T>(U(0) - static_cast<U>(a));
    } else {
      return -a;
    }
  }
};
template <typename T> struct PositiveOp { __device__ T operator()(T a) const { return a; } };
template <typename T> struct SquareOp { __device__ T operator()(T a) const { return a * a; } };
template <typename T> struct ExpOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) {
      return ::expf(a);
    } else {
      return static_cast<T>(exp(static_cast<double>(a)));
    }
  }
};
template <typename T> struct LogOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) {
      return ::logf(a);
    } else {
      return static_cast<T>(log(static_cast<double>(a)));
    }
  }
};
template <typename T> struct SqrtOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) {
      return ::sqrtf(a);
    } else if constexpr (std::is_same_v<T, __half>) {
      float x = __half2float(a);
      return __float2half_rn(::sqrtf(x));
#if VBT_CUDA_HAS_BF16
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      float x = __bfloat162float(a);
      return __float2bfloat16_rn(::sqrtf(x));
#endif
    } else {
      return static_cast<T>(::sqrt(static_cast<double>(a)));
    }
  }
};
template <typename T> struct RsqrtOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) {
      return ::rsqrtf(a);
    } else {
      return static_cast<T>(rsqrt(static_cast<double>(a)));
    }
  }
};
template <typename T> struct SinOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) {
      return ::sinf(a);
    } else {
      return static_cast<T>(sin(static_cast<double>(a)));
    }
  }
};
template <typename T> struct CosOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) {
      return ::cosf(a);
    } else {
      return static_cast<T>(cos(static_cast<double>(a)));
    }
  }
};
template <typename T> struct TanhOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) {
      return ::tanhf(a);
    } else {
      return static_cast<T>(tanh(static_cast<double>(a)));
    }
  }
};
template <typename T> struct SigmoidOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) {
      return 1.0f / (1.0f + ::expf(-a));
    } else {
      double x = static_cast<double>(a);
      return static_cast<T>(1.0 / (1.0 + exp(-x)));
    }
  }
};
template <typename T> struct ExpM1Op {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) {
      return ::expm1f(a);
    } else {
      return static_cast<T>(expm1(static_cast<double>(a)));
    }
  }
};
template <typename T> struct Log1pOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) {
      return ::log1pf(a);
    } else {
      return static_cast<T>(log1p(static_cast<double>(a)));
    }
  }
};
template <typename T> struct FloorOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_integral_v<T>) {
      return a;
    } else if constexpr (std::is_same_v<T, __half>) {
      float x = __half2float(a);
      return __float2half_rn(::floorf(x));
#if VBT_CUDA_HAS_BF16
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      float x = __bfloat162float(a);
      return __float2bfloat16_rn(::floorf(x));
#endif
    } else if constexpr (std::is_same_v<T, float>) {
      return ::floorf(a);
    } else {
      return static_cast<T>(::floor(static_cast<double>(a)));
    }
  }
};
template <typename T> struct CeilOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_integral_v<T>) {
      return a;
    } else if constexpr (std::is_same_v<T, __half>) {
      float x = __half2float(a);
      return __float2half_rn(::ceilf(x));
#if VBT_CUDA_HAS_BF16
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      float x = __bfloat162float(a);
      return __float2bfloat16_rn(::ceilf(x));
#endif
    } else if constexpr (std::is_same_v<T, float>) {
      return ::ceilf(a);
    } else {
      return static_cast<T>(::ceil(static_cast<double>(a)));
    }
  }
};
template <typename T> struct TruncOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_integral_v<T>) {
      return a;
    } else if constexpr (std::is_same_v<T, __half>) {
      float x = __half2float(a);
      return __float2half_rn(::truncf(x));
#if VBT_CUDA_HAS_BF16
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      float x = __bfloat162float(a);
      return __float2bfloat16_rn(::truncf(x));
#endif
    } else if constexpr (std::is_same_v<T, float>) {
      return ::truncf(a);
    } else {
      return static_cast<T>(::trunc(static_cast<double>(a)));
    }
  }
};
template <typename T> struct RoundOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_integral_v<T>) {
      return a;
    } else if constexpr (std::is_same_v<T, __half>) {
      float x = __half2float(a);
      return __float2half_rn(::nearbyintf(x));
#if VBT_CUDA_HAS_BF16
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      float x = __bfloat162float(a);
      return __float2bfloat16_rn(::nearbyintf(x));
#endif
    } else if constexpr (std::is_same_v<T, float>) {
      return ::nearbyintf(a);
    } else {
      return static_cast<T>(::nearbyint(static_cast<double>(a)));
    }
  }
};
template <typename T> struct FracOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    double i;
    double f = modf(x, &i);
    return static_cast<T>(f);
  }
};
template <typename T> struct ReciprocalOp { __device__ T operator()(T a) const { return static_cast<T>(1.0 / static_cast<double>(a)); } };
template <typename T> struct SincOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    if (x == 0.0) return static_cast<T>(1.0);
    double pix = 3.14159265358979323846 * x;
    return static_cast<T>(sin(pix) / pix);
  }
};
template <typename T> struct SignOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    return static_cast<T>((x > 0) - (x < 0));
  }
};

template <typename T> struct SgnOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    if (isnan(x)) return static_cast<T>(0);
    return static_cast<T>((x > 0) - (x < 0));
  }
};

template <typename T> struct ConjPhysicalOp {
  __device__ T operator()(T a) const {
    return a;
  }
};

template <typename T> struct AngleOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    if (isnan(x)) return a;
    if (x == 0.0) return static_cast<T>(0.0);
    return static_cast<T>(x < 0.0 ? 3.14159265358979323846 : 0.0);
  }
};

template <typename T> struct SigmoidBackwardOp {
  __device__ T operator()(T grad, T out) const {
    double g = static_cast<double>(grad);
    double o = static_cast<double>(out);
    return static_cast<T>(g * o * (1.0 - o));
  }
};

template <typename T> struct SigmoidBackwardROp {
  __device__ T operator()(T out, T grad) const {
    double g = static_cast<double>(grad);
    double o = static_cast<double>(out);
    return static_cast<T>(g * o * (1.0 - o));
  }
};

template <typename T> struct TanhBackwardOp {
  __device__ T operator()(T grad, T out) const {
    double g = static_cast<double>(grad);
    double o = static_cast<double>(out);
    return static_cast<T>(g * (1.0 - o * o));
  }
};

template <typename T> struct TanhBackwardROp {
  __device__ T operator()(T out, T grad) const {
    double g = static_cast<double>(grad);
    double o = static_cast<double>(out);
    return static_cast<T>(g * (1.0 - o * o));
  }
};

template <typename T> struct SiluBackwardOp {
  __device__ T operator()(T grad, T x) const {
    double g = static_cast<double>(grad);
    double xd = static_cast<double>(x);
    double s = 1.0 / (1.0 + exp(-xd));
    double deriv = s * (1.0 + xd * (1.0 - s));
    return static_cast<T>(g * deriv);
  }
};

template <typename T> struct SiluBackwardROp {
  __device__ T operator()(T x, T grad) const {
    double g = static_cast<double>(grad);
    double xd = static_cast<double>(x);
    double s = 1.0 / (1.0 + exp(-xd));
    double deriv = s * (1.0 + xd * (1.0 - s));
    return static_cast<T>(g * deriv);
  }
};

template <typename T> struct LogitBackwardOp {
  __device__ T operator()(T grad, T x) const {
    double g = static_cast<double>(grad);
    double xd = static_cast<double>(x);
    return static_cast<T>(g / (xd * (1.0 - xd)));
  }
};

template <typename T> struct LogitBackwardROp {
  __device__ T operator()(T x, T grad) const {
    double g = static_cast<double>(grad);
    double xd = static_cast<double>(x);
    return static_cast<T>(g / (xd * (1.0 - xd)));
  }
};

template <typename T> struct Relu6Op {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    if (x < 0.0) x = 0.0;
    if (x > 6.0) x = 6.0;
    return static_cast<T>(x);
  }
};

template <typename T> struct HardtanhOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    if (x < -1.0) x = -1.0;
    if (x > 1.0) x = 1.0;
    return static_cast<T>(x);
  }
};

template <typename T> struct HardsigmoidOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    double y = x / 6.0 + 0.5;
    if (y < 0.0) y = 0.0;
    if (y > 1.0) y = 1.0;
    return static_cast<T>(y);
  }
};

template <typename T> struct SiluOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    double s = 1.0 / (1.0 + exp(-x));
    return static_cast<T>(x * s);
  }
};

template <typename T> struct GeluOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    // 0.5*x*(1+erf(x/sqrt(2)))
    double y = 0.5 * x * (1.0 + erf(x * 0.70710678118654752440));
    return static_cast<T>(y);
  }
};

template <typename T> struct SoftplusOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    // Stable softplus
    if (x > 20.0) return static_cast<T>(x);
    if (x < -20.0) return static_cast<T>(exp(x));
    return static_cast<T>(log1p(exp(x)));
  }
};

template <typename T> struct MishOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    double sp = (x > 20.0) ? x : log1p(exp(x));
    double y = x * tanh(sp);
    return static_cast<T>(y);
  }
};

template <typename T> struct EluOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    return static_cast<T>(x > 0.0 ? x : expm1(x));
  }
};

template <typename T> struct CeluOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    // alpha=1.0 default: same as ELU
    return static_cast<T>(x > 0.0 ? x : expm1(x));
  }
};

template <typename T> struct SeluOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    constexpr double alpha = 1.6732632423543772848170429916717;
    constexpr double scale = 1.0507009873554804934193349852946;
    double y = x > 0.0 ? x : (alpha * expm1(x));
    return static_cast<T>(scale * y);
  }
};

template <typename T> struct HardshrinkOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    constexpr double lambd = 0.5;
    if (x > lambd || x < -lambd) return static_cast<T>(x);
    return static_cast<T>(0.0);
  }
};

template <typename T> struct SoftshrinkOp {
  __device__ T operator()(T a) const {
    double x = static_cast<double>(a);
    constexpr double lambd = 0.5;
    if (x > lambd) return static_cast<T>(x - lambd);
    if (x < -lambd) return static_cast<T>(x + lambd);
    return static_cast<T>(0.0);
  }
};

template <typename T> struct BitwiseNotOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_integral_v<T>) {
      return ~a;
    } else {
      // Unsupported dtype (dispatcher should reject). Return 0 to avoid a silent
      // no-op if this functor is ever instantiated for non-integral types.
      return T(0);
    }
  }
};

template <typename T> struct IsfiniteOp {
  __device__ bool operator()(T a) const {
    if constexpr (std::is_same_v<T, __half>) {
      return ::isfinite(__half2float(a));
#if VBT_CUDA_HAS_BF16
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return ::isfinite(__bfloat162float(a));
#endif
    } else if constexpr (std::is_floating_point_v<T>) {
      return ::isfinite(a);
    } else {
      return true;
    }
  }
};

template <typename T> struct IsinfOp {
  __device__ bool operator()(T a) const {
    if constexpr (std::is_same_v<T, __half>) {
      return ::isinf(__half2float(a));
#if VBT_CUDA_HAS_BF16
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return ::isinf(__bfloat162float(a));
#endif
    } else if constexpr (std::is_floating_point_v<T>) {
      return ::isinf(a);
    } else {
      return false;
    }
  }
};

template <typename T> struct IsnanOp {
  __device__ bool operator()(T a) const {
    if constexpr (std::is_same_v<T, __half>) {
      return ::isnan(__half2float(a));
#if VBT_CUDA_HAS_BF16
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return ::isnan(__bfloat162float(a));
#endif
    } else if constexpr (std::is_floating_point_v<T>) {
      return ::isnan(a);
    } else {
      return false;
    }
  }
};

template <typename T> struct IsneginfOp {
  __device__ bool operator()(T a) const {
    if constexpr (std::is_same_v<T, __half>) {
      float x = __half2float(a);
      return ::isinf(x) && (x < 0.0f);
#if VBT_CUDA_HAS_BF16
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      float x = __bfloat162float(a);
      return ::isinf(x) && (x < 0.0f);
#endif
    } else if constexpr (std::is_floating_point_v<T>) {
      return ::isinf(a) && (a < T(0));
    } else {
      return false;
    }
  }
};

template <typename T> struct IsposinfOp {
  __device__ bool operator()(T a) const {
    if constexpr (std::is_same_v<T, __half>) {
      float x = __half2float(a);
      return ::isinf(x) && (x > 0.0f);
#if VBT_CUDA_HAS_BF16
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      float x = __bfloat162float(a);
      return ::isinf(x) && (x > 0.0f);
#endif
    } else if constexpr (std::is_floating_point_v<T>) {
      return ::isinf(a) && (a > T(0));
    } else {
      return false;
    }
  }
};

template <typename T> struct SignbitOp {
  __device__ bool operator()(T a) const {
    if constexpr (std::is_same_v<T, __half>) {
      return ::signbit(__half2float(a));
#if VBT_CUDA_HAS_BF16
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return ::signbit(__bfloat162float(a));
#endif
    } else if constexpr (std::is_floating_point_v<T>) {
      return ::signbit(a);
    } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      return a < 0;
    } else {
      return false;
    }
  }
};

template <typename T> struct LogicalNotOp {
  __device__ bool operator()(T a) const {
    return !static_cast<bool>(a);
  }
};

template <typename T> struct Exp2Op {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::exp2f(a);
    else return static_cast<T>(exp2(static_cast<double>(a)));
  }
};
template <typename T> struct Log2Op {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::log2f(a);
    else return static_cast<T>(log2(static_cast<double>(a)));
  }
};
template <typename T> struct Log10Op {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::log10f(a);
    else return static_cast<T>(log10(static_cast<double>(a)));
  }
};
template <typename T> struct SinHOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::sinhf(a);
    else return static_cast<T>(sinh(static_cast<double>(a)));
  }
};
template <typename T> struct CosHOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::coshf(a);
    else return static_cast<T>(cosh(static_cast<double>(a)));
  }
};
template <typename T> struct AsinhOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::asinhf(a);
    else return static_cast<T>(asinh(static_cast<double>(a)));
  }
};
template <typename T> struct AcoshOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::acoshf(a);
    else return static_cast<T>(acosh(static_cast<double>(a)));
  }
};
template <typename T> struct AtanhOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::atanhf(a);
    else return static_cast<T>(atanh(static_cast<double>(a)));
  }
};
template <typename T> struct Deg2RadOp {
  __device__ T operator()(T a) const {
    constexpr double k = 3.14159265358979323846 / 180.0;
    return static_cast<T>(static_cast<double>(a) * k);
  }
};
template <typename T> struct Rad2DegOp {
  __device__ T operator()(T a) const {
    constexpr double k = 180.0 / 3.14159265358979323846;
    return static_cast<T>(static_cast<double>(a) * k);
  }
};
template <typename T> struct TanOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::tanf(a);
    else return static_cast<T>(tan(static_cast<double>(a)));
  }
};
template <typename T> struct AsinOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::asinf(a);
    else return static_cast<T>(asin(static_cast<double>(a)));
  }
};
template <typename T> struct AcosOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::acosf(a);
    else return static_cast<T>(acos(static_cast<double>(a)));
  }
};
template <typename T> struct AtanOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::atanf(a);
    else return static_cast<T>(atan(static_cast<double>(a)));
  }
};
template <typename T> struct ErfOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::erff(a);
    else return static_cast<T>(erf(static_cast<double>(a)));
  }
};
template <typename T> struct ErfcOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::erfcf(a);
    else return static_cast<T>(erfc(static_cast<double>(a)));
  }
};
template <typename T> struct LgammaOp {
  __device__ T operator()(T a) const {
    if constexpr (std::is_same_v<T, float>) return ::lgammaf(a);
    else return static_cast<T>(lgamma(static_cast<double>(a)));
  }
};
template <typename T> struct PowOp {
  __device__ T operator()(T a, T b) const {
    if constexpr (std::is_same_v<T, float>) return ::powf(a, b);
    else return static_cast<T>(pow(static_cast<double>(a), static_cast<double>(b)));
  }
};
template <typename T> struct ClampOp {
  __device__ T operator()(T v, T mn, T mx) const {
    if constexpr (std::is_floating_point_v<T>) {
      if (isnan(mn) || isnan(mx)) return static_cast<T>(NAN);
    }
    return v < mn ? mn : (v > mx ? mx : v);
  }
};
template <typename T> struct LerpOp {
  __device__ T operator()(T start, T end, T weight) const {
    return start + weight * (end - start);
  }
};

template <typename T> struct ThresholdOp {
  __device__ T operator()(T x, T thr, T value) const {
    return x > thr ? x : value;
  }
};

template <typename T> struct LogitOp {
  __device__ T operator()(T x) const {
    if constexpr (std::is_floating_point_v<T>) {
      // Basic implementation: log(x / (1 - x))
      return std::log(x / (T(1) - x));
    }
    return T(0);
  }
};
// PolygammaOp: Computes the polygamma function psi^(n)(x).
// LIMITATIONS:
// - Only order n=2 (trigamma derivative) is currently implemented.
// - Other orders return NaN. This is a known limitation vs PyTorch which supports n>=0.
// - Iteration is capped at 100 to prevent divergence on extreme inputs.
// - Poles at non-positive integers return NaN.
// - Values x < -1e6 return NaN to avoid excessive iteration.
template <typename T> struct PolygammaOp {
  __device__ T operator()(T n_in, T x_in) const {
    if constexpr (std::is_floating_point_v<T>) {
        // Only n==2 is implemented. Other orders return NaN.
        if (!(n_in == static_cast<T>(2.0))) {
            return static_cast<T>(::nanf(""));
        }

        double x = static_cast<double>(x_in);

        // Polygamma has poles at non-positive integers.
        if (x <= 0.0 && x == ::floor(x)) {
            return static_cast<T>(::nanf(""));
        }

        // Guard against extreme negative values that would cause excessive iteration
        if (x < -1e6) {
            return static_cast<T>(::nanf(""));
        }

        double sum = 0.0;
        // Shift for precision with bounded iteration count
        int max_iter = 100;
        while (x < 10.0 && max_iter-- > 0) {
            sum -= 2.0 / (x * x * x);
            x += 1.0;
        }
        // Asymptotic: -1/x^2 - 1/x^3 - 1/2x^4
        double inv_x = 1.0 / x;
        double inv_x2 = inv_x * inv_x;
        double inv_x3 = inv_x2 * inv_x;
        double inv_x4 = inv_x3 * inv_x;
        double ans = -inv_x2 - inv_x3 - 0.5 * inv_x4;
        return static_cast<T>(sum + ans);
    }
    return T(0);
  }
};
template <typename T> struct FmaxOp {
    __device__ T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) {
            return ::fmax(a, b);
        } else {
            return a > b ? a : b;
        }
    }
};
template <typename T> struct FminOp {
    __device__ T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) {
            return ::fmin(a, b);
        } else {
            return a < b ? a : b;
        }
    }
};

template <typename T> struct HuberLossOp {
  T delta;
  HuberLossOp(T d) : delta(d) {}
  __device__ T operator()(T input, T target) const {
    T diff = input - target;
    T abs_diff = abs(diff);
    if (abs_diff < delta) {
      return T(0.5) * diff * diff;
    } else {
      return delta * (abs_diff - T(0.5) * delta);
    }
  }
};
template <typename T> struct MseLossOp {
  __device__ T operator()(T input, T target) const {
    T diff = input - target;
    return diff * diff;
  }
};
template <typename T> struct SmoothL1LossOp {
  T beta;
  SmoothL1LossOp(T b) : beta(b) {}
  __device__ T operator()(T input, T target) const {
    T diff = input - target;
    T abs_diff = abs(diff);
    if (abs_diff < beta) {
      return T(0.5) * diff * diff / beta;
    } else {
      return abs_diff - T(0.5) * beta;
    }
  }
};

template <typename T> struct HuberLossTernaryOp {
    __device__ T operator()(T input, T target, T delta) const {
      T diff = input - target;
      T abs_diff = abs(diff);
      if (abs_diff < delta) {
        return T(0.5) * diff * diff;
      } else {
        return delta * (abs_diff - T(0.5) * delta);
      }
    }
};

template <typename T> struct SmoothL1TernaryOp {
    __device__ T operator()(T input, T target, T beta) const {
      T diff = input - target;
      T abs_diff = abs(diff);
      if (abs_diff < beta) {
        return T(0.5) * diff * diff / beta;
      } else {
        return abs_diff - T(0.5) * beta;
      }
    }
};

template <typename T, typename Op, typename index_t>
__global__ void binary_scalar_kernel(T* out, const T* a, T b, index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    out[i] = op(a[i], b);
  }
}

template <typename S, typename Op, typename index_t>
__global__ void binary_scalar_kernel_fp16bf16(S* out, const S* a, float b_val, index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    float ax = detail::load_as_float(a + i);
    detail::store_from_float(out + i, op(ax, b_val));
  }
}

template <typename T, typename Op, typename index_t>
__global__ void binary_dense_vectorized_kernel(T* out, const T* a, const T* b, index_t N, Op op) {
  static_assert(std::is_same_v<T, float>, "Vectorized kernel only supports float");
  using vec_t = float4;
  vec_t* out_v = reinterpret_cast<vec_t*>(out);
  const vec_t* a_v = reinterpret_cast<const vec_t*>(a);
  const vec_t* b_v = reinterpret_cast<const vec_t*>(b);
  index_t num_vec = N / 4;
  
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  
  for (; i < num_vec; i += stride) {
    vec_t av = a_v[i];
    vec_t bv = b_v[i];
    vec_t rv;
    rv.x = op(av.x, bv.x);
    rv.y = op(av.y, bv.y);
    rv.z = op(av.z, bv.z);
    rv.w = op(av.w, bv.w);
    out_v[i] = rv;
  }
}

template <typename T, typename Op, typename index_t>
__global__ void binary_dense_kernel(T* out, const T* a, const T* b, index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    out[i] = op(a[i], b[i]);
  }
}

template <typename T, typename Op, typename index_t>
__global__ void unary_dense_kernel(T* out, const T* a, index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    out[i] = op(a[i]);
  }
}

template <typename T, typename Op, typename index_t>
__global__ void unary_ti_strided_kernel(
    T* out, DeviceStrideMeta mo,
    const T* base_a, DeviceStrideMeta ma,
    index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    std::int64_t off_o = compute_offset_elems(static_cast<std::int64_t>(i), mo);
    std::int64_t off_a = compute_offset_elems(static_cast<std::int64_t>(i), ma);
    out[off_o] = op(base_a[off_a]);
  }
}

template <typename S, typename Op, typename index_t>
__global__ void unary_dense_kernel_fp16bf16(S* out, const S* a, index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    float ax = detail::load_as_float(a + i);
    detail::store_from_float(out + i, op(ax));
  }
}

template <typename S, typename Op, typename index_t>
__global__ void unary_ti_strided_kernel_fp16bf16(
    S* out, DeviceStrideMeta mo,
    const S* base_a, DeviceStrideMeta ma,
    index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    std::int64_t off_o = compute_offset_elems(static_cast<std::int64_t>(i), mo);
    std::int64_t off_a = compute_offset_elems(static_cast<std::int64_t>(i), ma);
    float ax = detail::load_as_float(base_a + off_a);
    detail::store_from_float(out + off_o, op(ax));
  }
}

template <typename T, typename Op, typename index_t>
__global__ void unary_to_bool_dense_kernel(uint8_t* out, const T* a, index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    out[i] = static_cast<uint8_t>(op(a[i]));
  }
}

template <typename S, typename Op, typename index_t>
__global__ void unary_to_bool_dense_kernel_fp16bf16(uint8_t* out, const S* a, index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    float ax = detail::load_as_float(a + i);
    out[i] = static_cast<uint8_t>(op(ax));
  }
}

template <typename T, typename Op, typename index_t>
__global__ void unary_to_bool_ti_strided_kernel(
    uint8_t* out, DeviceStrideMeta mo,
    const T* base_a, DeviceStrideMeta ma,
    index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    std::int64_t off_o = compute_offset_elems(static_cast<std::int64_t>(i), mo);
    std::int64_t off_a = compute_offset_elems(static_cast<std::int64_t>(i), ma);
    out[off_o] = static_cast<uint8_t>(op(base_a[off_a]));
  }
}

template <typename S, typename Op, typename index_t>
__global__ void unary_to_bool_ti_strided_kernel_fp16bf16(
    uint8_t* out, DeviceStrideMeta mo,
    const S* base_a, DeviceStrideMeta ma,
    index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    std::int64_t off_o = compute_offset_elems(static_cast<std::int64_t>(i), mo);
    std::int64_t off_a = compute_offset_elems(static_cast<std::int64_t>(i), ma);
    float ax = detail::load_as_float(base_a + off_a);
    out[off_o] = static_cast<uint8_t>(op(ax));
  }
}

template <typename index_t>
__global__ void nan_to_num_dense_kernel(float* out, const float* a, index_t N) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  const float kPos = FLT_MAX;
  const float kNeg = -FLT_MAX;
  for (; i < N; i += stride) {
    float v = a[i];
    if (isnan(v)) {
      v = 0.0f;
    } else if (isinf(v)) {
      v = (v > 0.0f) ? kPos : kNeg;
    }
    out[i] = v;
  }
}

template <typename index_t>
__global__ void nan_to_num_ti_strided_kernel(
    float* out, DeviceStrideMeta mo,
    const float* base_a, DeviceStrideMeta ma,
    index_t N) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  const float kPos = FLT_MAX;
  const float kNeg = -FLT_MAX;
  for (; i < N; i += stride) {
    std::int64_t off_o = compute_offset_elems(static_cast<std::int64_t>(i), mo);
    std::int64_t off_a = compute_offset_elems(static_cast<std::int64_t>(i), ma);
    float v = base_a[off_a];
    if (isnan(v)) {
      v = 0.0f;
    } else if (isinf(v)) {
      v = (v > 0.0f) ? kPos : kNeg;
    }
    out[off_o] = v;
  }
}

struct NanToNumStridedLauncher {
  void operator()(const TensorIter& iter, bool use32, void* stream_ptr) {
    const std::int64_t N_iter = iter.numel();
    if (N_iter == 0) return;

    auto* out = static_cast<float*>(iter.operand(0).data);
    auto* a = static_cast<const float*>(iter.operand(1).data);

    DeviceStrideMeta mo{}, ma{};
    iter.export_device_meta(0, &mo, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(1, &ma, VBT_CUDA_BROADCAST_MAX_NDIM);

    dim3 grid, block;
    launch_bounds_and_grid(N_iter, grid, block);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (use32) {
      nan_to_num_ti_strided_kernel<int32_t><<<grid, block, 0, s>>>(
          out, mo, a, ma, static_cast<int32_t>(N_iter));
    } else {
      nan_to_num_ti_strided_kernel<int64_t><<<grid, block, 0, s>>>(
          out, mo, a, ma, static_cast<int64_t>(N_iter));
    }
  }
};

template <typename T, typename index_t>
__global__ void relu_dense_kernel(T* out, const T* a, index_t N) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    T v = a[i];
    out[i] = v > T(0) ? v : T(0);
  }
}

// FP16/BF16 helpers: compute in float and convert
template <typename S, typename Op, typename index_t>
__global__ void binary_dense_kernel_fp16bf16(S* out, const S* a, const S* b, index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    float ax = detail::load_as_float(a + i);
    float bx = detail::load_as_float(b + i);
    detail::store_from_float(out + i, op(ax, bx));
  }
}

template <typename S, typename index_t>
__global__ void relu_dense_kernel_fp16bf16(S* out, const S* a, index_t N) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    float av = detail::load_as_float(a + i);
    float rv = av > 0.f ? av : 0.f;
    detail::store_from_float(out + i, rv);
  }
}

// Bool relu: identity
template <typename index_t>
__global__ void relu_bool_dense_kernel(uint8_t* out, const uint8_t* a, index_t N) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    out[i] = a[i];
  }
}

// TI-based strided kernels using DeviceStrideMeta (reused for broadcast fallback)

// Unary launcher for TI strided paths

template <typename T, typename Op>
struct UnaryStridedLauncher {
  Op op;
  void operator()(const TensorIter& iter, bool use32, void* stream_ptr) {
    const std::int64_t N_iter = iter.numel();
    if (N_iter == 0) return;

    auto* out = static_cast<T*>(iter.operand(0).data);
    auto* a = static_cast<const T*>(iter.operand(1).data);

    DeviceStrideMeta mo{}, ma{};
    iter.export_device_meta(0, &mo, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(1, &ma, VBT_CUDA_BROADCAST_MAX_NDIM);

    dim3 grid, block;
    launch_bounds_and_grid(N_iter, grid, block);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (use32) {
      unary_ti_strided_kernel<T, Op, int32_t><<<grid, block, 0, s>>>(
          out, mo, a, ma, static_cast<int32_t>(N_iter), op);
    } else {
      unary_ti_strided_kernel<T, Op, int64_t><<<grid, block, 0, s>>>(
          out, mo, a, ma, static_cast<int64_t>(N_iter), op);
    }
  }
};

template <typename S, typename Op>
struct UnaryStridedLauncherMixed {
  Op op;
  void operator()(const TensorIter& iter, bool use32, void* stream_ptr) {
    const std::int64_t N_iter = iter.numel();
    if (N_iter == 0) return;

    auto* out = static_cast<S*>(iter.operand(0).data);
    auto* a = static_cast<const S*>(iter.operand(1).data);

    DeviceStrideMeta mo{}, ma{};
    iter.export_device_meta(0, &mo, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(1, &ma, VBT_CUDA_BROADCAST_MAX_NDIM);

    dim3 grid, block;
    launch_bounds_and_grid(N_iter, grid, block);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (use32) {
      unary_ti_strided_kernel_fp16bf16<S, Op, int32_t><<<grid, block, 0, s>>>(
          out, mo, a, ma, static_cast<int32_t>(N_iter), op);
    } else {
      unary_ti_strided_kernel_fp16bf16<S, Op, int64_t><<<grid, block, 0, s>>>(
          out, mo, a, ma, static_cast<int64_t>(N_iter), op);
    }
  }
};

template <typename T, typename Op>
struct UnaryToBoolStridedLauncher {
  Op op;

  void operator()(const TensorIter& iter, bool use32, void* stream_ptr) {
    const std::int64_t N_iter = iter.numel();
    if (N_iter == 0) return;

    auto* out = static_cast<uint8_t*>(iter.operand(0).data);
    auto* a = static_cast<const T*>(iter.operand(1).data);

    DeviceStrideMeta mo{}, ma{};
    iter.export_device_meta(0, &mo, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(1, &ma, VBT_CUDA_BROADCAST_MAX_NDIM);

    dim3 grid, block;
    launch_bounds_and_grid(N_iter, grid, block);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (use32) {
      unary_to_bool_ti_strided_kernel<T, Op, int32_t><<<grid, block, 0, s>>>(
          out, mo, a, ma, static_cast<int32_t>(N_iter), op);
    } else {
      unary_to_bool_ti_strided_kernel<T, Op, int64_t><<<grid, block, 0, s>>>(
          out, mo, a, ma, static_cast<int64_t>(N_iter), op);
    }
  }
};

template <typename S, typename Op>
struct UnaryToBoolStridedLauncherMixed {
  Op op;

  void operator()(const TensorIter& iter, bool use32, void* stream_ptr) {
    const std::int64_t N_iter = iter.numel();
    if (N_iter == 0) return;

    auto* out = static_cast<uint8_t*>(iter.operand(0).data);
    auto* a = static_cast<const S*>(iter.operand(1).data);

    DeviceStrideMeta mo{}, ma{};
    iter.export_device_meta(0, &mo, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(1, &ma, VBT_CUDA_BROADCAST_MAX_NDIM);

    dim3 grid, block;
    launch_bounds_and_grid(N_iter, grid, block);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (use32) {
      unary_to_bool_ti_strided_kernel_fp16bf16<S, Op, int32_t><<<grid, block, 0, s>>>(
          out, mo, a, ma, static_cast<int32_t>(N_iter), op);
    } else {
      unary_to_bool_ti_strided_kernel_fp16bf16<S, Op, int64_t><<<grid, block, 0, s>>>(
          out, mo, a, ma, static_cast<int64_t>(N_iter), op);
    }
  }
};

// Unary dispatcher (TI-backed strided fallback)

template <template <class> class OpT>
static TensorImpl unary_op_dispatcher(const TensorImpl& a, const char* name) {
#if VBT_WITH_CUDA
  if (a.device().type != kDLCUDA) {
    throw std::runtime_error(std::string(name) + ": expected CUDA tensor");
  }
  DeviceGuard g(a.device().index);
  const int64_t N = numel_of(a);
  auto out = make_cuda_dense_out(a.sizes(), a.dtype(), a.device());
  if (N == 0) return out;

  bool use32 = vbt::cuda_detail::should_use_int32_index(N);
  dim3 grid, block; launch_bounds_and_grid(N, grid, block);
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(a.device().index));

  auto run_dense = [&](auto tag) {
    using T = decltype(tag);
    if (use32) unary_dense_kernel<T, OpT<T>, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
      static_cast<T*>(out.data()), static_cast<const T*>(a.data()), static_cast<int32_t>(N), OpT<T>{});
    else unary_dense_kernel<T, OpT<T>, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
      static_cast<T*>(out.data()), static_cast<const T*>(a.data()), static_cast<int64_t>(N), OpT<T>{});
  };

  if (a.is_contiguous()) {
    if (a.dtype() == ScalarType::Float32) {
      run_dense(float(0));
    } else if (a.dtype() == ScalarType::Int64) {
      run_dense(static_cast<long long>(0));
    } else if (a.dtype() == ScalarType::Float16) {
      if (use32) {
        unary_dense_kernel_fp16bf16<__half, OpT<float>, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<__half*>(out.data()),
            static_cast<const __half*>(a.data()),
            static_cast<int32_t>(N),
            OpT<float>{});
      } else {
        unary_dense_kernel_fp16bf16<__half, OpT<float>, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<__half*>(out.data()),
            static_cast<const __half*>(a.data()),
            static_cast<int64_t>(N),
            OpT<float>{});
      }
    }
#if VBT_CUDA_HAS_BF16
    else if (a.dtype() == ScalarType::BFloat16) {
      if (use32) {
        unary_dense_kernel_fp16bf16<__nv_bfloat16, OpT<float>, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<__nv_bfloat16*>(out.data()),
            static_cast<const __nv_bfloat16*>(a.data()),
            static_cast<int32_t>(N),
            OpT<float>{});
      } else {
        unary_dense_kernel_fp16bf16<__nv_bfloat16, OpT<float>, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<__nv_bfloat16*>(out.data()),
            static_cast<const __nv_bfloat16*>(a.data()),
            static_cast<int64_t>(N),
            OpT<float>{});
      }
    }
#endif
    else {
      throw std::invalid_argument(std::string(name) + ": unsupported dtype");
    }
    vbt::cuda::record_stream(out.storage(), stream);
    vbt::cuda::record_stream(a.storage(), stream);
    return out;
  }

  TensorIterConfig cfg;
  cfg.check_mem_overlap(true);
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true), IterOperandRole::WriteOnly, /*allow_resize=*/true);
  cfg.add_input(a);

  TensorIter iter;
  try {
    iter = cfg.build();
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string(name) + ": " + e.what());
  }

#ifdef VBT_TI_DEBUG
  {
    const std::int64_t N_iter = iter.numel();
    assert(N_iter == N);
    assert(iter.ndim() <= VBT_CUDA_BROADCAST_MAX_NDIM);
  }
#endif

  VBT_TI_STATS_INC(cuda_ti_kernel_launches);

  auto run_ti = [&](auto tag) {
    using T = decltype(tag);
    UnaryStridedLauncher<T, OpT<T>> launcher{OpT<T>{}};
    ::vbt::core::ti_gpu_kernel(iter, launcher);
  };

  auto run_ti_mixed = [&](auto storage_tag) {
    using S = decltype(storage_tag);
    UnaryStridedLauncherMixed<S, OpT<float>> launcher{OpT<float>{}};
    ::vbt::core::ti_gpu_kernel(iter, launcher);
  };

  if (a.dtype() == ScalarType::Float32) {
    run_ti(float(0));
  } else if (a.dtype() == ScalarType::Int64) {
    run_ti(static_cast<long long>(0));
  } else if (a.dtype() == ScalarType::Float16) {
    run_ti_mixed(__half{});
  }
#if VBT_CUDA_HAS_BF16
  else if (a.dtype() == ScalarType::BFloat16) {
    run_ti_mixed(__nv_bfloat16{});
  }
#endif
  else {
    throw std::invalid_argument(std::string(name) + ": unsupported dtype");
  }

  vbt::cuda::record_stream(out.storage(), stream);
  vbt::cuda::record_stream(a.storage(), stream);
  return out;
#else
  (void)a; (void)name; throw std::runtime_error("CUDA not built");
#endif
}

template <template <class> class OpT>
static TensorImpl unary_bitwise_dispatcher(const TensorImpl& a, const char* name) {
#if VBT_WITH_CUDA
  if (a.device().type != kDLCUDA) {
    throw std::runtime_error(std::string(name) + ": expected CUDA tensor");
  }
  DeviceGuard g(a.device().index);
  const int64_t N = numel_of(a);
  auto out = make_cuda_dense_out(a.sizes(), a.dtype(), a.device());
  if (N == 0) return out;

  bool use32 = vbt::cuda_detail::should_use_int32_index(N);
  dim3 grid, block;
  launch_bounds_and_grid(N, grid, block);
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(a.device().index));

  auto run_dense = [&](auto tag) {
    using T = decltype(tag);
    if (use32) {
      unary_dense_kernel<T, OpT<T>, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<T*>(out.data()),
          static_cast<const T*>(a.data()),
          static_cast<int32_t>(N),
          OpT<T>{});
    } else {
      unary_dense_kernel<T, OpT<T>, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<T*>(out.data()),
          static_cast<const T*>(a.data()),
          static_cast<int64_t>(N),
          OpT<T>{});
    }
  };

  if (a.is_contiguous()) {
    if (a.dtype() == ScalarType::Int32) {
      run_dense(static_cast<std::int32_t>(0));
    } else if (a.dtype() == ScalarType::Int64) {
      run_dense(static_cast<long long>(0));
    } else {
      throw std::invalid_argument(std::string(name) + ": unsupported dtype");
    }
    vbt::cuda::record_stream(out.storage(), stream);
    vbt::cuda::record_stream(a.storage(), stream);
    return out;
  }

  TensorIterConfig cfg;
  cfg.check_mem_overlap(true);
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true), IterOperandRole::WriteOnly, /*allow_resize=*/true);
  cfg.add_input(a);

  TensorIter iter;
  try {
    iter = cfg.build();
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string(name) + ": " + e.what());
  }

  VBT_TI_STATS_INC(cuda_ti_kernel_launches);

  auto run_ti = [&](auto tag) {
    using T = decltype(tag);
    UnaryStridedLauncher<T, OpT<T>> launcher{OpT<T>{}};
    ::vbt::core::ti_gpu_kernel(iter, launcher);
  };

  if (a.dtype() == ScalarType::Int32) {
    run_ti(static_cast<std::int32_t>(0));
  } else if (a.dtype() == ScalarType::Int64) {
    run_ti(static_cast<long long>(0));
  } else {
    throw std::invalid_argument(std::string(name) + ": unsupported dtype");
  }

  vbt::cuda::record_stream(out.storage(), stream);
  vbt::cuda::record_stream(a.storage(), stream);
  return out;
#else
  (void)a; (void)name; throw std::runtime_error("CUDA not built");
#endif
}


template <template <class> class OpT>
static TensorImpl unary_bool_op_dispatcher(const TensorImpl& a, const char* name) {
#if VBT_WITH_CUDA
  if (a.device().type != kDLCUDA) {
    throw std::runtime_error(std::string(name) + ": expected CUDA tensor");
  }
  DeviceGuard g(a.device().index);
  const int64_t N = numel_of(a);
  auto out = make_cuda_dense_out(a.sizes(), ScalarType::Bool, a.device());
  if (N == 0) return out;

  bool use32 = vbt::cuda_detail::should_use_int32_index(N);
  dim3 grid, block;
  launch_bounds_and_grid(N, grid, block);
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(a.device().index));

  if (a.is_contiguous()) {
    if (a.dtype() == ScalarType::Float32) {
      if (use32) {
        unary_to_bool_dense_kernel<float, OpT<float>, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<uint8_t*>(out.data()),
            static_cast<const float*>(a.data()),
            static_cast<int32_t>(N),
            OpT<float>{});
      } else {
        unary_to_bool_dense_kernel<float, OpT<float>, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<uint8_t*>(out.data()),
            static_cast<const float*>(a.data()),
            static_cast<int64_t>(N),
            OpT<float>{});
      }
    } else if (a.dtype() == ScalarType::Int64) {
      if (use32) {
        unary_to_bool_dense_kernel<long long, OpT<long long>, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<uint8_t*>(out.data()),
            static_cast<const long long*>(a.data()),
            static_cast<int32_t>(N),
            OpT<long long>{});
      } else {
        unary_to_bool_dense_kernel<long long, OpT<long long>, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<uint8_t*>(out.data()),
            static_cast<const long long*>(a.data()),
            static_cast<int64_t>(N),
            OpT<long long>{});
      }
    } else if (a.dtype() == ScalarType::Bool) {
      if (use32) {
        unary_to_bool_dense_kernel<uint8_t, OpT<uint8_t>, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<uint8_t*>(out.data()),
            static_cast<const uint8_t*>(a.data()),
            static_cast<int32_t>(N),
            OpT<uint8_t>{});
      } else {
        unary_to_bool_dense_kernel<uint8_t, OpT<uint8_t>, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<uint8_t*>(out.data()),
            static_cast<const uint8_t*>(a.data()),
            static_cast<int64_t>(N),
            OpT<uint8_t>{});
      }
    } else if (a.dtype() == ScalarType::Float16) {
      if (use32) {
        unary_to_bool_dense_kernel_fp16bf16<__half, OpT<float>, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<uint8_t*>(out.data()),
            static_cast<const __half*>(a.data()),
            static_cast<int32_t>(N),
            OpT<float>{});
      } else {
        unary_to_bool_dense_kernel_fp16bf16<__half, OpT<float>, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<uint8_t*>(out.data()),
            static_cast<const __half*>(a.data()),
            static_cast<int64_t>(N),
            OpT<float>{});
      }
    }
#if VBT_CUDA_HAS_BF16
    else if (a.dtype() == ScalarType::BFloat16) {
      if (use32) {
        unary_to_bool_dense_kernel_fp16bf16<__nv_bfloat16, OpT<float>, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<uint8_t*>(out.data()),
            static_cast<const __nv_bfloat16*>(a.data()),
            static_cast<int32_t>(N),
            OpT<float>{});
      } else {
        unary_to_bool_dense_kernel_fp16bf16<__nv_bfloat16, OpT<float>, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<uint8_t*>(out.data()),
            static_cast<const __nv_bfloat16*>(a.data()),
            static_cast<int64_t>(N),
            OpT<float>{});
      }
    }
#endif
    else {
      throw std::invalid_argument(std::string(name) + ": unsupported dtype");
    }

    vbt::cuda::record_stream(out.storage(), stream);
    vbt::cuda::record_stream(a.storage(), stream);
    return out;
  }

  TensorIterConfig cfg;
  cfg.check_mem_overlap(true);
  cfg.check_all_same_dtype(false);
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true), IterOperandRole::WriteOnly, /*allow_resize=*/true);
  cfg.add_input(a);

  TensorIter iter;
  try {
    iter = cfg.build();
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string(name) + ": " + e.what());
  }

#ifdef VBT_TI_DEBUG
  {
    const std::int64_t N_iter = iter.numel();
    assert(N_iter == N);
    assert(iter.ndim() <= VBT_CUDA_BROADCAST_MAX_NDIM);
  }
#endif

  VBT_TI_STATS_INC(cuda_ti_kernel_launches);

  auto run_ti = [&](auto tag) {
    using T = decltype(tag);
    UnaryToBoolStridedLauncher<T, OpT<T>> launcher{OpT<T>{}};
    ::vbt::core::ti_gpu_kernel(iter, launcher);
  };

  if (a.dtype() == ScalarType::Float32) {
    run_ti(float(0));
  } else if (a.dtype() == ScalarType::Int64) {
    run_ti(static_cast<long long>(0));
  } else if (a.dtype() == ScalarType::Bool) {
    run_ti(static_cast<uint8_t>(0));
  } else if (a.dtype() == ScalarType::Float16) {
    UnaryToBoolStridedLauncherMixed<__half, OpT<float>> launcher{OpT<float>{}};
    ::vbt::core::ti_gpu_kernel(iter, launcher);
  }
#if VBT_CUDA_HAS_BF16
  else if (a.dtype() == ScalarType::BFloat16) {
    UnaryToBoolStridedLauncherMixed<__nv_bfloat16, OpT<float>> launcher{OpT<float>{}};
    ::vbt::core::ti_gpu_kernel(iter, launcher);
  }
#endif
  else {
    throw std::invalid_argument(std::string(name) + ": unsupported dtype");
  }

  vbt::cuda::record_stream(out.storage(), stream);
  vbt::cuda::record_stream(a.storage(), stream);
  return out;
#else
  (void)a; (void)name;
  throw std::runtime_error("CUDA not built");
#endif
}

static TensorImpl nan_to_num_op_dispatcher(const TensorImpl& a, const char* name) {
#if VBT_WITH_CUDA
  if (a.device().type != kDLCUDA) {
    throw std::runtime_error(std::string(name) + ": expected CUDA tensor");
  }
  if (a.dtype() != ScalarType::Float32) {
    throw std::invalid_argument(std::string(name) + ": float32 tensor expected");
  }

  DeviceGuard g(a.device().index);
  const int64_t N = numel_of(a);
  auto out = make_cuda_dense_out(a.sizes(), a.dtype(), a.device());
  if (N == 0) return out;

  bool use32 = vbt::cuda_detail::should_use_int32_index(N);
  dim3 grid, block;
  launch_bounds_and_grid(N, grid, block);
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(a.device().index));

  if (a.is_contiguous()) {
    if (use32) {
      nan_to_num_dense_kernel<int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<float*>(out.data()),
          static_cast<const float*>(a.data()),
          static_cast<int32_t>(N));
    } else {
      nan_to_num_dense_kernel<int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<float*>(out.data()),
          static_cast<const float*>(a.data()),
          static_cast<int64_t>(N));
    }
    vbt::cuda::record_stream(out.storage(), stream);
    vbt::cuda::record_stream(a.storage(), stream);
    return out;
  }

  TensorIterConfig cfg;
  cfg.check_mem_overlap(true);
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true), IterOperandRole::WriteOnly, /*allow_resize=*/true);
  cfg.add_input(a);

  TensorIter iter;
  try {
    iter = cfg.build();
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string(name) + ": " + e.what());
  }

  VBT_TI_STATS_INC(cuda_ti_kernel_launches);

  NanToNumStridedLauncher launcher{};
  ::vbt::core::ti_gpu_kernel(iter, launcher);

  vbt::cuda::record_stream(out.storage(), stream);
  vbt::cuda::record_stream(a.storage(), stream);
  return out;
#else
  (void)a; (void)name;
  throw std::runtime_error("CUDA not built");
#endif
}

// Binary launcher

template <typename T, typename Op, typename index_t>
__global__ void binary_ti_strided_kernel(
    T* out,
    const T* base_a, DeviceStrideMeta ma,
    const T* base_b, DeviceStrideMeta mb,
    index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    std::int64_t off_a = compute_offset_elems(static_cast<std::int64_t>(i), ma);
    std::int64_t off_b = compute_offset_elems(static_cast<std::int64_t>(i), mb);
    out[i] = op(base_a[off_a], base_b[off_b]);
  }
}

template <typename S, typename Op, typename index_t>
__global__ void binary_ti_strided_kernel_fp16bf16(
    S* out,
    const S* base_a, DeviceStrideMeta ma,
    const S* base_b, DeviceStrideMeta mb,
    index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    std::int64_t off_a = compute_offset_elems(static_cast<std::int64_t>(i), ma);
    std::int64_t off_b = compute_offset_elems(static_cast<std::int64_t>(i), mb);
    float ax = detail::load_as_float(base_a + off_a);
    float bx = detail::load_as_float(base_b + off_b);
    detail::store_from_float(out + i, op(ax, bx));
  }
}




template <typename S, typename index_t>
__global__ void relu_strided_to_contig_kernel_fp16bf16(S* out, const S* base_a, DeviceStrideMeta ma, index_t N) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    int64_t off_a = compute_offset_elems(static_cast<int64_t>(i), ma);
    float av = detail::load_as_float(base_a + off_a);
    float rv = av > 0.f ? av : 0.f;
    detail::store_from_float(out + i, rv);
  }
}

template <typename T, typename index_t>
__global__ void relu_strided_to_contig_kernel(T* out, const T* base_a, DeviceStrideMeta ma, index_t N) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    int64_t off_a = compute_offset_elems(static_cast<int64_t>(i), ma);
    T v = base_a[off_a];
    out[i] = v > T(0) ? v : T(0);
  }
}

template <typename index_t>
__global__ void relu_bool_strided_to_contig_kernel(uint8_t* out, const uint8_t* base_a, DeviceStrideMeta ma, index_t N) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    int64_t off_a = compute_offset_elems(static_cast<int64_t>(i), ma);
    out[i] = base_a[off_a];
  }
}

// Broadcast helpers

static inline DeviceStrideMeta make_meta_aligned(const TensorImpl& t, const std::vector<int64_t>& out_sizes) {
  DeviceStrideMeta m{};
  const auto& sz = t.sizes();
  const auto& st = t.strides();
  const std::size_t out_nd = out_sizes.size();
  if (out_nd > static_cast<std::size_t>(VBT_CUDA_BROADCAST_MAX_NDIM)) {
    throw std::invalid_argument(
        std::string("vt::broadcast: CUDA broadcast fallback supports up to ") +
        std::to_string(VBT_CUDA_BROADCAST_MAX_NDIM) + "D (got more)");
  }
  m.ndim = static_cast<int64_t>(out_nd);
  const std::size_t in_nd = sz.size();
  const std::size_t pad = out_nd - in_nd;
  for (std::size_t i = 0; i < out_nd; ++i) {
    const int64_t out_dim = out_sizes[i];
    m.sizes[i] = out_dim;
    if (i < pad) {
      // Leading broadcasted dims for input
      m.strides[i] = 0;
    } else {
      const int64_t in_dim = sz[i - pad];
      if (in_dim == out_dim) {
        m.strides[i] = st[i - pad];
      } else if (in_dim == 1) {
        // Broadcast along this axis
        m.strides[i] = 0;
      } else {
        // Not broadcastable; should have been rejected earlier
        throw std::invalid_argument("vt::broadcast: shapes are not broadcastable");
      }
    }
  }
  return m;
}

static inline DeviceStrideMeta make_meta_ti_order(const TensorImpl& t) {
  DeviceStrideMeta m{};
  const auto& sz = t.sizes();
  const auto& st = t.strides();
  const std::size_t nd = sz.size();
  if (nd > static_cast<std::size_t>(VBT_CUDA_BROADCAST_MAX_NDIM)) {
    throw std::invalid_argument(
        std::string("vt::relu: CUDA strided kernel supports up to ") +
        std::to_string(VBT_CUDA_BROADCAST_MAX_NDIM) + "D (got " +
        std::to_string(nd) + "D)");
  }
  m.ndim = static_cast<int64_t>(nd);
  for (std::size_t i = 0; i < nd; ++i) {
    const std::size_t src = nd - 1 - i;
    m.sizes[i] = sz[src];
    m.strides[i] = st[src];
  }
  return m;
}

#endif // VBT_WITH_CUDA

template <typename T, typename Op>
struct BinaryStridedLauncher {
  Op op;

  void operator()(const TensorIter& iter, bool use32, void* stream_ptr) {
    const int64_t N_iter = iter.numel();
    if (N_iter == 0) return;

    auto* out = static_cast<T*>(iter.operand(0).data);
    auto* a = static_cast<const T*>(iter.operand(1).data);
    auto* b = static_cast<const T*>(iter.operand(2).data);

    DeviceStrideMeta ma{}, mb{};
    iter.export_device_meta(1, &ma, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(2, &mb, VBT_CUDA_BROADCAST_MAX_NDIM);

    dim3 grid, block;
    launch_bounds_and_grid(N_iter, grid, block);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    if (use32) {
        binary_ti_strided_kernel<T, Op, int32_t><<<grid, block, 0, stream>>>(out, a, ma, b, mb, static_cast<int32_t>(N_iter), op);
    } else {
        binary_ti_strided_kernel<T, Op, int64_t><<<grid, block, 0, stream>>>(out, a, ma, b, mb, static_cast<int64_t>(N_iter), op);
    }
  }
};

template <typename S, typename Op>
struct BinaryStridedLauncherMixed {
  Op op;

  void operator()(const TensorIter& iter, bool use32, void* stream_ptr) {
    const int64_t N_iter = iter.numel();
    if (N_iter == 0) return;

    auto* out = static_cast<S*>(iter.operand(0).data);
    auto* a = static_cast<const S*>(iter.operand(1).data);
    auto* b = static_cast<const S*>(iter.operand(2).data);

    DeviceStrideMeta ma{}, mb{};
    iter.export_device_meta(1, &ma, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(2, &mb, VBT_CUDA_BROADCAST_MAX_NDIM);

    dim3 grid, block;
    launch_bounds_and_grid(N_iter, grid, block);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    if (use32) {
        binary_ti_strided_kernel_fp16bf16<S, Op, int32_t><<<grid, block, 0, stream>>>(out, a, ma, b, mb, static_cast<int32_t>(N_iter), op);
    } else {
        binary_ti_strided_kernel_fp16bf16<S, Op, int64_t><<<grid, block, 0, stream>>>(out, a, ma, b, mb, static_cast<int64_t>(N_iter), op);
    }
  }
};

template <typename T> struct EqOp { __device__ bool operator()(T a, T b) const { return a == b; } };
template <typename T> struct NeOp { __device__ bool operator()(T a, T b) const { return a != b; } };
template <typename T> struct LtOp { __device__ bool operator()(T a, T b) const { return a < b; } };
template <typename T> struct GtOp { __device__ bool operator()(T a, T b) const { return a > b; } };
template <typename T> struct LeOp { __device__ bool operator()(T a, T b) const { return a <= b; } };
template <typename T> struct GeOp { __device__ bool operator()(T a, T b) const { return a >= b; } };

template <typename T, typename Op, typename index_t>
__global__ void comparison_ti_strided_kernel(
    uint8_t* out,
    const T* base_a, DeviceStrideMeta ma,
    const T* base_b, DeviceStrideMeta mb,
    index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    std::int64_t off_a = compute_offset_elems(static_cast<std::int64_t>(i), ma);
    std::int64_t off_b = compute_offset_elems(static_cast<std::int64_t>(i), mb);
    out[i] = static_cast<uint8_t>(op(base_a[off_a], base_b[off_b]));
  }
}

template <typename T, typename Op>
struct ComparisonLauncher {
  Op op;

  void operator()(const TensorIter& iter, bool use32, void* stream_ptr) {
    const int64_t N_iter = iter.numel();
    if (N_iter == 0) return;

    auto* out = static_cast<uint8_t*>(iter.operand(0).data);
    auto* a = static_cast<const T*>(iter.operand(1).data);
    auto* b = static_cast<const T*>(iter.operand(2).data);

    DeviceStrideMeta ma{}, mb{};
    iter.export_device_meta(1, &ma, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(2, &mb, VBT_CUDA_BROADCAST_MAX_NDIM);

    dim3 grid, block;
    launch_bounds_and_grid(N_iter, grid, block);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    if (use32) {
        comparison_ti_strided_kernel<T, Op, int32_t><<<grid, block, 0, stream>>>(out, a, ma, b, mb, static_cast<int32_t>(N_iter), op);
    } else {
        comparison_ti_strided_kernel<T, Op, int64_t><<<grid, block, 0, stream>>>(out, a, ma, b, mb, static_cast<int64_t>(N_iter), op);
    }
  }
};


template <typename T, typename Op, typename index_t>
__global__ void comparison_scalar_kernel(uint8_t* out, const T* a, T b, index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    out[i] = static_cast<uint8_t>(op(a[i], b));
  }
}

template <typename S, typename Op, typename index_t>
__global__ void comparison_scalar_kernel_fp16bf16(uint8_t* out, const S* a, float b_val, index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    float ax = detail::load_as_float(a + i);
    out[i] = static_cast<uint8_t>(op(ax, b_val));
  }
}

template <typename T, typename Op, typename index_t>
__global__ void comparison_dense_kernel(uint8_t* out, const T* a, const T* b, index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    out[i] = static_cast<uint8_t>(op(a[i], b[i]));
  }
}

template <typename S, typename Op, typename index_t>
__global__ void comparison_dense_kernel_fp16bf16(uint8_t* out, const S* a, const S* b, index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    float ax = detail::load_as_float(a + i);
    float bx = detail::load_as_float(b + i);
    out[i] = static_cast<uint8_t>(op(ax, bx));
  }
}

template <typename S, typename Op, typename index_t>
__global__ void comparison_ti_strided_kernel_fp16bf16(
    uint8_t* out,
    const S* base_a, DeviceStrideMeta ma,
    const S* base_b, DeviceStrideMeta mb,
    index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    std::int64_t off_a = compute_offset_elems(static_cast<std::int64_t>(i), ma);
    std::int64_t off_b = compute_offset_elems(static_cast<std::int64_t>(i), mb);
    float ax = detail::load_as_float(base_a + off_a);
    float bx = detail::load_as_float(base_b + off_b);
    out[i] = static_cast<uint8_t>(op(ax, bx));
  }
}

template <typename S, typename Op>
struct ComparisonLauncherMixed {
  Op op;

  void operator()(const TensorIter& iter, bool use32, void* stream_ptr) {
    const int64_t N_iter = iter.numel();
    if (N_iter == 0) return;

    auto* out = static_cast<uint8_t*>(iter.operand(0).data);
    auto* a = static_cast<const S*>(iter.operand(1).data);
    auto* b = static_cast<const S*>(iter.operand(2).data);

    DeviceStrideMeta ma{}, mb{};
    iter.export_device_meta(1, &ma, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(2, &mb, VBT_CUDA_BROADCAST_MAX_NDIM);

    dim3 grid, block;
    launch_bounds_and_grid(N_iter, grid, block);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    if (use32) {
        comparison_ti_strided_kernel_fp16bf16<S, Op, int32_t><<<grid, block, 0, stream>>>(out, a, ma, b, mb, static_cast<int32_t>(N_iter), op);
    } else {
        comparison_ti_strided_kernel_fp16bf16<S, Op, int64_t><<<grid, block, 0, stream>>>(out, a, ma, b, mb, static_cast<int64_t>(N_iter), op);
    }
  }
};

template <typename T, typename Op, typename index_t>
__global__ void ternary_ti_strided_kernel(
    T* out,
    const T* t1, DeviceStrideMeta t1_meta,
    const T* t2, DeviceStrideMeta t2_meta,
    const T* t3, DeviceStrideMeta t3_meta,
    index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    std::int64_t t1_offset = compute_offset_elems(static_cast<std::int64_t>(i), t1_meta);
    std::int64_t t2_offset = compute_offset_elems(static_cast<std::int64_t>(i), t2_meta);
    std::int64_t t3_offset = compute_offset_elems(static_cast<std::int64_t>(i), t3_meta);
    out[i] = op(t1[t1_offset], t2[t2_offset], t3[t3_offset]);
  }
}

template <typename S, typename Op, typename index_t>
__global__ void ternary_ti_strided_kernel_fp16bf16(
    S* out,
    const S* t1, DeviceStrideMeta t1_meta,
    const S* t2, DeviceStrideMeta t2_meta,
    const S* t3, DeviceStrideMeta t3_meta,
    index_t N, Op op) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    std::int64_t t1_offset = compute_offset_elems(static_cast<std::int64_t>(i), t1_meta);
    std::int64_t t2_offset = compute_offset_elems(static_cast<std::int64_t>(i), t2_meta);
    std::int64_t t3_offset = compute_offset_elems(static_cast<std::int64_t>(i), t3_meta);
    float a = detail::load_as_float(t1 + t1_offset);
    float b = detail::load_as_float(t2 + t2_offset);
    float c = detail::load_as_float(t3 + t3_offset);
    detail::store_from_float(out + i, op(a, b, c));
  }
}

template <typename S, typename Op>
struct TernaryLauncherMixed {
  Op op;

  void operator()(const TensorIter& iter, bool use32, void* stream_ptr) {
    const int64_t N_iter = iter.numel();
    if (N_iter == 0) return;

    auto* out = static_cast<S*>(iter.operand(0).data);
    auto* t1 = static_cast<const S*>(iter.operand(1).data);
    auto* t2 = static_cast<const S*>(iter.operand(2).data);
    auto* t3 = static_cast<const S*>(iter.operand(3).data);

    DeviceStrideMeta t1_meta{}, t2_meta{}, t3_meta{};
    iter.export_device_meta(1, &t1_meta, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(2, &t2_meta, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(3, &t3_meta, VBT_CUDA_BROADCAST_MAX_NDIM);

    dim3 grid, block;
    launch_bounds_and_grid(N_iter, grid, block);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    if (use32) {
      ternary_ti_strided_kernel_fp16bf16<S, Op, int32_t><<<grid, block, 0, stream>>>(
          out,
          t1, t1_meta,
          t2, t2_meta,
          t3, t3_meta,
          static_cast<int32_t>(N_iter),
          op);
    } else {
      ternary_ti_strided_kernel_fp16bf16<S, Op, int64_t><<<grid, block, 0, stream>>>(
          out,
          t1, t1_meta,
          t2, t2_meta,
          t3, t3_meta,
          static_cast<int64_t>(N_iter),
          op);
    }
  }
};

template <typename T, typename Op>
struct TernaryStridedLauncher {
  Op op;

  void operator()(const TensorIter& iter, bool use32, void* stream_ptr) {
    const int64_t N_iter = iter.numel();
    if (N_iter == 0) return;

    auto* out = static_cast<T*>(iter.operand(0).data);
    auto* t1 = static_cast<const T*>(iter.operand(1).data);
    auto* t2 = static_cast<const T*>(iter.operand(2).data);
    auto* t3 = static_cast<const T*>(iter.operand(3).data);

    DeviceStrideMeta t1_meta{}, t2_meta{}, t3_meta{};
    iter.export_device_meta(1, &t1_meta, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(2, &t2_meta, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(3, &t3_meta, VBT_CUDA_BROADCAST_MAX_NDIM);

    dim3 grid, block;
    launch_bounds_and_grid(N_iter, grid, block);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    if (use32) {
        ternary_ti_strided_kernel<T, Op, int32_t><<<grid, block, 0, stream>>>(
            out, t1, t1_meta, t2, t2_meta, t3, t3_meta, static_cast<int32_t>(N_iter), op);
    } else {
        ternary_ti_strided_kernel<T, Op, int64_t><<<grid, block, 0, stream>>>(
            out, t1, t1_meta, t2, t2_meta, t3, t3_meta, static_cast<int64_t>(N_iter), op);
    }
  }
};

template <typename T, typename index_t>
__global__ void where_ti_strided_kernel(
    T* out, DeviceStrideMeta mo,
    const uint8_t* cond, DeviceStrideMeta mc,
    const T* x, DeviceStrideMeta mx,
    const T* y, DeviceStrideMeta my,
    index_t N) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    std::int64_t off_o = compute_offset_elems(static_cast<std::int64_t>(i), mo);
    std::int64_t off_c = compute_offset_elems(static_cast<std::int64_t>(i), mc);
    std::int64_t off_x = compute_offset_elems(static_cast<std::int64_t>(i), mx);
    std::int64_t off_y = compute_offset_elems(static_cast<std::int64_t>(i), my);
    const bool take_x = (cond[off_c] != 0);
    out[off_o] = take_x ? x[off_x] : y[off_y];
  }
}

template <typename T>
struct WhereStridedLauncher {
  void operator()(const TensorIter& iter, bool use32, void* stream_ptr) {
    const int64_t N_iter = iter.numel();
    if (N_iter == 0) return;

    auto* out = static_cast<T*>(iter.operand(0).data);
    auto* cond = static_cast<const uint8_t*>(iter.operand(1).data);
    auto* x = static_cast<const T*>(iter.operand(2).data);
    auto* y = static_cast<const T*>(iter.operand(3).data);

    DeviceStrideMeta mo{}, mc{}, mx{}, my{};
    iter.export_device_meta(0, &mo, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(1, &mc, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(2, &mx, VBT_CUDA_BROADCAST_MAX_NDIM);
    iter.export_device_meta(3, &my, VBT_CUDA_BROADCAST_MAX_NDIM);

    dim3 grid, block;
    launch_bounds_and_grid(N_iter, grid, block);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    if (use32) {
      where_ti_strided_kernel<T, int32_t><<<grid, block, 0, stream>>>(
          out, mo, cond, mc, x, mx, y, my, static_cast<int32_t>(N_iter));
    } else {
      where_ti_strided_kernel<T, int64_t><<<grid, block, 0, stream>>>(
          out, mo, cond, mc, x, mx, y, my, static_cast<int64_t>(N_iter));
    }
  }
};

static TensorImpl where_op_dispatcher(
    const TensorImpl& cond,
    const TensorImpl& x,
    const TensorImpl& y,
    const char* name) {
#if VBT_WITH_CUDA
  if (cond.dtype() != ScalarType::Bool) {
    throw std::invalid_argument(std::string(name) + ": expected bool condition");
  }
  if (x.dtype() != y.dtype()) {
    throw std::invalid_argument(std::string(name) + ": dtype mismatch");
  }
  if (cond.device() != x.device() || cond.device() != y.device()) {
    throw std::invalid_argument(std::string(name) + ": device mismatch");
  }
  if (cond.device().type != kDLCUDA) {
    throw std::runtime_error(std::string(name) + ": expected CUDA tensors");
  }

  DeviceGuard g(x.device().index);

  std::vector<std::vector<int64_t>> shapes;
  shapes.push_back(cond.sizes());
  shapes.push_back(x.sizes());
  shapes.push_back(y.sizes());
  std::vector<int64_t> out_sizes;
  try {
    out_sizes = vbt::core::infer_broadcast_shape_nary(shapes);
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string(name) + ": " + e.what());
  }

  TensorImpl out = make_cuda_dense_out(out_sizes, x.dtype(), x.device());
  const int64_t N = numel_of(out);
  if (N == 0) return out;

  TensorIterConfig cfg;
  cfg.check_mem_overlap(true);
  cfg.check_all_same_dtype(false);
  cfg.add_output(OptionalTensorImplRef(&out, true));
  cfg.add_input(cond);
  cfg.add_input(x);
  cfg.add_input(y);

  TensorIter iter;
  try {
    iter = cfg.build();
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string(name) + ": " + e.what());
  }

  VBT_TI_STATS_INC(cuda_ti_kernel_launches);

  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(x.device().index));

  if (x.dtype() == ScalarType::Float32) {
    WhereStridedLauncher<float> launcher{};
    vbt::core::ti_gpu_kernel(iter, launcher);
  } else if (x.dtype() == ScalarType::Int64) {
    WhereStridedLauncher<std::int64_t> launcher{};
    vbt::core::ti_gpu_kernel(iter, launcher);
  } else if (x.dtype() == ScalarType::Bool) {
    WhereStridedLauncher<uint8_t> launcher{};
    vbt::core::ti_gpu_kernel(iter, launcher);
  } else if (x.dtype() == ScalarType::Float16) {
    WhereStridedLauncher<__half> launcher{};
    vbt::core::ti_gpu_kernel(iter, launcher);
  }
#if VBT_CUDA_HAS_BF16
  else if (x.dtype() == ScalarType::BFloat16) {
    WhereStridedLauncher<__nv_bfloat16> launcher{};
    vbt::core::ti_gpu_kernel(iter, launcher);
  }
#endif
  else {
    throw std::invalid_argument(std::string(name) + ": unsupported dtype");
  }

  vbt::cuda::record_stream(out.storage(), stream);
  vbt::cuda::record_stream(cond.storage(), stream);
  vbt::cuda::record_stream(x.storage(), stream);
  vbt::cuda::record_stream(y.storage(), stream);
  return out;
#else
  (void)cond; (void)x; (void)y; (void)name;
  throw std::runtime_error("CUDA not built");
#endif
}

template <template <typename> class Op>
static TensorImpl ternary_op_dispatcher(
    const TensorImpl& t1,
    const TensorImpl& t2,
    const TensorImpl& t3,
    const char* name) {
#if VBT_WITH_CUDA
  if (t1.dtype() != t2.dtype() || t1.dtype() != t3.dtype()) {
    throw std::invalid_argument(std::string(name) + ": dtype mismatch");
  }
  if (t1.device() != t2.device() || t1.device() != t3.device()) {
    throw std::invalid_argument(std::string(name) + ": device mismatch");
  }
  if (t1.device().type != kDLCUDA) {
    throw std::runtime_error(std::string(name) + ": expected CUDA tensor");
  }

  DeviceGuard g(t1.device().index);

  // Broadcast
  std::vector<std::vector<int64_t>> shapes;
  shapes.push_back(t1.sizes());
  shapes.push_back(t2.sizes());
  shapes.push_back(t3.sizes());

  std::vector<int64_t> out_sizes;
  try {
    out_sizes = vbt::core::infer_broadcast_shape_nary(shapes);
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string(name) + ": " + e.what());
  }

  if (out_sizes.size() > static_cast<std::size_t>(VBT_CUDA_BROADCAST_MAX_NDIM)) {
    throw std::invalid_argument(
        std::string(name) + ": CUDA broadcast supports up to " +
        std::to_string(VBT_CUDA_BROADCAST_MAX_NDIM) + "D (got " +
        std::to_string(out_sizes.size()) + "D)");
  }

  TensorImpl out = make_cuda_dense_out(out_sizes, t1.dtype(), t1.device());
  const int64_t N = numel_of(out);
  if (N == 0) return out;

  TensorIterConfig cfg;
  cfg.check_mem_overlap(true);
  cfg.add_output(OptionalTensorImplRef(&out, true));
  cfg.add_input(t1);
  cfg.add_input(t2);
  cfg.add_input(t3);

  TensorIter iter;
  try {
    iter = cfg.build();
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string(name) + ": " + e.what());
  }

  VBT_TI_STATS_INC(cuda_ti_kernel_launches);

  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(t1.device().index));

  if (t1.dtype() == ScalarType::Float32) {
    using T = float;
    using OpT = Op<T>;
    TernaryStridedLauncher<T, OpT> launcher{OpT{}};
    vbt::core::ti_gpu_kernel(iter, launcher);
  } else if (t1.dtype() == ScalarType::Float16) {
    using S = __half;
    using OpT = Op<float>;
    TernaryLauncherMixed<S, OpT> launcher{OpT{}};
    vbt::core::ti_gpu_kernel(iter, launcher);
  }
#if VBT_CUDA_HAS_BF16
  else if (t1.dtype() == ScalarType::BFloat16) {
    using S = __nv_bfloat16;
    using OpT = Op<float>;
    TernaryLauncherMixed<S, OpT> launcher{OpT{}};
    vbt::core::ti_gpu_kernel(iter, launcher);
  }
#endif
  else {
    throw std::invalid_argument(std::string(name) + ": unsupported dtype");
  }

  vbt::cuda::record_stream(out.storage(), stream);
  vbt::cuda::record_stream(t1.storage(), stream);
  vbt::cuda::record_stream(t2.storage(), stream);
  vbt::cuda::record_stream(t3.storage(), stream);
  return out;
#else
  (void)t1;
  (void)t2;
  (void)t3;
  (void)name;
  throw std::runtime_error("CUDA not built");
#endif
}

enum class OutputDtypeMode { Preserve, Bool };

template <template <typename> class Op, template <typename> class ROp, OutputDtypeMode OutMode>
static TensorImpl binary_op_dispatcher(const TensorImpl& a, const TensorImpl& b, const char* name) {
#if VBT_WITH_CUDA
  if (a.dtype() != b.dtype()) throw std::invalid_argument(std::string(name) + ": dtype mismatch");
  
  if (a.dtype() == ScalarType::Int64 && std::string(name) == "vt::div") {
    throw std::runtime_error(std::string(name) + ": Int64 not supported on CUDA (division by zero undefined)");
  }
  
  bool a_is_scalar = (a.sizes().size() == 0 && a.device().type == kDLCPU);
  bool b_is_scalar = (b.sizes().size() == 0 && b.device().type == kDLCPU);
  
  if (a.device() != b.device() && !a_is_scalar && !b_is_scalar) throw std::invalid_argument(std::string(name) + ": device mismatch");
  if (!a_is_scalar && a.device().type != kDLCUDA) throw std::runtime_error(std::string(name) + ": expected CUDA tensor");
  if (!b_is_scalar && b.device().type != kDLCUDA) throw std::runtime_error(std::string(name) + ": expected CUDA tensor");
  if (a_is_scalar && b_is_scalar) throw std::runtime_error(std::string(name) + ": both inputs CPU scalar (not implemented for CUDA backend)");

  Device out_dev = a_is_scalar ? b.device() : a.device();
  DeviceGuard g(out_dev.index);
  
  ScalarType out_dtype = (OutMode == OutputDtypeMode::Bool) ? ScalarType::Bool : a.dtype();
  
  if ((a_is_scalar || b_is_scalar) && (a_is_scalar ? b : a).is_contiguous()) {
     const TensorImpl& tensor = a_is_scalar ? b : a;
     const TensorImpl& scalar = a_is_scalar ? a : b;
     
     auto out = make_cuda_dense_out(tensor.sizes(), out_dtype, out_dev);
     const int64_t N = numel_of(tensor);
     if (N == 0) return out;
     
     bool use32 = vbt::cuda_detail::should_use_int32_index(N);
     dim3 grid, block; launch_bounds_and_grid(N, grid, block);
     auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(out_dev.index));
     
     if (tensor.dtype() == ScalarType::Float32) {
        float val = *static_cast<const float*>(scalar.data());
        auto run = [&](auto op_instance) {
           using OpT = decltype(op_instance);
           if constexpr (OutMode == OutputDtypeMode::Bool) {
               if (use32) comparison_scalar_kernel<float, OpT, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  static_cast<uint8_t*>(out.data()), static_cast<const float*>(tensor.data()), val, static_cast<int32_t>(N), op_instance);
               else comparison_scalar_kernel<float, OpT, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  static_cast<uint8_t*>(out.data()), static_cast<const float*>(tensor.data()), val, static_cast<int64_t>(N), op_instance);
           } else {
               if (use32) binary_scalar_kernel<float, OpT, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  static_cast<float*>(out.data()), static_cast<const float*>(tensor.data()), val, static_cast<int32_t>(N), op_instance);
               else binary_scalar_kernel<float, OpT, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  static_cast<float*>(out.data()), static_cast<const float*>(tensor.data()), val, static_cast<int64_t>(N), op_instance);
           }
        };
        if (a_is_scalar) run(ROp<float>()); else run(Op<float>());
     } else if (tensor.dtype() == ScalarType::Int64) {
        // Note: CUDA integer division by zero is undefined behavior (no check).
        long long val = *static_cast<const long long*>(scalar.data());
        auto run = [&](auto op_instance) {
           using OpT = decltype(op_instance);
           if constexpr (OutMode == OutputDtypeMode::Bool) {
               if (use32) comparison_scalar_kernel<long long, OpT, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  static_cast<uint8_t*>(out.data()), static_cast<const long long*>(tensor.data()), val, static_cast<int32_t>(N), op_instance);
               else comparison_scalar_kernel<long long, OpT, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  static_cast<uint8_t*>(out.data()), static_cast<const long long*>(tensor.data()), val, static_cast<int64_t>(N), op_instance);
           } else {
               if (use32) binary_scalar_kernel<long long, OpT, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  static_cast<long long*>(out.data()), static_cast<const long long*>(tensor.data()), static_cast<long long>(val), static_cast<int32_t>(N), op_instance);
               else binary_scalar_kernel<long long, OpT, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  static_cast<long long*>(out.data()), static_cast<const long long*>(tensor.data()), static_cast<long long>(val), static_cast<int64_t>(N), op_instance);
           }
        };
        if (a_is_scalar) run(ROp<long long>()); else run(Op<long long>());
     } else if (tensor.dtype() == ScalarType::Float16) {
        throw std::runtime_error(std::string(name) + ": Half CPU scalars not supported yet");
     } else {
        throw std::invalid_argument(std::string(name) + ": unsupported dtype for scalar op");
     }
     vbt::cuda::record_stream(out.storage(), stream);
     vbt::cuda::record_stream(tensor.storage(), stream);
     return out;
  }

  std::vector<int64_t> out_sizes;
  try {
    std::vector<std::vector<int64_t>> shapes;
    shapes.reserve(2);
    shapes.push_back(a.sizes());
    shapes.push_back(b.sizes());
    out_sizes = vbt::core::infer_broadcast_shape_nary(
        std::span<const std::vector<int64_t>>(shapes));
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string(name) + ": " + e.what());
  }
  if (out_sizes.size() > static_cast<std::size_t>(VBT_CUDA_BROADCAST_MAX_NDIM)) {
    throw std::invalid_argument(
        std::string(name) + ": CUDA broadcast supports up to " +
        std::to_string(VBT_CUDA_BROADCAST_MAX_NDIM) + "D (got " +
        std::to_string(out_sizes.size()) + "D)");
  }
  const int64_t N = [&](){ int64_t acc=1; for (auto s: out_sizes){ if (s==0){return int64_t(0);} if (acc <= std::numeric_limits<int64_t>::max()/s) acc*=s; else return int64_t(0);} return acc;}();
  auto out = make_cuda_dense_out(out_sizes, out_dtype, out_dev);
  if (N == 0) return out;
  bool use32 = vbt::cuda_detail::should_use_int32_index(N);
  dim3 grid, block; launch_bounds_and_grid(N, grid, block);
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(out_dev.index));

  if (a.is_contiguous() && b.is_contiguous() && a.sizes() == out_sizes && b.sizes() == out_sizes &&
      a.dtype() != ScalarType::Complex64 && a.dtype() != ScalarType::Complex128) {
    bool aligned = (reinterpret_cast<std::uintptr_t>(a.data()) % 16 == 0) &&
                   (reinterpret_cast<std::uintptr_t>(b.data()) % 16 == 0) &&
                   (reinterpret_cast<std::uintptr_t>(out.data()) % 16 == 0);
    bool vec_size = (N % 4 == 0);
    
    if constexpr (OutMode == OutputDtypeMode::Preserve) {
        if (a.dtype() == ScalarType::Float32 && aligned && vec_size) {
           if (use32) binary_dense_vectorized_kernel<float, Op<float>, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<float*>(out.data()), static_cast<const float*>(a.data()), static_cast<const float*>(b.data()), static_cast<int32_t>(N), Op<float>());
           else binary_dense_vectorized_kernel<float, Op<float>, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<float*>(out.data()), static_cast<const float*>(a.data()), static_cast<const float*>(b.data()), static_cast<int64_t>(N), Op<float>());
            
           vbt::cuda::record_stream(out.storage(), stream);
           vbt::cuda::record_stream(a.storage(), stream);
           vbt::cuda::record_stream(b.storage(), stream);
           return out;
        }
    }
    
    if (a.dtype() == ScalarType::Float32) {
      auto run = [&](auto op_instance) {
           using OpT = decltype(op_instance);
           if constexpr (OutMode == OutputDtypeMode::Bool) {
              if (use32) comparison_dense_kernel<float, OpT, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<uint8_t*>(out.data()), static_cast<const float*>(a.data()), static_cast<const float*>(b.data()), static_cast<int32_t>(N), op_instance);
              else comparison_dense_kernel<float, OpT, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<uint8_t*>(out.data()), static_cast<const float*>(a.data()), static_cast<const float*>(b.data()), static_cast<int64_t>(N), op_instance);
           } else {
              if (use32) binary_dense_kernel<float, OpT, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<float*>(out.data()), static_cast<const float*>(a.data()), static_cast<const float*>(b.data()), static_cast<int32_t>(N), op_instance);
              else binary_dense_kernel<float, OpT, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<float*>(out.data()), static_cast<const float*>(a.data()), static_cast<const float*>(b.data()), static_cast<int64_t>(N), op_instance);
           }
      };
      run(Op<float>());
    } else if (a.dtype() == ScalarType::Int64) {
      auto run = [&](auto op_instance) {
           using OpT = decltype(op_instance);
           if constexpr (OutMode == OutputDtypeMode::Bool) {
              if (use32) comparison_dense_kernel<long long, OpT, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<uint8_t*>(out.data()), static_cast<const long long*>(a.data()), static_cast<const long long*>(b.data()), static_cast<int32_t>(N), op_instance);
              else comparison_dense_kernel<long long, OpT, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<uint8_t*>(out.data()), static_cast<const long long*>(a.data()), static_cast<const long long*>(b.data()), static_cast<int64_t>(N), op_instance);
           } else {
              if (use32) binary_dense_kernel<long long, OpT, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<long long*>(out.data()), static_cast<const long long*>(a.data()), static_cast<const long long*>(b.data()), static_cast<int32_t>(N), op_instance);
              else binary_dense_kernel<long long, OpT, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<long long*>(out.data()), static_cast<const long long*>(a.data()), static_cast<const long long*>(b.data()), static_cast<int64_t>(N), op_instance);
           }
      };
      run(Op<long long>());
    } else if (a.dtype() == ScalarType::Float16) {
      auto run = [&](auto op_instance) {
           using OpT = decltype(op_instance);
           if constexpr (OutMode == OutputDtypeMode::Bool) {
              if (use32) comparison_dense_kernel_fp16bf16<__half, OpT, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<uint8_t*>(out.data()), static_cast<const __half*>(a.data()), static_cast<const __half*>(b.data()), static_cast<int32_t>(N), op_instance);
              else comparison_dense_kernel_fp16bf16<__half, OpT, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<uint8_t*>(out.data()), static_cast<const __half*>(a.data()), static_cast<const __half*>(b.data()), static_cast<int64_t>(N), op_instance);
           } else {
              if (use32) binary_dense_kernel_fp16bf16<__half, OpT, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<__half*>(out.data()), static_cast<const __half*>(a.data()), static_cast<const __half*>(b.data()), static_cast<int32_t>(N), op_instance);
              else binary_dense_kernel_fp16bf16<__half, OpT, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<__half*>(out.data()), static_cast<const __half*>(a.data()), static_cast<const __half*>(b.data()), static_cast<int64_t>(N), op_instance);
           }
      };
      run(Op<float>());
    }
#if VBT_CUDA_HAS_BF16
    else if (a.dtype() == ScalarType::BFloat16) {
      auto run = [&](auto op_instance) {
           using OpT = decltype(op_instance);
           if constexpr (OutMode == OutputDtypeMode::Bool) {
              if (use32) comparison_dense_kernel_fp16bf16<__nv_bfloat16, OpT, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<uint8_t*>(out.data()), static_cast<const __nv_bfloat16*>(a.data()), static_cast<const __nv_bfloat16*>(b.data()), static_cast<int32_t>(N), op_instance);
              else comparison_dense_kernel_fp16bf16<__nv_bfloat16, OpT, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<uint8_t*>(out.data()), static_cast<const __nv_bfloat16*>(a.data()), static_cast<const __nv_bfloat16*>(b.data()), static_cast<int64_t>(N), op_instance);
           } else {
              if (use32) binary_dense_kernel_fp16bf16<__nv_bfloat16, OpT, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<__nv_bfloat16*>(out.data()), static_cast<const __nv_bfloat16*>(a.data()), static_cast<const __nv_bfloat16*>(b.data()), static_cast<int32_t>(N), op_instance);
              else binary_dense_kernel_fp16bf16<__nv_bfloat16, OpT, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<__nv_bfloat16*>(out.data()), static_cast<const __nv_bfloat16*>(a.data()), static_cast<const __nv_bfloat16*>(b.data()), static_cast<int64_t>(N), op_instance);
           }
      };
      run(Op<float>());
    }
#endif
    else if (a.dtype() == ScalarType::Bool) {
      auto run = [&](auto op_instance) {
           using OpT = decltype(op_instance);
           if constexpr (OutMode == OutputDtypeMode::Bool) {
              if (use32) binary_dense_kernel<uint8_t, OpT, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<uint8_t*>(out.data()), static_cast<const uint8_t*>(a.data()), static_cast<const uint8_t*>(b.data()), static_cast<int32_t>(N), op_instance);
              else binary_dense_kernel<uint8_t, OpT, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<uint8_t*>(out.data()), static_cast<const uint8_t*>(a.data()), static_cast<const uint8_t*>(b.data()), static_cast<int64_t>(N), op_instance);
           } else {
              if (use32) binary_dense_kernel<uint8_t, OpT, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<uint8_t*>(out.data()), static_cast<const uint8_t*>(a.data()), static_cast<const uint8_t*>(b.data()), static_cast<int32_t>(N), op_instance);
              else binary_dense_kernel<uint8_t, OpT, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<uint8_t*>(out.data()), static_cast<const uint8_t*>(a.data()), static_cast<const uint8_t*>(b.data()), static_cast<int64_t>(N), op_instance);
           }
      };
      run(Op<uint8_t>());
    }
    else {
      throw std::invalid_argument(std::string(name) + ": unsupported dtype");
    }
  } else {
    TensorIterConfig cfg;
    cfg.check_mem_overlap(true);
    if constexpr (OutMode == OutputDtypeMode::Bool) {
         cfg.check_all_same_dtype(false);
    }
    cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true),
                   IterOperandRole::WriteOnly,
                   /*allow_resize=*/true);
    cfg.add_input(a);
    cfg.add_input(b);

    TensorIter iter;
    try {
      iter = cfg.build();
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument(std::string(name) + ": " + e.what());
    }

#ifdef VBT_TI_DEBUG
    {
      const std::int64_t N_iter = iter.numel();
      assert(N_iter == N);
      assert(iter.ndim() <= VBT_CUDA_BROADCAST_MAX_NDIM);
    }
#endif

    VBT_TI_STATS_INC(cuda_ti_kernel_launches);

    if (a.dtype() == ScalarType::Float32) {
      auto run = [&](auto op_instance) {
           using OpT = decltype(op_instance);
           if constexpr (OutMode == OutputDtypeMode::Bool) {
                ::vbt::core::ti_gpu_kernel(iter, ComparisonLauncher<float, OpT>{op_instance});
           } else {
                ::vbt::core::ti_gpu_kernel(iter, BinaryStridedLauncher<float, OpT>{op_instance});
           }
      };
      run(Op<float>());
    } else if (a.dtype() == ScalarType::Int64) {
      auto run = [&](auto op_instance) {
           using OpT = decltype(op_instance);
           if constexpr (OutMode == OutputDtypeMode::Bool) {
                ::vbt::core::ti_gpu_kernel(iter, ComparisonLauncher<long long, OpT>{op_instance});
           } else {
                ::vbt::core::ti_gpu_kernel(iter, BinaryStridedLauncher<long long, OpT>{op_instance});
           }
      };
      run(Op<long long>());
    } else if (a.dtype() == ScalarType::Float16) {
      auto run = [&](auto op_instance) {
           using OpT = decltype(op_instance);
           if constexpr (OutMode == OutputDtypeMode::Bool) {
                ::vbt::core::ti_gpu_kernel(iter, ComparisonLauncherMixed<__half, OpT>{op_instance});
           } else {
                ::vbt::core::ti_gpu_kernel(iter, BinaryStridedLauncherMixed<__half, OpT>{op_instance});
           }
      };
      run(Op<float>());
    }
#if VBT_CUDA_HAS_BF16
    else if (a.dtype() == ScalarType::BFloat16) {
      auto run = [&](auto op_instance) {
           using OpT = decltype(op_instance);
           if constexpr (OutMode == OutputDtypeMode::Bool) {
                ::vbt::core::ti_gpu_kernel(iter, ComparisonLauncherMixed<__nv_bfloat16, OpT>{op_instance});
           } else {
                ::vbt::core::ti_gpu_kernel(iter, BinaryStridedLauncherMixed<__nv_bfloat16, OpT>{op_instance});
           }
      };
      run(Op<float>());
    }
#endif
    else if (a.dtype() == ScalarType::Complex64) {
      using T = vbt::core::Complex64;
      if constexpr (OutMode == OutputDtypeMode::Preserve &&
                    (std::is_same_v<Op<T>, AddOp<T>> ||
                     std::is_same_v<Op<T>, MulOp<T>>)) {
        ::vbt::core::ti_gpu_kernel(iter, BinaryStridedLauncher<T, Op<T>>{Op<T>()});
      } else {
        throw std::invalid_argument(std::string(name) + ": unsupported dtype");
      }
    }
    else if (a.dtype() == ScalarType::Complex128) {
      using T = vbt::core::Complex128;
      if constexpr (OutMode == OutputDtypeMode::Preserve &&
                    (std::is_same_v<Op<T>, AddOp<T>> ||
                     std::is_same_v<Op<T>, MulOp<T>>)) {
        ::vbt::core::ti_gpu_kernel(iter, BinaryStridedLauncher<T, Op<T>>{Op<T>()});
      } else {
        throw std::invalid_argument(std::string(name) + ": unsupported dtype");
      }
    }
    else if (a.dtype() == ScalarType::Bool) {
      auto run = [&](auto op_instance) {
           using OpT = decltype(op_instance);
           if constexpr (OutMode == OutputDtypeMode::Bool) {
                ::vbt::core::ti_gpu_kernel(iter, BinaryStridedLauncher<uint8_t, OpT>{op_instance});
           } else {
                ::vbt::core::ti_gpu_kernel(iter, BinaryStridedLauncher<uint8_t, OpT>{op_instance});
           }
      };
      run(Op<uint8_t>());
    }
    else {
      throw std::invalid_argument(std::string(name) + ": unsupported dtype");
    }
  }
  vbt::cuda::record_stream(out.storage(), stream);
  if (!a_is_scalar) vbt::cuda::record_stream(a.storage(), stream);
  if (!b_is_scalar) vbt::cuda::record_stream(b.storage(), stream);
  return out;
#else
  (void)a; (void)b; (void)name; throw std::runtime_error("CUDA not built");
#endif
}


template <template <typename> class Op>
static TensorImpl binary_bitwise_dispatcher(const TensorImpl& a, const TensorImpl& b, const char* name) {
#if VBT_WITH_CUDA
  if (a.dtype() != b.dtype()) throw std::invalid_argument(std::string(name) + ": dtype mismatch");
  if (a.device() != b.device()) throw std::invalid_argument(std::string(name) + ": device mismatch");
  if (a.device().type != kDLCUDA) throw std::runtime_error(std::string(name) + ": expected CUDA tensor");

  if (!(a.dtype() == ScalarType::Bool || a.dtype() == ScalarType::Int32 || a.dtype() == ScalarType::Int64)) {
    throw std::invalid_argument(std::string(name) + ": unsupported dtype");
  }

  DeviceGuard g(a.device().index);

  std::vector<int64_t> out_sizes;
  try {
    std::vector<std::vector<int64_t>> shapes;
    shapes.reserve(2);
    shapes.push_back(a.sizes());
    shapes.push_back(b.sizes());
    out_sizes = vbt::core::infer_broadcast_shape_nary(
        std::span<const std::vector<int64_t>>(shapes));
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(std::string(name) + ": " + e.what());
  }

  if (out_sizes.size() > static_cast<std::size_t>(VBT_CUDA_BROADCAST_MAX_NDIM)) {
    throw std::invalid_argument(
        std::string(name) + ": CUDA broadcast supports up to " +
        std::to_string(VBT_CUDA_BROADCAST_MAX_NDIM) + "D (got " +
        std::to_string(out_sizes.size()) + "D)");
  }

  const int64_t N = [&](){ int64_t acc=1; for (auto s: out_sizes){ if (s==0){return int64_t(0);} if (acc <= std::numeric_limits<int64_t>::max()/s) acc*=s; else return int64_t(0);} return acc;}();
  auto out = make_cuda_dense_out(out_sizes, a.dtype(), a.device());
  if (N == 0) return out;

  bool use32 = vbt::cuda_detail::should_use_int32_index(N);
  dim3 grid, block; launch_bounds_and_grid(N, grid, block);
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(a.device().index));

  if (a.is_contiguous() && b.is_contiguous() && a.sizes() == out_sizes && b.sizes() == out_sizes) {
    if (a.dtype() == ScalarType::Int32) {
      using T = std::int32_t;
      if (use32) {
        binary_dense_kernel<T, Op<T>, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<T*>(out.data()),
            static_cast<const T*>(a.data()),
            static_cast<const T*>(b.data()),
            static_cast<int32_t>(N),
            Op<T>());
      } else {
        binary_dense_kernel<T, Op<T>, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<T*>(out.data()),
            static_cast<const T*>(a.data()),
            static_cast<const T*>(b.data()),
            static_cast<int64_t>(N),
            Op<T>());
      }
    } else if (a.dtype() == ScalarType::Int64) {
      using T = long long;
      if (use32) {
        binary_dense_kernel<T, Op<T>, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<T*>(out.data()),
            static_cast<const T*>(a.data()),
            static_cast<const T*>(b.data()),
            static_cast<int32_t>(N),
            Op<T>());
      } else {
        binary_dense_kernel<T, Op<T>, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<T*>(out.data()),
            static_cast<const T*>(a.data()),
            static_cast<const T*>(b.data()),
            static_cast<int64_t>(N),
            Op<T>());
      }
    } else if (a.dtype() == ScalarType::Bool) {
      using T = uint8_t;
      if (use32) {
        binary_dense_kernel<T, Op<T>, int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<T*>(out.data()),
            static_cast<const T*>(a.data()),
            static_cast<const T*>(b.data()),
            static_cast<int32_t>(N),
            Op<T>());
      } else {
        binary_dense_kernel<T, Op<T>, int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<T*>(out.data()),
            static_cast<const T*>(a.data()),
            static_cast<const T*>(b.data()),
            static_cast<int64_t>(N),
            Op<T>());
      }
    } else {
      throw std::invalid_argument(std::string(name) + ": unsupported dtype");
    }
  } else {
    TensorIterConfig cfg;
    cfg.check_mem_overlap(true);
    cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true),
                   IterOperandRole::WriteOnly,
                   /*allow_resize=*/true);
    cfg.add_input(a);
    cfg.add_input(b);

    TensorIter iter;
    try {
      iter = cfg.build();
    } catch (const std::invalid_argument& e) {
      throw std::invalid_argument(std::string(name) + ": " + e.what());
    }

    VBT_TI_STATS_INC(cuda_ti_kernel_launches);

    if (a.dtype() == ScalarType::Int32) {
      ::vbt::core::ti_gpu_kernel(iter, BinaryStridedLauncher<std::int32_t, Op<std::int32_t>>{Op<std::int32_t>()});
    } else if (a.dtype() == ScalarType::Int64) {
      ::vbt::core::ti_gpu_kernel(iter, BinaryStridedLauncher<long long, Op<long long>>{Op<long long>()});
    } else if (a.dtype() == ScalarType::Bool) {
      ::vbt::core::ti_gpu_kernel(iter, BinaryStridedLauncher<uint8_t, Op<uint8_t>>{Op<uint8_t>()});
    } else {
      throw std::invalid_argument(std::string(name) + ": unsupported dtype");
    }
  }

  vbt::cuda::record_stream(out.storage(), stream);
  vbt::cuda::record_stream(a.storage(), stream);
  vbt::cuda::record_stream(b.storage(), stream);
  return out;
#else
  (void)a; (void)b; (void)name; throw std::runtime_error("CUDA not built");
#endif
}


extern "C" {


TensorImpl vbt_cuda_add_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.dtype() == ScalarType::Bool || b.dtype() == ScalarType::Bool) throw std::invalid_argument("vt::add: boolean not supported");
  TensorImpl aa = vbt::core::resolve_conj(a);
  TensorImpl bb = vbt::core::resolve_conj(b);
  return binary_op_dispatcher<AddOp, AddOp, OutputDtypeMode::Preserve>(aa, bb, "vt::add");
}

TensorImpl vbt_cuda_mul_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.dtype() == ScalarType::Bool || b.dtype() == ScalarType::Bool) throw std::invalid_argument("vt::mul: boolean not supported");
  TensorImpl aa = vbt::core::resolve_conj(a);
  TensorImpl bb = vbt::core::resolve_conj(b);
  return binary_op_dispatcher<MulOp, MulOp, OutputDtypeMode::Preserve>(aa, bb, "vt::mul");
}

TensorImpl vbt_cuda_sub_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.dtype() == ScalarType::Bool || b.dtype() == ScalarType::Bool) {
    throw std::invalid_argument("vt::sub: boolean not supported");
  }
  return binary_op_dispatcher<SubOp, RSubOp, OutputDtypeMode::Preserve>(a, b, "vt::sub");
}

TensorImpl vbt_cuda_div_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.dtype() == ScalarType::Bool || b.dtype() == ScalarType::Bool) {
    throw std::invalid_argument("vt::div: boolean not supported");
  }
  return binary_op_dispatcher<DivOp, RDivOp, OutputDtypeMode::Preserve>(a, b, "vt::div");
}

TensorImpl vbt_cuda_rsub_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.dtype() == ScalarType::Bool || b.dtype() == ScalarType::Bool) {
    throw std::invalid_argument("vt::rsub: boolean not supported");
  }
  return binary_op_dispatcher<RSubOp, SubOp, OutputDtypeMode::Preserve>(a, b, "vt::rsub");
}

TensorImpl vbt_cuda_true_divide_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.dtype() == ScalarType::Bool || b.dtype() == ScalarType::Bool) {
    throw std::invalid_argument("vt::true_divide: boolean not supported");
  }
  if (a.dtype() == ScalarType::Int64) {
    throw std::runtime_error(
        "vt::true_divide: Int64 not supported on CUDA (division by zero undefined)");
  }
  return binary_op_dispatcher<DivOp, RDivOp, OutputDtypeMode::Preserve>(a, b, "vt::true_divide");
}

TensorImpl vbt_cuda_relu_impl(const TensorImpl& a) {
#if VBT_WITH_CUDA
  if (a.device().type != kDLCUDA) throw std::runtime_error("vt::relu: expected CUDA tensor");
  DeviceGuard g(a.device().index);
  const int64_t N = numel_of(a);
  auto out = make_cuda_dense_out(a.sizes(), a.dtype(), a.device());
  if (N == 0) return out;
  bool use32 = vbt::cuda_detail::should_use_int32_index(N);
  dim3 grid, block; launch_bounds_and_grid(N, grid, block);
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(a.device().index));
  if (a.is_contiguous()) {
    if (a.dtype() == ScalarType::Float32) {
      if (use32) relu_dense_kernel<float,int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<float*>(out.data()), static_cast<const float*>(a.data()), static_cast<int32_t>(N));
      else relu_dense_kernel<float,int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<float*>(out.data()), static_cast<const float*>(a.data()), static_cast<int64_t>(N));
    } else if (a.dtype() == ScalarType::Int64) {
      if (use32) relu_dense_kernel<long long,int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<long long*>(out.data()), static_cast<const long long*>(a.data()), static_cast<int32_t>(N));
      else relu_dense_kernel<long long,int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<long long*>(out.data()), static_cast<const long long*>(a.data()), static_cast<int64_t>(N));
    } else if (a.dtype() == ScalarType::Bool) {
      if (use32) relu_bool_dense_kernel<int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<uint8_t*>(out.data()), static_cast<const uint8_t*>(a.data()), static_cast<int32_t>(N));
      else relu_bool_dense_kernel<int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<uint8_t*>(out.data()), static_cast<const uint8_t*>(a.data()), static_cast<int64_t>(N));
    } else if (a.dtype() == ScalarType::Float16) {
      if (use32) relu_dense_kernel_fp16bf16<__half,int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<__half*>(out.data()), static_cast<const __half*>(a.data()), static_cast<int32_t>(N));
      else relu_dense_kernel_fp16bf16<__half,int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<__half*>(out.data()), static_cast<const __half*>(a.data()), static_cast<int64_t>(N));
    }
#if VBT_CUDA_HAS_BF16
    else if (a.dtype() == ScalarType::BFloat16) {
      if (use32) relu_dense_kernel_fp16bf16<__nv_bfloat16,int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<__nv_bfloat16*>(out.data()), static_cast<const __nv_bfloat16*>(a.data()), static_cast<int32_t>(N));
      else relu_dense_kernel_fp16bf16<__nv_bfloat16,int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<__nv_bfloat16*>(out.data()), static_cast<const __nv_bfloat16*>(a.data()), static_cast<int64_t>(N));
    }
#endif
    else {
      throw std::invalid_argument("vt::relu: unsupported dtype");
    }
  } else {
    DeviceStrideMeta ma = make_meta_ti_order(a);
    if (a.dtype() == ScalarType::Float32) {
      if (use32) relu_strided_to_contig_kernel<float,int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<float*>(out.data()), static_cast<const float*>(a.data()), ma, static_cast<int32_t>(N));
      else relu_strided_to_contig_kernel<float,int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<float*>(out.data()), static_cast<const float*>(a.data()), ma, static_cast<int64_t>(N));
    } else if (a.dtype() == ScalarType::Int64) {
      if (use32) relu_strided_to_contig_kernel<long long,int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<long long*>(out.data()), static_cast<const long long*>(a.data()), ma, static_cast<int32_t>(N));
      else relu_strided_to_contig_kernel<long long,int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<long long*>(out.data()), static_cast<const long long*>(a.data()), ma, static_cast<int64_t>(N));
    } else if (a.dtype() == ScalarType::Bool) {
      if (use32) relu_bool_strided_to_contig_kernel<int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<uint8_t*>(out.data()), static_cast<const uint8_t*>(a.data()), ma, static_cast<int32_t>(N));
      else relu_bool_strided_to_contig_kernel<int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<uint8_t*>(out.data()), static_cast<const uint8_t*>(a.data()), ma, static_cast<int64_t>(N));
    } else if (a.dtype() == ScalarType::Float16) {
      if (use32) relu_strided_to_contig_kernel_fp16bf16<__half,int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<__half*>(out.data()), static_cast<const __half*>(a.data()), ma, static_cast<int32_t>(N));
      else relu_strided_to_contig_kernel_fp16bf16<__half,int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<__half*>(out.data()), static_cast<const __half*>(a.data()), ma, static_cast<int64_t>(N));
    }
#if VBT_CUDA_HAS_BF16
    else if (a.dtype() == ScalarType::BFloat16) {
      if (use32) relu_strided_to_contig_kernel_fp16bf16<__nv_bfloat16,int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<__nv_bfloat16*>(out.data()), static_cast<const __nv_bfloat16*>(a.data()), ma, static_cast<int32_t>(N));
      else relu_strided_to_contig_kernel_fp16bf16<__nv_bfloat16,int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<__nv_bfloat16*>(out.data()), static_cast<const __nv_bfloat16*>(a.data()), ma, static_cast<int64_t>(N));
    }
#endif
    else {
      throw std::invalid_argument("vt::relu: unsupported dtype");
    }
  }
  vbt::cuda::record_stream(out.storage(), stream);
  vbt::cuda::record_stream(a.storage(), stream);
  return out;
#else
  (void)a; throw std::runtime_error("CUDA not built");
#endif
}

TensorImpl vbt_cuda_eq_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<EqOp, EqOp, OutputDtypeMode::Bool>(a, b, "vt::eq"); }
TensorImpl vbt_cuda_ne_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<NeOp, NeOp, OutputDtypeMode::Bool>(a, b, "vt::ne"); }
TensorImpl vbt_cuda_lt_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<LtOp, GtOp, OutputDtypeMode::Bool>(a, b, "vt::lt"); }
TensorImpl vbt_cuda_gt_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<GtOp, LtOp, OutputDtypeMode::Bool>(a, b, "vt::gt"); }
TensorImpl vbt_cuda_le_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<LeOp, GeOp, OutputDtypeMode::Bool>(a, b, "vt::le"); }
TensorImpl vbt_cuda_ge_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<GeOp, LeOp, OutputDtypeMode::Bool>(a, b, "vt::ge"); }
TensorImpl vbt_cuda_bitwise_and_impl(const TensorImpl& a, const TensorImpl& b) { 
  if (a.dtype() != ScalarType::Int64 && a.dtype() != ScalarType::Int32 && a.dtype() != ScalarType::Bool) throw std::invalid_argument("vt::bitwise_and: integral or bool tensors expected");
  return binary_bitwise_dispatcher<BitwiseAndOp>(a, b, "vt::bitwise_and"); 
}
TensorImpl vbt_cuda_bitwise_or_impl(const TensorImpl& a, const TensorImpl& b) { 
  if (a.dtype() != ScalarType::Int64 && a.dtype() != ScalarType::Int32 && a.dtype() != ScalarType::Bool) throw std::invalid_argument("vt::bitwise_or: integral or bool tensors expected");
  return binary_bitwise_dispatcher<BitwiseOrOp>(a, b, "vt::bitwise_or"); 
}
TensorImpl vbt_cuda_bitwise_xor_impl(const TensorImpl& a, const TensorImpl& b) { 
  if (a.dtype() != ScalarType::Int64 && a.dtype() != ScalarType::Int32 && a.dtype() != ScalarType::Bool) throw std::invalid_argument("vt::bitwise_xor: integral or bool tensors expected");
  return binary_bitwise_dispatcher<BitwiseXorOp>(a, b, "vt::bitwise_xor"); 
}
TensorImpl vbt_cuda_bitwise_not_impl(const TensorImpl& a) {
  if (a.dtype() == ScalarType::Bool) {
    return unary_bool_op_dispatcher<LogicalNotOp>(a, "vt::bitwise_not");
  }
  if (a.dtype() == ScalarType::Int64 || a.dtype() == ScalarType::Int32) {
    return unary_bitwise_dispatcher<BitwiseNotOp>(a, "vt::bitwise_not");
  }
  throw std::invalid_argument("vt::bitwise_not: integral or bool tensor expected");
}
TensorImpl vbt_cuda_logical_and_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<LogicalAndOp, LogicalAndOp, OutputDtypeMode::Bool>(a, b, "vt::logical_and"); }
TensorImpl vbt_cuda_logical_or_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<LogicalOrOp, LogicalOrOp, OutputDtypeMode::Bool>(a, b, "vt::logical_or"); }
TensorImpl vbt_cuda_logical_xor_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<LogicalXorOp, LogicalXorOp, OutputDtypeMode::Bool>(a, b, "vt::logical_xor"); }
TensorImpl vbt_cuda_lshift_impl(const TensorImpl& a, const TensorImpl& b) {
  if (!(a.dtype() == ScalarType::Int32 || a.dtype() == ScalarType::Int64) ||
      !(b.dtype() == ScalarType::Int32 || b.dtype() == ScalarType::Int64)) {
    throw std::invalid_argument("vt::lshift: expected Int32/Int64 tensors");
  }
  return binary_bitwise_dispatcher<LShiftOp>(a, b, "vt::lshift");
}
TensorImpl vbt_cuda_rshift_impl(const TensorImpl& a, const TensorImpl& b) {
  if (!(a.dtype() == ScalarType::Int32 || a.dtype() == ScalarType::Int64) ||
      !(b.dtype() == ScalarType::Int32 || b.dtype() == ScalarType::Int64)) {
    throw std::invalid_argument("vt::rshift: expected Int32/Int64 tensors");
  }
  return binary_bitwise_dispatcher<RShiftOp>(a, b, "vt::rshift");
}
TensorImpl vbt_cuda_bitwise_left_shift_impl(const TensorImpl& a, const TensorImpl& b) {
  if (!(a.dtype() == ScalarType::Int32 || a.dtype() == ScalarType::Int64) ||
      !(b.dtype() == ScalarType::Int32 || b.dtype() == ScalarType::Int64)) {
    throw std::invalid_argument("vt::bitwise_left_shift: expected Int32/Int64 tensors");
  }
  return binary_bitwise_dispatcher<LShiftOp>(a, b, "vt::bitwise_left_shift");
}
TensorImpl vbt_cuda_bitwise_right_shift_impl(const TensorImpl& a, const TensorImpl& b) {
  if (!(a.dtype() == ScalarType::Int32 || a.dtype() == ScalarType::Int64) ||
      !(b.dtype() == ScalarType::Int32 || b.dtype() == ScalarType::Int64)) {
    throw std::invalid_argument("vt::bitwise_right_shift: expected Int32/Int64 tensors");
  }
  return binary_bitwise_dispatcher<RShiftOp>(a, b, "vt::bitwise_right_shift");
}
TensorImpl vbt_cuda_fmod_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.dtype() == ScalarType::Bool || b.dtype() == ScalarType::Bool) {
    throw std::invalid_argument("vt::fmod: boolean not supported");
  }
  return binary_op_dispatcher<FmodOp, FmodOp, OutputDtypeMode::Preserve>(a, b, "vt::fmod");
}
TensorImpl vbt_cuda_remainder_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.dtype() == ScalarType::Bool || b.dtype() == ScalarType::Bool) {
    throw std::invalid_argument("vt::remainder: boolean not supported");
  }
  return binary_op_dispatcher<RemainderOp, RemainderOp, OutputDtypeMode::Preserve>(a, b, "vt::remainder");
}
TensorImpl vbt_cuda_atan2_impl(const TensorImpl& a, const TensorImpl& b) {
  reject_integral_inputs(a, "vt::atan2");
  reject_integral_inputs(b, "vt::atan2");
  return binary_op_dispatcher<Atan2Op, Atan2Op, OutputDtypeMode::Preserve>(a, b, "vt::atan2");
}
TensorImpl vbt_cuda_copysign_impl(const TensorImpl& a, const TensorImpl& b) {
  reject_integral_inputs(a, "vt::copysign");
  reject_integral_inputs(b, "vt::copysign");
  return binary_op_dispatcher<CopysignOp, CopysignOp, OutputDtypeMode::Preserve>(a, b, "vt::copysign");
}
TensorImpl vbt_cuda_hypot_impl(const TensorImpl& a, const TensorImpl& b) {
  reject_integral_inputs(a, "vt::hypot");
  reject_integral_inputs(b, "vt::hypot");
  return binary_op_dispatcher<HypotOp, HypotOp, OutputDtypeMode::Preserve>(a, b, "vt::hypot");
}
TensorImpl vbt_cuda_gcd_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.dtype() != ScalarType::Int64 || b.dtype() != ScalarType::Int64) {
    throw std::invalid_argument("vt::gcd: expected Int64 tensors");
  }
  return binary_op_dispatcher<GcdOp, GcdOp, OutputDtypeMode::Preserve>(a, b, "vt::gcd");
}
TensorImpl vbt_cuda_lcm_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.dtype() != ScalarType::Int64 || b.dtype() != ScalarType::Int64) {
    throw std::invalid_argument("vt::lcm: expected Int64 tensors");
  }
  return binary_op_dispatcher<LcmOp, LcmOp, OutputDtypeMode::Preserve>(a, b, "vt::lcm");
}
TensorImpl vbt_cuda_xlogy_impl(const TensorImpl& a, const TensorImpl& b) {
  reject_integral_inputs(a, "vt::xlogy");
  reject_integral_inputs(b, "vt::xlogy");
  return binary_op_dispatcher<XlogyOp, XlogyOp, OutputDtypeMode::Preserve>(a, b, "vt::xlogy");
}
TensorImpl vbt_cuda_xlog1py_impl(const TensorImpl& a, const TensorImpl& b) {
  reject_integral_inputs(a, "vt::xlog1py");
  reject_integral_inputs(b, "vt::xlog1py");
  return binary_op_dispatcher<Xlog1pyOp, Xlog1pyOp, OutputDtypeMode::Preserve>(a, b, "vt::xlog1py");
}

TensorImpl vbt_cuda_logaddexp_impl(const TensorImpl& a, const TensorImpl& b) {
  reject_integral_inputs(a, "vt::logaddexp");
  reject_integral_inputs(b, "vt::logaddexp");
  return binary_op_dispatcher<LogaddexpOp, LogaddexpOp, OutputDtypeMode::Preserve>(a, b, "vt::logaddexp");
}

TensorImpl vbt_cuda_logaddexp2_impl(const TensorImpl& a, const TensorImpl& b) {
  reject_integral_inputs(a, "vt::logaddexp2");
  reject_integral_inputs(b, "vt::logaddexp2");
  return binary_op_dispatcher<Logaddexp2Op, Logaddexp2Op, OutputDtypeMode::Preserve>(a, b, "vt::logaddexp2");
}

TensorImpl vbt_cuda_ldexp_impl(const TensorImpl& a, const TensorImpl& b) {
  reject_integral_inputs(a, "vt::ldexp");
  return binary_op_dispatcher<LdexpOp, LdexpROp, OutputDtypeMode::Preserve>(a, b, "vt::ldexp");
}

TensorImpl vbt_cuda_float_power_impl(const TensorImpl& a, const TensorImpl& b) {
  reject_integral_inputs(a, "vt::float_power");
  reject_integral_inputs(b, "vt::float_power");
  return binary_op_dispatcher<FloatPowerOp, FloatPowerROp, OutputDtypeMode::Preserve>(a, b, "vt::float_power");
}

TensorImpl vbt_cuda_special_xlog1py_impl(const TensorImpl& a, const TensorImpl& b) {
  reject_integral_inputs(a, "vt::special_xlog1py");
  reject_integral_inputs(b, "vt::special_xlog1py");
  return binary_op_dispatcher<Xlog1pyOp, Xlog1pyOp, OutputDtypeMode::Preserve>(a, b, "vt::special_xlog1py");
}
TensorImpl vbt_cuda_nextafter_impl(const TensorImpl& a, const TensorImpl& b) {
  reject_integral_inputs(a, "vt::nextafter");
  reject_integral_inputs(b, "vt::nextafter");
  return binary_op_dispatcher<NextAfterOp, NextAfterOp, OutputDtypeMode::Preserve>(a, b, "vt::nextafter");
}
TensorImpl vbt_cuda_heaviside_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<HeavisideOp, HeavisideOp, OutputDtypeMode::Preserve>(a, b, "vt::heaviside"); }
TensorImpl vbt_cuda_abs_impl(const TensorImpl& a) { return unary_op_dispatcher<AbsOp>(a, "vt::abs"); }
TensorImpl vbt_cuda_neg_impl(const TensorImpl& a) { return unary_op_dispatcher<NegOp>(a, "vt::neg"); }
TensorImpl vbt_cuda_positive_impl(const TensorImpl& a) { return unary_op_dispatcher<PositiveOp>(a, "vt::positive"); }
TensorImpl vbt_cuda_square_impl(const TensorImpl& a) { return unary_op_dispatcher<SquareOp>(a, "vt::square"); }

TensorImpl vbt_cuda_exp_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::exp");
  return unary_op_dispatcher<ExpOp>(a, "vt::exp");
}
TensorImpl vbt_cuda_log_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::log");
  return unary_op_dispatcher<LogOp>(a, "vt::log");
}
TensorImpl vbt_cuda_sqrt_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::sqrt");
  return unary_op_dispatcher<SqrtOp>(a, "vt::sqrt");
}
TensorImpl vbt_cuda_rsqrt_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::rsqrt");
  return unary_op_dispatcher<RsqrtOp>(a, "vt::rsqrt");
}
TensorImpl vbt_cuda_sin_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::sin");
  return unary_op_dispatcher<SinOp>(a, "vt::sin");
}
TensorImpl vbt_cuda_cos_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::cos");
  return unary_op_dispatcher<CosOp>(a, "vt::cos");
}
TensorImpl vbt_cuda_tanh_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::tanh");
  return unary_op_dispatcher<TanhOp>(a, "vt::tanh");
}
TensorImpl vbt_cuda_sigmoid_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::sigmoid");
  return unary_op_dispatcher<SigmoidOp>(a, "vt::sigmoid");
}
TensorImpl vbt_cuda_tanh_backward_impl(const TensorImpl& grad_output, const TensorImpl& output) {
  if (grad_output.dtype() == ScalarType::Int64 || grad_output.dtype() == ScalarType::Int32 || grad_output.dtype() == ScalarType::Bool) {
    throw std::invalid_argument("vt::tanh_backward: floating tensor expected");
  }
  return binary_op_dispatcher<TanhBackwardOp, TanhBackwardROp, OutputDtypeMode::Preserve>(
      grad_output, output, "vt::tanh_backward");
}
TensorImpl vbt_cuda_sigmoid_backward_impl(const TensorImpl& grad_output, const TensorImpl& output) {
  if (grad_output.dtype() == ScalarType::Int64 || grad_output.dtype() == ScalarType::Int32 || grad_output.dtype() == ScalarType::Bool) {
    throw std::invalid_argument("vt::sigmoid_backward: floating tensor expected");
  }
  return binary_op_dispatcher<SigmoidBackwardOp, SigmoidBackwardROp, OutputDtypeMode::Preserve>(
      grad_output, output, "vt::sigmoid_backward");
}
TensorImpl vbt_cuda_expm1_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::expm1");
  return unary_op_dispatcher<ExpM1Op>(a, "vt::expm1");
}
TensorImpl vbt_cuda_log1p_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::log1p");
  return unary_op_dispatcher<Log1pOp>(a, "vt::log1p");
}
TensorImpl vbt_cuda_floor_impl(const TensorImpl& a) { return unary_op_dispatcher<FloorOp>(a, "vt::floor"); }
TensorImpl vbt_cuda_ceil_impl(const TensorImpl& a) { return unary_op_dispatcher<CeilOp>(a, "vt::ceil"); }
TensorImpl vbt_cuda_trunc_impl(const TensorImpl& a) { return unary_op_dispatcher<TruncOp>(a, "vt::trunc"); }
TensorImpl vbt_cuda_round_impl(const TensorImpl& a) { return unary_op_dispatcher<RoundOp>(a, "vt::round"); }
TensorImpl vbt_cuda_frac_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::frac");
  return unary_op_dispatcher<FracOp>(a, "vt::frac");
}
TensorImpl vbt_cuda_reciprocal_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::reciprocal");
  return unary_op_dispatcher<ReciprocalOp>(a, "vt::reciprocal");
}
TensorImpl vbt_cuda_sign_impl(const TensorImpl& a) { return unary_op_dispatcher<SignOp>(a, "vt::sign"); }
TensorImpl vbt_cuda_sgn_impl(const TensorImpl& a) {
  if (a.dtype() == ScalarType::Bool) {
    throw std::invalid_argument("vt::sgn: non-bool tensor expected");
  }
  return unary_op_dispatcher<SgnOp>(a, "vt::sgn");
}
TensorImpl vbt_cuda_angle_impl(const TensorImpl& a) {
  if (a.dtype() == ScalarType::Int64 || a.dtype() == ScalarType::Int32 || a.dtype() == ScalarType::Bool) {
    throw std::invalid_argument("vt::angle: floating tensor expected");
  }
  return unary_op_dispatcher<AngleOp>(a, "vt::angle");
}
TensorImpl vbt_cuda_conj_physical_impl(const TensorImpl& a) {
  return unary_op_dispatcher<ConjPhysicalOp>(a, "vt::conj_physical");
}
TensorImpl vbt_cuda_isfinite_impl(const TensorImpl& a) { return unary_bool_op_dispatcher<IsfiniteOp>(a, "vt::isfinite"); }
TensorImpl vbt_cuda_isinf_impl(const TensorImpl& a) { return unary_bool_op_dispatcher<IsinfOp>(a, "vt::isinf"); }
TensorImpl vbt_cuda_isnan_impl(const TensorImpl& a) { return unary_bool_op_dispatcher<IsnanOp>(a, "vt::isnan"); }
TensorImpl vbt_cuda_isneginf_impl(const TensorImpl& a) { return unary_bool_op_dispatcher<IsneginfOp>(a, "vt::isneginf"); }
TensorImpl vbt_cuda_isposinf_impl(const TensorImpl& a) { return unary_bool_op_dispatcher<IsposinfOp>(a, "vt::isposinf"); }
TensorImpl vbt_cuda_logical_not_impl(const TensorImpl& a) { return unary_bool_op_dispatcher<LogicalNotOp>(a, "vt::logical_not"); }
TensorImpl vbt_cuda_signbit_impl(const TensorImpl& a) { return unary_bool_op_dispatcher<SignbitOp>(a, "vt::signbit"); }
TensorImpl vbt_cuda_nan_to_num_impl(const TensorImpl& a) { return nan_to_num_op_dispatcher(a, "vt::nan_to_num"); }
TensorImpl vbt_cuda_relu6_impl(const TensorImpl& a) { return unary_op_dispatcher<Relu6Op>(a, "vt::relu6"); }
TensorImpl vbt_cuda_hardtanh_impl(const TensorImpl& a) { return unary_op_dispatcher<HardtanhOp>(a, "vt::hardtanh"); }
TensorImpl vbt_cuda_hardsigmoid_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::hardsigmoid");
  return unary_op_dispatcher<HardsigmoidOp>(a, "vt::hardsigmoid");
}
TensorImpl vbt_cuda_silu_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::silu");
  return unary_op_dispatcher<SiluOp>(a, "vt::silu");
}
TensorImpl vbt_cuda_silu_backward_impl(const TensorImpl& grad_output, const TensorImpl& self) {
  if (grad_output.dtype() == ScalarType::Int64 || grad_output.dtype() == ScalarType::Int32 || grad_output.dtype() == ScalarType::Bool) {
    throw std::invalid_argument("vt::silu_backward: floating tensor expected");
  }
  return binary_op_dispatcher<SiluBackwardOp, SiluBackwardROp, OutputDtypeMode::Preserve>(
      grad_output, self, "vt::silu_backward");
}
TensorImpl vbt_cuda_gelu_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::gelu");
  return unary_op_dispatcher<GeluOp>(a, "vt::gelu");
}
TensorImpl vbt_cuda_mish_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::mish");
  return unary_op_dispatcher<MishOp>(a, "vt::mish");
}
TensorImpl vbt_cuda_selu_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::selu");
  return unary_op_dispatcher<SeluOp>(a, "vt::selu");
}
TensorImpl vbt_cuda_softplus_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::softplus");
  return unary_op_dispatcher<SoftplusOp>(a, "vt::softplus");
}
TensorImpl vbt_cuda_hardshrink_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::hardshrink");
  return unary_op_dispatcher<HardshrinkOp>(a, "vt::hardshrink");
}
TensorImpl vbt_cuda_softshrink_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::softshrink");
  return unary_op_dispatcher<SoftshrinkOp>(a, "vt::softshrink");
}
TensorImpl vbt_cuda_celu_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::celu");
  return unary_op_dispatcher<CeluOp>(a, "vt::celu");
}
TensorImpl vbt_cuda_elu_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::elu");
  return unary_op_dispatcher<EluOp>(a, "vt::elu");
}
TensorImpl vbt_cuda_exp2_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::exp2");
  return unary_op_dispatcher<Exp2Op>(a, "vt::exp2");
}
TensorImpl vbt_cuda_log2_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::log2");
  return unary_op_dispatcher<Log2Op>(a, "vt::log2");
}
TensorImpl vbt_cuda_log10_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::log10");
  return unary_op_dispatcher<Log10Op>(a, "vt::log10");
}
TensorImpl vbt_cuda_sinh_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::sinh");
  return unary_op_dispatcher<SinHOp>(a, "vt::sinh");
}
TensorImpl vbt_cuda_cosh_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::cosh");
  return unary_op_dispatcher<CosHOp>(a, "vt::cosh");
}
TensorImpl vbt_cuda_asinh_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::asinh");
  return unary_op_dispatcher<AsinhOp>(a, "vt::asinh");
}
TensorImpl vbt_cuda_acosh_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::acosh");
  return unary_op_dispatcher<AcoshOp>(a, "vt::acosh");
}
TensorImpl vbt_cuda_atanh_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::atanh");
  return unary_op_dispatcher<AtanhOp>(a, "vt::atanh");
}
TensorImpl vbt_cuda_deg2rad_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::deg2rad");
  return unary_op_dispatcher<Deg2RadOp>(a, "vt::deg2rad");
}
TensorImpl vbt_cuda_rad2deg_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::rad2deg");
  return unary_op_dispatcher<Rad2DegOp>(a, "vt::rad2deg");
}
TensorImpl vbt_cuda_tan_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::tan");
  return unary_op_dispatcher<TanOp>(a, "vt::tan");
}
TensorImpl vbt_cuda_asin_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::asin");
  return unary_op_dispatcher<AsinOp>(a, "vt::asin");
}
TensorImpl vbt_cuda_acos_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::acos");
  return unary_op_dispatcher<AcosOp>(a, "vt::acos");
}
TensorImpl vbt_cuda_atan_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::atan");
  return unary_op_dispatcher<AtanOp>(a, "vt::atan");
}
TensorImpl vbt_cuda_erf_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::erf");
  return unary_op_dispatcher<ErfOp>(a, "vt::erf");
}
TensorImpl vbt_cuda_erfc_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::erfc");
  return unary_op_dispatcher<ErfcOp>(a, "vt::erfc");
}
TensorImpl vbt_cuda_lgamma_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::lgamma");
  return unary_op_dispatcher<LgammaOp>(a, "vt::lgamma");
}
TensorImpl vbt_cuda_sinc_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::sinc");
  return unary_op_dispatcher<SincOp>(a, "vt::sinc");
}

TensorImpl vbt_cuda_logit_impl(const TensorImpl& a) {
  reject_integral_inputs(a, "vt::logit");
  return unary_op_dispatcher<LogitOp>(a, "vt::logit");
}
TensorImpl vbt_cuda_logit_backward_impl(const TensorImpl& grad_output, const TensorImpl& self) {
  if (grad_output.dtype() == ScalarType::Int64 || grad_output.dtype() == ScalarType::Int32 || grad_output.dtype() == ScalarType::Bool) {
    throw std::invalid_argument("vt::logit_backward: floating tensor expected");
  }
  return binary_op_dispatcher<LogitBackwardOp, LogitBackwardROp, OutputDtypeMode::Preserve>(
      grad_output, self, "vt::logit_backward");
}
TensorImpl vbt_cuda_polygamma_impl(const TensorImpl& n, const TensorImpl& a) {
#if VBT_WITH_CUDA
    if (n.numel() != 1) {
        throw std::invalid_argument("vt::polygamma: expected n to be a scalar tensor");
    }
    if (n.device().type != kDLCUDA || a.device().type != kDLCUDA) {
        throw std::invalid_argument("vt::polygamma: expected CUDA tensors");
    }
    if (n.device().index != a.device().index) {
        throw std::invalid_argument("vt::polygamma: device mismatch");
    }

    if (n.dtype() != ScalarType::Float32 && n.dtype() != ScalarType::Float16
#if VBT_CUDA_HAS_BF16
        && n.dtype() != ScalarType::BFloat16
#endif
    ) {
        throw std::invalid_argument("vt::polygamma: expected n to have floating dtype");
    }

    // For now, only n==2 is implemented; other orders return NaN (device-side).

    return binary_op_dispatcher<PolygammaOp, PolygammaOp, OutputDtypeMode::Preserve>(n, a, "vt::polygamma");
#else
    (void)n; (void)a;
    throw std::runtime_error("CUDA not built");
#endif
}
TensorImpl vbt_cuda_fmax_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<FmaxOp, FmaxOp, OutputDtypeMode::Preserve>(a, b, "vt::fmax"); }
TensorImpl vbt_cuda_fmin_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<FminOp, FminOp, OutputDtypeMode::Preserve>(a, b, "vt::fmin"); }
TensorImpl vbt_cuda_maximum_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<MaximumOp, MaximumOp, OutputDtypeMode::Preserve>(a, b, "vt::maximum"); }
TensorImpl vbt_cuda_minimum_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<MinimumOp, MinimumOp, OutputDtypeMode::Preserve>(a, b, "vt::minimum"); }

TensorImpl vbt_cuda_huber_loss_impl(const TensorImpl& input, const TensorImpl& target, const TensorImpl& delta) {
   return ternary_op_dispatcher<HuberLossTernaryOp>(input, target, delta, "vt::huber_loss");
}

TensorImpl vbt_cuda_mse_loss_impl(const TensorImpl& input, const TensorImpl& target) {
    return binary_op_dispatcher<MseLossOp, MseLossOp, OutputDtypeMode::Preserve>(input, target, "vt::mse_loss");
}

TensorImpl vbt_cuda_smooth_l1_loss_impl(const TensorImpl& input, const TensorImpl& target, const TensorImpl& beta) {
    return ternary_op_dispatcher<SmoothL1TernaryOp>(input, target, beta, "vt::smooth_l1_loss");
}

TensorImpl vbt_cuda_pow_impl(const TensorImpl& a, const TensorImpl& b) { return binary_op_dispatcher<PowOp, PowOp, OutputDtypeMode::Preserve>(a, b, "vt::pow"); }

TensorImpl vbt_cuda_clamp_impl(const TensorImpl& t, const TensorImpl& min, const TensorImpl& max) {
  return ternary_op_dispatcher<ClampOp>(t, min, max, "vt::clamp");
}
TensorImpl vbt_cuda_clip_impl(const TensorImpl& t, const TensorImpl& min, const TensorImpl& max) {
  return ternary_op_dispatcher<ClampOp>(t, min, max, "vt::clip");
}
TensorImpl vbt_cuda_threshold_impl(const TensorImpl& t, const TensorImpl& thr, const TensorImpl& value) {
  reject_integral_inputs(t, "vt::threshold");
  reject_integral_inputs(thr, "vt::threshold");
  reject_integral_inputs(value, "vt::threshold");
  return ternary_op_dispatcher<ThresholdOp>(t, thr, value, "vt::threshold");
}
TensorImpl vbt_cuda_clamp_min_impl(const TensorImpl& t, const TensorImpl& min) {
  return binary_op_dispatcher<MaximumOp, MaximumOp, OutputDtypeMode::Preserve>(t, min, "vt::clamp_min");
}
TensorImpl vbt_cuda_clamp_max_impl(const TensorImpl& t, const TensorImpl& max) {
  return binary_op_dispatcher<MinimumOp, MinimumOp, OutputDtypeMode::Preserve>(t, max, "vt::clamp_max");
}
TensorImpl vbt_cuda_lerp_impl(const TensorImpl& start, const TensorImpl& end, const TensorImpl& weight) {
  return ternary_op_dispatcher<LerpOp>(start, end, weight, "vt::lerp");
}
TensorImpl vbt_cuda_where_impl(const TensorImpl& cond, const TensorImpl& x, const TensorImpl& y) {
  return where_op_dispatcher(cond, x, y, "vt::where");
}
TensorImpl vbt_cuda_masked_fill_impl(const TensorImpl& self, const TensorImpl& mask, const TensorImpl& value) {
  return where_op_dispatcher(mask, value, self, "vt::masked_fill");
}
TensorImpl vbt_cuda_addcmul_impl(const TensorImpl& input, const TensorImpl& t1, const TensorImpl& t2, const TensorImpl& value) {
  TensorImpl prod = vbt_cuda_mul_impl(t1, t2);
  TensorImpl scaled = vbt_cuda_mul_impl(prod, value);
  return vbt_cuda_add_impl(input, scaled);
}

TensorImpl vbt_cuda_addcdiv_impl(const TensorImpl& input, const TensorImpl& t1, const TensorImpl& t2, const TensorImpl& value) {
   TensorImpl quot = vbt_cuda_div_impl(t1, t2);
   TensorImpl scaled = vbt_cuda_mul_impl(quot, value);
   return vbt_cuda_add_impl(input, scaled);
}

} // extern "C"

extern "C" void vbt_register_cuda_elementwise_kernels() {
#if VBT_WITH_CUDA
  static std::once_flag once;
  std::call_once(once, []() {
    using vbt::dispatch::Dispatcher;

    auto& D = Dispatcher::instance();

    if (D.has("vt::add")) {
      D.registerCudaKernel("vt::add", &vbt_cuda_add_impl);
    }
    if (D.has("vt::mul")) {
      D.registerCudaKernel("vt::mul", &vbt_cuda_mul_impl);
    }
    if (D.has("vt::sub")) {
      D.registerCudaKernel("vt::sub", &vbt_cuda_sub_impl);
    }
    if (D.has("vt::rsub")) {
      D.registerCudaKernel("vt::rsub", &vbt_cuda_rsub_impl);
    }
    if (D.has("vt::div")) {
      D.registerCudaKernel("vt::div", &vbt_cuda_div_impl);
    }
    if (D.has("vt::true_divide")) {
      D.registerCudaKernel("vt::true_divide", &vbt_cuda_true_divide_impl);
    }
    if (D.has("vt::relu")) {
      D.registerCudaKernel("vt::relu", &vbt_cuda_relu_impl);
    }
    if (D.has("vt::relu6")) {
      D.registerCudaKernel("vt::relu6", &vbt_cuda_relu6_impl);
    }
    if (D.has("vt::eq")) {
      D.registerCudaKernel("vt::eq", &vbt_cuda_eq_impl);
    }
    if (D.has("vt::ne")) {
      D.registerCudaKernel("vt::ne", &vbt_cuda_ne_impl);
    }
    if (D.has("vt::lt")) {
      D.registerCudaKernel("vt::lt", &vbt_cuda_lt_impl);
    }
    if (D.has("vt::gt")) {
      D.registerCudaKernel("vt::gt", &vbt_cuda_gt_impl);
    }
    if (D.has("vt::le")) {
      D.registerCudaKernel("vt::le", &vbt_cuda_le_impl);
    }
    if (D.has("vt::ge")) {
      D.registerCudaKernel("vt::ge", &vbt_cuda_ge_impl);
    }
    if (D.has("vt::bitwise_and")) {
      D.registerCudaKernel("vt::bitwise_and", &vbt_cuda_bitwise_and_impl);
    }
    if (D.has("vt::bitwise_or")) {
      D.registerCudaKernel("vt::bitwise_or", &vbt_cuda_bitwise_or_impl);
    }
    if (D.has("vt::bitwise_xor")) {
      D.registerCudaKernel("vt::bitwise_xor", &vbt_cuda_bitwise_xor_impl);
    }
    if (D.has("vt::bitwise_not")) {
      D.registerCudaKernel("vt::bitwise_not", &vbt_cuda_bitwise_not_impl);
    }
    if (D.has("vt::logical_and")) {
      D.registerCudaKernel("vt::logical_and", &vbt_cuda_logical_and_impl);
    }
    if (D.has("vt::logical_or")) {
      D.registerCudaKernel("vt::logical_or", &vbt_cuda_logical_or_impl);
    }
    if (D.has("vt::logical_xor")) {
      D.registerCudaKernel("vt::logical_xor", &vbt_cuda_logical_xor_impl);
    }
    if (D.has("vt::lshift")) {
      D.registerCudaKernel("vt::lshift", &vbt_cuda_lshift_impl);
    }
    if (D.has("vt::rshift")) {
      D.registerCudaKernel("vt::rshift", &vbt_cuda_rshift_impl);
    }
    if (D.has("vt::bitwise_left_shift")) {
      D.registerCudaKernel("vt::bitwise_left_shift", &vbt_cuda_bitwise_left_shift_impl);
    }
    if (D.has("vt::bitwise_right_shift")) {
      D.registerCudaKernel("vt::bitwise_right_shift", &vbt_cuda_bitwise_right_shift_impl);
    }
    if (D.has("vt::fmod")) {
      D.registerCudaKernel("vt::fmod", &vbt_cuda_fmod_impl);
    }
    if (D.has("vt::remainder")) {
      D.registerCudaKernel("vt::remainder", &vbt_cuda_remainder_impl);
    }
    if (D.has("vt::atan2")) {
      D.registerCudaKernel("vt::atan2", &vbt_cuda_atan2_impl);
    }
    if (D.has("vt::copysign")) {
      D.registerCudaKernel("vt::copysign", &vbt_cuda_copysign_impl);
    }
    if (D.has("vt::hypot")) {
      D.registerCudaKernel("vt::hypot", &vbt_cuda_hypot_impl);
    }
    if (D.has("vt::gcd")) {
      D.registerCudaKernel("vt::gcd", &vbt_cuda_gcd_impl);
    }
    if (D.has("vt::lcm")) {
      D.registerCudaKernel("vt::lcm", &vbt_cuda_lcm_impl);
    }
    if (D.has("vt::xlogy")) {
      D.registerCudaKernel("vt::xlogy", &vbt_cuda_xlogy_impl);
    }
    if (D.has("vt::xlog1py")) {
      D.registerCudaKernel("vt::xlog1py", &vbt_cuda_xlog1py_impl);
    }
    if (D.has("vt::logaddexp")) {
      D.registerCudaKernel("vt::logaddexp", &vbt_cuda_logaddexp_impl);
    }
    if (D.has("vt::logaddexp2")) {
      D.registerCudaKernel("vt::logaddexp2", &vbt_cuda_logaddexp2_impl);
    }
    if (D.has("vt::ldexp")) {
      D.registerCudaKernel("vt::ldexp", &vbt_cuda_ldexp_impl);
    }
    if (D.has("vt::float_power")) {
      D.registerCudaKernel("vt::float_power", &vbt_cuda_float_power_impl);
    }
    if (D.has("vt::special_xlog1py")) {
      D.registerCudaKernel("vt::special_xlog1py", &vbt_cuda_special_xlog1py_impl);
    }
    if (D.has("vt::nextafter")) {
      D.registerCudaKernel("vt::nextafter", &vbt_cuda_nextafter_impl);
    }
    if (D.has("vt::heaviside")) {
      D.registerCudaKernel("vt::heaviside", &vbt_cuda_heaviside_impl);
    }
    if (D.has("vt::abs")) {
      D.registerCudaKernel("vt::abs", &vbt_cuda_abs_impl);
    }
    if (D.has("vt::neg")) {
      D.registerCudaKernel("vt::neg", &vbt_cuda_neg_impl);
    }
    if (D.has("vt::positive")) {
      D.registerCudaKernel("vt::positive", &vbt_cuda_positive_impl);
    }
    if (D.has("vt::square")) {
      D.registerCudaKernel("vt::square", &vbt_cuda_square_impl);
    }
    if (D.has("vt::exp")) {
      D.registerCudaKernel("vt::exp", &vbt_cuda_exp_impl);
    }
    if (D.has("vt::log")) {
      D.registerCudaKernel("vt::log", &vbt_cuda_log_impl);
    }
    if (D.has("vt::sqrt")) {
      D.registerCudaKernel("vt::sqrt", &vbt_cuda_sqrt_impl);
    }
    if (D.has("vt::rsqrt")) {
      D.registerCudaKernel("vt::rsqrt", &vbt_cuda_rsqrt_impl);
    }
    if (D.has("vt::sin")) {
      D.registerCudaKernel("vt::sin", &vbt_cuda_sin_impl);
    }
    if (D.has("vt::cos")) {
      D.registerCudaKernel("vt::cos", &vbt_cuda_cos_impl);
    }
    if (D.has("vt::tanh")) {
      D.registerCudaKernel("vt::tanh", &vbt_cuda_tanh_impl);
    }
    if (D.has("vt::tanh_backward")) {
      D.registerCudaKernel("vt::tanh_backward", &vbt_cuda_tanh_backward_impl);
    }
    if (D.has("vt::sigmoid")) {
      D.registerCudaKernel("vt::sigmoid", &vbt_cuda_sigmoid_impl);
    }
    if (D.has("vt::sigmoid_backward")) {
      D.registerCudaKernel("vt::sigmoid_backward", &vbt_cuda_sigmoid_backward_impl);
    }
    if (D.has("vt::hardtanh")) {
      D.registerCudaKernel("vt::hardtanh", &vbt_cuda_hardtanh_impl);
    }
    if (D.has("vt::hardsigmoid")) {
      D.registerCudaKernel("vt::hardsigmoid", &vbt_cuda_hardsigmoid_impl);
    }
    if (D.has("vt::silu")) {
      D.registerCudaKernel("vt::silu", &vbt_cuda_silu_impl);
    }
    if (D.has("vt::silu_backward")) {
      D.registerCudaKernel("vt::silu_backward", &vbt_cuda_silu_backward_impl);
    }
    if (D.has("vt::gelu")) {
      D.registerCudaKernel("vt::gelu", &vbt_cuda_gelu_impl);
    }
    if (D.has("vt::mish")) {
      D.registerCudaKernel("vt::mish", &vbt_cuda_mish_impl);
    }
    if (D.has("vt::selu")) {
      D.registerCudaKernel("vt::selu", &vbt_cuda_selu_impl);
    }
    if (D.has("vt::softplus")) {
      D.registerCudaKernel("vt::softplus", &vbt_cuda_softplus_impl);
    }
    if (D.has("vt::hardshrink")) {
      D.registerCudaKernel("vt::hardshrink", &vbt_cuda_hardshrink_impl);
    }
    if (D.has("vt::softshrink")) {
      D.registerCudaKernel("vt::softshrink", &vbt_cuda_softshrink_impl);
    }
    if (D.has("vt::celu")) {
      D.registerCudaKernel("vt::celu", &vbt_cuda_celu_impl);
    }
    if (D.has("vt::elu")) {
      D.registerCudaKernel("vt::elu", &vbt_cuda_elu_impl);
    }
    if (D.has("vt::expm1")) {
      D.registerCudaKernel("vt::expm1", &vbt_cuda_expm1_impl);
    }
    if (D.has("vt::log1p")) {
      D.registerCudaKernel("vt::log1p", &vbt_cuda_log1p_impl);
    }
    if (D.has("vt::floor")) {
      D.registerCudaKernel("vt::floor", &vbt_cuda_floor_impl);
    }
    if (D.has("vt::ceil")) {
      D.registerCudaKernel("vt::ceil", &vbt_cuda_ceil_impl);
    }
    if (D.has("vt::trunc")) {
      D.registerCudaKernel("vt::trunc", &vbt_cuda_trunc_impl);
    }
    if (D.has("vt::round")) {
      D.registerCudaKernel("vt::round", &vbt_cuda_round_impl);
    }
    if (D.has("vt::frac")) {
      D.registerCudaKernel("vt::frac", &vbt_cuda_frac_impl);
    }
    if (D.has("vt::reciprocal")) {
      D.registerCudaKernel("vt::reciprocal", &vbt_cuda_reciprocal_impl);
    }
    if (D.has("vt::sign")) {
      D.registerCudaKernel("vt::sign", &vbt_cuda_sign_impl);
    }
    if (D.has("vt::sgn")) {
      D.registerCudaKernel("vt::sgn", &vbt_cuda_sgn_impl);
    }
    if (D.has("vt::isfinite")) {
      D.registerCudaKernel("vt::isfinite", &vbt_cuda_isfinite_impl);
    }
    if (D.has("vt::isinf")) {
      D.registerCudaKernel("vt::isinf", &vbt_cuda_isinf_impl);
    }
    if (D.has("vt::isnan")) {
      D.registerCudaKernel("vt::isnan", &vbt_cuda_isnan_impl);
    }
    if (D.has("vt::isneginf")) {
      D.registerCudaKernel("vt::isneginf", &vbt_cuda_isneginf_impl);
    }
    if (D.has("vt::isposinf")) {
      D.registerCudaKernel("vt::isposinf", &vbt_cuda_isposinf_impl);
    }
    if (D.has("vt::logical_not")) {
      D.registerCudaKernel("vt::logical_not", &vbt_cuda_logical_not_impl);
    }
    if (D.has("vt::signbit")) {
      D.registerCudaKernel("vt::signbit", &vbt_cuda_signbit_impl);
    }
    if (D.has("vt::nan_to_num")) {
      D.registerCudaKernel("vt::nan_to_num", &vbt_cuda_nan_to_num_impl);
    }
    if (D.has("vt::exp2")) {
      D.registerCudaKernel("vt::exp2", &vbt_cuda_exp2_impl);
    }
    if (D.has("vt::log2")) {
      D.registerCudaKernel("vt::log2", &vbt_cuda_log2_impl);
    }
    if (D.has("vt::log10")) {
      D.registerCudaKernel("vt::log10", &vbt_cuda_log10_impl);
    }
    if (D.has("vt::sinh")) {
      D.registerCudaKernel("vt::sinh", &vbt_cuda_sinh_impl);
    }
    if (D.has("vt::cosh")) {
      D.registerCudaKernel("vt::cosh", &vbt_cuda_cosh_impl);
    }
    if (D.has("vt::asinh")) {
      D.registerCudaKernel("vt::asinh", &vbt_cuda_asinh_impl);
    }
    if (D.has("vt::acosh")) {
      D.registerCudaKernel("vt::acosh", &vbt_cuda_acosh_impl);
    }
    if (D.has("vt::atanh")) {
      D.registerCudaKernel("vt::atanh", &vbt_cuda_atanh_impl);
    }
    if (D.has("vt::deg2rad")) {
      D.registerCudaKernel("vt::deg2rad", &vbt_cuda_deg2rad_impl);
    }
    if (D.has("vt::rad2deg")) {
      D.registerCudaKernel("vt::rad2deg", &vbt_cuda_rad2deg_impl);
    }
    if (D.has("vt::tan")) {
      D.registerCudaKernel("vt::tan", &vbt_cuda_tan_impl);
    }
    if (D.has("vt::asin")) {
      D.registerCudaKernel("vt::asin", &vbt_cuda_asin_impl);
    }
    if (D.has("vt::acos")) {
      D.registerCudaKernel("vt::acos", &vbt_cuda_acos_impl);
    }
    if (D.has("vt::atan")) {
      D.registerCudaKernel("vt::atan", &vbt_cuda_atan_impl);
    }
    if (D.has("vt::angle")) {
      D.registerCudaKernel("vt::angle", &vbt_cuda_angle_impl);
    }
    if (D.has("vt::conj_physical")) {
      D.registerCudaKernel("vt::conj_physical", &vbt_cuda_conj_physical_impl);
    }
    if (D.has("vt::erf")) D.registerCudaKernel("vt::erf", &vbt_cuda_erf_impl);
    if (D.has("vt::erfc")) D.registerCudaKernel("vt::erfc", &vbt_cuda_erfc_impl);
    if (D.has("vt::lgamma")) D.registerCudaKernel("vt::lgamma", &vbt_cuda_lgamma_impl);
    if (D.has("vt::sinc")) D.registerCudaKernel("vt::sinc", &vbt_cuda_sinc_impl);
    if (D.has("vt::pow")) D.registerCudaKernel("vt::pow", &vbt_cuda_pow_impl);
    if (D.has("vt::clamp")) D.registerCudaKernel("vt::clamp", &vbt_cuda_clamp_impl);
    if (D.has("vt::clamp_min")) D.registerCudaKernel("vt::clamp_min", &vbt_cuda_clamp_min_impl);
    if (D.has("vt::clamp_max")) D.registerCudaKernel("vt::clamp_max", &vbt_cuda_clamp_max_impl);
    if (D.has("vt::clip")) D.registerCudaKernel("vt::clip", &vbt_cuda_clip_impl);
    if (D.has("vt::threshold")) D.registerCudaKernel("vt::threshold", &vbt_cuda_threshold_impl);
    if (D.has("vt::lerp")) D.registerCudaKernel("vt::lerp", &vbt_cuda_lerp_impl);
    if (D.has("vt::where")) D.registerCudaKernel("vt::where", &vbt_cuda_where_impl);
    if (D.has("vt::masked_fill")) D.registerCudaKernel("vt::masked_fill", &vbt_cuda_masked_fill_impl);
    if (D.has("vt::addcmul")) D.registerCudaKernel("vt::addcmul", &vbt_cuda_addcmul_impl);
    if (D.has("vt::addcdiv")) D.registerCudaKernel("vt::addcdiv", &vbt_cuda_addcdiv_impl);
    if (D.has("vt::logit")) D.registerCudaKernel("vt::logit", &vbt_cuda_logit_impl);
    if (D.has("vt::logit_backward")) D.registerCudaKernel("vt::logit_backward", &vbt_cuda_logit_backward_impl);
    if (D.has("vt::polygamma")) D.registerCudaKernel("vt::polygamma", &vbt_cuda_polygamma_impl);
    if (D.has("vt::fmax")) D.registerCudaKernel("vt::fmax", &vbt_cuda_fmax_impl);
    if (D.has("vt::fmin")) D.registerCudaKernel("vt::fmin", &vbt_cuda_fmin_impl);
    if (D.has("vt::maximum")) D.registerCudaKernel("vt::maximum", &vbt_cuda_maximum_impl);
    if (D.has("vt::minimum")) D.registerCudaKernel("vt::minimum", &vbt_cuda_minimum_impl);
    if (D.has("vt::huber_loss")) D.registerCudaKernel("vt::huber_loss", &vbt_cuda_huber_loss_impl);
    if (D.has("vt::mse_loss")) D.registerCudaKernel("vt::mse_loss", &vbt_cuda_mse_loss_impl);
    if (D.has("vt::smooth_l1_loss")) D.registerCudaKernel("vt::smooth_l1_loss", &vbt_cuda_smooth_l1_loss_impl);
  });
#else
  // No-op when CUDA is not built.
#endif
}
