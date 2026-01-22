// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <limits>
#include <mutex>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <span>
#include <string>

#include "vbt/dispatch/registration.h"
#include "vbt/core/tensor.h"
#include "vbt/core/tensor_ops.h"
#include "vbt/core/storage.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/broadcast.h"
#include "vbt/core/tensor_iter.h"
#include "vbt/cuda/allocator.h"
#include "vbt/cuda/cub.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/device_caps.h"
#include "vbt/cuda/reduction_env.h"
#include "vbt/cuda/reduction_plan.h"
#include "vbt/cuda/reduction_workspace.h"
#include "vbt/cuda/graphs.h"
#include "vbt/core/tensor_iterator/cuda_loops.h"
#include "vbt/core/tensor_iterator/cuda.h"

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined"
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
using vbt::core::kTensorIterMaxRank;

namespace {

#if VBT_WITH_CUDA
using vbt::cuda::DeviceGuard;
#endif

#if VBT_INTERNAL_TESTS
static std::atomic<std::uint64_t> g_cub_reduce_all_sum_fastpath_calls{0};
#endif

// Opt-in internal env flag for the CUB reduce-all contiguous sum fast path.
static bool compute_opt_in_flag_from_env_raw(const char* raw) noexcept {
  if (raw == nullptr) {
    return false;
  }

  const unsigned char* p = reinterpret_cast<const unsigned char*>(raw);
  while (*p != '\0' && std::isspace(*p)) {
    ++p;
  }
  if (*p == '\0') {
    return false;  // all whitespace
  }

  const unsigned char* e = p;
  while (*e != '\0') {
    ++e;
  }
  while (e > p && std::isspace(*(e - 1))) {
    --e;
  }
  if (e == p) {
    return false;
  }

  auto equals_ci = [](const unsigned char* b,
                      const unsigned char* end,
                      const char* lit) noexcept {
    const unsigned char* s = b;
    const char* t = lit;
    while (s < end && *t != '\0') {
      if (static_cast<char>(std::tolower(*s)) != *t) {
        return false;
      }
      ++s;
      ++t;
    }
    return (s == end) && (*t == '\0');
  };

  if (equals_ci(p, e, "0") ||
      equals_ci(p, e, "false") ||
      equals_ci(p, e, "no") ||
      equals_ci(p, e, "off")) {
    return false;
  }

  return true;
}

static bool cuda_cub_reduce_all_sum_enabled() noexcept {
#if VBT_INTERNAL_TESTS
  static std::once_flag once;
  static bool enabled = false;
  std::call_once(once, []() {
    const char* raw = std::getenv("VBT_INTERNAL_CUDA_CUB_REDUCE_ALL_SUM");
    enabled = compute_opt_in_flag_from_env_raw(raw);
  });
  return enabled;
#else
  return false;
#endif
}

// --- Atomics ---

template <typename T>
__device__ void atomic_sum(T* addr, T val) {
    atomicAdd(addr, val);
}

// Specialization for int64 (unsigned long long atomicAdd is standard, signed might need cast)
template <> __device__ void atomic_sum<long long>(long long* addr, long long val) {
    atomicAdd((unsigned long long*)addr, (unsigned long long)val);
}

template <typename T>
__device__ void atomic_prod(T* addr, T val) {
    // CAS loop
    if constexpr (std::is_same_v<T, float>) {
        int* addr_as_int = (int*)addr;
        int old = *addr_as_int, assumed;
        do {
            assumed = old;
            float new_val = __int_as_float(assumed) * val;
            old = atomicCAS(addr_as_int, assumed, __float_as_int(new_val));
        } while (assumed != old);
    } else if constexpr (std::is_same_v<T, long long>) {
        unsigned long long* addr_as_ull = (unsigned long long*)addr;
        unsigned long long old = *addr_as_ull, assumed;
        do {
            assumed = old;
            long long new_val = (long long)assumed * val;
            old = atomicCAS(addr_as_ull, assumed, (unsigned long long)new_val);
        } while (assumed != old);
    }
}

template <typename T>
__device__ void atomic_min_op(T* addr, T val) {
    if constexpr (std::is_floating_point_v<T>) {
        int* addr_as_int = (int*)addr;
        int old = *addr_as_int, assumed;
        do {
            assumed = old;
            float old_val = __int_as_float(assumed);
            float new_val = val < old_val ? val : old_val;
            old = atomicCAS(addr_as_int, assumed, __float_as_int(new_val));
        } while (assumed != old);
    } else {
        atomicMin(addr, val);
    }
}
template <> __device__ void atomic_min_op<long long>(long long* addr, long long val) {
    // Standard atomicMin is for unsigned long long, so negative numbers are treated as large.
    // Use CAS loop for signed long long min.
    unsigned long long* address_as_ull = (unsigned long long*)addr;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        long long old_val = (long long)assumed;
        long long new_val = val < old_val ? val : old_val;
        old = atomicCAS(address_as_ull, assumed, (unsigned long long)new_val);
    } while (assumed != old);
}

template <typename T>
__device__ void atomic_max_op(T* addr, T val) {
    if constexpr (std::is_floating_point_v<T>) {
        int* addr_as_int = (int*)addr;
        int old = *addr_as_int, assumed;
        do {
            assumed = old;
            float old_val = __int_as_float(assumed);
            float new_val = val > old_val ? val : old_val;
            old = atomicCAS(addr_as_int, assumed, __float_as_int(new_val));
        } while (assumed != old);
    } else {
        atomicMax(addr, val);
    }
}
template <> __device__ void atomic_max_op<long long>(long long* addr, long long val) {
    // Standard atomicMax is for unsigned long long. Use CAS loop.
    unsigned long long* address_as_ull = (unsigned long long*)addr;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        long long old_val = (long long)assumed;
        long long new_val = val > old_val ? val : old_val;
        old = atomicCAS(address_as_ull, assumed, (unsigned long long)new_val);
    } while (assumed != old);
}

// --- Ops ---
template <typename T> struct SumOp {
    __device__ void operator()(T* addr, T val) const { atomic_sum(addr, val); }
    static constexpr T init() { return T(0); }
};

template <typename T> struct ProdOp {
    __device__ void operator()(T* addr, T val) const { atomic_prod(addr, val); }
    static constexpr T init() { return T(1); }
};

template <typename T> struct MinOp {
    __device__ void operator()(T* addr, T val) const { atomic_min_op(addr, val); }
    static constexpr T init() {
        if constexpr (std::is_floating_point_v<T>) {
            return std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::max();
        }
    }
};

template <typename T> struct MaxOp {
    __device__ void operator()(T* addr, T val) const { atomic_max_op(addr, val); }
    static constexpr T init() {
        if constexpr (std::is_floating_point_v<T>) {
            return -std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::lowest();
        }
    }
};

// --- Kernels ---

template <typename T, typename Op, typename index_t>
__global__ void atomic_reduce_kernel(
    T* out, DeviceStrideMeta out_meta,
    const T* in, DeviceStrideMeta in_meta,
    index_t N,
    Op op) {

    const std::int64_t N64 = static_cast<std::int64_t>(N);
    const std::int64_t start =
        static_cast<std::int64_t>(blockIdx.x) * static_cast<std::int64_t>(blockDim.x) +
        static_cast<std::int64_t>(threadIdx.x);
    const std::int64_t stride =
        static_cast<std::int64_t>(blockDim.x) * static_cast<std::int64_t>(gridDim.x);

    // Iterate over INPUT elements (grid-stride).
    for (std::int64_t i = start; i < N64; i += stride) {
        std::int64_t in_off = compute_offset_elems(i, in_meta);
        std::int64_t out_off = compute_offset_elems(i, out_meta);

        T val = in[in_off];
        op(&out[out_off], val);
    }
}

// --- Staged K2 kernels (C=1; 1 CTA per output) ---

using vbt::cuda::reduction::CudaReducePlanDevice;

__device__ __forceinline__ void k2_decode_out_idx(const CudaReducePlanDevice& plan,
                                                  std::int64_t out_idx,
                                                  std::int64_t* out_off,
                                                  std::int64_t* in_base) {
  std::int64_t idx_tmp = out_idx;
  std::int64_t out_acc = 0;
  std::int64_t in_acc = 0;

  const std::int32_t kept_ndim = plan.kept_ndim;
  for (std::int32_t j = 0; j < kept_ndim; ++j) {
    // Invariant: for eligible, non-empty plans, kept_sizes[j] comes from tensor
    // sizes (plan builder) and must be > 0. Empty out/slice cases return before
    // K2 is considered.
    const std::int64_t size = plan.kept_sizes[j];
#if VBT_INTERNAL_TESTS
    assert(size > 0);
#endif

    const std::int64_t coord = (size == 1) ? 0 : (idx_tmp % size);
    idx_tmp = (size == 1) ? idx_tmp : (idx_tmp / size);

    out_acc += coord * plan.kept_out_strides[j];
    in_acc += coord * plan.kept_in_strides[j];
  }

#if VBT_INTERNAL_TESTS
  assert(idx_tmp == 0);
#endif

  *out_off = out_acc;
  *in_base = in_acc;
}

template <typename T> struct K2SumOp {
  __device__ __forceinline__ static T init() { return static_cast<T>(0); }
  __device__ __forceinline__ static T combine(T a, T b) { return a + b; }
};

template <typename T> struct K2ProdOp {
  __device__ __forceinline__ static T init() { return static_cast<T>(1); }
  __device__ __forceinline__ static T combine(T a, T b) { return a * b; }
};

template <typename T, template <typename> class OpT>
__global__ void k2_reduce_kernel(CudaReducePlanDevice plan,
                                 const T* in,
                                 T* out) {
  const std::int64_t out_numel = plan.out_numel;
  const std::int64_t slice_len = plan.slice_len;
  const std::int64_t red_stride = plan.red_linear_stride;
  const std::int64_t grid_stride = static_cast<std::int64_t>(gridDim.x);

#if VBT_INTERNAL_TESTS
  assert(blockDim.x <= 256u);
  assert((blockDim.x & (blockDim.x - 1u)) == 0u);
#endif

  __shared__ T shm[256];

  for (std::int64_t out_idx = static_cast<std::int64_t>(blockIdx.x);
       out_idx < out_numel;
       out_idx += grid_stride) {
    std::int64_t out_off = 0;
    std::int64_t in_base = 0;
    k2_decode_out_idx(plan, out_idx, &out_off, &in_base);

    T acc = OpT<T>::init();
    for (std::int64_t red_i = static_cast<std::int64_t>(threadIdx.x);
         red_i < slice_len;
         red_i += static_cast<std::int64_t>(blockDim.x)) {
      const std::int64_t in_off = in_base + red_i * red_stride;
      acc = OpT<T>::combine(acc, in[in_off]);
    }

    const unsigned int tid = threadIdx.x;
    shm[tid] = acc;
    __syncthreads();

    // Requires power-of-two blockDim.x.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        shm[tid] = OpT<T>::combine(shm[tid], shm[tid + s]);
      }
      __syncthreads();
    }

    if (tid == 0u) {
      out[out_off] = shm[0];
    }
    __syncthreads();
  }
}

// --- Staged K3 kernels (tile-outputs; multiple outputs per CTA) ---

template <typename T, template <typename> class OpT>
__global__ void k3_tile_outputs_reduce_kernel(CudaReducePlanDevice plan,
                                             const T* in,
                                             T* out) {
  const std::int64_t out_numel = plan.out_numel;
  const std::int64_t slice_len = plan.slice_len;
  const std::int64_t red_stride = plan.red_linear_stride;

  const std::int64_t tile_outputs = static_cast<std::int64_t>(blockDim.x);
  const std::int64_t grid_stride =
      static_cast<std::int64_t>(gridDim.x) * tile_outputs;

#if VBT_INTERNAL_TESTS
  assert(tile_outputs > 0);
  assert(tile_outputs <= 256);
#endif

  for (std::int64_t tile_base =
           static_cast<std::int64_t>(blockIdx.x) * tile_outputs;
       tile_base < out_numel;
       tile_base += grid_stride) {
    const std::int64_t out_idx =
        tile_base + static_cast<std::int64_t>(threadIdx.x);
    if (out_idx >= out_numel) {
      continue;
    }

    std::int64_t out_off = 0;
    std::int64_t in_base = 0;
    k2_decode_out_idx(plan, out_idx, &out_off, &in_base);

    T acc = OpT<T>::init();
    for (std::int64_t red_i = 0; red_i < slice_len; ++red_i) {
      const std::int64_t in_off = in_base + red_i * red_stride;
      acc = OpT<T>::combine(acc, in[in_off]);
    }

    out[out_off] = acc;
  }
}

#if VBT_INTERNAL_TESTS
// --- Staged K2-multi kernels (C>=2; multiple CTAs per output) ---
using vbt::cuda::reduction::CudaK2MultiFaultMode;

static_assert(sizeof(unsigned int) == sizeof(std::uint32_t),
              "K2-multi semaphores must be 32-bit");

template <typename T, template <typename> class OpT>
__global__ void k2multi_reduce_kernel(CudaReducePlanDevice plan,
                                      const T* in,
                                      T* out,
                                      T* partials,
                                      unsigned int* semaphores,
                                      unsigned int ctas_per_output,
                                      CudaK2MultiFaultMode fault_mode) {
  const std::int64_t out_numel = plan.out_numel;
  const std::int64_t slice_len = plan.slice_len;
  const std::int64_t red_stride = plan.red_linear_stride;

  const std::int64_t C = static_cast<std::int64_t>(ctas_per_output);
  const std::int64_t cta_id = static_cast<std::int64_t>(blockIdx.y);
  const std::int64_t tid_i64 = static_cast<std::int64_t>(threadIdx.x);

#if VBT_INTERNAL_TESTS
  assert(blockDim.x <= 256u);
  assert((blockDim.x & (blockDim.x - 1u)) == 0u);
  assert(ctas_per_output >= 2u);
#endif

  __shared__ T shm[256];

  for (std::int64_t out_idx = static_cast<std::int64_t>(blockIdx.x);
       out_idx < out_numel;
       out_idx += static_cast<std::int64_t>(gridDim.x)) {
    std::int64_t out_off = 0;
    std::int64_t in_base = 0;
    k2_decode_out_idx(plan, out_idx, &out_off, &in_base);

    T acc = OpT<T>::init();
    const std::int64_t step = static_cast<std::int64_t>(blockDim.x) * C;
    for (std::int64_t red_i =
             cta_id * static_cast<std::int64_t>(blockDim.x) + tid_i64;
         red_i < slice_len;
         red_i += step) {
      const std::int64_t in_off = in_base + red_i * red_stride;
      acc = OpT<T>::combine(acc, in[in_off]);
    }

    const unsigned int tid = threadIdx.x;
    shm[tid] = acc;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        shm[tid] = OpT<T>::combine(shm[tid], shm[tid + s]);
      }
      __syncthreads();
    }

    if (tid == 0u) {
      const std::int64_t pidx = out_idx * C + cta_id;

      const bool inject =
          (fault_mode == CudaK2MultiFaultMode::SignalButSkipPartialWrite) &&
          (out_idx == 0) && (cta_id == 0);

      if (!inject) {
        partials[pidx] = shm[0];
      }

      __threadfence();
      const unsigned int prev = atomicAdd(&semaphores[out_idx], 1u);
      if (prev == (ctas_per_output - 1u)) {
        T total = OpT<T>::init();
        const std::int64_t base = out_idx * C;
        for (unsigned int j = 0; j < ctas_per_output; ++j) {
          total = OpT<T>::combine(total, partials[base + static_cast<std::int64_t>(j)]);
        }
        out[out_off] = total;
      }
    }
  }
}
#endif  // VBT_INTERNAL_TESTS

template <typename T, typename index_t>
__global__ void scalar_div_kernel(T* data, T divisor, index_t n) {
    const std::int64_t n64 = static_cast<std::int64_t>(n);
    const std::int64_t start =
        static_cast<std::int64_t>(blockIdx.x) * static_cast<std::int64_t>(blockDim.x) +
        static_cast<std::int64_t>(threadIdx.x);
    const std::int64_t stride =
        static_cast<std::int64_t>(blockDim.x) * static_cast<std::int64_t>(gridDim.x);

    for (std::int64_t i = start; i < n64; i += stride) {
        data[i] /= divisor;
    }
}

template <typename T, typename index_t>
__global__ void fill_kernel(T* data, T val, index_t n) {
    const std::int64_t n64 = static_cast<std::int64_t>(n);
    const std::int64_t start =
        static_cast<std::int64_t>(blockIdx.x) * static_cast<std::int64_t>(blockDim.x) +
        static_cast<std::int64_t>(threadIdx.x);
    const std::int64_t stride =
        static_cast<std::int64_t>(blockDim.x) * static_cast<std::int64_t>(gridDim.x);

    for (std::int64_t i = start; i < n64; i += stride) {
        data[i] = val;
    }
}

// --- Launchers ---

struct Launch1DConfig {
  dim3 grid;
  dim3 block;
};

static inline unsigned int floor_pow2_u32(unsigned int x) noexcept {
  if (x == 0u) {
    return 1u;
  }
  unsigned int p = 1u;
  while ((p << 1u) <= x) {
    p <<= 1u;
  }
  return p;
}

static inline Launch1DConfig make_1d_launch(std::int64_t N,
                                            int device_index,
                                            std::int64_t out_numel_for_test_cap) {
#if !VBT_WITH_CUDA
  (void)N;
  (void)device_index;
  (void)out_numel_for_test_cap;
  return Launch1DConfig{dim3(1u), dim3(1u)};
#else
  const auto caps = vbt::cuda::get_device_caps(
      static_cast<vbt::cuda::DeviceIndex>(device_index));

  unsigned int threads = 256u;
  if (caps.max_threads_per_block > 0) {
    threads = std::min<unsigned int>(threads, caps.max_threads_per_block);
  }
  if (threads == 0u) {
    threads = 1u;
  }

  dim3 block(threads);

  if (N <= 0) {
    return Launch1DConfig{dim3(1u), block};
  }

  const std::int64_t threads_i64 = static_cast<std::int64_t>(threads);
  std::int64_t requested_blocks = (N + threads_i64 - 1) / threads_i64;
  if (requested_blocks <= 0) {
    requested_blocks = 1;
  }

  const unsigned int device_max = std::max(1u, caps.max_grid_x);

  unsigned int env_cap = device_max;
  const auto& env = vbt::cuda::reduction::get_cuda_reduction_env_config();
  if (env.cuda_max_blocks_cap > 0) {
    const std::int64_t capped = std::min<std::int64_t>(
        env.cuda_max_blocks_cap,
        static_cast<std::int64_t>(device_max));
    env_cap = static_cast<unsigned int>(std::max<std::int64_t>(capped, 1));
  }

  const unsigned int max_blocks = std::max(1u, std::min(device_max, env_cap));

  std::int64_t final_blocks_i64 = std::min<std::int64_t>(
      requested_blocks,
      static_cast<std::int64_t>(max_blocks));
  if (final_blocks_i64 <= 0) {
    final_blocks_i64 = 1;
  }

  unsigned int grid_x = static_cast<unsigned int>(final_blocks_i64);

#if VBT_INTERNAL_TESTS
  const std::optional<unsigned int> cap =
      vbt::cuda::reduction::get_cuda_reduction_grid_x_cap_for_tests();
  if (cap.has_value()) {
    unsigned int c = *cap;
    if (out_numel_for_test_cap > 0) {
      const unsigned long long out_u64 =
          static_cast<unsigned long long>(out_numel_for_test_cap);
      const unsigned long long out_clamped = std::min<unsigned long long>(
          out_u64,
          static_cast<unsigned long long>(
              std::numeric_limits<unsigned int>::max()));
      c = std::min<unsigned int>(c, static_cast<unsigned int>(out_clamped));
    }
    if (c == 0u) {
      c = 1u;
    }
    grid_x = std::max(1u, std::min(grid_x, c));
  }
#endif

  return Launch1DConfig{dim3(grid_x), block};
#endif
}

static inline Launch1DConfig make_k2_launch(std::int64_t out_numel,
                                            int device_index) {
#if !VBT_WITH_CUDA
  (void)out_numel;
  (void)device_index;
  return Launch1DConfig{dim3(1u), dim3(1u)};
#else
  const auto caps = vbt::cuda::get_device_caps(
      static_cast<vbt::cuda::DeviceIndex>(device_index));

  unsigned int threads = 256u;
  if (caps.max_threads_per_block > 0) {
    threads = std::min<unsigned int>(threads, caps.max_threads_per_block);
  }
  if (threads == 0u) {
    threads = 1u;
  }
  threads = std::min<unsigned int>(256u, floor_pow2_u32(threads));
  if (threads == 0u) {
    threads = 1u;
  }

  dim3 block(threads);

  std::int64_t requested_blocks = out_numel;
  if (requested_blocks <= 0) {
    requested_blocks = 1;
  }

  const unsigned int device_max = std::max(1u, caps.max_grid_x);

  unsigned int env_cap = device_max;
  const auto& env = vbt::cuda::reduction::get_cuda_reduction_env_config();
  if (env.cuda_max_blocks_cap > 0) {
    const std::int64_t capped = std::min<std::int64_t>(
        env.cuda_max_blocks_cap,
        static_cast<std::int64_t>(device_max));
    env_cap = static_cast<unsigned int>(std::max<std::int64_t>(capped, 1));
  }

  const unsigned int max_blocks = std::max(1u, std::min(device_max, env_cap));

  std::int64_t final_blocks_i64 = std::min<std::int64_t>(
      requested_blocks,
      static_cast<std::int64_t>(max_blocks));
  if (final_blocks_i64 <= 0) {
    final_blocks_i64 = 1;
  }

  unsigned int grid_x = static_cast<unsigned int>(final_blocks_i64);

#if VBT_INTERNAL_TESTS
  const std::optional<unsigned int> cap =
      vbt::cuda::reduction::get_cuda_reduction_grid_x_cap_for_tests();
  if (cap.has_value()) {
    unsigned int c = *cap;
    if (out_numel > 0) {
      const unsigned long long out_u64 =
          static_cast<unsigned long long>(out_numel);
      const unsigned long long out_clamped = std::min<unsigned long long>(
          out_u64,
          static_cast<unsigned long long>(
              std::numeric_limits<unsigned int>::max()));
      c = std::min<unsigned int>(c, static_cast<unsigned int>(out_clamped));
    }
    if (c == 0u) {
      c = 1u;
    }
    grid_x = std::max(1u, std::min(grid_x, c));
  }
#endif

  return Launch1DConfig{dim3(grid_x), block};
#endif
}

template <typename T, typename Op>
struct ReductionLauncher {
  T* out;
  const T* in;
  dim3 grid;
  dim3 block;
  Op op;

  DeviceStrideMeta out_meta;
  DeviceStrideMeta in_meta;
  std::int64_t N{0};  // Number of INPUT elements

  void operator()(const TensorIter& /*iter*/, bool use32, void* stream_ptr) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    if (use32) {
      atomic_reduce_kernel<T, Op, std::int32_t><<<grid, block, 0, stream>>>(
          out,
          out_meta,
          in,
          in_meta,
          static_cast<std::int32_t>(N),
          op);
    } else {
      atomic_reduce_kernel<T, Op, std::int64_t><<<grid, block, 0, stream>>>(
          out,
          out_meta,
          in,
          in_meta,
          static_cast<std::int64_t>(N),
          op);
    }
  }
};

// --- Unified CUDA reduction dispatcher (K1 atomic + staged K2) ---

enum class CudaReduceKind : std::uint8_t {
  Sum,
  Prod,
  Min,
  Max,
  Mean,
};

static std::vector<int64_t> normalize_reduction_dims(std::vector<int64_t> dims, int64_t R, const char* name) {
  if (dims.empty()) {
    dims.resize(static_cast<std::size_t>(R));
    std::iota(dims.begin(), dims.end(), 0);
  }

  std::vector<int64_t> norm;
  norm.reserve(dims.size());
  for (int64_t d : dims) {
    if (d < 0) d += R;
    if (d < 0 || d >= R) {
      throw std::invalid_argument(std::string(name) + ": dim out of range");
    }
    norm.push_back(d);
  }
  std::sort(norm.begin(), norm.end());
  norm.erase(std::unique(norm.begin(), norm.end()), norm.end());
  return norm;
}

static TensorImpl allocate_reduction_out(const TensorImpl& self,
                                        const std::vector<int64_t>& norm_dims,
                                        bool keepdim) {
  const auto& in_sizes = self.sizes();
  const int64_t R = static_cast<int64_t>(in_sizes.size());

  std::vector<bool> reduced(static_cast<std::size_t>(R), false);
  for (int64_t d : norm_dims) {
    reduced[static_cast<std::size_t>(d)] = true;
  }

  std::vector<int64_t> out_sizes;
  if (keepdim) {
    out_sizes.assign(in_sizes.begin(), in_sizes.end());
    for (int64_t d : norm_dims) out_sizes[static_cast<std::size_t>(d)] = 1;
  } else {
    out_sizes.reserve(static_cast<std::size_t>(R - static_cast<int64_t>(norm_dims.size())));
    for (int64_t d = 0; d < R; ++d) {
      if (!reduced[static_cast<std::size_t>(d)]) out_sizes.push_back(in_sizes[static_cast<std::size_t>(d)]);
    }
  }

  int64_t n = 1;
  for (auto s : out_sizes) {
    if (!vbt::core::checked_mul_i64(n, s, n)) {
      throw std::runtime_error("vt::reduce: output size overflow");
    }
  }

  const size_t nbytes = static_cast<size_t>(n) * vbt::core::itemsize(self.dtype());
  auto storage = vbt::cuda::new_cuda_storage(nbytes, self.device().index);

  // Stride computation (contiguous)
  std::vector<int64_t> strides(out_sizes.size());
  int64_t acc = 1;
  for (ptrdiff_t i = (ptrdiff_t)out_sizes.size() - 1; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] = acc;
    acc *= out_sizes[static_cast<std::size_t>(i)];
  }

  return TensorImpl(storage, out_sizes, strides, 0, self.dtype(), self.device());
}

static bool try_compute_slice_len(const TensorImpl& self,
                                 const std::vector<int64_t>& norm_dims,
                                 int64_t& out_slice_len) noexcept {
  const auto& sizes = self.sizes();
  int64_t count = 1;
  for (int64_t d : norm_dims) {
    int64_t next = 0;
    if (!vbt::core::checked_mul_i64(count, sizes[static_cast<std::size_t>(d)], next)) {
      return false;
    }
    count = next;
  }
  out_slice_len = count;
  return true;
}

template <typename T>
static void launch_fill_constant(TensorImpl& out, T val, vbt::cuda::Stream stream) {
  const std::int64_t out_N = out.numel();
  if (out_N <= 0) return;
  const int device_index = out.device().index;
  Launch1DConfig cfg = make_1d_launch(out_N, device_index, out_N);
  fill_kernel<T, int64_t><<<cfg.grid, cfg.block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
      static_cast<T*>(out.data()), val, out_N);
}

template <template <typename> class OpT>
static void run_atomic_reduce_k1(const TensorImpl& self,
                                 TensorImpl& out,
                                 const TensorIter& iter,
                                 const DeviceStrideMeta& out_meta,
                                 const DeviceStrideMeta& in_meta,
                                 std::int64_t N,
                                 const Launch1DConfig& launch,
                                 vbt::cuda::Stream stream) {
  // Initialize output to the identity.
  if (self.dtype() == ScalarType::Float32) {
    launch_fill_constant<float>(out, OpT<float>::init(), stream);
  } else if (self.dtype() == ScalarType::Int64) {
    launch_fill_constant<long long>(out, OpT<long long>::init(), stream);
  } else {
    // Caller is expected to validate dtype before calling.
    throw std::invalid_argument("vt::reduce: unsupported dtype");
  }

  auto run = [&](auto tag) {
    using T = decltype(tag);
    using Op = OpT<T>;
    ReductionLauncher<T, Op> launcher{
        static_cast<T*>(out.data()),
        static_cast<const T*>(self.data()),
        launch.grid,
        launch.block,
        Op{},
        out_meta,
        in_meta,
        N};
    vbt::core::ti_gpu_kernel(iter, launcher);
  };

  if (self.dtype() == ScalarType::Float32) {
    run(float(0));
  } else {
    run(static_cast<long long>(0));
  }
}

#if VBT_INTERNAL_TESTS
static bool want_cuda_reduce_plan_for_tests(
    const vbt::cuda::reduction::CudaReductionEnvConfig& env,
    bool policy_override_active,
    vbt::cuda::reduction::CudaReductionKernelPolicy requested_policy) noexcept {
  const bool E = env.staged_default;
  if (!policy_override_active) {
    return E;
  }

  using vbt::cuda::reduction::CudaReductionKernelPolicy;
  switch (requested_policy) {
    case CudaReductionKernelPolicy::ForceK2IfEligible:
    case CudaReductionKernelPolicy::ForceK2Strict:
    case CudaReductionKernelPolicy::ForceK2MultiIfEligible:
    case CudaReductionKernelPolicy::ForceK2MultiStrict:
    case CudaReductionKernelPolicy::ForceK3IfEligible:
    case CudaReductionKernelPolicy::ForceK3Strict:
      return true;
    case CudaReductionKernelPolicy::ForceK1:
      return false;
    case CudaReductionKernelPolicy::Auto:
      return E;
  }
  return false;
}
#endif  // VBT_INTERNAL_TESTS

static TensorImpl cuda_reduce_dispatch(CudaReduceKind kind,
                                      const TensorImpl& self,
                                      std::vector<int64_t> dims,
                                      bool keepdim,
                                      const char* name) {
#if VBT_WITH_CUDA
  if (self.device().type != kDLCUDA) {
    throw std::runtime_error(std::string(name) + ": expected CUDA tensor");
  }

  DeviceGuard g(self.device().index);

#if VBT_INTERNAL_TESTS
  const auto requested_policy =
      vbt::cuda::reduction::get_cuda_reduction_kernel_policy_for_tests();
  const bool policy_override_active =
      vbt::cuda::reduction::cuda_reduction_kernel_policy_override_is_active_for_tests();
  const bool k2multi_stream_mismatch_injection =
      vbt::cuda::reduction::cuda_reduction_k2multi_stream_mismatch_injection_enabled_for_tests();
#endif

  const auto& in_sizes = self.sizes();
  const int64_t R = static_cast<int64_t>(in_sizes.size());

  // Dtype validation must happen before empty semantics.
  const ScalarType st = self.dtype();
  if (kind == CudaReduceKind::Mean) {
    if (st != ScalarType::Float32) {
#if VBT_INTERNAL_TESTS
      std::int64_t stats_out_numel = 0;
      std::int64_t stats_slice_len = 0;
      if (dims.empty() && R == 0) {
        stats_out_numel = 1;
        stats_slice_len = 1;
      } else {
        try {
          const std::vector<int64_t> stats_norm_dims =
              normalize_reduction_dims(dims, R, name);
          (void)try_compute_slice_len(self, stats_norm_dims, stats_slice_len);

          std::vector<bool> reduced(static_cast<std::size_t>(R), false);
          for (int64_t d : stats_norm_dims) {
            if (d >= 0 && d < R) {
              reduced[static_cast<std::size_t>(d)] = true;
            }
          }

          std::int64_t prod = 1;
          for (int64_t d = 0; d < R; ++d) {
            if (reduced[static_cast<std::size_t>(d)]) {
              continue;
            }
            std::int64_t next = 0;
            if (!vbt::core::checked_mul_i64(
                    prod, in_sizes[static_cast<std::size_t>(d)], next)) {
              prod = 0;
              break;
            }
            prod = next;
          }
          stats_out_numel = prod;
        } catch (...) {
        }
      }

      vbt::cuda::reduction::CudaReductionLastStats stats{};
      stats.selected_kernel = vbt::cuda::reduction::CudaReductionKernel::None;
      stats.requested_policy = requested_policy;
      stats.policy_override_active = policy_override_active;
      stats.want_plan = false;
      stats.ineligible_reason = vbt::cuda::reduction::CudaReduceIneligibleReason::None;
      stats.out_numel = stats_out_numel;
      stats.slice_len = stats_slice_len;
      vbt::cuda::reduction::set_cuda_reduction_last_stats_for_tests(stats);
#endif
      throw std::invalid_argument("mean: expected dtype=float32");
    }
  } else {
    if (!(st == ScalarType::Float32 || st == ScalarType::Int64)) {
#if VBT_INTERNAL_TESTS
      std::int64_t stats_out_numel = 0;
      std::int64_t stats_slice_len = 0;
      if (dims.empty() && R == 0) {
        stats_out_numel = 1;
        stats_slice_len = 1;
      } else {
        try {
          const std::vector<int64_t> stats_norm_dims =
              normalize_reduction_dims(dims, R, name);
          (void)try_compute_slice_len(self, stats_norm_dims, stats_slice_len);

          std::vector<bool> reduced(static_cast<std::size_t>(R), false);
          for (int64_t d : stats_norm_dims) {
            if (d >= 0 && d < R) {
              reduced[static_cast<std::size_t>(d)] = true;
            }
          }

          std::int64_t prod = 1;
          for (int64_t d = 0; d < R; ++d) {
            if (reduced[static_cast<std::size_t>(d)]) {
              continue;
            }
            std::int64_t next = 0;
            if (!vbt::core::checked_mul_i64(
                    prod, in_sizes[static_cast<std::size_t>(d)], next)) {
              prod = 0;
              break;
            }
            prod = next;
          }
          stats_out_numel = prod;
        } catch (...) {
        }
      }

      vbt::cuda::reduction::CudaReductionLastStats stats{};
      stats.selected_kernel = vbt::cuda::reduction::CudaReductionKernel::None;
      stats.requested_policy = requested_policy;
      stats.policy_override_active = policy_override_active;
      stats.want_plan = false;
      stats.ineligible_reason = vbt::cuda::reduction::CudaReduceIneligibleReason::None;
      stats.out_numel = stats_out_numel;
      stats.slice_len = stats_slice_len;
      vbt::cuda::reduction::set_cuda_reduction_last_stats_for_tests(stats);
#endif
      throw std::invalid_argument(std::string(name) + ": unsupported dtype");
    }
  }

  // dim=None path: reduce over all dims. Special-case scalars: TensorIterator
  // reductions require at least one dim.
  if (dims.empty() && R == 0) {
    TensorImpl out = vbt::core::clone_cuda(self);
    auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(self.device().index));

#if VBT_INTERNAL_TESTS
    vbt::cuda::reduction::CudaReductionLastStats stats{};
    stats.selected_kernel = vbt::cuda::reduction::CudaReductionKernel::None;
    stats.requested_policy = requested_policy;
    stats.policy_override_active = policy_override_active;
    stats.want_plan = false;
    stats.ineligible_reason = vbt::cuda::reduction::CudaReduceIneligibleReason::None;
    stats.out_numel = 1;
    stats.slice_len = 1;
    vbt::cuda::reduction::set_cuda_reduction_last_stats_for_tests(stats);
#endif

    vbt::cuda::record_stream(out.storage(), stream);
    return out;
  }

  std::vector<int64_t> norm_dims = normalize_reduction_dims(std::move(dims), R, name);

  TensorImpl out = allocate_reduction_out(self, norm_dims, keepdim);
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(self.device().index));

#if VBT_INTERNAL_TESTS
  // Test-only stream mismatch injection for K2-multi workspace record_stream hazards.
  // Keep the allocation stream (current) unchanged, but launch on a non-default stream.
  if (k2multi_stream_mismatch_injection &&
      policy_override_active &&
      requested_policy ==
          vbt::cuda::reduction::CudaReductionKernelPolicy::ForceK2MultiStrict) {
    stream = vbt::cuda::getStreamFromPool(/*high_priority=*/false,
                                         static_cast<vbt::cuda::DeviceIndex>(self.device().index));
  }
#endif

  // Reduce-all contiguous sum fast path (CUB).
  if (kind == CudaReduceKind::Sum) {
    bool env_enabled = cuda_cub_reduce_all_sum_enabled();
#if VBT_INTERNAL_TESTS
    // Forcing policies must not be bypassed by the CUB early-return path.
    if (policy_override_active &&
        requested_policy != vbt::cuda::reduction::CudaReductionKernelPolicy::Auto) {
      env_enabled = false;
    }
#endif
    if (env_enabled &&
        self.is_contiguous() &&
        out.numel() == 1 &&
        static_cast<std::int64_t>(norm_dims.size()) == R &&
        vbt::cuda::streamCaptureStatus(stream) == vbt::cuda::CaptureStatus::None) {

      bool reduce_all = true;
      for (std::int64_t i = 0; i < R; ++i) {
        if (norm_dims[static_cast<std::size_t>(i)] != i) {
          reduce_all = false;
          break;
        }
      }

      const std::int64_t n64 = self.numel();
      if (reduce_all && n64 <= static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        auto& alloc = vbt::cuda::Allocator::get(static_cast<vbt::cuda::DeviceIndex>(self.device().index));
        const int n = static_cast<int>(n64);

        if (st == ScalarType::Float32) {
          vbt::cuda::cub::reduce_all_contig_sum_f32(
              alloc,
              stream,
              static_cast<const float*>(self.data()),
              n,
              static_cast<float*>(out.data()));
        } else {
          vbt::cuda::cub::reduce_all_contig_sum_i64(
              alloc,
              stream,
              static_cast<const long long*>(self.data()),
              n,
              static_cast<long long*>(out.data()));
        }

#if VBT_INTERNAL_TESTS
        g_cub_reduce_all_sum_fastpath_calls.fetch_add(1, std::memory_order_relaxed);

        vbt::cuda::reduction::CudaReductionLastStats stats{};
        stats.selected_kernel = vbt::cuda::reduction::CudaReductionKernel::None;
        stats.requested_policy = requested_policy;
        stats.policy_override_active = policy_override_active;
        stats.want_plan = false;
        stats.ineligible_reason = vbt::cuda::reduction::CudaReduceIneligibleReason::None;
        stats.out_numel = 1;
        stats.slice_len = n64;
        vbt::cuda::reduction::set_cuda_reduction_last_stats_for_tests(stats);
#endif

        vbt::cuda::record_stream(out.storage(), stream);
        return out;
      }
    }
  }

  auto epilogue = [&]() { vbt::cuda::record_stream(out.storage(), stream); };


  const int64_t out_numel = out.numel();
  if (out_numel == 0) {
    // Case A: output is empty -> return empty output. No kernels launched.
#if VBT_INTERNAL_TESTS
    vbt::cuda::reduction::CudaReductionLastStats stats{};
    stats.selected_kernel = vbt::cuda::reduction::CudaReductionKernel::None;
    stats.requested_policy = requested_policy;
    stats.policy_override_active = policy_override_active;
    stats.want_plan = false;
    stats.ineligible_reason = vbt::cuda::reduction::CudaReduceIneligibleReason::EmptyOutNumel;
    stats.out_numel = out_numel;
    vbt::cuda::reduction::set_cuda_reduction_last_stats_for_tests(stats);
#endif
    epilogue();
    return out;
  }

  int64_t slice_len = 0;
  if (!try_compute_slice_len(self, norm_dims, slice_len)) {
#if VBT_INTERNAL_TESTS
    vbt::cuda::reduction::CudaReductionLastStats stats{};
    stats.selected_kernel = vbt::cuda::reduction::CudaReductionKernel::None;
    stats.requested_policy = requested_policy;
    stats.policy_override_active = policy_override_active;
    stats.want_plan = false;
    stats.ineligible_reason = vbt::cuda::reduction::CudaReduceIneligibleReason::Overflow;
    stats.out_numel = out_numel;
    vbt::cuda::reduction::set_cuda_reduction_last_stats_for_tests(stats);
#endif
    throw std::runtime_error(std::string(name) + ": slice_len overflow");
  }

  if (slice_len == 0) {
    // Case B: empty reduction domain with non-empty output.
#if VBT_INTERNAL_TESTS
    vbt::cuda::reduction::CudaReductionLastStats stats{};
    stats.selected_kernel = vbt::cuda::reduction::CudaReductionKernel::None;
    stats.requested_policy = requested_policy;
    stats.policy_override_active = policy_override_active;
    stats.want_plan = false;
    stats.ineligible_reason = vbt::cuda::reduction::CudaReduceIneligibleReason::EmptySlice;
    stats.out_numel = out_numel;
    stats.slice_len = slice_len;
    vbt::cuda::reduction::set_cuda_reduction_last_stats_for_tests(stats);
#endif
    switch (kind) {
      case CudaReduceKind::Sum:
        if (st == ScalarType::Float32) {
          launch_fill_constant<float>(out, 0.0f, stream);
        } else {
          launch_fill_constant<long long>(out, 0LL, stream);
        }
        epilogue();
        return out;
      case CudaReduceKind::Prod:
        if (st == ScalarType::Float32) {
          launch_fill_constant<float>(out, 1.0f, stream);
        } else {
          launch_fill_constant<long long>(out, 1LL, stream);
        }
        epilogue();
        return out;
      case CudaReduceKind::Mean:
        // mean is float32-only
        launch_fill_constant<float>(out, nanf(""), stream);
        epilogue();
        return out;
      case CudaReduceKind::Min:
        throw std::runtime_error("amin: empty");
      case CudaReduceKind::Max:
        throw std::runtime_error("amax: empty");
    }
  }

  // Non-empty reduction domain.
  const Launch1DConfig k1_launch =
      make_1d_launch(self.numel(), self.device().index, out_numel);

  const bool k2_kind_dtype_supported =
      ((kind == CudaReduceKind::Sum || kind == CudaReduceKind::Prod) &&
       (st == ScalarType::Float32 || st == ScalarType::Int64)) ||
      (kind == CudaReduceKind::Mean && st == ScalarType::Float32);

#if VBT_INTERNAL_TESTS
  using vbt::cuda::reduction::CudaReductionKernelPolicy;

  // Strict forcing policies remain internal-test-only.
  //
  // ForceK2Strict/ForceK2MultiStrict/ForceK3Strict are supported for staged
  // {sum, prod, mean} (on supported dtypes). Other ops/dtypes throw a
  // "... not implemented" error, but we still publish last-stats so tests can
  // inspect the throw path.
  auto throw_strict_not_implemented = [&](const char* msg) {
    vbt::cuda::reduction::CudaReductionLastStats stats{};
    stats.selected_kernel = vbt::cuda::reduction::CudaReductionKernel::None;
    stats.requested_policy = requested_policy;
    stats.policy_override_active = policy_override_active;
    stats.want_plan = false;
    stats.ineligible_reason = vbt::cuda::reduction::CudaReduceIneligibleReason::None;
    stats.out_numel = out_numel;
    stats.slice_len = slice_len;
    vbt::cuda::reduction::set_cuda_reduction_last_stats_for_tests(stats);

    throw std::invalid_argument(std::string(name) + msg);
  };

  switch (requested_policy) {
    case CudaReductionKernelPolicy::Auto:
    case CudaReductionKernelPolicy::ForceK1:
    case CudaReductionKernelPolicy::ForceK2IfEligible:
    case CudaReductionKernelPolicy::ForceK2MultiIfEligible:
    case CudaReductionKernelPolicy::ForceK3IfEligible:
      break;
    case CudaReductionKernelPolicy::ForceK2Strict:
      // Strict K2 is supported for staged {sum, prod, mean} on supported dtypes.
      if (!k2_kind_dtype_supported) {
        throw_strict_not_implemented(": forced kernel K2 not implemented");
      }
      break;
    case CudaReductionKernelPolicy::ForceK2MultiStrict:
      // Strict K2Multi is supported for staged {sum, prod, mean} on supported dtypes.
      if (!k2_kind_dtype_supported) {
        throw_strict_not_implemented(": forced kernel K2Multi not implemented");
      }
      break;
    case CudaReductionKernelPolicy::ForceK3Strict:
      // Strict K3 is supported for staged {sum, prod, mean} on supported dtypes.
      if (!k2_kind_dtype_supported) {
        throw_strict_not_implemented(": forced kernel K3 not implemented");
      }
      break;
  }
#endif

  TensorIterConfig cfg;
  cfg.check_mem_overlap(true);
  cfg.is_reduction(true);
  cfg.add_output(OptionalTensorImplRef(&out, true), IterOperandRole::ReduceOutput);
  cfg.add_input(self);
  cfg.set_reduce_dims(norm_dims, keepdim);
  TensorIter iter = cfg.build();

  DeviceStrideMeta out_meta{}, in_meta{};
  // Reductions support rank up to kTensorIterMaxRank (64). Export full-rank
  // metas here (instead of kTensorIterCudaMaxNdim=25 used by many elementwise
  // CUDA kernels) to avoid rejecting high-rank reductions.
  //
  // Perf note: compute_offset_elems() performs an O(ndim) loop in device code;
  // high-rank metas (even with many size==1 dims) can increase per-element
  // indexing overhead.
  iter.export_device_meta(0, &out_meta, kTensorIterMaxRank);
  iter.export_device_meta(1, &in_meta, kTensorIterMaxRank);
  const std::int64_t N = iter.numel();  // Number of INPUT elements

  using vbt::cuda::reduction::CudaReduceIneligibleReason;
  using vbt::cuda::reduction::CudaReducePlanBuildResult;

  const auto& env = vbt::cuda::reduction::get_cuda_reduction_env_config();

  bool want_plan = env.staged_default;
#if VBT_INTERNAL_TESTS
  want_plan =
      want_cuda_reduce_plan_for_tests(env, policy_override_active, requested_policy);
#endif
  if (kind == CudaReduceKind::Min || kind == CudaReduceKind::Max) {
    // min/max are not staged yet; never build a plan for non-empty reductions.
    want_plan = false;
  }

  CudaReducePlanBuildResult plan_result{};
  if (want_plan) {
    plan_result = vbt::cuda::reduction::build_cuda_reduce_plan_noalloc(
        out_meta,
        in_meta,
        std::span<const int64_t>(norm_dims.data(), norm_dims.size()),
        out_numel,
        slice_len);
  }

  const CudaReduceIneligibleReason plan_reason =
      want_plan ? plan_result.ineligible_reason : CudaReduceIneligibleReason::None;

#if VBT_INTERNAL_TESTS
  if (policy_override_active &&
      requested_policy == vbt::cuda::reduction::CudaReductionKernelPolicy::ForceK2Strict &&
      want_plan &&
      plan_reason != CudaReduceIneligibleReason::None) {
    // Strict ineligible: no kernel launched; publish stats before throwing.
    vbt::cuda::reduction::CudaReductionLastStats stats{};
    stats.selected_kernel = vbt::cuda::reduction::CudaReductionKernel::None;
    stats.requested_policy = requested_policy;
    stats.policy_override_active = policy_override_active;
    stats.want_plan = want_plan;
    stats.plan_iter_ndim = plan_result.plan.iter_ndim;
    stats.plan_kept_ndim = plan_result.plan.kept_ndim;
    stats.plan_red_ndim = plan_result.plan.red_ndim;
    stats.plan_red_linear_stride = plan_result.plan.red_linear_stride;
    stats.ineligible_reason = plan_reason;
    stats.out_numel = out_numel;
    stats.slice_len = slice_len;
    stats.grid_x = stats.grid_y = stats.grid_z = 0;
    stats.block_x = stats.block_y = stats.block_z = 0;
    vbt::cuda::reduction::set_cuda_reduction_last_stats_for_tests(stats);

    throw std::invalid_argument(std::string(name) + ": forced kernel K2 ineligible");
  }
#endif

#if VBT_INTERNAL_TESTS
  if (policy_override_active &&
      requested_policy ==
          vbt::cuda::reduction::CudaReductionKernelPolicy::ForceK3Strict) {
    using vbt::cuda::reduction::CudaReductionKernel;

    const bool k3_eligible =
        (want_plan &&
         plan_reason == CudaReduceIneligibleReason::None &&
         k2_kind_dtype_supported &&
         plan_result.plan.red_linear_stride >= 0);

    if (!k3_eligible) {
      // Strict ineligible: no kernel launched; publish stats before throwing.
      vbt::cuda::reduction::CudaReductionLastStats stats{};
      stats.selected_kernel = CudaReductionKernel::None;
      stats.requested_policy = requested_policy;
      stats.policy_override_active = policy_override_active;
      stats.want_plan = want_plan;
      if (want_plan) {
        stats.plan_iter_ndim = plan_result.plan.iter_ndim;
        stats.plan_kept_ndim = plan_result.plan.kept_ndim;
        stats.plan_red_ndim = plan_result.plan.red_ndim;
        stats.plan_red_linear_stride = plan_result.plan.red_linear_stride;
      }
      stats.ineligible_reason = plan_reason;
      if (stats.ineligible_reason == CudaReduceIneligibleReason::None) {
        // K3-specific gating failures (e.g. negative reduced stride) map to the
        // generic Overflow code for now.
        stats.ineligible_reason = CudaReduceIneligibleReason::Overflow;
      }
      stats.out_numel = out_numel;
      stats.slice_len = slice_len;
      stats.grid_x = stats.grid_y = stats.grid_z = 0;
      stats.block_x = stats.block_y = stats.block_z = 0;
      vbt::cuda::reduction::set_cuda_reduction_last_stats_for_tests(stats);

      throw std::invalid_argument(std::string(name) +
                                 ": forced kernel K3 ineligible");
    }
  }
#endif

#if VBT_INTERNAL_TESTS
  // K2-multi is forced-only; Auto must never select it.
  // ForceK2MultiStrict launches the real K2-multi kernel for staged {sum, prod, mean}
  // (same dtype allowlist as K2). If ineligible, strict throws "ineligible".
  if (policy_override_active &&
      requested_policy == vbt::cuda::reduction::CudaReductionKernelPolicy::ForceK2MultiStrict) {
    using vbt::cuda::reduction::CudaReductionKernel;

    const bool k2_eligible =
        (want_plan &&
         plan_reason == CudaReduceIneligibleReason::None &&
         k2_kind_dtype_supported);

    vbt::cuda::reduction::CudaK2MultiFaultMode fault_mode =
        vbt::cuda::reduction::get_cuda_reduction_k2multi_fault_mode_for_tests();
    if (st != ScalarType::Float32) {
      // Fault injection is defined only for float32.
      fault_mode = vbt::cuda::reduction::CudaK2MultiFaultMode::None;
    }

    unsigned int effective_ctas_per_output = 0;
    vbt::cuda::reduction::K2MultiWorkspaceLayout ws_layout{};

    vbt::core::StoragePtr workspace{};
    void* partials_ptr = nullptr;
    unsigned int* semaphores_ptr = nullptr;

    bool k2multi_gates_pass = false;

    if (k2_eligible) {
      constexpr unsigned int kDefaultCtasPerOutput = 2u;
      constexpr unsigned int kMaxCtasPerOutput = 32u;

      unsigned int requested_ctas = kDefaultCtasPerOutput;
      const std::optional<unsigned int> ctas_override =
          vbt::cuda::reduction::get_cuda_reduction_k2multi_ctas_per_output_for_tests();
      if (ctas_override.has_value()) {
        requested_ctas = *ctas_override;
      }

      const auto caps = vbt::cuda::get_device_caps(
          static_cast<vbt::cuda::DeviceIndex>(self.device().index));

      unsigned int max_ctas = kMaxCtasPerOutput;
      if (caps.max_grid_y > 0) {
        max_ctas = std::min<unsigned int>(max_ctas, caps.max_grid_y);
      }
      if (max_ctas == 0u) {
        max_ctas = 1u;
      }

      effective_ctas_per_output = requested_ctas;
      if (effective_ctas_per_output < 2u) {
        effective_ctas_per_output = 2u;
      }
      if (effective_ctas_per_output > max_ctas) {
        effective_ctas_per_output = max_ctas;
      }

      const bool ctas_ok = (effective_ctas_per_output >= 2u);
      const bool red_stride_ok = (plan_result.plan.red_linear_stride >= 0);
      const bool capture_ok =
          (vbt::cuda::streamCaptureStatus(stream) == vbt::cuda::CaptureStatus::None);

      if (ctas_ok && red_stride_ok && capture_ok) {
        const std::uint32_t C = static_cast<std::uint32_t>(effective_ctas_per_output);
        if (vbt::cuda::reduction::compute_k2multi_workspace_layout(
                out_numel,
                C,
                vbt::core::itemsize(st),
                &ws_layout)) {
          try {
            workspace = vbt::cuda::new_cuda_storage(
                ws_layout.total_bytes,
                self.device().index);
            vbt::cuda::record_stream(workspace, stream);
            // record_stream makes it safe to release workspace during stack unwinding:
            // the allocator defers reuse until this stream's work completes.

            // Zero semaphores on the launch stream.
            unsigned char* base = static_cast<unsigned char*>(workspace->data());
            partials_ptr = static_cast<void*>(base);
            void* sema_ptr_void = static_cast<void*>(base + ws_layout.sema_off);
            semaphores_ptr = static_cast<unsigned int*>(sema_ptr_void);
            cudaError_t st_memset = cudaMemsetAsync(
                sema_ptr_void,
                0,
                ws_layout.semaphores_bytes,
                reinterpret_cast<cudaStream_t>(stream.handle()));

            bool ok = (st_memset == cudaSuccess);
            if (!ok) {
              // Clear sticky error; treat as an ineligibility gate failure.
              (void)cudaGetLastError();
            }

            if (ok &&
                fault_mode ==
                    vbt::cuda::reduction::CudaK2MultiFaultMode::SignalButSkipPartialWrite) {
              // Deterministic fault injection: poison partials before launch so
              // the injected CTA leaves a stable, incorrect value.
              int poison_byte = 0xFF;  // sum/mean f32: 0xFFFFFFFF => NaN
              if (kind == CudaReduceKind::Prod) {
                poison_byte = 0x00;  // prod f32: 0.0f
              }
              cudaError_t st_poison = cudaMemsetAsync(
                  partials_ptr,
                  poison_byte,
                  ws_layout.partials_bytes,
                  reinterpret_cast<cudaStream_t>(stream.handle()));
              ok = (st_poison == cudaSuccess);
              if (!ok) {
                (void)cudaGetLastError();
              }
            }

            if (ok && k2multi_stream_mismatch_injection) {
              // Ensure the workspace initialization (semaphores + optional partial poisoning)
              // is complete before we launch the kernel. This makes the stream-mismatch
              // record_stream hazard test deterministic (clobbers must not be overwritten
              // by the initialization memset).
              cudaError_t st_sync_ws = cudaStreamSynchronize(
                  reinterpret_cast<cudaStream_t>(stream.handle()));
              ok = (st_sync_ws == cudaSuccess);
              if (!ok) {
                (void)cudaGetLastError();
              }
            }

            if (ok) {
              k2multi_gates_pass = true;
            }
          } catch (...) {
            // Allocation failure is treated as an ineligibility gate failure.
          }
        }
      }
    }

    if (k2_eligible && k2multi_gates_pass) {
      const Launch1DConfig base_launch = make_k2_launch(out_numel, self.device().index);
      const dim3 grid(base_launch.grid.x, effective_ctas_per_output, 1u);
      const dim3 block(base_launch.block.x, 1u, 1u);

      vbt::cuda::reduction::CudaReductionLastStats stats{};
      stats.selected_kernel = CudaReductionKernel::K2Multi;
      stats.requested_policy = requested_policy;
      stats.policy_override_active = policy_override_active;
      stats.want_plan = want_plan;
      stats.plan_iter_ndim = plan_result.plan.iter_ndim;
      stats.plan_kept_ndim = plan_result.plan.kept_ndim;
      stats.plan_red_ndim = plan_result.plan.red_ndim;
      stats.plan_red_linear_stride = plan_result.plan.red_linear_stride;

      stats.grid_x = grid.x;
      stats.grid_y = grid.y;
      stats.grid_z = grid.z;
      stats.block_x = block.x;
      stats.block_y = block.y;
      stats.block_z = block.z;

      stats.ineligible_reason = plan_reason;
      stats.out_numel = out_numel;
      stats.slice_len = slice_len;

      stats.k2multi_ctas_per_output = effective_ctas_per_output;
      stats.k2multi_workspace_partials_bytes = ws_layout.partials_bytes;
      stats.k2multi_workspace_sema_off = ws_layout.sema_off;
      stats.k2multi_workspace_total_bytes = ws_layout.total_bytes;
      stats.launch_stream_id = stream.id();

      vbt::cuda::reduction::set_cuda_reduction_last_stats_for_tests(stats);

      // Keep the grid-cap regression deterministic: when tests force a tiny
      // grid.x, missing out_idx looping would otherwise leave some outputs
      // uninitialized.
      if (vbt::cuda::reduction::get_cuda_reduction_grid_x_cap_for_tests().has_value()) {
        if (kind == CudaReduceKind::Prod) {
          if (st == ScalarType::Float32) {
            launch_fill_constant<float>(out, 1.0f, stream);
          } else if (st == ScalarType::Int64) {
            launch_fill_constant<long long>(out, 1LL, stream);
          } else {
            throw std::invalid_argument(std::string(name) +
                                       ": forced kernel K2Multi not implemented");
          }
        } else {
          if (st == ScalarType::Float32) {
            launch_fill_constant<float>(out, 0.0f, stream);
          } else if (st == ScalarType::Int64) {
            launch_fill_constant<long long>(out, 0LL, stream);
          } else {
            throw std::invalid_argument(std::string(name) +
                                       ": forced kernel K2Multi not implemented");
          }
        }
      }

      switch (kind) {
        case CudaReduceKind::Sum:
          if (st == ScalarType::Float32) {
            k2multi_reduce_kernel<float, K2SumOp>
                <<<grid,
                   block,
                   0,
                   reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                    plan_result.plan,
                    static_cast<const float*>(self.data()),
                    static_cast<float*>(out.data()),
                    static_cast<float*>(partials_ptr),
                    semaphores_ptr,
                    effective_ctas_per_output,
                    fault_mode);
          } else if (st == ScalarType::Int64) {
            k2multi_reduce_kernel<long long, K2SumOp>
                <<<grid,
                   block,
                   0,
                   reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                    plan_result.plan,
                    static_cast<const long long*>(self.data()),
                    static_cast<long long*>(out.data()),
                    static_cast<long long*>(partials_ptr),
                    semaphores_ptr,
                    effective_ctas_per_output,
                    fault_mode);
          } else {
            throw std::invalid_argument(std::string(name) +
                                       ": forced kernel K2Multi not implemented");
          }
          break;
        case CudaReduceKind::Prod:
          if (st == ScalarType::Float32) {
            k2multi_reduce_kernel<float, K2ProdOp>
                <<<grid,
                   block,
                   0,
                   reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                    plan_result.plan,
                    static_cast<const float*>(self.data()),
                    static_cast<float*>(out.data()),
                    static_cast<float*>(partials_ptr),
                    semaphores_ptr,
                    effective_ctas_per_output,
                    fault_mode);
          } else if (st == ScalarType::Int64) {
            k2multi_reduce_kernel<long long, K2ProdOp>
                <<<grid,
                   block,
                   0,
                   reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                    plan_result.plan,
                    static_cast<const long long*>(self.data()),
                    static_cast<long long*>(out.data()),
                    static_cast<long long*>(partials_ptr),
                    semaphores_ptr,
                    effective_ctas_per_output,
                    fault_mode);
          } else {
            throw std::invalid_argument(std::string(name) +
                                       ": forced kernel K2Multi not implemented");
          }
          break;
        case CudaReduceKind::Mean: {
          if (st != ScalarType::Float32) {
            throw std::invalid_argument(std::string(name) +
                                       ": forced kernel K2Multi not implemented");
          }

          // mean is implemented as sum + div, matching the K1/K2 paths.
          k2multi_reduce_kernel<float, K2SumOp>
              <<<grid,
                 block,
                 0,
                 reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  plan_result.plan,
                  static_cast<const float*>(self.data()),
                  static_cast<float*>(out.data()),
                  static_cast<float*>(partials_ptr),
                  semaphores_ptr,
                  effective_ctas_per_output,
                  fault_mode);

          const int64_t out_N = out.numel();
          const Launch1DConfig div_launch =
              make_1d_launch(out_N, self.device().index, out_numel);
          scalar_div_kernel<float, int64_t>
              <<<div_launch.grid,
                 div_launch.block,
                 0,
                 reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  static_cast<float*>(out.data()),
                  static_cast<float>(slice_len),
                  out_N);
          break;
        }
        default:
          // ForceK2MultiStrict is only exposed for staged {sum, prod, mean}.
          throw std::invalid_argument(std::string(name) +
                                     ": forced kernel K2Multi not implemented");
      }

      epilogue();
      return out;
    }

    // Strict ineligible: no kernel launched; publish stats before throwing.
    vbt::cuda::reduction::CudaReductionLastStats stats{};
    stats.selected_kernel = CudaReductionKernel::None;
    stats.requested_policy = requested_policy;
    stats.policy_override_active = policy_override_active;
    stats.want_plan = want_plan;
    if (want_plan) {
      stats.plan_iter_ndim = plan_result.plan.iter_ndim;
      stats.plan_kept_ndim = plan_result.plan.kept_ndim;
      stats.plan_red_ndim = plan_result.plan.red_ndim;
      stats.plan_red_linear_stride = plan_result.plan.red_linear_stride;
    }
    stats.ineligible_reason = plan_reason;
    if (stats.ineligible_reason == CudaReduceIneligibleReason::None) {
      // No dedicated reason for K2-multi-only gate failures yet.
      stats.ineligible_reason = CudaReduceIneligibleReason::Overflow;
    }
    stats.out_numel = out_numel;
    stats.slice_len = slice_len;
    stats.grid_x = stats.grid_y = stats.grid_z = 0;
    stats.block_x = stats.block_y = stats.block_z = 0;

    stats.k2multi_ctas_per_output = effective_ctas_per_output;
    stats.k2multi_workspace_partials_bytes = ws_layout.partials_bytes;
    stats.k2multi_workspace_sema_off = ws_layout.sema_off;
    stats.k2multi_workspace_total_bytes = ws_layout.total_bytes;
    stats.launch_stream_id = stream.id();

    vbt::cuda::reduction::set_cuda_reduction_last_stats_for_tests(stats);

    throw std::invalid_argument(std::string(name) +
                               ": forced kernel K2Multi ineligible");
  }
#endif

  bool k2_requested = env.staged_default;
  bool k3_requested = false;
#if VBT_INTERNAL_TESTS
  if (policy_override_active) {
    using vbt::cuda::reduction::CudaReductionKernelPolicy;
    switch (requested_policy) {
      case CudaReductionKernelPolicy::ForceK2IfEligible:
      case CudaReductionKernelPolicy::ForceK2Strict:
      case CudaReductionKernelPolicy::ForceK2MultiIfEligible:
        k2_requested = true;
        k3_requested = false;
        break;
      case CudaReductionKernelPolicy::ForceK3IfEligible:
      case CudaReductionKernelPolicy::ForceK3Strict:
        k2_requested = false;
        k3_requested = true;
        break;
      case CudaReductionKernelPolicy::Auto:
        k2_requested = env.staged_default;
        k3_requested = false;
        break;
      default:
        k2_requested = false;
        k3_requested = false;
        break;
    }
  }
#endif

  const bool use_k3 =
      (k3_requested &&
       want_plan &&
       plan_reason == CudaReduceIneligibleReason::None &&
       k2_kind_dtype_supported &&
       plan_result.plan.red_linear_stride >= 0);

  const bool use_k2 =
      (!use_k3 &&
       k2_requested &&
       want_plan &&
       plan_reason == CudaReduceIneligibleReason::None &&
       k2_kind_dtype_supported);

  Launch1DConfig k2_launch{dim3(1u), dim3(1u)};
  if (use_k2) {
    k2_launch = make_k2_launch(out_numel, self.device().index);
  }

  Launch1DConfig k3_launch{dim3(1u), dim3(1u)};
  if (use_k3) {
    k3_launch = make_1d_launch(out_numel, self.device().index, out_numel);
  }

  const Launch1DConfig& selected_launch =
      use_k3 ? k3_launch : (use_k2 ? k2_launch : k1_launch);

#if VBT_INTERNAL_TESTS
  using vbt::cuda::reduction::CudaReductionKernel;

  vbt::cuda::reduction::CudaReductionLastStats stats{};
  stats.selected_kernel =
      use_k3   ? CudaReductionKernel::K3
      : use_k2 ? CudaReductionKernel::K2
               : CudaReductionKernel::K1Atomic;
  stats.requested_policy = requested_policy;
  stats.policy_override_active = policy_override_active;
  stats.want_plan = want_plan;
  if (want_plan) {
    stats.plan_iter_ndim = plan_result.plan.iter_ndim;
    stats.plan_kept_ndim = plan_result.plan.kept_ndim;
    stats.plan_red_ndim = plan_result.plan.red_ndim;
    stats.plan_red_linear_stride = plan_result.plan.red_linear_stride;
  }
  stats.grid_x = selected_launch.grid.x;
  stats.grid_y = selected_launch.grid.y;
  stats.grid_z = selected_launch.grid.z;
  stats.block_x = selected_launch.block.x;
  stats.block_y = selected_launch.block.y;
  stats.block_z = selected_launch.block.z;
  stats.ineligible_reason = plan_reason;
  stats.out_numel = out_numel;
  stats.slice_len = slice_len;
  if (use_k2) {
    // k2_smem_bytes is a conceptual per-CTA scratch requirement for block
    // reduction; it is not interpreted as CUDA dynamic shared memory bytes.
    stats.k2_smem_bytes = static_cast<std::uint32_t>(
        static_cast<size_t>(selected_launch.block.x) * vbt::core::itemsize(st));
  }
  vbt::cuda::reduction::set_cuda_reduction_last_stats_for_tests(stats);
#endif

  switch (kind) {
    case CudaReduceKind::Sum:
      if (use_k3) {
#if VBT_INTERNAL_TESTS
        // Keep the grid-cap regression deterministic: when tests force a tiny
        // grid.x, missing tile_base looping would otherwise leave some outputs
        // uninitialized.
        if (vbt::cuda::reduction::get_cuda_reduction_grid_x_cap_for_tests().has_value()) {
          if (st == ScalarType::Float32) {
            launch_fill_constant<float>(out, 0.0f, stream);
          } else {
            launch_fill_constant<long long>(out, 0LL, stream);
          }
        }
#endif
        if (st == ScalarType::Float32) {
          k3_tile_outputs_reduce_kernel<float, K2SumOp>
              <<<selected_launch.grid,
                 selected_launch.block,
                 0,
                 reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  plan_result.plan,
                  static_cast<const float*>(self.data()),
                  static_cast<float*>(out.data()));
        } else {
          k3_tile_outputs_reduce_kernel<long long, K2SumOp>
              <<<selected_launch.grid,
                 selected_launch.block,
                 0,
                 reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  plan_result.plan,
                  static_cast<const long long*>(self.data()),
                  static_cast<long long*>(out.data()));
        }
      } else if (use_k2) {
#if VBT_INTERNAL_TESTS
        // Keep the grid-cap regression deterministic: when tests force a tiny
        // grid.x, missing out_idx looping would otherwise leave some outputs
        // uninitialized.
        if (vbt::cuda::reduction::get_cuda_reduction_grid_x_cap_for_tests().has_value()) {
          if (st == ScalarType::Float32) {
            launch_fill_constant<float>(out, 0.0f, stream);
          } else {
            launch_fill_constant<long long>(out, 0LL, stream);
          }
        }
#endif
        if (st == ScalarType::Float32) {
          k2_reduce_kernel<float, K2SumOp>
              <<<selected_launch.grid,
                 selected_launch.block,
                 0,
                 reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  plan_result.plan,
                  static_cast<const float*>(self.data()),
                  static_cast<float*>(out.data()));
        } else {
          k2_reduce_kernel<long long, K2SumOp>
              <<<selected_launch.grid,
                 selected_launch.block,
                 0,
                 reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  plan_result.plan,
                  static_cast<const long long*>(self.data()),
                  static_cast<long long*>(out.data()));
        }
      } else {
        run_atomic_reduce_k1<SumOp>(
            self, out, iter, out_meta, in_meta, N, k1_launch, stream);
      }
      break;
    case CudaReduceKind::Prod:
      if (use_k3) {
#if VBT_INTERNAL_TESTS
        // Keep the grid-cap regression deterministic: when tests force a tiny
        // grid.x, missing tile_base looping would otherwise leave some outputs
        // uninitialized.
        if (vbt::cuda::reduction::get_cuda_reduction_grid_x_cap_for_tests().has_value()) {
          if (st == ScalarType::Float32) {
            launch_fill_constant<float>(out, 1.0f, stream);
          } else {
            launch_fill_constant<long long>(out, 1LL, stream);
          }
        }
#endif
        if (st == ScalarType::Float32) {
          k3_tile_outputs_reduce_kernel<float, K2ProdOp>
              <<<selected_launch.grid,
                 selected_launch.block,
                 0,
                 reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  plan_result.plan,
                  static_cast<const float*>(self.data()),
                  static_cast<float*>(out.data()));
        } else {
          k3_tile_outputs_reduce_kernel<long long, K2ProdOp>
              <<<selected_launch.grid,
                 selected_launch.block,
                 0,
                 reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  plan_result.plan,
                  static_cast<const long long*>(self.data()),
                  static_cast<long long*>(out.data()));
        }
      } else if (use_k2) {
#if VBT_INTERNAL_TESTS
        // Keep the grid-cap regression deterministic: when tests force a tiny
        // grid.x, missing out_idx looping would otherwise leave some outputs
        // uninitialized.
        if (vbt::cuda::reduction::get_cuda_reduction_grid_x_cap_for_tests().has_value()) {
          if (st == ScalarType::Float32) {
            launch_fill_constant<float>(out, 1.0f, stream);
          } else {
            launch_fill_constant<long long>(out, 1LL, stream);
          }
        }
#endif
        if (st == ScalarType::Float32) {
          k2_reduce_kernel<float, K2ProdOp>
              <<<selected_launch.grid,
                 selected_launch.block,
                 0,
                 reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  plan_result.plan,
                  static_cast<const float*>(self.data()),
                  static_cast<float*>(out.data()));
        } else {
          k2_reduce_kernel<long long, K2ProdOp>
              <<<selected_launch.grid,
                 selected_launch.block,
                 0,
                 reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                  plan_result.plan,
                  static_cast<const long long*>(self.data()),
                  static_cast<long long*>(out.data()));
        }
      } else {
        run_atomic_reduce_k1<ProdOp>(
            self, out, iter, out_meta, in_meta, N, k1_launch, stream);
      }
      break;
    case CudaReduceKind::Min:
      run_atomic_reduce_k1<MinOp>(self, out, iter, out_meta, in_meta, N, k1_launch, stream);
      break;
    case CudaReduceKind::Max:
      run_atomic_reduce_k1<MaxOp>(self, out, iter, out_meta, in_meta, N, k1_launch, stream);
      break;
    case CudaReduceKind::Mean: {
      // mean is implemented as sum + div, but with a unified epilogue.
      if (use_k3) {
#if VBT_INTERNAL_TESTS
        // Keep the grid-cap regression deterministic: when tests force a tiny
        // grid.x, missing tile_base looping would otherwise leave some outputs
        // uninitialized.
        if (vbt::cuda::reduction::get_cuda_reduction_grid_x_cap_for_tests().has_value()) {
          launch_fill_constant<float>(out, 0.0f, stream);
        }
#endif
        k3_tile_outputs_reduce_kernel<float, K2SumOp>
            <<<selected_launch.grid,
               selected_launch.block,
               0,
               reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                plan_result.plan,
                static_cast<const float*>(self.data()),
                static_cast<float*>(out.data()));
      } else if (use_k2) {
#if VBT_INTERNAL_TESTS
        // Keep the grid-cap regression deterministic: when tests force a tiny
        // grid.x, missing out_idx looping would otherwise leave some outputs
        // uninitialized.
        if (vbt::cuda::reduction::get_cuda_reduction_grid_x_cap_for_tests().has_value()) {
          launch_fill_constant<float>(out, 0.0f, stream);
        }
#endif
        k2_reduce_kernel<float, K2SumOp>
            <<<selected_launch.grid,
               selected_launch.block,
               0,
               reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                plan_result.plan,
                static_cast<const float*>(self.data()),
                static_cast<float*>(out.data()));
      } else {
        run_atomic_reduce_k1<SumOp>(
            self, out, iter, out_meta, in_meta, N, k1_launch, stream);
      }

      const int64_t out_N = out.numel();
      Launch1DConfig div_launch =
          make_1d_launch(out_N, self.device().index, out_numel);
      scalar_div_kernel<float, int64_t><<<div_launch.grid, div_launch.block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<float*>(out.data()), static_cast<float>(slice_len), out_N);
      break;
    }
  }

  epilogue();
  return out;
#else
  (void)kind;
  (void)self;
  (void)dims;
  (void)keepdim;
  (void)name;
  throw std::runtime_error("CUDA not built");
#endif
}

} // namespace

#if VBT_INTERNAL_TESTS
extern "C" std::uint64_t vbt_cuda_debug_get_cub_reduce_all_sum_fastpath_calls() {
  return g_cub_reduce_all_sum_fastpath_calls.load(std::memory_order_relaxed);
}

extern "C" void vbt_cuda_debug_reset_cub_reduce_all_sum_fastpath_calls() {
  g_cub_reduce_all_sum_fastpath_calls.store(0, std::memory_order_relaxed);
}
#endif

extern "C" {

TensorImpl vbt_cuda_sum_impl(const TensorImpl& self, std::vector<int64_t> dims, bool keepdim) {
  return cuda_reduce_dispatch(CudaReduceKind::Sum, self, std::move(dims), keepdim, "vt::sum");
}

TensorImpl vbt_cuda_prod_impl(const TensorImpl& self, std::vector<int64_t> dims, bool keepdim) {
  return cuda_reduce_dispatch(CudaReduceKind::Prod, self, std::move(dims), keepdim, "vt::prod");
}

TensorImpl vbt_cuda_min_impl(const TensorImpl& self, std::vector<int64_t> dims, bool keepdim) {
  return cuda_reduce_dispatch(CudaReduceKind::Min, self, std::move(dims), keepdim, "vt::min");
}

TensorImpl vbt_cuda_max_impl(const TensorImpl& self, std::vector<int64_t> dims, bool keepdim) {
  return cuda_reduce_dispatch(CudaReduceKind::Max, self, std::move(dims), keepdim, "vt::max");
}

TensorImpl vbt_cuda_mean_impl(const TensorImpl& self, std::vector<int64_t> dims, bool keepdim) {
  return cuda_reduce_dispatch(CudaReduceKind::Mean, self, std::move(dims), keepdim, "vt::mean");
}

} // extern "C"

extern "C" void vbt_register_cuda_reduction_kernels() {
#if VBT_WITH_CUDA
  // NOTE: vt reductions have schemas with non-Tensor arguments (int[], bool). The
  // dispatcher only supports Tensor inputs, so these ops are schema-only and are
  // not registered as unboxed kernels.
  static std::once_flag once;
  std::call_once(once, []() {});
#endif
}
