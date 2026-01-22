// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/tensor_ops.h"

#include <stdexcept>
#include <vector>
#include <limits>
#include <string>

#include "vbt/core/checked_math.h"
#include "vbt/core/dtype.h"
#include "vbt/core/complex.h"
#include "vbt/core/device.h"
#include "vbt/core/error_text.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/guard.h"

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

#if VBT_WITH_CUDA
using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;

namespace {

#if VBT_WITH_CUDA
using vbt::cuda::DeviceGuard;
#endif

static inline std::vector<int64_t> make_contiguous_strides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> st(sizes.size(), 0);
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    st[idx] = acc;

    int64_t dim = sizes[idx];
    if (dim < 0) {
      throw std::invalid_argument("clone_cuda: negative size");
    }
    if (dim == 0) {
      dim = 1;
    }

    int64_t tmp = 0;
    if (!vbt::core::checked_mul_i64(acc, dim, tmp)) {
      throw std::overflow_error("clone_cuda: contiguous stride overflow");
    }
    acc = tmp;
  }
  return st;
}

struct StridedMeta {
  int64_t sizes[8];
  int64_t strides[8];
  int64_t ndim; // <= 8
};

__device__ inline int64_t compute_offset_elems(int64_t li, const StridedMeta& m) {
  if (m.ndim == 0) return 0;
  int64_t off = 0;
  for (int64_t d = m.ndim - 1; d >= 0; --d) {
    int64_t size_d = m.sizes[d] == 0 ? 1 : m.sizes[d];
    int64_t idx_d = (size_d == 1) ? 0 : (li % size_d);
    li = (size_d == 1) ? li : (li / size_d);
    off += idx_d * m.strides[d];
  }
  return off;
}

template <typename T, typename index_t>
__global__ void clone_strided_to_contig_kernel(T* out, const T* base_in, StridedMeta m, index_t N) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    int64_t off = compute_offset_elems(static_cast<int64_t>(i), m);
    out[i] = base_in[off];
  }
}

template <typename ComplexT, typename index_t>
__global__ void conj_inplace_kernel(ComplexT* out, index_t N) {
  index_t i = blockIdx.x * blockDim.x + threadIdx.x;
  index_t stride = blockDim.x * gridDim.x;
  for (; i < N; i += stride) {
    out[i].im = -out[i].im;
  }
}

static inline void launch_bounds_and_grid(int64_t N, dim3& grid, dim3& block) {
  const int threads = 256;
  block = dim3(threads);
  int64_t blocks = (N + threads - 1) / threads;
  if (blocks <= 0) blocks = 1;
  if (blocks > 65535) blocks = 65535;
  grid = dim3(static_cast<unsigned int>(blocks));
}

[[nodiscard]] static inline int64_t numel_or_throw(const TensorImpl& t,
                                                  const char* ctx) {
  const auto& sz = t.sizes();
  if (sz.empty()) {
    return 1;  // scalar
  }

  int64_t n = 1;
  for (int64_t s : sz) {
    if (s < 0) {
      throw std::invalid_argument(std::string(ctx) + ": negative size");
    }
    if (s == 0) {
      return 0;
    }
    int64_t tmp = 0;
    if (!vbt::core::checked_mul_i64(n, s, tmp)) {
      throw std::overflow_error(std::string(ctx) + ": numel overflow");
    }
    n = tmp;
  }

  return n;
}

[[nodiscard]] static inline std::size_t nbytes_or_throw(ScalarType dtype,
                                                       int64_t N,
                                                       const char* ctx) {
  if (N < 0) {
    throw std::invalid_argument(std::string(ctx) + ": negative numel");
  }

  const std::size_t item_b =
      static_cast<std::size_t>(vbt::core::itemsize(dtype));

  if (N == 0) {
    return 0;
  }
  if (item_b == 0) {
    throw std::invalid_argument(
        std::string(ctx) + ": itemsize is zero for non-empty tensor");
  }

  if (item_b > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error(std::string(ctx) + ": itemsize too large");
  }
  const int64_t item_b_i64 = static_cast<int64_t>(item_b);

  int64_t bytes_i64 = 0;
  if (!vbt::core::checked_mul_i64(N, item_b_i64, bytes_i64)) {
    throw std::overflow_error(std::string(ctx) + ": nbytes overflow");
  }
  if (bytes_i64 < 0) {
    throw std::overflow_error(std::string(ctx) + ": nbytes underflow");
  }

  const auto bytes_u64 = static_cast<std::uint64_t>(bytes_i64);
  if (bytes_u64 >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error(std::string(ctx) + ": nbytes too large");
  }

  return static_cast<std::size_t>(bytes_u64);
}

static inline TensorImpl make_cuda_dense_out_like(const TensorImpl& like,
                                                 int64_t N) {
  const std::size_t nbytes = nbytes_or_throw(like.dtype(), N, "clone_cuda");
  auto storage = vbt::cuda::new_cuda_storage(nbytes, like.device().index);
  auto sizes = like.sizes();
  auto strides = make_contiguous_strides(sizes);
  return TensorImpl(storage,
                    std::move(sizes),
                    std::move(strides),
                    /*storage_offset=*/0,
                    like.dtype(),
                    like.device());
}

#endif // VBT_WITH_CUDA

} // anonymous

namespace vbt { namespace core {

#if VBT_WITH_CUDA
TensorImpl clone_cuda(const TensorImpl& self) {
  if (self.device().type != kDLCUDA) {
    throw std::invalid_argument("clone_cuda: expected a CUDA tensor");
  }
  if (!(self.dtype() == ScalarType::Float32 ||
        self.dtype() == ScalarType::Float64 ||
        self.dtype() == ScalarType::Int64 ||
        self.dtype() == ScalarType::Complex64 ||
        self.dtype() == ScalarType::Complex128)) {
    throw std::invalid_argument(kCloneCudaDtypeAllowlistMsg);
  }

  DeviceGuard g(self.device().index);
  const int64_t N = numel_or_throw(self, "clone_cuda");
  auto out = make_cuda_dense_out_like(self, N);
  if (N == 0) return out;

  // Pre-clear any sticky error
  (void)cudaGetLastError();

  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(self.device().index));

  const std::size_t nbytes = nbytes_or_throw(self.dtype(), N, "clone_cuda");
  if (self.is_contiguous()) {
    // memcpy fast path for truly contiguous inputs
    cudaError_t st = cudaMemcpyAsync(out.data(), self.data(), nbytes, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream.handle()));
    if (st != cudaSuccess) {
      const char* msg = cudaGetErrorString(st);
      std::string m = std::string(kCloneKernelLaunchFailedPrefix) + (msg ? msg : "");
      throw std::runtime_error(m);
    }
  } else {
    // General strided → contiguous kernel
    bool use32 = (N <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
    dim3 grid, block; launch_bounds_and_grid(N, grid, block);
    StridedMeta m{};
    const auto& sz = self.sizes();
    const auto& st = self.strides();
    if (sz.size() > 8) {
      throw std::invalid_argument("clone_cuda: tensors with ndim > 8 not supported in P2");
    }
    m.ndim = static_cast<int64_t>(sz.size());
    for (std::size_t i = 0; i < sz.size(); ++i) { m.sizes[i] = sz[i]; m.strides[i] = st[i]; }

    if (self.dtype() == ScalarType::Float32) {
      if (use32) clone_strided_to_contig_kernel<float,int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<float*>(out.data()), static_cast<const float*>(self.data()), m, static_cast<int32_t>(N));
      else clone_strided_to_contig_kernel<float,int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<float*>(out.data()), static_cast<const float*>(self.data()), m, static_cast<int64_t>(N));
    } else if (self.dtype() == ScalarType::Float64) {
      if (use32) clone_strided_to_contig_kernel<double,int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<double*>(out.data()), static_cast<const double*>(self.data()), m, static_cast<int32_t>(N));
      else clone_strided_to_contig_kernel<double,int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<double*>(out.data()), static_cast<const double*>(self.data()), m, static_cast<int64_t>(N));
    } else if (self.dtype() == ScalarType::Int64) {
      if (use32) clone_strided_to_contig_kernel<long long,int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<long long*>(out.data()), static_cast<const long long*>(self.data()), m, static_cast<int32_t>(N));
      else clone_strided_to_contig_kernel<long long,int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<long long*>(out.data()), static_cast<const long long*>(self.data()), m, static_cast<int64_t>(N));
    } else if (self.dtype() == ScalarType::Complex64) {
      if (use32) clone_strided_to_contig_kernel<vbt::core::Complex64,int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<vbt::core::Complex64*>(out.data()), static_cast<const vbt::core::Complex64*>(self.data()), m, static_cast<int32_t>(N));
      else clone_strided_to_contig_kernel<vbt::core::Complex64,int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<vbt::core::Complex64*>(out.data()), static_cast<const vbt::core::Complex64*>(self.data()), m, static_cast<int64_t>(N));
    } else if (self.dtype() == ScalarType::Complex128) {
      if (use32) clone_strided_to_contig_kernel<vbt::core::Complex128,int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<vbt::core::Complex128*>(out.data()), static_cast<const vbt::core::Complex128*>(self.data()), m, static_cast<int32_t>(N));
      else clone_strided_to_contig_kernel<vbt::core::Complex128,int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
        static_cast<vbt::core::Complex128*>(out.data()), static_cast<const vbt::core::Complex128*>(self.data()), m, static_cast<int64_t>(N));
    }
  }

  if (self.is_conj() &&
      (self.dtype() == ScalarType::Complex64 || self.dtype() == ScalarType::Complex128)) {
    const bool use32 = (N <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
    dim3 grid, block;
    launch_bounds_and_grid(N, grid, block);

    if (self.dtype() == ScalarType::Complex64) {
      if (use32) {
        conj_inplace_kernel<vbt::core::Complex64, int32_t><<<
            grid,
            block,
            0,
            reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<vbt::core::Complex64*>(out.data()),
            static_cast<int32_t>(N));
      } else {
        conj_inplace_kernel<vbt::core::Complex64, int64_t><<<
            grid,
            block,
            0,
            reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<vbt::core::Complex64*>(out.data()),
            static_cast<int64_t>(N));
      }
    } else {  // Complex128
      if (use32) {
        conj_inplace_kernel<vbt::core::Complex128, int32_t><<<
            grid,
            block,
            0,
            reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<vbt::core::Complex128*>(out.data()),
            static_cast<int32_t>(N));
      } else {
        conj_inplace_kernel<vbt::core::Complex128, int64_t><<<
            grid,
            block,
            0,
            reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<vbt::core::Complex128*>(out.data()),
            static_cast<int64_t>(N));
      }
    }
  }

  // Surface any pending launch failure from earlier kernels on this stream
  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    std::string m = std::string(kCloneKernelLaunchFailedPrefix) + (msg ? msg : "");
    throw std::runtime_error(m);
  }

  vbt::cuda::record_stream(out.storage(), stream);
  vbt::cuda::record_stream(self.storage(), stream);

  return out;
}
#endif

}} // namespace vbt::core
