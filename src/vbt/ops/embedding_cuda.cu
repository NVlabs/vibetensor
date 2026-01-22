// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "vbt/cuda/stream.h"

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1,
              "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

#if VBT_WITH_CUDA

namespace vbt {
namespace ops {

namespace {

static_assert(sizeof(std::int64_t) == sizeof(unsigned long long),
              "expected 64-bit unsigned long long");

__global__ void embedding_bounds_check_i64_kernel(
    const std::int64_t* __restrict__ idx,
    std::int64_t N,
    std::int64_t V,
    int* __restrict__ error_flag) {
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < N;
       i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t v = idx[i];
    if (v < 0 || v >= V) {
      atomicExch(error_flag, 1);
    }
  }
}

__global__ void embedding_gather_f32_kernel(
    const float* __restrict__ w,
    std::int64_t ws0,
    std::int64_t ws1,
    const std::int64_t* __restrict__ idx,
    float* __restrict__ out,
    std::int64_t N,
    std::int64_t D) {
  const std::int64_t total = N * D;
  for (std::int64_t li = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       li < total;
       li += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t i = li / D;
    const std::int64_t j = li - i * D;
    const std::int64_t row = idx[i];
    out[li] = w[row * ws0 + j * ws1];
  }
}

__global__ void embedding_backward_accum_f32_kernel(
    const std::int64_t* __restrict__ idx,
    const float* __restrict__ grad,
    float* __restrict__ grad_weight,
    std::int64_t N,
    std::int64_t D,
    std::int64_t padding_idx) {
  const std::int64_t total = N * D;
  for (std::int64_t li = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       li < total;
       li += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t i = li / D;
    const std::int64_t j = li - i * D;
    const std::int64_t row = idx[i];
    if (padding_idx >= 0 && row == padding_idx) {
      continue;
    }
    // NOTE: atomicAdd accumulation is potentially nondeterministic when
    // indices contain duplicates (the update order is undefined).
    atomicAdd(&grad_weight[row * D + j], grad[li]);
  }
}

__global__ void embedding_count_by_freq_i64_kernel(
    const std::int64_t* __restrict__ idx,
    std::int64_t N,
    std::int64_t padding_idx,
    unsigned long long* __restrict__ counts) {
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < N;
       i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t row = idx[i];
    if (padding_idx >= 0 && row == padding_idx) {
      continue;
    }
    atomicAdd(&counts[row], 1ULL);
  }
}

__global__ void embedding_backward_accum_scaled_f32_kernel(
    const std::int64_t* __restrict__ idx,
    const float* __restrict__ grad,
    const unsigned long long* __restrict__ counts,
    float* __restrict__ grad_weight,
    std::int64_t N,
    std::int64_t D,
    std::int64_t padding_idx) {
  const std::int64_t total = N * D;
  for (std::int64_t li = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       li < total;
       li += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t i = li / D;
    const std::int64_t j = li - i * D;
    const std::int64_t row = idx[i];
    if (padding_idx >= 0 && row == padding_idx) {
      continue;
    }
    const unsigned long long c = counts[row];
    if (c == 0ULL) {
      // Defensive: should never happen after a correct count pass.
      continue;
    }
    const float scale = 1.0f / static_cast<float>(c);
    atomicAdd(&grad_weight[row * D + j], grad[li] * scale);
  }
}

__global__ void embedding_pack_keys_vals_u64_i64_kernel(
    const std::int64_t* __restrict__ idx,
    std::uint64_t* __restrict__ keys,
    long long* __restrict__ vals,
    int n) {
  for (int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n;
       i += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x)) {
    keys[i] = static_cast<std::uint64_t>(idx[i]);
    vals[i] = 1LL;
  }
}

__global__ void embedding_renorm_f32_kernel(
    const std::uint64_t* __restrict__ unique_keys,
    const int* __restrict__ d_num_unique,
    int n_max,
    float* __restrict__ w,
    std::int64_t ws0,
    std::int64_t ws1,
    std::int64_t D,
    float max_norm,
    float p,
    int* __restrict__ d_mutated) {
  constexpr float kEps = 1e-7f;
  const int K = d_num_unique ? d_num_unique[0] : 0;

  for (int k = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       k < n_max;
       k += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x)) {
    if (k >= K) {
      continue;
    }

    const std::int64_t row = static_cast<std::int64_t>(unique_keys[k]);
    float* row_ptr = w + row * ws0;

    // Compute p-norm (match CPU branching).
    float norm = 0.0f;
    if (isinf(p)) {
      float m = 0.0f;
      for (std::int64_t j = 0; j < D; ++j) {
        const float v = row_ptr[j * ws1];
        m = fmaxf(m, fabsf(v));
      }
      norm = m;
    } else if (p == 1.0f) {
      float acc = 0.0f;
      for (std::int64_t j = 0; j < D; ++j) {
        const float v = row_ptr[j * ws1];
        acc += fabsf(v);
      }
      norm = acc;
    } else if (p == 2.0f) {
      float acc = 0.0f;
      for (std::int64_t j = 0; j < D; ++j) {
        const float v = row_ptr[j * ws1];
        acc += v * v;
      }
      norm = sqrtf(acc);
    } else {
      float acc = 0.0f;
      for (std::int64_t j = 0; j < D; ++j) {
        const float v = row_ptr[j * ws1];
        acc += powf(fabsf(v), p);
      }
      norm = powf(acc, 1.0f / p);
    }

    if (norm > max_norm) {
      const float scale = max_norm / (norm + kEps);
      for (std::int64_t j = 0; j < D; ++j) {
        row_ptr[j * ws1] *= scale;
      }
      if (d_mutated) {
        atomicExch(d_mutated, 1);
      }
    }
  }
}

static inline void launch_1d(int64_t N, dim3& grid, dim3& block) {
  const int threads = 256;
  block = dim3(threads);
  int64_t blocks = (N + threads - 1) / threads;
  if (blocks <= 0) blocks = 1;
  if (blocks > 65535) blocks = 65535;
  grid = dim3(static_cast<unsigned int>(blocks));
}

}  // namespace

bool embedding_cuda_bounds_check_i64(
    const std::int64_t* idx,
    std::int64_t N,
    std::int64_t V,
    vbt::cuda::Stream stream,
    const char* op_name) {
  if (N <= 0) {
    return false;
  }

  int* d_error = nullptr;
  cudaError_t st_alloc =
      cudaMalloc(reinterpret_cast<void**>(&d_error), sizeof(int));
  if (st_alloc != cudaSuccess) {
    const char* msg = cudaGetErrorString(st_alloc);
    std::string m = std::string(op_name) +
                    ": CUDA embedding bounds check alloc failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  struct DeviceIntGuard {
    int* ptr;
    ~DeviceIntGuard() {
      if (ptr) {
        (void)cudaFree(ptr);
      }
    }
  } guard{d_error};

  cudaError_t st = cudaMemsetAsync(d_error, 0, sizeof(int),
                                  reinterpret_cast<cudaStream_t>(stream.handle()));
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    std::string m = std::string(op_name) +
                    ": CUDA embedding bounds check memset failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  dim3 grid, block;
  launch_1d(N, grid, block);

  (void)cudaGetLastError();
  embedding_bounds_check_i64_kernel<<<grid, block, 0,
                                     reinterpret_cast<cudaStream_t>(stream.handle())>>>(
      idx, N, V, d_error);

  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    std::string m = std::string(op_name) +
                    ": CUDA embedding bounds check launch failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  int h_error = 0;
  st = cudaMemcpyAsync(&h_error, d_error, sizeof(int), cudaMemcpyDeviceToHost,
                       reinterpret_cast<cudaStream_t>(stream.handle()));
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    std::string m = std::string(op_name) +
                    ": CUDA embedding bounds check D2H failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  st = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.handle()));
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    std::string m = std::string(op_name) + ": CUDA embedding bounds check sync failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  return h_error != 0;
}

void embedding_cuda_gather_f32(
    const float* w,
    std::int64_t ws0,
    std::int64_t ws1,
    const std::int64_t* idx,
    float* out,
    std::int64_t N,
    std::int64_t D,
    vbt::cuda::Stream stream,
    const char* op_name) {
  if (N <= 0 || D <= 0) {
    return;
  }

  // Pre-clear any sticky error.
  (void)cudaGetLastError();

  const std::int64_t total = N * D;
  dim3 grid, block;
  launch_1d(total, grid, block);

  embedding_gather_f32_kernel<<<grid, block, 0,
                               reinterpret_cast<cudaStream_t>(stream.handle())>>>(
      w, ws0, ws1, idx, out, N, D);

  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    std::string m = std::string(op_name) + ": CUDA embedding gather launch failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }
}

void embedding_cuda_backward_accum_f32(
    const std::int64_t* idx,
    const float* grad,
    float* grad_weight,
    std::int64_t N,
    std::int64_t D,
    std::int64_t padding_idx,
    vbt::cuda::Stream stream,
    const char* op_name) {
  if (N <= 0 || D <= 0) {
    return;
  }

  // Guard N*D overflow.
  if (D > 0 && N > std::numeric_limits<std::int64_t>::max() / D) {
    throw std::overflow_error(std::string(op_name) + ": CUDA embedding backward size overflow");
  }

  // Pre-clear any sticky error.
  (void)cudaGetLastError();

  const std::int64_t total = N * D;
  dim3 grid, block;
  launch_1d(total, grid, block);

  embedding_backward_accum_f32_kernel<<<grid, block, 0,
                                       reinterpret_cast<cudaStream_t>(stream.handle())>>>(
      idx, grad, grad_weight, N, D, padding_idx);

  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    std::string m = std::string(op_name) + ": CUDA embedding backward launch failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }
}

void embedding_cuda_count_by_freq_i64(
    const std::int64_t* idx,
    std::int64_t N,
    std::int64_t padding_idx,
    std::int64_t* counts,
    vbt::cuda::Stream stream,
    const char* op_name) {
  if (N <= 0) {
    return;
  }

  // Pre-clear any sticky error.
  (void)cudaGetLastError();

  dim3 grid, block;
  launch_1d(N, grid, block);

  embedding_count_by_freq_i64_kernel<<<grid, block, 0,
                                      reinterpret_cast<cudaStream_t>(stream.handle())>>>(
      idx, N, padding_idx,
      reinterpret_cast<unsigned long long*>(counts));

  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    std::string m = std::string(op_name) + ": CUDA embedding count_by_freq launch failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }
}

void embedding_cuda_backward_accum_scaled_f32(
    const std::int64_t* idx,
    const float* grad,
    const std::int64_t* counts,
    float* grad_weight,
    std::int64_t N,
    std::int64_t D,
    std::int64_t padding_idx,
    vbt::cuda::Stream stream,
    const char* op_name) {
  if (N <= 0 || D <= 0) {
    return;
  }

  // Guard N*D overflow.
  if (D > 0 && N > std::numeric_limits<std::int64_t>::max() / D) {
    throw std::overflow_error(std::string(op_name) + ": CUDA embedding backward size overflow");
  }

  // Pre-clear any sticky error.
  (void)cudaGetLastError();

  const std::int64_t total = N * D;
  dim3 grid, block;
  launch_1d(total, grid, block);

  embedding_backward_accum_scaled_f32_kernel<<<grid, block, 0,
                                              reinterpret_cast<cudaStream_t>(stream.handle())>>>(
      idx, grad,
      reinterpret_cast<const unsigned long long*>(counts),
      grad_weight, N, D, padding_idx);

  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    std::string m = std::string(op_name) + ": CUDA embedding backward_scaled launch failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }
}

void embedding_cuda_pack_keys_vals_u64_i64(
    const std::int64_t* idx,
    std::uint64_t* keys,
    long long* vals,
    int n,
    vbt::cuda::Stream stream,
    const char* op_name) {
  if (n <= 0) {
    return;
  }

  // Pre-clear any sticky error.
  (void)cudaGetLastError();

  dim3 grid, block;
  launch_1d(static_cast<int64_t>(n), grid, block);

  embedding_pack_keys_vals_u64_i64_kernel<<<grid, block, 0,
                                           reinterpret_cast<cudaStream_t>(stream.handle())>>>(
      idx, keys, vals, n);

  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    std::string m = std::string(op_name) + ": CUDA embedding renorm pack launch failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }
}

void embedding_cuda_renorm_f32(
    const std::uint64_t* unique_keys,
    const int* d_num_unique,
    int n_max,
    float* w,
    std::int64_t ws0,
    std::int64_t ws1,
    std::int64_t D,
    float max_norm,
    float p,
    int* d_mutated,
    vbt::cuda::Stream stream,
    const char* op_name) {
  if (n_max <= 0 || D <= 0) {
    return;
  }

  // Pre-clear any sticky error.
  (void)cudaGetLastError();

  dim3 grid, block;
  launch_1d(static_cast<int64_t>(n_max), grid, block);

  embedding_renorm_f32_kernel<<<grid, block, 0,
                               reinterpret_cast<cudaStream_t>(stream.handle())>>>(
      unique_keys, d_num_unique, n_max, w, ws0, ws1, D, max_norm, p, d_mutated);

  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    std::string m = std::string(op_name) + ": CUDA embedding renorm launch failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }
}

}  // namespace ops
}  // namespace vbt

#endif  // VBT_WITH_CUDA
