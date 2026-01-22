// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/rng/kernels_cuda.h"

#include <stdexcept>
#include <string>
#include <cmath>

#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/core/write_guard.h"
#include "vbt/rng/philox_util.h"
#include "vbt/rng/graph_capture.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/graphs.h"

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace vbt {
namespace rng {
namespace cuda {

#if VBT_WITH_CUDA

using vbt::core::TensorImpl;
using vbt::core::ScalarType;

namespace {

namespace gc = vbt::rng::graph_capture;
using vbt::cuda::CaptureStatus;
using vbt::cuda::DeviceIndex;

static bool rng_stream_is_capturing_for_tensor(
    const TensorImpl& t,
    vbt::rng::CudaGenerator& gen) {
  using vbt::rng::graph_capture::is_generator_capture_active;

  if (!is_generator_capture_active(gen)) {
    return false;
  }

  DeviceIndex dev_idx = static_cast<DeviceIndex>(t.device().index);
  CaptureStatus status = vbt::cuda::currentStreamCaptureStatus(dev_idx);
  return status == CaptureStatus::Active;
}

static vbt::rng::PhiloxState reserve_philox_for_cuda_op(
    TensorImpl&                 t,
    vbt::rng::CudaGenerator&    gen,
    std::uint64_t               total_blocks,
    std::uint32_t               outputs_per_block,
    gc::RngOpTag                op_tag) {
  const bool stream_is_capturing = rng_stream_is_capturing_for_tensor(t, gen);
  return gc::reserve_blocks_for_graph_aware_cuda_op(
      gen,
      total_blocks,
      outputs_per_block,
      op_tag,
      stream_is_capturing);
}

static constexpr float kTwoPi = 6.28318530717958647692f; // 2*pi in float32

struct DeviceIndexer {
  // Fixed-rank (<=8) indexer to keep kernel arguments small; higher ranks
  // are rejected at the host boundary with a clear error message.
  std::int64_t sizes[8];
  std::int64_t strides[8];
  std::int64_t ndim; // <= 8
  std::uint8_t* base;
};

struct KernelArgs {
  std::uint64_t seed;
  std::uint64_t base_offset;
  std::uint64_t total_blocks;
  std::uint32_t outputs_per_block;
  std::uint64_t numel;
  float low;
  float high;
};

__device__ inline float* ptr_for_linear_e(DeviceIndexer meta, std::uint64_t e) {
  std::uint64_t rem = e;
  std::uint8_t* base = meta.base;
  const std::int64_t ndim = meta.ndim;
  for (std::int64_t rev = 0; rev < ndim; ++rev) {
    const std::int64_t i = ndim - 1 - rev;
    const std::uint64_t dim = static_cast<std::uint64_t>(meta.sizes[i] >= 0 ? meta.sizes[i] : 0);
    if (dim == 0ull) {
      return reinterpret_cast<float*>(base);
    }
    const std::uint64_t idx = (dim == 0ull) ? 0ull : (rem % dim);
    rem = (dim == 0ull) ? 0ull : (rem / dim);
    const std::int64_t st = meta.strides[i];
    const std::int64_t step_bytes = static_cast<std::int64_t>(sizeof(float)) * st;
    base += step_bytes * static_cast<std::int64_t>(idx);
  }
  return reinterpret_cast<float*>(base);
}

__device__ inline long long* ptr_for_linear_e_i64(DeviceIndexer meta, std::uint64_t e) {
  std::uint64_t rem = e;
  std::uint8_t* base = meta.base;
  const std::int64_t ndim = meta.ndim;
  for (std::int64_t rev = 0; rev < ndim; ++rev) {
    const std::int64_t i = ndim - 1 - rev;
    const std::uint64_t dim = static_cast<std::uint64_t>(meta.sizes[i] >= 0 ? meta.sizes[i] : 0);
    if (dim == 0ull) {
      return reinterpret_cast<long long*>(base);
    }
    const std::uint64_t idx = (dim == 0ull) ? 0ull : (rem % dim);
    rem = (dim == 0ull) ? 0ull : (rem / dim);
    const std::int64_t st = meta.strides[i];
    const std::int64_t step_bytes = static_cast<std::int64_t>(sizeof(long long)) * st;
    base += step_bytes * static_cast<std::int64_t>(idx);
  }
  return reinterpret_cast<long long*>(base);
}

__global__ void uniform_f32_kernel(float* out,
                                   DeviceIndexer meta,
                                   bool is_contig,
                                   KernelArgs args,
                                   std::uint32_t blocks_per_thread) {
  const std::uint64_t start   = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                                static_cast<std::uint64_t>(threadIdx.x);
  const std::uint64_t tstride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

  std::uint32_t key[2];
  vbt::rng::seed_to_key(args.seed, key);

  const std::uint64_t total_blocks = args.total_blocks;
  const std::uint64_t numel = args.numel;
  const float low = args.low;
  const float high = args.high;

  if (blocks_per_thread == 0u) {
    return;
  }

  for (std::uint64_t rb0 = start; rb0 < total_blocks; rb0 += tstride * static_cast<std::uint64_t>(blocks_per_thread)) {
    for (std::uint32_t k = 0; k < blocks_per_thread; ++k) {
      const std::uint64_t rb = rb0 + static_cast<std::uint64_t>(k) * tstride;
      if (rb >= total_blocks) {
        break;
      }
      const std::uint64_t B = args.base_offset + rb;
      std::uint32_t ctr[4];
      vbt::rng::block_to_counter(B, ctr);
      std::uint32_t lanes[4];
      vbt::rng::philox10(ctr, key, lanes);
      const std::uint64_t base_e = rb * 4ull;
      for (std::uint32_t lane = 0; lane < 4u; ++lane) {
        const std::uint64_t e = base_e + static_cast<std::uint64_t>(lane);
        if (e >= numel) {
          break;
        }
        const float U = vbt::rng::u32_to_uniform_f32(lanes[lane]);
        const float scale = high - low;
        const float val = low + scale * U;
        if (is_contig) {
          out[e] = val;
        } else {
          float* pf = ptr_for_linear_e(meta, e);
          *pf = val;
        }
      }
    }
  }
}

__global__ void normal_f32_kernel(float* out,
                                  DeviceIndexer meta,
                                  bool is_contig,
                                  std::uint64_t seed,
                                  std::uint64_t base_offset,
                                  std::uint64_t total_blocks,
                                  std::uint64_t numel,
                                  float mean,
                                  float std,
                                  std::uint32_t blocks_per_thread) {
  const std::uint64_t start   = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                                static_cast<std::uint64_t>(threadIdx.x);
  const std::uint64_t tstride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

  std::uint32_t key[2];
  vbt::rng::seed_to_key(seed, key);

  if (blocks_per_thread == 0u) return;

  for (std::uint64_t rb0 = start; rb0 < total_blocks; rb0 += tstride * static_cast<std::uint64_t>(blocks_per_thread)) {
    for (std::uint32_t k = 0; k < blocks_per_thread; ++k) {
      const std::uint64_t rb = rb0 + static_cast<std::uint64_t>(k) * tstride;
      if (rb >= total_blocks) break;
      const std::uint64_t B = base_offset + rb;
      std::uint32_t ctr[4]; vbt::rng::block_to_counter(B, ctr);
      std::uint32_t lanes[4]; vbt::rng::philox10(ctr, key, lanes);
      const std::uint64_t base_e = rb * 4ull;
      // Pair 0: lanes[0], lanes[1]
      if (base_e < numel) {
        float U0 = vbt::rng::u01_open_open(lanes[0]);
        float U1 = vbt::rng::u01_closed_open(lanes[1]);
        float R = sqrtf(-2.0f * logf(U0));
        float Theta = kTwoPi * U1;
        float Z0 = R * cosf(Theta);
        float Z1 = R * sinf(Theta);
        if (is_contig) {
          out[base_e] = mean + std * Z0;
          if (base_e + 1ull < numel) out[base_e + 1ull] = mean + std * Z1;
        } else {
          float* p0 = ptr_for_linear_e(meta, base_e);
          *p0 = mean + std * Z0;
          if (base_e + 1ull < numel) {
            float* p1 = ptr_for_linear_e(meta, base_e + 1ull);
            *p1 = mean + std * Z1;
          }
        }
      }
      // Pair 1: lanes[2], lanes[3]
      if (base_e + 2ull < numel) {
        float U0 = vbt::rng::u01_open_open(lanes[2]);
        float U1 = vbt::rng::u01_closed_open(lanes[3]);
        float R = sqrtf(-2.0f * logf(U0));
        float Theta = kTwoPi * U1;
        float Z0 = R * cosf(Theta);
        float Z1 = R * sinf(Theta);
        if (is_contig) {
          out[base_e + 2ull] = mean + std * Z0;
          if (base_e + 3ull < numel) out[base_e + 3ull] = mean + std * Z1;
        } else {
          float* p2 = ptr_for_linear_e(meta, base_e + 2ull);
          *p2 = mean + std * Z0;
          if (base_e + 3ull < numel) {
            float* p3 = ptr_for_linear_e(meta, base_e + 3ull);
            *p3 = mean + std * Z1;
          }
        }
      }
    }
  }
}

__global__ void bernoulli_f32_kernel(float* out,
                                     DeviceIndexer meta,
                                     bool is_contig,
                                     std::uint64_t seed,
                                     std::uint64_t base_offset,
                                     std::uint64_t total_blocks,
                                     std::uint64_t numel,
                                     float p,
                                     std::uint32_t blocks_per_thread) {
  const std::uint64_t start   = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                                static_cast<std::uint64_t>(threadIdx.x);
  const std::uint64_t tstride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  std::uint32_t key[2]; vbt::rng::seed_to_key(seed, key);
  if (blocks_per_thread == 0u) return;
  for (std::uint64_t rb0 = start; rb0 < total_blocks; rb0 += tstride * static_cast<std::uint64_t>(blocks_per_thread)) {
    for (std::uint32_t k = 0; k < blocks_per_thread; ++k) {
      const std::uint64_t rb = rb0 + static_cast<std::uint64_t>(k) * tstride;
      if (rb >= total_blocks) break;
      const std::uint64_t B = base_offset + rb;
      std::uint32_t ctr[4]; vbt::rng::block_to_counter(B, ctr);
      std::uint32_t lanes[4]; vbt::rng::philox10(ctr, key, lanes);
      const std::uint64_t base_e = rb * 4ull;
      for (std::uint32_t lane = 0; lane < 4u; ++lane) {
        const std::uint64_t e = base_e + static_cast<std::uint64_t>(lane);
        if (e >= numel) break;
        float U = vbt::rng::u32_to_uniform_f32(lanes[lane]);
        float val = (U < p) ? 1.0f : 0.0f;
        if (is_contig) out[e] = val; else *ptr_for_linear_e(meta, e) = val;
      }
    }
  }
}

__device__ inline void mul_64x64_128_dev(std::uint64_t a, std::uint64_t b, std::uint64_t& lo, std::uint64_t& hi) {
  lo = a * b;
  hi = __umul64hi(a, b);
}

__global__ void randint_i64_kernel(long long* out,
                                   DeviceIndexer meta,
                                   bool is_contig,
                                   std::uint64_t seed,
                                   std::uint64_t base_offset,
                                   std::uint64_t numel,
                                   long long low,
                                   std::uint64_t n) {
  const std::uint64_t start   = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                                static_cast<std::uint64_t>(threadIdx.x);
  const std::uint64_t tstride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  std::uint32_t key[2]; vbt::rng::seed_to_key(seed, key);
  const std::uint64_t threshold = (static_cast<std::uint64_t>(0) - n) % n;
  for (std::uint64_t e = start; e < numel; e += tstride) {
    const std::uint64_t lane = (e & 1ull);
    const std::uint64_t block_rel = (e >> 1);
    const std::uint64_t B = base_offset + block_rel;
    std::uint32_t attempt = 0u;
    while (true) {
      std::uint32_t ctr[4]; vbt::rng::block_to_counter(B, ctr); ctr[2] = attempt;
      std::uint32_t lanes[4]; vbt::rng::philox10(ctr, key, lanes);
      std::uint64_t R = (lane == 0ull)
                        ? vbt::rng::pack_u64(lanes[0], lanes[1])
                        : vbt::rng::pack_u64(lanes[2], lanes[3]);
      std::uint64_t lo128, hi128; mul_64x64_128_dev(R, n, lo128, hi128);
      if (lo128 < threshold) { ++attempt; continue; }
      // Write result
      long long val = static_cast<long long>(low) + static_cast<long long>(hi128);
      if (is_contig) {
        out[e] = val;
      } else {
        long long* pd = ptr_for_linear_e_i64(meta, e);
        *pd = val;
      }
      break;
    }
  }
}

static inline void cudaCheckLast(const char* what_prefix) {
  cudaError_t st = cudaGetLastError();
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    std::string m = std::string(what_prefix) + (msg ? msg : "");
    throw std::runtime_error(m);
  }
}

} // anonymous namespace

void uniform_(TensorImpl& t, float low, float high, vbt::rng::CudaGenerator& gen) {
  vbt::core::check_writable(t);

  if (t.dtype() != ScalarType::Float32) {
    throw std::runtime_error("uniform_: expected dtype=float32");
  }
  if (!std::isfinite(low) || !std::isfinite(high)) {
    throw std::invalid_argument("uniform_: low and high must be finite");
  }
  if (!(low <= high)) {
    throw std::invalid_argument("uniform_: low must be <= high");
  }

  if (t.device().type != kDLCUDA) {
    throw std::runtime_error("uniform_cuda: expected CUDA tensor");
  }

  const std::int64_t n64 = t.numel();
  if (n64 <= 0) {
    return;
  }
  const std::uint64_t N = static_cast<std::uint64_t>(n64);
  const std::uint64_t total_blocks = vbt::rng::ceil_div_u64(N, 4ull);

  constexpr std::uint32_t kOutputsPerBlock = 4u;
  PhiloxState st = reserve_philox_for_cuda_op(
      t,
      gen,
      total_blocks,
      kOutputsPerBlock,
      gc::RngOpTag::Uniform);

  ExecutionPolicy pol = calc_execution_policy(n64, 4u, 4u);
  if (pol.grid_x == 0u || pol.block_x == 0u || pol.blocks_per_thread == 0u) {
    // Fallback to something sane
    pol.grid_x = 1u;
    pol.block_x = 256u;
    pol.blocks_per_thread = 1u;
  }

  vbt::cuda::DeviceGuard guard(static_cast<vbt::cuda::DeviceIndex>(t.device().index));
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(t.device().index));

  bool is_contig = t.is_contiguous();
  DeviceIndexer meta{};
  meta.base = reinterpret_cast<std::uint8_t*>(t.data());
  if (!is_contig) {
    const auto& sizes = t.sizes();
    const auto& strides = t.strides();
    if (sizes.size() > 8) {
      throw std::invalid_argument("uniform_: rank>8 not supported on CUDA (DeviceIndexer supports up to 8 dims)");
    }
    meta.ndim = static_cast<std::int64_t>(sizes.size());
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      meta.sizes[i] = sizes[i];
      meta.strides[i] = strides[i];
    }
  } else {
    meta.ndim = 0;
  }

  KernelArgs args{};
  args.seed = st.seed;
  args.base_offset = st.offset;
  args.total_blocks = total_blocks;
  args.outputs_per_block = 4u;
  args.numel = N;
  args.low = low;
  args.high = high;

  // Clear any sticky error before launch, then launch and check.
  (void)cudaGetLastError();

  dim3 grid(pol.grid_x, 1u, 1u);
  dim3 block(pol.block_x, 1u, 1u);

  uniform_f32_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
      reinterpret_cast<float*>(t.data()),
      meta,
      is_contig,
      args,
      pol.blocks_per_thread);

  cudaCheckLast("uniform_cuda: kernel launch failed: ");
}

void normal_(TensorImpl& t, float mean, float std, vbt::rng::CudaGenerator& gen) {
  vbt::core::check_writable(t);
  if (t.dtype() != ScalarType::Float32) {
    throw std::runtime_error("expected floating dtype for normal_");
  }
  if (t.device().type != kDLCUDA) {
    throw std::runtime_error("normal_cuda: expected CUDA tensor");
  }
  const std::int64_t n64 = t.numel();
  if (n64 <= 0) return;
  const std::uint64_t N = static_cast<std::uint64_t>(n64);
  const std::uint64_t total_blocks = vbt::rng::ceil_div_u64(N, 4ull);

  constexpr std::uint32_t kOutputsPerBlock = 4u;
  PhiloxState st = reserve_philox_for_cuda_op(
      t,
      gen,
      total_blocks,
      kOutputsPerBlock,
      gc::RngOpTag::Normal);

  ExecutionPolicy pol = calc_execution_policy(n64, 4u, 4u);
  if (pol.grid_x == 0u || pol.block_x == 0u || pol.blocks_per_thread == 0u) {
    pol.grid_x = 1u; pol.block_x = 256u; pol.blocks_per_thread = 1u;
  }

  vbt::cuda::DeviceGuard guard(static_cast<vbt::cuda::DeviceIndex>(t.device().index));
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(t.device().index));

  bool is_contig = t.is_contiguous();
  DeviceIndexer meta{}; meta.base = reinterpret_cast<std::uint8_t*>(t.data());
  if (!is_contig) {
    const auto& sizes = t.sizes(); const auto& strides = t.strides();
    if (sizes.size() > 8) throw std::invalid_argument("normal_: rank>8 not supported on CUDA (DeviceIndexer supports up to 8 dims)");
    meta.ndim = static_cast<std::int64_t>(sizes.size());
    for (std::size_t i = 0; i < sizes.size(); ++i) { meta.sizes[i] = sizes[i]; meta.strides[i] = strides[i]; }
  } else { meta.ndim = 0; }

  (void)cudaGetLastError();
  dim3 grid(pol.grid_x, 1u, 1u); dim3 block(pol.block_x, 1u, 1u);
  normal_f32_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
      reinterpret_cast<float*>(t.data()), meta, is_contig,
      st.seed, st.offset, total_blocks, N, mean, std, pol.blocks_per_thread);
  cudaCheckLast("normal_cuda: kernel launch failed: ");
}

void bernoulli_(TensorImpl& t, float p, vbt::rng::CudaGenerator& gen) {
  vbt::core::check_writable(t);
  if (t.dtype() != ScalarType::Float32) {
    throw std::runtime_error("expected floating dtype for bernoulli_");
  }
  if (!(p >= 0.0f && p <= 1.0f)) {
    throw std::runtime_error("bernoulli_: p must be in [0, 1]");
  }
  if (t.device().type != kDLCUDA) {
    throw std::runtime_error("bernoulli_cuda: expected CUDA tensor");
  }
  const std::int64_t n64 = t.numel(); if (n64 <= 0) return;
  const std::uint64_t N = static_cast<std::uint64_t>(n64);
  const std::uint64_t total_blocks = vbt::rng::ceil_div_u64(N, 4ull);

  constexpr std::uint32_t kOutputsPerBlock = 4u;
  PhiloxState st = reserve_philox_for_cuda_op(
      t,
      gen,
      total_blocks,
      kOutputsPerBlock,
      gc::RngOpTag::Bernoulli);

  ExecutionPolicy pol = calc_execution_policy(n64, 4u, 4u);
  if (pol.grid_x == 0u || pol.block_x == 0u || pol.blocks_per_thread == 0u) { pol.grid_x = 1u; pol.block_x = 256u; pol.blocks_per_thread = 1u; }

  vbt::cuda::DeviceGuard guard(static_cast<vbt::cuda::DeviceIndex>(t.device().index));
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(t.device().index));

  bool is_contig = t.is_contiguous();
  DeviceIndexer meta{}; meta.base = reinterpret_cast<std::uint8_t*>(t.data());
  if (!is_contig) {
    const auto& sizes = t.sizes(); const auto& strides = t.strides();
    if (sizes.size() > 8) throw std::invalid_argument("bernoulli_: rank>8 not supported on CUDA (DeviceIndexer supports up to 8 dims)");
    meta.ndim = static_cast<std::int64_t>(sizes.size());
    for (std::size_t i = 0; i < sizes.size(); ++i) { meta.sizes[i] = sizes[i]; meta.strides[i] = strides[i]; }
  } else { meta.ndim = 0; }

  (void)cudaGetLastError();
  dim3 grid(pol.grid_x, 1u, 1u); dim3 block(pol.block_x, 1u, 1u);
  bernoulli_f32_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
      reinterpret_cast<float*>(t.data()), meta, is_contig,
      st.seed, st.offset, total_blocks, N, p, pol.blocks_per_thread);
  cudaCheckLast("bernoulli_cuda: kernel launch failed: ");
}

void randint_(TensorImpl& t, std::int64_t low, std::int64_t high, vbt::rng::CudaGenerator& gen) {
  vbt::core::check_writable(t);
  if (t.dtype() != ScalarType::Int64) {
    throw std::runtime_error("randint: output dtype must be int64");
  }
  if (!(low < high)) {
    throw std::runtime_error("randint: require low < high and (high - low) in [1, 2^63 - 1]");
  }
  const std::uint64_t lo_u = static_cast<std::uint64_t>(low);
  const std::uint64_t hi_u = static_cast<std::uint64_t>(high);
  std::uint64_t n = hi_u - lo_u; // unsigned subtraction avoids UB on signed overflow
  if (n == 0 || n > 0x7FFFFFFFFFFFFFFFull) {
    throw std::runtime_error("randint: require low < high and (high - low) in [1, 2^63 - 1]");
  }
  if (t.device().type != kDLCUDA) {
    throw std::runtime_error("randint_cuda: expected CUDA tensor");
  }
  const std::int64_t n64 = t.numel(); if (n64 <= 0) return;
  const std::uint64_t N = static_cast<std::uint64_t>(n64);
  const std::uint64_t total_blocks = vbt::rng::ceil_div_u64(N, 2ull);

  constexpr std::uint32_t kOutputsPerBlock = 2u;  // Philox accounting only
  PhiloxState st = reserve_philox_for_cuda_op(
      t,
      gen,
      total_blocks,
      kOutputsPerBlock,
      gc::RngOpTag::Randint);

  // Size grid for per-element iteration; outputs_per_block=1, BPT_default=1
  ExecutionPolicy pol = calc_execution_policy(n64, 1u, 1u);
  if (pol.grid_x == 0u || pol.block_x == 0u) { pol.grid_x = 1u; pol.block_x = 256u; }

  vbt::cuda::DeviceGuard guard(static_cast<vbt::cuda::DeviceIndex>(t.device().index));
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(t.device().index));

  bool is_contig = t.is_contiguous();
  DeviceIndexer meta{}; meta.base = reinterpret_cast<std::uint8_t*>(t.data());
  if (!is_contig) {
    const auto& sizes = t.sizes(); const auto& strides = t.strides();
    if (sizes.size() > 8) throw std::invalid_argument("randint_: rank>8 not supported on CUDA (DeviceIndexer supports up to 8 dims)");
    meta.ndim = static_cast<std::int64_t>(sizes.size());
    for (std::size_t i = 0; i < sizes.size(); ++i) { meta.sizes[i] = sizes[i]; meta.strides[i] = strides[i]; }
  } else { meta.ndim = 0; }

  (void)cudaGetLastError();
  dim3 grid(pol.grid_x, 1u, 1u); dim3 block(pol.block_x, 1u, 1u);
  randint_i64_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
      reinterpret_cast<long long*>(t.data()), meta, is_contig,
      st.seed, st.offset, N, static_cast<long long>(low), n);
  cudaCheckLast("randint_cuda: kernel launch failed: ");
}

#endif // VBT_WITH_CUDA

} // namespace cuda
} // namespace rng
} // namespace vbt
