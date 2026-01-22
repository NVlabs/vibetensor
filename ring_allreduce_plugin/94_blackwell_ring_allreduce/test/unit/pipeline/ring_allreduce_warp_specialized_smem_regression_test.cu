// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/***************************************************************************************************
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Targeted regression tests for the warp-specialized SMEM RS path (partial/empty tiles and failure gating).
*/

// CUTLASS_RING_ALLREDUCE_ENABLE_WARP_SPECIALIZED_SMEM is enabled for this test binary via CMake.

#include "../common/cutlass_unit_test.h"

#include "cutlass/experimental/distributed/collective/ring_allreduce_kernel_sm100.cuh"
#include "cutlass/experimental/distributed/collective/ring_allreduce_tiling.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>

namespace {

using cutlass::distributed::collective::RingAllreduceDrainConfig;
using cutlass::distributed::collective::RingAllreduceError;
using cutlass::distributed::collective::RingAllreduceParams;
using cutlass::distributed::collective::RingAllreduceSystemAtomicU32;
using cutlass::distributed::collective::compute_ring_allreduce_tiling;

static bool is_sm100_or_sm103() {
  int device = 0;
  cudaError_t st = cudaGetDevice(&device);
  if (st != cudaSuccess) {
    return false;
  }

  cudaDeviceProp prop{};
  st = cudaGetDeviceProperties(&prop, device);
  if (st != cudaSuccess) {
    return false;
  }

  int cc = prop.major * 10 + prop.minor;
  return cc == 100 || cc == 103;
}

__global__ void ring_allreduce_warp_specialized_smem_regression_probe_kernel() {}

static cudaError_t ring_allreduce_warp_specialized_smem_regression_probe_launch() {
  // Clear any pre-existing per-thread CUDA error state.
  (void)cudaGetLastError();

  ring_allreduce_warp_specialized_smem_regression_probe_kernel<<<1, 1>>>();
  return cudaGetLastError();
}

static cudaError_t wait_or_timeout(cudaEvent_t done_event, std::chrono::steady_clock::time_point deadline) {
  while (true) {
    cudaError_t q = cudaEventQuery(done_event);
    if (q == cudaSuccess) {
      return cudaSuccess;
    }
    if (q != cudaErrorNotReady) {
      return q;
    }

    if (std::chrono::steady_clock::now() > deadline) {
      return cudaErrorTimeout;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

static void construct_atomic_array(RingAllreduceSystemAtomicU32* p, uint64_t len) {
  for (uint64_t i = 0; i < len; ++i) {
    new (p + i) RingAllreduceSystemAtomicU32{};
  }
}

static void destroy_atomic_array(RingAllreduceSystemAtomicU32* p, uint64_t len) {
  for (uint64_t i = 0; i < len; ++i) {
    p[i].~RingAllreduceSystemAtomicU32();
  }
}

__global__ void ring_allreduce_warp_specialized_smem_sim_allreduce_kernel(
    RingAllreduceParams<float, 8> const* params,
    uint32_t N,
    uint64_t flags_len) {
#if defined(__CUDA_ARCH__)
  uint32_t rank = static_cast<uint32_t>(blockIdx.x);
  if (rank >= N) {
    return;
  }

  RingAllreduceDrainConfig cfg{};

  RingAllreduceParams<float, 8> p = params[rank];

  uint32_t r_u32 = static_cast<uint32_t>(p.rank);
  int32_t left = (p.rank + p.world_size - 1) % p.world_size;

  // Single-tile test: channel_id=0, tile_in_chunk=0, tile_linear=0.
  cutlass::distributed::collective::detail::ring_allreduce_sm100_tile_warp_specialized_smem(
      p,
      cfg,
      /*tile_linear=*/0,
      /*channel_id=*/0,
      /*tile_in_chunk=*/0,
      N,
      r_u32,
      left,
      flags_len);
#else
  CUTLASS_UNUSED(params);
  CUTLASS_UNUSED(N);
  CUTLASS_UNUSED(flags_len);
#endif
}

__global__ void ring_allreduce_warp_specialized_smem_invalid_publish_ptr_tile_kernel(
    RingAllreduceParams<float, 8> p,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len) {
#if defined(__CUDA_ARCH__)
  RingAllreduceDrainConfig cfg{};

  cutlass::distributed::collective::detail::ring_allreduce_sm100_tile_warp_specialized_smem(
      p,
      cfg,
      /*tile_linear=*/0,
      /*channel_id=*/0,
      /*tile_in_chunk=*/0,
      N,
      r_u32,
      left,
      flags_len);
#else
  CUTLASS_UNUSED(p);
  CUTLASS_UNUSED(N);
  CUTLASS_UNUSED(r_u32);
  CUTLASS_UNUSED(left);
  CUTLASS_UNUSED(flags_len);
#endif
}

} // namespace

TEST(RingAllreduceWarpSpecializedSmemRegression, PartialTileCorrectness) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_warp_specialized_smem_regression_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  constexpr uint32_t kWorldSize = 4;
  constexpr int32_t kNumChannels = 1;
  constexpr uint32_t kTileElems = 256;
  constexpr uint64_t kCount = 800;
  constexpr uint32_t kEpoch = 1;

  auto tiling_r = compute_ring_allreduce_tiling(kCount, /*world_size=*/kWorldSize, kNumChannels, kTileElems);
  ASSERT_TRUE(tiling_r.ok()) << (tiling_r.error_reason ? tiling_r.error_reason : "<no error>");

  const uint32_t num_tiles_total = tiling_r.tiling.num_tiles_total;
  ASSERT_EQ(num_tiles_total, 1u);

  const uint64_t flags_len = uint64_t(kWorldSize) * uint64_t(num_tiles_total);

  // Allocate payload + per-rank flags.
  float* data = nullptr;
  RingAllreduceSystemAtomicU32* rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* ag_ready = nullptr;
  RingAllreduceSystemAtomicU32* abort_flags = nullptr;
  RingAllreduceSystemAtomicU32* error_flags = nullptr;
  RingAllreduceParams<float, 8>* params = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&data), kWorldSize * kCount * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&rs_ready), kWorldSize * flags_len * sizeof(RingAllreduceSystemAtomicU32)),
            cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&ag_ready), kWorldSize * flags_len * sizeof(RingAllreduceSystemAtomicU32)),
            cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&abort_flags), kWorldSize * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&error_flags), kWorldSize * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&params), kWorldSize * sizeof(RingAllreduceParams<float, 8>)), cudaSuccess);

  // Construct atomics.
  construct_atomic_array(rs_ready, kWorldSize * flags_len);
  construct_atomic_array(ag_ready, kWorldSize * flags_len);
  construct_atomic_array(abort_flags, kWorldSize);
  construct_atomic_array(error_flags, kWorldSize);

  // Initialize per-rank state.
  for (uint32_t r = 0; r < kWorldSize; ++r) {
    abort_flags[r].store(0u, cuda::memory_order_relaxed);
    error_flags[r].store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);

    for (uint64_t i = 0; i < flags_len; ++i) {
      rs_ready[r * flags_len + i].store(0u, cuda::memory_order_relaxed);
      ag_ready[r * flags_len + i].store(0u, cuda::memory_order_relaxed);
    }

    float* self_data = data + uint64_t(r) * kCount;
    for (uint64_t i = 0; i < kCount; ++i) {
      self_data[i] = static_cast<float>(r * 1000u + static_cast<uint32_t>(i));
    }

    RingAllreduceParams<float, 8> p{};
    p.world_size = static_cast<int32_t>(kWorldSize);
    p.rank = static_cast<int32_t>(r);
    p.epoch = kEpoch;

    p.count = kCount;
    p.num_channels = kNumChannels;

    p.tile_elems = tiling_r.tiling.tile_elems;
    p.num_chunks_total = tiling_r.tiling.num_chunks_total;
    p.max_chunk_elems = tiling_r.tiling.max_chunk_elems;
    p.tiles_per_chunk = tiling_r.tiling.tiles_per_chunk;
    p.num_tiles_total = tiling_r.tiling.num_tiles_total;

    p.timeout_iters = 0;
    p.timeout_cycles = 0;
    p.poll_sleep_start = 0;
    p.poll_sleep_ns = 0;

    p.self_data = self_data;
    p.self_rs_ready = rs_ready + uint64_t(r) * flags_len;
    p.self_ag_ready = ag_ready + uint64_t(r) * flags_len;
    p.self_abort = abort_flags + r;
    p.self_error = error_flags + r;

    // Peers: point directly at each rank's local allocations.
    for (uint32_t peer = 0; peer < 8; ++peer) {
      p.peer_data[peer] = nullptr;
      p.peer_rs_ready[peer] = nullptr;
      p.peer_ag_ready[peer] = nullptr;
      p.peer_abort[peer] = nullptr;

      p.peer_barrier_gather_token[peer] = nullptr;
      p.peer_barrier_gather_status[peer] = nullptr;
      p.peer_barrier_release_token[peer] = nullptr;
      p.peer_barrier_release_status[peer] = nullptr;
    }

    for (uint32_t peer = 0; peer < kWorldSize; ++peer) {
      p.peer_data[peer] = data + uint64_t(peer) * kCount;
      p.peer_rs_ready[peer] = rs_ready + uint64_t(peer) * flags_len;
      p.peer_ag_ready[peer] = ag_ready + uint64_t(peer) * flags_len;
      p.peer_abort[peer] = abort_flags + peer;
    }

    // Not used by the tile helper.
    p.self_tiles_finished = nullptr;
    p.self_barrier_gather_token = nullptr;
    p.self_barrier_gather_status = nullptr;
    p.self_barrier_release_token = nullptr;
    p.self_barrier_release_status = nullptr;

    p.debug_abort_rank = 0;
    p.debug_abort_ag_step = 0;
    p.debug_abort_before_ag_publish = 0;
    p.debug_abort_after_ag_publish = 0;

    p.debug_release_delay_rank = 0;
    p.debug_release_delay_iters = 0;

    p.debug_jitter_seed = 0;
    p.debug_jitter_max_iters = 0;
    p.debug_jitter_mask = 0;

    params[r] = p;
  }

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&done_event), cudaSuccess);

  ring_allreduce_warp_specialized_smem_sim_allreduce_kernel<<<kWorldSize, 256, 0, stream>>>(params, kWorldSize, flags_len);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    // Avoid wedging CI on a deadlock: hard-abort matches existing ring_allreduce tests.
    std::fprintf(stderr, "ring_allreduce_warp_specialized_smem_regression_test: watchdog timeout (PartialTileCorrectness)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  // Validate status and payload.
  for (uint32_t r = 0; r < kWorldSize; ++r) {
    EXPECT_EQ(abort_flags[r].load(cuda::memory_order_relaxed), 0u);
    EXPECT_EQ(error_flags[r].load(cuda::memory_order_relaxed), static_cast<uint32_t>(RingAllreduceError::kOk));

    float* self_data = data + uint64_t(r) * kCount;
    for (uint64_t i = 0; i < kCount; ++i) {
      float expected = static_cast<float>(6000u + 4u * static_cast<uint32_t>(i));
      EXPECT_EQ(self_data[i], expected);
    }

    // Publish coverage: RS publishes indices 1..N-1 and AG publishes 0..N-1.
    for (uint32_t s = 1; s < kWorldSize; ++s) {
      EXPECT_EQ(rs_ready[r * flags_len + s].load(cuda::memory_order_relaxed), kEpoch);
    }
    for (uint32_t s = 0; s < kWorldSize; ++s) {
      EXPECT_EQ(ag_ready[r * flags_len + s].load(cuda::memory_order_relaxed), kEpoch);
    }
  }

  destroy_atomic_array(error_flags, kWorldSize);
  destroy_atomic_array(abort_flags, kWorldSize);
  destroy_atomic_array(ag_ready, kWorldSize * flags_len);
  destroy_atomic_array(rs_ready, kWorldSize * flags_len);

  ASSERT_EQ(cudaFree(params), cudaSuccess);
  ASSERT_EQ(cudaFree(error_flags), cudaSuccess);
  ASSERT_EQ(cudaFree(abort_flags), cudaSuccess);
  ASSERT_EQ(cudaFree(ag_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(rs_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(data), cudaSuccess);

  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(RingAllreduceWarpSpecializedSmemRegression, EmptyTilePublishesFlags) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_warp_specialized_smem_regression_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  constexpr uint32_t kWorldSize = 4;
  constexpr int32_t kNumChannels = 1;
  constexpr uint32_t kTileElems = 256;
  constexpr uint64_t kCount = 1;
  constexpr uint32_t kEpoch = 1;

  auto tiling_r = compute_ring_allreduce_tiling(kCount, /*world_size=*/kWorldSize, kNumChannels, kTileElems);
  ASSERT_TRUE(tiling_r.ok()) << (tiling_r.error_reason ? tiling_r.error_reason : "<no error>");

  const uint32_t num_tiles_total = tiling_r.tiling.num_tiles_total;
  ASSERT_EQ(num_tiles_total, 1u);

  const uint64_t flags_len = uint64_t(kWorldSize) * uint64_t(num_tiles_total);

  float* data = nullptr;
  RingAllreduceSystemAtomicU32* rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* ag_ready = nullptr;
  RingAllreduceSystemAtomicU32* abort_flags = nullptr;
  RingAllreduceSystemAtomicU32* error_flags = nullptr;
  RingAllreduceParams<float, 8>* params = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&data), kWorldSize * kCount * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&rs_ready), kWorldSize * flags_len * sizeof(RingAllreduceSystemAtomicU32)),
            cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&ag_ready), kWorldSize * flags_len * sizeof(RingAllreduceSystemAtomicU32)),
            cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&abort_flags), kWorldSize * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&error_flags), kWorldSize * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&params), kWorldSize * sizeof(RingAllreduceParams<float, 8>)), cudaSuccess);

  construct_atomic_array(rs_ready, kWorldSize * flags_len);
  construct_atomic_array(ag_ready, kWorldSize * flags_len);
  construct_atomic_array(abort_flags, kWorldSize);
  construct_atomic_array(error_flags, kWorldSize);

  for (uint32_t r = 0; r < kWorldSize; ++r) {
    abort_flags[r].store(0u, cuda::memory_order_relaxed);
    error_flags[r].store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);

    for (uint64_t i = 0; i < flags_len; ++i) {
      rs_ready[r * flags_len + i].store(0u, cuda::memory_order_relaxed);
      ag_ready[r * flags_len + i].store(0u, cuda::memory_order_relaxed);
    }

    float* self_data = data + uint64_t(r) * kCount;
    self_data[0] = static_cast<float>(r + 1u);

    RingAllreduceParams<float, 8> p{};
    p.world_size = static_cast<int32_t>(kWorldSize);
    p.rank = static_cast<int32_t>(r);
    p.epoch = kEpoch;

    p.count = kCount;
    p.num_channels = kNumChannels;

    p.tile_elems = tiling_r.tiling.tile_elems;
    p.num_chunks_total = tiling_r.tiling.num_chunks_total;
    p.max_chunk_elems = tiling_r.tiling.max_chunk_elems;
    p.tiles_per_chunk = tiling_r.tiling.tiles_per_chunk;
    p.num_tiles_total = tiling_r.tiling.num_tiles_total;

    p.timeout_iters = 0;
    p.timeout_cycles = 0;
    p.poll_sleep_start = 0;
    p.poll_sleep_ns = 0;

    p.self_data = self_data;
    p.self_rs_ready = rs_ready + uint64_t(r) * flags_len;
    p.self_ag_ready = ag_ready + uint64_t(r) * flags_len;
    p.self_abort = abort_flags + r;
    p.self_error = error_flags + r;

    for (uint32_t peer = 0; peer < 8; ++peer) {
      p.peer_data[peer] = nullptr;
      p.peer_rs_ready[peer] = nullptr;
      p.peer_ag_ready[peer] = nullptr;
      p.peer_abort[peer] = nullptr;

      p.peer_barrier_gather_token[peer] = nullptr;
      p.peer_barrier_gather_status[peer] = nullptr;
      p.peer_barrier_release_token[peer] = nullptr;
      p.peer_barrier_release_status[peer] = nullptr;
    }

    for (uint32_t peer = 0; peer < kWorldSize; ++peer) {
      p.peer_data[peer] = data + uint64_t(peer) * kCount;
      p.peer_rs_ready[peer] = rs_ready + uint64_t(peer) * flags_len;
      p.peer_ag_ready[peer] = ag_ready + uint64_t(peer) * flags_len;
      p.peer_abort[peer] = abort_flags + peer;
    }

    p.self_tiles_finished = nullptr;
    p.self_barrier_gather_token = nullptr;
    p.self_barrier_gather_status = nullptr;
    p.self_barrier_release_token = nullptr;
    p.self_barrier_release_status = nullptr;

    params[r] = p;
  }

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&done_event), cudaSuccess);

  ring_allreduce_warp_specialized_smem_sim_allreduce_kernel<<<kWorldSize, 256, 0, stream>>>(params, kWorldSize, flags_len);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    // Avoid wedging CI on a deadlock: hard-abort matches existing ring_allreduce tests.
    std::fprintf(stderr, "ring_allreduce_warp_specialized_smem_regression_test: watchdog timeout (EmptyTilePublishesFlags)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  for (uint32_t r = 0; r < kWorldSize; ++r) {
    EXPECT_EQ(abort_flags[r].load(cuda::memory_order_relaxed), 0u);
    EXPECT_EQ(error_flags[r].load(cuda::memory_order_relaxed), static_cast<uint32_t>(RingAllreduceError::kOk));

    float* self_data = data + uint64_t(r) * kCount;
    EXPECT_EQ(self_data[0], 10.0f);

    for (uint32_t s = 1; s < kWorldSize; ++s) {
      EXPECT_EQ(rs_ready[r * flags_len + s].load(cuda::memory_order_relaxed), kEpoch);
    }
    for (uint32_t s = 0; s < kWorldSize; ++s) {
      EXPECT_EQ(ag_ready[r * flags_len + s].load(cuda::memory_order_relaxed), kEpoch);
    }
  }

  destroy_atomic_array(error_flags, kWorldSize);
  destroy_atomic_array(abort_flags, kWorldSize);
  destroy_atomic_array(ag_ready, kWorldSize * flags_len);
  destroy_atomic_array(rs_ready, kWorldSize * flags_len);

  ASSERT_EQ(cudaFree(params), cudaSuccess);
  ASSERT_EQ(cudaFree(error_flags), cudaSuccess);
  ASSERT_EQ(cudaFree(abort_flags), cudaSuccess);
  ASSERT_EQ(cudaFree(ag_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(rs_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(data), cudaSuccess);

  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(RingAllreduceWarpSpecializedSmemRegression, InvalidRsPublishPointerNoAg) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_warp_specialized_smem_regression_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  constexpr uint32_t kWorldSize = 4;
  constexpr uint32_t kEpoch = 1;
  constexpr uint64_t kFlagsLen = kWorldSize;

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;
  RingAllreduceSystemAtomicU32* self_ag_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_ag_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_abort = nullptr;

  float* self_data = nullptr;
  float* peer_data = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_ag_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_rs_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_ag_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_data), sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_data), sizeof(float)), cudaSuccess);

  new (self_abort) RingAllreduceSystemAtomicU32{};
  new (self_error) RingAllreduceSystemAtomicU32{};
  new (peer_abort) RingAllreduceSystemAtomicU32{};

  construct_atomic_array(self_ag_ready, kFlagsLen);
  construct_atomic_array(peer_rs_ready, kFlagsLen);
  construct_atomic_array(peer_ag_ready, kFlagsLen);

  self_abort->store(0u, cuda::memory_order_relaxed);
  self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  peer_abort->store(0u, cuda::memory_order_relaxed);

  for (uint64_t i = 0; i < kFlagsLen; ++i) {
    self_ag_ready[i].store(0u, cuda::memory_order_relaxed);
    peer_rs_ready[i].store(kEpoch, cuda::memory_order_release);
    peer_ag_ready[i].store(kEpoch, cuda::memory_order_release);
  }

  self_data[0] = 1.0f;
  peer_data[0] = 2.0f;

  RingAllreduceParams<float, 8> p{};
  p.world_size = static_cast<int32_t>(kWorldSize);
  p.rank = 1;
  p.epoch = kEpoch;

  p.count = 1;
  p.num_channels = 1;
  p.tile_elems = 1;
  p.num_chunks_total = kWorldSize;
  p.max_chunk_elems = 1;
  p.tiles_per_chunk = 1;
  p.num_tiles_total = 1;

  p.timeout_iters = 0;
  p.timeout_cycles = 0;
  p.poll_sleep_start = 0;
  p.poll_sleep_ns = 0;

  p.self_data = self_data;
  p.self_rs_ready = nullptr; // force invalid publish pointer
  p.self_ag_ready = self_ag_ready;
  p.self_abort = self_abort;
  p.self_error = self_error;

  for (uint32_t i = 0; i < 8; ++i) {
    p.peer_data[i] = nullptr;
    p.peer_rs_ready[i] = nullptr;
    p.peer_ag_ready[i] = nullptr;
    p.peer_abort[i] = nullptr;
  }

  p.peer_data[0] = peer_data;
  p.peer_rs_ready[0] = peer_rs_ready;
  p.peer_ag_ready[0] = peer_ag_ready;
  p.peer_abort[0] = peer_abort;

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&done_event), cudaSuccess);

  ring_allreduce_warp_specialized_smem_invalid_publish_ptr_tile_kernel<<<1, 256, 0, stream>>>(
      p,
      /*N=*/kWorldSize,
      /*r_u32=*/1,
      /*left=*/0,
      /*flags_len=*/kFlagsLen);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    // Avoid wedging CI on a deadlock: hard-abort matches existing ring_allreduce tests.
    std::fprintf(stderr, "ring_allreduce_warp_specialized_smem_regression_test: watchdog timeout (InvalidRsPublishPointerNoAg)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  EXPECT_EQ(self_abort->load(cuda::memory_order_relaxed), 1u);
  EXPECT_EQ(self_error->load(cuda::memory_order_relaxed), static_cast<uint32_t>(RingAllreduceError::kInvalidParams));

  // RS failure must gate AG (do not publish AG step 0).
  EXPECT_EQ(self_ag_ready[0].load(cuda::memory_order_relaxed), 0u);

  destroy_atomic_array(peer_ag_ready, kFlagsLen);
  destroy_atomic_array(peer_rs_ready, kFlagsLen);
  destroy_atomic_array(self_ag_ready, kFlagsLen);

  peer_abort->~RingAllreduceSystemAtomicU32();
  self_error->~RingAllreduceSystemAtomicU32();
  self_abort->~RingAllreduceSystemAtomicU32();

  ASSERT_EQ(cudaFree(peer_data), cudaSuccess);
  ASSERT_EQ(cudaFree(self_data), cudaSuccess);

  ASSERT_EQ(cudaFree(peer_abort), cudaSuccess);
  ASSERT_EQ(cudaFree(peer_ag_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(peer_rs_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(self_ag_ready), cudaSuccess);

  ASSERT_EQ(cudaFree(self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self_abort), cudaSuccess);

  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
