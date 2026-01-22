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
    \brief Negative/edge-case unit tests for the warp-specialized SMEM RS path.
*/

// CUTLASS_RING_ALLREDUCE_TEST_WARP_SPECIALIZED_SMEM_RS_IMPL_TAG is enabled for this test binary via CMake.

#include "../common/cutlass_unit_test.h"

#include "cutlass/experimental/distributed/collective/ring_allreduce_kernel_sm100.cuh"

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
using cutlass::distributed::collective::detail::ring_allreduce_warp_specialized_smem_rs_impl_tag;

#if CUTLASS_RING_ALLREDUCE_TEST_WARP_SPECIALIZED_SMEM_RS_IMPL_TAG
using cutlass::distributed::collective::detail::kRingAllreduceWarpSpecializedSmemRsImplTagPingPongNoOverlap;
using cutlass::distributed::collective::detail::kRingAllreduceWarpSpecializedSmemRsImplTagPingPongPrefetch;
using cutlass::distributed::collective::detail::kRingAllreduceWarpSpecializedSmemRsImplTagSingleBuffer;
#endif

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

__global__ void ring_allreduce_warp_specialized_smem_negative_probe_kernel() {}

static cudaError_t ring_allreduce_warp_specialized_smem_negative_probe_launch() {
  // Clear any pre-existing per-thread CUDA error state.
  (void)cudaGetLastError();

  ring_allreduce_warp_specialized_smem_negative_probe_kernel<<<1, 1>>>();
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

__global__ void ring_allreduce_warp_specialized_smem_negative_rs_kernel(
    RingAllreduceParams<float, 8> p,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len,
    uint32_t* out_rs_ok) {
#if defined(__CUDA_ARCH__)
  RingAllreduceDrainConfig cfg;
  cfg.timeout_iters = p.timeout_iters;
  cfg.timeout_cycles = p.timeout_cycles;
  cfg.poll_sleep_start = p.poll_sleep_start;
  cfg.poll_sleep_ns = p.poll_sleep_ns;

  bool rs_ok = cutlass::distributed::collective::detail::ring_allreduce_sm100_tile_warp_specialized_smem_rs(
      p,
      cfg,
      /*tile_linear=*/0,
      /*channel_id=*/0,
      /*tile_in_chunk=*/0,
      N,
      r_u32,
      left,
      flags_len);

  if (out_rs_ok && threadIdx.x == 0) {
    out_rs_ok[0] = rs_ok ? 1u : 0u;
  }
#else
  CUTLASS_UNUSED(p);
  CUTLASS_UNUSED(N);
  CUTLASS_UNUSED(r_u32);
  CUTLASS_UNUSED(left);
  CUTLASS_UNUSED(flags_len);

  if (out_rs_ok && threadIdx.x == 0) {
    out_rs_ok[0] = 0u;
  }
#endif
}

__global__ void ring_allreduce_warp_specialized_smem_negative_tile_kernel(
    RingAllreduceParams<float, 8> p,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len) {
#if defined(__CUDA_ARCH__)
  RingAllreduceDrainConfig cfg;
  cfg.timeout_iters = p.timeout_iters;
  cfg.timeout_cycles = p.timeout_cycles;
  cfg.poll_sleep_start = p.poll_sleep_start;
  cfg.poll_sleep_ns = p.poll_sleep_ns;

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

TEST(RingAllreduceWarpSpecializedSmemNegative, EC_BADLEN) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_warp_specialized_smem_negative_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  constexpr uint32_t N = 4;
  constexpr uint32_t kNumTilesTotal = 1;
  constexpr uint32_t kEpoch = 1;
  constexpr uint64_t kFlagsLen = uint64_t(N) * uint64_t(kNumTilesTotal);

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;
  RingAllreduceSystemAtomicU32* self_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* self_ag_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_ag_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_abort = nullptr;

  float* self_data = nullptr;
  float* peer_data = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_rs_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_ag_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_rs_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_ag_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_data), N * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_data), N * sizeof(float)), cudaSuccess);

  new (self_abort) RingAllreduceSystemAtomicU32{};
  new (self_error) RingAllreduceSystemAtomicU32{};
  new (peer_abort) RingAllreduceSystemAtomicU32{};

  construct_atomic_array(self_rs_ready, kFlagsLen);
  construct_atomic_array(self_ag_ready, kFlagsLen);
  construct_atomic_array(peer_rs_ready, kFlagsLen);
  construct_atomic_array(peer_ag_ready, kFlagsLen);

  self_abort->store(0u, cuda::memory_order_relaxed);
  self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  peer_abort->store(0u, cuda::memory_order_relaxed);

  for (uint64_t i = 0; i < kFlagsLen; ++i) {
    self_rs_ready[i].store(0u, cuda::memory_order_relaxed);
    self_ag_ready[i].store(0u, cuda::memory_order_relaxed);
    // If RS/AG is accidentally entered, make the peer-side flags immediately ready.
    peer_rs_ready[i].store(kEpoch, cuda::memory_order_release);
    peer_ag_ready[i].store(kEpoch, cuda::memory_order_release);
  }

  for (uint32_t i = 0; i < N; ++i) {
    self_data[i] = 1.0f;
    peer_data[i] = 2.0f;
  }

  RingAllreduceParams<float, 8> p{};
  p.world_size = N;
  p.rank = 1;
  p.epoch = kEpoch;

  p.count = N;
  p.num_channels = 1;
  p.tile_elems = 1;
  p.num_chunks_total = N;
  p.max_chunk_elems = 1;
  p.tiles_per_chunk = 1;
  p.num_tiles_total = kNumTilesTotal;

  p.timeout_iters = 0;
  p.timeout_cycles = 0;
  p.poll_sleep_start = 0;
  p.poll_sleep_ns = 0;

  p.self_data = self_data;
  p.self_rs_ready = self_rs_ready;
  p.self_ag_ready = self_ag_ready;
  p.self_abort = self_abort;
  p.self_error = self_error;

  p.peer_data[0] = peer_data;
  p.peer_rs_ready[0] = peer_rs_ready;
  p.peer_ag_ready[0] = peer_ag_ready;
  p.peer_abort[0] = peer_abort;

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&done_event), cudaSuccess);

  // flags_len==0 => safe_ptr_add must fail (kInvalidParams). Failure must gate AG.
  ring_allreduce_warp_specialized_smem_negative_tile_kernel<<<1, 256, 0, stream>>>(p, /*N=*/N, /*r_u32=*/1, /*left=*/0, /*flags_len=*/0);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    std::fprintf(stderr, "ring_allreduce_warp_specialized_smem_negative_test: watchdog timeout (EC_BADLEN)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  EXPECT_EQ(self_abort->load(cuda::memory_order_relaxed), 1u);
  EXPECT_EQ(self_error->load(cuda::memory_order_relaxed), static_cast<uint32_t>(RingAllreduceError::kInvalidParams));
  // RS failure must gate AG.
  EXPECT_EQ(self_ag_ready[0].load(cuda::memory_order_relaxed), 0u);
  EXPECT_EQ(self_rs_ready[1].load(cuda::memory_order_relaxed), 0u);

  destroy_atomic_array(peer_ag_ready, kFlagsLen);
  destroy_atomic_array(peer_rs_ready, kFlagsLen);
  destroy_atomic_array(self_ag_ready, kFlagsLen);
  destroy_atomic_array(self_rs_ready, kFlagsLen);

  peer_abort->~RingAllreduceSystemAtomicU32();
  self_error->~RingAllreduceSystemAtomicU32();
  self_abort->~RingAllreduceSystemAtomicU32();

  ASSERT_EQ(cudaFree(peer_data), cudaSuccess);
  ASSERT_EQ(cudaFree(self_data), cudaSuccess);

  ASSERT_EQ(cudaFree(peer_abort), cudaSuccess);
  ASSERT_EQ(cudaFree(peer_ag_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(peer_rs_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(self_ag_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(self_rs_ready), cudaSuccess);

  ASSERT_EQ(cudaFree(self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self_abort), cudaSuccess);

  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(RingAllreduceWarpSpecializedSmemNegative, EC_NULL_PEER_DATA) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_warp_specialized_smem_negative_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;
  RingAllreduceSystemAtomicU32* self_rs_ready = nullptr;
  float* self_data = nullptr;
  uint32_t* out_rs_ok = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_rs_ready), 2 * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_data), sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&out_rs_ok), sizeof(uint32_t)), cudaSuccess);

  new (self_abort) RingAllreduceSystemAtomicU32{};
  new (self_error) RingAllreduceSystemAtomicU32{};
  construct_atomic_array(self_rs_ready, /*len=*/2);

  self_abort->store(0u, cuda::memory_order_relaxed);
  self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  self_rs_ready[0].store(0u, cuda::memory_order_relaxed);
  self_rs_ready[1].store(0u, cuda::memory_order_relaxed);

  self_data[0] = 1.0f;
  out_rs_ok[0] = 0xFFFF'FFFFu;

  RingAllreduceParams<float, 8> p{};
  p.world_size = 2;
  p.rank = 1;
  p.epoch = 1;

  p.count = 1;
  p.num_channels = 1;
  p.tile_elems = 1;
  p.num_chunks_total = 2;
  p.max_chunk_elems = 1;
  p.tiles_per_chunk = 1;
  p.num_tiles_total = 1;

  p.timeout_iters = 0;
  p.timeout_cycles = 0;
  p.poll_sleep_start = 0;
  p.poll_sleep_ns = 0;

  p.self_data = self_data;
  p.self_rs_ready = self_rs_ready;
  p.self_abort = self_abort;
  p.self_error = self_error;

  // peer_data[left] == nullptr must publish kInvalidParams without hanging.
  p.peer_data[0] = nullptr;

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&done_event), cudaSuccess);

  ring_allreduce_warp_specialized_smem_negative_rs_kernel<<<1, 256, 0, stream>>>(p, /*N=*/2, /*r_u32=*/1, /*left=*/0, /*flags_len=*/2, out_rs_ok);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    std::fprintf(stderr, "ring_allreduce_warp_specialized_smem_negative_test: watchdog timeout (EC_NULL_PEER_DATA)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  EXPECT_EQ(out_rs_ok[0], 0u);
  EXPECT_EQ(self_abort->load(cuda::memory_order_relaxed), 1u);
  EXPECT_EQ(self_error->load(cuda::memory_order_relaxed), static_cast<uint32_t>(RingAllreduceError::kInvalidParams));
  EXPECT_EQ(self_rs_ready[1].load(cuda::memory_order_relaxed), 0u);

  destroy_atomic_array(self_rs_ready, /*len=*/2);
  self_error->~RingAllreduceSystemAtomicU32();
  self_abort->~RingAllreduceSystemAtomicU32();

  ASSERT_EQ(cudaFree(out_rs_ok), cudaSuccess);
  ASSERT_EQ(cudaFree(self_data), cudaSuccess);
  ASSERT_EQ(cudaFree(self_rs_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self_abort), cudaSuccess);

  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(RingAllreduceWarpSpecializedSmemNegative, EC_OVERSIZE_TILE) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_warp_specialized_smem_negative_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  constexpr uint32_t kStageElemsMax = cutlass::distributed::collective::detail::kStageElemsMax;
  constexpr uint32_t kOversizeTileElems = kStageElemsMax + 1;

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;
  RingAllreduceSystemAtomicU32* self_rs_ready = nullptr;
  float* self_data = nullptr;
  float* peer_data = nullptr;
  uint32_t* out_rs_ok = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_rs_ready), 2 * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_data), kOversizeTileElems * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_data), kOversizeTileElems * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&out_rs_ok), sizeof(uint32_t)), cudaSuccess);

  new (self_abort) RingAllreduceSystemAtomicU32{};
  new (self_error) RingAllreduceSystemAtomicU32{};
  construct_atomic_array(self_rs_ready, /*len=*/2);

  self_abort->store(0u, cuda::memory_order_relaxed);
  self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  self_rs_ready[0].store(0u, cuda::memory_order_relaxed);
  self_rs_ready[1].store(0u, cuda::memory_order_relaxed);

  for (uint32_t i = 0; i < kOversizeTileElems; ++i) {
    self_data[i] = static_cast<float>(i);
    peer_data[i] = static_cast<float>(i);
  }

  out_rs_ok[0] = 0xFFFF'FFFFu;

  RingAllreduceParams<float, 8> p{};
  p.world_size = 2;
  p.rank = 1;
  p.epoch = 1;

  p.count = kOversizeTileElems;
  p.num_channels = 1;
  p.tile_elems = kOversizeTileElems;
  p.num_chunks_total = 2;
  p.max_chunk_elems = kOversizeTileElems;
  p.tiles_per_chunk = 1;
  p.num_tiles_total = 1;

  p.timeout_iters = 0;
  p.timeout_cycles = 0;
  p.poll_sleep_start = 0;
  p.poll_sleep_ns = 0;

  p.self_data = self_data;
  p.self_rs_ready = self_rs_ready;
  p.self_abort = self_abort;
  p.self_error = self_error;

  // Keep peer_data non-null so the oversize guard is the only failure mode.
  p.peer_data[0] = peer_data;

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&done_event), cudaSuccess);

  ring_allreduce_warp_specialized_smem_negative_rs_kernel<<<1, 256, 0, stream>>>(p, /*N=*/2, /*r_u32=*/1, /*left=*/0, /*flags_len=*/2, out_rs_ok);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    std::fprintf(stderr, "ring_allreduce_warp_specialized_smem_negative_test: watchdog timeout (EC_OVERSIZE_TILE)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  EXPECT_EQ(out_rs_ok[0], 0u);
  EXPECT_EQ(self_abort->load(cuda::memory_order_relaxed), 1u);
  EXPECT_EQ(self_error->load(cuda::memory_order_relaxed), static_cast<uint32_t>(RingAllreduceError::kInvalidParams));
  EXPECT_EQ(self_rs_ready[1].load(cuda::memory_order_relaxed), 0u);

  destroy_atomic_array(self_rs_ready, /*len=*/2);
  self_error->~RingAllreduceSystemAtomicU32();
  self_abort->~RingAllreduceSystemAtomicU32();

  ASSERT_EQ(cudaFree(out_rs_ok), cudaSuccess);
  ASSERT_EQ(cudaFree(peer_data), cudaSuccess);
  ASSERT_EQ(cudaFree(self_data), cudaSuccess);
  ASSERT_EQ(cudaFree(self_rs_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self_abort), cudaSuccess);

  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(RingAllreduceWarpSpecializedSmemNegative, EC_TIMEOUT) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_warp_specialized_smem_negative_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  constexpr uint32_t N = 4;
  constexpr uint64_t kFlagsLen = N;

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;
  RingAllreduceSystemAtomicU32* self_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* self_ag_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_ag_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_abort = nullptr;
  float* self_data = nullptr;
  float* peer_data = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_rs_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_ag_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_rs_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_ag_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_data), N * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_data), N * sizeof(float)), cudaSuccess);

  new (self_abort) RingAllreduceSystemAtomicU32{};
  new (self_error) RingAllreduceSystemAtomicU32{};
  new (peer_abort) RingAllreduceSystemAtomicU32{};

  construct_atomic_array(self_rs_ready, kFlagsLen);
  construct_atomic_array(self_ag_ready, kFlagsLen);
  construct_atomic_array(peer_rs_ready, kFlagsLen);
  construct_atomic_array(peer_ag_ready, kFlagsLen);

  self_abort->store(0u, cuda::memory_order_relaxed);
  self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  peer_abort->store(0u, cuda::memory_order_relaxed);

  for (uint64_t i = 0; i < kFlagsLen; ++i) {
    self_rs_ready[i].store(0u, cuda::memory_order_relaxed);
    self_ag_ready[i].store(0u, cuda::memory_order_relaxed);
    peer_rs_ready[i].store(0u, cuda::memory_order_relaxed);
    // If AG is accidentally entered, make the peer-side AG flags immediately ready.
    peer_ag_ready[i].store(1u, cuda::memory_order_relaxed);
  }

  for (uint32_t i = 0; i < N; ++i) {
    self_data[i] = static_cast<float>(i);
    peer_data[i] = static_cast<float>(i + 100);
  }

  RingAllreduceParams<float, 8> p{};
  p.world_size = N;
  p.rank = 1;
  p.epoch = 1;

  p.count = N;
  p.num_channels = 1;
  p.tile_elems = 1;
  p.num_chunks_total = N;
  p.max_chunk_elems = 1;
  p.tiles_per_chunk = 1;
  p.num_tiles_total = 1;

  // Enable a fast timeout (iteration-based).
  p.timeout_iters = 128;
  p.timeout_cycles = 0;
  p.poll_sleep_start = 0;
  p.poll_sleep_ns = 0;

  p.self_data = self_data;
  p.self_rs_ready = self_rs_ready;
  p.self_ag_ready = self_ag_ready;
  p.self_abort = self_abort;
  p.self_error = self_error;

  p.peer_data[0] = peer_data;
  p.peer_rs_ready[0] = peer_rs_ready;
  p.peer_ag_ready[0] = peer_ag_ready;
  p.peer_abort[0] = peer_abort;

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&done_event), cudaSuccess);

  // r_u32==1 => step 0 uses chunk 0 (base 0), step 1 uses chunk 3 (base 3).
  ring_allreduce_warp_specialized_smem_negative_tile_kernel<<<1, 256, 0, stream>>>(p, /*N=*/N, /*r_u32=*/1, /*left=*/0, /*flags_len=*/kFlagsLen);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    std::fprintf(stderr, "ring_allreduce_warp_specialized_smem_negative_test: watchdog timeout (EC_TIMEOUT)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  EXPECT_EQ(self_abort->load(cuda::memory_order_relaxed), 1u);
  EXPECT_EQ(self_error->load(cuda::memory_order_relaxed), static_cast<uint32_t>(RingAllreduceError::kTimeout));
  // RS failure must gate AG.
  EXPECT_EQ(self_ag_ready[0].load(cuda::memory_order_relaxed), 0u);
  // Step 0 should have published before timing out at step 1.
  EXPECT_EQ(self_rs_ready[1].load(cuda::memory_order_relaxed), 1u);

  destroy_atomic_array(peer_ag_ready, kFlagsLen);
  destroy_atomic_array(peer_rs_ready, kFlagsLen);
  destroy_atomic_array(self_ag_ready, kFlagsLen);
  destroy_atomic_array(self_rs_ready, kFlagsLen);

  peer_abort->~RingAllreduceSystemAtomicU32();
  self_error->~RingAllreduceSystemAtomicU32();
  self_abort->~RingAllreduceSystemAtomicU32();

  ASSERT_EQ(cudaFree(peer_data), cudaSuccess);
  ASSERT_EQ(cudaFree(self_data), cudaSuccess);

  ASSERT_EQ(cudaFree(peer_abort), cudaSuccess);
  ASSERT_EQ(cudaFree(peer_ag_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(peer_rs_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(self_ag_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(self_rs_ready), cudaSuccess);

  ASSERT_EQ(cudaFree(self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self_abort), cudaSuccess);

  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(RingAllreduceWarpSpecializedSmemNegative, EC_TIGHT_TIMEOUT_OK) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_warp_specialized_smem_negative_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

#if CUTLASS_RING_ALLREDUCE_TEST_FORCE_NAMED_BARRIER_DISABLED
  GTEST_SKIP() << "requires NamedBarrier ping-pong RS";
#endif

  constexpr uint32_t N = 4;
  constexpr uint64_t kFlagsLen = N;

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;
  RingAllreduceSystemAtomicU32* self_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_abort = nullptr;

  float* self_data = nullptr;
  float* peer_data = nullptr;
  uint32_t* out_rs_ok = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_rs_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_rs_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_data), N * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_data), N * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&out_rs_ok), sizeof(uint32_t)), cudaSuccess);

  new (self_abort) RingAllreduceSystemAtomicU32{};
  new (self_error) RingAllreduceSystemAtomicU32{};
  new (peer_abort) RingAllreduceSystemAtomicU32{};

  construct_atomic_array(self_rs_ready, kFlagsLen);
  construct_atomic_array(peer_rs_ready, kFlagsLen);

  self_abort->store(0u, cuda::memory_order_relaxed);
  self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  peer_abort->store(0u, cuda::memory_order_relaxed);

  for (uint64_t i = 0; i < kFlagsLen; ++i) {
    self_rs_ready[i].store(0u, cuda::memory_order_relaxed);
    peer_rs_ready[i].store(/*epoch=*/1u, cuda::memory_order_release);
  }

  for (uint32_t i = 0; i < N; ++i) {
    self_data[i] = static_cast<float>(i);
    peer_data[i] = static_cast<float>(i + 100);
  }

  out_rs_ok[0] = 0xFFFF'FFFFu;

  RingAllreduceParams<float, 8> p{};
  p.world_size = N;
  p.rank = 1;
  p.epoch = 1;

  p.count = N;
  p.num_channels = 1;
  p.tile_elems = 1;
  p.num_chunks_total = N;
  p.max_chunk_elems = 1;
  p.tiles_per_chunk = 1;
  p.num_tiles_total = 1;

  // Timeouts enabled => prefetch must be disabled.
  p.timeout_iters = 128;
  p.timeout_cycles = 0;
  p.poll_sleep_start = 0;
  p.poll_sleep_ns = 0;

  p.self_data = self_data;
  p.self_rs_ready = self_rs_ready;
  p.self_abort = self_abort;
  p.self_error = self_error;

  p.peer_data[0] = peer_data;
  p.peer_rs_ready[0] = peer_rs_ready;
  p.peer_abort[0] = peer_abort;

  uint32_t zero = 0u;
  ASSERT_EQ(cudaMemcpyToSymbol(ring_allreduce_warp_specialized_smem_rs_impl_tag, &zero, sizeof(zero)), cudaSuccess);

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&done_event), cudaSuccess);

  ring_allreduce_warp_specialized_smem_negative_rs_kernel<<<1, 256, 0, stream>>>(p, /*N=*/N, /*r_u32=*/1, /*left=*/0, /*flags_len=*/kFlagsLen, out_rs_ok);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    std::fprintf(stderr, "ring_allreduce_warp_specialized_smem_negative_test: watchdog timeout (EC_TIGHT_TIMEOUT_OK)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  uint32_t tag = 0u;
  ASSERT_EQ(cudaMemcpyFromSymbol(&tag, ring_allreduce_warp_specialized_smem_rs_impl_tag, sizeof(tag)), cudaSuccess);

  EXPECT_EQ(out_rs_ok[0], 1u);
  EXPECT_EQ(tag, kRingAllreduceWarpSpecializedSmemRsImplTagPingPongNoOverlap);

  EXPECT_EQ(self_abort->load(cuda::memory_order_relaxed), 0u);
  EXPECT_EQ(self_error->load(cuda::memory_order_relaxed), static_cast<uint32_t>(RingAllreduceError::kOk));

  EXPECT_EQ(self_rs_ready[1].load(cuda::memory_order_relaxed), 1u);
  EXPECT_EQ(self_rs_ready[2].load(cuda::memory_order_relaxed), 1u);
  EXPECT_EQ(self_rs_ready[3].load(cuda::memory_order_relaxed), 1u);

  destroy_atomic_array(peer_rs_ready, kFlagsLen);
  destroy_atomic_array(self_rs_ready, kFlagsLen);

  peer_abort->~RingAllreduceSystemAtomicU32();
  self_error->~RingAllreduceSystemAtomicU32();
  self_abort->~RingAllreduceSystemAtomicU32();

  ASSERT_EQ(cudaFree(out_rs_ok), cudaSuccess);
  ASSERT_EQ(cudaFree(peer_data), cudaSuccess);
  ASSERT_EQ(cudaFree(self_data), cudaSuccess);

  ASSERT_EQ(cudaFree(peer_abort), cudaSuccess);
  ASSERT_EQ(cudaFree(peer_rs_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(self_rs_ready), cudaSuccess);

  ASSERT_EQ(cudaFree(self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self_abort), cudaSuccess);

  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(RingAllreduceWarpSpecializedSmemNegative, EC_NO_TIMEOUT_PREFETCH_TAG) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_warp_specialized_smem_negative_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

#if CUTLASS_RING_ALLREDUCE_TEST_FORCE_NAMED_BARRIER_DISABLED
  GTEST_SKIP() << "requires NamedBarrier ping-pong RS";
#endif

  constexpr uint32_t N = 4;
  constexpr uint64_t kFlagsLen = N;

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;
  RingAllreduceSystemAtomicU32* self_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_abort = nullptr;

  float* self_data = nullptr;
  float* peer_data = nullptr;
  uint32_t* out_rs_ok = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_rs_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_rs_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_data), N * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_data), N * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&out_rs_ok), sizeof(uint32_t)), cudaSuccess);

  new (self_abort) RingAllreduceSystemAtomicU32{};
  new (self_error) RingAllreduceSystemAtomicU32{};
  new (peer_abort) RingAllreduceSystemAtomicU32{};

  construct_atomic_array(self_rs_ready, kFlagsLen);
  construct_atomic_array(peer_rs_ready, kFlagsLen);

  self_abort->store(0u, cuda::memory_order_relaxed);
  self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  peer_abort->store(0u, cuda::memory_order_relaxed);

  for (uint64_t i = 0; i < kFlagsLen; ++i) {
    self_rs_ready[i].store(0u, cuda::memory_order_relaxed);
    peer_rs_ready[i].store(/*epoch=*/1u, cuda::memory_order_release);
  }

  for (uint32_t i = 0; i < N; ++i) {
    self_data[i] = static_cast<float>(i);
    peer_data[i] = static_cast<float>(i + 100);
  }

  out_rs_ok[0] = 0xFFFF'FFFFu;

  RingAllreduceParams<float, 8> p{};
  p.world_size = N;
  p.rank = 1;
  p.epoch = 1;

  p.count = N;
  p.num_channels = 1;
  p.tile_elems = 1;
  p.num_chunks_total = N;
  p.max_chunk_elems = 1;
  p.tiles_per_chunk = 1;
  p.num_tiles_total = 1;

  // Timeouts disabled => prefetch overlap is allowed.
  p.timeout_iters = 0;
  p.timeout_cycles = 0;
  p.poll_sleep_start = 0;
  p.poll_sleep_ns = 0;

  p.self_data = self_data;
  p.self_rs_ready = self_rs_ready;
  p.self_abort = self_abort;
  p.self_error = self_error;

  p.peer_data[0] = peer_data;
  p.peer_rs_ready[0] = peer_rs_ready;
  p.peer_abort[0] = peer_abort;

  uint32_t zero = 0u;
  ASSERT_EQ(cudaMemcpyToSymbol(ring_allreduce_warp_specialized_smem_rs_impl_tag, &zero, sizeof(zero)), cudaSuccess);

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&done_event), cudaSuccess);

  ring_allreduce_warp_specialized_smem_negative_rs_kernel<<<1, 256, 0, stream>>>(p, /*N=*/N, /*r_u32=*/1, /*left=*/0, /*flags_len=*/kFlagsLen, out_rs_ok);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    std::fprintf(stderr, "ring_allreduce_warp_specialized_smem_negative_test: watchdog timeout (EC_NO_TIMEOUT_PREFETCH_TAG)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  uint32_t tag = 0u;
  ASSERT_EQ(cudaMemcpyFromSymbol(&tag, ring_allreduce_warp_specialized_smem_rs_impl_tag, sizeof(tag)), cudaSuccess);

  EXPECT_EQ(out_rs_ok[0], 1u);
  EXPECT_EQ(tag, kRingAllreduceWarpSpecializedSmemRsImplTagPingPongPrefetch);

  EXPECT_EQ(self_abort->load(cuda::memory_order_relaxed), 0u);
  EXPECT_EQ(self_error->load(cuda::memory_order_relaxed), static_cast<uint32_t>(RingAllreduceError::kOk));

  EXPECT_EQ(self_rs_ready[1].load(cuda::memory_order_relaxed), 1u);
  EXPECT_EQ(self_rs_ready[2].load(cuda::memory_order_relaxed), 1u);
  EXPECT_EQ(self_rs_ready[3].load(cuda::memory_order_relaxed), 1u);

  destroy_atomic_array(peer_rs_ready, kFlagsLen);
  destroy_atomic_array(self_rs_ready, kFlagsLen);

  peer_abort->~RingAllreduceSystemAtomicU32();
  self_error->~RingAllreduceSystemAtomicU32();
  self_abort->~RingAllreduceSystemAtomicU32();

  ASSERT_EQ(cudaFree(out_rs_ok), cudaSuccess);
  ASSERT_EQ(cudaFree(peer_data), cudaSuccess);
  ASSERT_EQ(cudaFree(self_data), cudaSuccess);

  ASSERT_EQ(cudaFree(peer_abort), cudaSuccess);
  ASSERT_EQ(cudaFree(peer_rs_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(self_rs_ready), cudaSuccess);

  ASSERT_EQ(cudaFree(self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self_abort), cudaSuccess);

  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(RingAllreduceWarpSpecializedSmemNegative, EC_NB_DISABLED_IMPL_TAG) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_warp_specialized_smem_negative_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

#if !CUTLASS_RING_ALLREDUCE_TEST_FORCE_NAMED_BARRIER_DISABLED
  GTEST_SKIP() << "requires CUTLASS_RING_ALLREDUCE_TEST_FORCE_NAMED_BARRIER_DISABLED=1";
#endif

  constexpr uint32_t N = 4;
  constexpr uint64_t kFlagsLen = N;

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;
  RingAllreduceSystemAtomicU32* self_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* peer_abort = nullptr;

  float* self_data = nullptr;
  float* peer_data = nullptr;
  uint32_t* out_rs_ok = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_rs_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_rs_ready), kFlagsLen * sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_data), N * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_data), N * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&out_rs_ok), sizeof(uint32_t)), cudaSuccess);

  new (self_abort) RingAllreduceSystemAtomicU32{};
  new (self_error) RingAllreduceSystemAtomicU32{};
  new (peer_abort) RingAllreduceSystemAtomicU32{};

  construct_atomic_array(self_rs_ready, kFlagsLen);
  construct_atomic_array(peer_rs_ready, kFlagsLen);

  self_abort->store(0u, cuda::memory_order_relaxed);
  self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  peer_abort->store(0u, cuda::memory_order_relaxed);

  for (uint64_t i = 0; i < kFlagsLen; ++i) {
    self_rs_ready[i].store(0u, cuda::memory_order_relaxed);
    peer_rs_ready[i].store(/*epoch=*/1u, cuda::memory_order_release);
  }

  for (uint32_t i = 0; i < N; ++i) {
    self_data[i] = static_cast<float>(i);
    peer_data[i] = static_cast<float>(i + 100);
  }

  out_rs_ok[0] = 0xFFFF'FFFFu;

  RingAllreduceParams<float, 8> p{};
  p.world_size = N;
  p.rank = 1;
  p.epoch = 1;

  p.count = N;
  p.num_channels = 1;
  p.tile_elems = 1;
  p.num_chunks_total = N;
  p.max_chunk_elems = 1;
  p.tiles_per_chunk = 1;
  p.num_tiles_total = 1;

  // Timeouts disabled (also fine for single-buffer). Tag must indicate fallback.
  p.timeout_iters = 0;
  p.timeout_cycles = 0;
  p.poll_sleep_start = 0;
  p.poll_sleep_ns = 0;

  p.self_data = self_data;
  p.self_rs_ready = self_rs_ready;
  p.self_abort = self_abort;
  p.self_error = self_error;

  p.peer_data[0] = peer_data;
  p.peer_rs_ready[0] = peer_rs_ready;
  p.peer_abort[0] = peer_abort;

  uint32_t zero = 0u;
  ASSERT_EQ(cudaMemcpyToSymbol(ring_allreduce_warp_specialized_smem_rs_impl_tag, &zero, sizeof(zero)), cudaSuccess);

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&done_event), cudaSuccess);

  ring_allreduce_warp_specialized_smem_negative_rs_kernel<<<1, 256, 0, stream>>>(p, /*N=*/N, /*r_u32=*/1, /*left=*/0, /*flags_len=*/kFlagsLen, out_rs_ok);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    std::fprintf(stderr, "ring_allreduce_warp_specialized_smem_negative_test: watchdog timeout (EC_NB_DISABLED_IMPL_TAG)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  uint32_t tag = 0u;
  ASSERT_EQ(cudaMemcpyFromSymbol(&tag, ring_allreduce_warp_specialized_smem_rs_impl_tag, sizeof(tag)), cudaSuccess);

  EXPECT_EQ(out_rs_ok[0], 1u);
  EXPECT_EQ(tag, kRingAllreduceWarpSpecializedSmemRsImplTagSingleBuffer);

  EXPECT_EQ(self_abort->load(cuda::memory_order_relaxed), 0u);
  EXPECT_EQ(self_error->load(cuda::memory_order_relaxed), static_cast<uint32_t>(RingAllreduceError::kOk));

  EXPECT_EQ(self_rs_ready[1].load(cuda::memory_order_relaxed), 1u);
  EXPECT_EQ(self_rs_ready[2].load(cuda::memory_order_relaxed), 1u);
  EXPECT_EQ(self_rs_ready[3].load(cuda::memory_order_relaxed), 1u);

  destroy_atomic_array(peer_rs_ready, kFlagsLen);
  destroy_atomic_array(self_rs_ready, kFlagsLen);

  peer_abort->~RingAllreduceSystemAtomicU32();
  self_error->~RingAllreduceSystemAtomicU32();
  self_abort->~RingAllreduceSystemAtomicU32();

  ASSERT_EQ(cudaFree(out_rs_ok), cudaSuccess);
  ASSERT_EQ(cudaFree(peer_data), cudaSuccess);
  ASSERT_EQ(cudaFree(self_data), cudaSuccess);

  ASSERT_EQ(cudaFree(peer_abort), cudaSuccess);
  ASSERT_EQ(cudaFree(peer_rs_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(self_rs_ready), cudaSuccess);

  ASSERT_EQ(cudaFree(self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self_abort), cudaSuccess);

  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
