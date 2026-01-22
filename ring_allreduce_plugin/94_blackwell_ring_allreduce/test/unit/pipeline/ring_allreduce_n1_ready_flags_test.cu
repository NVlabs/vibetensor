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
    \brief Unit test for N==1 per-tile self_ag_ready publication.
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/experimental/distributed/collective/ring_allreduce_kernel_sm100.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <new>

namespace {

using cutlass::distributed::collective::RingAllreduceDeviceAtomicU32;
using cutlass::distributed::collective::RingAllreduceError;
using cutlass::distributed::collective::RingAllreduceParams;
using cutlass::distributed::collective::RingAllreduceSystemAtomicU32;
using cutlass::distributed::collective::flag_index_u64;

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

__global__ void ring_allreduce_n1_ready_flags_probe_kernel() {}

static cudaError_t ring_allreduce_n1_ready_flags_probe_launch() {
  // Clear any pre-existing per-thread CUDA error state.
  (void)cudaGetLastError();

  ring_allreduce_n1_ready_flags_probe_kernel<<<1, 1>>>();
  return cudaGetLastError();
}

} // namespace

TEST(RingAllreduceN1ReadyFlags, N1F0_PublishPerTileAgReadyStep0) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_n1_ready_flags_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  constexpr int32_t kWorldSize = 1;
  constexpr int32_t kRank = 0;
  constexpr uint32_t kEpoch = 1u;

  constexpr uint32_t kNumTilesTotal = 4;
  constexpr int32_t kNumChannels = 1;

  constexpr uint32_t kTileElems = 256;
  constexpr uint32_t kTilesPerChunk = kNumTilesTotal;
  constexpr uint32_t kNumChunksTotal = static_cast<uint32_t>(kWorldSize) * static_cast<uint32_t>(kNumChannels);

  constexpr uint64_t kMaxChunkElems = uint64_t(kTileElems) * uint64_t(kTilesPerChunk);
  constexpr uint64_t kCount = kMaxChunkElems;

  constexpr uint32_t kFlagsLen = static_cast<uint32_t>(kWorldSize) * kNumTilesTotal;

  float* self_data = nullptr;
  RingAllreduceSystemAtomicU32* self_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* self_ag_ready = nullptr;

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;

  RingAllreduceDeviceAtomicU32* self_tiles_finished = nullptr;

  RingAllreduceSystemAtomicU32* self_barrier_gather_token = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_gather_status = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_release_token = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_release_status = nullptr;

  uint32_t* out_status = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_data), sizeof(float) * kCount), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_rs_ready), sizeof(RingAllreduceSystemAtomicU32) * kFlagsLen), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_ag_ready), sizeof(RingAllreduceSystemAtomicU32) * kFlagsLen), cudaSuccess);

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_tiles_finished), sizeof(RingAllreduceDeviceAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_barrier_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_barrier_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_barrier_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_barrier_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&out_status), sizeof(uint32_t)), cudaSuccess);

  // Begin lifetime for atomic objects (avoid C++ object-lifetime UB).
  for (uint32_t i = 0; i < kFlagsLen; ++i) {
    new (self_rs_ready + i) RingAllreduceSystemAtomicU32{};
    new (self_ag_ready + i) RingAllreduceSystemAtomicU32{};
  }
  new (self_abort) RingAllreduceSystemAtomicU32{};
  new (self_error) RingAllreduceSystemAtomicU32{};
  new (self_tiles_finished) RingAllreduceDeviceAtomicU32{};

  new (self_barrier_gather_token) RingAllreduceSystemAtomicU32{};
  new (self_barrier_gather_status) RingAllreduceSystemAtomicU32{};
  new (self_barrier_release_token) RingAllreduceSystemAtomicU32{};
  new (self_barrier_release_status) RingAllreduceSystemAtomicU32{};

  // Initialize payload.
  for (uint64_t i = 0; i < kCount; ++i) {
    self_data[i] = static_cast<float>(i);
  }

  // Initialize flags / status.
  for (uint32_t i = 0; i < kFlagsLen; ++i) {
    self_rs_ready[i].store(0u, cuda::memory_order_relaxed);
    self_ag_ready[i].store(0u, cuda::memory_order_relaxed);
  }

  self_abort->store(0u, cuda::memory_order_relaxed);
  self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);

  self_tiles_finished->store(0u, cuda::memory_order_relaxed);

  self_barrier_gather_token->store(0u, cuda::memory_order_relaxed);
  self_barrier_gather_status->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  self_barrier_release_token->store(0u, cuda::memory_order_relaxed);
  self_barrier_release_status->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);

  out_status[0] = 0xFFFF'FFFFu;

  RingAllreduceParams<float, 8> p{};

  p.world_size = kWorldSize;
  p.rank = kRank;
  p.epoch = kEpoch;

  p.count = kCount;
  p.num_channels = kNumChannels;

  p.tile_elems = kTileElems;
  p.num_chunks_total = kNumChunksTotal;
  p.max_chunk_elems = kMaxChunkElems;
  p.tiles_per_chunk = kTilesPerChunk;
  p.num_tiles_total = kNumTilesTotal;

  p.timeout_iters = 1u << 20;
  p.timeout_cycles = 0;

  p.poll_sleep_start = 0;
  p.poll_sleep_ns = 0;

  p.self_data = self_data;

  p.self_rs_ready = self_rs_ready;
  p.self_ag_ready = self_ag_ready;
  p.self_abort = self_abort;
  p.self_error = self_error;

  p.self_tiles_finished = self_tiles_finished;

  p.self_barrier_gather_token = self_barrier_gather_token;
  p.self_barrier_gather_status = self_barrier_gather_status;
  p.self_barrier_release_token = self_barrier_release_token;
  p.self_barrier_release_status = self_barrier_release_status;

  // Peer pointer tables can be null for world_size==1.
  for (int i = 0; i < 8; ++i) {
    p.peer_data[i] = nullptr;
    p.peer_rs_ready[i] = nullptr;
    p.peer_ag_ready[i] = nullptr;
    p.peer_abort[i] = nullptr;

    p.peer_barrier_gather_token[i] = nullptr;
    p.peer_barrier_gather_status[i] = nullptr;
    p.peer_barrier_release_token[i] = nullptr;
    p.peer_barrier_release_status[i] = nullptr;
  }

  // Debug/test hooks disabled.
  p.debug_abort_rank = 0u;
  p.debug_abort_ag_step = 0u;
  p.debug_abort_before_ag_publish = 0u;
  p.debug_abort_after_ag_publish = 0u;
  p.debug_release_delay_rank = 0u;
  p.debug_release_delay_iters = 0u;
  p.debug_jitter_seed = 0u;
  p.debug_jitter_max_iters = 0u;
  p.debug_jitter_mask = 0u;

  cutlass::distributed::collective::ring_allreduce_sm100<float><<<kNumTilesTotal, 256>>>(p, out_status);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(out_status[0], static_cast<uint32_t>(RingAllreduceError::kOk));

  // Validate per-tile step0 publication (idx == tile_linear for N==1).
  for (uint32_t tile_linear = 0; tile_linear < kNumTilesTotal; ++tile_linear) {
    uint64_t idx = flag_index_u64(/*step=*/0, tile_linear, kNumTilesTotal);
    uint32_t got = self_ag_ready[idx].load(cuda::memory_order_relaxed);
    EXPECT_EQ(got, kEpoch) << "tile_linear=" << tile_linear << " idx=" << idx;
  }

  ASSERT_EQ(self_abort->load(cuda::memory_order_relaxed), 0u);
  ASSERT_EQ(self_error->load(cuda::memory_order_relaxed), static_cast<uint32_t>(RingAllreduceError::kOk));

  ASSERT_EQ(self_tiles_finished->load(cuda::memory_order_relaxed), kNumTilesTotal);

  // Cleanup.
  for (uint32_t i = 0; i < kFlagsLen; ++i) {
    self_ag_ready[i].~RingAllreduceSystemAtomicU32();
    self_rs_ready[i].~RingAllreduceSystemAtomicU32();
  }

  self_barrier_release_status->~RingAllreduceSystemAtomicU32();
  self_barrier_release_token->~RingAllreduceSystemAtomicU32();
  self_barrier_gather_status->~RingAllreduceSystemAtomicU32();
  self_barrier_gather_token->~RingAllreduceSystemAtomicU32();

  self_tiles_finished->~RingAllreduceDeviceAtomicU32();
  self_error->~RingAllreduceSystemAtomicU32();
  self_abort->~RingAllreduceSystemAtomicU32();

  ASSERT_EQ(cudaFree(out_status), cudaSuccess);

  ASSERT_EQ(cudaFree(self_barrier_release_status), cudaSuccess);
  ASSERT_EQ(cudaFree(self_barrier_release_token), cudaSuccess);
  ASSERT_EQ(cudaFree(self_barrier_gather_status), cudaSuccess);
  ASSERT_EQ(cudaFree(self_barrier_gather_token), cudaSuccess);

  ASSERT_EQ(cudaFree(self_tiles_finished), cudaSuccess);
  ASSERT_EQ(cudaFree(self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self_abort), cudaSuccess);

  ASSERT_EQ(cudaFree(self_ag_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(self_rs_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(self_data), cudaSuccess);
}
