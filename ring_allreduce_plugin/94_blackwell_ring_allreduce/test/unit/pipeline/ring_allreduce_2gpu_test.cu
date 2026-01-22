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
    \brief L2 multi-GPU unit test harness for the ring_allreduce kernel.
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/experimental/distributed/collective/ring_allreduce_host.hpp"
#include "cutlass/experimental/distributed/collective/ring_allreduce_kernel_sm100.cuh"

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <thread>
#include <vector>

namespace {

using cutlass::distributed::collective::RingAllreduceError;
using cutlass::distributed::collective::RingAllreduceParams;
using cutlass::distributed::collective::RingAllreduceDeviceAtomicU32;
using cutlass::distributed::collective::RingAllreduceSystemAtomicU32;
using cutlass::distributed::collective::RingAllreduceTiling;
using cutlass::distributed::collective::validate_ring_allreduce_host_tiling;
using cutlass::distributed::collective::validate_ring_p2p_caps_and_enable_peer_access;

struct RingAllreduce2GpuAtomics {
  RingAllreduceSystemAtomicU32* self_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* self_ag_ready = nullptr;
  uint32_t flags_len = 0u;

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;

  RingAllreduceDeviceAtomicU32* self_tiles_finished = nullptr;

  RingAllreduceSystemAtomicU32* self_barrier_gather_token = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_gather_status = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_release_token = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_release_status = nullptr;
};

struct RingAllreduce2GpuAtomicsSnapshot {
  uint32_t abort = 0u;
  uint32_t error = 0u;

  uint32_t tiles_finished = 0u;

  uint32_t gather_token = 0u;
  uint32_t gather_status = 0u;

  uint32_t release_token = 0u;
  uint32_t release_status = 0u;
};

__global__ void construct_2gpu_atomics_kernel(RingAllreduce2GpuAtomics a) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.flags_len) {
    if (a.self_rs_ready) {
      new (a.self_rs_ready + idx) RingAllreduceSystemAtomicU32{};
    }
    if (a.self_ag_ready) {
      new (a.self_ag_ready + idx) RingAllreduceSystemAtomicU32{};
    }
  }

  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (a.self_abort) {
    new (a.self_abort) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_error) {
    new (a.self_error) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_tiles_finished) {
    new (a.self_tiles_finished) RingAllreduceDeviceAtomicU32{};
  }

  if (a.self_barrier_gather_token) {
    new (a.self_barrier_gather_token) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_barrier_gather_status) {
    new (a.self_barrier_gather_status) RingAllreduceSystemAtomicU32{};
  }

  if (a.self_barrier_release_token) {
    new (a.self_barrier_release_token) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_barrier_release_status) {
    new (a.self_barrier_release_status) RingAllreduceSystemAtomicU32{};
  }
}

__global__ void reset_2gpu_atomics_kernel(RingAllreduce2GpuAtomics a, uint32_t epoch) {
  // Tokens are reset to 0 between runs. Keep epoch in the signature for future
  // per-epoch reset logic.
  CUTLASS_UNUSED(epoch);

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.flags_len) {
    if (a.self_rs_ready) {
      a.self_rs_ready[idx].store(0u, cuda::memory_order_relaxed);
    }
    if (a.self_ag_ready) {
      a.self_ag_ready[idx].store(0u, cuda::memory_order_relaxed);
    }
  }

  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (a.self_abort) {
    a.self_abort->store(0u, cuda::memory_order_relaxed);
  }
  if (a.self_error) {
    a.self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  }
  if (a.self_tiles_finished) {
    a.self_tiles_finished->store(0u, cuda::memory_order_relaxed);
  }

  if (a.self_barrier_gather_token) {
    a.self_barrier_gather_token->store(0u, cuda::memory_order_relaxed);
  }
  if (a.self_barrier_gather_status) {
    a.self_barrier_gather_status->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  }

  if (a.self_barrier_release_token) {
    a.self_barrier_release_token->store(0u, cuda::memory_order_relaxed);
  }
  if (a.self_barrier_release_status) {
    a.self_barrier_release_status->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  }
}

__global__ void destroy_2gpu_atomics_kernel(RingAllreduce2GpuAtomics a) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.flags_len) {
    if (a.self_ag_ready) {
      a.self_ag_ready[idx].~RingAllreduceSystemAtomicU32();
    }
    if (a.self_rs_ready) {
      a.self_rs_ready[idx].~RingAllreduceSystemAtomicU32();
    }
  }

  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (a.self_barrier_release_status) {
    a.self_barrier_release_status->~RingAllreduceSystemAtomicU32();
  }
  if (a.self_barrier_release_token) {
    a.self_barrier_release_token->~RingAllreduceSystemAtomicU32();
  }

  if (a.self_barrier_gather_status) {
    a.self_barrier_gather_status->~RingAllreduceSystemAtomicU32();
  }
  if (a.self_barrier_gather_token) {
    a.self_barrier_gather_token->~RingAllreduceSystemAtomicU32();
  }

  if (a.self_tiles_finished) {
    a.self_tiles_finished->~RingAllreduceDeviceAtomicU32();
  }
  if (a.self_error) {
    a.self_error->~RingAllreduceSystemAtomicU32();
  }
  if (a.self_abort) {
    a.self_abort->~RingAllreduceSystemAtomicU32();
  }
}

__global__ void snapshot_2gpu_atomics_kernel(
    RingAllreduce2GpuAtomics a,
    RingAllreduce2GpuAtomicsSnapshot* out,
    uint32_t* rs_ready_out,
    uint32_t* ag_ready_out) {

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.flags_len) {
    if (rs_ready_out && a.self_rs_ready) {
      rs_ready_out[idx] = a.self_rs_ready[idx].load(cuda::memory_order_relaxed);
    }
    if (ag_ready_out && a.self_ag_ready) {
      ag_ready_out[idx] = a.self_ag_ready[idx].load(cuda::memory_order_relaxed);
    }
  }

  if (blockIdx.x != 0 || threadIdx.x != 0 || !out) {
    return;
  }

  RingAllreduce2GpuAtomicsSnapshot snap;

  snap.abort = a.self_abort ? a.self_abort->load(cuda::memory_order_relaxed) : 0u;
  snap.error = a.self_error ? a.self_error->load(cuda::memory_order_relaxed) : 0u;

  snap.tiles_finished = a.self_tiles_finished ? a.self_tiles_finished->load(cuda::memory_order_relaxed) : 0u;

  snap.gather_token = a.self_barrier_gather_token ? a.self_barrier_gather_token->load(cuda::memory_order_relaxed) : 0u;
  snap.gather_status = a.self_barrier_gather_status ? a.self_barrier_gather_status->load(cuda::memory_order_relaxed) : 0u;

  snap.release_token = a.self_barrier_release_token ? a.self_barrier_release_token->load(cuda::memory_order_relaxed) : 0u;
  snap.release_status = a.self_barrier_release_status ? a.self_barrier_release_status->load(cuda::memory_order_relaxed) : 0u;

  *out = snap;
}

static bool is_sm100_or_sm103(int device) {
  cudaDeviceProp prop{};
  cudaError_t st = cudaGetDeviceProperties(&prop, device);
  if (st != cudaSuccess) {
    return false;
  }

  int cc = prop.major * 10 + prop.minor;
  return cc == 100 || cc == 103;
}

__global__ void ring_allreduce_2gpu_probe_kernel() {}

static cudaError_t ring_allreduce_2gpu_probe_launch(int device) {
  // Clear any pre-existing per-thread CUDA error state.
  (void)cudaGetLastError();

  cudaError_t st = cudaSetDevice(device);
  if (st != cudaSuccess) {
    return st;
  }

  ring_allreduce_2gpu_probe_kernel<<<1, 1>>>();
  return cudaGetLastError();
}

} // namespace

CUTLASS_TEST_L2(RingAllreduce2Gpu, S0_HarnessAlive, {

  constexpr int kWorldSize = 2;
  constexpr uint32_t kEpoch = 1;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  // Pick SM100/SM103 devices that have a valid kernel image in this binary.
  std::vector<int> candidates;
  candidates.reserve(device_count);

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_2gpu_probe_launch(dev);
    if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
      continue;
    }
    ASSERT_EQ(probe_st, cudaSuccess);

    candidates.push_back(dev);
  }

  if (static_cast<int>(candidates.size()) < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " SM100/SM103 devices with a valid kernel image";
  }

  std::array<int, kWorldSize> devices{};
  bool found_pair = false;

  for (size_t i = 0; i < candidates.size() && !found_pair; ++i) {
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      devices[0] = candidates[i];
      devices[1] = candidates[j];

      auto p2p = validate_ring_p2p_caps_and_enable_peer_access(kWorldSize, devices.data());
      if (p2p.ok()) {
        found_pair = true;
        break;
      }
    }
  }

  if (!found_pair) {
    GTEST_SKIP() << "no suitable P2P pair with native peer atomics";
  }

  std::vector<uint32_t> host_status(kWorldSize, 0xffff'ffffu);
  std::vector<uint32_t*> device_status(kWorldSize, nullptr);

  std::vector<cudaStream_t> streams(kWorldSize);
  std::vector<cudaEvent_t> done_events(kWorldSize);

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&streams[i]), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&done_events[i], cudaEventDisableTiming), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status[i]), sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(device_status[i], 0xFF, sizeof(uint32_t), streams[i]), cudaSuccess);

    // M1 harness: intentionally pass invalid params (null pointers) to validate
    // end-to-end launch safety while the kernel is under development.
    RingAllreduceParams<uint8_t, 8> p{};

    p.world_size = kWorldSize;
    p.rank = i;
    p.epoch = kEpoch;

    // Minimal tiling to launch a single CTA.
    p.count = 1;
    p.num_channels = 1;
    p.tile_elems = 1;
    p.num_chunks_total = 2;
    p.max_chunk_elems = 1;
    p.tiles_per_chunk = 1;
    p.num_tiles_total = 1;

    // Safety-net timeouts (stub kernel should never loop indefinitely).
    p.timeout_iters = 1u << 18;
    p.timeout_cycles = 0;
    p.poll_sleep_start = 0;
    p.poll_sleep_ns = 0;

    p.debug_jitter_seed = 0u;
    p.debug_jitter_max_iters = 0u;
    p.debug_jitter_mask = 0u;

    cutlass::distributed::collective::ring_allreduce_sm100<uint8_t><<<1, 256, 0, streams[i]>>>(
        p,
        device_status[i]);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(done_events[i], streams[i]), cudaSuccess);
  }

  // Host watchdog: ensure no hangs even while the allreduce kernel is under development.
  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  std::vector<bool> done(kWorldSize, false);
  while (true) {
    bool all_done = true;

    for (int i = 0; i < kWorldSize; ++i) {
      if (done[i]) {
        continue;
      }

      all_done = false;

      ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
      cudaError_t q = cudaEventQuery(done_events[i]);
      if (q == cudaSuccess) {
        done[i] = true;
        continue;
      }
      if (q != cudaErrorNotReady) {
        // Unexpected failure (e.g., launch failure).
        ASSERT_EQ(q, cudaSuccess);
      }
    }

    if (all_done) {
      break;
    }

    if (std::chrono::steady_clock::now() > deadline) {
      std::fprintf(stderr, "ring_allreduce_2gpu_test: watchdog timeout (world_size=%d)\n", kWorldSize);
      std::abort();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&host_status[i], device_status[i], sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    EXPECT_EQ(host_status[i], static_cast<uint32_t>(RingAllreduceError::kInvalidParams));
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaFree(device_status[i]), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(done_events[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
  }
});

CUTLASS_TEST_L2(RingAllreduce2Gpu, S0_ArchGuardNoEarlyReturn, {

  constexpr int kWorldSize = 1;
  constexpr uint32_t kEpoch = 1;

  // Launch multiple CTAs so missing epilogue signals are observable.
  constexpr int kLaunchBlocks = 4;
  constexpr int kThreads = 256;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < 1) {
    GTEST_SKIP() << "requires >= 1 CUDA device";
  }

  // Pick an SM100/SM103 device that has a valid kernel image in this binary.
  int device = -1;
  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_2gpu_probe_launch(dev);
    if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
      continue;
    }
    ASSERT_EQ(probe_st, cudaSuccess);

    device = dev;
    break;
  }

  if (device < 0) {
    GTEST_SKIP() << "requires an SM100/SM103 device with a valid kernel image";
  }

  ASSERT_EQ(cudaSetDevice(device), cudaSuccess);

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming), cudaSuccess);

  constexpr uint32_t kNumTilesTotal = static_cast<uint32_t>(kLaunchBlocks);
  constexpr uint32_t kFlagsLen = kWorldSize * kNumTilesTotal;

  int blocks = static_cast<int>((kFlagsLen + kThreads - 1) / kThreads);
  // Ensure CTA0/thread0 runs scalar init even if flags_len==0.
  if (blocks == 0) {
    blocks = 1;
  }

  RingAllreduce2GpuAtomics atomics{};
  atomics.flags_len = kFlagsLen;

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_rs_ready), sizeof(RingAllreduceSystemAtomicU32) * kFlagsLen), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_ag_ready), sizeof(RingAllreduceSystemAtomicU32) * kFlagsLen), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_tiles_finished), sizeof(RingAllreduceDeviceAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_barrier_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_barrier_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_barrier_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_barrier_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  construct_2gpu_atomics_kernel<<<blocks, kThreads, 0, stream>>>(atomics);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  reset_2gpu_atomics_kernel<<<blocks, kThreads, 0, stream>>>(atomics, kEpoch);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  uint8_t* self_data = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&self_data), sizeof(uint8_t)), cudaSuccess);
  ASSERT_EQ(cudaMemset(self_data, 0, sizeof(uint8_t)), cudaSuccess);

  uint32_t* device_status = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status), sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemset(device_status, 0xFF, sizeof(uint32_t)), cudaSuccess);

  RingAllreduce2GpuAtomicsSnapshot* device_snap = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_snap), sizeof(RingAllreduce2GpuAtomicsSnapshot)), cudaSuccess);
  ASSERT_EQ(cudaMemset(device_snap, 0xFF, sizeof(RingAllreduce2GpuAtomicsSnapshot)), cudaSuccess);

  RingAllreduceParams<uint8_t, 8> p{};
  p.world_size = kWorldSize;
  p.rank = 0;
  p.epoch = kEpoch;

  // Minimal tiling to satisfy the validator (num_tiles_total == gridDim.x).
  p.count = 1;
  p.num_channels = 1;
  p.tile_elems = 1;
  p.num_chunks_total = kWorldSize;
  p.max_chunk_elems = 1;
  p.tiles_per_chunk = kNumTilesTotal;
  p.num_tiles_total = kNumTilesTotal;

  // Timeouts disabled: even in invalid-params paths, the epilogue must complete.
  p.timeout_iters = 0;
  p.timeout_cycles = 0;
  p.poll_sleep_start = 0;
  p.poll_sleep_ns = 0;

  p.self_data = self_data;

  p.self_rs_ready = atomics.self_rs_ready;
  p.self_ag_ready = atomics.self_ag_ready;
  p.self_abort = atomics.self_abort;
  p.self_error = atomics.self_error;
  p.self_tiles_finished = atomics.self_tiles_finished;

  p.self_barrier_gather_token = atomics.self_barrier_gather_token;
  p.self_barrier_gather_status = atomics.self_barrier_gather_status;
  p.self_barrier_release_token = atomics.self_barrier_release_token;
  p.self_barrier_release_status = atomics.self_barrier_release_status;

  p.debug_abort_rank = 0u;
  p.debug_abort_ag_step = 0u;
  p.debug_abort_before_ag_publish = 0u;
  p.debug_abort_after_ag_publish = 0u;

  p.debug_release_delay_rank = 0u;
  p.debug_release_delay_iters = 0u;

  p.debug_jitter_seed = 0u;
  p.debug_jitter_max_iters = 0u;
  p.debug_jitter_mask = 0u;

  // Force the arch guard to fail to validate that the epilogue still runs.
  cutlass::distributed::collective::ring_allreduce_sm100<uint8_t, true><<<kLaunchBlocks, kThreads, 0, stream>>>(
      p,
      device_status);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  snapshot_2gpu_atomics_kernel<<<blocks, kThreads, 0, stream>>>(atomics, device_snap, nullptr, nullptr);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  while (true) {
    cudaError_t q = cudaEventQuery(done_event);
    if (q == cudaSuccess) {
      break;
    }
    if (q != cudaErrorNotReady) {
      ASSERT_EQ(q, cudaSuccess);
    }
    if (std::chrono::steady_clock::now() > deadline) {
      std::fprintf(stderr, "ring_allreduce_2gpu_test: watchdog timeout (S0_ArchGuardNoEarlyReturn)\n");
      std::abort();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  uint32_t host_status = 0u;
  RingAllreduce2GpuAtomicsSnapshot host_snap{};

  ASSERT_EQ(cudaMemcpy(&host_status, device_status, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&host_snap, device_snap, sizeof(RingAllreduce2GpuAtomicsSnapshot), cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_EQ(host_status, static_cast<uint32_t>(RingAllreduceError::kInvalidParams));

  EXPECT_EQ(host_snap.error, static_cast<uint32_t>(RingAllreduceError::kInvalidParams));
  EXPECT_NE(host_snap.abort, 0u);

  EXPECT_EQ(host_snap.tiles_finished, static_cast<uint32_t>(kLaunchBlocks));

  EXPECT_EQ(host_snap.gather_token, kEpoch);
  EXPECT_EQ(host_snap.gather_status, static_cast<uint32_t>(RingAllreduceError::kInvalidParams));

  EXPECT_EQ(host_snap.release_token, kEpoch);
  EXPECT_EQ(host_snap.release_status, static_cast<uint32_t>(RingAllreduceError::kInvalidParams));

  destroy_2gpu_atomics_kernel<<<blocks, kThreads, 0, stream>>>(atomics);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  ASSERT_EQ(cudaFree(device_snap), cudaSuccess);
  ASSERT_EQ(cudaFree(device_status), cudaSuccess);
  ASSERT_EQ(cudaFree(self_data), cudaSuccess);

  ASSERT_EQ(cudaFree(atomics.self_barrier_release_status), cudaSuccess);
  ASSERT_EQ(cudaFree(atomics.self_barrier_release_token), cudaSuccess);
  ASSERT_EQ(cudaFree(atomics.self_barrier_gather_status), cudaSuccess);
  ASSERT_EQ(cudaFree(atomics.self_barrier_gather_token), cudaSuccess);

  ASSERT_EQ(cudaFree(atomics.self_tiles_finished), cudaSuccess);
  ASSERT_EQ(cudaFree(atomics.self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(atomics.self_abort), cudaSuccess);

  ASSERT_EQ(cudaFree(atomics.self_ag_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(atomics.self_rs_ready), cudaSuccess);

  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
});

CUTLASS_TEST_L2(RingAllreduce2Gpu, S1_AtomicsConstructReset, {

  constexpr int kWorldSize = 2;
  constexpr uint32_t kEpoch = 1;

  // Note: This test is local (it does not dereference peer pointers), but we
  // keep the full 2-GPU + P2P/native-atomics runtime gating because these
  // atomics are ultimately used for cross-device synchronization in the
  // ring_allreduce kernel.

  // Use a non-trivial length to exercise multi-block init/reset kernels.
  constexpr uint32_t kNumTilesTotal = 257;
  constexpr uint32_t kFlagsLen = kWorldSize * kNumTilesTotal;

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((kFlagsLen + kThreads - 1) / kThreads);
  // Ensure CTA0/thread0 runs scalar init even if flags_len==0.
  if (blocks == 0) {
    blocks = 1;
  }

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  // Pick SM100/SM103 devices that have a valid kernel image in this binary.
  std::vector<int> candidates;
  candidates.reserve(device_count);

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_2gpu_probe_launch(dev);
    if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
      continue;
    }
    ASSERT_EQ(probe_st, cudaSuccess);

    candidates.push_back(dev);
  }

  if (static_cast<int>(candidates.size()) < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " SM100/SM103 devices with a valid kernel image";
  }

  std::array<int, kWorldSize> devices{};
  bool found_pair = false;

  for (size_t i = 0; i < candidates.size() && !found_pair; ++i) {
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      devices[0] = candidates[i];
      devices[1] = candidates[j];

      auto p2p = validate_ring_p2p_caps_and_enable_peer_access(kWorldSize, devices.data());
      if (p2p.ok()) {
        found_pair = true;
        break;
      }
    }
  }

  if (!found_pair) {
    GTEST_SKIP() << "no suitable P2P pair with native peer atomics";
  }

  std::array<RingAllreduce2GpuAtomics, kWorldSize> atomics{};
  std::array<RingAllreduce2GpuAtomicsSnapshot, kWorldSize> host_snap{};

  std::vector<uint32_t> host_rs_ready(kFlagsLen, 0xffff'ffffu);
  std::vector<uint32_t> host_ag_ready(kFlagsLen, 0xffff'ffffu);

  std::vector<cudaStream_t> streams(kWorldSize);

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&streams[i]), cudaSuccess);

    atomics[i].flags_len = kFlagsLen;

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_rs_ready), sizeof(RingAllreduceSystemAtomicU32) * kFlagsLen), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_ag_ready), sizeof(RingAllreduceSystemAtomicU32) * kFlagsLen), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_tiles_finished), sizeof(RingAllreduceDeviceAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    construct_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    reset_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[i]>>>(atomics[i], kEpoch);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    RingAllreduce2GpuAtomicsSnapshot* device_snap = nullptr;
    uint32_t* device_rs_ready = nullptr;
    uint32_t* device_ag_ready = nullptr;

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_snap), sizeof(RingAllreduce2GpuAtomicsSnapshot)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_rs_ready), sizeof(uint32_t) * kFlagsLen), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_ag_ready), sizeof(uint32_t) * kFlagsLen), cudaSuccess);

    ASSERT_EQ(cudaMemsetAsync(device_snap, 0xFF, sizeof(RingAllreduce2GpuAtomicsSnapshot), streams[i]), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(device_rs_ready, 0xFF, sizeof(uint32_t) * kFlagsLen, streams[i]), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(device_ag_ready, 0xFF, sizeof(uint32_t) * kFlagsLen, streams[i]), cudaSuccess);

    snapshot_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[i]>>>(
        atomics[i],
        device_snap,
        device_rs_ready,
        device_ag_ready);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&host_snap[i], device_snap, sizeof(RingAllreduce2GpuAtomicsSnapshot), cudaMemcpyDeviceToHost), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(host_rs_ready.data(), device_rs_ready, sizeof(uint32_t) * kFlagsLen, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_ag_ready.data(), device_ag_ready, sizeof(uint32_t) * kFlagsLen, cudaMemcpyDeviceToHost), cudaSuccess);

    ASSERT_EQ(cudaFree(device_ag_ready), cudaSuccess);
    ASSERT_EQ(cudaFree(device_rs_ready), cudaSuccess);
    ASSERT_EQ(cudaFree(device_snap), cudaSuccess);

    for (uint32_t v : host_rs_ready) {
      EXPECT_EQ(v, 0u);
    }
    for (uint32_t v : host_ag_ready) {
      EXPECT_EQ(v, 0u);
    }

    EXPECT_EQ(host_snap[i].abort, 0u);
    EXPECT_EQ(host_snap[i].error, static_cast<uint32_t>(RingAllreduceError::kOk));
    EXPECT_EQ(host_snap[i].tiles_finished, 0u);

    EXPECT_EQ(host_snap[i].gather_token, 0u);
    EXPECT_EQ(host_snap[i].gather_status, static_cast<uint32_t>(RingAllreduceError::kOk));
    EXPECT_EQ(host_snap[i].release_token, 0u);
    EXPECT_EQ(host_snap[i].release_status, static_cast<uint32_t>(RingAllreduceError::kOk));
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    destroy_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_barrier_release_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_barrier_release_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_barrier_gather_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_barrier_gather_token), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_tiles_finished), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_error), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_abort), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_ag_ready), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_rs_ready), cudaSuccess);

    ASSERT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
  }
});

static void run_ring_allreduce_2gpu_correctness_test(uint64_t count, int32_t num_channels, uint32_t tile_elems) {

  constexpr int kWorldSize = 2;
  constexpr uint32_t kEpoch = 1;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  // Pick SM100/SM103 devices that have a valid kernel image in this binary.
  std::vector<int> candidates;
  candidates.reserve(device_count);

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_2gpu_probe_launch(dev);
    if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
      continue;
    }
    ASSERT_EQ(probe_st, cudaSuccess);

    candidates.push_back(dev);
  }

  if (static_cast<int>(candidates.size()) < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " SM100/SM103 devices with a valid kernel image";
  }

  std::array<int, kWorldSize> devices{};
  bool found_pair = false;

  for (size_t i = 0; i < candidates.size() && !found_pair; ++i) {
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      devices[0] = candidates[i];
      devices[1] = candidates[j];

      auto p2p = validate_ring_p2p_caps_and_enable_peer_access(kWorldSize, devices.data());
      if (p2p.ok()) {
        found_pair = true;
        break;
      }
    }
  }

  if (!found_pair) {
    GTEST_SKIP() << "no suitable P2P pair with native peer atomics";
  }

  RingAllreduceTiling tiling{};
  auto tiling_r = validate_ring_allreduce_host_tiling(
      count,
      kWorldSize,
      num_channels,
      tile_elems,
      &tiling,
      devices.data());
  ASSERT_TRUE(tiling_r.ok()) << (tiling_r.error_reason ? tiling_r.error_reason : "tiling validation failed");

  uint32_t flags_len = static_cast<uint32_t>(kWorldSize) * tiling.num_tiles_total;

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((flags_len + kThreads - 1) / kThreads);
  // Ensure CTA0/thread0 runs scalar init even if flags_len==0.
  if (blocks == 0) {
    blocks = 1;
  }

  std::array<RingAllreduce2GpuAtomics, kWorldSize> atomics{};

  std::array<float*, kWorldSize> device_data{};
  std::array<uint32_t*, kWorldSize> device_status{};
  std::array<uint32_t, kWorldSize> host_status{};

  std::array<cudaStream_t, kWorldSize> streams{};
  std::array<cudaEvent_t, kWorldSize> done_events{};

  std::vector<float> host_in0(static_cast<size_t>(count));
  std::vector<float> host_in1(static_cast<size_t>(count));

  for (uint64_t i = 0; i < count; ++i) {
    host_in0[i] = static_cast<float>(i);
    host_in1[i] = static_cast<float>(2.0f * static_cast<float>(i));
  }

  std::array<std::vector<float>, kWorldSize> host_out;
  host_out[0].resize(static_cast<size_t>(count));
  host_out[1].resize(static_cast<size_t>(count));

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    ASSERT_EQ(cudaStreamCreate(&streams[i]), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&done_events[i], cudaEventDisableTiming), cudaSuccess);

    atomics[i].flags_len = flags_len;

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_data[i]), sizeof(float) * count), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status[i]), sizeof(uint32_t)), cudaSuccess);

    ASSERT_EQ(cudaMemsetAsync(device_status[i], 0xFF, sizeof(uint32_t), streams[i]), cudaSuccess);

    auto const& host_in = (i == 0) ? host_in0 : host_in1;
    ASSERT_EQ(cudaMemcpyAsync(device_data[i], host_in.data(), sizeof(float) * count, cudaMemcpyHostToDevice, streams[i]), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_rs_ready), sizeof(RingAllreduceSystemAtomicU32) * flags_len), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_ag_ready), sizeof(RingAllreduceSystemAtomicU32) * flags_len), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_tiles_finished), sizeof(RingAllreduceDeviceAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    construct_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    reset_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[i]>>>(atomics[i], kEpoch);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  // Ensure per-rank initialization is complete before any cross-device accesses.
  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    RingAllreduceParams<float, 8> p{};

    p.world_size = kWorldSize;
    p.rank = i;
    p.epoch = kEpoch;

    p.count = count;
    p.num_channels = num_channels;

    p.tile_elems = tiling.tile_elems;
    p.num_chunks_total = tiling.num_chunks_total;
    p.max_chunk_elems = tiling.max_chunk_elems;
    p.tiles_per_chunk = tiling.tiles_per_chunk;
    p.num_tiles_total = tiling.num_tiles_total;

    // Hang-resistant defaults for development/CI safety.
    p.timeout_iters = 1u << 18;
    p.timeout_cycles = 0;
    p.poll_sleep_start = 0;
    p.poll_sleep_ns = 0;

    p.self_data = device_data[i];

    p.self_rs_ready = atomics[i].self_rs_ready;
    p.self_ag_ready = atomics[i].self_ag_ready;
    p.self_abort = atomics[i].self_abort;
    p.self_error = atomics[i].self_error;

    p.self_tiles_finished = atomics[i].self_tiles_finished;

    p.self_barrier_gather_token = atomics[i].self_barrier_gather_token;
    p.self_barrier_gather_status = atomics[i].self_barrier_gather_status;
    p.self_barrier_release_token = atomics[i].self_barrier_release_token;
    p.self_barrier_release_status = atomics[i].self_barrier_release_status;

    for (int peer = 0; peer < kWorldSize; ++peer) {
      p.peer_data[peer] = device_data[peer];
      p.peer_rs_ready[peer] = atomics[peer].self_rs_ready;
      p.peer_ag_ready[peer] = atomics[peer].self_ag_ready;
      p.peer_abort[peer] = atomics[peer].self_abort;

      p.peer_barrier_gather_token[peer] = atomics[peer].self_barrier_gather_token;
      p.peer_barrier_gather_status[peer] = atomics[peer].self_barrier_gather_status;
      p.peer_barrier_release_token[peer] = atomics[peer].self_barrier_release_token;
      p.peer_barrier_release_status[peer] = atomics[peer].self_barrier_release_status;
    }

    // No abort injection for correctness tests.
    p.debug_abort_rank = 0xffff'ffffu;
    p.debug_abort_ag_step = 0u;
    p.debug_abort_before_ag_publish = 0u;
    p.debug_abort_after_ag_publish = 0u;

    p.debug_release_delay_rank = 0xffff'ffffu;
    p.debug_release_delay_iters = 0u;

    p.debug_jitter_seed = 0u;
    p.debug_jitter_max_iters = 0u;
    p.debug_jitter_mask = 0u;

    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    cutlass::distributed::collective::ring_allreduce_sm100<float><<<tiling.num_tiles_total, 256, 0, streams[i]>>>(
        p,
        device_status[i]);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(done_events[i], streams[i]), cudaSuccess);
  }

  // Host watchdog: ensure no hangs.
  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  std::vector<bool> done(kWorldSize, false);
  while (true) {
    bool all_done = true;

    for (int i = 0; i < kWorldSize; ++i) {
      if (done[i]) {
        continue;
      }

      all_done = false;

      ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
      cudaError_t q = cudaEventQuery(done_events[i]);
      if (q == cudaSuccess) {
        done[i] = true;
        continue;
      }
      if (q != cudaErrorNotReady) {
        ASSERT_EQ(q, cudaSuccess);
      }
    }

    if (all_done) {
      break;
    }

    if (std::chrono::steady_clock::now() > deadline) {
      std::fprintf(stderr, "ring_allreduce_2gpu_test: watchdog timeout (world_size=%d)\n", kWorldSize);
      std::abort();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&host_status[i], device_status[i], sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    EXPECT_EQ(host_status[i], static_cast<uint32_t>(RingAllreduceError::kOk));
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_out[i].data(), device_data[i], sizeof(float) * count, cudaMemcpyDeviceToHost), cudaSuccess);
  }

  for (uint64_t i = 0; i < count; ++i) {
    float expected = static_cast<float>(3.0f * static_cast<float>(i));
    EXPECT_FLOAT_EQ(host_out[0][static_cast<size_t>(i)], expected);
    EXPECT_FLOAT_EQ(host_out[1][static_cast<size_t>(i)], expected);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    destroy_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_barrier_release_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_barrier_release_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_barrier_gather_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_barrier_gather_token), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_tiles_finished), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_error), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_abort), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_ag_ready), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_rs_ready), cudaSuccess);

    ASSERT_EQ(cudaFree(device_status[i]), cudaSuccess);
    ASSERT_EQ(cudaFree(device_data[i]), cudaSuccess);

    ASSERT_EQ(cudaEventDestroy(done_events[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
  }
}

CUTLASS_TEST_L2(RingAllreduce2Gpu, R0_Correctness, {

  run_ring_allreduce_2gpu_correctness_test(/*count=*/1024, /*num_channels=*/1, /*tile_elems=*/256);
});

CUTLASS_TEST_L2(RingAllreduce2Gpu, R2_Correctness_Small, {

  constexpr int kWorldSize = 2;
  constexpr int32_t kNumChannels = 2;
  constexpr uint32_t kTileElems = 256;

  constexpr uint64_t kCount =
      uint64_t(kWorldSize) * uint64_t(kNumChannels) * uint64_t(kTileElems) * 1ull + 13ull;

  run_ring_allreduce_2gpu_correctness_test(kCount, kNumChannels, kTileElems);
});

CUTLASS_TEST_L2(RingAllreduce2Gpu, R3_Correctness_Large, {

  constexpr int kWorldSize = 2;
  constexpr int32_t kNumChannels = 2;
  constexpr uint32_t kTileElems = 256;

  constexpr uint64_t kCount =
      uint64_t(kWorldSize) * uint64_t(kNumChannels) * uint64_t(kTileElems) * 16ull + 13ull;

  run_ring_allreduce_2gpu_correctness_test(kCount, kNumChannels, kTileElems);
});

CUTLASS_TEST_L2(RingAllreduce2Gpu, R1_AbortAfterAgPublish, {

  constexpr int kWorldSize = 2;
  constexpr uint32_t kEpoch = 1;

  constexpr uint64_t kCount = 1024;
  constexpr int32_t kNumChannels = 1;
  constexpr uint32_t kTileElems = 256;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  // Pick SM100/SM103 devices that have a valid kernel image in this binary.
  std::vector<int> candidates;
  candidates.reserve(device_count);

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_2gpu_probe_launch(dev);
    if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
      continue;
    }
    ASSERT_EQ(probe_st, cudaSuccess);

    candidates.push_back(dev);
  }

  if (static_cast<int>(candidates.size()) < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " SM100/SM103 devices with a valid kernel image";
  }

  std::array<int, kWorldSize> devices{};
  bool found_pair = false;

  for (size_t i = 0; i < candidates.size() && !found_pair; ++i) {
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      devices[0] = candidates[i];
      devices[1] = candidates[j];

      auto p2p = validate_ring_p2p_caps_and_enable_peer_access(kWorldSize, devices.data());
      if (p2p.ok()) {
        found_pair = true;
        break;
      }
    }
  }

  if (!found_pair) {
    GTEST_SKIP() << "no suitable P2P pair with native peer atomics";
  }

  RingAllreduceTiling tiling{};
  auto tiling_r = validate_ring_allreduce_host_tiling(
      kCount,
      kWorldSize,
      kNumChannels,
      kTileElems,
      &tiling,
      devices.data());
  ASSERT_TRUE(tiling_r.ok()) << (tiling_r.error_reason ? tiling_r.error_reason : "tiling validation failed");

  uint32_t flags_len = static_cast<uint32_t>(kWorldSize) * tiling.num_tiles_total;

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((flags_len + kThreads - 1) / kThreads);
  // Ensure CTA0/thread0 runs scalar init even if flags_len==0.
  if (blocks == 0) {
    blocks = 1;
  }

  std::array<RingAllreduce2GpuAtomics, kWorldSize> atomics{};

  std::array<float*, kWorldSize> device_data{};
  std::array<uint32_t*, kWorldSize> device_status{};
  std::array<uint32_t, kWorldSize> host_status{};

  std::array<cudaStream_t, kWorldSize> streams{};
  std::array<cudaEvent_t, kWorldSize> done_events{};

  std::vector<float> host_in0(kCount);
  std::vector<float> host_in1(kCount);

  for (uint64_t i = 0; i < kCount; ++i) {
    host_in0[i] = static_cast<float>(i);
    host_in1[i] = static_cast<float>(2.0f * static_cast<float>(i));
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    ASSERT_EQ(cudaStreamCreate(&streams[i]), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&done_events[i], cudaEventDisableTiming), cudaSuccess);

    atomics[i].flags_len = flags_len;

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_data[i]), sizeof(float) * kCount), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status[i]), sizeof(uint32_t)), cudaSuccess);

    ASSERT_EQ(cudaMemsetAsync(device_status[i], 0xFF, sizeof(uint32_t), streams[i]), cudaSuccess);

    auto const& host_in = (i == 0) ? host_in0 : host_in1;
    ASSERT_EQ(cudaMemcpyAsync(device_data[i], host_in.data(), sizeof(float) * kCount, cudaMemcpyHostToDevice, streams[i]), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_rs_ready), sizeof(RingAllreduceSystemAtomicU32) * flags_len), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_ag_ready), sizeof(RingAllreduceSystemAtomicU32) * flags_len), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_tiles_finished), sizeof(RingAllreduceDeviceAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    construct_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    reset_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[i]>>>(atomics[i], kEpoch);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  // Ensure per-rank initialization is complete before any cross-device accesses.
  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    RingAllreduceParams<float, 8> p{};

    p.world_size = kWorldSize;
    p.rank = i;
    p.epoch = kEpoch;

    p.count = kCount;
    p.num_channels = kNumChannels;

    p.tile_elems = tiling.tile_elems;
    p.num_chunks_total = tiling.num_chunks_total;
    p.max_chunk_elems = tiling.max_chunk_elems;
    p.tiles_per_chunk = tiling.tiles_per_chunk;
    p.num_tiles_total = tiling.num_tiles_total;

    // Hang-resistant defaults for development/CI safety.
    p.timeout_iters = 1u << 18;
    p.timeout_cycles = 0;
    p.poll_sleep_start = 0;
    p.poll_sleep_ns = 0;

    p.self_data = device_data[i];

    p.self_rs_ready = atomics[i].self_rs_ready;
    p.self_ag_ready = atomics[i].self_ag_ready;
    p.self_abort = atomics[i].self_abort;
    p.self_error = atomics[i].self_error;

    p.self_tiles_finished = atomics[i].self_tiles_finished;

    p.self_barrier_gather_token = atomics[i].self_barrier_gather_token;
    p.self_barrier_gather_status = atomics[i].self_barrier_gather_status;
    p.self_barrier_release_token = atomics[i].self_barrier_release_token;
    p.self_barrier_release_status = atomics[i].self_barrier_release_status;

    for (int peer = 0; peer < kWorldSize; ++peer) {
      p.peer_data[peer] = device_data[peer];
      p.peer_rs_ready[peer] = atomics[peer].self_rs_ready;
      p.peer_ag_ready[peer] = atomics[peer].self_ag_ready;
      p.peer_abort[peer] = atomics[peer].self_abort;

      p.peer_barrier_gather_token[peer] = atomics[peer].self_barrier_gather_token;
      p.peer_barrier_gather_status[peer] = atomics[peer].self_barrier_gather_status;
      p.peer_barrier_release_token[peer] = atomics[peer].self_barrier_release_token;
      p.peer_barrier_release_status[peer] = atomics[peer].self_barrier_release_status;
    }

    // Inject abort-after-publish at the final AG step (world_size==2 => s==0) on rank0.
    p.debug_abort_rank = 0u;
    p.debug_abort_ag_step = 0u;
    p.debug_abort_before_ag_publish = 0u;
    p.debug_abort_after_ag_publish = 1u;

    p.debug_release_delay_rank = 0xffff'ffffu;
    p.debug_release_delay_iters = 0u;

    p.debug_jitter_seed = 0u;
    p.debug_jitter_max_iters = 0u;
    p.debug_jitter_mask = 0u;

    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    cutlass::distributed::collective::ring_allreduce_sm100<float><<<tiling.num_tiles_total, 256, 0, streams[i]>>>(
        p,
        device_status[i]);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(done_events[i], streams[i]), cudaSuccess);
  }

  // Host watchdog: ensure no hangs.
  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  std::vector<bool> done(kWorldSize, false);
  while (true) {
    bool all_done = true;

    for (int i = 0; i < kWorldSize; ++i) {
      if (done[i]) {
        continue;
      }

      all_done = false;

      ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
      cudaError_t q = cudaEventQuery(done_events[i]);
      if (q == cudaSuccess) {
        done[i] = true;
        continue;
      }
      if (q != cudaErrorNotReady) {
        ASSERT_EQ(q, cudaSuccess);
      }
    }

    if (all_done) {
      break;
    }

    if (std::chrono::steady_clock::now() > deadline) {
      std::fprintf(stderr, "ring_allreduce_2gpu_test: watchdog timeout (world_size=%d)\n", kWorldSize);
      std::abort();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&host_status[i], device_status[i], sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    EXPECT_EQ(host_status[i], static_cast<uint32_t>(RingAllreduceError::kAbortObserved));
  }

  // The R1 injection is "abort-after-publish" at the AG forwarding publish site.
  // Ensure CTA0 published its AG forwarding flag before the abort.
  {
    std::vector<uint32_t> host_ag_ready(flags_len, 0xffff'ffffu);
    uint32_t* device_ag_ready_out = nullptr;

    ASSERT_EQ(cudaSetDevice(devices[0]), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_ag_ready_out), sizeof(uint32_t) * flags_len), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(device_ag_ready_out, 0xFF, sizeof(uint32_t) * flags_len, streams[0]), cudaSuccess);

    snapshot_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[0]>>>(
        atomics[0],
        /*out=*/nullptr,
        /*rs_ready_out=*/nullptr,
        /*ag_ready_out=*/device_ag_ready_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[0]), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(host_ag_ready.data(), device_ag_ready_out, sizeof(uint32_t) * flags_len, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(device_ag_ready_out), cudaSuccess);

    uint32_t ag_fwd_idx = tiling.num_tiles_total; // step=1, tile_linear=0
    EXPECT_EQ(host_ag_ready[ag_fwd_idx], kEpoch);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    destroy_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_barrier_release_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_barrier_release_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_barrier_gather_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_barrier_gather_token), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_tiles_finished), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_error), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_abort), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_ag_ready), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_rs_ready), cudaSuccess);

    ASSERT_EQ(cudaFree(device_status[i]), cudaSuccess);
    ASSERT_EQ(cudaFree(device_data[i]), cudaSuccess);

    ASSERT_EQ(cudaEventDestroy(done_events[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
  }
});

CUTLASS_TEST_L2(RingAllreduce2Gpu, R2_NullPeerAgReadyInvalidParams, {

  constexpr int kWorldSize = 2;
  constexpr uint32_t kEpoch = 1;

  constexpr uint64_t kCount = 1024;
  constexpr int32_t kNumChannels = 1;
  constexpr uint32_t kTileElems = 256;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  // Pick SM100/SM103 devices that have a valid kernel image in this binary.
  std::vector<int> candidates;
  candidates.reserve(device_count);

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_2gpu_probe_launch(dev);
    if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
      continue;
    }
    ASSERT_EQ(probe_st, cudaSuccess);

    candidates.push_back(dev);
  }

  if (static_cast<int>(candidates.size()) < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " SM100/SM103 devices with a valid kernel image";
  }

  std::array<int, kWorldSize> devices{};
  bool found_pair = false;

  for (size_t i = 0; i < candidates.size() && !found_pair; ++i) {
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      devices[0] = candidates[i];
      devices[1] = candidates[j];

      auto p2p = validate_ring_p2p_caps_and_enable_peer_access(kWorldSize, devices.data());
      if (p2p.ok()) {
        found_pair = true;
        break;
      }
    }
  }

  if (!found_pair) {
    GTEST_SKIP() << "no suitable P2P pair with native peer atomics";
  }

  RingAllreduceTiling tiling{};
  auto tiling_r = validate_ring_allreduce_host_tiling(
      kCount,
      kWorldSize,
      kNumChannels,
      kTileElems,
      &tiling,
      devices.data());
  ASSERT_TRUE(tiling_r.ok()) << (tiling_r.error_reason ? tiling_r.error_reason : "tiling validation failed");

  uint32_t flags_len = static_cast<uint32_t>(kWorldSize) * tiling.num_tiles_total;

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((flags_len + kThreads - 1) / kThreads);
  // Ensure CTA0/thread0 runs scalar init even if flags_len==0.
  if (blocks == 0) {
    blocks = 1;
  }

  std::array<RingAllreduce2GpuAtomics, kWorldSize> atomics{};

  std::array<float*, kWorldSize> device_data{};
  std::array<uint32_t*, kWorldSize> device_status{};
  std::array<uint32_t, kWorldSize> host_status{};

  std::array<cudaStream_t, kWorldSize> streams{};
  std::array<cudaEvent_t, kWorldSize> done_events{};

  std::vector<float> host_in0(kCount);
  std::vector<float> host_in1(kCount);

  for (uint64_t i = 0; i < kCount; ++i) {
    host_in0[i] = static_cast<float>(i);
    host_in1[i] = static_cast<float>(2.0f * static_cast<float>(i));
  }

  std::array<std::vector<float>, kWorldSize> host_out;
  host_out[0].resize(kCount);
  host_out[1].resize(kCount);

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    ASSERT_EQ(cudaStreamCreate(&streams[i]), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&done_events[i], cudaEventDisableTiming), cudaSuccess);

    atomics[i].flags_len = flags_len;

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_data[i]), sizeof(float) * kCount), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status[i]), sizeof(uint32_t)), cudaSuccess);

    ASSERT_EQ(cudaMemsetAsync(device_status[i], 0xFF, sizeof(uint32_t), streams[i]), cudaSuccess);

    auto const& host_in = (i == 0) ? host_in0 : host_in1;
    ASSERT_EQ(cudaMemcpyAsync(device_data[i], host_in.data(), sizeof(float) * kCount, cudaMemcpyHostToDevice, streams[i]), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_rs_ready), sizeof(RingAllreduceSystemAtomicU32) * flags_len), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_ag_ready), sizeof(RingAllreduceSystemAtomicU32) * flags_len), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_tiles_finished), sizeof(RingAllreduceDeviceAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_barrier_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    construct_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    reset_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[i]>>>(atomics[i], kEpoch);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  // Ensure per-rank initialization is complete before any cross-device accesses.
  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    RingAllreduceParams<float, 8> p{};

    p.world_size = kWorldSize;
    p.rank = i;
    p.epoch = kEpoch;

    p.count = kCount;
    p.num_channels = kNumChannels;

    p.tile_elems = tiling.tile_elems;
    p.num_chunks_total = tiling.num_chunks_total;
    p.max_chunk_elems = tiling.max_chunk_elems;
    p.tiles_per_chunk = tiling.tiles_per_chunk;
    p.num_tiles_total = tiling.num_tiles_total;

    // M4 liveness requirement: no hangs even with timeouts disabled.
    p.timeout_iters = 0;
    p.timeout_cycles = 0;
    p.poll_sleep_start = 0;
    p.poll_sleep_ns = 0;

    p.self_data = device_data[i];

    p.self_rs_ready = atomics[i].self_rs_ready;
    p.self_ag_ready = atomics[i].self_ag_ready;
    p.self_abort = atomics[i].self_abort;
    p.self_error = atomics[i].self_error;

    p.self_tiles_finished = atomics[i].self_tiles_finished;

    p.self_barrier_gather_token = atomics[i].self_barrier_gather_token;
    p.self_barrier_gather_status = atomics[i].self_barrier_gather_status;
    p.self_barrier_release_token = atomics[i].self_barrier_release_token;
    p.self_barrier_release_status = atomics[i].self_barrier_release_status;

    for (int peer = 0; peer < kWorldSize; ++peer) {
      p.peer_data[peer] = device_data[peer];
      p.peer_rs_ready[peer] = atomics[peer].self_rs_ready;
      p.peer_ag_ready[peer] = atomics[peer].self_ag_ready;
      p.peer_abort[peer] = atomics[peer].self_abort;

      p.peer_barrier_gather_token[peer] = atomics[peer].self_barrier_gather_token;
      p.peer_barrier_gather_status[peer] = atomics[peer].self_barrier_gather_status;
      p.peer_barrier_release_token[peer] = atomics[peer].self_barrier_release_token;
      p.peer_barrier_release_status[peer] = atomics[peer].self_barrier_release_status;
    }

    // Negative test for UB-proof readiness pointer formation: make a required
    // readiness flag base pointer null.
    int left = (i + kWorldSize - 1) % kWorldSize;
    p.peer_ag_ready[left] = nullptr;

    // No abort injection.
    p.debug_abort_rank = 0xffff'ffffu;
    p.debug_abort_ag_step = 0u;
    p.debug_abort_before_ag_publish = 0u;
    p.debug_abort_after_ag_publish = 0u;

    p.debug_release_delay_rank = 0xffff'ffffu;
    p.debug_release_delay_iters = 0u;

    p.debug_jitter_seed = 0u;
    p.debug_jitter_max_iters = 0u;
    p.debug_jitter_mask = 0u;

    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    cutlass::distributed::collective::ring_allreduce_sm100<float><<<tiling.num_tiles_total, 256, 0, streams[i]>>>(
        p,
        device_status[i]);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(done_events[i], streams[i]), cudaSuccess);
  }

  // Host watchdog: ensure no hangs.
  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  std::vector<bool> done(kWorldSize, false);
  while (true) {
    bool all_done = true;

    for (int i = 0; i < kWorldSize; ++i) {
      if (done[i]) {
        continue;
      }

      all_done = false;

      ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
      cudaError_t q = cudaEventQuery(done_events[i]);
      if (q == cudaSuccess) {
        done[i] = true;
        continue;
      }
      if (q != cudaErrorNotReady) {
        ASSERT_EQ(q, cudaSuccess);
      }
    }

    if (all_done) {
      break;
    }

    if (std::chrono::steady_clock::now() > deadline) {
      std::fprintf(stderr, "ring_allreduce_2gpu_test: watchdog timeout (world_size=%d)\n", kWorldSize);
      std::abort();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(&host_status[i], device_status[i], sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    EXPECT_EQ(host_status[i], static_cast<uint32_t>(RingAllreduceError::kInvalidParams));
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host_out[i].data(), device_data[i], sizeof(float) * kCount, cudaMemcpyDeviceToHost), cudaSuccess);
  }

  for (uint64_t i = 0; i < kCount; ++i) {
    EXPECT_FLOAT_EQ(host_out[0][i], host_in0[i]);
    EXPECT_FLOAT_EQ(host_out[1][i], host_in1[i]);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    destroy_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_barrier_release_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_barrier_release_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_barrier_gather_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_barrier_gather_token), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_tiles_finished), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_error), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_abort), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_ag_ready), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_rs_ready), cudaSuccess);

    ASSERT_EQ(cudaFree(device_status[i]), cudaSuccess);
    ASSERT_EQ(cudaFree(device_data[i]), cudaSuccess);

    ASSERT_EQ(cudaEventDestroy(done_events[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
  }
});

