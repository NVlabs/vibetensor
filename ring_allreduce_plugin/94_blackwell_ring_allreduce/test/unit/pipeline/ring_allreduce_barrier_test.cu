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
    \brief L2 multi-GPU unit test harness for the ring_allreduce completion barrier.
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/experimental/distributed/collective/ring_allreduce_barrier_sm100.cuh"
#include "cutlass/experimental/distributed/collective/ring_allreduce_host.hpp"

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

using cutlass::distributed::collective::RingAllreduceBarrierState;
using cutlass::distributed::collective::RingAllreduceDrainConfig;
using cutlass::distributed::collective::RingAllreduceError;
using cutlass::distributed::collective::RingAllreduceSystemAtomicU32;
using cutlass::distributed::collective::RingAllreduceParams;
using cutlass::distributed::collective::ring_allreduce_barrier_gather_release;
using cutlass::distributed::collective::ring_allreduce_is_cta0;
using cutlass::distributed::collective::ring_allreduce_is_thread0;
using cutlass::distributed::collective::validate_ring_p2p_caps_and_enable_peer_access;

static bool is_sm100_or_sm103(int device) {
  cudaDeviceProp prop{};
  cudaError_t st = cudaGetDeviceProperties(&prop, device);
  if (st != cudaSuccess) {
    return false;
  }

  int cc = prop.major * 10 + prop.minor;
  return cc == 100 || cc == 103;
}

__global__ void ring_allreduce_barrier_probe_kernel() {}

static cudaError_t ring_allreduce_barrier_probe_launch(int device) {
  // Clear any pre-existing per-thread CUDA error state.
  (void)cudaGetLastError();

  cudaError_t st = cudaSetDevice(device);
  if (st != cudaSuccess) {
    return st;
  }

  ring_allreduce_barrier_probe_kernel<<<1, 1>>>();
  return cudaGetLastError();
}

__global__ void ring_allreduce_barrier_harness_kernel(
    int32_t rank,
    int32_t world_size,
    uint32_t epoch,
    uint32_t* out_status) {

  if (!ring_allreduce_is_cta0() || !ring_allreduce_is_thread0()) {
    return;
  }

  RingAllreduceBarrierState st;
  RingAllreduceDrainConfig cfg;

  RingAllreduceError result = ring_allreduce_barrier_gather_release(
      rank,
      world_size,
      epoch,
      st,
      /*self_abort=*/nullptr,
      /*self_error=*/nullptr,
      cfg,
      /*debug_release_delay_rank=*/0u,
      /*debug_release_delay_iters=*/0u);

  if (out_status) {
    out_status[0] = static_cast<uint32_t>(result);
  }
}

struct RingAllreduceBarrierAtomics {
  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;

  RingAllreduceSystemAtomicU32* self_gather_token = nullptr;
  RingAllreduceSystemAtomicU32* self_gather_status = nullptr;

  RingAllreduceSystemAtomicU32* self_release_token = nullptr;
  RingAllreduceSystemAtomicU32* self_release_status = nullptr;
};

__global__ void ring_allreduce_barrier_microkernel(
    int32_t rank,
    int32_t world_size,
    uint32_t epoch,
    RingAllreduceBarrierAtomics self,
    RingAllreduceBarrierAtomics left,
    int invalidate_left_release_token,
    RingAllreduceDrainConfig cfg,
    uint32_t debug_release_delay_rank,
    uint32_t debug_release_delay_iters,
    uint32_t* out_status) {

  if (!ring_allreduce_is_cta0() || !ring_allreduce_is_thread0()) {
    return;
  }

  RingAllreduceBarrierState st;

  st.self_gather_token = self.self_gather_token;
  st.self_gather_status = self.self_gather_status;
  st.self_release_token = self.self_release_token;
  st.self_release_status = self.self_release_status;

  st.left_gather_token = left.self_gather_token;
  st.left_gather_status = left.self_gather_status;
  st.left_release_token = left.self_release_token;
  st.left_release_status = left.self_release_status;

  st.left_abort = left.self_abort;

  if (invalidate_left_release_token) {
    st.left_release_token = nullptr;
  }

  RingAllreduceError result = ring_allreduce_barrier_gather_release(
      rank,
      world_size,
      epoch,
      st,
      self.self_abort,
      self.self_error,
      cfg,
      debug_release_delay_rank,
      debug_release_delay_iters);

  if (out_status) {
    out_status[0] = static_cast<uint32_t>(result);
  }
}

struct RingAllreduceBarrierAtomicsSnapshot {
  uint32_t abort = 0u;
  uint32_t error = 0u;

  uint32_t gather_token = 0u;
  uint32_t gather_status = 0u;

  uint32_t release_token = 0u;
  uint32_t release_status = 0u;
};

__global__ void construct_barrier_atomics_kernel(RingAllreduceBarrierAtomics a) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (a.self_abort) {
    new (a.self_abort) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_error) {
    new (a.self_error) RingAllreduceSystemAtomicU32{};
  }

  if (a.self_gather_token) {
    new (a.self_gather_token) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_gather_status) {
    new (a.self_gather_status) RingAllreduceSystemAtomicU32{};
  }

  if (a.self_release_token) {
    new (a.self_release_token) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_release_status) {
    new (a.self_release_status) RingAllreduceSystemAtomicU32{};
  }
}

__global__ void destroy_barrier_atomics_kernel(RingAllreduceBarrierAtomics a) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (a.self_release_status) {
    a.self_release_status->~RingAllreduceSystemAtomicU32();
  }
  if (a.self_release_token) {
    a.self_release_token->~RingAllreduceSystemAtomicU32();
  }

  if (a.self_gather_status) {
    a.self_gather_status->~RingAllreduceSystemAtomicU32();
  }
  if (a.self_gather_token) {
    a.self_gather_token->~RingAllreduceSystemAtomicU32();
  }

  if (a.self_error) {
    a.self_error->~RingAllreduceSystemAtomicU32();
  }
  if (a.self_abort) {
    a.self_abort->~RingAllreduceSystemAtomicU32();
  }
}

__global__ void reset_barrier_atomics_kernel(RingAllreduceBarrierAtomics a, uint32_t epoch) {
  // Tokens are reset to 0 between runs. Keep epoch in the signature for future
  // per-epoch reset logic.
  CUTLASS_UNUSED(epoch);

  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (a.self_abort) {
    a.self_abort->store(0u, cuda::memory_order_relaxed);
  }
  if (a.self_error) {
    a.self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  }

  if (a.self_gather_token) {
    a.self_gather_token->store(0u, cuda::memory_order_relaxed);
  }
  if (a.self_gather_status) {
    a.self_gather_status->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  }

  if (a.self_release_token) {
    a.self_release_token->store(0u, cuda::memory_order_relaxed);
  }
  if (a.self_release_status) {
    a.self_release_status->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  }
}

__global__ void set_barrier_error_kernel(RingAllreduceBarrierAtomics a, RingAllreduceError error) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (a.self_error) {
    a.self_error->store(static_cast<uint32_t>(error), cuda::memory_order_relaxed);
  }
}

__global__ void misbehave_rank2_kernel(
    RingAllreduceBarrierAtomics a,
    uint32_t epoch,
    uint32_t* out_status) {

  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  // Simulate a peer that sets abort and publishes GATHER, but never forwards RELEASE.
  if (a.self_abort) {
    a.self_abort->store(1u, cuda::memory_order_release);
  }

  if (a.self_gather_status && a.self_gather_token) {
    a.self_gather_status->store(
        static_cast<uint32_t>(RingAllreduceError::kAbortObserved), cuda::memory_order_relaxed);
    a.self_gather_token->store(epoch, cuda::memory_order_release);
  }

  if (out_status) {
    out_status[0] = static_cast<uint32_t>(RingAllreduceError::kAbortObserved);
  }
}

using BarrierParamsP0 = RingAllreduceParams<uint8_t, 4>;

__global__ void ring_allreduce_barrier_params_microkernel(BarrierParamsP0 p, uint32_t* out_status) {
  if (!ring_allreduce_is_cta0() || !ring_allreduce_is_thread0()) {
    return;
  }

  RingAllreduceError result = ring_allreduce_barrier_gather_release(p);

  if (out_status) {
    out_status[0] = static_cast<uint32_t>(result);
  }
}

__global__ void snapshot_barrier_atomics_kernel(
    RingAllreduceBarrierAtomics a,
    RingAllreduceBarrierAtomicsSnapshot* out) {

  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (!out) {
    return;
  }

  RingAllreduceBarrierAtomicsSnapshot snap;

  snap.abort = a.self_abort ? a.self_abort->load(cuda::memory_order_relaxed) : 0u;
  snap.error = a.self_error ? a.self_error->load(cuda::memory_order_relaxed) : 0u;

  snap.gather_token = a.self_gather_token ? a.self_gather_token->load(cuda::memory_order_relaxed) : 0u;
  snap.gather_status = a.self_gather_status ? a.self_gather_status->load(cuda::memory_order_relaxed) : 0u;

  snap.release_token = a.self_release_token ? a.self_release_token->load(cuda::memory_order_relaxed) : 0u;
  snap.release_status = a.self_release_status ? a.self_release_status->load(cuda::memory_order_relaxed) : 0u;

  *out = snap;
}

} // namespace

CUTLASS_TEST_L2(RingAllreduceBarrier, S0_HarnessAlive, {

  constexpr int kWorldSize = 4;
  constexpr uint32_t kEpoch = 1;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  // Pick a set of kWorldSize devices that (a) are SM100/SM103, (b) have a valid
  // kernel image in this binary, and (c) satisfy the ring P2P requirements.
  std::vector<int> candidates;
  candidates.reserve(device_count);

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_barrier_probe_launch(dev);
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
  bool found_ring = false;

  for (size_t start = 0; start + kWorldSize <= candidates.size(); ++start) {
    for (int i = 0; i < kWorldSize; ++i) {
      devices[i] = candidates[start + static_cast<size_t>(i)];
    }

    auto p2p = validate_ring_p2p_caps_and_enable_peer_access(kWorldSize, devices.data());
    if (p2p.ok()) {
      found_ring = true;
      break;
    }
  }

  if (!found_ring) {
    GTEST_SKIP() << "no suitable P2P ring with native peer atomics";
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

    ring_allreduce_barrier_harness_kernel<<<1, 1, 0, streams[i]>>>(
        /*rank=*/i,
        /*world_size=*/kWorldSize,
        /*epoch=*/kEpoch,
        device_status[i]);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(done_events[i], streams[i]), cudaSuccess);
  }

  // Host watchdog: ensure no hangs even while the barrier is under development.
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
      std::fprintf(stderr, "ring_allreduce_barrier_test: watchdog timeout (world_size=%d)\n", kWorldSize);
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
    // M1 scaffold: the barrier helper is a stub that returns kInvalidParams.
    EXPECT_EQ(host_status[i], static_cast<uint32_t>(RingAllreduceError::kInvalidParams));
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaFree(device_status[i]), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(done_events[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
  }
});

CUTLASS_TEST_L2(RingAllreduceBarrier, S1_AtomicsConstructReset, {

  constexpr int kWorldSize = 4;
  constexpr uint32_t kEpoch = 1;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  // Pick a set of kWorldSize devices that (a) are SM100/SM103, (b) have a valid
  // kernel image in this binary, and (c) satisfy the ring P2P requirements.
  std::vector<int> candidates;
  candidates.reserve(device_count);

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_barrier_probe_launch(dev);
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
  bool found_ring = false;

  for (size_t start = 0; start + kWorldSize <= candidates.size(); ++start) {
    for (int i = 0; i < kWorldSize; ++i) {
      devices[i] = candidates[start + static_cast<size_t>(i)];
    }

    auto p2p = validate_ring_p2p_caps_and_enable_peer_access(kWorldSize, devices.data());
    if (p2p.ok()) {
      found_ring = true;
      break;
    }
  }

  if (!found_ring) {
    GTEST_SKIP() << "no suitable P2P ring with native peer atomics";
  }

  std::array<RingAllreduceBarrierAtomics, kWorldSize> atomics{};
  std::array<RingAllreduceBarrierAtomicsSnapshot, kWorldSize> host_snap{};

  std::vector<cudaStream_t> streams(kWorldSize);

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&streams[i]), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    construct_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    reset_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i], kEpoch);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    RingAllreduceBarrierAtomicsSnapshot* device_snap = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_snap), sizeof(RingAllreduceBarrierAtomicsSnapshot)), cudaSuccess);

    snapshot_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i], device_snap);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&host_snap[i], device_snap, sizeof(RingAllreduceBarrierAtomicsSnapshot), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(device_snap), cudaSuccess);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    EXPECT_EQ(host_snap[i].abort, 0u);
    EXPECT_EQ(host_snap[i].error, static_cast<uint32_t>(RingAllreduceError::kOk));
    EXPECT_EQ(host_snap[i].gather_token, 0u);
    EXPECT_EQ(host_snap[i].gather_status, static_cast<uint32_t>(RingAllreduceError::kOk));
    EXPECT_EQ(host_snap[i].release_token, 0u);
    EXPECT_EQ(host_snap[i].release_status, static_cast<uint32_t>(RingAllreduceError::kOk));
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    destroy_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_release_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_release_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_gather_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_gather_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_error), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_abort), cudaSuccess);

    ASSERT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
  }
});

CUTLASS_TEST_L2(RingAllreduceBarrier, I0_InvalidParamsLiveness, {

  constexpr int kWorldSize = 4;
  constexpr uint32_t kEpoch = 1;
  constexpr int kInvalidRank = 2;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  // Pick a set of kWorldSize devices that (a) are SM100/SM103, (b) have a valid
  // kernel image in this binary, and (c) satisfy the ring P2P requirements.
  std::vector<int> candidates;
  candidates.reserve(device_count);

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_barrier_probe_launch(dev);
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
  bool found_ring = false;

  for (size_t start = 0; start + kWorldSize <= candidates.size(); ++start) {
    for (int i = 0; i < kWorldSize; ++i) {
      devices[i] = candidates[start + static_cast<size_t>(i)];
    }

    auto p2p = validate_ring_p2p_caps_and_enable_peer_access(kWorldSize, devices.data());
    if (p2p.ok()) {
      found_ring = true;
      break;
    }
  }

  if (!found_ring) {
    GTEST_SKIP() << "no suitable P2P ring with native peer atomics";
  }

  RingAllreduceDrainConfig cfg;
  cfg.timeout_iters = 0;   // infinite waits in barrier logic
  cfg.timeout_cycles = 0;  // disable cycle-based timeouts
  cfg.poll_sleep_start = 0;
  cfg.poll_sleep_ns = 0;

  std::array<RingAllreduceBarrierAtomics, kWorldSize> atomics{};
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

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    construct_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    reset_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i], kEpoch);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    int left = (i + kWorldSize - 1) % kWorldSize;
    int invalidate = (i == kInvalidRank) ? 1 : 0;

    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    ring_allreduce_barrier_microkernel<<<1, 1, 0, streams[i]>>>(
        /*rank=*/i,
        /*world_size=*/kWorldSize,
        /*epoch=*/kEpoch,
        atomics[i],
        atomics[left],
        invalidate,
        cfg,
        /*debug_release_delay_rank=*/0u,
        /*debug_release_delay_iters=*/0u,
        device_status[i]);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(done_events[i], streams[i]), cudaSuccess);
  }

  // Host watchdog: ensure no hangs even with infinite-wait timeouts.
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
      std::fprintf(stderr, "ring_allreduce_barrier_test: watchdog timeout (I0, world_size=%d)\n", kWorldSize);
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
    EXPECT_NE(host_status[i], static_cast<uint32_t>(RingAllreduceError::kOk));
  }
  EXPECT_EQ(host_status[kInvalidRank], static_cast<uint32_t>(RingAllreduceError::kInvalidParams));

  // Tightened assertions: verify invalid rank published gather+release tokens for epoch.
  RingAllreduceBarrierAtomicsSnapshot invalid_snap{};
  RingAllreduceBarrierAtomicsSnapshot* device_snap = nullptr;

  ASSERT_EQ(cudaSetDevice(devices[kInvalidRank]), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_snap), sizeof(RingAllreduceBarrierAtomicsSnapshot)), cudaSuccess);

  snapshot_barrier_atomics_kernel<<<1, 1, 0, streams[kInvalidRank]>>>(atomics[kInvalidRank], device_snap);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(streams[kInvalidRank]), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&invalid_snap, device_snap, sizeof(RingAllreduceBarrierAtomicsSnapshot), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(device_snap), cudaSuccess);

  EXPECT_EQ(invalid_snap.abort, 1u);
  EXPECT_EQ(invalid_snap.error, static_cast<uint32_t>(RingAllreduceError::kInvalidParams));
  EXPECT_EQ(invalid_snap.gather_token, kEpoch);
  EXPECT_EQ(invalid_snap.gather_status, static_cast<uint32_t>(RingAllreduceError::kInvalidParams));
  EXPECT_EQ(invalid_snap.release_token, kEpoch);
  EXPECT_EQ(invalid_snap.release_status, static_cast<uint32_t>(RingAllreduceError::kInvalidParams));

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    destroy_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_release_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_release_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_gather_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_gather_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_error), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_abort), cudaSuccess);

    ASSERT_EQ(cudaFree(device_status[i]), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(done_events[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
  }
});

CUTLASS_TEST_L2(RingAllreduceBarrier, W0_WorldSizeOne, {

  constexpr int kWorldSize = 1;
  constexpr uint32_t kEpoch = 1;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA device";
  }

  int device = -1;
  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_barrier_probe_launch(dev);
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

  RingAllreduceDrainConfig cfg;
  cfg.timeout_cycles = 0;
  cfg.timeout_iters = 1u << 18;
  cfg.poll_sleep_start = 0;
  cfg.poll_sleep_ns = 40;

  RingAllreduceBarrierAtomics self{};
  RingAllreduceBarrierAtomics left{};

  uint32_t host_status = 0xffff'ffffu;
  uint32_t* device_status = nullptr;

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;

  ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status), sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(device_status, 0xFF, sizeof(uint32_t), stream), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&self.self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&self.self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&self.self_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&self.self_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&self.self_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&self.self_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  construct_barrier_atomics_kernel<<<1, 1, 0, stream>>>(self);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  reset_barrier_atomics_kernel<<<1, 1, 0, stream>>>(self, kEpoch);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  ring_allreduce_barrier_microkernel<<<1, 1, 0, stream>>>(
      /*rank=*/0,
      /*world_size=*/kWorldSize,
      /*epoch=*/kEpoch,
      self,
      left,
      /*invalidate_left_release_token=*/0,
      cfg,
      /*debug_release_delay_rank=*/0u,
      /*debug_release_delay_iters=*/0u,
      device_status);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  while (true) {
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
    cudaError_t q = cudaEventQuery(done_event);
    if (q == cudaSuccess) {
      break;
    }
    if (q != cudaErrorNotReady) {
      ASSERT_EQ(q, cudaSuccess);
    }

    if (std::chrono::steady_clock::now() > deadline) {
      std::fprintf(stderr, "ring_allreduce_barrier_test: watchdog timeout (W0, world_size=%d)\n", kWorldSize);
      std::abort();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&host_status, device_status, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_EQ(host_status, static_cast<uint32_t>(RingAllreduceError::kOk));

  RingAllreduceBarrierAtomicsSnapshot host_snap{};
  RingAllreduceBarrierAtomicsSnapshot* device_snap = nullptr;

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_snap), sizeof(RingAllreduceBarrierAtomicsSnapshot)), cudaSuccess);

  snapshot_barrier_atomics_kernel<<<1, 1, 0, stream>>>(self, device_snap);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&host_snap, device_snap, sizeof(RingAllreduceBarrierAtomicsSnapshot), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(device_snap), cudaSuccess);

  EXPECT_EQ(host_snap.abort, 0u);
  EXPECT_EQ(host_snap.error, static_cast<uint32_t>(RingAllreduceError::kOk));
  EXPECT_EQ(host_snap.gather_token, kEpoch);
  EXPECT_EQ(host_snap.gather_status, static_cast<uint32_t>(RingAllreduceError::kOk));
  EXPECT_EQ(host_snap.release_token, kEpoch);
  EXPECT_EQ(host_snap.release_status, static_cast<uint32_t>(RingAllreduceError::kOk));

  destroy_barrier_atomics_kernel<<<1, 1, 0, stream>>>(self);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  ASSERT_EQ(cudaFree(self.self_release_status), cudaSuccess);
  ASSERT_EQ(cudaFree(self.self_release_token), cudaSuccess);
  ASSERT_EQ(cudaFree(self.self_gather_status), cudaSuccess);
  ASSERT_EQ(cudaFree(self.self_gather_token), cudaSuccess);
  ASSERT_EQ(cudaFree(self.self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self.self_abort), cudaSuccess);

  ASSERT_EQ(cudaFree(device_status), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
});

CUTLASS_TEST_L2(RingAllreduceBarrier, E0_EpochZeroInvalid, {

  constexpr int kWorldSize = 1;
  constexpr uint32_t kEpoch = 0;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA device";
  }

  int device = -1;
  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_barrier_probe_launch(dev);
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

  RingAllreduceDrainConfig cfg;
  cfg.timeout_cycles = 0;
  cfg.timeout_iters = 1u << 18;
  cfg.poll_sleep_start = 0;
  cfg.poll_sleep_ns = 40;

  RingAllreduceBarrierAtomics self{};
  RingAllreduceBarrierAtomics left{};

  uint32_t host_status = 0xffff'ffffu;
  uint32_t* device_status = nullptr;

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;

  ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status), sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(device_status, 0xFF, sizeof(uint32_t), stream), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&self.self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&self.self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&self.self_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&self.self_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&self.self_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&self.self_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  construct_barrier_atomics_kernel<<<1, 1, 0, stream>>>(self);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  reset_barrier_atomics_kernel<<<1, 1, 0, stream>>>(self, kEpoch);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  ring_allreduce_barrier_microkernel<<<1, 1, 0, stream>>>(
      /*rank=*/0,
      /*world_size=*/kWorldSize,
      /*epoch=*/kEpoch,
      self,
      left,
      /*invalidate_left_release_token=*/0,
      cfg,
      /*debug_release_delay_rank=*/0u,
      /*debug_release_delay_iters=*/0u,
      device_status);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  while (true) {
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
    cudaError_t q = cudaEventQuery(done_event);
    if (q == cudaSuccess) {
      break;
    }
    if (q != cudaErrorNotReady) {
      ASSERT_EQ(q, cudaSuccess);
    }

    if (std::chrono::steady_clock::now() > deadline) {
      std::fprintf(stderr, "ring_allreduce_barrier_test: watchdog timeout (E0, world_size=%d)\n", kWorldSize);
      std::abort();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&host_status, device_status, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_EQ(host_status, static_cast<uint32_t>(RingAllreduceError::kInvalidParams));

  RingAllreduceBarrierAtomicsSnapshot host_snap{};
  RingAllreduceBarrierAtomicsSnapshot* device_snap = nullptr;

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_snap), sizeof(RingAllreduceBarrierAtomicsSnapshot)), cudaSuccess);

  snapshot_barrier_atomics_kernel<<<1, 1, 0, stream>>>(self, device_snap);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&host_snap, device_snap, sizeof(RingAllreduceBarrierAtomicsSnapshot), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(device_snap), cudaSuccess);

  EXPECT_EQ(host_snap.abort, 1u);
  EXPECT_EQ(host_snap.error, static_cast<uint32_t>(RingAllreduceError::kInvalidParams));
  EXPECT_EQ(host_snap.gather_token, 0u);
  EXPECT_EQ(host_snap.gather_status, static_cast<uint32_t>(RingAllreduceError::kOk));
  EXPECT_EQ(host_snap.release_token, 0u);
  EXPECT_EQ(host_snap.release_status, static_cast<uint32_t>(RingAllreduceError::kOk));

  destroy_barrier_atomics_kernel<<<1, 1, 0, stream>>>(self);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  ASSERT_EQ(cudaFree(self.self_release_status), cudaSuccess);
  ASSERT_EQ(cudaFree(self.self_release_token), cudaSuccess);
  ASSERT_EQ(cudaFree(self.self_gather_status), cudaSuccess);
  ASSERT_EQ(cudaFree(self.self_gather_token), cudaSuccess);
  ASSERT_EQ(cudaFree(self.self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self.self_abort), cudaSuccess);

  ASSERT_EQ(cudaFree(device_status), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
});

CUTLASS_TEST_L2(RingAllreduceBarrier, B0_SuccessReleaseDelay, {

  constexpr int kWorldSize = 4;
  constexpr uint32_t kEpoch = 1;
  constexpr uint32_t kDelayRank = 2;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  // Pick a set of kWorldSize devices that (a) are SM100/SM103, (b) have a valid
  // kernel image in this binary, and (c) satisfy the ring P2P requirements.
  std::vector<int> candidates;
  candidates.reserve(device_count);

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_barrier_probe_launch(dev);
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
  bool found_ring = false;

  for (size_t start = 0; start + kWorldSize <= candidates.size(); ++start) {
    for (int i = 0; i < kWorldSize; ++i) {
      devices[i] = candidates[start + static_cast<size_t>(i)];
    }

    auto p2p = validate_ring_p2p_caps_and_enable_peer_access(kWorldSize, devices.data());
    if (p2p.ok()) {
      found_ring = true;
      break;
    }
  }

  if (!found_ring) {
    GTEST_SKIP() << "no suitable P2P ring with native peer atomics";
  }

  RingAllreduceDrainConfig cfg;
  cfg.timeout_cycles = 0;
  cfg.timeout_iters = 1u << 18;
  cfg.poll_sleep_start = 0;
  cfg.poll_sleep_ns = 40;

  uint32_t debug_release_delay_rank = kDelayRank;
  uint32_t debug_release_delay_iters = cfg.timeout_iters * 4u;

  std::array<RingAllreduceBarrierAtomics, kWorldSize> atomics{};
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

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    construct_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    reset_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i], kEpoch);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    int left = (i + kWorldSize - 1) % kWorldSize;

    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    ring_allreduce_barrier_microkernel<<<1, 1, 0, streams[i]>>>(
        /*rank=*/i,
        /*world_size=*/kWorldSize,
        /*epoch=*/kEpoch,
        atomics[i],
        atomics[left],
        /*invalidate_left_release_token=*/0,
        cfg,
        debug_release_delay_rank,
        debug_release_delay_iters,
        device_status[i]);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(done_events[i], streams[i]), cudaSuccess);
  }

  // Host watchdog: ensures RELEASE delay can't cause false timeouts.
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
      std::fprintf(stderr, "ring_allreduce_barrier_test: watchdog timeout (B0, world_size=%d)\n", kWorldSize);
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

  // Sanity: rank0 must not publish abort/error on success.
  RingAllreduceBarrierAtomicsSnapshot rank0_snap{};
  RingAllreduceBarrierAtomicsSnapshot* device_snap = nullptr;

  ASSERT_EQ(cudaSetDevice(devices[0]), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_snap), sizeof(RingAllreduceBarrierAtomicsSnapshot)), cudaSuccess);

  snapshot_barrier_atomics_kernel<<<1, 1, 0, streams[0]>>>(atomics[0], device_snap);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(streams[0]), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&rank0_snap, device_snap, sizeof(RingAllreduceBarrierAtomicsSnapshot), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(device_snap), cudaSuccess);

  EXPECT_EQ(rank0_snap.abort, 0u);
  EXPECT_EQ(rank0_snap.error, static_cast<uint32_t>(RingAllreduceError::kOk));
  EXPECT_EQ(rank0_snap.gather_token, kEpoch);
  EXPECT_EQ(rank0_snap.gather_status, static_cast<uint32_t>(RingAllreduceError::kOk));
  EXPECT_EQ(rank0_snap.release_token, kEpoch);
  EXPECT_EQ(rank0_snap.release_status, static_cast<uint32_t>(RingAllreduceError::kOk));

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    destroy_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_release_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_release_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_gather_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_gather_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_error), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_abort), cudaSuccess);

    ASSERT_EQ(cudaFree(device_status[i]), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(done_events[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
  }
});

CUTLASS_TEST_L2(RingAllreduceBarrier, B1_NonOkPropagationRank0Latch, {

  constexpr int kWorldSize = 4;
  constexpr uint32_t kEpoch = 1;
  constexpr int kErrorRank = 2;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  // Pick a set of kWorldSize devices that (a) are SM100/SM103, (b) have a valid
  // kernel image in this binary, and (c) satisfy the ring P2P requirements.
  std::vector<int> candidates;
  candidates.reserve(device_count);

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_barrier_probe_launch(dev);
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
  bool found_ring = false;

  for (size_t start = 0; start + kWorldSize <= candidates.size(); ++start) {
    for (int i = 0; i < kWorldSize; ++i) {
      devices[i] = candidates[start + static_cast<size_t>(i)];
    }

    auto p2p = validate_ring_p2p_caps_and_enable_peer_access(kWorldSize, devices.data());
    if (p2p.ok()) {
      found_ring = true;
      break;
    }
  }

  if (!found_ring) {
    GTEST_SKIP() << "no suitable P2P ring with native peer atomics";
  }

  RingAllreduceDrainConfig cfg;
  cfg.timeout_cycles = 0;
  cfg.timeout_iters = 1u << 18;
  cfg.poll_sleep_start = 0;
  cfg.poll_sleep_ns = 40;

  std::array<RingAllreduceBarrierAtomics, kWorldSize> atomics{};
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

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    construct_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    reset_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i], kEpoch);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
  }

  // Inject a non-Ok local error on one rank.
  ASSERT_EQ(cudaSetDevice(devices[kErrorRank]), cudaSuccess);
  set_barrier_error_kernel<<<1, 1, 0, streams[kErrorRank]>>>(atomics[kErrorRank], RingAllreduceError::kTimeout);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(streams[kErrorRank]), cudaSuccess);

  for (int i = 0; i < kWorldSize; ++i) {
    int left = (i + kWorldSize - 1) % kWorldSize;

    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    ring_allreduce_barrier_microkernel<<<1, 1, 0, streams[i]>>>(
        /*rank=*/i,
        /*world_size=*/kWorldSize,
        /*epoch=*/kEpoch,
        atomics[i],
        atomics[left],
        /*invalidate_left_release_token=*/0,
        cfg,
        /*debug_release_delay_rank=*/0u,
        /*debug_release_delay_iters=*/0u,
        device_status[i]);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(done_events[i], streams[i]), cudaSuccess);
  }

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
      std::fprintf(stderr, "ring_allreduce_barrier_test: watchdog timeout (B1, world_size=%d)\n", kWorldSize);
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
    EXPECT_EQ(host_status[i], static_cast<uint32_t>(RingAllreduceError::kTimeout));
  }

  // Rank0 must latch the non-Ok final_status into self_error/self_abort.
  RingAllreduceBarrierAtomicsSnapshot rank0_snap{};
  RingAllreduceBarrierAtomicsSnapshot* device_snap = nullptr;

  ASSERT_EQ(cudaSetDevice(devices[0]), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_snap), sizeof(RingAllreduceBarrierAtomicsSnapshot)), cudaSuccess);

  snapshot_barrier_atomics_kernel<<<1, 1, 0, streams[0]>>>(atomics[0], device_snap);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(streams[0]), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&rank0_snap, device_snap, sizeof(RingAllreduceBarrierAtomicsSnapshot), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(device_snap), cudaSuccess);

  EXPECT_EQ(rank0_snap.abort, 1u);
  EXPECT_EQ(rank0_snap.error, static_cast<uint32_t>(RingAllreduceError::kTimeout));
  EXPECT_EQ(rank0_snap.gather_token, kEpoch);
  EXPECT_EQ(rank0_snap.gather_status, static_cast<uint32_t>(RingAllreduceError::kOk));
  EXPECT_EQ(rank0_snap.release_token, kEpoch);
  EXPECT_EQ(rank0_snap.release_status, static_cast<uint32_t>(RingAllreduceError::kTimeout));

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    destroy_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_release_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_release_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_gather_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_gather_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_error), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_abort), cudaSuccess);

    ASSERT_EQ(cudaFree(device_status[i]), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(done_events[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
  }
});

CUTLASS_TEST_L2(RingAllreduceBarrier, P0_ParamsOverloadOobGuard, {

  constexpr int kWorldSize = 5;
  constexpr uint32_t kEpoch = 1;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < 1) {
    GTEST_SKIP() << "requires a CUDA device";
  }

  int device = -1;
  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_barrier_probe_launch(dev);
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

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  RingAllreduceBarrierAtomics atomics{};
  uint32_t* device_status = nullptr;
  uint32_t host_status = 0xffff'ffffu;

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status), sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(device_status, 0xFF, sizeof(uint32_t), stream), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics.self_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  construct_barrier_atomics_kernel<<<1, 1, 0, stream>>>(atomics);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  reset_barrier_atomics_kernel<<<1, 1, 0, stream>>>(atomics, kEpoch);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  BarrierParamsP0 p{};
  p.world_size = kWorldSize;
  p.rank = 0;
  p.epoch = kEpoch;

  p.timeout_iters = 0;
  p.timeout_cycles = 0;
  p.poll_sleep_start = 0;
  p.poll_sleep_ns = 0;

  p.self_abort = atomics.self_abort;
  p.self_error = atomics.self_error;
  p.self_barrier_gather_token = atomics.self_gather_token;
  p.self_barrier_gather_status = atomics.self_gather_status;
  p.self_barrier_release_token = atomics.self_release_token;
  p.self_barrier_release_status = atomics.self_release_status;

  p.debug_release_delay_rank = 0;
  p.debug_release_delay_iters = 0;

  ring_allreduce_barrier_params_microkernel<<<1, 1, 0, stream>>>(p, device_status);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&host_status, device_status, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_EQ(host_status, static_cast<uint32_t>(RingAllreduceError::kInvalidParams));

  RingAllreduceBarrierAtomicsSnapshot snap{};
  RingAllreduceBarrierAtomicsSnapshot* device_snap = nullptr;

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_snap), sizeof(RingAllreduceBarrierAtomicsSnapshot)), cudaSuccess);

  snapshot_barrier_atomics_kernel<<<1, 1, 0, stream>>>(atomics, device_snap);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&snap, device_snap, sizeof(RingAllreduceBarrierAtomicsSnapshot), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(device_snap), cudaSuccess);

  EXPECT_EQ(snap.abort, 1u);
  EXPECT_EQ(snap.error, static_cast<uint32_t>(RingAllreduceError::kInvalidParams));
  EXPECT_EQ(snap.gather_token, kEpoch);
  EXPECT_EQ(snap.gather_status, static_cast<uint32_t>(RingAllreduceError::kInvalidParams));
  EXPECT_EQ(snap.release_token, kEpoch);
  EXPECT_EQ(snap.release_status, static_cast<uint32_t>(RingAllreduceError::kInvalidParams));

  destroy_barrier_atomics_kernel<<<1, 1, 0, stream>>>(atomics);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  ASSERT_EQ(cudaFree(atomics.self_release_status), cudaSuccess);
  ASSERT_EQ(cudaFree(atomics.self_release_token), cudaSuccess);
  ASSERT_EQ(cudaFree(atomics.self_gather_status), cudaSuccess);
  ASSERT_EQ(cudaFree(atomics.self_gather_token), cudaSuccess);
  ASSERT_EQ(cudaFree(atomics.self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(atomics.self_abort), cudaSuccess);

  ASSERT_EQ(cudaFree(device_status), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
});

CUTLASS_TEST_L2(RingAllreduceBarrier, R0_ReleasePeerAbortArmingRegression, {

  constexpr int kWorldSize = 4;
  constexpr uint32_t kEpoch = 1;
  constexpr int kMisbehaveRank = 2;
  constexpr int kTimeoutRank = 3;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  // Pick a set of kWorldSize devices that (a) are SM100/SM103, (b) have a valid
  // kernel image in this binary, and (c) satisfy the ring P2P requirements.
  std::vector<int> candidates;
  candidates.reserve(device_count);

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_barrier_probe_launch(dev);
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
  bool found_ring = false;

  for (size_t start = 0; start + kWorldSize <= candidates.size(); ++start) {
    for (int i = 0; i < kWorldSize; ++i) {
      devices[i] = candidates[start + static_cast<size_t>(i)];
    }

    auto p2p = validate_ring_p2p_caps_and_enable_peer_access(kWorldSize, devices.data());
    if (p2p.ok()) {
      found_ring = true;
      break;
    }
  }

  if (!found_ring) {
    GTEST_SKIP() << "no suitable P2P ring with native peer atomics";
  }

  RingAllreduceDrainConfig cfg;
  cfg.timeout_cycles = 0;
  cfg.timeout_iters = 1u << 18;
  cfg.poll_sleep_start = 0;
  cfg.poll_sleep_ns = 40;

  std::array<RingAllreduceBarrierAtomics, kWorldSize> atomics{};
  std::vector<uint32_t> host_status(kWorldSize, 0xffff'ffffu);
  std::vector<uint32_t*> device_status(kWorldSize, nullptr);

  std::vector<cudaStream_t> streams(kWorldSize);
  std::vector<cudaStream_t> diag_streams(kWorldSize);
  std::vector<cudaEvent_t> done_events(kWorldSize);

  std::vector<RingAllreduceBarrierAtomicsSnapshot*> device_snaps(kWorldSize, nullptr);

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&streams[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&diag_streams[i], cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&done_events[i], cudaEventDisableTiming), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status[i]), sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(device_status[i], 0xFF, sizeof(uint32_t), streams[i]), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[i].self_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    construct_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    reset_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i], kEpoch);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_snaps[i]), sizeof(RingAllreduceBarrierAtomicsSnapshot)), cudaSuccess);
  }

  // Reset phase must complete on all devices before the R0 choreography begins.
  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
  }

  // Misbehaving prelude (rank2): publish gather token/status and set abort, but
  // do not publish RELEASE.
  ASSERT_EQ(cudaSetDevice(devices[kMisbehaveRank]), cudaSuccess);

  misbehave_rank2_kernel<<<1, 1, 0, streams[kMisbehaveRank]>>>(
      atomics[kMisbehaveRank],
      kEpoch,
      device_status[kMisbehaveRank]);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(streams[kMisbehaveRank]), cudaSuccess);

  // Barrier kernels (ranks 0,1,3).
  for (int i = 0; i < kWorldSize; ++i) {
    if (i == kMisbehaveRank) {
      continue;
    }

    int left = (i + kWorldSize - 1) % kWorldSize;

    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    ring_allreduce_barrier_microkernel<<<1, 1, 0, streams[i]>>>(
        /*rank=*/i,
        /*world_size=*/kWorldSize,
        /*epoch=*/kEpoch,
        atomics[i],
        atomics[left],
        /*invalidate_left_release_token=*/0,
        cfg,
        /*debug_release_delay_rank=*/0u,
        /*debug_release_delay_iters=*/0u,
        device_status[i]);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(done_events[i], streams[i]), cudaSuccess);
  }

  // Host watchdog with best-effort barrier snapshot diagnostics.
  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  std::vector<bool> done(kWorldSize, false);
  done[kMisbehaveRank] = true;

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
      std::fprintf(stderr, "ring_allreduce_barrier_test: watchdog timeout (R0, world_size=%d)\n", kWorldSize);

      std::array<RingAllreduceBarrierAtomicsSnapshot, kWorldSize> host_snap{};

      for (int i = 0; i < kWorldSize; ++i) {
        ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

        snapshot_barrier_atomics_kernel<<<1, 1, 0, diag_streams[i]>>>(atomics[i], device_snaps[i]);
        (void)cudaGetLastError();

        ASSERT_EQ(cudaMemcpyAsync(
            &host_snap[i],
            device_snaps[i],
            sizeof(RingAllreduceBarrierAtomicsSnapshot),
            cudaMemcpyDeviceToHost,
            diag_streams[i]), cudaSuccess);
      }

      for (int i = 0; i < kWorldSize; ++i) {
        ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
        (void)cudaStreamSynchronize(diag_streams[i]);

        std::fprintf(
            stderr,
            "  rank %d: abort=%u error=%u gather={token=%u status=%u} release={token=%u status=%u}\n",
            i,
            host_snap[i].abort,
            host_snap[i].error,
            host_snap[i].gather_token,
            host_snap[i].gather_status,
            host_snap[i].release_token,
            host_snap[i].release_status);
      }

      std::abort();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&host_status[i], device_status[i], sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
  }

  EXPECT_EQ(host_status[kTimeoutRank], static_cast<uint32_t>(RingAllreduceError::kTimeout));
  EXPECT_NE(host_status[0], static_cast<uint32_t>(RingAllreduceError::kOk));

  // Verify rank3 timed out and published its release token/status.
  RingAllreduceBarrierAtomicsSnapshot rank3_snap{};
  RingAllreduceBarrierAtomicsSnapshot* device_snap = nullptr;

  ASSERT_EQ(cudaSetDevice(devices[kTimeoutRank]), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_snap), sizeof(RingAllreduceBarrierAtomicsSnapshot)), cudaSuccess);

  snapshot_barrier_atomics_kernel<<<1, 1, 0, streams[kTimeoutRank]>>>(atomics[kTimeoutRank], device_snap);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(streams[kTimeoutRank]), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&rank3_snap, device_snap, sizeof(RingAllreduceBarrierAtomicsSnapshot), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(device_snap), cudaSuccess);

  EXPECT_EQ(rank3_snap.abort, 1u);
  EXPECT_EQ(rank3_snap.error, static_cast<uint32_t>(RingAllreduceError::kTimeout));
  EXPECT_EQ(rank3_snap.release_token, kEpoch);
  EXPECT_EQ(rank3_snap.release_status, static_cast<uint32_t>(RingAllreduceError::kTimeout));

  // Rank0 must return non-Ok and latch non-Ok into self_abort/self_error.
  RingAllreduceBarrierAtomicsSnapshot rank0_snap{};
  device_snap = nullptr;

  ASSERT_EQ(cudaSetDevice(devices[0]), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_snap), sizeof(RingAllreduceBarrierAtomicsSnapshot)), cudaSuccess);

  snapshot_barrier_atomics_kernel<<<1, 1, 0, streams[0]>>>(atomics[0], device_snap);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(streams[0]), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&rank0_snap, device_snap, sizeof(RingAllreduceBarrierAtomicsSnapshot), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(device_snap), cudaSuccess);

  EXPECT_EQ(rank0_snap.abort, 1u);
  EXPECT_NE(rank0_snap.error, static_cast<uint32_t>(RingAllreduceError::kOk));
  EXPECT_EQ(rank0_snap.release_token, kEpoch);
  EXPECT_NE(rank0_snap.release_status, static_cast<uint32_t>(RingAllreduceError::kOk));

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[i]), cudaSuccess);

    destroy_barrier_atomics_kernel<<<1, 1, 0, streams[i]>>>(atomics[i]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);

    ASSERT_EQ(cudaFree(device_snaps[i]), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[i].self_release_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_release_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_gather_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_gather_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_error), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[i].self_abort), cudaSuccess);

    ASSERT_EQ(cudaFree(device_status[i]), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(done_events[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(diag_streams[i]), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
  }
});
