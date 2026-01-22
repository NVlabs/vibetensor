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
    \brief L2 multi-GPU harness for ring_allreduce (N-GPU correctness + abort scaffolding).
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
#include <initializer_list>
#include <new>
#include <thread>
#include <vector>

namespace {

using cutlass::distributed::collective::RingAllreduceDeviceAtomicU32;
using cutlass::distributed::collective::RingAllreduceError;
using cutlass::distributed::collective::RingAllreduceHostResult;
using cutlass::distributed::collective::RingAllreduceParams;
using cutlass::distributed::collective::RingAllreduceSystemAtomicU32;
using cutlass::distributed::collective::RingAllreduceTiling;
using cutlass::distributed::collective::validate_ring_allreduce_host_tiling;
using cutlass::distributed::collective::validate_ring_p2p_caps_and_enable_peer_access;

struct ScopedCudaDeviceRestore {
  int device = 0;
  cudaError_t st = cudaSuccess;

  ScopedCudaDeviceRestore() {
    st = cudaGetDevice(&device);
  }

  ~ScopedCudaDeviceRestore() {
    if (st == cudaSuccess) {
      (void)cudaSetDevice(device);
    }
  }

  bool ok() const {
    return st == cudaSuccess;
  }
};

struct ExpectedAgReadyValue {
  uint32_t step = 0u;
  uint32_t value = 0u;
};

static bool is_sm100_or_sm103(int device) {
  cudaDeviceProp prop{};
  cudaError_t st = cudaGetDeviceProperties(&prop, device);
  if (st != cudaSuccess) {
    return false;
  }

  int cc = prop.major * 10 + prop.minor;
  return cc == 100 || cc == 103;
}

__global__ void ring_allreduce_ngpu_abort_probe_kernel() {}

static cudaError_t ring_allreduce_ngpu_abort_probe_launch(int device) {
  // Clear any pre-existing per-thread CUDA error state.
  (void)cudaGetLastError();

  int original_device = 0;
  cudaError_t st = cudaGetDevice(&original_device);
  if (st != cudaSuccess) {
    return st;
  }

  st = cudaSetDevice(device);
  if (st != cudaSuccess) {
    return st;
  }

  ring_allreduce_ngpu_abort_probe_kernel<<<1, 1>>>();
  cudaError_t probe_st = cudaGetLastError();

  // Restore the device to avoid leaking CUDA global state into other tests.
  (void)cudaSetDevice(original_device);

  return probe_st;
}

__global__ void ring_allreduce_ngpu_abort_noop_kernel(uint32_t value, uint32_t* out_status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (out_status) {
    out_status[0] = value;
  }
}

static bool p2p_pair_ok(int a, int b) {
  int can_ab = 0;
  int can_ba = 0;

  cudaError_t st = cudaDeviceCanAccessPeer(&can_ab, a, b);
  if (st != cudaSuccess) {
    return false;
  }
  st = cudaDeviceCanAccessPeer(&can_ba, b, a);
  if (st != cudaSuccess) {
    return false;
  }

  if (!can_ab || !can_ba) {
    return false;
  }

  int atomic_ab = 0;
  int atomic_ba = 0;

  st = cudaDeviceGetP2PAttribute(&atomic_ab, cudaDevP2PAttrNativeAtomicSupported, a, b);
  if (st != cudaSuccess) {
    return false;
  }
  st = cudaDeviceGetP2PAttribute(&atomic_ba, cudaDevP2PAttrNativeAtomicSupported, b, a);
  if (st != cudaSuccess) {
    return false;
  }

  return (atomic_ab == 1) && (atomic_ba == 1);
}

static bool dfs_pick_ring(
    int32_t world_size,
    std::vector<std::vector<uint8_t>> const& adj,
    int start,
    std::vector<int>& path,
    std::vector<uint8_t>& used) {

  if (static_cast<int32_t>(path.size()) == world_size) {
    return adj[static_cast<size_t>(path.back())][static_cast<size_t>(start)] != 0u;
  }

  int last = path.back();

  for (size_t next = 0; next < used.size(); ++next) {
    if (used[next]) {
      continue;
    }
    if (!adj[static_cast<size_t>(last)][next]) {
      continue;
    }

    used[next] = 1u;
    path.push_back(static_cast<int>(next));

    if (dfs_pick_ring(world_size, adj, start, path, used)) {
      return true;
    }

    path.pop_back();
    used[next] = 0u;
  }

  return false;
}

static bool try_pick_sm100_or_sm103_ring(
    int32_t world_size,
    std::vector<int> const& candidates,
    std::vector<int>& devices_out) {

  devices_out.clear();

  if (world_size <= 0) {
    return false;
  }

  if (static_cast<int32_t>(candidates.size()) < world_size) {
    return false;
  }

  if (world_size == 1) {
    devices_out.assign(1, candidates[0]);
    return true;
  }

  size_t n = candidates.size();
  std::vector<std::vector<uint8_t>> adj(n, std::vector<uint8_t>(n, 0u));

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }

      if (p2p_pair_ok(candidates[i], candidates[j])) {
        adj[i][j] = 1u;
      }
    }
  }

  std::vector<int> path;
  std::vector<uint8_t> used;

  for (size_t start = 0; start < n; ++start) {
    path.clear();
    used.assign(n, 0u);

    path.reserve(static_cast<size_t>(world_size));
    used[start] = 1u;
    path.push_back(static_cast<int>(start));

    if (!dfs_pick_ring(world_size, adj, static_cast<int>(start), path, used)) {
      continue;
    }

    devices_out.resize(static_cast<size_t>(world_size));
    for (int i = 0; i < world_size; ++i) {
      devices_out[static_cast<size_t>(i)] = candidates[static_cast<size_t>(path[static_cast<size_t>(i)])];
    }

    // Enable peer access for the selected ring.
    RingAllreduceHostResult p2p = validate_ring_p2p_caps_and_enable_peer_access(world_size, devices_out.data());
    if (p2p.ok()) {
      return true;
    }

    devices_out.clear();
  }

  return false;
}

struct RingAllreduceTestAtomics {
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

__global__ void construct_ngpu_atomics_kernel(RingAllreduceTestAtomics a) {
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

__global__ void reset_ngpu_atomics_kernel(RingAllreduceTestAtomics a, uint32_t epoch) {
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

__global__ void destroy_ngpu_atomics_kernel(RingAllreduceTestAtomics a) {
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

__global__ void snapshot_ngpu_ag_ready_kernel(RingAllreduceTestAtomics a, uint32_t* ag_ready_out) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.flags_len) {
    if (ag_ready_out && a.self_ag_ready) {
      ag_ready_out[idx] = a.self_ag_ready[idx].load(cuda::memory_order_relaxed);
    }
  }
}

template <int kWorldSize, uint32_t kCountMult = 2>
static void run_ring_allreduce_correctness_test() {

  constexpr uint32_t kEpoch = 1;

  // Pick parameters that exercise:
  //  - multiple channels (channel_id != 0),
  //  - multiple tiles per chunk (tile_in_chunk != 0),
  //  - tail/partial tiles (chunk sizes not divisible by tile_elems),
  // so readiness-flag indexing covers tile_linear != 0.
  constexpr int32_t kNumChannels = 2;
  static_assert(kNumChannels > 1, "RingAllreduceNgpuAbort correctness test requires multiple channels");
  constexpr uint32_t kTileElems = 256;
  constexpr uint64_t kCount =
      uint64_t(kWorldSize) * uint64_t(kNumChannels) * uint64_t(kTileElems) * uint64_t(kCountMult) + 13ull;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  std::vector<int> candidates;
  candidates.reserve(static_cast<size_t>(device_count));

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_ngpu_abort_probe_launch(dev);
    if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
      continue;
    }
    ASSERT_EQ(probe_st, cudaSuccess);

    candidates.push_back(dev);
  }

  if (static_cast<int>(candidates.size()) < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " SM100/SM103 devices with a valid kernel image";
  }

  std::vector<int> devices;
  if (!try_pick_sm100_or_sm103_ring(kWorldSize, candidates, devices)) {
    GTEST_SKIP() << "no suitable P2P ring with native peer atomics";
  }

  std::array<int, kWorldSize> ring_devices{};
  for (int i = 0; i < kWorldSize; ++i) {
    ring_devices[static_cast<size_t>(i)] = devices[static_cast<size_t>(i)];
  }

  RingAllreduceTiling tiling{};
  auto tiling_r = validate_ring_allreduce_host_tiling(
      kCount,
      kWorldSize,
      kNumChannels,
      kTileElems,
      &tiling,
      ring_devices.data());
  ASSERT_TRUE(tiling_r.ok()) << (tiling_r.error_reason ? tiling_r.error_reason : "tiling validation failed");

  // Sanity-check that this test is exercising the intended multi-tile / multi-channel paths.
  ASSERT_GT(tiling.num_tiles_total, 1u);
  ASSERT_GT(tiling.tiles_per_chunk, 1u);

  uint64_t flags_len_u64 = uint64_t(kWorldSize) * uint64_t(tiling.num_tiles_total);
  ASSERT_LE(flags_len_u64, 0xffff'ffffull);
  uint32_t flags_len = static_cast<uint32_t>(flags_len_u64);

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((flags_len + kThreads - 1) / kThreads);
  // Ensure CTA0/thread0 runs scalar init even if flags_len==0.
  if (blocks == 0) {
    blocks = 1;
  }

  std::array<RingAllreduceTestAtomics, kWorldSize> atomics{};

  std::array<float*, kWorldSize> device_data{};
  std::array<uint32_t*, kWorldSize> device_status{};
  std::array<uint32_t, kWorldSize> host_status{};

  std::array<cudaStream_t, kWorldSize> streams{};
  std::array<cudaEvent_t, kWorldSize> done_events{};

  std::array<std::vector<float>, kWorldSize> host_in;
  std::array<std::vector<float>, kWorldSize> host_out;

  for (int r = 0; r < kWorldSize; ++r) {
    host_in[static_cast<size_t>(r)].resize(kCount);
    host_out[static_cast<size_t>(r)].resize(kCount);

    float scale = static_cast<float>(r + 1);
    for (uint64_t i = 0; i < kCount; ++i) {
      host_in[static_cast<size_t>(r)][i] = scale * static_cast<float>(i);
    }
  }

  for (int r = 0; r < kWorldSize; ++r) {
    int dev = ring_devices[static_cast<size_t>(r)];
    ASSERT_EQ(cudaSetDevice(dev), cudaSuccess);

    ASSERT_EQ(cudaStreamCreate(&streams[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(
        cudaEventCreateWithFlags(&done_events[static_cast<size_t>(r)], cudaEventDisableTiming),
        cudaSuccess);

    atomics[static_cast<size_t>(r)].flags_len = flags_len;

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_data[static_cast<size_t>(r)]), sizeof(float) * kCount), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status[static_cast<size_t>(r)]), sizeof(uint32_t)), cudaSuccess);

    ASSERT_EQ(cudaMemsetAsync(device_status[static_cast<size_t>(r)], 0xFF, sizeof(uint32_t), streams[static_cast<size_t>(r)]), cudaSuccess);

    ASSERT_EQ(
        cudaMemcpyAsync(
            device_data[static_cast<size_t>(r)],
            host_in[static_cast<size_t>(r)].data(),
            sizeof(float) * kCount,
            cudaMemcpyHostToDevice,
            streams[static_cast<size_t>(r)]),
        cudaSuccess);

    ASSERT_EQ(
        cudaMalloc(
            reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_rs_ready),
            sizeof(RingAllreduceSystemAtomicU32) * flags_len),
        cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(
            reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_ag_ready),
            sizeof(RingAllreduceSystemAtomicU32) * flags_len),
        cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_tiles_finished), sizeof(RingAllreduceDeviceAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_barrier_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_barrier_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_barrier_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_barrier_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    construct_ngpu_atomics_kernel<<<blocks, kThreads, 0, streams[static_cast<size_t>(r)]>>>(atomics[static_cast<size_t>(r)]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    reset_ngpu_atomics_kernel<<<blocks, kThreads, 0, streams[static_cast<size_t>(r)]>>>(atomics[static_cast<size_t>(r)], kEpoch);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  // Ensure per-rank initialization is complete before any cross-device accesses.
  for (int r = 0; r < kWorldSize; ++r) {
    ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[static_cast<size_t>(r)]), cudaSuccess);
  }

  for (int r = 0; r < kWorldSize; ++r) {
    RingAllreduceParams<float, 8> p{};

    p.world_size = kWorldSize;
    p.rank = r;
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

    p.self_data = device_data[static_cast<size_t>(r)];

    p.self_rs_ready = atomics[static_cast<size_t>(r)].self_rs_ready;
    p.self_ag_ready = atomics[static_cast<size_t>(r)].self_ag_ready;
    p.self_abort = atomics[static_cast<size_t>(r)].self_abort;
    p.self_error = atomics[static_cast<size_t>(r)].self_error;

    p.self_tiles_finished = atomics[static_cast<size_t>(r)].self_tiles_finished;

    p.self_barrier_gather_token = atomics[static_cast<size_t>(r)].self_barrier_gather_token;
    p.self_barrier_gather_status = atomics[static_cast<size_t>(r)].self_barrier_gather_status;
    p.self_barrier_release_token = atomics[static_cast<size_t>(r)].self_barrier_release_token;
    p.self_barrier_release_status = atomics[static_cast<size_t>(r)].self_barrier_release_status;

    if constexpr (kWorldSize > 1) {
      int left = (r + kWorldSize - 1) % kWorldSize;

      p.peer_data[left] = device_data[static_cast<size_t>(left)];
      p.peer_rs_ready[left] = atomics[static_cast<size_t>(left)].self_rs_ready;
      p.peer_ag_ready[left] = atomics[static_cast<size_t>(left)].self_ag_ready;
      p.peer_abort[left] = atomics[static_cast<size_t>(left)].self_abort;

      p.peer_barrier_gather_token[left] = atomics[static_cast<size_t>(left)].self_barrier_gather_token;
      p.peer_barrier_gather_status[left] = atomics[static_cast<size_t>(left)].self_barrier_gather_status;
      p.peer_barrier_release_token[left] = atomics[static_cast<size_t>(left)].self_barrier_release_token;
      p.peer_barrier_release_status[left] = atomics[static_cast<size_t>(left)].self_barrier_release_status;
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

    ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);

    cutlass::distributed::collective::ring_allreduce_sm100<float><<<tiling.num_tiles_total, kThreads, 0, streams[static_cast<size_t>(r)]>>>(
        p,
        device_status[static_cast<size_t>(r)]);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(done_events[static_cast<size_t>(r)], streams[static_cast<size_t>(r)]), cudaSuccess);
  }

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  std::vector<uint8_t> done(static_cast<size_t>(kWorldSize), 0u);

  while (true) {
    bool all_done = true;

    for (int r = 0; r < kWorldSize; ++r) {
      if (done[static_cast<size_t>(r)]) {
        continue;
      }

      all_done = false;

      ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);
      cudaError_t q = cudaEventQuery(done_events[static_cast<size_t>(r)]);
      if (q == cudaSuccess) {
        done[static_cast<size_t>(r)] = 1u;
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
      std::fprintf(
          stderr,
          "ring_allreduce_ngpu_abort_test: watchdog timeout (world_size=%d, devices=",
          kWorldSize);
      for (int i = 0; i < kWorldSize; ++i) {
        std::fprintf(
            stderr,
            "%d%s",
            ring_devices[static_cast<size_t>(i)],
            (i + 1 < kWorldSize) ? "," : "");
      }
      std::fprintf(stderr, ")\n");
      std::fflush(stderr);
      std::abort();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  for (int r = 0; r < kWorldSize; ++r) {
    ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[static_cast<size_t>(r)]), cudaSuccess);

    ASSERT_EQ(
        cudaMemcpy(
            &host_status[static_cast<size_t>(r)],
            device_status[static_cast<size_t>(r)],
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost),
        cudaSuccess);

    ASSERT_EQ(
        cudaMemcpy(
            host_out[static_cast<size_t>(r)].data(),
            device_data[static_cast<size_t>(r)],
            sizeof(float) * kCount,
            cudaMemcpyDeviceToHost),
        cudaSuccess);
  }

  for (int r = 0; r < kWorldSize; ++r) {
    EXPECT_EQ(host_status[static_cast<size_t>(r)], static_cast<uint32_t>(RingAllreduceError::kOk));
  }

  float coeff = static_cast<float>(kWorldSize * (kWorldSize + 1) / 2);
  for (uint64_t i = 0; i < kCount; ++i) {
    float expected = coeff * static_cast<float>(i);
    for (int r = 0; r < kWorldSize; ++r) {
      EXPECT_FLOAT_EQ(host_out[static_cast<size_t>(r)][i], expected) << "rank=" << r << " i=" << i;
    }
  }

  for (int r = 0; r < kWorldSize; ++r) {
    ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);

    destroy_ngpu_atomics_kernel<<<blocks, kThreads, 0, streams[static_cast<size_t>(r)]>>>(atomics[static_cast<size_t>(r)]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[static_cast<size_t>(r)]), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_barrier_release_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_barrier_release_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_barrier_gather_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_barrier_gather_token), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_tiles_finished), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_error), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_abort), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_ag_ready), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_rs_ready), cudaSuccess);

    ASSERT_EQ(cudaFree(device_status[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(cudaFree(device_data[static_cast<size_t>(r)]), cudaSuccess);

    ASSERT_EQ(cudaEventDestroy(done_events[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streams[static_cast<size_t>(r)]), cudaSuccess);
  }
}

static void run_ring_allreduce_abort_ag_publish_test(
    uint32_t abort_ag_step,
    uint32_t abort_before_ag_publish,
    uint32_t abort_after_ag_publish,
    std::initializer_list<ExpectedAgReadyValue> expected_ag_ready) {

  constexpr int kWorldSize = 4;
  constexpr uint32_t kEpoch = 1;

  // Single-tile configuration: num_tiles_total is expected to be small, keeping
  // flag snapshots minimal while still exercising the AG publish site.
  constexpr uint64_t kCount = 1024;
  constexpr int32_t kNumChannels = 1;
  constexpr uint32_t kTileElems = 256;

  // Exactly one of before/after must be enabled.
  ASSERT_NE(abort_before_ag_publish | abort_after_ag_publish, 0u);
  ASSERT_EQ((abort_before_ag_publish != 0u) + (abort_after_ag_publish != 0u), 1);

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  std::vector<int> candidates;
  candidates.reserve(static_cast<size_t>(device_count));

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_ngpu_abort_probe_launch(dev);
    if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
      continue;
    }
    ASSERT_EQ(probe_st, cudaSuccess);

    candidates.push_back(dev);
  }

  if (static_cast<int>(candidates.size()) < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " SM100/SM103 devices with a valid kernel image";
  }

  std::vector<int> devices;
  if (!try_pick_sm100_or_sm103_ring(kWorldSize, candidates, devices)) {
    GTEST_SKIP() << "no suitable P2P ring with native peer atomics";
  }

  std::array<int, kWorldSize> ring_devices{};
  for (int i = 0; i < kWorldSize; ++i) {
    ring_devices[static_cast<size_t>(i)] = devices[static_cast<size_t>(i)];
  }

  RingAllreduceTiling tiling{};
  auto tiling_r = validate_ring_allreduce_host_tiling(
      kCount,
      kWorldSize,
      kNumChannels,
      kTileElems,
      &tiling,
      ring_devices.data());
  ASSERT_TRUE(tiling_r.ok()) << (tiling_r.error_reason ? tiling_r.error_reason : "tiling validation failed");

  uint64_t flags_len_u64 = uint64_t(kWorldSize) * uint64_t(tiling.num_tiles_total);
  ASSERT_LE(flags_len_u64, 0xffff'ffffull);
  uint32_t flags_len = static_cast<uint32_t>(flags_len_u64);

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((flags_len + kThreads - 1) / kThreads);
  // Ensure CTA0/thread0 runs scalar init even if flags_len==0.
  if (blocks == 0) {
    blocks = 1;
  }

  std::array<RingAllreduceTestAtomics, kWorldSize> atomics{};

  std::array<float*, kWorldSize> device_data{};
  std::array<uint32_t*, kWorldSize> device_status{};
  std::array<uint32_t, kWorldSize> host_status{};

  std::array<cudaStream_t, kWorldSize> streams{};
  std::array<cudaEvent_t, kWorldSize> done_events{};

  for (int r = 0; r < kWorldSize; ++r) {
    ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);

    ASSERT_EQ(cudaStreamCreate(&streams[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(
        cudaEventCreateWithFlags(&done_events[static_cast<size_t>(r)], cudaEventDisableTiming),
        cudaSuccess);

    atomics[static_cast<size_t>(r)].flags_len = flags_len;

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_data[static_cast<size_t>(r)]), sizeof(float) * kCount), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status[static_cast<size_t>(r)]), sizeof(uint32_t)), cudaSuccess);

    ASSERT_EQ(cudaMemsetAsync(device_status[static_cast<size_t>(r)], 0xFF, sizeof(uint32_t), streams[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(device_data[static_cast<size_t>(r)], 0, sizeof(float) * kCount, streams[static_cast<size_t>(r)]), cudaSuccess);

    ASSERT_EQ(
        cudaMalloc(
            reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_rs_ready),
            sizeof(RingAllreduceSystemAtomicU32) * flags_len),
        cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(
            reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_ag_ready),
            sizeof(RingAllreduceSystemAtomicU32) * flags_len),
        cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_tiles_finished), sizeof(RingAllreduceDeviceAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_barrier_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_barrier_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_barrier_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_barrier_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    construct_ngpu_atomics_kernel<<<blocks, kThreads, 0, streams[static_cast<size_t>(r)]>>>(atomics[static_cast<size_t>(r)]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    reset_ngpu_atomics_kernel<<<blocks, kThreads, 0, streams[static_cast<size_t>(r)]>>>(atomics[static_cast<size_t>(r)], kEpoch);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  // Ensure per-rank initialization is complete before any cross-device accesses.
  for (int r = 0; r < kWorldSize; ++r) {
    ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[static_cast<size_t>(r)]), cudaSuccess);
  }

  for (int r = 0; r < kWorldSize; ++r) {
    RingAllreduceParams<float, 8> p{};

    p.world_size = kWorldSize;
    p.rank = r;
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

    p.self_data = device_data[static_cast<size_t>(r)];

    p.self_rs_ready = atomics[static_cast<size_t>(r)].self_rs_ready;
    p.self_ag_ready = atomics[static_cast<size_t>(r)].self_ag_ready;
    p.self_abort = atomics[static_cast<size_t>(r)].self_abort;
    p.self_error = atomics[static_cast<size_t>(r)].self_error;

    p.self_tiles_finished = atomics[static_cast<size_t>(r)].self_tiles_finished;

    p.self_barrier_gather_token = atomics[static_cast<size_t>(r)].self_barrier_gather_token;
    p.self_barrier_gather_status = atomics[static_cast<size_t>(r)].self_barrier_gather_status;
    p.self_barrier_release_token = atomics[static_cast<size_t>(r)].self_barrier_release_token;
    p.self_barrier_release_status = atomics[static_cast<size_t>(r)].self_barrier_release_status;

    int left = (r + kWorldSize - 1) % kWorldSize;

    p.peer_data[left] = device_data[static_cast<size_t>(left)];
    p.peer_rs_ready[left] = atomics[static_cast<size_t>(left)].self_rs_ready;
    p.peer_ag_ready[left] = atomics[static_cast<size_t>(left)].self_ag_ready;
    p.peer_abort[left] = atomics[static_cast<size_t>(left)].self_abort;

    p.peer_barrier_gather_token[left] = atomics[static_cast<size_t>(left)].self_barrier_gather_token;
    p.peer_barrier_gather_status[left] = atomics[static_cast<size_t>(left)].self_barrier_gather_status;
    p.peer_barrier_release_token[left] = atomics[static_cast<size_t>(left)].self_barrier_release_token;
    p.peer_barrier_release_status[left] = atomics[static_cast<size_t>(left)].self_barrier_release_status;

    // Inject an abort around the AG forwarding publish site on rank0 (CTA0 only).
    p.debug_abort_rank = 0u;
    p.debug_abort_ag_step = abort_ag_step;
    p.debug_abort_before_ag_publish = abort_before_ag_publish;
    p.debug_abort_after_ag_publish = abort_after_ag_publish;

    p.debug_release_delay_rank = 0xffff'ffffu;
    p.debug_release_delay_iters = 0u;

    p.debug_jitter_seed = 0u;
    p.debug_jitter_max_iters = 0u;
    p.debug_jitter_mask = 0u;

    ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);

    cutlass::distributed::collective::ring_allreduce_sm100<float><<<tiling.num_tiles_total, kThreads, 0, streams[static_cast<size_t>(r)]>>>(
        p,
        device_status[static_cast<size_t>(r)]);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(done_events[static_cast<size_t>(r)], streams[static_cast<size_t>(r)]), cudaSuccess);
  }

  // Host watchdog: ensure no hangs.
  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  std::vector<uint8_t> done(static_cast<size_t>(kWorldSize), 0u);
  while (true) {
    bool all_done = true;

    for (int r = 0; r < kWorldSize; ++r) {
      if (done[static_cast<size_t>(r)]) {
        continue;
      }

      all_done = false;

      ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);
      cudaError_t q = cudaEventQuery(done_events[static_cast<size_t>(r)]);
      if (q == cudaSuccess) {
        done[static_cast<size_t>(r)] = 1u;
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
      std::fprintf(
          stderr,
          "ring_allreduce_ngpu_abort_test: watchdog timeout (world_size=%d, devices=",
          kWorldSize);
      for (int i = 0; i < kWorldSize; ++i) {
        std::fprintf(
            stderr,
            "%d%s",
            ring_devices[static_cast<size_t>(i)],
            (i + 1 < kWorldSize) ? "," : "");
      }
      std::fprintf(stderr, ")\n");
      std::fflush(stderr);
      std::abort();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  for (int r = 0; r < kWorldSize; ++r) {
    ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[static_cast<size_t>(r)]), cudaSuccess);

    ASSERT_EQ(
        cudaMemcpy(
            &host_status[static_cast<size_t>(r)],
            device_status[static_cast<size_t>(r)],
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
  }

  for (int r = 0; r < kWorldSize; ++r) {
    EXPECT_NE(host_status[static_cast<size_t>(r)], static_cast<uint32_t>(RingAllreduceError::kOk));
  }

  // Publish-side assertions: abort injection is CTA0-only, so sample tile_linear=0.
  {
    std::vector<uint32_t> host_ag_ready(static_cast<size_t>(flags_len), 0xffff'ffffu);
    uint32_t* device_ag_ready_out = nullptr;

    constexpr size_t kAbortRank = 0;
    ASSERT_EQ(cudaSetDevice(ring_devices[kAbortRank]), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_ag_ready_out), sizeof(uint32_t) * flags_len), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(device_ag_ready_out, 0xFF, sizeof(uint32_t) * flags_len, streams[kAbortRank]), cudaSuccess);

    snapshot_ngpu_ag_ready_kernel<<<blocks, kThreads, 0, streams[kAbortRank]>>>(
        atomics[kAbortRank],
        device_ag_ready_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[kAbortRank]), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(host_ag_ready.data(), device_ag_ready_out, sizeof(uint32_t) * flags_len, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(device_ag_ready_out), cudaSuccess);

    for (auto const& exp : expected_ag_ready) {
      uint64_t idx_u64 = uint64_t(exp.step) * uint64_t(tiling.num_tiles_total) + 0ull;
      ASSERT_LT(idx_u64, flags_len_u64);
      EXPECT_EQ(host_ag_ready[static_cast<size_t>(idx_u64)], exp.value);
    }
  }

  for (int r = 0; r < kWorldSize; ++r) {
    ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);

    destroy_ngpu_atomics_kernel<<<blocks, kThreads, 0, streams[static_cast<size_t>(r)]>>>(atomics[static_cast<size_t>(r)]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[static_cast<size_t>(r)]), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_barrier_release_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_barrier_release_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_barrier_gather_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_barrier_gather_token), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_tiles_finished), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_error), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_abort), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_ag_ready), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_rs_ready), cudaSuccess);

    ASSERT_EQ(cudaFree(device_status[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(cudaFree(device_data[static_cast<size_t>(r)]), cudaSuccess);

    ASSERT_EQ(cudaEventDestroy(done_events[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streams[static_cast<size_t>(r)]), cudaSuccess);
  }
}

static void run_ring_allreduce_jitter_stress_test(uint64_t kCount, uint32_t kIterations) {

  constexpr int kWorldSize = 4;

  // Keep this test reasonably small while still exercising:
  //  - multiple channels (channel_id != 0),
  //  - multiple tiles per chunk (tile_in_chunk != 0),
  //  - tail/partial tiles,
  //  - multiple RS/AG steps (N-1 for N=4).
  constexpr int32_t kNumChannels = 2;
  constexpr uint32_t kTileElems = 256;

  // Stress parameters: caller controls tensor size and iteration count.

  // Enable all four jitter sites (RS/AG after-wait + before-publish).
  constexpr uint32_t kJitterMask = 0xFu;
  constexpr uint32_t kJitterMaxIters = 256u;
  constexpr uint32_t kJitterBaseSeed = 0xC001D00Du;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  std::vector<int> candidates;
  candidates.reserve(static_cast<size_t>(device_count));

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_ngpu_abort_probe_launch(dev);
    if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
      continue;
    }
    ASSERT_EQ(probe_st, cudaSuccess);

    candidates.push_back(dev);
  }

  if (static_cast<int>(candidates.size()) < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " SM100/SM103 devices with a valid kernel image";
  }

  std::vector<int> devices;
  if (!try_pick_sm100_or_sm103_ring(kWorldSize, candidates, devices)) {
    GTEST_SKIP() << "no suitable P2P ring with native peer atomics";
  }

  std::array<int, kWorldSize> ring_devices{};
  for (int i = 0; i < kWorldSize; ++i) {
    ring_devices[static_cast<size_t>(i)] = devices[static_cast<size_t>(i)];
  }

  RingAllreduceTiling tiling{};
  auto tiling_r = validate_ring_allreduce_host_tiling(
      kCount,
      kWorldSize,
      kNumChannels,
      kTileElems,
      &tiling,
      ring_devices.data());
  ASSERT_TRUE(tiling_r.ok()) << (tiling_r.error_reason ? tiling_r.error_reason : "tiling validation failed");

  // Ensure this test actually exercises multi-CTA paths.
  ASSERT_GT(tiling.num_tiles_total, 1u);
  ASSERT_GT(tiling.tiles_per_chunk, 1u);

  uint64_t flags_len_u64 = uint64_t(kWorldSize) * uint64_t(tiling.num_tiles_total);
  ASSERT_LE(flags_len_u64, 0xffff'ffffull);
  uint32_t flags_len = static_cast<uint32_t>(flags_len_u64);

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((flags_len + kThreads - 1) / kThreads);
  // Ensure CTA0/thread0 runs scalar init even if flags_len==0.
  if (blocks == 0) {
    blocks = 1;
  }

  std::array<RingAllreduceTestAtomics, kWorldSize> atomics{};

  std::array<float*, kWorldSize> device_data{};
  std::array<uint32_t*, kWorldSize> device_status{};
  std::array<uint32_t, kWorldSize> host_status{};

  std::array<cudaStream_t, kWorldSize> streams{};
  std::array<cudaEvent_t, kWorldSize> done_events{};

  std::array<std::vector<float>, kWorldSize> host_in;
  std::array<std::vector<float>, kWorldSize> host_out;

  for (int r = 0; r < kWorldSize; ++r) {
    host_in[static_cast<size_t>(r)].resize(kCount);
    host_out[static_cast<size_t>(r)].resize(kCount);

    float scale = static_cast<float>(r + 1);
    for (uint64_t i = 0; i < kCount; ++i) {
      host_in[static_cast<size_t>(r)][i] = scale * static_cast<float>(i);
    }
  }

  for (int r = 0; r < kWorldSize; ++r) {
    int dev = ring_devices[static_cast<size_t>(r)];
    ASSERT_EQ(cudaSetDevice(dev), cudaSuccess);

    ASSERT_EQ(cudaStreamCreate(&streams[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(
        cudaEventCreateWithFlags(&done_events[static_cast<size_t>(r)], cudaEventDisableTiming),
        cudaSuccess);

    atomics[static_cast<size_t>(r)].flags_len = flags_len;

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_data[static_cast<size_t>(r)]), sizeof(float) * kCount), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status[static_cast<size_t>(r)]), sizeof(uint32_t)), cudaSuccess);

    ASSERT_EQ(cudaMemsetAsync(device_status[static_cast<size_t>(r)], 0xFF, sizeof(uint32_t), streams[static_cast<size_t>(r)]), cudaSuccess);

    ASSERT_EQ(
        cudaMemcpyAsync(
            device_data[static_cast<size_t>(r)],
            host_in[static_cast<size_t>(r)].data(),
            sizeof(float) * kCount,
            cudaMemcpyHostToDevice,
            streams[static_cast<size_t>(r)]),
        cudaSuccess);

    ASSERT_EQ(
        cudaMalloc(
            reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_rs_ready),
            sizeof(RingAllreduceSystemAtomicU32) * flags_len),
        cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(
            reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_ag_ready),
            sizeof(RingAllreduceSystemAtomicU32) * flags_len),
        cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_tiles_finished), sizeof(RingAllreduceDeviceAtomicU32)), cudaSuccess);

    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_barrier_gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_barrier_gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_barrier_release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&atomics[static_cast<size_t>(r)].self_barrier_release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

    construct_ngpu_atomics_kernel<<<blocks, kThreads, 0, streams[static_cast<size_t>(r)]>>>(atomics[static_cast<size_t>(r)]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    reset_ngpu_atomics_kernel<<<blocks, kThreads, 0, streams[static_cast<size_t>(r)]>>>(atomics[static_cast<size_t>(r)], /*epoch=*/1u);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  // Ensure per-rank initialization is complete before any cross-device accesses.
  for (int r = 0; r < kWorldSize; ++r) {
    ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[static_cast<size_t>(r)]), cudaSuccess);
  }

  // Base parameter set-up (per-rank pointers do not change across iterations).
  std::array<RingAllreduceParams<float, 8>, kWorldSize> base_params{};

  for (int r = 0; r < kWorldSize; ++r) {
    RingAllreduceParams<float, 8> p{};

    p.world_size = kWorldSize;
    p.rank = r;

    p.count = kCount;
    p.num_channels = kNumChannels;

    p.tile_elems = tiling.tile_elems;
    p.num_chunks_total = tiling.num_chunks_total;
    p.max_chunk_elems = tiling.max_chunk_elems;
    p.tiles_per_chunk = tiling.tiles_per_chunk;
    p.num_tiles_total = tiling.num_tiles_total;

    // Generous timeouts + host watchdog: avoid flaky timeouts under stress.
    p.timeout_iters = 1u << 20;
    p.timeout_cycles = 0;
    p.poll_sleep_start = 0;
    p.poll_sleep_ns = 0;

    p.self_data = device_data[static_cast<size_t>(r)];

    p.self_rs_ready = atomics[static_cast<size_t>(r)].self_rs_ready;
    p.self_ag_ready = atomics[static_cast<size_t>(r)].self_ag_ready;
    p.self_abort = atomics[static_cast<size_t>(r)].self_abort;
    p.self_error = atomics[static_cast<size_t>(r)].self_error;

    p.self_tiles_finished = atomics[static_cast<size_t>(r)].self_tiles_finished;

    p.self_barrier_gather_token = atomics[static_cast<size_t>(r)].self_barrier_gather_token;
    p.self_barrier_gather_status = atomics[static_cast<size_t>(r)].self_barrier_gather_status;
    p.self_barrier_release_token = atomics[static_cast<size_t>(r)].self_barrier_release_token;
    p.self_barrier_release_status = atomics[static_cast<size_t>(r)].self_barrier_release_status;

    int left = (r + kWorldSize - 1) % kWorldSize;

    p.peer_data[left] = device_data[static_cast<size_t>(left)];
    p.peer_rs_ready[left] = atomics[static_cast<size_t>(left)].self_rs_ready;
    p.peer_ag_ready[left] = atomics[static_cast<size_t>(left)].self_ag_ready;
    p.peer_abort[left] = atomics[static_cast<size_t>(left)].self_abort;

    p.peer_barrier_gather_token[left] = atomics[static_cast<size_t>(left)].self_barrier_gather_token;
    p.peer_barrier_gather_status[left] = atomics[static_cast<size_t>(left)].self_barrier_gather_status;
    p.peer_barrier_release_token[left] = atomics[static_cast<size_t>(left)].self_barrier_release_token;
    p.peer_barrier_release_status[left] = atomics[static_cast<size_t>(left)].self_barrier_release_status;

    // No abort injection.
    p.debug_abort_rank = 0xffff'ffffu;
    p.debug_abort_ag_step = 0u;
    p.debug_abort_before_ag_publish = 0u;
    p.debug_abort_after_ag_publish = 0u;

    p.debug_release_delay_rank = 0xffff'ffffu;
    p.debug_release_delay_iters = 0u;

    base_params[static_cast<size_t>(r)] = p;
  }

  float coeff = static_cast<float>(kWorldSize * (kWorldSize + 1) / 2);

  for (uint32_t iter = 0; iter < kIterations; ++iter) {
    uint32_t epoch = 1u + iter;

    // Reset atomics + payload for a clean iteration.
    for (int r = 0; r < kWorldSize; ++r) {
      ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);

      reset_ngpu_atomics_kernel<<<blocks, kThreads, 0, streams[static_cast<size_t>(r)]>>>(
          atomics[static_cast<size_t>(r)],
          epoch);
      ASSERT_EQ(cudaGetLastError(), cudaSuccess);

      ASSERT_EQ(cudaMemsetAsync(device_status[static_cast<size_t>(r)], 0xFF, sizeof(uint32_t), streams[static_cast<size_t>(r)]), cudaSuccess);

      ASSERT_EQ(
          cudaMemcpyAsync(
              device_data[static_cast<size_t>(r)],
              host_in[static_cast<size_t>(r)].data(),
              sizeof(float) * kCount,
              cudaMemcpyHostToDevice,
              streams[static_cast<size_t>(r)]),
          cudaSuccess);
    }

    // Ensure resets are globally visible before launching the multi-GPU kernel.
    for (int r = 0; r < kWorldSize; ++r) {
      ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);
      ASSERT_EQ(cudaStreamSynchronize(streams[static_cast<size_t>(r)]), cudaSuccess);
    }

    for (int r = 0; r < kWorldSize; ++r) {
      RingAllreduceParams<float, 8> p = base_params[static_cast<size_t>(r)];

      p.epoch = epoch;

      // Deterministic jitter: vary the seed per-iteration, while rank/tile/step
      // are mixed in device-side.
      p.debug_jitter_seed = kJitterBaseSeed ^ (epoch * 0x9e3779b9u);
      p.debug_jitter_max_iters = kJitterMaxIters;
      p.debug_jitter_mask = kJitterMask;

      ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);

      cutlass::distributed::collective::ring_allreduce_sm100<float><<<tiling.num_tiles_total, kThreads, 0, streams[static_cast<size_t>(r)]>>>(
          p,
          device_status[static_cast<size_t>(r)]);

      ASSERT_EQ(cudaGetLastError(), cudaSuccess);
      ASSERT_EQ(cudaEventRecord(done_events[static_cast<size_t>(r)], streams[static_cast<size_t>(r)]), cudaSuccess);
    }

    // Host watchdog: ensure no hangs.
    constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
    auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

    std::vector<uint8_t> done(static_cast<size_t>(kWorldSize), 0u);

    while (true) {
      bool all_done = true;

      for (int r = 0; r < kWorldSize; ++r) {
        if (done[static_cast<size_t>(r)]) {
          continue;
        }

        all_done = false;

        ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);
        cudaError_t q = cudaEventQuery(done_events[static_cast<size_t>(r)]);
        if (q == cudaSuccess) {
          done[static_cast<size_t>(r)] = 1u;
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
        // Hard abort: a GPU hang can leave the process wedged in CUDA runtime
        // state; aborting avoids a deadlocked test runner.
        std::fprintf(stderr, "ring_allreduce_ngpu_abort_test: watchdog timeout (jitter iter=%u)\n", iter);
        std::fflush(stderr);
        std::abort();
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    for (int r = 0; r < kWorldSize; ++r) {
      ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);
      ASSERT_EQ(cudaStreamSynchronize(streams[static_cast<size_t>(r)]), cudaSuccess);

      ASSERT_EQ(
          cudaMemcpy(
              &host_status[static_cast<size_t>(r)],
              device_status[static_cast<size_t>(r)],
              sizeof(uint32_t),
              cudaMemcpyDeviceToHost),
          cudaSuccess);

      ASSERT_EQ(
          cudaMemcpy(
              host_out[static_cast<size_t>(r)].data(),
              device_data[static_cast<size_t>(r)],
              sizeof(float) * kCount,
              cudaMemcpyDeviceToHost),
          cudaSuccess);
    }

    for (int r = 0; r < kWorldSize; ++r) {
      ASSERT_EQ(
          host_status[static_cast<size_t>(r)],
          static_cast<uint32_t>(RingAllreduceError::kOk))
          << "iter=" << iter << " rank=" << r;
    }

    uint64_t mismatch_count = 0;
    int first_mismatch_rank = -1;
    uint64_t first_mismatch_i = 0;
    float first_mismatch_got = 0.0f;
    float first_mismatch_expected = 0.0f;

    // Inputs are small integer-valued FP32 and the full sum stays < 2^24, so
    // FP32 arithmetic is exact here. Use exact equality to catch any overwrite.
    for (uint64_t i = 0; i < kCount; ++i) {
      float expected = coeff * static_cast<float>(i);
      for (int r = 0; r < kWorldSize; ++r) {
        float got = host_out[static_cast<size_t>(r)][i];
        if (got != expected) {
          if (mismatch_count == 0) {
            first_mismatch_rank = r;
            first_mismatch_i = i;
            first_mismatch_got = got;
            first_mismatch_expected = expected;
          }
          ++mismatch_count;
        }
      }
    }

    ASSERT_EQ(mismatch_count, 0ull)
        << "overwrite-safety mismatch under kOk (iter=" << iter
        << ", first_rank=" << first_mismatch_rank
        << ", first_i=" << first_mismatch_i
        << ", got=" << first_mismatch_got
        << ", expected=" << first_mismatch_expected
        << ", mismatches=" << mismatch_count << ")";
  }

  for (int r = 0; r < kWorldSize; ++r) {
    ASSERT_EQ(cudaSetDevice(ring_devices[static_cast<size_t>(r)]), cudaSuccess);

    destroy_ngpu_atomics_kernel<<<blocks, kThreads, 0, streams[static_cast<size_t>(r)]>>>(atomics[static_cast<size_t>(r)]);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[static_cast<size_t>(r)]), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_barrier_release_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_barrier_release_token), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_barrier_gather_status), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_barrier_gather_token), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_tiles_finished), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_error), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_abort), cudaSuccess);

    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_ag_ready), cudaSuccess);
    ASSERT_EQ(cudaFree(atomics[static_cast<size_t>(r)].self_rs_ready), cudaSuccess);

    ASSERT_EQ(cudaFree(device_status[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(cudaFree(device_data[static_cast<size_t>(r)]), cudaSuccess);

    ASSERT_EQ(cudaEventDestroy(done_events[static_cast<size_t>(r)]), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streams[static_cast<size_t>(r)]), cudaSuccess);
  }
}

} // namespace

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, S0_HarnessAlive, {

  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  // M6b scaffold: validate multi-GPU orchestration and P2P-ring selection.
  // Abort publish-side assertions are covered by A0/A1 tests below.

  constexpr int kWorldSize = 4;
  constexpr uint32_t kMagicStatus = 0xA11CE0DEu;

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " CUDA devices";
  }

  std::vector<int> candidates;
  candidates.reserve(static_cast<size_t>(device_count));

  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_ngpu_abort_probe_launch(dev);
    if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
      continue;
    }
    ASSERT_EQ(probe_st, cudaSuccess);

    candidates.push_back(dev);
  }

  if (static_cast<int>(candidates.size()) < kWorldSize) {
    GTEST_SKIP() << "requires >= " << kWorldSize << " SM100/SM103 devices with a valid kernel image";
  }

  std::vector<int> devices;
  if (!try_pick_sm100_or_sm103_ring(kWorldSize, candidates, devices)) {
    GTEST_SKIP() << "no suitable P2P ring with native peer atomics";
  }

  std::vector<uint32_t> host_status(kWorldSize, 0xffff'ffffu);
  std::vector<uint32_t*> device_status(kWorldSize, nullptr);

  std::vector<cudaStream_t> streams(kWorldSize);
  std::vector<cudaEvent_t> done_events(kWorldSize);

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[static_cast<size_t>(i)]), cudaSuccess);

    ASSERT_EQ(cudaStreamCreate(&streams[static_cast<size_t>(i)]), cudaSuccess);
    ASSERT_EQ(
        cudaEventCreateWithFlags(&done_events[static_cast<size_t>(i)], cudaEventDisableTiming),
        cudaSuccess);

    ASSERT_EQ(
        cudaMalloc(reinterpret_cast<void**>(&device_status[static_cast<size_t>(i)]), sizeof(uint32_t)),
        cudaSuccess);

    ASSERT_EQ(
        cudaMemsetAsync(device_status[static_cast<size_t>(i)], 0xFF, sizeof(uint32_t), streams[static_cast<size_t>(i)]),
        cudaSuccess);

    ring_allreduce_ngpu_abort_noop_kernel<<<1, 1, 0, streams[static_cast<size_t>(i)]>>>(
        kMagicStatus,
        device_status[static_cast<size_t>(i)]);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(done_events[static_cast<size_t>(i)], streams[static_cast<size_t>(i)]), cudaSuccess);
  }

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  std::vector<uint8_t> done(kWorldSize, 0u);

  while (true) {
    bool all_done = true;

    for (int i = 0; i < kWorldSize; ++i) {
      if (done[static_cast<size_t>(i)]) {
        continue;
      }

      all_done = false;

      ASSERT_EQ(cudaSetDevice(devices[static_cast<size_t>(i)]), cudaSuccess);
      cudaError_t q = cudaEventQuery(done_events[static_cast<size_t>(i)]);
      if (q == cudaSuccess) {
        done[static_cast<size_t>(i)] = 1u;
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
      std::fprintf(
          stderr,
          "ring_allreduce_ngpu_abort_test: watchdog timeout (world_size=%d, devices=",
          kWorldSize);
      for (int i = 0; i < kWorldSize; ++i) {
        std::fprintf(stderr, "%d%s", devices[static_cast<size_t>(i)], (i + 1 < kWorldSize) ? "," : "");
      }
      std::fprintf(stderr, ")\n");
      std::fflush(stderr);
      std::abort();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[static_cast<size_t>(i)]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[static_cast<size_t>(i)]), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(
            &host_status[static_cast<size_t>(i)],
            device_status[static_cast<size_t>(i)],
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
  }

  for (int i = 0; i < kWorldSize; ++i) {
    EXPECT_EQ(host_status[static_cast<size_t>(i)], kMagicStatus);
  }

  // M6f abort-before/after-publish coverage is provided by A0/A1 below.

  for (int i = 0; i < kWorldSize; ++i) {
    ASSERT_EQ(cudaSetDevice(devices[static_cast<size_t>(i)]), cudaSuccess);

    ASSERT_EQ(cudaFree(device_status[static_cast<size_t>(i)]), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(done_events[static_cast<size_t>(i)]), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(streams[static_cast<size_t>(i)]), cudaSuccess);
  }
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, R0_Correctness_N1_NoOp, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  run_ring_allreduce_correctness_test<1>();
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, R5_Correctness_N2, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  run_ring_allreduce_correctness_test<2>();
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, R6_Correctness_N2_Small, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  run_ring_allreduce_correctness_test<2, 1>();
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, R7_Correctness_N2_Large, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  run_ring_allreduce_correctness_test<2, 16>();
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, R1_Correctness_N4, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  run_ring_allreduce_correctness_test<4>();
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, R3_Correctness_N4_Small, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  run_ring_allreduce_correctness_test<4, 1>();
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, R4_Correctness_N4_Large, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  run_ring_allreduce_correctness_test<4, 16>();
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, R2_Correctness_N8, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  run_ring_allreduce_correctness_test<8>();
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, R8_Correctness_N8_Small, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  run_ring_allreduce_correctness_test<8, 1>();
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, R9_Correctness_N8_Large, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  run_ring_allreduce_correctness_test<8, 16>();
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, A0_AbortBeforeAgPublish_N4, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  // A0: abort-before-publish at AG step s=0 on rank0.
  run_ring_allreduce_abort_ag_publish_test(
      /*abort_ag_step=*/0u,
      /*abort_before_ag_publish=*/1u,
      /*abort_after_ag_publish=*/0u,
      {
          {0u, 1u},
          {1u, 0u},
      });
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, A1_AbortAfterAgPublish_N4, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  // A1: abort-after-publish at the final AG step (N=4 => s=2) on rank0.
  run_ring_allreduce_abort_ag_publish_test(
      /*abort_ag_step=*/2u,
      /*abort_before_ag_publish=*/0u,
      /*abort_after_ag_publish=*/1u,
      {
          {0u, 1u},
          {1u, 1u},
          {2u, 1u},
          {3u, 1u},
      });
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, J0_JitterStress_N4, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  constexpr int kWorldSize = 4;
  constexpr int32_t kNumChannels = 2;
  constexpr uint32_t kTileElems = 256;

  constexpr uint64_t kCountMid =
      uint64_t(kWorldSize) * uint64_t(kNumChannels) * uint64_t(kTileElems) * 2ull + 13ull;

  run_ring_allreduce_jitter_stress_test(kCountMid, /*kIterations=*/100u);
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, J1_JitterStress_N4_Small, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  constexpr int kWorldSize = 4;
  constexpr int32_t kNumChannels = 2;
  constexpr uint32_t kTileElems = 256;

  constexpr uint64_t kCountSmall =
      uint64_t(kWorldSize) * uint64_t(kNumChannels) * uint64_t(kTileElems) * 1ull + 13ull;

  run_ring_allreduce_jitter_stress_test(kCountSmall, /*kIterations=*/20u);
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, J2_JitterStress_N4_Large, {
  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  constexpr int kWorldSize = 4;
  constexpr int32_t kNumChannels = 2;
  constexpr uint32_t kTileElems = 256;

  constexpr uint64_t kCountLarge =
      uint64_t(kWorldSize) * uint64_t(kNumChannels) * uint64_t(kTileElems) * 16ull + 13ull;

  run_ring_allreduce_jitter_stress_test(kCountLarge, /*kIterations=*/20u);
});

CUTLASS_TEST_L2(RingAllreduceNgpuAbort, I0_InvalidWorldSize3NoHang, {

  ScopedCudaDeviceRestore original_device;
  if (!original_device.ok()) {
    GTEST_SKIP() << "cudaGetDevice failed";
  }

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < 1) {
    GTEST_SKIP() << "requires >= 1 CUDA device";
  }

  int device = -1;
  for (int dev = 0; dev < device_count; ++dev) {
    if (!is_sm100_or_sm103(dev)) {
      continue;
    }

    cudaError_t probe_st = ring_allreduce_ngpu_abort_probe_launch(dev);
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

  constexpr int32_t kWorldSize = 3;
  constexpr uint32_t kEpoch = 1;

  constexpr uint32_t kNumTilesTotal = 1;
  constexpr uint32_t kFlagsLen = static_cast<uint32_t>(kWorldSize) * kNumTilesTotal;

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((kFlagsLen + kThreads - 1) / kThreads);
  if (blocks == 0) {
    blocks = 1;
  }

  RingAllreduceTestAtomics atomics{};
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

  construct_ngpu_atomics_kernel<<<blocks, kThreads, 0, stream>>>(atomics);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  reset_ngpu_atomics_kernel<<<blocks, kThreads, 0, stream>>>(atomics, kEpoch);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  uint8_t* self_data = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&self_data), sizeof(uint8_t)), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(self_data, 0, sizeof(uint8_t), stream), cudaSuccess);

  uint32_t* device_status = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status), sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(device_status, 0xFF, sizeof(uint32_t), stream), cudaSuccess);

  RingAllreduceParams<uint8_t, 8> p{};
  p.world_size = kWorldSize;
  p.rank = 0;
  p.epoch = kEpoch;

  // Minimal tiling for a single CTA.
  p.count = 1;
  p.num_channels = 1;
  p.tile_elems = 1;
  p.num_chunks_total = static_cast<uint32_t>(kWorldSize);
  p.max_chunk_elems = 1;
  p.tiles_per_chunk = 1;
  p.num_tiles_total = kNumTilesTotal;

  // Timeouts disabled: invalid-params paths must still complete.
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

  p.debug_abort_rank = 0xffff'ffffu;
  p.debug_abort_ag_step = 0u;
  p.debug_abort_before_ag_publish = 0u;
  p.debug_abort_after_ag_publish = 0u;

  p.debug_release_delay_rank = 0xffff'ffffu;
  p.debug_release_delay_iters = 0u;

  p.debug_jitter_seed = 0u;
  p.debug_jitter_max_iters = 0u;
  p.debug_jitter_mask = 0u;

  cutlass::distributed::collective::ring_allreduce_sm100<uint8_t><<<kNumTilesTotal, kThreads, 0, stream>>>(p, device_status);
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
      std::fprintf(stderr, "ring_allreduce_ngpu_abort_test: watchdog timeout (world_size=3 invalid params)\n");
      std::abort();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  uint32_t host_status = 0u;
  ASSERT_EQ(cudaMemcpy(&host_status, device_status, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(host_status, static_cast<uint32_t>(RingAllreduceError::kInvalidParams));

  destroy_ngpu_atomics_kernel<<<blocks, kThreads, 0, stream>>>(atomics);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

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
