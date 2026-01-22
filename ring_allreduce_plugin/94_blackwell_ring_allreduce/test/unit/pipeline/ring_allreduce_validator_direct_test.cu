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
    \brief Validator-direct unit tests for ring_allreduce_params_valid_ngpu.
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/experimental/distributed/collective/ring_allreduce_kernel_sm100.cuh"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <new>

namespace {

using cutlass::distributed::collective::RingAllreduceDeviceAtomicU32;
using cutlass::distributed::collective::RingAllreduceParams;
using cutlass::distributed::collective::RingAllreduceSystemAtomicU32;

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

static bool is_sm100_or_sm103(int device) {
  cudaDeviceProp prop{};
  cudaError_t st = cudaGetDeviceProperties(&prop, device);
  if (st != cudaSuccess) {
    return false;
  }

  int cc = prop.major * 10 + prop.minor;
  return cc == 100 || cc == 103;
}

__global__ void ring_allreduce_validator_probe_kernel() {}

static cudaError_t ring_allreduce_validator_probe_launch(int device) {
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

  ring_allreduce_validator_probe_kernel<<<1, 1>>>();
  cudaError_t probe_st = cudaGetLastError();

  // Restore the device to avoid leaking CUDA global state into other tests.
  (void)cudaSetDevice(original_device);

  return probe_st;
}

struct RingAllreduceValidatorPtrs {
  uint8_t* self_data = nullptr;

  RingAllreduceSystemAtomicU32* self_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* self_ag_ready = nullptr;

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;

  RingAllreduceDeviceAtomicU32* self_tiles_finished = nullptr;

  RingAllreduceSystemAtomicU32* self_barrier_gather_token = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_gather_status = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_release_token = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_release_status = nullptr;
};

struct RingAllreduceValidatorAtomics {
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

__global__ void construct_validator_atomics_kernel(RingAllreduceValidatorAtomics a) {
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

__global__ void destroy_validator_atomics_kernel(RingAllreduceValidatorAtomics a) {
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

template <typename T>
__global__ void ring_allreduce_validator_direct_kernel(
    RingAllreduceParams<T, 8> p,
    uint32_t* out_valid) {

  if (blockIdx.x != 0 || threadIdx.x != 0 || !out_valid) {
    return;
  }

  out_valid[0] = cutlass::distributed::collective::detail::ring_allreduce_params_valid_ngpu(p) ? 1u : 0u;
}

static RingAllreduceParams<uint8_t, 8> make_valid_params(
    int32_t world_size,
    int32_t rank,
    uint32_t epoch,
    RingAllreduceValidatorPtrs const& ptrs,
    bool set_peer_rs_ready) {

  RingAllreduceParams<uint8_t, 8> p{};

  p.world_size = world_size;
  p.rank = rank;
  p.epoch = epoch;

  p.count = 1;
  p.num_channels = 1;

  p.tile_elems = 1;
  p.num_chunks_total = static_cast<uint32_t>(world_size) * static_cast<uint32_t>(p.num_channels);
  p.max_chunk_elems = 1;
  p.tiles_per_chunk = 1;
  p.num_tiles_total = 1;

  p.timeout_iters = 0;
  p.timeout_cycles = 0;
  p.poll_sleep_start = 0;
  p.poll_sleep_ns = 0;

  p.self_data = ptrs.self_data;
  p.self_rs_ready = ptrs.self_rs_ready;
  p.self_ag_ready = ptrs.self_ag_ready;
  p.self_abort = ptrs.self_abort;
  p.self_error = ptrs.self_error;
  p.self_tiles_finished = ptrs.self_tiles_finished;
  p.self_barrier_gather_token = ptrs.self_barrier_gather_token;
  p.self_barrier_gather_status = ptrs.self_barrier_gather_status;
  p.self_barrier_release_token = ptrs.self_barrier_release_token;
  p.self_barrier_release_status = ptrs.self_barrier_release_status;

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

  if (world_size > 1 && world_size > 0) {
    // Spec orientation: left = (rank - 1 + N) % N.
    int32_t left = (rank + world_size - 1) % world_size;

    p.peer_data[left] = ptrs.self_data;
    p.peer_ag_ready[left] = ptrs.self_ag_ready;
    p.peer_abort[left] = ptrs.self_abort;

    p.peer_barrier_gather_token[left] = ptrs.self_barrier_gather_token;
    p.peer_barrier_gather_status[left] = ptrs.self_barrier_gather_status;
    p.peer_barrier_release_token[left] = ptrs.self_barrier_release_token;
    p.peer_barrier_release_status[left] = ptrs.self_barrier_release_status;

    if (set_peer_rs_ready) {
      p.peer_rs_ready[left] = ptrs.self_rs_ready;
    }
  }

  return p;
}

} // namespace

CUTLASS_TEST_L2(RingAllreduceValidatorDirect, S0_ParamsValidNgpu, {

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

    cudaError_t probe_st = ring_allreduce_validator_probe_launch(dev);
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

  // Allocate a minimal set of dummy storage that can be shared across cases.
  constexpr uint32_t kMaxWorldSize = 8;
  constexpr uint32_t kNumTilesTotal = 1;
  constexpr uint32_t kFlagsLen = kMaxWorldSize * kNumTilesTotal;

  uint8_t* data = nullptr;
  RingAllreduceSystemAtomicU32* rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* ag_ready = nullptr;

  RingAllreduceSystemAtomicU32* abort = nullptr;
  RingAllreduceSystemAtomicU32* error = nullptr;

  RingAllreduceDeviceAtomicU32* tiles_finished = nullptr;

  RingAllreduceSystemAtomicU32* gather_token = nullptr;
  RingAllreduceSystemAtomicU32* gather_status = nullptr;
  RingAllreduceSystemAtomicU32* release_token = nullptr;
  RingAllreduceSystemAtomicU32* release_status = nullptr;

  uint32_t* device_valid = nullptr;

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&data), sizeof(uint8_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&rs_ready), sizeof(RingAllreduceSystemAtomicU32) * kFlagsLen), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&ag_ready), sizeof(RingAllreduceSystemAtomicU32) * kFlagsLen), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&tiles_finished), sizeof(RingAllreduceDeviceAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&gather_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&gather_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&release_token), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&release_status), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_valid), sizeof(uint32_t)), cudaSuccess);

  RingAllreduceValidatorAtomics atomics{};
  atomics.flags_len = kFlagsLen;
  atomics.self_rs_ready = rs_ready;
  atomics.self_ag_ready = ag_ready;
  atomics.self_abort = abort;
  atomics.self_error = error;
  atomics.self_tiles_finished = tiles_finished;
  atomics.self_barrier_gather_token = gather_token;
  atomics.self_barrier_gather_status = gather_status;
  atomics.self_barrier_release_token = release_token;
  atomics.self_barrier_release_status = release_status;

  construct_validator_atomics_kernel<<<(kFlagsLen + 255) / 256, 256>>>(atomics);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  RingAllreduceValidatorPtrs ptrs{};
  ptrs.self_data = data;
  ptrs.self_rs_ready = rs_ready;
  ptrs.self_ag_ready = ag_ready;
  ptrs.self_abort = abort;
  ptrs.self_error = error;
  ptrs.self_tiles_finished = tiles_finished;
  ptrs.self_barrier_gather_token = gather_token;
  ptrs.self_barrier_gather_status = gather_status;
  ptrs.self_barrier_release_token = release_token;
  ptrs.self_barrier_release_status = release_status;

  auto run_case = [&](char const* name, RingAllreduceParams<uint8_t, 8> const& p, bool expected) {
    ASSERT_EQ(cudaMemset(device_valid, 0xFF, sizeof(uint32_t)), cudaSuccess);

    ring_allreduce_validator_direct_kernel<uint8_t><<<1, 1>>>(p, device_valid);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << name;
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << name;

    uint32_t host_valid = 0u;
    ASSERT_EQ(cudaMemcpy(&host_valid, device_valid, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host_valid, expected ? 1u : 0u) << name;
  };

  run_case(
      "valid_n1_no_peers",
      make_valid_params(/*world_size=*/1, /*rank=*/0, /*epoch=*/1, ptrs, /*set_peer_rs_ready=*/false),
      /*expected=*/true);

  run_case(
      "invalid_world_size_0",
      make_valid_params(/*world_size=*/0, /*rank=*/0, /*epoch=*/1, ptrs, /*set_peer_rs_ready=*/true),
      /*expected=*/false);

  run_case(
      "invalid_world_size_3",
      make_valid_params(/*world_size=*/3, /*rank=*/0, /*epoch=*/1, ptrs, /*set_peer_rs_ready=*/true),
      /*expected=*/false);

  run_case(
      "invalid_world_size_5",
      make_valid_params(/*world_size=*/5, /*rank=*/0, /*epoch=*/1, ptrs, /*set_peer_rs_ready=*/true),
      /*expected=*/false);

  run_case(
      "invalid_rank_out_of_range",
      make_valid_params(/*world_size=*/2, /*rank=*/2, /*epoch=*/1, ptrs, /*set_peer_rs_ready=*/true),
      /*expected=*/false);

  run_case(
      "invalid_epoch_0",
      make_valid_params(/*world_size=*/1, /*rank=*/0, /*epoch=*/0, ptrs, /*set_peer_rs_ready=*/false),
      /*expected=*/false);

  {
    RingAllreduceParams<uint8_t, 8> p = make_valid_params(/*world_size=*/1, /*rank=*/0, /*epoch=*/1, ptrs, /*set_peer_rs_ready=*/false);
    p.self_data = nullptr;
    run_case("invalid_missing_self_pointer", p, /*expected=*/false);
  }

  {
    RingAllreduceParams<uint8_t, 8> p = make_valid_params(/*world_size=*/1, /*rank=*/0, /*epoch=*/1, ptrs, /*set_peer_rs_ready=*/false);
    p.self_error = nullptr;
    run_case("invalid_missing_self_error", p, /*expected=*/false);
  }

  run_case(
      "valid_n2_peer_rs_ready_null",
      make_valid_params(/*world_size=*/2, /*rank=*/0, /*epoch=*/1, ptrs, /*set_peer_rs_ready=*/false),
      /*expected=*/true);

  {
    RingAllreduceParams<uint8_t, 8> p = make_valid_params(/*world_size=*/2, /*rank=*/0, /*epoch=*/1, ptrs, /*set_peer_rs_ready=*/false);
    int32_t left = (p.rank + p.world_size - 1) % p.world_size;
    p.peer_abort[left] = nullptr;
    run_case("invalid_missing_peer_abort", p, /*expected=*/false);
  }

  run_case(
      "valid_n4_peer_rs_ready_present",
      make_valid_params(/*world_size=*/4, /*rank=*/0, /*epoch=*/1, ptrs, /*set_peer_rs_ready=*/true),
      /*expected=*/true);

  run_case(
      "invalid_n4_peer_rs_ready_null",
      make_valid_params(/*world_size=*/4, /*rank=*/0, /*epoch=*/1, ptrs, /*set_peer_rs_ready=*/false),
      /*expected=*/false);

  run_case(
      "valid_n8_rank1_smoke",
      make_valid_params(/*world_size=*/8, /*rank=*/1, /*epoch=*/1, ptrs, /*set_peer_rs_ready=*/true),
      /*expected=*/true);

  destroy_validator_atomics_kernel<<<(kFlagsLen + 255) / 256, 256>>>(atomics);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  ASSERT_EQ(cudaFree(device_valid), cudaSuccess);

  ASSERT_EQ(cudaFree(release_status), cudaSuccess);
  ASSERT_EQ(cudaFree(release_token), cudaSuccess);
  ASSERT_EQ(cudaFree(gather_status), cudaSuccess);
  ASSERT_EQ(cudaFree(gather_token), cudaSuccess);

  ASSERT_EQ(cudaFree(tiles_finished), cudaSuccess);

  ASSERT_EQ(cudaFree(error), cudaSuccess);
  ASSERT_EQ(cudaFree(abort), cudaSuccess);

  ASSERT_EQ(cudaFree(ag_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(rs_ready), cudaSuccess);
  ASSERT_EQ(cudaFree(data), cudaSuccess);
});
