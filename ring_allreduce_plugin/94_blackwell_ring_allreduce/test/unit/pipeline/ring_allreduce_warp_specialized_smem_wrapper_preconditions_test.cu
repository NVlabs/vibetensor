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
    \brief Unit tests for warp-specialized SMEM wrapper runtime preconditions.
*/

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

__global__ void ring_allreduce_warp_specialized_smem_wrapper_preconditions_probe_kernel() {}

static cudaError_t ring_allreduce_warp_specialized_smem_wrapper_preconditions_probe_launch() {
  // Clear any pre-existing per-thread CUDA error state.
  (void)cudaGetLastError();

  ring_allreduce_warp_specialized_smem_wrapper_preconditions_probe_kernel<<<1, 1>>>();
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

__global__ void ring_allreduce_warp_specialized_smem_wrapper_preconditions_kernel(
    RingAllreduceParams<float, 8> p,
    uint32_t N,
    uint32_t* out_rs_ok) {
#if defined(__CUDA_ARCH__)
  RingAllreduceDrainConfig cfg;
  cfg.timeout_iters = 0;
  cfg.timeout_cycles = 0;
  cfg.poll_sleep_start = 0;
  cfg.poll_sleep_ns = 0;

  bool rs_ok = cutlass::distributed::collective::detail::ring_allreduce_sm100_tile_warp_specialized_smem_rs(
      p,
      cfg,
      /*tile_linear=*/0,
      /*channel_id=*/0,
      /*tile_in_chunk=*/0,
      N,
      /*r_u32=*/0,
      /*left=*/0,
      /*flags_len=*/0);

  if (out_rs_ok && threadIdx.x == 0) {
    out_rs_ok[0] = rs_ok ? 1u : 0u;
  }
#else
  if (out_rs_ok && threadIdx.x == 0) {
    out_rs_ok[0] = 0u;
  }
#endif
}

} // namespace

TEST(RingAllreduceWarpSpecializedSmemWrapperPreconditions, RS_BAD_BLOCK_SHAPE) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_warp_specialized_smem_wrapper_preconditions_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;
  uint32_t* out_rs_ok = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&out_rs_ok), sizeof(uint32_t)), cudaSuccess);

  new (self_abort) RingAllreduceSystemAtomicU32{};
  new (self_error) RingAllreduceSystemAtomicU32{};

  self_abort->store(0u, cuda::memory_order_relaxed);
  self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  out_rs_ok[0] = 0xFFFF'FFFFu;

  RingAllreduceParams<float, 8> p{};
  p.self_abort = self_abort;
  p.self_error = self_error;

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&done_event), cudaSuccess);

  // Wrong block shape: 128 threads.
  ring_allreduce_warp_specialized_smem_wrapper_preconditions_kernel<<<1, 128, 0, stream>>>(p, /*N=*/2, out_rs_ok);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    std::fprintf(stderr, "ring_allreduce_warp_specialized_smem_wrapper_preconditions_test: watchdog timeout (RS_BAD_BLOCK_SHAPE)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  EXPECT_EQ(out_rs_ok[0], 0u);
  EXPECT_EQ(self_abort->load(cuda::memory_order_relaxed), 1u);
  EXPECT_EQ(self_error->load(cuda::memory_order_relaxed), static_cast<uint32_t>(RingAllreduceError::kInvalidParams));

  self_error->~RingAllreduceSystemAtomicU32();
  self_abort->~RingAllreduceSystemAtomicU32();

  ASSERT_EQ(cudaFree(out_rs_ok), cudaSuccess);
  ASSERT_EQ(cudaFree(self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self_abort), cudaSuccess);

  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(RingAllreduceWarpSpecializedSmemWrapperPreconditions, RS_N1_INVALID) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_warp_specialized_smem_wrapper_preconditions_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;
  uint32_t* out_rs_ok = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&out_rs_ok), sizeof(uint32_t)), cudaSuccess);

  new (self_abort) RingAllreduceSystemAtomicU32{};
  new (self_error) RingAllreduceSystemAtomicU32{};

  self_abort->store(0u, cuda::memory_order_relaxed);
  self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  out_rs_ok[0] = 0xFFFF'FFFFu;

  RingAllreduceParams<float, 8> p{};
  p.self_abort = self_abort;
  p.self_error = self_error;

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&done_event), cudaSuccess);

  // Invalid world size for RS: N==1.
  ring_allreduce_warp_specialized_smem_wrapper_preconditions_kernel<<<1, 256, 0, stream>>>(p, /*N=*/1, out_rs_ok);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    std::fprintf(stderr, "ring_allreduce_warp_specialized_smem_wrapper_preconditions_test: watchdog timeout (RS_N1_INVALID)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  EXPECT_EQ(out_rs_ok[0], 0u);
  EXPECT_EQ(self_abort->load(cuda::memory_order_relaxed), 1u);
  EXPECT_EQ(self_error->load(cuda::memory_order_relaxed), static_cast<uint32_t>(RingAllreduceError::kInvalidParams));

  self_error->~RingAllreduceSystemAtomicU32();
  self_abort->~RingAllreduceSystemAtomicU32();

  ASSERT_EQ(cudaFree(out_rs_ok), cudaSuccess);
  ASSERT_EQ(cudaFree(self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self_abort), cudaSuccess);

  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
