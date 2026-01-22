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
    \brief Unit test for ring_allreduce_wait_flag abort propagation semantics.
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/experimental/distributed/collective/ring_allreduce_kernel_sm100.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <new>

namespace {

using cutlass::distributed::collective::RingAllreduceDrainConfig;
using cutlass::distributed::collective::RingAllreduceError;
using cutlass::distributed::collective::RingAllreduceSystemAtomicU32;
using cutlass::distributed::collective::detail::ring_allreduce_wait_flag;

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

__global__ void ring_allreduce_wait_flag_probe_kernel() {}

static cudaError_t ring_allreduce_wait_flag_probe_launch() {
  // Clear any pre-existing per-thread CUDA error state.
  (void)cudaGetLastError();

  ring_allreduce_wait_flag_probe_kernel<<<1, 1>>>();
  return cudaGetLastError();
}

__global__ void ring_allreduce_wait_flag_abort_kernel(
    RingAllreduceSystemAtomicU32* flag,
    uint32_t epoch,
    RingAllreduceSystemAtomicU32* self_abort,
    RingAllreduceSystemAtomicU32* self_error,
    RingAllreduceSystemAtomicU32* peer_abort,
    uint32_t* out_wait_ok) {

  // Block1: asynchronously trigger a peer abort.
  if (blockIdx.x == 1) {
    if (threadIdx.x == 0) {
      // Small delay to reduce the chance the store happens before block0 begins polling.
      for (int i = 0; i < 1024; ++i) {
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
          __nanosleep(50);
        #endif
      }

      peer_abort->store(1u, cuda::memory_order_release);
    }
    return;
  }

  RingAllreduceDrainConfig cfg{};
  cfg.timeout_iters = 1u << 20; // safety net: should not trigger on abort propagation
  cfg.timeout_cycles = 0;
  cfg.poll_sleep_start = 0;
  cfg.poll_sleep_ns = 0;

  bool ok = ring_allreduce_wait_flag(
      flag,
      epoch,
      self_abort,
      self_error,
      peer_abort,
      cfg);

  if (threadIdx.x == 0 && out_wait_ok) {
    out_wait_ok[0] = ok ? 1u : 0u;
  }
}

} // namespace

TEST(RingAllreduceWaitFlag, M5_AbortPropagationInWaitHelper) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_wait_flag_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  RingAllreduceSystemAtomicU32* flag = nullptr;
  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;
  RingAllreduceSystemAtomicU32* peer_abort = nullptr;

  uint32_t* out_wait_ok = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&flag), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&peer_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&out_wait_ok), sizeof(uint32_t)), cudaSuccess);

  // Begin lifetime for atomic objects (avoid C++ object-lifetime UB).
  new (flag) RingAllreduceSystemAtomicU32{};
  new (self_abort) RingAllreduceSystemAtomicU32{};
  new (self_error) RingAllreduceSystemAtomicU32{};
  new (peer_abort) RingAllreduceSystemAtomicU32{};

  flag->store(0u, cuda::memory_order_relaxed);
  self_abort->store(0u, cuda::memory_order_relaxed);
  self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  peer_abort->store(0u, cuda::memory_order_relaxed);

  out_wait_ok[0] = 0xFFFF'FFFFu;

  ring_allreduce_wait_flag_abort_kernel<<<2, 256>>>(
      flag,
      /*epoch=*/1u,
      self_abort,
      self_error,
      peer_abort,
      out_wait_ok);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t abort = self_abort->load(cuda::memory_order_relaxed);
  uint32_t err = self_error->load(cuda::memory_order_relaxed);

  EXPECT_EQ(out_wait_ok[0], 0u);
  EXPECT_EQ(abort, 1u);
  EXPECT_EQ(err, static_cast<uint32_t>(RingAllreduceError::kOk));

  ASSERT_EQ(cudaFree(out_wait_ok), cudaSuccess);

  peer_abort->~RingAllreduceSystemAtomicU32();
  self_error->~RingAllreduceSystemAtomicU32();
  self_abort->~RingAllreduceSystemAtomicU32();
  flag->~RingAllreduceSystemAtomicU32();

  ASSERT_EQ(cudaFree(peer_abort), cudaSuccess);
  ASSERT_EQ(cudaFree(self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self_abort), cudaSuccess);
  ASSERT_EQ(cudaFree(flag), cudaSuccess);
}
