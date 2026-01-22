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
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Unit test for ring_allreduce drain counter atomic contract helpers.
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/experimental/distributed/collective/ring_allreduce_drain.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <new>
#include <vector>

namespace {

using cutlass::distributed::collective::RingAllreduceDeviceAtomicU32;
using cutlass::distributed::collective::RingAllreduceDrainConfig;
using cutlass::distributed::collective::RingAllreduceError;
using cutlass::distributed::collective::RingAllreduceSystemAtomicU32;
using cutlass::distributed::collective::ring_allreduce_drain_tiles_finished;
using cutlass::distributed::collective::ring_allreduce_is_cta0;
using cutlass::distributed::collective::ring_allreduce_is_thread0;
using cutlass::distributed::collective::ring_allreduce_signal_tile_finished;

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

__global__ void ring_allreduce_drain_probe_kernel() {}

static cudaError_t ring_allreduce_drain_probe_launch() {
  // Clear any pre-existing per-thread CUDA error state.
  (void)cudaGetLastError();

  ring_allreduce_drain_probe_kernel<<<1, 1>>>();
  return cudaGetLastError();
}

__global__ void ring_allreduce_drain_d0_kernel(
    RingAllreduceDeviceAtomicU32* self_tiles_finished,
    uint32_t expected_tiles,
    RingAllreduceSystemAtomicU32* self_abort,
    RingAllreduceSystemAtomicU32* self_error,
    uint32_t* cta_old,
    uint32_t* out_status) {

  if (!ring_allreduce_is_thread0()) {
    return;
  }

  uint32_t old = ring_allreduce_signal_tile_finished(self_tiles_finished);
  if (cta_old) {
    cta_old[blockIdx.x] = old;
  }

  if (ring_allreduce_is_cta0()) {
    RingAllreduceDrainConfig cfg;
    cfg.timeout_iters = 1u << 20; // safety net: should not trigger on success
    cfg.timeout_cycles = 0;
    cfg.poll_sleep_start = 0;
    cfg.poll_sleep_ns = 0;

    RingAllreduceError st = ring_allreduce_drain_tiles_finished(
        self_tiles_finished,
        expected_tiles,
        self_abort,
        self_error,
        cfg);

    if (out_status) {
      out_status[0] = static_cast<uint32_t>(st);
    }
  }
}

__global__ void ring_allreduce_drain_d1_kernel(
    RingAllreduceDeviceAtomicU32* self_tiles_finished,
    uint32_t expected_tiles,
    RingAllreduceSystemAtomicU32* self_abort,
    RingAllreduceSystemAtomicU32* self_error,
    int spin_block,
    uint32_t timeout_iters,
    uint32_t max_spin_iters,
    uint32_t* spinner_result,
    uint32_t* out_status) {

  if (!ring_allreduce_is_thread0()) {
    return;
  }

  if (static_cast<int>(blockIdx.x) == spin_block) {
    uint32_t it = 0;
    while (self_abort->load(cuda::memory_order_acquire) == 0u && it < max_spin_iters) {
      #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        __nanosleep(40);
      #endif
      ++it;
    }

    if (spinner_result) {
      spinner_result[0] = (it < max_spin_iters) ? 1u : 2u; // 1=abort observed, 2=watchdog
    }

    (void)ring_allreduce_signal_tile_finished(self_tiles_finished);
    return;
  }

  // Non-spinner CTAs signal immediately.
  (void)ring_allreduce_signal_tile_finished(self_tiles_finished);

  if (ring_allreduce_is_cta0()) {
    RingAllreduceDrainConfig cfg;
    cfg.timeout_iters = timeout_iters;
    cfg.timeout_cycles = 0;
    cfg.poll_sleep_start = 0;
    cfg.poll_sleep_ns = 0;

    RingAllreduceError st = ring_allreduce_drain_tiles_finished(
        self_tiles_finished,
        expected_tiles,
        self_abort,
        self_error,
        cfg);

    if (out_status) {
      out_status[0] = static_cast<uint32_t>(st);
    }
  }
}

} // namespace

TEST(RingAllreduceDrainAtomic, D0_DrainSuccess) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP();
  }

  cudaError_t probe_st = ring_allreduce_drain_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP();
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  constexpr int kBlocks = 64;
  constexpr int kThreads = 256;

  RingAllreduceDeviceAtomicU32* self_tiles_finished = nullptr;
  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;

  uint32_t* cta_old = nullptr;
  uint32_t* out_status = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_tiles_finished), sizeof(RingAllreduceDeviceAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&cta_old), sizeof(uint32_t) * kBlocks), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&out_status), sizeof(uint32_t)), cudaSuccess);

  // Begin lifetime for atomic objects (avoid C++ object-lifetime UB).
  new (self_tiles_finished) RingAllreduceDeviceAtomicU32{};
  new (self_abort) RingAllreduceSystemAtomicU32{};
  new (self_error) RingAllreduceSystemAtomicU32{};

  // Initialize atomics within the C++ atomic object model.
  self_tiles_finished->store(0u, cuda::memory_order_relaxed);
  self_abort->store(0u, cuda::memory_order_relaxed);
  self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);

  for (int i = 0; i < kBlocks; ++i) {
    cta_old[i] = 0u;
  }
  out_status[0] = 0u;

  ring_allreduce_drain_d0_kernel<<<kBlocks, kThreads>>>(
      self_tiles_finished,
      static_cast<uint32_t>(kBlocks),
      self_abort,
      self_error,
      cta_old,
      out_status);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t done = self_tiles_finished->load(cuda::memory_order_relaxed);
  uint32_t abort = self_abort->load(cuda::memory_order_relaxed);
  uint32_t err = self_error->load(cuda::memory_order_relaxed);

  EXPECT_EQ(done, static_cast<uint32_t>(kBlocks));
  EXPECT_EQ(abort, 0u);
  EXPECT_EQ(err, static_cast<uint32_t>(RingAllreduceError::kOk));
  EXPECT_EQ(out_status[0], static_cast<uint32_t>(RingAllreduceError::kOk));

  std::vector<uint32_t> vals(cta_old, cta_old + kBlocks);
  std::sort(vals.begin(), vals.end());
  for (int i = 0; i < kBlocks; ++i) {
    EXPECT_EQ(vals[i], static_cast<uint32_t>(i));
  }

  ASSERT_EQ(cudaFree(out_status), cudaSuccess);
  ASSERT_EQ(cudaFree(cta_old), cudaSuccess);

  self_error->~RingAllreduceSystemAtomicU32();
  self_abort->~RingAllreduceSystemAtomicU32();
  self_tiles_finished->~RingAllreduceDeviceAtomicU32();

  ASSERT_EQ(cudaFree(self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self_abort), cudaSuccess);
  ASSERT_EQ(cudaFree(self_tiles_finished), cudaSuccess);
}

TEST(RingAllreduceDrainAtomic, D1_TimeoutPublishesAbortAndTerminates) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP();
  }

  cudaError_t probe_st = ring_allreduce_drain_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP();
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  constexpr int kBlocks = 8;
  constexpr int kThreads = 256;
  constexpr int kSpinBlock = kBlocks - 1;

  RingAllreduceDeviceAtomicU32* self_tiles_finished = nullptr;
  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;

  uint32_t* spinner_result = nullptr;
  uint32_t* out_status = nullptr;

  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_tiles_finished), sizeof(RingAllreduceDeviceAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_abort), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&self_error), sizeof(RingAllreduceSystemAtomicU32)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&spinner_result), sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMallocManaged(reinterpret_cast<void**>(&out_status), sizeof(uint32_t)), cudaSuccess);

  // Begin lifetime for atomic objects (avoid C++ object-lifetime UB).
  new (self_tiles_finished) RingAllreduceDeviceAtomicU32{};
  new (self_abort) RingAllreduceSystemAtomicU32{};
  new (self_error) RingAllreduceSystemAtomicU32{};

  self_tiles_finished->store(0u, cuda::memory_order_relaxed);
  self_abort->store(0u, cuda::memory_order_relaxed);
  self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);

  spinner_result[0] = 0u;
  out_status[0] = 0u;

  ring_allreduce_drain_d1_kernel<<<kBlocks, kThreads>>>(
      self_tiles_finished,
      static_cast<uint32_t>(kBlocks),
      self_abort,
      self_error,
      /*spin_block=*/kSpinBlock,
      /*timeout_iters=*/1024u,
      /*max_spin_iters=*/(1u << 20),
      spinner_result,
      out_status);

  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  uint32_t done = self_tiles_finished->load(cuda::memory_order_relaxed);
  uint32_t abort = self_abort->load(cuda::memory_order_relaxed);
  uint32_t err = self_error->load(cuda::memory_order_relaxed);

  EXPECT_EQ(abort, 1u);
  EXPECT_EQ(err, static_cast<uint32_t>(RingAllreduceError::kTimeout));
  EXPECT_EQ(out_status[0], static_cast<uint32_t>(RingAllreduceError::kTimeout));
  EXPECT_EQ(spinner_result[0], 1u) << "spinner_result: 1=abort observed, 2=watchdog";
  EXPECT_EQ(done, static_cast<uint32_t>(kBlocks));

  ASSERT_EQ(cudaFree(out_status), cudaSuccess);
  ASSERT_EQ(cudaFree(spinner_result), cudaSuccess);

  self_error->~RingAllreduceSystemAtomicU32();
  self_abort->~RingAllreduceSystemAtomicU32();
  self_tiles_finished->~RingAllreduceDeviceAtomicU32();

  ASSERT_EQ(cudaFree(self_error), cudaSuccess);
  ASSERT_EQ(cudaFree(self_abort), cudaSuccess);
  ASSERT_EQ(cudaFree(self_tiles_finished), cudaSuccess);
}
