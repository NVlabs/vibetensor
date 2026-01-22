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
    \brief Smoke tests for NamedBarrier choreography used by the warp-specialized SMEM path.
*/

#include "../common/cutlass_unit_test.h"

#include "cutlass/experimental/distributed/collective/ring_allreduce_kernel_sm100.cuh"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

using cutlass::arch::NamedBarrier;

struct TestNamedBarrierConstants {
  static constexpr uint32_t kStageThreads = 256;
  static constexpr uint32_t kPublishThreads = 160;
  static constexpr uint32_t kStepBarrierThreads = 256;

  static constexpr uint32_t kStageId0 = 0;
  static constexpr uint32_t kStageId1 = 1;
  static constexpr uint32_t kPublishId0 = 2;
  static constexpr uint32_t kPublishId1 = 3;
  static constexpr uint32_t kStepBarrierId = 4;
};

static_assert(TestNamedBarrierConstants::kStageThreads % 32 == 0,
              "Stage barrier threads must be warp-aligned.");
static_assert(TestNamedBarrierConstants::kPublishThreads % 32 == 0,
              "Publish barrier threads must be warp-aligned.");
static_assert(TestNamedBarrierConstants::kStepBarrierThreads % 32 == 0,
              "Step barrier threads must be warp-aligned.");

static_assert(
    TestNamedBarrierConstants::kStepBarrierId + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount <
        cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers,
    "NamedBarrier IDs must fit within hardware limits.");

#if CUTLASS_RING_ALLREDUCE_ENABLE_WARP_SPECIALIZED_SMEM
using BarrierConsts = cutlass::distributed::collective::detail::RingAllreduceWarpSpecializedSmemNamedBarrierConstants;
static_assert(BarrierConsts::kStageThreads == TestNamedBarrierConstants::kStageThreads);
static_assert(BarrierConsts::kPublishThreads == TestNamedBarrierConstants::kPublishThreads);
static_assert(BarrierConsts::kStepBarrierThreads == TestNamedBarrierConstants::kStepBarrierThreads);

static_assert(BarrierConsts::kStageId0 == TestNamedBarrierConstants::kStageId0);
static_assert(BarrierConsts::kStageId1 == TestNamedBarrierConstants::kStageId1);
static_assert(BarrierConsts::kPublishId0 == TestNamedBarrierConstants::kPublishId0);
static_assert(BarrierConsts::kPublishId1 == TestNamedBarrierConstants::kPublishId1);
static_assert(BarrierConsts::kStepBarrierId == TestNamedBarrierConstants::kStepBarrierId);
#else
using BarrierConsts = TestNamedBarrierConstants;
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

__global__ void ring_allreduce_named_barrier_smoke_probe_kernel() {}

static cudaError_t ring_allreduce_named_barrier_smoke_probe_launch() {
  // Clear any pre-existing per-thread CUDA error state.
  (void)cudaGetLastError();

  ring_allreduce_named_barrier_smoke_probe_kernel<<<1, 1>>>();
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

__global__ void ring_allreduce_named_barrier_smoke_nb0_kernel(uint32_t* out_status) {
#if defined(__CUDA_ARCH__) && CUDA_BARRIER_ENABLED
  NamedBarrier::arrive_and_wait(BarrierConsts::kStageThreads, BarrierConsts::kStageId0);
  NamedBarrier::arrive_and_wait(BarrierConsts::kStageThreads, BarrierConsts::kStageId1);
  NamedBarrier::arrive_and_wait(BarrierConsts::kStepBarrierThreads, BarrierConsts::kStepBarrierId);

  if (out_status && threadIdx.x == 0) {
    out_status[0] = 1u;
  }
#else
  if (out_status && threadIdx.x == 0) {
    out_status[0] = 0u;
  }
#endif
}

__global__ void ring_allreduce_named_barrier_smoke_nb1_kernel(uint32_t* out_sums) {
#if defined(__CUDA_ARCH__) && CUDA_BARRIER_ENABLED
  __shared__ int data[128];

  uint32_t warp_id = threadIdx.x >> 5;
  uint32_t lane = threadIdx.x & 0x1Fu;

  if (threadIdx.x < 128) {
    data[threadIdx.x] = 0;
  }

  NamedBarrier::arrive_and_wait(BarrierConsts::kStageThreads, BarrierConsts::kStageId0);

  // Round 0 (PublishId0): data[i] = i.
  if (warp_id >= 2 && warp_id <= 5) {
    uint32_t tid = (warp_id - 2) * 32 + lane;
    data[tid] = static_cast<int>(tid);
  }

  // Split-phase PublishBarrier: compute warps arrive, store warp arrives+waits.
  if (warp_id >= 2 && warp_id <= 5) {
    NamedBarrier::arrive(BarrierConsts::kPublishThreads, BarrierConsts::kPublishId0);
  }
  else if (warp_id == 6) {
    NamedBarrier::arrive_and_wait(BarrierConsts::kPublishThreads, BarrierConsts::kPublishId0);
  }

  if (warp_id == 6 && lane == 0 && out_sums) {
    int sum = 0;
    for (int i = 0; i < 128; ++i) {
      sum += data[i];
    }
    out_sums[0] = static_cast<uint32_t>(sum);
  }

  // Throttle + allow safe reuse of `data` for the second publish barrier ID.
  NamedBarrier::arrive_and_wait(BarrierConsts::kStageThreads, BarrierConsts::kStageId1);

  // Round 1 (PublishId1): data[i] = i + 1.
  if (warp_id >= 2 && warp_id <= 5) {
    uint32_t tid = (warp_id - 2) * 32 + lane;
    data[tid] = static_cast<int>(tid + 1u);
  }

  if (warp_id >= 2 && warp_id <= 5) {
    NamedBarrier::arrive(BarrierConsts::kPublishThreads, BarrierConsts::kPublishId1);
  }
  else if (warp_id == 6) {
    NamedBarrier::arrive_and_wait(BarrierConsts::kPublishThreads, BarrierConsts::kPublishId1);
  }

  if (warp_id == 6 && lane == 0 && out_sums) {
    int sum = 0;
    for (int i = 0; i < 128; ++i) {
      sum += data[i];
    }
    out_sums[1] = static_cast<uint32_t>(sum);
  }

  NamedBarrier::arrive_and_wait(BarrierConsts::kStepBarrierThreads, BarrierConsts::kStepBarrierId);
#else
  if (out_sums && threadIdx.x == 0) {
    out_sums[0] = 0u;
    out_sums[1] = 0u;
  }
#endif
}

__global__ void ring_allreduce_named_barrier_smoke_nb2_kernel(uint32_t* out_status) {
#if defined(__CUDA_ARCH__) && CUDA_BARRIER_ENABLED
  __shared__ uint32_t in_valid;

  uint32_t warp_id = threadIdx.x >> 5;

  if (threadIdx.x == 0) {
    in_valid = 0u;
  }

  NamedBarrier::arrive_and_wait(BarrierConsts::kStageThreads, BarrierConsts::kStageId0);

  // Uniform early exit: everyone must skip PublishBarrier.
  if (in_valid != 0u) {
    if (warp_id >= 2 && warp_id <= 6) {
      NamedBarrier::arrive_and_wait(BarrierConsts::kPublishThreads, BarrierConsts::kPublishId0);
    }
  }

  NamedBarrier::arrive_and_wait(BarrierConsts::kStepBarrierThreads, BarrierConsts::kStepBarrierId);

  if (out_status && threadIdx.x == 0) {
    out_status[0] = 1u;
  }
#else
  if (out_status && threadIdx.x == 0) {
    out_status[0] = 0u;
  }
#endif
}

__global__ void ring_allreduce_named_barrier_smoke_nb3_kernel(uint32_t* out_status) {
#if defined(__CUDA_ARCH__) && CUDA_BARRIER_ENABLED
  __shared__ uint32_t rs_ok;
  __shared__ uint32_t stage_out[128];

  uint32_t warp_id = threadIdx.x >> 5;
  uint32_t lane = threadIdx.x & 0x1Fu;

  if (threadIdx.x == 0) {
    rs_ok = 0u;
  }

  if (threadIdx.x < 128) {
    stage_out[threadIdx.x] = 0u;
  }

  NamedBarrier::arrive_and_wait(BarrierConsts::kStageThreads, BarrierConsts::kStageId0);

  if (warp_id >= 2 && warp_id <= 5) {
    uint32_t tid = (warp_id - 2) * 32 + lane;
    stage_out[tid] = tid;
  }

  if (warp_id >= 2 && warp_id <= 6) {
    NamedBarrier::arrive_and_wait(BarrierConsts::kPublishThreads, BarrierConsts::kPublishId0);
  }

  // Controlled delay in warp6 after PublishBarrier, then publish rs_ok.
  if (warp_id == 6 && lane == 0) {
    uint32_t sum = 0u;
    for (int i = 0; i < 128; ++i) {
      sum += stage_out[i];
    }

    uint64_t start = clock64();
    // Small delay (order of ~us) to stress RS->AG uniform-gating barriers.
    while (clock64() - start < 50'000u) {
    }

    rs_ok = (sum == 8128u) ? 1u : 0u;
  }

  NamedBarrier::arrive_and_wait(BarrierConsts::kStepBarrierThreads, BarrierConsts::kStepBarrierId);

  bool ok = (rs_ok != 0u);

  // Emulate legacy AG entry barrier; this must be CTA-uniform.
  if (ok) {
    __syncthreads();
  }

  if (out_status && threadIdx.x == 0) {
    out_status[0] = ok ? 1u : 0u;
  }
#else
  if (out_status && threadIdx.x == 0) {
    out_status[0] = 0u;
  }
#endif
}

__global__ void ring_allreduce_named_barrier_smoke_nb4_kernel(uint32_t* out_sums) {
#if defined(__CUDA_ARCH__) && CUDA_BARRIER_ENABLED
  __shared__ int data[128];

  uint32_t warp_id = threadIdx.x >> 5;
  uint32_t lane = threadIdx.x & 0x1Fu;

  for (uint32_t iter = 0; iter < 3; ++iter) {
    uint32_t buf = (iter + 1u) & 1u;
    uint32_t stage_id = buf ? BarrierConsts::kStageId1 : BarrierConsts::kStageId0;
    uint32_t publish_id = buf ? BarrierConsts::kPublishId1 : BarrierConsts::kPublishId0;

    NamedBarrier::arrive_and_wait(BarrierConsts::kStageThreads, stage_id);

    // Round `iter`: data[i] = i + (iter + 1).
    if (warp_id >= 2 && warp_id <= 5) {
      uint32_t tid = (warp_id - 2) * 32 + lane;
      data[tid] = static_cast<int>(tid + iter + 1u);
    }

    // Full PublishBarrier with warps 2-6. Alternate PublishId1 -> PublishId0 -> PublishId1.
    if (warp_id >= 2 && warp_id <= 6) {
      NamedBarrier::arrive_and_wait(BarrierConsts::kPublishThreads, publish_id);
    }

    if (warp_id == 6 && lane == 0 && out_sums) {
      int sum = 0;
      for (int i = 0; i < 128; ++i) {
        sum += data[i];
      }
      out_sums[iter] = static_cast<uint32_t>(sum);
    }

    NamedBarrier::arrive_and_wait(BarrierConsts::kStepBarrierThreads, BarrierConsts::kStepBarrierId);
  }
#else
  if (out_sums && threadIdx.x == 0) {
    out_sums[0] = 0u;
    out_sums[1] = 0u;
    out_sums[2] = 0u;
  }
#endif
}

} // namespace

TEST(RingAllreduceNamedBarrierSmoke, NB0_BasicStageStepBarrier) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_named_barrier_smoke_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;

  uint32_t* device_status = nullptr;
  uint32_t host_status = 0u;

  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status), sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(device_status, 0, sizeof(uint32_t), stream), cudaSuccess);

  ring_allreduce_named_barrier_smoke_nb0_kernel<<<1, BarrierConsts::kStageThreads, 0, stream>>>(device_status);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    std::fprintf(stderr, "ring_allreduce_named_barrier_smoke_test: watchdog timeout (NB0)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&host_status, device_status, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_EQ(host_status, 1u);

  ASSERT_EQ(cudaFree(device_status), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(RingAllreduceNamedBarrierSmoke, NB1_SplitArrivalPublishBarrier) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_named_barrier_smoke_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;

  uint32_t* device_sums = nullptr;
  uint32_t host_sums[2] = {0u, 0u};

  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_sums), 2 * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(device_sums, 0, 2 * sizeof(uint32_t), stream), cudaSuccess);

  ring_allreduce_named_barrier_smoke_nb1_kernel<<<1, BarrierConsts::kStageThreads, 0, stream>>>(device_sums);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    std::fprintf(stderr, "ring_allreduce_named_barrier_smoke_test: watchdog timeout (NB1)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(host_sums, device_sums, sizeof(host_sums), cudaMemcpyDeviceToHost), cudaSuccess);

  // Sum_{i=0..127} i = 8128.
  EXPECT_EQ(host_sums[0], 8128u);
  // Sum_{i=0..127} (i + 1) = 8256.
  EXPECT_EQ(host_sums[1], 8256u);

  ASSERT_EQ(cudaFree(device_sums), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(RingAllreduceNamedBarrierSmoke, NB2_UniformEarlyExitSkipsPublishBarrier) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_named_barrier_smoke_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;

  uint32_t* device_status = nullptr;
  uint32_t host_status = 0u;

  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status), sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(device_status, 0, sizeof(uint32_t), stream), cudaSuccess);

  ring_allreduce_named_barrier_smoke_nb2_kernel<<<1, BarrierConsts::kStageThreads, 0, stream>>>(device_status);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    std::fprintf(stderr, "ring_allreduce_named_barrier_smoke_test: watchdog timeout (NB2)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&host_status, device_status, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_EQ(host_status, 1u);

  ASSERT_EQ(cudaFree(device_status), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(RingAllreduceNamedBarrierSmoke, NB3_RsAgUniformGatingStress) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_named_barrier_smoke_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;

  uint32_t* device_status = nullptr;
  uint32_t host_status = 0u;

  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_status), sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(device_status, 0, sizeof(uint32_t), stream), cudaSuccess);

  ring_allreduce_named_barrier_smoke_nb3_kernel<<<1, BarrierConsts::kStageThreads, 0, stream>>>(device_status);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  // Host watchdog: ensure no hangs even while RS/AG choreography is under development.
  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    std::fprintf(stderr, "ring_allreduce_named_barrier_smoke_test: watchdog timeout (NB3)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&host_status, device_status, sizeof(uint32_t), cudaMemcpyDeviceToHost), cudaSuccess);

  EXPECT_EQ(host_status, 1u);

  ASSERT_EQ(cudaFree(device_status), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(RingAllreduceNamedBarrierSmoke, NB4_PublishId1Alternation) {
  if (!is_sm100_or_sm103()) {
    GTEST_SKIP() << "requires SM100/SM103";
  }

  cudaError_t probe_st = ring_allreduce_named_barrier_smoke_probe_launch();
  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    GTEST_SKIP() << "requires a valid kernel image";
  }
  ASSERT_EQ(probe_st, cudaSuccess);

  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;

  uint32_t* device_sums = nullptr;
  uint32_t host_sums[3] = {0u, 0u, 0u};

  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  ASSERT_EQ(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming), cudaSuccess);

  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_sums), sizeof(host_sums)), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(device_sums, 0, sizeof(host_sums), stream), cudaSuccess);

  ring_allreduce_named_barrier_smoke_nb4_kernel<<<1, BarrierConsts::kStageThreads, 0, stream>>>(device_sums);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(done_event, stream), cudaSuccess);

  constexpr auto kHostWatchdogTimeout = std::chrono::seconds(10);
  auto deadline = std::chrono::steady_clock::now() + kHostWatchdogTimeout;

  cudaError_t q = wait_or_timeout(done_event, deadline);
  if (q == cudaErrorTimeout) {
    std::fprintf(stderr, "ring_allreduce_named_barrier_smoke_test: watchdog timeout (NB4)\n");
    std::abort();
  }
  ASSERT_EQ(q, cudaSuccess);

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(host_sums, device_sums, sizeof(host_sums), cudaMemcpyDeviceToHost), cudaSuccess);

  // Sum_{i=0..127} (i + (iter + 1)).
  EXPECT_EQ(host_sums[0], 8256u);
  EXPECT_EQ(host_sums[1], 8384u);
  EXPECT_EQ(host_sums[2], 8512u);

  ASSERT_EQ(cudaFree(device_sums), cudaSuccess);
  ASSERT_EQ(cudaEventDestroy(done_event), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
