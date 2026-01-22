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

#include "../common/cutlass_unit_test.h"

#include "cutlass/experimental/distributed/collective/ring_allreduce_host.hpp"

#include <cuda_runtime_api.h>

namespace {

using cutlass::distributed::collective::RingAllreduceP2POptions;
using cutlass::distributed::collective::RingAllreduceTiling;
using cutlass::distributed::collective::validate_ring_allreduce_host_tiling;
using cutlass::distributed::collective::validate_ring_p2p_caps_and_enable_peer_access;

} // namespace

TEST(RingAllreduceHost, P2PRejectsInvalidWorldSize) {

  auto r = validate_ring_p2p_caps_and_enable_peer_access(/*world_size=*/0);
  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status, cutlass::Status::kErrorInvalidProblem);
}

TEST(RingAllreduceHost, P2PRejectsOutOfRangeDeviceId) {

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count <= 0) {
    GTEST_SKIP();
  }

  int invalid_dev = device_count;
  auto r = validate_ring_p2p_caps_and_enable_peer_access(/*world_size=*/1, &invalid_dev);
  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status, cutlass::Status::kErrorInvalidProblem);
}

TEST(RingAllreduceHost, P2PRejectsDuplicateDeviceIds) {

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < 2) {
    GTEST_SKIP();
  }

  int devices[2] = {0, 0};
  auto r = validate_ring_p2p_caps_and_enable_peer_access(/*world_size=*/2, devices);
  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status, cutlass::Status::kErrorInvalidProblem);
}

TEST(RingAllreduceHost, P2PCapsMatchRuntimeQueries) {

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < 2) {
    GTEST_SKIP();
  }

  int can01 = 0;
  int can10 = 0;
  EXPECT_EQ(cudaDeviceCanAccessPeer(&can01, 0, 1), cudaSuccess);
  EXPECT_EQ(cudaDeviceCanAccessPeer(&can10, 1, 0), cudaSuccess);

  int nat01 = 0;
  int nat10 = 0;
  EXPECT_EQ(cudaDeviceGetP2PAttribute(&nat01, cudaDevP2PAttrNativeAtomicSupported, 0, 1), cudaSuccess);
  EXPECT_EQ(cudaDeviceGetP2PAttribute(&nat10, cudaDevP2PAttrNativeAtomicSupported, 1, 0), cudaSuccess);

  bool expected_ok = (can01 != 0) && (can10 != 0) && (nat01 == 1) && (nat10 == 1);

  int devices[2] = {0, 1};
  RingAllreduceP2POptions opts;
  opts.enable_peer_access = true;
  opts.require_native_atomics = true;

  auto r = validate_ring_p2p_caps_and_enable_peer_access(/*world_size=*/2, devices, opts);

  if (expected_ok) {
    EXPECT_TRUE(r.ok()) << (r.error_reason ? r.error_reason : "<no error>")
                        << " (cuda_error=" << cudaGetErrorString(r.cuda_error) << ")";

    // Idempotency: enabling twice should not fail.
    auto r2 = validate_ring_p2p_caps_and_enable_peer_access(/*world_size=*/2, devices, opts);
    EXPECT_TRUE(r2.ok()) << (r2.error_reason ? r2.error_reason : "<no error>");
  }
  else {
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status, cutlass::Status::kErrorNotSupported);
  }
}

TEST(RingAllreduceHost, TilingRejectsNullOutTiling) {

  auto r = validate_ring_allreduce_host_tiling(
      /*count=*/1024,
      /*world_size=*/1,
      /*num_channels=*/1,
      /*tile_elems=*/256,
      /*out_tiling=*/nullptr);

  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status, cutlass::Status::kErrorInvalidProblem);
}

TEST(RingAllreduceHost, TilingRejectsOutOfRangeDeviceId) {

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count <= 0) {
    GTEST_SKIP();
  }

  RingAllreduceTiling tiling;
  int invalid_dev = device_count;
  auto r = validate_ring_allreduce_host_tiling(
      /*count=*/1024,
      /*world_size=*/1,
      /*num_channels=*/1,
      /*tile_elems=*/256,
      &tiling,
      &invalid_dev);

  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status, cutlass::Status::kErrorInvalidProblem);
}

TEST(RingAllreduceHost, TilingRejectsDuplicateDeviceIds) {

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < 2) {
    GTEST_SKIP();
  }

  RingAllreduceTiling tiling;
  int devices[2] = {0, 0};
  auto r = validate_ring_allreduce_host_tiling(
      /*count=*/1024,
      /*world_size=*/2,
      /*num_channels=*/1,
      /*tile_elems=*/256,
      &tiling,
      devices);

  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status, cutlass::Status::kErrorInvalidProblem);
}

TEST(RingAllreduceHost, TilingCountZeroIsNoOp) {

  RingAllreduceTiling tiling;
  auto r = validate_ring_allreduce_host_tiling(
      /*count=*/0,
      /*world_size=*/1,
      /*num_channels=*/1,
      /*tile_elems=*/256,
      &tiling);

  EXPECT_TRUE(r.ok()) << (r.error_reason ? r.error_reason : "<no error>");
  EXPECT_EQ(tiling.max_chunk_elems, 0u);
  EXPECT_EQ(tiling.tiles_per_chunk, 0u);
  EXPECT_EQ(tiling.num_tiles_total, 0u);
}

TEST(RingAllreduceHost, TilingCountZeroSkipsDeviceCountEnforcement) {

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess) {
    GTEST_SKIP();
  }

  RingAllreduceTiling tiling;
  int32_t world_size = static_cast<int32_t>(device_count) + 1;

  auto r = validate_ring_allreduce_host_tiling(
      /*count=*/0,
      /*world_size=*/world_size,
      /*num_channels=*/1,
      /*tile_elems=*/256,
      &tiling);

  EXPECT_TRUE(r.ok()) << (r.error_reason ? r.error_reason : "<no error>");
}

TEST(RingAllreduceHost, TilingRejectsExceedMaxGridSize) {

  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count <= 0) {
    GTEST_SKIP();
  }

  cudaDeviceProp prop{};
  EXPECT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);

  uint64_t count = static_cast<uint64_t>(prop.maxGridSize[0]) + 1ull;

  RingAllreduceTiling tiling;
  auto r = validate_ring_allreduce_host_tiling(
      count,
      /*world_size=*/1,
      /*num_channels=*/1,
      /*tile_elems=*/1,
      &tiling);

  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status, cutlass::Status::kErrorInvalidProblem);
}
