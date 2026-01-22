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

#include "cutlass/experimental/distributed/collective/ring_allreduce_tiling.hpp"

#include <cstdint>

namespace {

using cutlass::distributed::collective::compute_ring_allreduce_tiling;

} // namespace

TEST(RingAllreduceTiling, ValidConfig) {

  auto r = compute_ring_allreduce_tiling(/*count=*/1024, /*world_size=*/4, /*num_channels=*/2, /*tile_elems=*/256);

  ASSERT_TRUE(r.ok()) << (r.error_reason ? r.error_reason : "<no error>");

  EXPECT_EQ(r.tiling.tile_elems, 256u);
  EXPECT_EQ(r.tiling.num_chunks_total, 8u);
  EXPECT_EQ(r.tiling.max_chunk_elems, 128u);
  EXPECT_EQ(r.tiling.tiles_per_chunk, 1u);
  EXPECT_EQ(r.tiling.num_tiles_total, 2u);
}

TEST(RingAllreduceTiling, CountZeroIsNoOp) {

  auto r = compute_ring_allreduce_tiling(/*count=*/0, /*world_size=*/4, /*num_channels=*/2, /*tile_elems=*/256);

  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.tiling.max_chunk_elems, 0u);
  EXPECT_EQ(r.tiling.tiles_per_chunk, 0u);
  EXPECT_EQ(r.tiling.num_tiles_total, 0u);
}

TEST(RingAllreduceTiling, RejectsNumChunksOverflow) {

  // 65536 * 65536 == 2^32, which does not fit in uint32_t.
  auto r = compute_ring_allreduce_tiling(/*count=*/1, /*world_size=*/65536, /*num_channels=*/65536, /*tile_elems=*/1);

  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status, cutlass::Status::kErrorInvalidProblem);
}

TEST(RingAllreduceTiling, RejectsTilesPerChunkOverflow) {

  // tiles_per_chunk == count when world_size=num_channels=tile_elems=1.
  constexpr uint64_t kCount = 0x1'0000'0000ull; // 2^32
  auto r = compute_ring_allreduce_tiling(/*count=*/kCount, /*world_size=*/1, /*num_channels=*/1, /*tile_elems=*/1);

  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status, cutlass::Status::kErrorInvalidProblem);
}

TEST(RingAllreduceTiling, RejectsNumTilesTotalOverflow) {

  // max_chunk_elems == 0xffffffff and tiles_per_chunk == 0xffffffff, so num_tiles_total == 2 * 0xffffffff.
  constexpr uint64_t kCount = 0xffff'ffffull * 2ull;
  auto r = compute_ring_allreduce_tiling(/*count=*/kCount, /*world_size=*/1, /*num_channels=*/2, /*tile_elems=*/1);

  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status, cutlass::Status::kErrorInvalidProblem);
}
