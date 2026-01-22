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

#include "cutlass/experimental/distributed/collective/ring_allreduce_types.hpp"

#include <cstdint>

namespace {

using cutlass::distributed::collective::flag_index_u64;
using cutlass::distributed::collective::safe_ptr_add;
using cutlass::distributed::collective::world_size_enabled;
using cutlass::distributed::collective::world_size_supported;

static inline int32_t expected_left(int32_t rank, int32_t N) {
  return (rank == 0) ? (N - 1) : (rank - 1);
}

static inline int32_t expected_right(int32_t rank, int32_t N) {
  return (rank + 1 == N) ? 0 : (rank + 1);
}

static inline int32_t dec_wrap(int32_t x, int32_t N) {
  return (x == 0) ? (N - 1) : (x - 1);
}

} // namespace

TEST(RingAllreduceHelpers, WorldSizeSupportedAndEnabledAllowlist) {

  struct Case {
    int32_t world_size;
    bool supported;
    bool enabled;
  };

  // Supported allowlist from ring_allreduce/README.md.
  // Enabled is a staging gate (currently {1,2,4,8}).
  Case cases[] = {
      {-1, false, false},
      {0, false, false},
      {1, true, true},
      {2, true, true},
      {3, false, false},
      {4, true, true},
      {5, false, false},
      {8, true, true},
      {9, false, false},
  };

  for (auto const& c : cases) {
    EXPECT_EQ(world_size_supported(c.world_size), c.supported) << "world_size=" << c.world_size;
    EXPECT_EQ(world_size_enabled(c.world_size), c.enabled) << "world_size=" << c.world_size;

    if (c.enabled) {
      EXPECT_TRUE(c.supported) << "enabled must imply supported";
    }
  }
}

TEST(RingAllreduceHelpers, RingMathFormulas) {

  // Spec formulas in ring_allreduce/README.md.
  constexpr int32_t kNs[] = {1, 2, 4, 8};

  for (int32_t N : kNs) {
    for (int32_t rank = 0; rank < N; ++rank) {

      int32_t left = (rank + N - 1) % N;
      int32_t right = (rank + 1) % N;

      EXPECT_EQ(left, expected_left(rank, N)) << "N=" << N << " rank=" << rank;
      EXPECT_EQ(right, expected_right(rank, N)) << "N=" << N << " rank=" << rank;

      // Owned chunk after reduce-scatter.
      int32_t owned_chunk = (rank + 1) % N;
      EXPECT_EQ(owned_chunk, right) << "N=" << N << " rank=" << rank;

      // RS recv chunk id sequence.
      int32_t exp_rs = expected_left(rank, N);
      for (int32_t s = 0; s + 1 < N; ++s) {
        int32_t recv_chunk_rs = (rank + N - s - 1) % N;
        EXPECT_EQ(recv_chunk_rs, exp_rs) << "N=" << N << " rank=" << rank << " s=" << s;
        exp_rs = dec_wrap(exp_rs, N);
      }

      // AG recv chunk id sequence.
      int32_t exp_ag = rank;
      for (int32_t s = 0; s + 1 < N; ++s) {
        int32_t recv_chunk_ag = (rank + N - s) % N;
        EXPECT_EQ(recv_chunk_ag, exp_ag) << "N=" << N << " rank=" << rank << " s=" << s;
        exp_ag = dec_wrap(exp_ag, N);
      }
    }
  }
}

TEST(RingAllreduceHelpers, FlagIndexUsesU64Intermediates) {

  EXPECT_EQ(flag_index_u64(/*step=*/0, /*tile_linear=*/0, /*num_tiles_total=*/1), 0u);
  EXPECT_EQ(flag_index_u64(/*step=*/1, /*tile_linear=*/7, /*num_tiles_total=*/10), 17u);

  // Discriminating case: 65536 * 65536 == 2^32, which overflows uint32_t but
  // fits in uint64_t. A broken implementation that multiplies as uint32_t would
  // return just tile_linear.
  uint32_t step = 0x1'0000u;
  uint32_t tile_linear = 7u;
  uint32_t num_tiles_total = 0x1'0000u;

  uint64_t expected = uint64_t(step) * uint64_t(num_tiles_total) + uint64_t(tile_linear);
  EXPECT_EQ(flag_index_u64(step, tile_linear, num_tiles_total), expected);
}

TEST(RingAllreduceHelpers, SafePtrAddRejectsNullAndOOB) {

  int data[4] = {0, 1, 2, 3};

  EXPECT_EQ(safe_ptr_add(data, /*idx=*/0, /*len=*/4), data);
  EXPECT_EQ(safe_ptr_add(data, /*idx=*/3, /*len=*/4), data + 3);

  EXPECT_EQ(safe_ptr_add(data, /*idx=*/4, /*len=*/4), nullptr);
  EXPECT_EQ(safe_ptr_add(data, /*idx=*/0, /*len=*/0), nullptr);

  int* null_data = nullptr;
  EXPECT_EQ(safe_ptr_add(null_data, /*idx=*/0, /*len=*/4), nullptr);

  int const cdata[4] = {0, 1, 2, 3};
  EXPECT_EQ(safe_ptr_add(cdata, /*idx=*/2, /*len=*/4), cdata + 2);
}
