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
    \brief Host-side tiling utilities for the Blackwell ring allreduce prototype.

    NOTE: This API is experimental and may change without notice.
*/

#pragma once

#include "cutlass/cutlass.h"

#include <cstdint>

namespace cutlass::distributed::collective {

/// Derived tiling quantities computed by the host.
///
/// Design invariants (for count > 0):
///   - tile_elems > 0
///   - num_chunks_total > 0
///   - max_chunk_elems > 0
///   - tiles_per_chunk > 0
///   - num_tiles_total > 0
struct RingAllreduceTiling {
  uint32_t tile_elems = 0;
  uint32_t num_chunks_total = 0;
  uint64_t max_chunk_elems = 0;
  uint32_t tiles_per_chunk = 0;
  uint32_t num_tiles_total = 0;
};

/// Result of host-side tiling computation.
///
/// On failure, status != kSuccess and error_reason is a non-null static string.
struct RingAllreduceTilingResult {
  cutlass::Status status = cutlass::Status::kSuccess;
  RingAllreduceTiling tiling{};
  char const* error_reason = nullptr;

  static inline CUTLASS_HOST
  RingAllreduceTilingResult success(RingAllreduceTiling tiling_) {
    RingAllreduceTilingResult r;
    r.status = cutlass::Status::kSuccess;
    r.tiling = tiling_;
    r.error_reason = nullptr;
    return r;
  }

  static inline CUTLASS_HOST
  RingAllreduceTilingResult failure(cutlass::Status status_, char const* reason) {
    RingAllreduceTilingResult r;
    r.status = status_;
    r.error_reason = reason;
    return r;
  }

  CUTLASS_HOST
  bool ok() const {
    return status == cutlass::Status::kSuccess;
  }
};

/// Overflow-safe ceil-div for unsigned 64-bit integers.
static inline CUTLASS_HOST
uint64_t ceil_div_u64(uint64_t dividend, uint64_t divisor) {
  // divisor==0 is treated as invalid by the callers.
  return dividend / divisor + (dividend % divisor != 0);
}

/// Computes host-side tiling quantities with overflow-safe arithmetic.
///
/// This follows the rules defined in ring_allreduce/design/README.md:
///   - num_chunks_total = world_size * num_channels   (must fit uint32_t)
///   - max_chunk_elems = ceil_div(count, num_chunks_total)
///   - tiles_per_chunk = ceil_div(max_chunk_elems, tile_elems) (must fit uint32_t and be > 0)
///   - num_tiles_total = num_channels * tiles_per_chunk (must fit uint32_t and be > 0)
///
/// Special case:
///   - count == 0 returns success with num_tiles_total == 0 and tiles_per_chunk == 0.
static inline CUTLASS_HOST
RingAllreduceTilingResult compute_ring_allreduce_tiling(
    uint64_t count,
    int32_t world_size,
    int32_t num_channels,
    uint32_t tile_elems) {

  constexpr uint64_t kMaxU32 = 0xffff'ffffull;

  if (world_size <= 0) {
    return RingAllreduceTilingResult::failure(cutlass::Status::kErrorInvalidProblem,
                                              "world_size must be > 0");
  }

  if (num_channels <= 0) {
    return RingAllreduceTilingResult::failure(cutlass::Status::kErrorInvalidProblem,
                                              "num_channels must be > 0");
  }

  if (tile_elems == 0) {
    return RingAllreduceTilingResult::failure(cutlass::Status::kErrorInvalidProblem,
                                              "tile_elems must be > 0");
  }

  uint64_t num_chunks_total_u64 = uint64_t(world_size) * uint64_t(num_channels);
  if (num_chunks_total_u64 == 0 || num_chunks_total_u64 > kMaxU32) {
    return RingAllreduceTilingResult::failure(cutlass::Status::kErrorInvalidProblem,
                                              "num_chunks_total overflows uint32_t");
  }

  RingAllreduceTiling tiling;
  tiling.tile_elems = tile_elems;
  tiling.num_chunks_total = static_cast<uint32_t>(num_chunks_total_u64);

  if (count == 0) {
    tiling.max_chunk_elems = 0;
    tiling.tiles_per_chunk = 0;
    tiling.num_tiles_total = 0;
    return RingAllreduceTilingResult::success(tiling);
  }

  tiling.max_chunk_elems = ceil_div_u64(count, num_chunks_total_u64);
  if (tiling.max_chunk_elems == 0) {
    return RingAllreduceTilingResult::failure(cutlass::Status::kErrorInternal,
                                              "max_chunk_elems computed as 0 for count>0");
  }

  uint64_t tiles_per_chunk_u64 = ceil_div_u64(tiling.max_chunk_elems, static_cast<uint64_t>(tile_elems));
  if (tiles_per_chunk_u64 == 0 || tiles_per_chunk_u64 > kMaxU32) {
    return RingAllreduceTilingResult::failure(cutlass::Status::kErrorInvalidProblem,
                                              "tiles_per_chunk overflows uint32_t");
  }

  tiling.tiles_per_chunk = static_cast<uint32_t>(tiles_per_chunk_u64);

  uint64_t num_tiles_total_u64 = uint64_t(num_channels) * tiles_per_chunk_u64;
  if (num_tiles_total_u64 == 0 || num_tiles_total_u64 > kMaxU32) {
    return RingAllreduceTilingResult::failure(cutlass::Status::kErrorInvalidProblem,
                                              "num_tiles_total overflows uint32_t");
  }

  tiling.num_tiles_total = static_cast<uint32_t>(num_tiles_total_u64);

  return RingAllreduceTilingResult::success(tiling);
}

} // namespace cutlass::distributed::collective
