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
    \brief Core ABI and helper types for the Blackwell ring allreduce prototype.

    NOTE: This API is experimental and may change without notice.
*/

#pragma once

#include "cutlass/cutlass.h"

#include <cuda/atomic>

#include <cstdint>

namespace cutlass::distributed::collective {

using RingAllreduceSystemAtomicU32 = cuda::atomic<uint32_t, cuda::thread_scope_system>;
using RingAllreduceDeviceAtomicU32 = cuda::atomic<uint32_t, cuda::thread_scope_device>;

/// Error/status codes reported by the ring allreduce prototype.
///
/// NOTE: Do not assume numeric ordering matches precedence. Use merge_status().
enum class RingAllreduceError : uint32_t {
  kOk = 0,
  kInvalidParams = 1,
  kTimeout = 2,
  kAbortObserved = 3,
};

/// Deterministically merges two status values using the precedence:
///
///   kInvalidParams > kTimeout > kAbortObserved > kOk
///
/// This must not rely on enum numeric ordering.
CUTLASS_HOST_DEVICE
RingAllreduceError merge_status(RingAllreduceError a, RingAllreduceError b) {
  if (a == RingAllreduceError::kInvalidParams || b == RingAllreduceError::kInvalidParams) {
    return RingAllreduceError::kInvalidParams;
  }
  if (a == RingAllreduceError::kTimeout || b == RingAllreduceError::kTimeout) {
    return RingAllreduceError::kTimeout;
  }
  if (a == RingAllreduceError::kAbortObserved || b == RingAllreduceError::kAbortObserved) {
    return RingAllreduceError::kAbortObserved;
  }
  return RingAllreduceError::kOk;
}

/// Params ABI for ring_allreduce_sm100.
///
/// This is a single source of truth for the device kernel's launch parameters.
template <typename T, int kMaxWorldSize>
struct RingAllreduceParams {
  static_assert(kMaxWorldSize > 0, "kMaxWorldSize must be > 0");

  // Identity
  int32_t world_size;
  int32_t rank;
  uint32_t epoch;

  // Shape
  uint64_t count;
  int32_t num_channels;

  // Host-precomputed tiling (MUST be host-computed)
  uint32_t tile_elems;
  uint32_t num_chunks_total;
  uint64_t max_chunk_elems;
  uint32_t tiles_per_chunk;
  uint32_t num_tiles_total;

  // Timeouts
  uint32_t timeout_iters;  // 0 = infinite
  uint64_t timeout_cycles; // 0 = disabled

  // Backoff
  uint32_t poll_sleep_start;
  uint32_t poll_sleep_ns;

  // Payload
  T* self_data;

  // Local flags
  RingAllreduceSystemAtomicU32* self_rs_ready; // len world_size * num_tiles_total
  RingAllreduceSystemAtomicU32* self_ag_ready; // len world_size * num_tiles_total
  RingAllreduceSystemAtomicU32* self_abort;    // single element
  RingAllreduceSystemAtomicU32* self_error;    // single element (RingAllreduceError value)

  // Rank-local drain counter (atomic object)
  RingAllreduceDeviceAtomicU32* self_tiles_finished; // single element

  // Ring-token barrier state (system-scope, epoch-tagged tokens)
  RingAllreduceSystemAtomicU32* self_barrier_gather_token;
  RingAllreduceSystemAtomicU32* self_barrier_gather_status;  // RingAllreduceError value
  RingAllreduceSystemAtomicU32* self_barrier_release_token;
  RingAllreduceSystemAtomicU32* self_barrier_release_status; // RingAllreduceError value

  // Peer pointers
  T* peer_data[kMaxWorldSize];
  RingAllreduceSystemAtomicU32* peer_rs_ready[kMaxWorldSize];
  RingAllreduceSystemAtomicU32* peer_ag_ready[kMaxWorldSize];
  RingAllreduceSystemAtomicU32* peer_abort[kMaxWorldSize];

  RingAllreduceSystemAtomicU32* peer_barrier_gather_token[kMaxWorldSize];
  RingAllreduceSystemAtomicU32* peer_barrier_gather_status[kMaxWorldSize];
  RingAllreduceSystemAtomicU32* peer_barrier_release_token[kMaxWorldSize];
  RingAllreduceSystemAtomicU32* peer_barrier_release_status[kMaxWorldSize];

  // Debug/test hooks
  uint32_t debug_abort_rank;
  uint32_t debug_abort_ag_step;
  uint32_t debug_abort_before_ag_publish; // 0/1
  uint32_t debug_abort_after_ag_publish;  // 0/1

  // Release-phase delay hook (for testing RELEASE semantics)
  uint32_t debug_release_delay_rank;  // rank to delay
  uint32_t debug_release_delay_iters; // iterations of __nanosleep

  // Deterministic jitter hook (overwrite-safety stress)
  uint32_t debug_jitter_seed;
  uint32_t debug_jitter_max_iters;
  uint32_t debug_jitter_mask;
};

/// True iff the world size is in the supported allowlist.
///
/// The ring allreduce prototype only supports power-of-two world sizes up to 8.
CUTLASS_HOST_DEVICE
bool world_size_supported(int32_t world_size) {
  return (world_size == 1) || (world_size == 2) || (world_size == 4) || (world_size == 8);
}

/// True iff the world size is currently enabled for end-to-end execution.
///
/// This is intentionally separate from world_size_supported() to allow staged
/// rollout while keeping validators and math helpers aligned with the final
/// spec. Currently, the enabled set matches the supported allowlist.
CUTLASS_HOST_DEVICE
bool world_size_enabled(int32_t world_size) {
  return world_size_supported(world_size);
}

/// Computes a readiness-flag array index using 64-bit intermediates.
CUTLASS_HOST_DEVICE
uint64_t flag_index_u64(uint32_t step, uint32_t tile_linear, uint32_t num_tiles_total) {
  return uint64_t(step) * uint64_t(num_tiles_total) + uint64_t(tile_linear);
}

/// Bounds-checking pointer addition that avoids undefined behavior.
///
/// NOTE: Pointer arithmetic on a null pointer is undefined even if the result is
/// never dereferenced. This helper ensures `base + idx` is only evaluated when
/// it is well-defined.
template <typename PtrT>
CUTLASS_HOST_DEVICE
PtrT* safe_ptr_add(PtrT* base, uint64_t idx, uint64_t len) {
  if (!base) {
    return nullptr;
  }
  if (idx >= len) {
    return nullptr;
  }
  return base + idx;
}

} // namespace cutlass::distributed::collective
