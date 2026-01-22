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
    \brief Device-side drain counter helpers for the Blackwell ring allreduce prototype.

    NOTE: This API is experimental and may change without notice.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "ring_allreduce_types.hpp"

#include <cuda/atomic>

#include <cstdint>

namespace cutlass::distributed::collective {

struct RingAllreduceDrainConfig {
  uint32_t timeout_iters = 0;   // 0 = infinite
  uint64_t timeout_cycles = 0;  // 0 = disabled
  uint32_t poll_sleep_start = 0;
  uint32_t poll_sleep_ns = 0;   // 0 = backoff disabled
};

CUTLASS_DEVICE
bool ring_allreduce_is_thread0() {
#if defined(__CUDA_ARCH__)
  return ((threadIdx.x | threadIdx.y | threadIdx.z) == 0);
#else
  return false;
#endif
}

CUTLASS_DEVICE
bool ring_allreduce_is_cta0() {
#if defined(__CUDA_ARCH__)
  return ((blockIdx.x | blockIdx.y | blockIdx.z) == 0);
#else
  return false;
#endif
}

// Writer-side helper. Must be called exactly once per participating CTA by thread0.
//
// Returns the pre-increment counter value (as returned by fetch_add).
CUTLASS_DEVICE
uint32_t ring_allreduce_signal_tile_finished(
    RingAllreduceDeviceAtomicU32* self_tiles_finished) {

  if (!ring_allreduce_is_thread0() || !self_tiles_finished) {
    return 0u;
  }

  return self_tiles_finished->fetch_add(1u, cuda::memory_order_relaxed);
}

// First-writer-wins error publication, then abort publication.
CUTLASS_DEVICE
void ring_allreduce_publish_error_and_abort(
    RingAllreduceSystemAtomicU32* self_error,
    RingAllreduceSystemAtomicU32* self_abort,
    RingAllreduceError desired_error) {

  if (!ring_allreduce_is_thread0() || !self_error || !self_abort) {
    return;
  }

  uint32_t expected_ok = static_cast<uint32_t>(RingAllreduceError::kOk);
  (void)self_error->compare_exchange_strong(
      expected_ok,
      static_cast<uint32_t>(desired_error),
      cuda::memory_order_release,
      cuda::memory_order_acquire);

  self_abort->store(1u, cuda::memory_order_release);
}

// Any-thread variant of ring_allreduce_publish_error_and_abort. NOT thread0-gated.
//
// Only call from the warp-specialized SMEM allowlist guard (verbatim):
//   if (ring_allreduce_is_cta0() && warp_id == 6 && lane == 0) { ... }
//
// Only intended for warp6 lane0 in CTA0. Do not call from polling loops.
CUTLASS_DEVICE
void ring_allreduce_publish_error_and_abort_any_thread(
    RingAllreduceSystemAtomicU32* self_error,
    RingAllreduceSystemAtomicU32* self_abort,
    RingAllreduceError desired_error) {

  if (!self_error || !self_abort) {
    return;
  }

  // Defense in depth: restrict error publication to CTA0.
  if (!ring_allreduce_is_cta0()) {
    return;
  }

  uint32_t expected_ok = static_cast<uint32_t>(RingAllreduceError::kOk);
  (void)self_error->compare_exchange_strong(
      expected_ok,
      static_cast<uint32_t>(desired_error),
      cuda::memory_order_release,
      cuda::memory_order_acquire);

  self_abort->store(1u, cuda::memory_order_release);
}

// Local status derivation:
//   local_status = (error != kOk) ? error : (abort ? kAbortObserved : kOk)
CUTLASS_DEVICE
RingAllreduceError ring_allreduce_local_status(
    RingAllreduceSystemAtomicU32 const* self_abort,
    RingAllreduceSystemAtomicU32 const* self_error) {

  if (!self_abort || !self_error) {
    return RingAllreduceError::kInvalidParams;
  }

  uint32_t abort = self_abort->load(cuda::memory_order_acquire);
  uint32_t err_u32 = self_error->load(cuda::memory_order_relaxed);

  RingAllreduceError err = static_cast<RingAllreduceError>(err_u32);
  if (err != RingAllreduceError::kOk) {
    return err;
  }

  return abort ? RingAllreduceError::kAbortObserved : RingAllreduceError::kOk;
}

// CTA0 drain wait. Must be called only by (CTA0 && thread0).
CUTLASS_DEVICE
RingAllreduceError ring_allreduce_drain_tiles_finished(
    RingAllreduceDeviceAtomicU32* self_tiles_finished,
    uint32_t expected_tiles,
    RingAllreduceSystemAtomicU32* self_abort,
    RingAllreduceSystemAtomicU32* self_error,
    RingAllreduceDrainConfig cfg) {

#if defined(__CUDA_ARCH__)

  if (!ring_allreduce_is_cta0() || !ring_allreduce_is_thread0()) {
    return ring_allreduce_local_status(self_abort, self_error);
  }

  if (expected_tiles == 0u || !self_tiles_finished || !self_abort || !self_error) {
    ring_allreduce_publish_error_and_abort(self_error, self_abort, RingAllreduceError::kInvalidParams);
    return ring_allreduce_local_status(self_abort, self_error);
  }

  uint64_t iters = 0;
  uint64_t start_cycles = (cfg.timeout_cycles != 0) ? clock64() : 0;

  while (true) {
    uint32_t done = self_tiles_finished->load(cuda::memory_order_relaxed);
    if (done == expected_tiles) {
      break;
    }

    // Abort-aware poll (does not change logic; keeps loop responsive).
    (void)self_abort->load(cuda::memory_order_acquire);

    bool timed_out_iters = (cfg.timeout_iters != 0) &&
        (iters >= static_cast<uint64_t>(cfg.timeout_iters));

    bool timed_out_cycles = false;
    if (cfg.timeout_cycles != 0) {
      timed_out_cycles = (clock64() - start_cycles) >= cfg.timeout_cycles;
    }

    if (timed_out_iters || timed_out_cycles) {
      ring_allreduce_publish_error_and_abort(self_error, self_abort, RingAllreduceError::kTimeout);
      break; // proceed (never deadlock when timeouts enabled)
    }

    if (cfg.poll_sleep_ns > 0 && iters >= static_cast<uint64_t>(cfg.poll_sleep_start)) {
      #if (__CUDA_ARCH__ >= 700)
        __nanosleep(cfg.poll_sleep_ns);
      #endif
    }

    ++iters;
  }

  return ring_allreduce_local_status(self_abort, self_error);

#else

  CUTLASS_UNUSED(self_tiles_finished);
  CUTLASS_UNUSED(expected_tiles);
  CUTLASS_UNUSED(cfg);

  return ring_allreduce_local_status(self_abort, self_error);

#endif
}

// Convenience wrapper for CTA0:
// 1) signal CTA0 tile finished
// 2) drain
CUTLASS_DEVICE
RingAllreduceError ring_allreduce_cta0_signal_and_drain(
    RingAllreduceDeviceAtomicU32* self_tiles_finished,
    uint32_t expected_tiles,
    RingAllreduceSystemAtomicU32* self_abort,
    RingAllreduceSystemAtomicU32* self_error,
    RingAllreduceDrainConfig cfg) {

  if (!ring_allreduce_is_cta0() || !ring_allreduce_is_thread0()) {
    return ring_allreduce_local_status(self_abort, self_error);
  }

  (void)ring_allreduce_signal_tile_finished(self_tiles_finished);

  return ring_allreduce_drain_tiles_finished(
      self_tiles_finished, expected_tiles, self_abort, self_error, cfg);
}

} // namespace cutlass::distributed::collective
