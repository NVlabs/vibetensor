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
    \brief Shared-memory helpers for the Blackwell ring allreduce prototype.

    NOTE: This API is experimental and may change without notice.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "ring_allreduce_drain.hpp"

#include <cstdint>

// Compile-time feature gate for the warp-specialized SMEM path.
//
// Enable with CUTLASS_RING_ALLREDUCE_ENABLE_WARP_SPECIALIZED_SMEM=1 (defaults to 1).
#ifndef CUTLASS_RING_ALLREDUCE_ENABLE_WARP_SPECIALIZED_SMEM
#define CUTLASS_RING_ALLREDUCE_ENABLE_WARP_SPECIALIZED_SMEM 1
#endif

// Compile-time gate for selecting the warp-specialized SMEM all-gather implementation.
//
// This gates only the AG call-site selection inside
// detail::ring_allreduce_sm100_tile_warp_specialized_smem(...).
//
// IMPORTANT: This macro affects header-defined device code; it must remain
// consistent across all translation units in a linked binary.
//
// Default 1: RS+AG use warp-specialized SMEM.
#ifndef CUTLASS_RING_ALLREDUCE_ENABLE_WARP_SPECIALIZED_SMEM_AG
#define CUTLASS_RING_ALLREDUCE_ENABLE_WARP_SPECIALIZED_SMEM_AG 1
#endif

// Test-only knob to force-disable NamedBarrier-based implementations.
//
// When set, all code that would call cutlass::arch::NamedBarrier must be
// compiled out at the preprocessor level to avoid brkpt traps when
// CUDA_BARRIER_ENABLED==0.
#ifndef CUTLASS_RING_ALLREDUCE_TEST_FORCE_NAMED_BARRIER_DISABLED
#define CUTLASS_RING_ALLREDUCE_TEST_FORCE_NAMED_BARRIER_DISABLED 0
#endif

// Test-only instrumentation for wait_flag_warp acquire-confirm counts.
//
// IMPORTANT: this must remain disabled by default to avoid ODR violations when
// this header is included in multiple translation units.
#ifndef CUTLASS_RING_ALLREDUCE_TEST_WAIT_FLAG_WARP_ACQUIRE_COUNTER
#define CUTLASS_RING_ALLREDUCE_TEST_WAIT_FLAG_WARP_ACQUIRE_COUNTER 0
#endif

namespace cutlass::distributed::collective {
namespace detail {

// Maximum number of elements staged per ping-pong buffer stage.
//
// TODO: Derive from the final SMEM layout / byte budget. For now this matches
// existing tests which use tile_elems == 256.
static constexpr uint32_t kStageElemsMax = 256;

#if CUTLASS_RING_ALLREDUCE_TEST_WAIT_FLAG_WARP_ACQUIRE_COUNTER
// Counts the number of acquire-confirm loads executed by wait_flag_warp.
// Expected: 32 on success (one per lane), 0 on abort/timeout/invalid.
//
// Declared static to avoid ODR violations if multiple TUs enable the test macro.
static __device__ uint32_t ring_allreduce_wait_flag_warp_acquire_count = 0;
#endif

// Abort + timeout-aware wait for a system-scope atomic readiness flag.
//
// Contract:
// - Must be called by a full warp with warp-uniform control flow.
// - If error/timeout publication is required, this must be called by warp0 so
//   lane0 is thread0 (ring_allreduce_publish_error_and_abort is thread0-gated).
// - Lane0 performs relaxed polling.
// - After observing flag==epoch, each payload-reading lane performs an
//   acquire-confirmation load before reading payload.
CUTLASS_DEVICE
bool wait_flag_warp(
    RingAllreduceSystemAtomicU32 const* flag,
    uint32_t epoch,
    RingAllreduceSystemAtomicU32* self_abort,
    RingAllreduceSystemAtomicU32* self_error,
    RingAllreduceSystemAtomicU32 const* peer_abort,
    RingAllreduceDrainConfig cfg) {

#if defined(__CUDA_ARCH__)

  uint32_t lane = threadIdx.x & 0x1Fu;

  uint32_t ready = 0u;

  if (lane == 0u) {
    bool ptrs_ok = flag && self_abort && self_error && peer_abort && (epoch != 0);
    if (!ptrs_ok) {
      ring_allreduce_publish_error_and_abort(self_error, self_abort, RingAllreduceError::kInvalidParams);
    }
    else {
      bool timeouts_enabled = (cfg.timeout_iters != 0) || (cfg.timeout_cycles != 0);

      uint64_t iters = 0;
      uint64_t start_cycles = (timeouts_enabled && cfg.timeout_cycles != 0) ? clock64() : 0;

      while (true) {
        if (flag->load(cuda::memory_order_relaxed) == epoch) {
          ready = 1u;
          break;
        }

        // Abort-aware wait.
        uint32_t s = self_abort->load(cuda::memory_order_acquire);
        uint32_t l = peer_abort->load(cuda::memory_order_acquire);
        if ((s | l) != 0u) {
          // Hop-by-hop abort propagation: if we observe the left neighbor aborted,
          // set our local abort flag so our right neighbor can observe it.
          if (l != 0u) {
            self_abort->store(1u, cuda::memory_order_release);
          }
          break;
        }

        if (timeouts_enabled) {
          bool timed_out_iters = (cfg.timeout_iters != 0) && (iters >= uint64_t(cfg.timeout_iters));
          bool timed_out_cycles = false;
          if (cfg.timeout_cycles != 0) {
            timed_out_cycles = (clock64() - start_cycles) >= cfg.timeout_cycles;
          }
          if (timed_out_iters || timed_out_cycles) {
            ring_allreduce_publish_error_and_abort(self_error, self_abort, RingAllreduceError::kTimeout);
            break;
          }
        }

        if (cfg.poll_sleep_ns > 0 && iters >= uint64_t(cfg.poll_sleep_start)) {
          #if (__CUDA_ARCH__ >= 700)
            __nanosleep(cfg.poll_sleep_ns);
          #endif
        }

        ++iters;
      }
    }
  }

  // Broadcast lane0's ready state.
  ready = __shfl_sync(0xFFFF'FFFFu, ready, 0);

  if (ready == 0u) {
    return false;
  }

  // Acquire-confirmation: each payload-reading lane establishes ordering for its
  // subsequent GMEM loads.
  uint32_t observed = flag->load(cuda::memory_order_acquire);

  #if CUTLASS_RING_ALLREDUCE_TEST_WAIT_FLAG_WARP_ACQUIRE_COUNTER
    atomicAdd(&ring_allreduce_wait_flag_warp_acquire_count, 1u);
  #endif

  bool confirmed = (observed == epoch);
  int all_confirmed = __all_sync(0xFFFF'FFFFu, confirmed ? 1 : 0);

  if (!all_confirmed) {
    if (lane == 0u) {
      ring_allreduce_publish_error_and_abort(self_error, self_abort, RingAllreduceError::kInvalidParams);
    }
    return false;
  }

  return true;

#else

  CUTLASS_UNUSED(flag);
  CUTLASS_UNUSED(epoch);
  CUTLASS_UNUSED(self_abort);
  CUTLASS_UNUSED(self_error);
  CUTLASS_UNUSED(peer_abort);
  CUTLASS_UNUSED(cfg);
  return false;

#endif
}

} // namespace detail
} // namespace cutlass::distributed::collective
