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
    \brief Device kernel for the Blackwell ring allreduce prototype.

    NOTE: This API is experimental and may change without notice.
*/

#pragma once

#include "cutlass/cutlass.h"

// Required for CUDA_BARRIER_ENABLED and cutlass::arch::NamedBarrier constants.
#include "cutlass/arch/barrier.h"

#include "ring_allreduce_barrier_sm100.cuh"
#include "ring_allreduce_drain.hpp"
#include "ring_allreduce_types.hpp"
#include "ring_allreduce_smem.hpp"

#include <cstdint>

// Test-only knob to force-disable NamedBarrier-based implementations.
#ifndef CUTLASS_RING_ALLREDUCE_TEST_FORCE_NAMED_BARRIER_DISABLED
#define CUTLASS_RING_ALLREDUCE_TEST_FORCE_NAMED_BARRIER_DISABLED 0
#endif

// Test-only instrumentation: expose which warp-specialized SMEM RS implementation was used.
//
// Disabled by default to avoid adding global device symbols in normal builds.
#ifndef CUTLASS_RING_ALLREDUCE_TEST_WARP_SPECIALIZED_SMEM_RS_IMPL_TAG
#define CUTLASS_RING_ALLREDUCE_TEST_WARP_SPECIALIZED_SMEM_RS_IMPL_TAG 0
#endif

namespace cutlass::distributed::collective {

namespace detail {


struct RingAllreduceTileRange {
  uint64_t base = 0;
  uint32_t len = 0;
};


template <typename T>
CUTLASS_DEVICE
RingAllreduceTileRange ring_allreduce_compute_tile_range(
    RingAllreduceParams<T, 8> const& p,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t chunk_id) {

#if defined(__CUDA_ARCH__)

  // global_chunk_id = channel_id * world_size + chunk_id
  uint32_t global_chunk_id = channel_id * static_cast<uint32_t>(p.world_size) + chunk_id;

  uint64_t chunk_base = uint64_t(global_chunk_id) * p.max_chunk_elems;
  uint64_t tile_base = chunk_base + uint64_t(tile_in_chunk) * uint64_t(p.tile_elems);

  uint64_t chunk_end = chunk_base + p.max_chunk_elems;
  if (chunk_end > p.count) {
    chunk_end = p.count;
  }

  uint64_t tile_end = tile_base + uint64_t(p.tile_elems);
  if (tile_end > chunk_end) {
    tile_end = chunk_end;
  }

  RingAllreduceTileRange r;
  r.base = tile_base;
  r.len = (tile_base < tile_end) ? static_cast<uint32_t>(tile_end - tile_base) : 0u;
  return r;

#else

  CUTLASS_UNUSED(p);
  CUTLASS_UNUSED(channel_id);
  CUTLASS_UNUSED(tile_in_chunk);
  CUTLASS_UNUSED(chunk_id);
  return {};

#endif
}

template <typename T>
CUTLASS_DEVICE
bool ring_allreduce_params_valid_ngpu(RingAllreduceParams<T, 8> const& p) {

#if defined(__CUDA_ARCH__)

  // Validate supported world sizes first (avoid modulo-by-zero and OOB peer table access).
  bool world_size_ok = (p.world_size == 1) || (p.world_size == 2) || (p.world_size == 4) || (p.world_size == 8);
  if (!world_size_ok) {
    return false;
  }

  if (p.rank < 0 || p.rank >= p.world_size) {
    return false;
  }

  if (p.epoch == 0) {
    return false;
  }

  // Minimal tiling validation (avoids divide-by-zero and obvious inconsistencies).
  if (p.num_channels <= 0) {
    return false;
  }

  if (p.tile_elems == 0 || p.tiles_per_chunk == 0 || p.num_tiles_total == 0) {
    return false;
  }

  if (p.max_chunk_elems == 0) {
    return false;
  }

  uint64_t expected_tiles_total = uint64_t(p.num_channels) * uint64_t(p.tiles_per_chunk);
  if (expected_tiles_total != uint64_t(p.num_tiles_total)) {
    return false;
  }

  uint32_t expected_chunks = static_cast<uint32_t>(p.world_size) * static_cast<uint32_t>(p.num_channels);
  if (p.num_chunks_total != expected_chunks) {
    return false;
  }

  // Required self pointers.
  bool self_ok = p.self_data &&
      p.self_rs_ready &&
      p.self_ag_ready &&
      p.self_abort &&
      p.self_error &&
      p.self_tiles_finished &&
      p.self_barrier_gather_token &&
      p.self_barrier_gather_status &&
      p.self_barrier_release_token &&
      p.self_barrier_release_status;

  if (!self_ok) {
    return false;
  }

  // Degenerate ring: no peer pointers required.
  if (p.world_size == 1) {
    return true;
  }

  // Spec orientation: left = (rank - 1 + N) % N.
  int32_t left = (p.rank + p.world_size - 1) % p.world_size;

  bool peer_ok = p.peer_data[left] &&
      p.peer_ag_ready[left] &&
      p.peer_abort[left] &&
      p.peer_barrier_gather_token[left] &&
      p.peer_barrier_gather_status[left] &&
      p.peer_barrier_release_token[left] &&
      p.peer_barrier_release_status[left];

  if (!peer_ok) {
    return false;
  }

  // RS uses peer_rs_ready waits only for N>2 (for N==2, the s>0 wait never executes).
  if (p.world_size > 2 && !p.peer_rs_ready[left]) {
    return false;
  }

  return true;

#else

  CUTLASS_UNUSED(p);
  return false;

#endif
}

// Abort + timeout-aware wait for a system-scope atomic readiness flag.
//
// Contract:
// - Thread0 performs relaxed polling.
// - After observing flag==epoch, each payload-reading thread performs an
//   acquire-confirmation load before reading payload.
CUTLASS_DEVICE
bool ring_allreduce_wait_flag(
    RingAllreduceSystemAtomicU32 const* flag,
    uint32_t epoch,
    RingAllreduceSystemAtomicU32* self_abort,
    RingAllreduceSystemAtomicU32* self_error,
    RingAllreduceSystemAtomicU32 const* peer_abort,
    RingAllreduceDrainConfig cfg) {

#if defined(__CUDA_ARCH__)

  __shared__ uint32_t ready;

  if (ring_allreduce_is_thread0()) {
    ready = 0u;

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

  __syncthreads();

  if (ready == 0u) {
    return false;
  }

  // Acquire-confirmation: each payload-reading thread establishes ordering for
  // its subsequent GMEM loads.
  bool confirmed = (flag->load(cuda::memory_order_acquire) == epoch);
  int all_confirmed = __syncthreads_and(confirmed ? 1 : 0);

  if (!all_confirmed) {
    if (ring_allreduce_is_thread0()) {
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


// Deterministic jitter injection for overwrite-safety stress.
//
// The jitter hooks intentionally add randomized latency at specific sites in the
// RS/AG loops to increase the likelihood of catching overwrite-safety and
// ordering bugs. The jitter is deterministic given the (seed, max_iters, mask)
// tuple and the per-call metadata (rank/tile/step/site).
//
// Mask bit assignments:
//  - bit0: RS after wait acquire-confirmation
//  - bit1: RS before publishing self_rs_ready
//  - bit2: AG after wait acquire-confirmation
//  - bit3: AG before publishing self_ag_ready
enum class RingAllreduceJitterSite : uint32_t {
  kRsAfterWait = 0u,
  kRsBeforePublish = 1u,
  kAgAfterWait = 2u,
  kAgBeforePublish = 3u,
};

CUTLASS_DEVICE
uint32_t ring_allreduce_jitter_hash(uint32_t x) {
#if defined(__CUDA_ARCH__)
  // Finalizer-style mix (adapted from MurmurHash3 finalizer).
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
#else
  CUTLASS_UNUSED(x);
  return 0u;
#endif
}

CUTLASS_DEVICE
void ring_allreduce_maybe_jitter(
    uint32_t seed,
    uint32_t max_iters,
    uint32_t mask,
    RingAllreduceJitterSite site,
    uint32_t step,
    uint32_t tile_linear,
    uint32_t rank,
    uint32_t epoch) {

#if defined(__CUDA_ARCH__)

  uint32_t site_bit = 1u << static_cast<uint32_t>(site);
  if (max_iters == 0u || (mask & site_bit) == 0u) {
    return;
  }

  // Derive a deterministic per-site seed.
  uint32_t x = seed;
  x ^= static_cast<uint32_t>(site) * 0x9e3779b9u;
  x ^= step * 0x85ebca6bu;
  x ^= tile_linear * 0xc2b2ae35u;
  x ^= rank * 0x27d4eb2fu;
  x ^= epoch * 0x165667b1u;

  uint32_t iters = ring_allreduce_jitter_hash(x) % max_iters;
  if (iters == 0u) {
    return;
  }

  // Delay the whole CTA via thread0. Other threads will naturally wait at the
  // next synchronization point (or at the end of the per-element loop).
  if (ring_allreduce_is_thread0()) {
    for (uint32_t i = 0; i < iters; ++i) {
      #if (__CUDA_ARCH__ >= 700)
        __nanosleep(40);
      #endif
    }
  }

#else

  CUTLASS_UNUSED(seed);
  CUTLASS_UNUSED(max_iters);
  CUTLASS_UNUSED(mask);
  CUTLASS_UNUSED(site);
  CUTLASS_UNUSED(step);
  CUTLASS_UNUSED(tile_linear);
  CUTLASS_UNUSED(rank);
  CUTLASS_UNUSED(epoch);

#endif
}

CUTLASS_DEVICE
void ring_allreduce_maybe_jitter_warp_lane0(
    uint32_t seed,
    uint32_t max_iters,
    uint32_t mask,
    RingAllreduceJitterSite site,
    uint32_t step,
    uint32_t tile_linear,
    uint32_t rank,
    uint32_t epoch,
    uint32_t lane) {

#if defined(__CUDA_ARCH__)

  uint32_t site_bit = 1u << static_cast<uint32_t>(site);
  if (max_iters == 0u || (mask & site_bit) == 0u) {
    return;
  }

  // Derive a deterministic per-site seed. Must match ring_allreduce_maybe_jitter.
  uint32_t x = seed;
  x ^= static_cast<uint32_t>(site) * 0x9e3779b9u;
  x ^= step * 0x85ebca6bu;
  x ^= tile_linear * 0xc2b2ae35u;
  x ^= rank * 0x27d4eb2fu;
  x ^= epoch * 0x165667b1u;

  uint32_t iters = ring_allreduce_jitter_hash(x) % max_iters;
  if (iters == 0u) {
    return;
  }

  if (lane == 0u) {
    for (uint32_t i = 0; i < iters; ++i) {
      #if (__CUDA_ARCH__ >= 700)
        __nanosleep(40);
      #endif
    }
  }

#else

  CUTLASS_UNUSED(seed);
  CUTLASS_UNUSED(max_iters);
  CUTLASS_UNUSED(mask);
  CUTLASS_UNUSED(site);
  CUTLASS_UNUSED(step);
  CUTLASS_UNUSED(tile_linear);
  CUTLASS_UNUSED(rank);
  CUTLASS_UNUSED(epoch);
  CUTLASS_UNUSED(lane);

#endif
}



template <typename T>
CUTLASS_DEVICE
bool ring_allreduce_sm100_tile_legacy_rs(
    RingAllreduceParams<T, 8> const& p,
    RingAllreduceDrainConfig cfg,
    uint32_t tile_linear,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len) {

  // -------------------------------
  // Reduce-scatter
  // -------------------------------
  bool rs_ok = true;

  for (uint32_t s = 0; s + 1 < N; ++s) {

    // recv_chunk_id_rs(s) = (rank - s - 1 + N) % N
    uint32_t chunk_rs = (r_u32 + N - s - 1) % N;

    // If s>0: wait on peer_rs_ready[left][flag_index(s, tile_linear)] == epoch.
    if (s > 0) {
      uint64_t wait_idx = flag_index_u64(s, tile_linear, p.num_tiles_total);
      RingAllreduceSystemAtomicU32 const* peer_flag = safe_ptr_add(
          p.peer_rs_ready[left],
          wait_idx,
          flags_len);

      bool got_peer = detail::ring_allreduce_wait_flag(
          peer_flag,
          p.epoch,
          p.self_abort,
          p.self_error,
          p.peer_abort[left],
          cfg);

      if (!got_peer) {
        rs_ok = false;
        break;
      }

      if (p.debug_jitter_max_iters != 0u &&
          (p.debug_jitter_mask & (1u << uint32_t(detail::RingAllreduceJitterSite::kRsAfterWait))) != 0u) {
        detail::ring_allreduce_maybe_jitter(
            p.debug_jitter_seed,
            p.debug_jitter_max_iters,
            p.debug_jitter_mask,
            detail::RingAllreduceJitterSite::kRsAfterWait,
            s,
            tile_linear,
            r_u32,
            p.epoch);
        // IMPORTANT: This barrier must remain CTA-uniform (do not move under
        // per-thread control flow), otherwise it can deadlock.
        __syncthreads();
      }
    }

    auto rs_tile = detail::ring_allreduce_compute_tile_range(
        p,
        channel_id,
        tile_in_chunk,
        chunk_rs);

    T* peer_left = p.peer_data[left];
    if (!peer_left) {
      rs_ok = false;
      ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
      break;
    }

    for (uint32_t e = static_cast<uint32_t>(threadIdx.x); e < rs_tile.len; e += static_cast<uint32_t>(blockDim.x)) {
      uint64_t idx = rs_tile.base + uint64_t(e);
      p.self_data[idx] = p.self_data[idx] + peer_left[idx];
    }

    if (p.debug_jitter_max_iters != 0u &&
        (p.debug_jitter_mask & (1u << uint32_t(detail::RingAllreduceJitterSite::kRsBeforePublish))) != 0u) {
      detail::ring_allreduce_maybe_jitter(
          p.debug_jitter_seed,
          p.debug_jitter_max_iters,
          p.debug_jitter_mask,
          detail::RingAllreduceJitterSite::kRsBeforePublish,
          s,
          tile_linear,
          r_u32,
          p.epoch);
    }

    // Publish self_rs_ready[flag_index(s+1, tile_linear)] = epoch.
    __syncthreads();
    __threadfence_system();
    __syncthreads();

    uint64_t rs_ready_idx = flag_index_u64(s + 1, tile_linear, p.num_tiles_total);
    RingAllreduceSystemAtomicU32* self_rs_flag = safe_ptr_add(p.self_rs_ready, rs_ready_idx, flags_len);

    if (ring_allreduce_is_thread0()) {
      if (!self_rs_flag) {
        ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
      }
      else {
        self_rs_flag->store(p.epoch, cuda::memory_order_release);
      }
    }

    if (!self_rs_flag) {
      rs_ok = false;
      break;
    }
  }

  return rs_ok;
}

template <typename T>
CUTLASS_DEVICE
void ring_allreduce_sm100_tile_legacy_ag(
    RingAllreduceParams<T, 8> const& p,
    RingAllreduceDrainConfig cfg,
    uint32_t tile_linear,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len) {

  // -------------------------------
  // Allgather
  // -------------------------------

  // Publish self_ag_ready[flag_index(0, tile_linear)] = epoch for owned_chunk.
  __syncthreads();
  __threadfence_system();
  __syncthreads();

  uint64_t ag_ready_idx = flag_index_u64(/*step=*/0, tile_linear, p.num_tiles_total);
  RingAllreduceSystemAtomicU32* self_ag_flag = safe_ptr_add(p.self_ag_ready, ag_ready_idx, flags_len);

  if (ring_allreduce_is_thread0()) {
    if (!self_ag_flag) {
      ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
    }
    else {
      self_ag_flag->store(p.epoch, cuda::memory_order_release);
    }
  }

  if (!self_ag_flag) {
    return;
  }

  for (uint32_t s = 0; s + 1 < N; ++s) {

    // recv_chunk_id_ag(s) = (rank - s + N) % N
    uint32_t chunk_ag = (r_u32 + N - s) % N;

    // Wait on peer_ag_ready[left][flag_index(s, tile_linear)] == epoch.
    uint64_t wait_idx = flag_index_u64(s, tile_linear, p.num_tiles_total);
    RingAllreduceSystemAtomicU32 const* peer_flag = safe_ptr_add(
        p.peer_ag_ready[left],
        wait_idx,
        flags_len);

    bool got_peer = detail::ring_allreduce_wait_flag(
        peer_flag,
        p.epoch,
        p.self_abort,
        p.self_error,
        p.peer_abort[left],
        cfg);

    if (!got_peer) {
      break;
    }

    if (p.debug_jitter_max_iters != 0u &&
        (p.debug_jitter_mask & (1u << uint32_t(detail::RingAllreduceJitterSite::kAgAfterWait))) != 0u) {
      detail::ring_allreduce_maybe_jitter(
          p.debug_jitter_seed,
          p.debug_jitter_max_iters,
          p.debug_jitter_mask,
          detail::RingAllreduceJitterSite::kAgAfterWait,
          s,
          tile_linear,
          r_u32,
          p.epoch);
      // IMPORTANT: This barrier must remain CTA-uniform (do not move under
      // per-thread control flow), otherwise it can deadlock.
      __syncthreads();
    }

    auto ag_tile = detail::ring_allreduce_compute_tile_range(
        p,
        channel_id,
        tile_in_chunk,
        chunk_ag);

    T* peer_left = p.peer_data[left];
    if (!peer_left) {
      ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
      break;
    }

    for (uint32_t e = static_cast<uint32_t>(threadIdx.x); e < ag_tile.len; e += static_cast<uint32_t>(blockDim.x)) {
      uint64_t idx = ag_tile.base + uint64_t(e);
      p.self_data[idx] = peer_left[idx];
    }

    if (p.debug_jitter_max_iters != 0u &&
        (p.debug_jitter_mask & (1u << uint32_t(detail::RingAllreduceJitterSite::kAgBeforePublish))) != 0u) {
      detail::ring_allreduce_maybe_jitter(
          p.debug_jitter_seed,
          p.debug_jitter_max_iters,
          p.debug_jitter_mask,
          detail::RingAllreduceJitterSite::kAgBeforePublish,
          s,
          tile_linear,
          r_u32,
          p.epoch);
    }

    // Publish self_ag_ready[flag_index(s+1, tile_linear)] = epoch best effort.
    //
    // This publish is also the injection point for the spec's debug abort hooks.
    __syncthreads();
    __threadfence_system();
    __syncthreads();

    bool debug_abort_match = (static_cast<uint32_t>(p.rank) == p.debug_abort_rank) &&
        (p.debug_abort_ag_step == s) &&
        ring_allreduce_is_cta0();

    uint64_t ag_fwd_idx = flag_index_u64(s + 1, tile_linear, p.num_tiles_total);
    RingAllreduceSystemAtomicU32* self_ag_fwd_flag = safe_ptr_add(p.self_ag_ready, ag_fwd_idx, flags_len);

    if (ring_allreduce_is_thread0()) {
      if (!self_ag_fwd_flag) {
        ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
      }
      else if (debug_abort_match && (p.debug_abort_before_ag_publish != 0u)) {
        // Simulate a failure before the readiness flag is published.
        ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kAbortObserved);
      }
      else {
        self_ag_fwd_flag->store(p.epoch, cuda::memory_order_release);

        if (debug_abort_match && (p.debug_abort_after_ag_publish != 0u)) {
          ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kAbortObserved);
        }
      }
    }

    if (!self_ag_fwd_flag) {
      break;
    }
  }

}

template <typename T>
CUTLASS_DEVICE
void ring_allreduce_sm100_tile_legacy(
    RingAllreduceParams<T, 8> const& p,
    RingAllreduceDrainConfig cfg,
    uint32_t tile_linear,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len) {

  bool rs_ok = ring_allreduce_sm100_tile_legacy_rs(
      p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len);

  // rs_ok must be CTA-uniform: ring_allreduce_sm100_tile_legacy_ag() begins with __syncthreads().
  if (rs_ok) {
    ring_allreduce_sm100_tile_legacy_ag(
        p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len);
  }
}

#if CUTLASS_RING_ALLREDUCE_ENABLE_WARP_SPECIALIZED_SMEM

// NamedBarrier choreography constants for the warp-specialized SMEM RS ping-pong path.
// Barrier IDs are user IDs (offset by cutlass::arch::NamedBarrier::ReservedNamedBarrierCount).
//
// NOTE: NamedBarrier's ID constants are compile-time available even when
// CUDA_BARRIER_ENABLED==0; only the barrier methods trap in device code when
// disabled.
struct RingAllreduceWarpSpecializedSmemNamedBarrierConstants {
  static constexpr uint32_t kStageThreads = 256;
  static constexpr uint32_t kPublishThreads = 160;
  static constexpr uint32_t kStepBarrierThreads = 256;

  static constexpr uint32_t kStageId0 = 0;
  static constexpr uint32_t kStageId1 = 1;
  static constexpr uint32_t kPublishId0 = 2;
  static constexpr uint32_t kPublishId1 = 3;
  static constexpr uint32_t kStepBarrierId = 4;
};

static_assert(RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStageThreads % 32 == 0,
              "Stage barrier threads must be warp-aligned.");
static_assert(RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kPublishThreads % 32 == 0,
              "Publish barrier threads must be warp-aligned.");
static_assert(RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStepBarrierThreads % 32 == 0,
              "Step barrier threads must be warp-aligned.");

static_assert(
    RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStepBarrierId + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount <
        cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers,
    "NamedBarrier IDs must fit within hardware limits.");

#if CUTLASS_RING_ALLREDUCE_TEST_WARP_SPECIALIZED_SMEM_RS_IMPL_TAG
// Test-only tag indicating which RS implementation was selected.
static constexpr uint32_t kRingAllreduceWarpSpecializedSmemRsImplTagPingPongNoOverlap = 1u;
static constexpr uint32_t kRingAllreduceWarpSpecializedSmemRsImplTagPingPongPrefetch = 2u;
static constexpr uint32_t kRingAllreduceWarpSpecializedSmemRsImplTagSingleBuffer = 3u;

extern __device__ uint32_t ring_allreduce_warp_specialized_smem_rs_impl_tag;
#endif

template <typename T>
CUTLASS_DEVICE
void ring_allreduce_sm100_tile_warp_specialized_smem_n1_publisher_only(
    RingAllreduceParams<T, 8> const& p,
    uint32_t tile_linear,
    uint64_t flags_len) {

#if defined(__CUDA_ARCH__)

  uint32_t warp_id = threadIdx.x >> 5;
  uint32_t lane = threadIdx.x & 0x1Fu;

  uint64_t ag_ready_idx = flag_index_u64(/*step=*/0, tile_linear, p.num_tiles_total);
  RingAllreduceSystemAtomicU32* self_ag_flag = safe_ptr_add(p.self_ag_ready, ag_ready_idx, flags_len);

  if (!self_ag_flag) {
    ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
  }

  // Preserve the baseline publish ordering (fence before ready flag store).
  __syncthreads();
  __threadfence_system();
  __syncthreads();

  // Role plan: warp 6 lane0 is the readiness-flag publisher.
  static constexpr uint32_t kPublishWarpId = 6;
  static constexpr uint32_t kPublishLane = 0u;

  if (self_ag_flag && warp_id == kPublishWarpId && lane == kPublishLane) {
    self_ag_flag->store(p.epoch, cuda::memory_order_release);
  }

#else

  CUTLASS_UNUSED(p);
  CUTLASS_UNUSED(tile_linear);
  CUTLASS_UNUSED(flags_len);

#endif
}

template <typename T>
struct RingAllreduceWarpSpecializedSmem {
  // Unified RS/AG stage buffers (ping-pong). Single-buffer fallback uses only buf==0.
  // Footprint (kStageElemsMax==256): float ~6.5–7 KiB, double ~12.5–13 KiB.
  T stage_peer[2][detail::kStageElemsMax];
  T stage_local[2][detail::kStageElemsMax];
  T stage_out[2][detail::kStageElemsMax];

  // Per-step metadata written by warp0 lane0.
  uint32_t step_participate[2];
  uint32_t in_len[2];

  alignas(8) uint64_t in_base[2];
  alignas(8) RingAllreduceSystemAtomicU32* out_rs_flag[2];

  // CTA-uniform RS status written by warp0 lane0.
  uint32_t rs_ok;

  // CTA-uniform AG step0 pointer broadcast written by warp0 lane0.
  alignas(8) RingAllreduceSystemAtomicU32* step0_ag_flag;
  uint32_t step0_ok;

  // Per-step AG readiness publish pointers written by warp0 lane0.
  alignas(8) RingAllreduceSystemAtomicU32* out_ag_flag[2];
};

// Keep shared storage comfortably below the per-CTA budget.
static_assert(sizeof(RingAllreduceWarpSpecializedSmem<double>) <= 32 * 1024, "Shared storage must remain within 32KiB.");

template <typename T>
CUTLASS_DEVICE
bool ring_allreduce_sm100_tile_warp_specialized_smem_rs_single_buffer(
    RingAllreduceParams<T, 8> const& p,
    RingAllreduceDrainConfig cfg,
    uint32_t tile_linear,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len,
    RingAllreduceWarpSpecializedSmem<T>& smem) {

#if defined(__CUDA_ARCH__)

  // Precondition: this helper is called only on the warp-specialized SMEM path:
  // blockDim==(256,1,1), N>1, and p.tile_elems<=kStageElemsMax.
  uint32_t warp_id = threadIdx.x >> 5;
  uint32_t lane = threadIdx.x & 0x1Fu;

  // Single-buffer fallback uses only buf==0.
  static constexpr uint32_t buf = 0u;

  if (warp_id == 0 && lane == 0) {
    smem.rs_ok = 1u;
  }
  __syncthreads();

#if CUTLASS_RING_ALLREDUCE_TEST_WARP_SPECIALIZED_SMEM_RS_IMPL_TAG
  // Stable tag for selection tests (race-free: only CTA0 thread0 writes).
  if (ring_allreduce_is_cta0() && ring_allreduce_is_thread0()) {
    ring_allreduce_warp_specialized_smem_rs_impl_tag = kRingAllreduceWarpSpecializedSmemRsImplTagSingleBuffer;
  }
#endif

  for (uint32_t s = 0; s + 1 < N; ++s) {

    // ----- Phase A: compute tile + validate pointers + form publish ptr (warp0 lane0) -----
    if (warp_id == 0) {
      if (lane == 0) {
        smem.step_participate[buf] = 1u;

        // recv_chunk_id_rs(s) = (rank - s - 1 + N) % N
        uint32_t chunk_rs = (r_u32 + N - s - 1) % N;
        auto rs_tile = detail::ring_allreduce_compute_tile_range(p, channel_id, tile_in_chunk, chunk_rs);

        smem.in_base[buf] = rs_tile.base;
        smem.in_len[buf] = rs_tile.len;

        // Defense-in-depth: should be implied by smem_eligible.
        if (smem.in_len[buf] > detail::kStageElemsMax) {
          ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
          smem.step_participate[buf] = 0u;
          smem.rs_ok = 0u;
        }

        // Form RS publish pointer for step s+1 using safe_ptr_add.
        uint64_t rs_ready_idx = flag_index_u64(s + 1, tile_linear, p.num_tiles_total);
        smem.out_rs_flag[buf] = safe_ptr_add(p.self_rs_ready, rs_ready_idx, flags_len);

        if (!smem.out_rs_flag[buf]) {
          ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
          smem.step_participate[buf] = 0u;
          smem.rs_ok = 0u;
        }
      }

      // Ensure warp0 observes lane0's metadata updates (independent thread scheduling).
      __syncwarp();

      // ----- Phase B: wait (warp0 only; s>0) -----
      bool got_peer = true;
      if (s > 0) {
        // For N==2, s>0 never executes; validator allows peer_rs_ready[left]==nullptr.
        uint64_t wait_idx = flag_index_u64(s, tile_linear, p.num_tiles_total);
        RingAllreduceSystemAtomicU32 const* peer_flag = safe_ptr_add(p.peer_rs_ready[left], wait_idx, flags_len);

        got_peer = detail::wait_flag_warp(
            peer_flag,
            p.epoch,
            p.self_abort,
            p.self_error,
            p.peer_abort[left],
            cfg);

        if (!got_peer && lane == 0) {
          // wait_flag_warp already published timeout/abort/invalid if needed (thread0-gated),
          // but we must communicate failure to the CTA.
          smem.step_participate[buf] = 0u;
          smem.rs_ok = 0u;
        }
      }

      // Ensure warp0 observes any lane0 failure updates before staging.
      __syncwarp();

      // ----- Phase C: stage peer (warp0) -----
      // Warp0 must not read peer payload unless wait succeeded.
      uint32_t ok = __shfl_sync(0xFFFF'FFFFu, got_peer ? 1u : 0u, 0);

      // (Optional safety) if peer pointer invalid, publish error via thread0.
      if (ok && lane == 0 && !p.peer_data[left]) {
        ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
        smem.step_participate[buf] = 0u;
        smem.rs_ok = 0u;
        ok = 0u;
      }

      ok = __shfl_sync(0xFFFF'FFFFu, ok, 0);

      if (ok && smem.step_participate[buf] != 0u) {
        T const* peer_left = p.peer_data[left];
        uint32_t in_len = smem.in_len[buf];
        uint64_t in_base = smem.in_base[buf];
        for (uint32_t e = lane; e < in_len; e += 32) {
          smem.stage_peer[buf][e] = peer_left[in_base + uint64_t(e)];
        }
      }
    }

    // ----- Phase C: stage local (warp1) -----
    if (warp_id == 1) {
      // IMPORTANT: recompute rs_tile locally rather than reading smem.in_base/in_len here,
      // to avoid cross-warp ordering races before StageBarrier.
      uint32_t chunk_rs = (r_u32 + N - s - 1) % N;
      auto rs_tile = detail::ring_allreduce_compute_tile_range(p, channel_id, tile_in_chunk, chunk_rs);
      uint32_t safe_len = rs_tile.len;
      if (safe_len > detail::kStageElemsMax) {
        safe_len = detail::kStageElemsMax;
      }
      for (uint32_t e = lane; e < safe_len; e += 32) {
        smem.stage_local[buf][e] = p.self_data[rs_tile.base + uint64_t(e)];
      }
    }

    __syncthreads(); // StageBarrier

    if (smem.step_participate[buf] == 0u) {
      break; // CTA-uniform
    }

    // ----- Phase D: compute (warps2–5) -----
    if (warp_id >= 2 && warp_id <= 5) {
      uint32_t tid = (warp_id - 2) * 32 + lane; // 0..127
      uint32_t in_len = smem.in_len[buf];
      for (uint32_t e = tid; e < in_len; e += 128) {
        smem.stage_out[buf][e] = smem.stage_local[buf][e] + smem.stage_peer[buf][e];
      }
    }

    __syncthreads(); // ComputeBarrier

    // ----- Phase E: store + publish (warp6 only) -----
    if (warp_id == 6) {
      uint32_t in_len = smem.in_len[buf];
      uint64_t in_base = smem.in_base[buf];

      for (uint32_t e = lane; e < in_len; e += 32) {
        p.self_data[in_base + uint64_t(e)] = smem.stage_out[buf][e];
      }

      // Publish ordering: payload stores (warp6) → fence (all warp6 lanes) → ready flag store(release) (lane0).
      __syncwarp();
      __threadfence_system();
      __syncwarp();

      if (lane == 0) {
        smem.out_rs_flag[buf]->store(p.epoch, cuda::memory_order_release);
      }
    }

    __syncthreads(); // PublishBarrier
  }

  return (smem.rs_ok != 0u);

#else

  CUTLASS_UNUSED(p);
  CUTLASS_UNUSED(cfg);
  CUTLASS_UNUSED(tile_linear);
  CUTLASS_UNUSED(channel_id);
  CUTLASS_UNUSED(tile_in_chunk);
  CUTLASS_UNUSED(N);
  CUTLASS_UNUSED(r_u32);
  CUTLASS_UNUSED(left);
  CUTLASS_UNUSED(flags_len);
  CUTLASS_UNUSED(smem);
  return false;

#endif
}

// Selector for warp-specialized RS implementation.
//
// IMPORTANT: cutlass::arch::NamedBarrier traps (brkpt) when CUDA_BARRIER_ENABLED==0.
// Any RS implementation that calls NamedBarrier must be compiled out when
// NamedBarriers are unavailable or forcibly disabled (test knob).
#if CUDA_BARRIER_ENABLED && !CUTLASS_RING_ALLREDUCE_TEST_FORCE_NAMED_BARRIER_DISABLED

template <typename T>
CUTLASS_DEVICE
void ring_allreduce_sm100_tile_warp_specialized_smem_rs_ping_pong_stage_step(
    RingAllreduceParams<T, 8> const& p,
    RingAllreduceDrainConfig cfg,
    uint32_t tile_linear,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len,
    uint32_t step,
    uint32_t buf,
    RingAllreduceWarpSpecializedSmem<T>& smem,
    uint32_t warp_id,
    uint32_t lane) {

#if defined(__CUDA_ARCH__)

  // ----- Phase A: compute tile + validate pointers + form publish ptr (warp0 lane0) -----
  if (warp_id == 0) {
    if (lane == 0) {
      smem.step_participate[buf] = 1u;

      // recv_chunk_id_rs(step) = (rank - step - 1 + N) % N
      uint32_t chunk_rs = (r_u32 + N - step - 1) % N;
      auto rs_tile = detail::ring_allreduce_compute_tile_range(p, channel_id, tile_in_chunk, chunk_rs);

      smem.in_base[buf] = rs_tile.base;
      smem.in_len[buf] = rs_tile.len;

      // Defense-in-depth: should be implied by smem_eligible.
      if (smem.in_len[buf] > detail::kStageElemsMax) {
        ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
        smem.step_participate[buf] = 0u;
        smem.rs_ok = 0u;
      }

      // Form RS publish pointer for step step+1 using safe_ptr_add.
      uint64_t rs_ready_idx = flag_index_u64(step + 1, tile_linear, p.num_tiles_total);
      smem.out_rs_flag[buf] = safe_ptr_add(p.self_rs_ready, rs_ready_idx, flags_len);

      if (!smem.out_rs_flag[buf]) {
        ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
        smem.step_participate[buf] = 0u;
        smem.rs_ok = 0u;
      }
    }

    // Ensure warp0 observes lane0's metadata updates (independent thread scheduling).
    __syncwarp();

    // ----- Phase B: wait (warp0 only; step>0) -----
    bool got_peer = true;
    if (step > 0) {
      // For N==2, step>0 never executes; validator allows peer_rs_ready[left]==nullptr.
      uint64_t wait_idx = flag_index_u64(step, tile_linear, p.num_tiles_total);
      RingAllreduceSystemAtomicU32 const* peer_flag = safe_ptr_add(p.peer_rs_ready[left], wait_idx, flags_len);

      got_peer = detail::wait_flag_warp(
          peer_flag,
          p.epoch,
          p.self_abort,
          p.self_error,
          p.peer_abort[left],
          cfg);

      if (!got_peer && lane == 0) {
        // wait_flag_warp already published timeout/abort/invalid if needed (thread0-gated),
        // but we must communicate failure to the CTA.
        smem.step_participate[buf] = 0u;
        smem.rs_ok = 0u;
      }
    }

    // Ensure warp0 observes any lane0 failure updates before staging.
    __syncwarp();

    // ----- Phase C: stage peer (warp0) -----
    // Warp0 must not read peer payload unless wait succeeded.
    uint32_t ok = __shfl_sync(0xFFFF'FFFFu, got_peer ? 1u : 0u, 0);

    // (Optional safety) if peer pointer invalid, publish error via thread0.
    if (ok && lane == 0 && !p.peer_data[left]) {
      ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
      smem.step_participate[buf] = 0u;
      smem.rs_ok = 0u;
      ok = 0u;
    }

    ok = __shfl_sync(0xFFFF'FFFFu, ok, 0);

    if (ok && smem.step_participate[buf] != 0u) {
      T const* peer_left = p.peer_data[left];
      uint32_t in_len = smem.in_len[buf];
      uint64_t in_base = smem.in_base[buf];
      for (uint32_t e = lane; e < in_len; e += 32) {
        smem.stage_peer[buf][e] = peer_left[in_base + uint64_t(e)];
      }
    }
  }

  // ----- Phase C: stage local (warp1) -----
  if (warp_id == 1) {
    // IMPORTANT: recompute rs_tile locally rather than reading smem.in_base/in_len here,
    // to avoid cross-warp ordering races before StageBarrier.
    uint32_t chunk_rs = (r_u32 + N - step - 1) % N;
    auto rs_tile = detail::ring_allreduce_compute_tile_range(p, channel_id, tile_in_chunk, chunk_rs);
    uint32_t safe_len = rs_tile.len;
    if (safe_len > detail::kStageElemsMax) {
      safe_len = detail::kStageElemsMax;
    }
    for (uint32_t e = lane; e < safe_len; e += 32) {
      smem.stage_local[buf][e] = p.self_data[rs_tile.base + uint64_t(e)];
    }
  }

#else

  CUTLASS_UNUSED(p);
  CUTLASS_UNUSED(cfg);
  CUTLASS_UNUSED(tile_linear);
  CUTLASS_UNUSED(channel_id);
  CUTLASS_UNUSED(tile_in_chunk);
  CUTLASS_UNUSED(N);
  CUTLASS_UNUSED(r_u32);
  CUTLASS_UNUSED(left);
  CUTLASS_UNUSED(flags_len);
  CUTLASS_UNUSED(step);
  CUTLASS_UNUSED(buf);
  CUTLASS_UNUSED(smem);
  CUTLASS_UNUSED(warp_id);
  CUTLASS_UNUSED(lane);

#endif
}

template <typename T>
CUTLASS_DEVICE
bool ring_allreduce_sm100_tile_warp_specialized_smem_rs_ping_pong(
    RingAllreduceParams<T, 8> const& p,
    RingAllreduceDrainConfig cfg,
    uint32_t tile_linear,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len,
    RingAllreduceWarpSpecializedSmem<T>& smem) {

#if defined(__CUDA_ARCH__)

  // Defense-in-depth: mirror wrapper preconditions so accidental direct calls
  // cannot deadlock on NamedBarrier.
  if ((blockDim.x != RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStageThreads) || (blockDim.y != 1) ||
      (blockDim.z != 1) || (N <= 1)) {
    ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
    return false; // CTA-uniform
  }

  uint32_t warp_id = threadIdx.x >> 5;
  uint32_t lane = threadIdx.x & 0x1Fu;

  if (warp_id == 0 && lane == 0) {
    smem.rs_ok = 1u;
  }
  __syncthreads();

  // Overlap policy: wait_flag_warp measures timeout budget from its call site.
  // To preserve baseline timeout behavior, only prefetch when timeouts are disabled.
  bool timeouts_enabled = (cfg.timeout_iters != 0) || (cfg.timeout_cycles != 0);
  bool allow_prefetch = !timeouts_enabled;

#if CUTLASS_RING_ALLREDUCE_TEST_WARP_SPECIALIZED_SMEM_RS_IMPL_TAG
  if (ring_allreduce_is_cta0() && ring_allreduce_is_thread0()) {
    ring_allreduce_warp_specialized_smem_rs_impl_tag = allow_prefetch ? kRingAllreduceWarpSpecializedSmemRsImplTagPingPongPrefetch
                                                    : kRingAllreduceWarpSpecializedSmemRsImplTagPingPongNoOverlap;
  }
#endif

  // No-overlap schedule: stage each step just-in-time.
  if (!allow_prefetch) {
    for (uint32_t s = 0; s + 1 < N; ++s) {
      uint32_t buf = s & 1u;
      uint32_t stage_id = buf ? RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStageId1
                              : RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStageId0;
      uint32_t publish_id = buf ? RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kPublishId1
                                : RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kPublishId0;

      ring_allreduce_sm100_tile_warp_specialized_smem_rs_ping_pong_stage_step(
          p,
          cfg,
          tile_linear,
          channel_id,
          tile_in_chunk,
          N,
          r_u32,
          left,
          flags_len,
          /*step=*/s,
          buf,
          smem,
          warp_id,
          lane);

      cutlass::arch::NamedBarrier::arrive_and_wait(RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStageThreads, stage_id);

      uint32_t rs_participate = smem.step_participate[buf];
      if (rs_participate == 0u) {
        // CTA-uniform early exit before PublishBarrier.
        // Safe to skip StepBarrier since we return immediately.
        break;
      }

      // ----- Phase D: compute (warps2–5) -----
      if (warp_id >= 2 && warp_id <= 5) {
        uint32_t tid = (warp_id - 2) * 32 + lane; // 0..127
        uint32_t in_len = smem.in_len[buf];
        for (uint32_t e = tid; e < in_len; e += 128) {
          smem.stage_out[buf][e] = smem.stage_local[buf][e] + smem.stage_peer[buf][e];
        }
      }

      // ----- Phase E: PublishBarrier (warps2–6) -----
      // Split-phase: compute warps arrive, store warp arrives+waits.
      if (warp_id >= 2 && warp_id <= 5) {
        cutlass::arch::NamedBarrier::arrive(RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kPublishThreads, publish_id);
      }
      else if (warp_id == 6) {
        cutlass::arch::NamedBarrier::arrive_and_wait(RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kPublishThreads, publish_id);
      }

      // ----- Phase F: store + publish (warp6 only) -----
      if (warp_id == 6) {
        uint32_t in_len = smem.in_len[buf];
        uint64_t in_base = smem.in_base[buf];

        for (uint32_t e = lane; e < in_len; e += 32) {
          p.self_data[in_base + uint64_t(e)] = smem.stage_out[buf][e];
        }

        // Publish ordering: payload stores (warp6) → fence (all warp6 lanes) → ready flag store(release) (lane0).
        __syncwarp();
        __threadfence_system();
        __syncwarp();

        if (lane == 0) {
          smem.out_rs_flag[buf]->store(p.epoch, cuda::memory_order_release);
        }
      }

      // Step serialization barrier: all 256 threads participate when rs_participate==1.
      cutlass::arch::NamedBarrier::arrive_and_wait(
          RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStepBarrierThreads,
          RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStepBarrierId);
    }

    return (smem.rs_ok != 0u);
  }

  // Prefetch schedule: stage step0 up front, then overlap stage_step(s+1) with compute/store of step s.
  ring_allreduce_sm100_tile_warp_specialized_smem_rs_ping_pong_stage_step(
      p,
      cfg,
      tile_linear,
      channel_id,
      tile_in_chunk,
      N,
      r_u32,
      left,
      flags_len,
      /*step=*/0,
      /*buf=*/0u,
      smem,
      warp_id,
      lane);

  for (uint32_t s = 0; s + 1 < N; ++s) {
    uint32_t buf = s & 1u;
    uint32_t stage_id = buf ? RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStageId1
                            : RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStageId0;
    uint32_t publish_id = buf ? RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kPublishId1
                              : RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kPublishId0;

    cutlass::arch::NamedBarrier::arrive_and_wait(RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStageThreads, stage_id);

    uint32_t rs_participate = smem.step_participate[buf];
    if (rs_participate == 0u) {
      // CTA-uniform early exit before PublishBarrier.
      // Safe to skip StepBarrier since we return immediately.
      break;
    }

    // Stage the next step's payload/metadata into the alternate buffer while compute runs.
    if ((s + 2) < N) {
      ring_allreduce_sm100_tile_warp_specialized_smem_rs_ping_pong_stage_step(
          p,
          cfg,
          tile_linear,
          channel_id,
          tile_in_chunk,
          N,
          r_u32,
          left,
          flags_len,
          /*step=*/s + 1,
          /*buf=*/buf ^ 1u,
          smem,
          warp_id,
          lane);
    }

    // ----- Phase D: compute (warps2–5) -----
    if (warp_id >= 2 && warp_id <= 5) {
      uint32_t tid = (warp_id - 2) * 32 + lane; // 0..127
      uint32_t in_len = smem.in_len[buf];
      for (uint32_t e = tid; e < in_len; e += 128) {
        smem.stage_out[buf][e] = smem.stage_local[buf][e] + smem.stage_peer[buf][e];
      }
    }

    // ----- Phase E: PublishBarrier (warps2–6) -----
    // Split-phase: compute warps arrive, store warp arrives+waits.
    if (warp_id >= 2 && warp_id <= 5) {
      cutlass::arch::NamedBarrier::arrive(RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kPublishThreads, publish_id);
    }
    else if (warp_id == 6) {
      cutlass::arch::NamedBarrier::arrive_and_wait(RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kPublishThreads, publish_id);
    }

    // ----- Phase F: store + publish (warp6 only) -----
    if (warp_id == 6) {
      uint32_t in_len = smem.in_len[buf];
      uint64_t in_base = smem.in_base[buf];

      for (uint32_t e = lane; e < in_len; e += 32) {
        p.self_data[in_base + uint64_t(e)] = smem.stage_out[buf][e];
      }

      // Publish ordering: payload stores (warp6) → fence (all warp6 lanes) → ready flag store(release) (lane0).
      __syncwarp();
      __threadfence_system();
      __syncwarp();

      if (lane == 0) {
        smem.out_rs_flag[buf]->store(p.epoch, cuda::memory_order_release);
      }
    }

    // Step serialization barrier: all 256 threads participate when rs_participate==1.
    cutlass::arch::NamedBarrier::arrive_and_wait(
        RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStepBarrierThreads,
        RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStepBarrierId);
  }

  return (smem.rs_ok != 0u);

#else

  CUTLASS_UNUSED(p);
  CUTLASS_UNUSED(cfg);
  CUTLASS_UNUSED(tile_linear);
  CUTLASS_UNUSED(channel_id);
  CUTLASS_UNUSED(tile_in_chunk);
  CUTLASS_UNUSED(N);
  CUTLASS_UNUSED(r_u32);
  CUTLASS_UNUSED(left);
  CUTLASS_UNUSED(flags_len);
  CUTLASS_UNUSED(smem);
  return false;

#endif
}

#endif

template <typename T>
CUTLASS_DEVICE
bool ring_allreduce_sm100_tile_warp_specialized_smem_rs_impl(
    RingAllreduceParams<T, 8> const& p,
    RingAllreduceDrainConfig cfg,
    uint32_t tile_linear,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len,
    RingAllreduceWarpSpecializedSmem<T>& smem) {

#if defined(__CUDA_ARCH__)
  // Wrapper-level runtime preconditions:
  // - blockDim must be 256 threads (8 warps), matching expected warp roles and NamedBarrier contracts.
  // - N must be > 1 (N==1 uses a separate publisher-only path at the kernel level).
  if ((blockDim.x != 256) || (blockDim.y != 1) || (blockDim.z != 1) || (N <= 1)) {
    ring_allreduce_publish_error_and_abort(
        p.self_error,
        p.self_abort,
        RingAllreduceError::kInvalidParams);
    return false; // CTA-uniform
  }
#endif

#if CUDA_BARRIER_ENABLED && !CUTLASS_RING_ALLREDUCE_TEST_FORCE_NAMED_BARRIER_DISABLED
  return ring_allreduce_sm100_tile_warp_specialized_smem_rs_ping_pong(
      p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len, smem);
#else
  return ring_allreduce_sm100_tile_warp_specialized_smem_rs_single_buffer(
      p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len, smem);
#endif
}

template <typename T>
CUTLASS_DEVICE
bool ring_allreduce_sm100_tile_warp_specialized_smem_rs(
    RingAllreduceParams<T, 8> const& p,
    RingAllreduceDrainConfig cfg,
    uint32_t tile_linear,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len) {

#if defined(__CUDA_ARCH__)

  // NOTE: This RS-only wrapper allocates its own shared storage for unit tests.
  // The full warp-specialized tile wrapper (ring_allreduce_sm100_tile_warp_specialized_smem) allocates shared
  // storage once so RS and AG can reuse it.
  __shared__ RingAllreduceWarpSpecializedSmem<T> smem;
  return ring_allreduce_sm100_tile_warp_specialized_smem_rs_impl(
      p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len, smem);

#else

  CUTLASS_UNUSED(p);
  CUTLASS_UNUSED(cfg);
  CUTLASS_UNUSED(tile_linear);
  CUTLASS_UNUSED(channel_id);
  CUTLASS_UNUSED(tile_in_chunk);
  CUTLASS_UNUSED(N);
  CUTLASS_UNUSED(r_u32);
  CUTLASS_UNUSED(left);
  CUTLASS_UNUSED(flags_len);
  return false;

#endif
}

static CUTLASS_DEVICE
void ring_allreduce_warp_specialized_smem_ag_publish_flag_warp6(
    RingAllreduceSystemAtomicU32* flag,
    uint32_t epoch,
    uint32_t lane) {

#if defined(__CUDA_ARCH__)

  // Must be called by a full warp with warp-uniform control flow.
  // Publish ordering: payload stores (warp6) -> fence (all warp6 lanes) -> ready flag store(release) (lane0).
  __syncwarp();
  __threadfence_system();
  __syncwarp();

  if (lane == 0u && flag) {
    flag->store(epoch, cuda::memory_order_release);
  }

#else

  CUTLASS_UNUSED(flag);
  CUTLASS_UNUSED(epoch);
  CUTLASS_UNUSED(lane);

#endif
}

template <typename T>
CUTLASS_DEVICE
void ring_allreduce_sm100_tile_warp_specialized_smem_ag_single_buffer(
    RingAllreduceParams<T, 8> const& p,
    RingAllreduceDrainConfig cfg,
    uint32_t tile_linear,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len,
    RingAllreduceWarpSpecializedSmem<T>& smem) {

#if defined(__CUDA_ARCH__)

  // AG runtime preconditions (CTA-uniform; must run before any barrier).
  // All arguments must be CTA-uniform (only call from ring_allreduce_sm100_tile_warp_specialized_smem()).
  bool N_ok = (N >= 2u) && (N <= 8u);
  bool left_ok = N_ok && (left >= 0) && (left < static_cast<int32_t>(N));

  bool ok = (blockDim.x == 256) && (blockDim.y == 1) && (blockDim.z == 1) &&
      N_ok &&
      left_ok &&
      (p.tile_elems != 0u) &&
      (p.tile_elems <= detail::kStageElemsMax) &&
      (p.num_tiles_total != 0u) &&
      (p.epoch != 0u) &&
      (p.self_data != nullptr) &&
      (p.self_ag_ready != nullptr) &&
      (p.self_abort != nullptr) &&
      (p.self_error != nullptr);

  if (left_ok) {
    ok = ok &&
        (p.peer_data[left] != nullptr) &&
        (p.peer_ag_ready[left] != nullptr) &&
        (p.peer_abort[left] != nullptr);
  }

  if (!ok) {
    ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
    return; // CTA-uniform
  }

  uint32_t warp_id = threadIdx.x >> 5;
  uint32_t lane = threadIdx.x & 0x1Fu;

  // Step0 publish protocol (single-buffer fallback):
  // 1) thread0 computes pointer + ok bit into shared
  // 2) CTA-wide __syncthreads() broadcast
  // 3) if !ok: return CTA-uniformly
  // 4) warp6 publishes epoch with fence ordering
  // 5) CTA-wide __syncthreads()
  if (warp_id == 0 && lane == 0) {
    uint64_t ag_ready_idx = flag_index_u64(/*step=*/0, tile_linear, p.num_tiles_total);
    smem.step0_ag_flag = safe_ptr_add(p.self_ag_ready, ag_ready_idx, flags_len);
    smem.step0_ok = smem.step0_ag_flag ? 1u : 0u;

    if (smem.step0_ok == 0u) {
      ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
    }
  }

  __syncthreads();

  if (smem.step0_ok == 0u) {
    return; // CTA-uniform
  }

  // Role plan: warp 6 lane0 is the readiness-flag publisher.
  static constexpr uint32_t kPublishWarpId = 6;
  static_assert(kPublishWarpId == 6,
                "Invariant: publish warp must be warp6 (matches abort allowlist guard)");
  if (warp_id == kPublishWarpId) {
    ring_allreduce_warp_specialized_smem_ag_publish_flag_warp6(smem.step0_ag_flag, p.epoch, lane);
  }

  __syncthreads();

  // Single-buffer fallback uses only buf==0.
  static constexpr uint32_t buf = 0u;

  for (uint32_t s = 0; s + 1 < N; ++s) {

    // ----- Phase A: stage step (warp0) -----
    if (warp_id == 0) {

      // A0) lane0 writes per-step metadata + forwarding flag pointer.
      if (lane == 0) {
        smem.step_participate[buf] = 1u; // IMPORTANT: keep 1 even if in_len==0

        // recv_chunk_id_ag(s) = (rank - s + N) % N
        uint32_t chunk_ag = (r_u32 + N - s) % N;
        auto ag_tile = detail::ring_allreduce_compute_tile_range(p, channel_id, tile_in_chunk, chunk_ag);

        smem.in_base[buf] = ag_tile.base;
        smem.in_len[buf] = ag_tile.len;

        // Defense-in-depth: should be implied by smem_eligible.
        if (smem.in_len[buf] > detail::kStageElemsMax) {
          ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
          smem.step_participate[buf] = 0u;
        }

        // Form AG publish pointer for step s+1 using safe_ptr_add.
        uint64_t ag_ready_idx = flag_index_u64(s + 1, tile_linear, p.num_tiles_total);
        smem.out_ag_flag[buf] = safe_ptr_add(p.self_ag_ready, ag_ready_idx, flags_len);

        if (!smem.out_ag_flag[buf]) {
          ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
          smem.step_participate[buf] = 0u;
        }
      }

      // Ensure warp0 observes lane0's metadata updates (independent thread scheduling).
      __syncwarp();

      // A1) wait on peer_ag_ready[left][s] with acquire-confirm semantics.
      bool got_peer = false;
      if (smem.step_participate[buf] != 0u) {
        uint64_t wait_idx = flag_index_u64(s, tile_linear, p.num_tiles_total);
        RingAllreduceSystemAtomicU32 const* peer_flag = safe_ptr_add(p.peer_ag_ready[left], wait_idx, flags_len);

        got_peer = detail::wait_flag_warp(
            peer_flag,
            p.epoch,
            p.self_abort,
            p.self_error,
            p.peer_abort[left],
            cfg);

        if (!got_peer && lane == 0) {
          // wait_flag_warp already published timeout/abort/invalid if needed (thread0-gated),
          // but we must communicate failure to the CTA.
          smem.step_participate[buf] = 0u;
        }
      }

      // Ensure warp0 observes any lane0 failure updates before staging.
      __syncwarp();

      // A2) kAgAfterWait jitter parity (success-only, lane0 only) + immediate __syncwarp.
      if (got_peer) {
        uint32_t site_bit = 1u << uint32_t(detail::RingAllreduceJitterSite::kAgAfterWait);
        if (lane == 0 && p.debug_jitter_max_iters != 0u && (p.debug_jitter_mask & site_bit) != 0u) {
          detail::ring_allreduce_maybe_jitter(
              p.debug_jitter_seed,
              p.debug_jitter_max_iters,
              p.debug_jitter_mask,
              detail::RingAllreduceJitterSite::kAgAfterWait,
              s,
              tile_linear,
              r_u32,
              p.epoch);
        }

        // Prevent partial staging while lane0 sleeps.
        __syncwarp();
      }

      // A3) stage peer payload into SMEM (gated on wait success).
      uint32_t ok = __shfl_sync(0xFFFF'FFFFu, got_peer ? 1u : 0u, 0);

      // Defense-in-depth: validate payload pointer before deref.
      if (ok && lane == 0 && p.peer_data[left] == nullptr) {
        ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
        smem.step_participate[buf] = 0u;
        ok = 0u;
      }

      ok = __shfl_sync(0xFFFF'FFFFu, ok, 0);

      if (ok && smem.step_participate[buf] != 0u) {
        T const* peer_left = p.peer_data[left];
        uint64_t in_base = smem.in_base[buf];
        uint32_t in_len = smem.in_len[buf];
        for (uint32_t e = lane; e < in_len; e += 32) {
          smem.stage_peer[buf][e] = peer_left[in_base + uint64_t(e)];
        }
      }
    }

    __syncthreads(); // StageBarrier

    if (smem.step_participate[buf] == 0u) {
      break; // CTA-uniform
    }

    // ----- Phase B: SMEM→SMEM copy (warps2–5) -----
    if (warp_id >= 2 && warp_id <= 5) {
      uint32_t tid = (warp_id - 2) * 32 + lane; // 0..127
      uint32_t in_len = smem.in_len[buf];
      for (uint32_t e = tid; e < in_len; e += 128) {
        smem.stage_out[buf][e] = smem.stage_peer[buf][e];
      }
    }

    __syncthreads(); // PublishBarrier

    // ----- Phase C: store + forward publish (warp6 only) -----
    if (warp_id == kPublishWarpId) {
      uint64_t in_base = smem.in_base[buf];
      uint32_t in_len = smem.in_len[buf];

      // Payload stores (0-iter if in_len==0).
      for (uint32_t e = lane; e < in_len; e += 32) {
        p.self_data[in_base + uint64_t(e)] = smem.stage_out[buf][e];
      }

      // Required ordering prefix.
      __syncwarp();

      // kAgBeforePublish jitter parity: warp6 lane0.
      detail::ring_allreduce_maybe_jitter_warp_lane0(
          p.debug_jitter_seed,
          p.debug_jitter_max_iters,
          p.debug_jitter_mask,
          detail::RingAllreduceJitterSite::kAgBeforePublish,
          s,
          tile_linear,
          r_u32,
          p.epoch,
          lane);

      // Publish ordering: payload stores (warp6) → fence (all warp6 lanes) → ready flag store(release) (lane0).
      __threadfence_system();
      __syncwarp();

      // Debug abort parity at forwarding publish.
      bool debug_abort_match = (static_cast<uint32_t>(p.rank) == p.debug_abort_rank) &&
          (p.debug_abort_ag_step == s) &&
          ring_allreduce_is_cta0();

      bool abort_before = debug_abort_match && (p.debug_abort_before_ag_publish != 0u);
      bool abort_after = debug_abort_match && (p.debug_abort_after_ag_publish != 0u);

      if (lane == 0) {
        if (abort_before) {
          // abort-before: SKIP forwarding flag store.
          if (ring_allreduce_is_cta0() && warp_id == 6 && lane == 0) {
            ring_allreduce_publish_error_and_abort_any_thread(
                p.self_error,
                p.self_abort,
                RingAllreduceError::kAbortObserved);
          }
        }
        else {
          smem.out_ag_flag[buf]->store(p.epoch, cuda::memory_order_release);

          if (abort_after) {
            if (ring_allreduce_is_cta0() && warp_id == 6 && lane == 0) {
              ring_allreduce_publish_error_and_abort_any_thread(
                  p.self_error,
                  p.self_abort,
                  RingAllreduceError::kAbortObserved);
            }
          }
        }
      }
    }

    __syncthreads(); // StepBarrier
  }

#else

  CUTLASS_UNUSED(p);
  CUTLASS_UNUSED(cfg);
  CUTLASS_UNUSED(tile_linear);
  CUTLASS_UNUSED(channel_id);
  CUTLASS_UNUSED(tile_in_chunk);
  CUTLASS_UNUSED(N);
  CUTLASS_UNUSED(r_u32);
  CUTLASS_UNUSED(left);
  CUTLASS_UNUSED(flags_len);
  CUTLASS_UNUSED(smem);

#endif
}

#if CUDA_BARRIER_ENABLED && !CUTLASS_RING_ALLREDUCE_TEST_FORCE_NAMED_BARRIER_DISABLED

template <typename T>
CUTLASS_DEVICE
void ring_allreduce_sm100_tile_warp_specialized_smem_ag_ping_pong_stage_step(
    RingAllreduceParams<T, 8> const& p,
    RingAllreduceDrainConfig cfg,
    uint32_t tile_linear,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len,
    uint32_t step,
    uint32_t buf,
    RingAllreduceWarpSpecializedSmem<T>& smem,
    uint32_t warp_id,
    uint32_t lane) {

#if defined(__CUDA_ARCH__)

  // ----- Phase A: stage step (warp0) -----
  if (warp_id == 0) {

    // A0) lane0 writes per-step metadata + forwarding flag pointer.
    if (lane == 0) {
      smem.step_participate[buf] = 1u; // IMPORTANT: keep 1 even if in_len==0

      // recv_chunk_id_ag(step) = (rank - step + N) % N
      uint32_t chunk_ag = (r_u32 + N - step) % N;
      auto ag_tile = detail::ring_allreduce_compute_tile_range(p, channel_id, tile_in_chunk, chunk_ag);

      smem.in_base[buf] = ag_tile.base;
      smem.in_len[buf] = ag_tile.len;

      // Defense-in-depth: should be implied by smem_eligible.
      if (smem.in_len[buf] > detail::kStageElemsMax) {
        ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
        smem.step_participate[buf] = 0u;
      }

      // Form AG publish pointer for step step+1 using safe_ptr_add.
      uint64_t ag_ready_idx = flag_index_u64(step + 1, tile_linear, p.num_tiles_total);
      smem.out_ag_flag[buf] = safe_ptr_add(p.self_ag_ready, ag_ready_idx, flags_len);

      if (!smem.out_ag_flag[buf]) {
        ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
        smem.step_participate[buf] = 0u;
      }
    }

    // Ensure warp0 observes lane0's metadata updates (independent thread scheduling).
    __syncwarp();

    // A1) wait on peer_ag_ready[left][step] with acquire-confirm semantics.
    bool got_peer = false;
    if (smem.step_participate[buf] != 0u) {
      uint64_t wait_idx = flag_index_u64(step, tile_linear, p.num_tiles_total);
      RingAllreduceSystemAtomicU32 const* peer_flag = safe_ptr_add(p.peer_ag_ready[left], wait_idx, flags_len);

      got_peer = detail::wait_flag_warp(
          peer_flag,
          p.epoch,
          p.self_abort,
          p.self_error,
          p.peer_abort[left],
          cfg);

      if (!got_peer && lane == 0) {
        // wait_flag_warp already published timeout/abort/invalid if needed (thread0-gated),
        // but we must communicate failure to the CTA.
        smem.step_participate[buf] = 0u;
      }
    }

    // Ensure warp0 observes any lane0 failure updates before staging.
    __syncwarp();

    // A2) kAgAfterWait jitter parity (success-only, lane0 only) + immediate __syncwarp.
    if (got_peer) {
      uint32_t site_bit = 1u << uint32_t(detail::RingAllreduceJitterSite::kAgAfterWait);
      if (lane == 0 && p.debug_jitter_max_iters != 0u && (p.debug_jitter_mask & site_bit) != 0u) {
        detail::ring_allreduce_maybe_jitter(
            p.debug_jitter_seed,
            p.debug_jitter_max_iters,
            p.debug_jitter_mask,
            detail::RingAllreduceJitterSite::kAgAfterWait,
            step,
            tile_linear,
            r_u32,
            p.epoch);
      }

      // Prevent partial staging while lane0 sleeps.
      __syncwarp();
    }

    // A3) stage peer payload into SMEM (gated on wait success).
    uint32_t ok = __shfl_sync(0xFFFF'FFFFu, got_peer ? 1u : 0u, 0);

    // Defense-in-depth: validate payload pointer before deref.
    if (ok && lane == 0 && p.peer_data[left] == nullptr) {
      ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
      smem.step_participate[buf] = 0u;
      ok = 0u;
    }

    ok = __shfl_sync(0xFFFF'FFFFu, ok, 0);

    if (ok && smem.step_participate[buf] != 0u) {
      T const* peer_left = p.peer_data[left];
      uint64_t in_base = smem.in_base[buf];
      uint32_t in_len = smem.in_len[buf];
      for (uint32_t e = lane; e < in_len; e += 32) {
        smem.stage_peer[buf][e] = peer_left[in_base + uint64_t(e)];
      }
    }
  }

#else

  CUTLASS_UNUSED(p);
  CUTLASS_UNUSED(cfg);
  CUTLASS_UNUSED(tile_linear);
  CUTLASS_UNUSED(channel_id);
  CUTLASS_UNUSED(tile_in_chunk);
  CUTLASS_UNUSED(N);
  CUTLASS_UNUSED(r_u32);
  CUTLASS_UNUSED(left);
  CUTLASS_UNUSED(flags_len);
  CUTLASS_UNUSED(step);
  CUTLASS_UNUSED(buf);
  CUTLASS_UNUSED(smem);
  CUTLASS_UNUSED(warp_id);
  CUTLASS_UNUSED(lane);

#endif
}

// AG ping-pong NO-OVERLAP per-step loop.
// Precondition: step0 publish protocol + StepBarrier complete.
// Called only from ring_allreduce_sm100_tile_warp_specialized_smem_ag_ping_pong().
template <typename T>
CUTLASS_DEVICE
void ring_allreduce_sm100_tile_warp_specialized_smem_ag_ping_pong_no_overlap(
    RingAllreduceParams<T, 8> const& p,
    RingAllreduceDrainConfig cfg,
    uint32_t tile_linear,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len,
    RingAllreduceWarpSpecializedSmem<T>& smem,
    uint32_t warp_id,
    uint32_t lane) {

#if defined(__CUDA_ARCH__)

  // Role plan: warp 6 lane0 is the readiness-flag publisher.
  static constexpr uint32_t kPublishWarpId = 6;
  static_assert(kPublishWarpId == 6,
                "Invariant: publish warp must be warp6 (matches abort allowlist guard)");

  // AG ping-pong per-step invariants (do not change):
  //  1) stage_step(s, buf)
  //  2) StageBarrier: arrive_and_wait(256, stage_id)
  //  3) if (step_participate==0) break;   // CTA-uniform; BEFORE Publish/Step
  //  4) copy (warps2–5)
  //  5) PublishBarrier: warps2–5 arrive(160), warp6 arrive_and_wait(160)
  //     NOTE: barrier calls are thread-level: ALL 32 lanes of each participating warp must call.
  //  6) warp6 store + forward publish (jitter + abort parity)
  //  7) StepBarrier: arrive_and_wait(256, kStepBarrierId)
  for (uint32_t s = 0; s + 1 < N; ++s) {
    uint32_t buf = s & 1u;
    uint32_t stage_id = (buf == 0u) ? RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStageId0
                                    : RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStageId1;
    uint32_t publish_id = (buf == 0u) ? RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kPublishId0
                                      : RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kPublishId1;

    ring_allreduce_sm100_tile_warp_specialized_smem_ag_ping_pong_stage_step(
        p,
        cfg,
        tile_linear,
        channel_id,
        tile_in_chunk,
        N,
        r_u32,
        left,
        flags_len,
        /*step=*/s,
        buf,
        smem,
        warp_id,
        lane);

    cutlass::arch::NamedBarrier::arrive_and_wait(
        RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStageThreads,
        stage_id);

    uint32_t step_participate = smem.step_participate[buf];
    if (step_participate == 0u) {
      // CTA-uniform early exit before PublishBarrier / StepBarrier.
      break;
    }

    // ----- Phase D: copy (warps2–5) -----
    // NOTE: This SMEM->SMEM copy is intentional overhead (role symmetry with RS and
    // a future overlap schedule). Do not remove.
    if (warp_id >= 2 && warp_id <= 5) {
      uint32_t tid = (warp_id - 2) * 32 + lane; // 0..127
      uint32_t in_len = smem.in_len[buf];
      for (uint32_t e = tid; e < in_len; e += 128) {
        smem.stage_out[buf][e] = smem.stage_peer[buf][e];
      }
    }

    // ----- Phase E: PublishBarrier (warps2–6) -----
    // Split-phase: copy warps arrive, store warp arrives+waits.
    // PublishBarrier participant checklist (thread-counted, 5 warps = kPublishThreads):
    // - warps2–5: arrive(kPublishThreads, publish_id) (ALL lanes)
    // - warp6:    arrive_and_wait(kPublishThreads, publish_id) (ALL lanes)
    // - only warps2–6 participate
    if (warp_id >= 2 && warp_id <= 5) {
      cutlass::arch::NamedBarrier::arrive(
          RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kPublishThreads,
          publish_id);
    }
    else if (warp_id == 6) {
      cutlass::arch::NamedBarrier::arrive_and_wait(
          RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kPublishThreads,
          publish_id);
    }

    // ----- Phase F: store + forward publish (warp6 only) -----
    if (warp_id == kPublishWarpId) {
      uint64_t in_base = smem.in_base[buf];
      uint32_t in_len = smem.in_len[buf];

      // Payload stores (0-iter if in_len==0).
      for (uint32_t e = lane; e < in_len; e += 32) {
        p.self_data[in_base + uint64_t(e)] = smem.stage_out[buf][e];
      }

      // Required ordering prefix.
      __syncwarp();

      // kAgBeforePublish jitter parity: warp6 lane0.
      detail::ring_allreduce_maybe_jitter_warp_lane0(
          p.debug_jitter_seed,
          p.debug_jitter_max_iters,
          p.debug_jitter_mask,
          detail::RingAllreduceJitterSite::kAgBeforePublish,
          s,
          tile_linear,
          r_u32,
          p.epoch,
          lane);

      // Publish ordering: payload stores (warp6) → fence (all warp6 lanes) → ready flag store(release) (lane0).
      __threadfence_system();
      __syncwarp();

      // Debug abort parity at forwarding publish.
      bool debug_abort_match = (static_cast<uint32_t>(p.rank) == p.debug_abort_rank) &&
          (p.debug_abort_ag_step == s) &&
          ring_allreduce_is_cta0();

      bool abort_before = debug_abort_match && (p.debug_abort_before_ag_publish != 0u);
      bool abort_after = debug_abort_match && (p.debug_abort_after_ag_publish != 0u);

      if (lane == 0) {
        if (abort_before) {
          // abort-before: SKIP forwarding flag store.
          if (ring_allreduce_is_cta0() && warp_id == 6 && lane == 0) {
            ring_allreduce_publish_error_and_abort_any_thread(
                p.self_error,
                p.self_abort,
                RingAllreduceError::kAbortObserved);
          }
        }
        else {
          smem.out_ag_flag[buf]->store(p.epoch, cuda::memory_order_release);

          if (abort_after) {
            if (ring_allreduce_is_cta0() && warp_id == 6 && lane == 0) {
              ring_allreduce_publish_error_and_abort_any_thread(
                  p.self_error,
                  p.self_abort,
                  RingAllreduceError::kAbortObserved);
            }
          }
        }
      }
    }

    cutlass::arch::NamedBarrier::arrive_and_wait(
        RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStepBarrierThreads,
        RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStepBarrierId);
  }

#else

  CUTLASS_UNUSED(p);
  CUTLASS_UNUSED(cfg);
  CUTLASS_UNUSED(tile_linear);
  CUTLASS_UNUSED(channel_id);
  CUTLASS_UNUSED(tile_in_chunk);
  CUTLASS_UNUSED(N);
  CUTLASS_UNUSED(r_u32);
  CUTLASS_UNUSED(left);
  CUTLASS_UNUSED(flags_len);
  CUTLASS_UNUSED(smem);
  CUTLASS_UNUSED(warp_id);
  CUTLASS_UNUSED(lane);

#endif
}

template <typename T>
CUTLASS_DEVICE
void ring_allreduce_sm100_tile_warp_specialized_smem_ag_ping_pong(
    RingAllreduceParams<T, 8> const& p,
    RingAllreduceDrainConfig cfg,
    uint32_t tile_linear,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len,
    RingAllreduceWarpSpecializedSmem<T>& smem) {

#if defined(__CUDA_ARCH__)

  // AG runtime preconditions (CTA-uniform; must run before any barrier).
  // All arguments must be CTA-uniform (only call from ring_allreduce_sm100_tile_warp_specialized_smem()).
  bool N_ok = (N >= 2u) && (N <= 8u);
  bool left_ok = N_ok && (left >= 0) && (left < static_cast<int32_t>(N));

  bool ok = (blockDim.x == 256) && (blockDim.y == 1) && (blockDim.z == 1) &&
      N_ok &&
      left_ok &&
      (p.tile_elems != 0u) &&
      (p.tile_elems <= detail::kStageElemsMax) &&
      (p.num_tiles_total != 0u) &&
      (p.epoch != 0u) &&
      (p.self_data != nullptr) &&
      (p.self_ag_ready != nullptr) &&
      (p.self_abort != nullptr) &&
      (p.self_error != nullptr);

  if (left_ok) {
    ok = ok &&
        (p.peer_data[left] != nullptr) &&
        (p.peer_ag_ready[left] != nullptr) &&
        (p.peer_abort[left] != nullptr);
  }

  if (!ok) {
    ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
    return; // CTA-uniform
  }

  uint32_t warp_id = threadIdx.x >> 5;
  uint32_t lane = threadIdx.x & 0x1Fu;

  // Step0 publish protocol (NamedBarrier):
  // 1) thread0 computes pointer + ok bit into shared
  // 2) StepBarrier(256) broadcast
  // 3) if !ok: return CTA-uniformly
  // 4) warp6 publishes epoch with fence ordering
  // 5) StepBarrier(256)
  if (warp_id == 0 && lane == 0) {
    uint64_t ag_ready_idx = flag_index_u64(/*step=*/0, tile_linear, p.num_tiles_total);
    smem.step0_ag_flag = safe_ptr_add(p.self_ag_ready, ag_ready_idx, flags_len);
    smem.step0_ok = smem.step0_ag_flag ? 1u : 0u;

    if (smem.step0_ok == 0u) {
      ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
    }
  }

  cutlass::arch::NamedBarrier::arrive_and_wait(
      RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStepBarrierThreads,
      RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStepBarrierId);

  if (smem.step0_ok == 0u) {
    return; // CTA-uniform
  }

  // Role plan: warp 6 lane0 is the readiness-flag publisher.
  static constexpr uint32_t kPublishWarpId = 6;
  static_assert(kPublishWarpId == 6,
                "Invariant: publish warp must be warp6 (matches abort allowlist guard)");
  if (warp_id == kPublishWarpId) {
    ring_allreduce_warp_specialized_smem_ag_publish_flag_warp6(smem.step0_ag_flag, p.epoch, lane);
  }

  cutlass::arch::NamedBarrier::arrive_and_wait(
      RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStepBarrierThreads,
      RingAllreduceWarpSpecializedSmemNamedBarrierConstants::kStepBarrierId);

  // Overlap policy: wait_flag_warp measures timeout budget from its call site.
  // To preserve baseline timeout behavior, only prefetch when timeouts are disabled.
  bool timeouts_enabled = (cfg.timeout_iters != 0) || (cfg.timeout_cycles != 0);
  bool allow_prefetch = !timeouts_enabled;

  // No-overlap schedule: stage each step just-in-time.
  // IMPORTANT: allow_prefetch must be CTA-uniform (divergence would deadlock due to barriers).
  if (!allow_prefetch) {
    ring_allreduce_sm100_tile_warp_specialized_smem_ag_ping_pong_no_overlap(
        p,
        cfg,
        tile_linear,
        channel_id,
        tile_in_chunk,
        N,
        r_u32,
        left,
        flags_len,
        smem,
        warp_id,
        lane);
  }
  else {
    // Prefetch schedule hook: allow_prefetch==true implies timeouts are disabled.
    // Prefetch/overlap is not implemented yet for AG ping-pong; run no-overlap for now.
    ring_allreduce_sm100_tile_warp_specialized_smem_ag_ping_pong_no_overlap(
        p,
        cfg,
        tile_linear,
        channel_id,
        tile_in_chunk,
        N,
        r_u32,
        left,
        flags_len,
        smem,
        warp_id,
        lane);
  }

#else

  CUTLASS_UNUSED(p);
  CUTLASS_UNUSED(cfg);
  CUTLASS_UNUSED(tile_linear);
  CUTLASS_UNUSED(channel_id);
  CUTLASS_UNUSED(tile_in_chunk);
  CUTLASS_UNUSED(N);
  CUTLASS_UNUSED(r_u32);
  CUTLASS_UNUSED(left);
  CUTLASS_UNUSED(flags_len);
  CUTLASS_UNUSED(smem);

#endif
}

#endif

template <typename T>
CUTLASS_DEVICE
void ring_allreduce_sm100_tile_warp_specialized_smem_ag(
    RingAllreduceParams<T, 8> const& p,
    RingAllreduceDrainConfig cfg,
    uint32_t tile_linear,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len,
    RingAllreduceWarpSpecializedSmem<T>& smem) {

#if defined(__CUDA_ARCH__)

  // AG runtime preconditions (CTA-uniform; must run before any barrier).
  // All arguments must be CTA-uniform (only call from ring_allreduce_sm100_tile_warp_specialized_smem()).
  bool N_ok = (N >= 2u) && (N <= 8u);
  bool left_ok = N_ok && (left >= 0) && (left < static_cast<int32_t>(N));

  bool ok = (blockDim.x == 256) && (blockDim.y == 1) && (blockDim.z == 1) &&
      N_ok &&
      left_ok &&
      (p.tile_elems != 0u) &&
      (p.tile_elems <= detail::kStageElemsMax) &&
      (p.num_tiles_total != 0u) &&
      (p.epoch != 0u) &&
      (p.self_data != nullptr) &&
      (p.self_ag_ready != nullptr) &&
      (p.self_abort != nullptr) &&
      (p.self_error != nullptr);

  if (left_ok) {
    ok = ok &&
        (p.peer_data[left] != nullptr) &&
        (p.peer_ag_ready[left] != nullptr) &&
        (p.peer_abort[left] != nullptr);
  }

  if (!ok) {
    ring_allreduce_publish_error_and_abort(p.self_error, p.self_abort, RingAllreduceError::kInvalidParams);
    return; // CTA-uniform
  }

#if CUDA_BARRIER_ENABLED && !CUTLASS_RING_ALLREDUCE_TEST_FORCE_NAMED_BARRIER_DISABLED
  ring_allreduce_sm100_tile_warp_specialized_smem_ag_ping_pong(
      p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len, smem);
#else
  ring_allreduce_sm100_tile_warp_specialized_smem_ag_single_buffer(
      p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len, smem);
#endif

#else

  CUTLASS_UNUSED(p);
  CUTLASS_UNUSED(cfg);
  CUTLASS_UNUSED(tile_linear);
  CUTLASS_UNUSED(channel_id);
  CUTLASS_UNUSED(tile_in_chunk);
  CUTLASS_UNUSED(N);
  CUTLASS_UNUSED(r_u32);
  CUTLASS_UNUSED(left);
  CUTLASS_UNUSED(flags_len);
  CUTLASS_UNUSED(smem);

#endif
}

template <typename T>
CUTLASS_DEVICE
void ring_allreduce_sm100_tile_warp_specialized_smem(
    RingAllreduceParams<T, 8> const& p,
    RingAllreduceDrainConfig cfg,
    uint32_t tile_linear,
    uint32_t channel_id,
    uint32_t tile_in_chunk,
    uint32_t N,
    uint32_t r_u32,
    int32_t left,
    uint64_t flags_len) {

#if defined(__CUDA_ARCH__)

  // Unified wrapper-level shared storage reused across RS and AG.
  __shared__ RingAllreduceWarpSpecializedSmem<T> smem;

  bool rs_ok = ring_allreduce_sm100_tile_warp_specialized_smem_rs_impl(
      p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len, smem);

  // rs_ok must be CTA-uniform: the selected AG path begins with CTA-wide synchronization.
  if (rs_ok) {
#if CUTLASS_RING_ALLREDUCE_ENABLE_WARP_SPECIALIZED_SMEM_AG
    ring_allreduce_sm100_tile_warp_specialized_smem_ag(
        p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len, smem);
#else
    ring_allreduce_sm100_tile_legacy_ag(
        p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len);
#endif
  }

#else

  CUTLASS_UNUSED(p);
  CUTLASS_UNUSED(cfg);
  CUTLASS_UNUSED(tile_linear);
  CUTLASS_UNUSED(channel_id);
  CUTLASS_UNUSED(tile_in_chunk);
  CUTLASS_UNUSED(N);
  CUTLASS_UNUSED(r_u32);
  CUTLASS_UNUSED(left);
  CUTLASS_UNUSED(flags_len);

#endif
}

#endif

} // namespace detail

/// Correctness-first ring allreduce kernel (SM100/SM103).
///
/// Supports world_size ∈ {1,2,4,8} (kMaxWorldSize==8).
///
/// Algorithm (see ring_allreduce/README.md):
/// - RS: for s in [0, N-2], reduce recv_chunk_id_rs(s) from the left neighbor
///       into local storage and publish self_rs_ready[s+1].
/// - AG: publish self_ag_ready[0] for the owned chunk, then for s in [0, N-2]
///       wait on peer_ag_ready[left][s], copy recv_chunk_id_ag(s), and publish
///       self_ag_ready[s+1].
/// - CTA0 drains the rank-local completion counter and runs the status-coherent
///   completion barrier.
///
/// Liveness: even on invalid params or arch mismatch, the kernel must still run
/// the epilogue (tile finished signal + CTA0 drain + barrier).
template <typename T, bool kForceUnsupportedArch = false>
__global__ void ring_allreduce_sm100(
    RingAllreduceParams<T, 8> p,
    uint32_t* out_status = nullptr) {

  RingAllreduceDrainConfig cfg;
  cfg.timeout_iters = p.timeout_iters;
  cfg.timeout_cycles = p.timeout_cycles;
  cfg.poll_sleep_start = p.poll_sleep_start;
  cfg.poll_sleep_ns = p.poll_sleep_ns;

  bool valid = detail::ring_allreduce_params_valid_ngpu(p);
  bool enabled = world_size_enabled(p.world_size);
  bool ok = valid && enabled;

#if defined(__CUDA_ARCH__)

  // Compile guard: this kernel is intended for SM100/SM103 only.
  // kForceUnsupportedArch is a test-only knob to exercise the mismatch path.
  bool arch_supported = (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030) && !kForceUnsupportedArch;
  if (!arch_supported) {
    ok = false;
    ring_allreduce_publish_error_and_abort(
        p.self_error,
        p.self_abort,
        RingAllreduceError::kInvalidParams);
  }

#endif

  if (!ok) {
    ring_allreduce_publish_error_and_abort(
        p.self_error,
        p.self_abort,
        RingAllreduceError::kInvalidParams);
  }

  uint32_t tile_linear = static_cast<uint32_t>(blockIdx.x);

  bool tile_in_range = ok && (tile_linear < p.num_tiles_total) && (p.tiles_per_chunk != 0u);

  if (tile_in_range) {
    uint32_t channel_id = tile_linear / p.tiles_per_chunk;
    uint32_t tile_in_chunk = tile_linear - channel_id * p.tiles_per_chunk;

    uint32_t N = static_cast<uint32_t>(p.world_size);
    uint32_t r_u32 = static_cast<uint32_t>(p.rank);

    // Spec orientation: left = (rank - 1 + N) % N.
    int32_t left = (p.rank + p.world_size - 1) % p.world_size;

    uint64_t flags_len = uint64_t(N) * uint64_t(p.num_tiles_total);

#if CUTLASS_RING_ALLREDUCE_ENABLE_WARP_SPECIALIZED_SMEM

    bool block_shape_ok = (blockDim.x == 256) && (blockDim.y == 1) && (blockDim.z == 1);

    bool n1_publisher_only = block_shape_ok && (N == 1);

    bool smem_eligible = ok
        && tile_in_range
        && block_shape_ok
        && (N > 1)
        && (p.tile_elems <= detail::kStageElemsMax);

    if (n1_publisher_only) {
      detail::ring_allreduce_sm100_tile_warp_specialized_smem_n1_publisher_only(
          p, tile_linear, flags_len);
    }
    else if (smem_eligible) {
      detail::ring_allreduce_sm100_tile_warp_specialized_smem(
          p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len);
    }
    else {
      detail::ring_allreduce_sm100_tile_legacy(
          p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len);
    }

#else

    detail::ring_allreduce_sm100_tile_legacy(
        p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len);

#endif

  }

  // All CTAs must participate in the drain counter protocol even if this tile is
  // empty or params are invalid.
  (void)ring_allreduce_signal_tile_finished(p.self_tiles_finished);

  if (ring_allreduce_is_cta0() && ring_allreduce_is_thread0()) {

    // Rank-local drain.
    //
    // The drain helper uses strict equality (done == expected_tiles) and does
    // not early-exit on abort when timeouts are disabled. To keep this kernel
    // hang-resistant, treat any mismatch between the launched grid and
    // p.num_tiles_total as invalid params and use gridDim.x for the drain
    // expected count.
    uint32_t expected_tiles = p.num_tiles_total;
    if (expected_tiles != static_cast<uint32_t>(gridDim.x)) {
      ring_allreduce_publish_error_and_abort(
          p.self_error,
          p.self_abort,
          RingAllreduceError::kInvalidParams);
      expected_tiles = static_cast<uint32_t>(gridDim.x);
    }

    (void)ring_allreduce_drain_tiles_finished(
        p.self_tiles_finished,
        expected_tiles,
        p.self_abort,
        p.self_error,
        cfg);

    // Status-coherent completion barrier.
    (void)ring_allreduce_barrier_gather_release(p);

    if (out_status) {
      out_status[0] = static_cast<uint32_t>(ring_allreduce_local_status(p.self_abort, p.self_error));
    }
  }
}

} // namespace cutlass::distributed::collective
