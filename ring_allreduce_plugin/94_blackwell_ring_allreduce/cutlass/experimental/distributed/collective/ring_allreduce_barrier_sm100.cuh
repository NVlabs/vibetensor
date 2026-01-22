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
    \brief Device-side ring-token barrier helpers for the Blackwell ring allreduce prototype.

    NOTE: This API is experimental and may change without notice.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "ring_allreduce_drain.hpp"
#include "ring_allreduce_types.hpp"

#include <cstdint>

namespace cutlass::distributed::collective {

struct RingAllreduceBarrierState {
  // Self (writers)
  RingAllreduceSystemAtomicU32* self_gather_token = nullptr;
  RingAllreduceSystemAtomicU32* self_gather_status = nullptr;
  RingAllreduceSystemAtomicU32* self_release_token = nullptr;
  RingAllreduceSystemAtomicU32* self_release_status = nullptr;

  // Left neighbor (readers)
  RingAllreduceSystemAtomicU32 const* left_gather_token = nullptr;
  RingAllreduceSystemAtomicU32 const* left_gather_status = nullptr;
  RingAllreduceSystemAtomicU32 const* left_release_token = nullptr;
  RingAllreduceSystemAtomicU32 const* left_release_status = nullptr;

  // Left neighbor abort flag (peer_abort[left])
  RingAllreduceSystemAtomicU32 const* left_abort = nullptr;
};

namespace detail {

CUTLASS_DEVICE
void publish_invalid_params_tokens(
    uint32_t epoch,
    RingAllreduceBarrierState st,
    RingAllreduceSystemAtomicU32* self_abort,
    RingAllreduceSystemAtomicU32* self_error) {

  // 1) Publish error+abort best effort.
  ring_allreduce_publish_error_and_abort(self_error, self_abort, RingAllreduceError::kInvalidParams);

  // 2) Publish barrier tokens best effort.
  if (epoch == 0) {
    // Cannot publish meaningful epoch-tagged tokens.
    return;
  }

  if (st.self_gather_status && st.self_gather_token) {
    st.self_gather_status->store(
        static_cast<uint32_t>(RingAllreduceError::kInvalidParams), cuda::memory_order_relaxed);
    st.self_gather_token->store(epoch, cuda::memory_order_release);
  }

  if (st.self_release_status && st.self_release_token) {
    st.self_release_status->store(
        static_cast<uint32_t>(RingAllreduceError::kInvalidParams), cuda::memory_order_relaxed);
    st.self_release_token->store(epoch, cuda::memory_order_release);
  }
}

struct AbortObservation {
  RingAllreduceSystemAtomicU32 const* self_abort = nullptr;
  RingAllreduceSystemAtomicU32 const* left_abort = nullptr;

  CUTLASS_DEVICE
  bool observed() const {
    uint32_t s = self_abort ? self_abort->load(cuda::memory_order_acquire) : 0u;
    uint32_t l = left_abort ? left_abort->load(cuda::memory_order_acquire) : 0u;
    return (s | l) != 0u;
  }
};

struct WaitTokenAndLoadStatusResult {
  bool got_token = false;
  RingAllreduceError status = RingAllreduceError::kInvalidParams;
};

CUTLASS_DEVICE
WaitTokenAndLoadStatusResult wait_token_and_load_status(
    RingAllreduceSystemAtomicU32 const* token,
    RingAllreduceSystemAtomicU32 const* status,
    uint32_t epoch,
    RingAllreduceDrainConfig cfg,
    bool timeouts_enabled,
    AbortObservation abort_obs,
    RingAllreduceBarrierState st,
    RingAllreduceSystemAtomicU32* self_abort,
    RingAllreduceSystemAtomicU32* self_error,
    RingAllreduceError timeout_error) {

#if defined(__CUDA_ARCH__)

  // Caller must have validated non-null pointers. If violated, treat as invalid params
  // and publish tokens best-effort (defensive).
  if (!token || !status || epoch == 0) {
    publish_invalid_params_tokens(epoch, st, self_abort, self_error);
    return {false, ring_allreduce_local_status(self_abort, self_error)};
  }

  uint64_t iters = 0;
  uint64_t start_cycles = (timeouts_enabled && cfg.timeout_cycles != 0) ? clock64() : 0;

  while (true) {
    if (token->load(cuda::memory_order_relaxed) == epoch) {
      break;
    }

    // Abort-aware poll (does not early exit).
    (void)abort_obs.observed();

    if (timeouts_enabled) {
      bool timed_out_iters = (cfg.timeout_iters != 0) && (iters >= uint64_t(cfg.timeout_iters));
      bool timed_out_cycles = false;
      if (cfg.timeout_cycles != 0) {
        timed_out_cycles = (clock64() - start_cycles) >= cfg.timeout_cycles;
      }
      if (timed_out_iters || timed_out_cycles) {
        ring_allreduce_publish_error_and_abort(self_error, self_abort, timeout_error);
        return {false, ring_allreduce_local_status(self_abort, self_error)};
      }
    }

    if (cfg.poll_sleep_ns > 0 && iters >= uint64_t(cfg.poll_sleep_start)) {
      #if (__CUDA_ARCH__ >= 700)
        __nanosleep(cfg.poll_sleep_ns);
      #endif
    }

    ++iters;
  }

  // Acquire-confirmation.
  if (token->load(cuda::memory_order_acquire) != epoch) {
    publish_invalid_params_tokens(epoch, st, self_abort, self_error);
    return {false, ring_allreduce_local_status(self_abort, self_error)};
  }

  RingAllreduceError s = static_cast<RingAllreduceError>(
      status->load(cuda::memory_order_relaxed));

  return {true, s};

#else

  CUTLASS_UNUSED(token);
  CUTLASS_UNUSED(status);
  CUTLASS_UNUSED(epoch);
  CUTLASS_UNUSED(cfg);
  CUTLASS_UNUSED(timeouts_enabled);
  CUTLASS_UNUSED(abort_obs);
  CUTLASS_UNUSED(st);
  CUTLASS_UNUSED(timeout_error);

  return {false, ring_allreduce_local_status(self_abort, self_error)};

#endif
}

} // namespace detail

CUTLASS_DEVICE
RingAllreduceError ring_allreduce_barrier_gather_release(
    int32_t rank,
    int32_t world_size,
    uint32_t epoch,
    RingAllreduceBarrierState st,
    RingAllreduceSystemAtomicU32* self_abort,
    RingAllreduceSystemAtomicU32* self_error,
    RingAllreduceDrainConfig cfg,
    uint32_t debug_release_delay_rank,
    uint32_t debug_release_delay_iters) {

#if defined(__CUDA_ARCH__)

  if (!ring_allreduce_is_cta0() || !ring_allreduce_is_thread0()) {
    return ring_allreduce_local_status(self_abort, self_error);
  }

  // Validate scalar params.
  bool scalar_ok = (world_size > 0) && (rank >= 0) && (rank < world_size) && (epoch != 0);

  // Validate required self pointers.
  bool self_ok = self_abort && self_error &&
      st.self_gather_token && st.self_gather_status &&
      st.self_release_token && st.self_release_status;

  // Validate required left pointers.
  bool left_ok = true;
  if (world_size > 1) {
    left_ok = st.left_gather_token && st.left_gather_status &&
              st.left_release_token && st.left_release_status &&
              st.left_abort;
  }

  if (!(scalar_ok && self_ok && left_ok)) {
    detail::publish_invalid_params_tokens(epoch, st, self_abort, self_error);
    return ring_allreduce_local_status(self_abort, self_error);
  }

  detail::AbortObservation abort_obs{self_abort, (world_size > 1) ? st.left_abort : nullptr};

  // Degenerate ring: publish gather+release locally without dereferencing any peer pointers.
  if (world_size == 1) {
    RingAllreduceError local = ring_allreduce_local_status(self_abort, self_error);

    st.self_gather_status->store(static_cast<uint32_t>(local), cuda::memory_order_relaxed);
    st.self_gather_token->store(epoch, cuda::memory_order_release);

    st.self_release_status->store(static_cast<uint32_t>(local), cuda::memory_order_relaxed);
    st.self_release_token->store(epoch, cuda::memory_order_release);

    if (local != RingAllreduceError::kOk) {
      ring_allreduce_publish_error_and_abort(self_error, self_abort, local);
    }

    return ring_allreduce_local_status(self_abort, self_error);
  }

  // Phase A (GATHER)
  RingAllreduceError local = ring_allreduce_local_status(self_abort, self_error);
  RingAllreduceError agg = local;

  if (rank == 0) {
    st.self_gather_status->store(static_cast<uint32_t>(local), cuda::memory_order_relaxed);
    st.self_gather_token->store(epoch, cuda::memory_order_release);

    auto r = detail::wait_token_and_load_status(
        st.left_gather_token,
        st.left_gather_status,
        epoch,
        cfg,
        /*timeouts_enabled=*/true,
        abort_obs,
        st,
        self_abort,
        self_error,
        /*timeout_error=*/RingAllreduceError::kTimeout);

    agg = r.got_token ? r.status : ring_allreduce_local_status(self_abort, self_error);
  }
  else {
    auto r = detail::wait_token_and_load_status(
        st.left_gather_token,
        st.left_gather_status,
        epoch,
        cfg,
        /*timeouts_enabled=*/true,
        abort_obs,
        st,
        self_abort,
        self_error,
        /*timeout_error=*/RingAllreduceError::kTimeout);

    if (!r.got_token) {
      // Timeout-forwarding: publish local_status and token so ring can converge.
      RingAllreduceError now = ring_allreduce_local_status(self_abort, self_error);
      st.self_gather_status->store(static_cast<uint32_t>(now), cuda::memory_order_relaxed);
      st.self_gather_token->store(epoch, cuda::memory_order_release);
      agg = now;
    }
    else {
      RingAllreduceError incoming = r.status;
      agg = merge_status(local, incoming);
      st.self_gather_status->store(static_cast<uint32_t>(agg), cuda::memory_order_relaxed);
      st.self_gather_token->store(epoch, cuda::memory_order_release);
    }
  }

  RingAllreduceError final_status = agg;

  // Phase B (RELEASE)
  if (rank == 0) {
    st.self_release_status->store(static_cast<uint32_t>(final_status), cuda::memory_order_relaxed);
    st.self_release_token->store(epoch, cuda::memory_order_release);

    // If final_status==kOk, rank0 must not time out waiting for the release token to return.
    bool timeouts_enabled = (final_status != RingAllreduceError::kOk);

    (void)detail::wait_token_and_load_status(
        st.left_release_token,
        st.left_release_status,
        epoch,
        cfg,
        /*timeouts_enabled=*/timeouts_enabled,
        abort_obs,
        st,
        self_abort,
        self_error,
        /*timeout_error=*/RingAllreduceError::kTimeout);

    // After determining final_status, latch non-Ok into local abort/error state.
    if (final_status != RingAllreduceError::kOk) {
      ring_allreduce_publish_error_and_abort(self_error, self_abort, final_status);
    }

    return ring_allreduce_local_status(self_abort, self_error);
  }

  // rank != 0: wait for broadcast with "arm pre-token timeouts only after abort observed".
  bool timeouts_enabled = false;
  uint64_t iters = 0;
  uint64_t start_cycles = 0;

  while (st.left_release_token->load(cuda::memory_order_relaxed) != epoch) {
    bool abort_seen = abort_obs.observed();

    if (!timeouts_enabled && abort_seen) {
      timeouts_enabled = true;
      iters = 0;
      start_cycles = (cfg.timeout_cycles != 0) ? clock64() : 0;
    }

    if (timeouts_enabled) {
      bool timed_out_iters = (cfg.timeout_iters != 0) && (iters >= uint64_t(cfg.timeout_iters));
      bool timed_out_cycles = false;
      if (cfg.timeout_cycles != 0) {
        timed_out_cycles = (clock64() - start_cycles) >= cfg.timeout_cycles;
      }
      if (timed_out_iters || timed_out_cycles) {
        ring_allreduce_publish_error_and_abort(self_error, self_abort, RingAllreduceError::kTimeout);

        // Liveness: publish release token/status best-effort so right neighbor isn't stranded.
        RingAllreduceError now = ring_allreduce_local_status(self_abort, self_error);
        st.self_release_status->store(static_cast<uint32_t>(now), cuda::memory_order_relaxed);
        st.self_release_token->store(epoch, cuda::memory_order_release);

        return ring_allreduce_local_status(self_abort, self_error);
      }
    }

    if (cfg.poll_sleep_ns > 0 && iters >= uint64_t(cfg.poll_sleep_start)) {
      #if (__CUDA_ARCH__ >= 700)
        __nanosleep(cfg.poll_sleep_ns);
      #endif
    }

    ++iters;
  }

  // Acquire-confirm.
  if (st.left_release_token->load(cuda::memory_order_acquire) != epoch) {
    detail::publish_invalid_params_tokens(epoch, st, self_abort, self_error);
    return ring_allreduce_local_status(self_abort, self_error);
  }

  RingAllreduceError broadcast = static_cast<RingAllreduceError>(
      st.left_release_status->load(cuda::memory_order_relaxed));

  if (rank == static_cast<int32_t>(debug_release_delay_rank)) {
    for (uint32_t i = 0; i < debug_release_delay_iters; ++i) {
      #if (__CUDA_ARCH__ >= 700)
        __nanosleep(40);
      #endif
    }
  }

  // Must-forward exactly once, even if abort observed.
  st.self_release_status->store(static_cast<uint32_t>(broadcast), cuda::memory_order_relaxed);
  st.self_release_token->store(epoch, cuda::memory_order_release);

  // After receiving final_status:
  if (broadcast != RingAllreduceError::kOk) {
    ring_allreduce_publish_error_and_abort(self_error, self_abort, broadcast);
  }

  return ring_allreduce_local_status(self_abort, self_error);

#else

  CUTLASS_UNUSED(rank);
  CUTLASS_UNUSED(world_size);
  CUTLASS_UNUSED(epoch);
  CUTLASS_UNUSED(st);
  CUTLASS_UNUSED(cfg);
  CUTLASS_UNUSED(debug_release_delay_rank);
  CUTLASS_UNUSED(debug_release_delay_iters);

  return ring_allreduce_local_status(self_abort, self_error);

#endif
}

template <typename T, int kMaxWorldSize>
CUTLASS_DEVICE
RingAllreduceError ring_allreduce_barrier_gather_release(
    RingAllreduceParams<T, kMaxWorldSize> const& p) {

#if defined(__CUDA_ARCH__)

  if (!ring_allreduce_is_cta0() || !ring_allreduce_is_thread0()) {
    return ring_allreduce_local_status(p.self_abort, p.self_error);
  }

  RingAllreduceDrainConfig cfg;
  cfg.timeout_iters = p.timeout_iters;
  cfg.timeout_cycles = p.timeout_cycles;
  cfg.poll_sleep_start = p.poll_sleep_start;
  cfg.poll_sleep_ns = p.poll_sleep_ns;

  // Populate self pointers up front so invalid-params publication can still
  // best-effort publish barrier tokens for liveness.
  RingAllreduceBarrierState st;
  st.self_gather_token = p.self_barrier_gather_token;
  st.self_gather_status = p.self_barrier_gather_status;
  st.self_release_token = p.self_barrier_release_token;
  st.self_release_status = p.self_barrier_release_status;

  // Validation order is important: validate world_size bounds before computing
  // left and before indexing any peer_*[kMaxWorldSize] arrays.
  if (p.world_size <= 0) {
    detail::publish_invalid_params_tokens(p.epoch, st, p.self_abort, p.self_error);
    return ring_allreduce_local_status(p.self_abort, p.self_error);
  }

  if (p.world_size > kMaxWorldSize) {
    detail::publish_invalid_params_tokens(p.epoch, st, p.self_abort, p.self_error);
    return ring_allreduce_local_status(p.self_abort, p.self_error);
  }

  if (p.rank < 0 || p.rank >= p.world_size) {
    detail::publish_invalid_params_tokens(p.epoch, st, p.self_abort, p.self_error);
    return ring_allreduce_local_status(p.self_abort, p.self_error);
  }

  if (p.epoch == 0) {
    detail::publish_invalid_params_tokens(p.epoch, st, p.self_abort, p.self_error);
    return ring_allreduce_local_status(p.self_abort, p.self_error);
  }

  // Required self pointers.
  bool self_ok = p.self_abort && p.self_error &&
      st.self_gather_token && st.self_gather_status &&
      st.self_release_token && st.self_release_status;

  if (!self_ok) {
    detail::publish_invalid_params_tokens(p.epoch, st, p.self_abort, p.self_error);
    return ring_allreduce_local_status(p.self_abort, p.self_error);
  }

  // Degenerate ring: no peer pointers required.
  if (p.world_size == 1) {
    return ring_allreduce_barrier_gather_release(
        p.rank,
        p.world_size,
        p.epoch,
        st,
        p.self_abort,
        p.self_error,
        cfg,
        p.debug_release_delay_rank,
        p.debug_release_delay_iters);
  }

  int32_t left = (p.rank + p.world_size - 1) % p.world_size;

  st.left_gather_token = p.peer_barrier_gather_token[left];
  st.left_gather_status = p.peer_barrier_gather_status[left];
  st.left_release_token = p.peer_barrier_release_token[left];
  st.left_release_status = p.peer_barrier_release_status[left];
  st.left_abort = p.peer_abort[left];

  bool left_ok = st.left_gather_token && st.left_gather_status &&
      st.left_release_token && st.left_release_status &&
      st.left_abort;

  if (!left_ok) {
    detail::publish_invalid_params_tokens(p.epoch, st, p.self_abort, p.self_error);
    return ring_allreduce_local_status(p.self_abort, p.self_error);
  }

  return ring_allreduce_barrier_gather_release(
      p.rank,
      p.world_size,
      p.epoch,
      st,
      p.self_abort,
      p.self_error,
      cfg,
      p.debug_release_delay_rank,
      p.debug_release_delay_iters);

#else

  CUTLASS_UNUSED(p);
  return RingAllreduceError::kInvalidParams;

#endif
}

} // namespace cutlass::distributed::collective
