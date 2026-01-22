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
    \brief Compile-only TU wired into the ring_allreduce warp-specialized SMEM pipeline tests.

    This TU forces device compilation of the warp-specialized SMEM all-gather helpers even while
    CUTLASS_RING_ALLREDUCE_ENABLE_WARP_SPECIALIZED_SMEM_AG remains disabled (runtime uses the
    legacy direct-GMEM all-gather).
*/

#include "cutlass/experimental/distributed/collective/ring_allreduce_kernel_sm100.cuh"

#include <cstdint>

#if defined(CUTLASS_RING_ALLREDUCE_ENABLE_WARP_SPECIALIZED_SMEM)
#if CUTLASS_RING_ALLREDUCE_ENABLE_WARP_SPECIALIZED_SMEM

__global__ void ring_allreduce_warp_specialized_smem_ag_compile_only_kernel() {
#if defined(__CUDA_ARCH__)

// Restrict instantiation to SM100/SM103 device passes to keep fatbin builds light.
#if (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)

  using Element = float;
  using Params = cutlass::distributed::collective::RingAllreduceParams<Element, 8>;

  __shared__ cutlass::distributed::collective::detail::RingAllreduceWarpSpecializedSmem<Element> smem;

  Params p{};
  cutlass::distributed::collective::RingAllreduceDrainConfig cfg{};

  uint32_t tile_linear = 0u;
  uint32_t channel_id = 0u;
  uint32_t tile_in_chunk = 0u;
  uint32_t N = 2u;
  uint32_t r_u32 = 0u;
  int32_t left = 0;
  uint64_t flags_len = 0ull;

  // Compile-check the AG wrapper entrypoint (even while selection remains legacy).
  cutlass::distributed::collective::detail::ring_allreduce_sm100_tile_warp_specialized_smem_ag(
      p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len, smem);

  // Compile-check ping-pong vs fallback across binaries using the exact RS guard.
#if CUDA_BARRIER_ENABLED && !CUTLASS_RING_ALLREDUCE_TEST_FORCE_NAMED_BARRIER_DISABLED
  cutlass::distributed::collective::detail::ring_allreduce_sm100_tile_warp_specialized_smem_ag_ping_pong(
      p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len, smem);
#else
  cutlass::distributed::collective::detail::ring_allreduce_sm100_tile_warp_specialized_smem_ag_single_buffer(
      p, cfg, tile_linear, channel_id, tile_in_chunk, N, r_u32, left, flags_len, smem);
#endif

  // Compile-check the allowlisted any-thread error publisher.
  uint32_t warp_id = threadIdx.x >> 5;
  uint32_t lane = threadIdx.x & 0x1Fu;

  using cutlass::distributed::collective::RingAllreduceError;
  using cutlass::distributed::collective::ring_allreduce_is_cta0;
  using cutlass::distributed::collective::ring_allreduce_publish_error_and_abort_any_thread;

  if (ring_allreduce_is_cta0() && warp_id == 6 && lane == 0) {
    ring_allreduce_publish_error_and_abort_any_thread(
        p.self_error,
        p.self_abort,
        RingAllreduceError::kAbortObserved);
  }

#endif // (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
#endif // defined(__CUDA_ARCH__)
}

#endif
#endif
