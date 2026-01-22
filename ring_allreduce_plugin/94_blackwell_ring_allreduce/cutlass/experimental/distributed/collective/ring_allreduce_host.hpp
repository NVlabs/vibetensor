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
    \brief Host-side capability validation helpers for the Blackwell ring allreduce prototype.

    NOTE: This API is experimental and may change without notice.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "ring_allreduce_tiling.hpp"
#include "ring_allreduce_types.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace cutlass::distributed::collective {

/// Result type for host-side validation helpers.
struct RingAllreduceHostResult {
  cutlass::Status status = cutlass::Status::kSuccess;
  cudaError_t cuda_error = cudaSuccess;

  // For capability failures involving a device pair.
  int device_a = -1;
  int device_b = -1;

  char const* error_reason = nullptr;

  static inline CUTLASS_HOST
  RingAllreduceHostResult success() {
    return RingAllreduceHostResult{};
  }

  static inline CUTLASS_HOST
  RingAllreduceHostResult failure(cutlass::Status status_, char const* reason) {
    RingAllreduceHostResult r;
    r.status = status_;
    r.error_reason = reason;
    return r;
  }

  static inline CUTLASS_HOST
  RingAllreduceHostResult cuda_failure(cudaError_t cuda_error_, char const* reason, int a = -1, int b = -1) {
    RingAllreduceHostResult r;
    r.status = cutlass::Status::kErrorInternal;
    r.cuda_error = cuda_error_;
    r.device_a = a;
    r.device_b = b;
    r.error_reason = reason;
    return r;
  }

  CUTLASS_HOST
  bool ok() const {
    return status == cutlass::Status::kSuccess;
  }
};

struct RingAllreduceP2POptions {
  bool enable_peer_access = true;
  bool require_native_atomics = true;

  // Debug-only escape hatch. Never set this in production paths.
  //
  // NOTE: This bypasses capability checks but does *not* bypass failures from
  // cudaDeviceEnablePeerAccess(). If you want a dry-run capability query on a
  // system that does not support P2P, also set enable_peer_access=false.
  bool allow_unsupported = false;
};

/// Enables peer access and validates P2P capabilities for a ring of devices.
///
/// Per ring neighbor pair (a, b) where b = (a+1) % world_size, the host validates:
///   - cudaDeviceCanAccessPeer in both directions.
///   - cudaDevP2PAttrNativeAtomicSupported in both directions (unless disabled).
///
/// If options.enable_peer_access is true, peer access is enabled in both directions.
///
/// device_ids may be null; in that case device i is assumed to be i.
static inline CUTLASS_HOST
RingAllreduceHostResult validate_ring_p2p_caps_and_enable_peer_access(
    int32_t world_size,
    int const* device_ids = nullptr,
    RingAllreduceP2POptions options = {}) {

  if (world_size <= 0) {
    return RingAllreduceHostResult::failure(cutlass::Status::kErrorInvalidProblem,
                                            "world_size must be > 0");
  }

  int device_count = 0;
  cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess) {
    return RingAllreduceHostResult::cuda_failure(cuda_status, "cudaGetDeviceCount failed");
  }

  if (device_count < world_size) {
    return RingAllreduceHostResult::failure(cutlass::Status::kErrorNotSupported,
                                            "insufficient CUDA devices for requested world_size");
  }

  auto device_at = [&](int i) -> int {
    return device_ids ? device_ids[i] : i;
  };

  // Validate device ids are in range and unique.
  for (int i = 0; i < world_size; ++i) {
    int di = device_at(i);
    if (di < 0 || di >= device_count) {
      return RingAllreduceHostResult::failure(cutlass::Status::kErrorInvalidProblem,
                                              "device id out of range");
    }
    for (int j = i + 1; j < world_size; ++j) {
      int dj = device_at(j);
      if (di == dj) {
        return RingAllreduceHostResult::failure(cutlass::Status::kErrorInvalidProblem,
                                                "device id list contains duplicates");
      }
    }
  }

  int original_device = 0;
  cuda_status = cudaGetDevice(&original_device);
  if (cuda_status != cudaSuccess) {
    return RingAllreduceHostResult::cuda_failure(cuda_status, "cudaGetDevice failed");
  }

  auto check_direction = [&](int a, int b) -> RingAllreduceHostResult {
    int can_access = 0;
    cudaError_t st = cudaDeviceCanAccessPeer(&can_access, a, b);
    if (st != cudaSuccess) {
      return RingAllreduceHostResult::cuda_failure(st, "cudaDeviceCanAccessPeer failed", a, b);
    }

    if (!can_access) {
      if (options.allow_unsupported) {
        return RingAllreduceHostResult::success();
      }
      RingAllreduceHostResult r;
      r.status = cutlass::Status::kErrorNotSupported;
      r.device_a = a;
      r.device_b = b;
      r.error_reason = "peer access unsupported";
      return r;
    }

    if (options.require_native_atomics) {
      int native_atomics = 0;
      st = cudaDeviceGetP2PAttribute(&native_atomics, cudaDevP2PAttrNativeAtomicSupported, a, b);
      if (st != cudaSuccess) {
        return RingAllreduceHostResult::cuda_failure(st, "cudaDeviceGetP2PAttribute failed", a, b);
      }
      if (native_atomics != 1) {
        if (options.allow_unsupported) {
          return RingAllreduceHostResult::success();
        }
        RingAllreduceHostResult r;
        r.status = cutlass::Status::kErrorNotSupported;
        r.device_a = a;
        r.device_b = b;
        r.error_reason = "native peer atomics unsupported";
        return r;
      }
    }

    return RingAllreduceHostResult::success();
  };

  auto enable_one_direction = [&](int a, int b) -> RingAllreduceHostResult {
    if (!options.enable_peer_access) {
      return RingAllreduceHostResult::success();
    }

    cudaError_t st = cudaSetDevice(a);
    if (st != cudaSuccess) {
      return RingAllreduceHostResult::cuda_failure(st, "cudaSetDevice failed", a, b);
    }

    st = cudaDeviceEnablePeerAccess(b, 0);
    if (st == cudaErrorPeerAccessAlreadyEnabled) {
      // Clear the per-thread CUDA error state.
      (void)cudaGetLastError();
      return RingAllreduceHostResult::success();
    }

    if (st != cudaSuccess) {
      return RingAllreduceHostResult::cuda_failure(st, "cudaDeviceEnablePeerAccess failed", a, b);
    }

    return RingAllreduceHostResult::success();
  };

  if (world_size == 1) {
    // No peer access required.
    (void)cudaSetDevice(original_device);
    return RingAllreduceHostResult::success();
  }

  // Validate all neighbor pairs in both directions.
  for (int i = 0; i < world_size; ++i) {
    int a = device_at(i);
    int b = device_at((i + 1) % world_size);

    if (a == b) {
      (void)cudaSetDevice(original_device);
      return RingAllreduceHostResult::failure(cutlass::Status::kErrorInvalidProblem,
                                              "neighbor devices must be distinct");
    }

    if (auto r = check_direction(a, b); !r.ok()) {
      (void)cudaSetDevice(original_device);
      return r;
    }
    if (auto r = check_direction(b, a); !r.ok()) {
      (void)cudaSetDevice(original_device);
      return r;
    }
  }

  // Enable access for the same set of neighbor pairs in both directions.
  for (int i = 0; i < world_size; ++i) {
    int a = device_at(i);
    int b = device_at((i + 1) % world_size);

    if (auto r = enable_one_direction(a, b); !r.ok()) {
      (void)cudaSetDevice(original_device);
      return r;
    }
    if (auto r = enable_one_direction(b, a); !r.ok()) {
      (void)cudaSetDevice(original_device);
      return r;
    }
  }

  cuda_status = cudaSetDevice(original_device);
  if (cuda_status != cudaSuccess) {
    return RingAllreduceHostResult::cuda_failure(cuda_status, "cudaSetDevice restore failed");
  }

  return RingAllreduceHostResult::success();
}

/// Validates host-derived tiling against per-device launch limits.
///
/// This checks:
///   - num_tiles_total <= deviceProp.maxGridSize[0] for each device.
///   - readiness flag allocation sizes fit size_t.
///
/// device_ids may be null; in that case device i is assumed to be i.
static inline CUTLASS_HOST
RingAllreduceHostResult validate_ring_allreduce_host_tiling(
    uint64_t count,
    int32_t world_size,
    int32_t num_channels,
    uint32_t tile_elems,
    RingAllreduceTiling* out_tiling,
    int const* device_ids = nullptr) {

  if (!out_tiling) {
    return RingAllreduceHostResult::failure(cutlass::Status::kErrorInvalidProblem,
                                            "out_tiling must be non-null");
  }

  auto tiling_result = compute_ring_allreduce_tiling(count, world_size, num_channels, tile_elems);
  if (!tiling_result.ok()) {
    RingAllreduceHostResult r;
    r.status = tiling_result.status;
    r.error_reason = tiling_result.error_reason;
    return r;
  }

  *out_tiling = tiling_result.tiling;

  // count==0 is a host-level no-op; launch limits are irrelevant.
  if (count == 0) {
    return RingAllreduceHostResult::success();
  }

  int device_count = 0;
  cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess) {
    return RingAllreduceHostResult::cuda_failure(cuda_status, "cudaGetDeviceCount failed");
  }

  if (device_count < world_size) {
    return RingAllreduceHostResult::failure(cutlass::Status::kErrorNotSupported,
                                            "insufficient CUDA devices for requested world_size");
  }

  auto device_at = [&](int i) -> int {
    return device_ids ? device_ids[i] : i;
  };

  // Validate device ids are in range and unique.
  for (int i = 0; i < world_size; ++i) {
    int di = device_at(i);
    if (di < 0 || di >= device_count) {
      return RingAllreduceHostResult::failure(cutlass::Status::kErrorInvalidProblem,
                                              "device id out of range");
    }

    for (int j = i + 1; j < world_size; ++j) {
      if (di == device_at(j)) {
        return RingAllreduceHostResult::failure(cutlass::Status::kErrorInvalidProblem,
                                                "device id list contains duplicates");
      }
    }
  }

  for (int i = 0; i < world_size; ++i) {
    cudaDeviceProp prop;
    int dev = device_at(i);
    cuda_status = cudaGetDeviceProperties(&prop, dev);
    if (cuda_status != cudaSuccess) {
      return RingAllreduceHostResult::cuda_failure(cuda_status, "cudaGetDeviceProperties failed", dev);
    }

    // maxGridSize is int[3]. Only x is relevant for 1D grid.
    if (out_tiling->num_tiles_total > static_cast<uint32_t>(prop.maxGridSize[0])) {
      RingAllreduceHostResult r;
      r.status = cutlass::Status::kErrorInvalidProblem;
      r.device_a = dev;
      r.error_reason = "num_tiles_total exceeds maxGridSize.x";
      return r;
    }
  }

  // Overflow-safe readiness flag allocation size checks.
  // flags_len = world_size * num_tiles_total
  constexpr size_t kMaxSizeT = static_cast<size_t>(-1);

  uint64_t flags_len_u64 = uint64_t(world_size) * uint64_t(out_tiling->num_tiles_total);
  if (flags_len_u64 > static_cast<uint64_t>(kMaxSizeT)) {
    return RingAllreduceHostResult::failure(cutlass::Status::kErrorInvalidProblem,
                                            "flag element count overflows size_t");
  }

  size_t flags_len = static_cast<size_t>(flags_len_u64);
  if (flags_len != 0 && flags_len > kMaxSizeT / sizeof(RingAllreduceSystemAtomicU32)) {
    return RingAllreduceHostResult::failure(cutlass::Status::kErrorInvalidProblem,
                                            "flag allocation size overflows size_t");
  }

  return RingAllreduceHostResult::success();
}

} // namespace cutlass::distributed::collective
