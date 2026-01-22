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
  \brief Minimal 2-GPU ring allreduce (sum) sample for Blackwell (SM100/SM103).

  This example launches one ring_allreduce_sm100 kernel per device and validates:

    - R0: basic correctness (no abort injection)
    - R1-like: abort-before/abort-after AG publish (status-coherence check)

  Requirements:

    - >=2 CUDA devices (SM100 or SM103)
    - P2P access enabled between the two devices
    - Native peer atomics supported between the two devices

  Usage:

    $ ./94_blackwell_ring_allreduce --count=1024 --tile_elems=256 --num_channels=1

    # Inject abort-after-publish at rank0 (world_size==2 => final AG step is s==0)
    $ ./94_blackwell_ring_allreduce --abort_after_ag_publish
*/

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <new>
#include <thread>
#include <vector>

#include "cutlass/cutlass.h"

#include "cutlass/experimental/distributed/collective/ring_allreduce_host.hpp"
#include "cutlass/experimental/distributed/collective/ring_allreduce_kernel_sm100.cuh"

#include "cutlass/util/command_line.h"

#include "helper.h"

namespace {

using cutlass::distributed::collective::RingAllreduceDeviceAtomicU32;
using cutlass::distributed::collective::RingAllreduceError;
using cutlass::distributed::collective::RingAllreduceHostResult;
using cutlass::distributed::collective::RingAllreduceParams;
using cutlass::distributed::collective::RingAllreduceSystemAtomicU32;
using cutlass::distributed::collective::RingAllreduceTiling;
using cutlass::distributed::collective::validate_ring_allreduce_host_tiling;
using cutlass::distributed::collective::validate_ring_p2p_caps_and_enable_peer_access;

constexpr int kWorldSize = 2;

static char const* ring_allreduce_error_to_string(RingAllreduceError e) {
  switch (e) {
    case RingAllreduceError::kOk: return "kOk";
    case RingAllreduceError::kInvalidParams: return "kInvalidParams";
    case RingAllreduceError::kTimeout: return "kTimeout";
    case RingAllreduceError::kAbortObserved: return "kAbortObserved";
    default: return "<unknown>";
  }
}

static void print_host_result(char const* what, RingAllreduceHostResult const& r) {
  std::cerr << what << " failed: status=" << cutlassGetStatusString(r.status)
            << " cuda_error=" << cudaGetErrorString(r.cuda_error);

  if (r.device_a >= 0) {
    std::cerr << " device_a=" << r.device_a;
  }
  if (r.device_b >= 0) {
    std::cerr << " device_b=" << r.device_b;
  }
  if (r.error_reason) {
    std::cerr << " reason='" << r.error_reason << "'";
  }
  std::cerr << "\n";
}

struct Options {
  bool help = false;
  bool error = false;

  uint64_t count = 1024;
  int32_t num_channels = 1;
  uint32_t tile_elems = 256;
  uint32_t epoch = 1;

  int device0 = -1;
  int device1 = -1;

  uint32_t abort_rank = 0;
  bool abort_before_ag_publish = false;
  bool abort_after_ag_publish = false;

  int watchdog_ms = 10'000;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("count", count, uint64_t(1024));
    cmd.get_cmd_line_argument("num_channels", num_channels, int32_t(1));
    cmd.get_cmd_line_argument("tile_elems", tile_elems, uint32_t(256));
    cmd.get_cmd_line_argument("epoch", epoch, uint32_t(1));

    cmd.get_cmd_line_argument("device0", device0, -1);
    cmd.get_cmd_line_argument("device1", device1, -1);

    bool device0_set = device0 >= 0;
    bool device1_set = device1 >= 0;
    if (device0_set != device1_set) {
      std::cerr << "device0/device1 must be specified together (or not at all)\n";
      error = true;
    }

    cmd.get_cmd_line_argument("abort_rank", abort_rank, uint32_t(0));
    abort_before_ag_publish = cmd.check_cmd_line_flag("abort_before_ag_publish");
    abort_after_ag_publish = cmd.check_cmd_line_flag("abort_after_ag_publish");

    cmd.get_cmd_line_argument("watchdog_ms", watchdog_ms, 10'000);

    if (abort_rank >= uint32_t(kWorldSize)) {
      std::cerr << "abort_rank must be 0 or 1\n";
      error = true;
    }

    if (abort_before_ag_publish && abort_after_ag_publish) {
      std::cerr << "abort_before_ag_publish and abort_after_ag_publish are mutually exclusive\n";
      error = true;
    }

    if (count == 0) {
      std::cerr << "count must be > 0\n";
      error = true;
    }

    if (num_channels <= 0) {
      std::cerr << "num_channels must be > 0\n";
      error = true;
    }

    if (tile_elems == 0) {
      std::cerr << "tile_elems must be > 0\n";
      error = true;
    }

    if (watchdog_ms <= 0) {
      std::cerr << "watchdog_ms must be > 0\n";
      error = true;
    }
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "94_blackwell_ring_allreduce\n\n"
        << "  2-GPU ring allreduce (sum) prototype for Blackwell (SM100/SM103).\n\n"
        << "Options:\n\n"
        << "  --help                        Display this usage statement\n"
        << "  --count=<int>                 Element count (default: 1024)\n"
        << "  --num_channels=<int>          Number of channels (default: 1)\n"
        << "  --tile_elems=<int>            Tile size in elements (default: 256)\n"
        << "  --epoch=<int>                 Epoch tag (default: 1)\n"
        << "  --device0=<int>               Optional explicit device0 id\n"
        << "  --device1=<int>               Optional explicit device1 id\n"
        << "  --watchdog_ms=<int>           Host watchdog timeout (default: 10000)\n\n"
        << "Abort injection (R1-like):\n\n"
        << "  --abort_rank=<int>            Rank to inject abort on (default: 0)\n"
        << "  --abort_before_ag_publish     Abort before publishing AG forwarding flag\n"
        << "  --abort_after_ag_publish      Abort after publishing AG forwarding flag\n";

    return out;
  }
};

struct RingAllreduce2GpuAtomics {
  RingAllreduceSystemAtomicU32* self_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* self_ag_ready = nullptr;
  uint32_t flags_len = 0u;

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;

  RingAllreduceDeviceAtomicU32* self_tiles_finished = nullptr;

  RingAllreduceSystemAtomicU32* self_barrier_gather_token = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_gather_status = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_release_token = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_release_status = nullptr;
};

__global__ void construct_2gpu_atomics_kernel(RingAllreduce2GpuAtomics a) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.flags_len) {
    if (a.self_rs_ready) {
      new (a.self_rs_ready + idx) RingAllreduceSystemAtomicU32{};
    }
    if (a.self_ag_ready) {
      new (a.self_ag_ready + idx) RingAllreduceSystemAtomicU32{};
    }
  }

  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (a.self_abort) {
    new (a.self_abort) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_error) {
    new (a.self_error) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_tiles_finished) {
    new (a.self_tiles_finished) RingAllreduceDeviceAtomicU32{};
  }

  if (a.self_barrier_gather_token) {
    new (a.self_barrier_gather_token) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_barrier_gather_status) {
    new (a.self_barrier_gather_status) RingAllreduceSystemAtomicU32{};
  }

  if (a.self_barrier_release_token) {
    new (a.self_barrier_release_token) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_barrier_release_status) {
    new (a.self_barrier_release_status) RingAllreduceSystemAtomicU32{};
  }
}

__global__ void reset_2gpu_atomics_kernel(RingAllreduce2GpuAtomics a, uint32_t epoch) {
  // Tokens are reset to 0 between runs. Keep epoch in the signature for future
  // per-epoch reset logic.
  CUTLASS_UNUSED(epoch);

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.flags_len) {
    if (a.self_rs_ready) {
      a.self_rs_ready[idx].store(0u, cuda::memory_order_relaxed);
    }
    if (a.self_ag_ready) {
      a.self_ag_ready[idx].store(0u, cuda::memory_order_relaxed);
    }
  }

  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (a.self_abort) {
    a.self_abort->store(0u, cuda::memory_order_relaxed);
  }
  if (a.self_error) {
    a.self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  }
  if (a.self_tiles_finished) {
    a.self_tiles_finished->store(0u, cuda::memory_order_relaxed);
  }

  if (a.self_barrier_gather_token) {
    a.self_barrier_gather_token->store(0u, cuda::memory_order_relaxed);
  }
  if (a.self_barrier_gather_status) {
    a.self_barrier_gather_status->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  }

  if (a.self_barrier_release_token) {
    a.self_barrier_release_token->store(0u, cuda::memory_order_relaxed);
  }
  if (a.self_barrier_release_status) {
    a.self_barrier_release_status->store(static_cast<uint32_t>(RingAllreduceError::kOk), cuda::memory_order_relaxed);
  }
}

__global__ void destroy_2gpu_atomics_kernel(RingAllreduce2GpuAtomics a) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.flags_len) {
    if (a.self_ag_ready) {
      a.self_ag_ready[idx].~RingAllreduceSystemAtomicU32();
    }
    if (a.self_rs_ready) {
      a.self_rs_ready[idx].~RingAllreduceSystemAtomicU32();
    }
  }

  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (a.self_barrier_release_status) {
    a.self_barrier_release_status->~RingAllreduceSystemAtomicU32();
  }
  if (a.self_barrier_release_token) {
    a.self_barrier_release_token->~RingAllreduceSystemAtomicU32();
  }

  if (a.self_barrier_gather_status) {
    a.self_barrier_gather_status->~RingAllreduceSystemAtomicU32();
  }
  if (a.self_barrier_gather_token) {
    a.self_barrier_gather_token->~RingAllreduceSystemAtomicU32();
  }

  if (a.self_tiles_finished) {
    a.self_tiles_finished->~RingAllreduceDeviceAtomicU32();
  }
  if (a.self_error) {
    a.self_error->~RingAllreduceSystemAtomicU32();
  }
  if (a.self_abort) {
    a.self_abort->~RingAllreduceSystemAtomicU32();
  }
}

static bool is_sm100_or_sm103(int device) {
  cudaDeviceProp prop{};
  cudaError_t st = cudaGetDeviceProperties(&prop, device);
  if (st != cudaSuccess) {
    return false;
  }

  int cc = prop.major * 10 + prop.minor;
  return cc == 100 || cc == 103;
}

__global__ void ring_allreduce_2gpu_probe_kernel() {}

static cudaError_t ring_allreduce_2gpu_probe_launch(int device) {
  // Clear any pre-existing per-thread CUDA error state.
  (void)cudaGetLastError();

  cudaError_t st = cudaSetDevice(device);
  if (st != cudaSuccess) {
    return st;
  }

  ring_allreduce_2gpu_probe_kernel<<<1, 1>>>();
  return cudaGetLastError();
}

static bool select_devices(std::array<int, kWorldSize>& out_devices, Options const& options) {
  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess || device_count < kWorldSize) {
    std::cerr << "requires >= " << kWorldSize << " CUDA devices\n";
    return false;
  }

  auto device_is_compatible = [&](int dev) -> bool {
    if (!is_sm100_or_sm103(dev)) {
      return false;
    }

    cudaError_t probe_st = ring_allreduce_2gpu_probe_launch(dev);
    if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
      return false;
    }

    if (probe_st != cudaSuccess) {
      std::cerr << "probe launch failed on device " << dev << ": " << cudaGetErrorString(probe_st) << "\n";
      return false;
    }

    return true;
  };

  if (options.device0 >= 0 && options.device1 >= 0) {
    if (options.device0 == options.device1) {
      std::cerr << "device0 and device1 must be distinct\n";
      return false;
    }
    if (options.device0 >= device_count || options.device1 >= device_count) {
      std::cerr << "device id out of range\n";
      return false;
    }
    if (!device_is_compatible(options.device0) || !device_is_compatible(options.device1)) {
      std::cerr << "specified devices are not compatible (need SM100/SM103 with a valid kernel image)\n";
      return false;
    }

    out_devices[0] = options.device0;
    out_devices[1] = options.device1;

    auto p2p = validate_ring_p2p_caps_and_enable_peer_access(kWorldSize, out_devices.data());
    if (!p2p.ok()) {
      print_host_result("validate_ring_p2p_caps_and_enable_peer_access", p2p);
      return false;
    }

    return true;
  }

  // Auto-select a suitable SM100/SM103 P2P pair.
  std::vector<int> candidates;
  candidates.reserve(device_count);

  for (int dev = 0; dev < device_count; ++dev) {
    if (device_is_compatible(dev)) {
      candidates.push_back(dev);
    }
  }

  if (static_cast<int>(candidates.size()) < kWorldSize) {
    std::cerr << "requires >= " << kWorldSize << " SM100/SM103 devices with a valid kernel image\n";
    return false;
  }

  for (size_t i = 0; i < candidates.size(); ++i) {
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      out_devices[0] = candidates[i];
      out_devices[1] = candidates[j];

      auto p2p = validate_ring_p2p_caps_and_enable_peer_access(kWorldSize, out_devices.data());
      if (p2p.ok()) {
        return true;
      }
    }
  }

  std::cerr << "no suitable SM100/SM103 P2P pair with native peer atomics\n";
  return false;
}

} // namespace

int main(int argc, char const** argv) {

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout);
    return 0;
  }

  if (options.error) {
    options.print_usage(std::cerr);
    return 1;
  }

  bool devices_requested = (options.device0 >= 0 && options.device1 >= 0);

  std::array<int, kWorldSize> devices{};
  if (!select_devices(devices, options)) {
    // If devices were explicitly requested, treat selection failure as an error.
    // Otherwise, treat capability unavailability as a "skip" for this sample.
    return devices_requested ? 1 : 0;
  }

  std::cout << "Using devices: [" << devices[0] << ", " << devices[1] << "]\n";
  std::cout << "Config: count=" << options.count
            << " num_channels=" << options.num_channels
            << " tile_elems=" << options.tile_elems
            << " epoch=" << options.epoch
            << " watchdog_ms=" << options.watchdog_ms
            << "\n";

  bool do_abort_injection = options.abort_before_ag_publish || options.abort_after_ag_publish;
  if (do_abort_injection) {
    std::cout << "Abort injection: rank=" << options.abort_rank
              << " before_ag_publish=" << (options.abort_before_ag_publish ? 1 : 0)
              << " after_ag_publish=" << (options.abort_after_ag_publish ? 1 : 0)
              << "\n";
  }

  RingAllreduceTiling tiling{};
  auto tiling_r = validate_ring_allreduce_host_tiling(
      options.count,
      kWorldSize,
      options.num_channels,
      options.tile_elems,
      &tiling,
      devices.data());

  if (!tiling_r.ok()) {
    print_host_result("validate_ring_allreduce_host_tiling", tiling_r);
    return 1;
  }

  uint32_t flags_len = uint32_t(kWorldSize) * tiling.num_tiles_total;

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((flags_len + kThreads - 1) / kThreads);
  if (blocks == 0) {
    // Ensure CTA0/thread0 runs scalar init even if flags_len==0.
    blocks = 1;
  }

  std::array<RingAllreduce2GpuAtomics, kWorldSize> atomics{};
  std::array<float*, kWorldSize> device_data{};
  std::array<uint32_t*, kWorldSize> device_status{};

  std::array<cudaStream_t, kWorldSize> streams{};
  std::array<cudaEvent_t, kWorldSize> done_events{};

  std::vector<float> host_in0(options.count);
  std::vector<float> host_in1(options.count);

  for (uint64_t i = 0; i < options.count; ++i) {
    host_in0[i] = static_cast<float>(i);
    host_in1[i] = static_cast<float>(2.0f * static_cast<float>(i));
  }

  for (int rank = 0; rank < kWorldSize; ++rank) {
    CUDA_CHECK(cudaSetDevice(devices[rank]));

    CUDA_CHECK(cudaStreamCreate(&streams[rank]));
    CUDA_CHECK(cudaEventCreateWithFlags(&done_events[rank], cudaEventDisableTiming));

    atomics[rank].flags_len = flags_len;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_data[rank]), sizeof(float) * options.count));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_status[rank]), sizeof(uint32_t)));

    CUDA_CHECK(cudaMemsetAsync(device_status[rank], 0xFF, sizeof(uint32_t), streams[rank]));

    auto const& host_in = (rank == 0) ? host_in0 : host_in1;
    CUDA_CHECK(cudaMemcpyAsync(
        device_data[rank],
        host_in.data(),
        sizeof(float) * options.count,
        cudaMemcpyHostToDevice,
        streams[rank]));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_rs_ready), sizeof(RingAllreduceSystemAtomicU32) * flags_len));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_ag_ready), sizeof(RingAllreduceSystemAtomicU32) * flags_len));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_abort), sizeof(RingAllreduceSystemAtomicU32)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_error), sizeof(RingAllreduceSystemAtomicU32)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_tiles_finished), sizeof(RingAllreduceDeviceAtomicU32)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_barrier_gather_token), sizeof(RingAllreduceSystemAtomicU32)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_barrier_gather_status), sizeof(RingAllreduceSystemAtomicU32)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_barrier_release_token), sizeof(RingAllreduceSystemAtomicU32)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_barrier_release_status), sizeof(RingAllreduceSystemAtomicU32)));

    construct_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[rank]>>>(atomics[rank]);
    CUDA_CHECK(cudaGetLastError());

    reset_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[rank]>>>(atomics[rank], options.epoch);
    CUDA_CHECK(cudaGetLastError());
  }

  // Ensure per-rank initialization is complete before any cross-device accesses.
  for (int rank = 0; rank < kWorldSize; ++rank) {
    CUDA_CHECK(cudaSetDevice(devices[rank]));
    CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
  }

  for (int rank = 0; rank < kWorldSize; ++rank) {
    RingAllreduceParams<float, 8> p{};

    p.world_size = kWorldSize;
    p.rank = rank;
    p.epoch = options.epoch;

    p.count = options.count;
    p.num_channels = options.num_channels;

    p.tile_elems = tiling.tile_elems;
    p.num_chunks_total = tiling.num_chunks_total;
    p.max_chunk_elems = tiling.max_chunk_elems;
    p.tiles_per_chunk = tiling.tiles_per_chunk;
    p.num_tiles_total = tiling.num_tiles_total;

    // Hang-resistant defaults.
    p.timeout_iters = 1u << 18;
    p.timeout_cycles = 0;
    p.poll_sleep_start = 0;
    p.poll_sleep_ns = 0;

    p.self_data = device_data[rank];

    p.self_rs_ready = atomics[rank].self_rs_ready;
    p.self_ag_ready = atomics[rank].self_ag_ready;
    p.self_abort = atomics[rank].self_abort;
    p.self_error = atomics[rank].self_error;

    p.self_tiles_finished = atomics[rank].self_tiles_finished;

    p.self_barrier_gather_token = atomics[rank].self_barrier_gather_token;
    p.self_barrier_gather_status = atomics[rank].self_barrier_gather_status;
    p.self_barrier_release_token = atomics[rank].self_barrier_release_token;
    p.self_barrier_release_status = atomics[rank].self_barrier_release_status;

    for (int peer = 0; peer < kWorldSize; ++peer) {
      p.peer_data[peer] = device_data[peer];
      p.peer_rs_ready[peer] = atomics[peer].self_rs_ready;
      p.peer_ag_ready[peer] = atomics[peer].self_ag_ready;
      p.peer_abort[peer] = atomics[peer].self_abort;

      p.peer_barrier_gather_token[peer] = atomics[peer].self_barrier_gather_token;
      p.peer_barrier_gather_status[peer] = atomics[peer].self_barrier_gather_status;
      p.peer_barrier_release_token[peer] = atomics[peer].self_barrier_release_token;
      p.peer_barrier_release_status[peer] = atomics[peer].self_barrier_release_status;
    }

    if (do_abort_injection) {
      p.debug_abort_rank = options.abort_rank;
      p.debug_abort_ag_step = 0u;
      p.debug_abort_before_ag_publish = options.abort_before_ag_publish ? 1u : 0u;
      p.debug_abort_after_ag_publish = options.abort_after_ag_publish ? 1u : 0u;
    }
    else {
      p.debug_abort_rank = 0xffff'ffffu;
      p.debug_abort_ag_step = 0u;
      p.debug_abort_before_ag_publish = 0u;
      p.debug_abort_after_ag_publish = 0u;
    }

    p.debug_release_delay_rank = 0xffff'ffffu;
    p.debug_release_delay_iters = 0u;

    p.debug_jitter_seed = 0u;
    p.debug_jitter_max_iters = 0u;
    p.debug_jitter_mask = 0u;

    CUDA_CHECK(cudaSetDevice(devices[rank]));

    cutlass::distributed::collective::ring_allreduce_sm100<float><<<tiling.num_tiles_total, 256, 0, streams[rank]>>>(
        p,
        device_status[rank]);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(done_events[rank], streams[rank]));
  }

  // Host watchdog: ensure no hangs.
  auto const host_timeout = std::chrono::milliseconds(options.watchdog_ms);
  auto deadline = std::chrono::steady_clock::now() + host_timeout;

  std::vector<bool> done(kWorldSize, false);
  while (true) {
    bool all_done = true;

    for (int rank = 0; rank < kWorldSize; ++rank) {
      if (done[rank]) {
        continue;
      }

      all_done = false;

      CUDA_CHECK(cudaSetDevice(devices[rank]));
      cudaError_t q = cudaEventQuery(done_events[rank]);
      if (q == cudaSuccess) {
        done[rank] = true;
        continue;
      }
      if (q != cudaErrorNotReady) {
        std::cerr << "cudaEventQuery failed on rank " << rank << ": " << cudaGetErrorString(q) << "\n";
        return 1;
      }
    }

    if (all_done) {
      break;
    }

    if (std::chrono::steady_clock::now() > deadline) {
      std::fprintf(stderr, "94_blackwell_ring_allreduce: watchdog timeout (world_size=%d)\n", kWorldSize);
      std::abort();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  std::array<uint32_t, kWorldSize> host_status{};
  for (int rank = 0; rank < kWorldSize; ++rank) {
    CUDA_CHECK(cudaSetDevice(devices[rank]));
    CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
    CUDA_CHECK(cudaMemcpy(&host_status[rank], device_status[rank], sizeof(uint32_t), cudaMemcpyDeviceToHost));
  }

  RingAllreduceError first_error = static_cast<RingAllreduceError>(host_status[0]);
  bool all_ok = true;
  bool all_non_ok = true;
  bool all_same = true;

  for (int rank = 0; rank < kWorldSize; ++rank) {
    auto e = static_cast<RingAllreduceError>(host_status[rank]);
    std::cout << "rank" << rank << " status: " << ring_allreduce_error_to_string(e) << " (" << host_status[rank] << ")\n";
    all_ok = all_ok && (e == RingAllreduceError::kOk);
    all_non_ok = all_non_ok && (e != RingAllreduceError::kOk);
    all_same = all_same && (e == first_error);
  }

  if (do_abort_injection) {
    if (!all_non_ok) {
      std::cerr << "status coherence violated: at least one rank returned kOk under abort injection\n";
      return 1;
    }
    if (!all_same) {
      std::cerr << "status coherence violated: ranks returned different error codes under abort injection\n";
      return 1;
    }

    std::cout << "Abort-injection run completed (payload is unspecified on non-kOk).\n";
  }
  else {
    if (!all_ok) {
      std::cerr << "unexpected non-kOk status in the no-injection path\n";
      return 1;
    }

    std::vector<float> host_out0(options.count);
    std::vector<float> host_out1(options.count);

    CUDA_CHECK(cudaSetDevice(devices[0]));
    CUDA_CHECK(cudaMemcpy(host_out0.data(), device_data[0], sizeof(float) * options.count, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaSetDevice(devices[1]));
    CUDA_CHECK(cudaMemcpy(host_out1.data(), device_data[1], sizeof(float) * options.count, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (uint64_t i = 0; i < options.count; ++i) {
      float expected = static_cast<float>(3.0f * static_cast<float>(i));
      float e0 = host_out0[i];
      float e1 = host_out1[i];

      float tol = 1e-5f * (1.0f + std::fabs(expected));
      if (std::fabs(e0 - expected) > tol || std::fabs(e1 - expected) > tol) {
        std::cerr << "mismatch at i=" << i << ": got [" << e0 << ", " << e1 << "] expected " << expected << "\n";
        correct = false;
        break;
      }
    }

    if (!correct) {
      std::cerr << "Verification failed\n";
      return 1;
    }

    std::cout << "Verification passed (allreduce sum correct).\n";
  }

  // Cleanup.
  for (int rank = 0; rank < kWorldSize; ++rank) {
    CUDA_CHECK(cudaSetDevice(devices[rank]));

    destroy_2gpu_atomics_kernel<<<blocks, kThreads, 0, streams[rank]>>>(atomics[rank]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(streams[rank]));

    CUDA_CHECK(cudaFree(atomics[rank].self_barrier_release_status));
    CUDA_CHECK(cudaFree(atomics[rank].self_barrier_release_token));
    CUDA_CHECK(cudaFree(atomics[rank].self_barrier_gather_status));
    CUDA_CHECK(cudaFree(atomics[rank].self_barrier_gather_token));

    CUDA_CHECK(cudaFree(atomics[rank].self_tiles_finished));

    CUDA_CHECK(cudaFree(atomics[rank].self_error));
    CUDA_CHECK(cudaFree(atomics[rank].self_abort));

    CUDA_CHECK(cudaFree(atomics[rank].self_ag_ready));
    CUDA_CHECK(cudaFree(atomics[rank].self_rs_ready));

    CUDA_CHECK(cudaFree(device_status[rank]));
    CUDA_CHECK(cudaFree(device_data[rank]));

    CUDA_CHECK(cudaEventDestroy(done_events[rank]));
    CUDA_CHECK(cudaStreamDestroy(streams[rank]));
  }

  return 0;
}
