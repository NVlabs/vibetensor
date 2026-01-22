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
  \brief Bandwidth benchmark for the Blackwell ring allreduce prototype.

  This benchmark launches one ring_allreduce_sm100 kernel per device and measures
  kernel-only latency using CUDA events. It reports both algorithmic bandwidth
  and NCCL-style bus bandwidth.

  Requirements:

    - >=2 CUDA devices (SM100 or SM103)
    - P2P access + native peer atomics along the selected ring neighbors

  Example:

    $ ./94_blackwell_ring_allreduce_benchmark \
        --world_sizes=2,4,8 \
        --sizes=4KiB,16KiB,64KiB,256KiB,1MiB,4MiB,16MiB,64MiB,256MiB \
        --num_channels=1 --tile_elems=256 --warmup_iters=10 --measure_iters=50

  Notes:

    - By default, kernel timeouts are disabled (best performance); use
      --timeouts_enabled to enable hang-resistant timeouts.
*/

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
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
using cutlass::distributed::collective::RingAllreduceP2POptions;
using cutlass::distributed::collective::RingAllreduceParams;
using cutlass::distributed::collective::RingAllreduceSystemAtomicU32;
using cutlass::distributed::collective::RingAllreduceTiling;
using cutlass::distributed::collective::validate_ring_allreduce_host_tiling;
using cutlass::distributed::collective::validate_ring_p2p_caps_and_enable_peer_access;

constexpr int kMaxWorldSize = 8;

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

struct SizeSpec {
  uint64_t bytes_per_rank = 0;
  uint64_t count_elems = 0;
  std::string token;
};

static std::string trim_ascii(std::string s) {
  auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };

  while (!s.empty() && is_space(static_cast<unsigned char>(s.front()))) {
    s.erase(s.begin());
  }
  while (!s.empty() && is_space(static_cast<unsigned char>(s.back()))) {
    s.pop_back();
  }
  return s;
}

static std::string to_lower_ascii(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

static bool ends_with(std::string const& s, std::string const& suffix) {
  if (suffix.size() > s.size()) {
    return false;
  }
  return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool parse_u64(std::string const& s, uint64_t* out) {
  if (!out) {
    return false;
  }
  if (s.empty()) {
    return false;
  }

  // Disallow leading '+'/'-'.
  if (s[0] == '+' || s[0] == '-') {
    return false;
  }

  char* end = nullptr;
  errno = 0;
  unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (errno != 0 || end == s.c_str() || *end != '\0') {
    return false;
  }
  *out = static_cast<uint64_t>(v);
  return true;
}

static std::vector<std::string> split_csv(std::string const& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ',') {
      out.push_back(cur);
      cur.clear();
    }
    else {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

static std::string format_bytes_binary(uint64_t bytes) {
  struct Unit {
    char const* suffix;
    uint64_t denom;
  };

  static Unit const units[] = {
    {"GiB", 1024ull * 1024ull * 1024ull},
    {"MiB", 1024ull * 1024ull},
    {"KiB", 1024ull},
    {"B", 1ull},
  };

  for (auto const& u : units) {
    if (bytes >= u.denom && (bytes % u.denom == 0)) {
      std::ostringstream oss;
      oss << (bytes / u.denom) << u.suffix;
      return oss.str();
    }
  }

  // Fallback: non-integral unit.
  std::ostringstream oss;
  if (bytes >= 1024ull * 1024ull * 1024ull) {
    oss << std::fixed << std::setprecision(2)
        << (static_cast<double>(bytes) / static_cast<double>(1024ull * 1024ull * 1024ull)) << "GiB";
  }
  else if (bytes >= 1024ull * 1024ull) {
    oss << std::fixed << std::setprecision(2)
        << (static_cast<double>(bytes) / static_cast<double>(1024ull * 1024ull)) << "MiB";
  }
  else if (bytes >= 1024ull) {
    oss << std::fixed << std::setprecision(2)
        << (static_cast<double>(bytes) / static_cast<double>(1024ull)) << "KiB";
  }
  else {
    oss << bytes << "B";
  }
  return oss.str();
}

struct Options {
  bool help = false;
  bool error = false;

  std::vector<int> world_sizes = {2, 4, 8};

  std::string sizes_raw = "4KiB,16KiB,64KiB,256KiB,1MiB,4MiB,16MiB,64MiB,256MiB";

  int32_t num_channels = 1;
  uint32_t tile_elems = 256;

  int warmup_iters = 10;
  int measure_iters = 50;

  bool verify = false;
  uint64_t verify_seed = 1u;
  int verify_samples = 4096;

  std::vector<int> devices;

  bool timeouts_enabled = false;
  int watchdog_ms = 20'000;

  bool verbose = false;

  std::string csv_path;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_arguments("world_sizes", world_sizes);
    cmd.get_cmd_line_argument("sizes", sizes_raw, sizes_raw);

    cmd.get_cmd_line_argument("num_channels", num_channels, int32_t(1));
    cmd.get_cmd_line_argument("tile_elems", tile_elems, uint32_t(256));

    cmd.get_cmd_line_argument("warmup_iters", warmup_iters, 10);
    cmd.get_cmd_line_argument("measure_iters", measure_iters, 50);

    verify = cmd.check_cmd_line_flag("verify");
    cmd.get_cmd_line_argument("verify_seed", verify_seed, uint64_t(1));
    cmd.get_cmd_line_argument("verify_samples", verify_samples, 4096);

    cmd.get_cmd_line_arguments("devices", devices);

    timeouts_enabled = cmd.check_cmd_line_flag("timeouts_enabled");
    verbose = cmd.check_cmd_line_flag("verbose");

    cmd.get_cmd_line_argument("watchdog_ms", watchdog_ms, 20'000);

    cmd.get_cmd_line_argument("csv", csv_path, std::string{});

    // Validate numeric args.
    if (world_sizes.empty()) {
      std::cerr << "world_sizes must be non-empty\n";
      error = true;
    }

    for (int ws : world_sizes) {
      if (!(ws == 1 || ws == 2 || ws == 4 || ws == 8)) {
        std::cerr << "world_sizes entries must be in {1,2,4,8}\n";
        error = true;
        break;
      }
    }

    if (num_channels <= 0) {
      std::cerr << "num_channels must be > 0\n";
      error = true;
    }

    if (tile_elems == 0) {
      std::cerr << "tile_elems must be > 0\n";
      error = true;
    }

    if (warmup_iters < 0) {
      std::cerr << "warmup_iters must be >= 0\n";
      error = true;
    }

    if (measure_iters <= 0) {
      std::cerr << "measure_iters must be > 0\n";
      error = true;
    }

    if (verify_samples <= 0) {
      std::cerr << "verify_samples must be > 0\n";
      error = true;
    }

    if (watchdog_ms <= 0) {
      std::cerr << "watchdog_ms must be > 0\n";
      error = true;
    }
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "94_blackwell_ring_allreduce_benchmark\n\n"
        << "  Bandwidth benchmark for Blackwell ring allreduce (SM100/SM103).\n\n"
        << "Options:\n\n"
        << "  --help                        Display this usage statement\n"
        << "  --world_sizes=<csv>            World sizes to benchmark (default: 2,4,8)\n"
        << "  --sizes=<csv>                  Tensor sizes per rank (default: 4KiB,...,256MiB)\n"
        << "                                Units: B/KiB/MiB/GiB, or element forms like 262144e\n"
        << "  --num_channels=<int>           Number of channels (default: 1)\n"
        << "  --tile_elems=<int>             Tile size in elements (default: 256; <=256 for warp-specialized SMEM)\n"
        << "  --warmup_iters=<int>           Warmup iterations (default: 10)\n"
        << "  --measure_iters=<int>          Measured iterations (default: 50)\n"
        << "  --verify                      Verify numerical correctness (random inputs)\n"
        << "  --verify_seed=<int>            RNG seed for --verify (default: 1)\n"
        << "  --verify_samples=<int>         Number of elements to spot-check (default: 4096)\n"
        << "  --timeouts_enabled             Enable hang-resistant timeouts (slower)\n"
        << "  --devices=<csv>                Optional explicit device ring order; prefix used per world size\n"
        << "  --watchdog_ms=<int>            Host watchdog timeout (default: 20000)\n"
        << "  --csv=<path>                   Optional CSV output path (append; header if new/empty)\n"
        << "  --verbose                      Print extra diagnostics\n";

    return out;
  }
};

static bool is_sm100_or_sm103(int device) {
  cudaDeviceProp prop{};
  cudaError_t st = cudaGetDeviceProperties(&prop, device);
  if (st != cudaSuccess) {
    return false;
  }

  int cc = prop.major * 10 + prop.minor;
  return cc == 100 || cc == 103;
}

__global__ void ring_allreduce_probe_kernel() {}

static cudaError_t ring_allreduce_probe_launch(int device) {
  // Clear any pre-existing per-thread CUDA error state.
  (void)cudaGetLastError();

  cudaError_t st = cudaSetDevice(device);
  if (st != cudaSuccess) {
    return st;
  }

  ring_allreduce_probe_kernel<<<1, 1>>>();
  return cudaGetLastError();
}

static bool device_has_kernel_image(int device) {
  cudaError_t probe_st = ring_allreduce_probe_launch(device);

  if (probe_st == cudaErrorNoKernelImageForDevice || probe_st == cudaErrorInvalidDeviceFunction) {
    return false;
  }

  if (probe_st != cudaSuccess) {
    std::cerr << "probe launch failed on device " << device << ": " << cudaGetErrorString(probe_st) << "\n";
    return false;
  }

  return true;
}

static bool device_is_eligible(int device) {
  return is_sm100_or_sm103(device) && device_has_kernel_image(device);
}

static std::vector<int> enumerate_eligible_devices() {
  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(st) << "\n";
    return {};
  }

  std::vector<int> eligible;
  for (int dev = 0; dev < device_count; ++dev) {
    if (device_is_eligible(dev)) {
      eligible.push_back(dev);
    }
  }

  std::sort(eligible.begin(), eligible.end());
  return eligible;
}

static bool validate_devices_unique_and_in_range(std::vector<int> const& devices) {
  int device_count = 0;
  cudaError_t st = cudaGetDeviceCount(&device_count);
  if (st != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(st) << "\n";
    return false;
  }

  for (size_t i = 0; i < devices.size(); ++i) {
    int di = devices[i];
    if (di < 0 || di >= device_count) {
      std::cerr << "device id out of range: " << di << "\n";
      return false;
    }

    for (size_t j = i + 1; j < devices.size(); ++j) {
      if (di == devices[j]) {
        std::cerr << "device list contains duplicates\n";
        return false;
      }
    }
  }

  return true;
}

static bool parse_sizes_float(Options const& options, std::vector<SizeSpec>* out_sizes) {
  if (!out_sizes) {
    return false;
  }

  out_sizes->clear();

  // Only float is supported for now.
  constexpr uint64_t kElemSize = sizeof(float);

  std::vector<std::string> tokens = split_csv(options.sizes_raw);
  for (auto const& raw : tokens) {
    std::string tok = trim_ascii(raw);
    if (tok.empty()) {
      std::cerr << "sizes contains an empty token\n";
      return false;
    }

    std::string lower = to_lower_ascii(tok);

    // Element-count suffixes.
    std::string elem_suffix;
    if (ends_with(lower, "elements")) elem_suffix = "elements";
    else if (ends_with(lower, "element")) elem_suffix = "element";
    else if (ends_with(lower, "elems")) elem_suffix = "elems";
    else if (ends_with(lower, "elem")) elem_suffix = "elem";
    else if (ends_with(lower, "e")) elem_suffix = "e";

    SizeSpec spec;
    spec.token = tok;

    if (!elem_suffix.empty()) {
      std::string num = lower.substr(0, lower.size() - elem_suffix.size());
      num = trim_ascii(num);

      uint64_t count = 0;
      if (!parse_u64(num, &count) || count == 0) {
        std::cerr << "invalid element-count size token: '" << tok << "'\n";
        return false;
      }

      spec.count_elems = count;
      spec.bytes_per_rank = count * kElemSize;
    }
    else {
      // Bytes with optional units.
      uint64_t mul = 1;
      std::string num = lower;

      auto take_suffix = [&](std::string const& suf, uint64_t factor) {
        if (ends_with(num, suf)) {
          mul = factor;
          num = num.substr(0, num.size() - suf.size());
          num = trim_ascii(num);
          return true;
        }
        return false;
      };

      // Note: order matters (kib before b).
      if (take_suffix("kib", 1024ull) || take_suffix("k", 1024ull) ||
          take_suffix("mib", 1024ull * 1024ull) || take_suffix("m", 1024ull * 1024ull) ||
          take_suffix("gib", 1024ull * 1024ull * 1024ull) || take_suffix("g", 1024ull * 1024ull * 1024ull) ||
          take_suffix("b", 1ull)) {
        // parsed
      }

      uint64_t base = 0;
      if (!parse_u64(num, &base) || base == 0) {
        std::cerr << "invalid byte size token: '" << tok << "'\n";
        return false;
      }

      // Overflow check.
      if (base > (std::numeric_limits<uint64_t>::max() / mul)) {
        std::cerr << "byte size overflows uint64: '" << tok << "'\n";
        return false;
      }

      spec.bytes_per_rank = base * mul;
      if (spec.bytes_per_rank == 0) {
        std::cerr << "byte size must be > 0: '" << tok << "'\n";
        return false;
      }

      if (spec.bytes_per_rank % kElemSize != 0) {
        std::cerr << "byte size is not divisible by sizeof(float): '" << tok << "'\n";
        return false;
      }

      spec.count_elems = spec.bytes_per_rank / kElemSize;
      if (spec.count_elems == 0) {
        std::cerr << "element count must be > 0: '" << tok << "'\n";
        return false;
      }
    }

    out_sizes->push_back(spec);
  }

  return true;
}

CUTLASS_HOST_DEVICE uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

CUTLASS_HOST_DEVICE float random_value_float(uint64_t seed, uint32_t rank, uint64_t idx) {
  uint64_t x = seed;
  x ^= (static_cast<uint64_t>(rank) + 1ull) * 0x9e3779b97f4a7c15ull;
  x ^= idx * 0xbf58476d1ce4e5b9ull;
  x = splitmix64(x);

  // Take the top 24 bits and map to [-1, 1).
  uint32_t u24 = static_cast<uint32_t>(x >> 40);
  float f01 = static_cast<float>(u24) * (1.0f / 16777216.0f); // 2^24
  return f01 * 2.0f - 1.0f;
}

__global__ void fill_random_float_kernel(float* data, uint64_t count, uint64_t seed, uint32_t rank) {
  uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

  for (uint64_t i = tid; i < count; i += stride) {
    data[i] = random_value_float(seed, rank, i);
  }
}

__global__ void gather_samples_float_kernel(float const* data, uint64_t const* indices, float* out, uint32_t n) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    uint64_t idx = indices[tid];
    out[tid] = data[idx];
  }
}

static bool find_ring_auto(
    int world_size,
    std::vector<int> const& eligible_devices,
    std::vector<int>* out_ring,
    bool verbose) {

  if (!out_ring) {
    return false;
  }

  out_ring->clear();

  if (world_size <= 0) {
    return false;
  }

  if (static_cast<int>(eligible_devices.size()) < world_size) {
    return false;
  }

  // Precompute pairwise connectivity using dry-run P2P checks (no peer-access side effects).
  std::unordered_map<int, int> idx;
  idx.reserve(eligible_devices.size());
  for (int i = 0; i < static_cast<int>(eligible_devices.size()); ++i) {
    idx[eligible_devices[i]] = i;
  }

  std::vector<std::vector<bool>> ok(eligible_devices.size(), std::vector<bool>(eligible_devices.size(), false));

  RingAllreduceP2POptions opts;
  opts.enable_peer_access = false;
  opts.require_native_atomics = true;

  for (size_t i = 0; i < eligible_devices.size(); ++i) {
    for (size_t j = 0; j < eligible_devices.size(); ++j) {
      if (i == j) {
        continue;
      }
      int a = eligible_devices[i];
      int b = eligible_devices[j];
      int pair[2] = {a, b};

      auto r = validate_ring_p2p_caps_and_enable_peer_access(2, pair, opts);
      ok[i][j] = r.ok();
    }
  }

  // Deterministic DFS for a Hamiltonian cycle of length world_size.
  // Try start devices in ascending order; restrict pool to devices >= start so
  // start is the minimal ID in the selected ring.
  auto dfs = [&](auto&& self,
                 std::vector<int> const& pool,
                 std::vector<int> const& pool_to_global,
                 std::vector<int>& path,
                 std::vector<bool>& used) -> bool {
    if (static_cast<int>(path.size()) == world_size) {
      int first = path.front();
      int last = path.back();
      int gi = idx[last];
      int gj = idx[first];
      return ok[gi][gj];
    }

    int last = path.back();
    int last_g = idx[last];

    for (size_t j = 0; j < pool.size(); ++j) {
      if (used[j]) {
        continue;
      }

      int cand = pool[j];
      int cand_g = pool_to_global[j];
      if (!ok[last_g][cand_g]) {
        continue;
      }

      used[j] = true;
      path.push_back(cand);

      if (self(self, pool, pool_to_global, path, used)) {
        return true;
      }

      path.pop_back();
      used[j] = false;
    }

    return false;
  };

  for (size_t start_pos = 0; start_pos < eligible_devices.size(); ++start_pos) {
    if (eligible_devices.size() - start_pos < static_cast<size_t>(world_size)) {
      break;
    }

    std::vector<int> pool;
    pool.reserve(eligible_devices.size() - start_pos);

    std::vector<int> pool_to_global;
    pool_to_global.reserve(eligible_devices.size() - start_pos);

    for (size_t i = start_pos; i < eligible_devices.size(); ++i) {
      pool.push_back(eligible_devices[i]);
      pool_to_global.push_back(static_cast<int>(i));
    }

    std::vector<int> path;
    path.reserve(world_size);
    path.push_back(pool[0]);

    std::vector<bool> used(pool.size(), false);
    used[0] = true;

    if (dfs(dfs, pool, pool_to_global, path, used)) {
      *out_ring = path;
      if (verbose) {
        std::cerr << "auto-selected ring for world_size=" << world_size << ":";
        for (int d : *out_ring) {
          std::cerr << " " << d;
        }
        std::cerr << "\n";
      }
      return true;
    }
  }

  return false;
}

static void wait_for_events_or_die(
    std::vector<int> const& devices,
    std::vector<cudaEvent_t> const& done_events,
    int watchdog_ms,
    char const* context) {

  int world_size = static_cast<int>(devices.size());
  auto const host_timeout = std::chrono::milliseconds(watchdog_ms);
  auto deadline = std::chrono::steady_clock::now() + host_timeout;

  std::vector<bool> done(world_size, false);
  while (true) {
    bool all_done = true;

    for (int rank = 0; rank < world_size; ++rank) {
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
        std::_Exit(2);
      }
    }

    if (all_done) {
      break;
    }

    if (std::chrono::steady_clock::now() > deadline) {
      std::fprintf(stderr, "94_blackwell_ring_allreduce_benchmark: watchdog timeout (%s, world_size=%d)\n",
                   context ? context : "unknown", world_size);
      std::fflush(stderr);
      std::_Exit(2);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

struct RingAllreduceRankAtomics {
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

__global__ void construct_rank_atomics_kernel(RingAllreduceRankAtomics a) {
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

__global__ void reset_rank_atomics_init_kernel(RingAllreduceRankAtomics a, uint32_t epoch) {
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

__global__ void destroy_rank_atomics_kernel(RingAllreduceRankAtomics a) {
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

static void print_table_header() {
  std::cout << std::left
            << std::setw(6) << "world"
            << std::setw(12) << "bytes"
            << std::setw(12) << "elems"
            << std::setw(14) << "result"
            << std::setw(12) << "avg_ms"
            << std::setw(14) << "algbw(GB/s)"
            << std::setw(14) << "busbw(GB/s)"
            << "devices"
            << "\n";
}

static void print_row(
    int world_size,
    SizeSpec const& spec,
    std::string const& result,
    double avg_ms,
    double algbw,
    double busbw,
    std::vector<int> const& devices) {

  std::ostringstream devs;
  devs << "[";
  for (size_t i = 0; i < devices.size(); ++i) {
    devs << devices[i];
    if (i + 1 < devices.size()) {
      devs << ",";
    }
  }
  devs << "]";

  std::cout << std::left
            << std::setw(6) << world_size
            << std::setw(12) << format_bytes_binary(spec.bytes_per_rank)
            << std::setw(12) << spec.count_elems
            << std::setw(14) << result;

  if (result == "OK") {
    std::cout << std::setw(12) << std::fixed << std::setprecision(3) << avg_ms
              << std::setw(14) << std::fixed << std::setprecision(2) << algbw
              << std::setw(14) << std::fixed << std::setprecision(2) << busbw;
  }
  else {
    std::cout << std::setw(12) << "-" << std::setw(14) << "-" << std::setw(14) << "-";
  }

  std::cout << devs.str() << "\n";
}

static void maybe_write_csv_header(std::ofstream& csv) {
  csv << "world_size,dtype,bytes,element_count,num_channels,tile_elems,warmup_iters,measure_iters,result,avg_ms,algbw_GBps,busbw_GBps,devices\n";
}

static void write_csv_row(
    std::ofstream& csv,
    int world_size,
    SizeSpec const& spec,
    Options const& options,
    std::string const& result,
    double avg_ms,
    double algbw,
    double busbw,
    std::vector<int> const& devices) {

  std::ostringstream devs;
  for (size_t i = 0; i < devices.size(); ++i) {
    devs << devices[i];
    if (i + 1 < devices.size()) {
      devs << ":";
    }
  }

  csv << world_size
      << ",float"
      << "," << spec.bytes_per_rank
      << "," << spec.count_elems
      << "," << options.num_channels
      << "," << options.tile_elems
      << "," << options.warmup_iters
      << "," << options.measure_iters
      << "," << result;

  if (result == "OK") {
    csv << "," << std::fixed << std::setprecision(6) << avg_ms
        << "," << std::fixed << std::setprecision(6) << algbw
        << "," << std::fixed << std::setprecision(6) << busbw;
  }
  else {
    csv << ",,,";
  }

  csv << "," << devs.str() << "\n";
  csv.flush();
}

static bool run_world_size_float(
    Options const& options,
    int world_size,
    std::vector<int> const& ring_devices,
    std::vector<SizeSpec> const& sizes,
    std::ofstream* csv_out) {

  // Streams and timing events (per rank, reused across sizes).
  std::vector<cudaStream_t> streams(world_size);
  std::vector<cudaEvent_t> start_events(world_size);
  std::vector<cudaEvent_t> stop_events(world_size);

  for (int rank = 0; rank < world_size; ++rank) {
    CUDA_CHECK(cudaSetDevice(ring_devices[rank]));
    CUDA_CHECK(cudaStreamCreate(&streams[rank]));
    CUDA_CHECK(cudaEventCreate(&start_events[rank]));
    CUDA_CHECK(cudaEventCreate(&stop_events[rank]));
  }

  bool all_ok = true;

  // Per-size allocations.
  for (auto const& spec : sizes) {

    RingAllreduceTiling tiling{};
    auto tiling_r = validate_ring_allreduce_host_tiling(
        spec.count_elems,
        world_size,
        options.num_channels,
        options.tile_elems,
        &tiling,
        ring_devices.data());

    if (!tiling_r.ok()) {
      if (options.verbose) {
        print_host_result("validate_ring_allreduce_host_tiling", tiling_r);
      }
      print_row(world_size, spec, "SKIP_TILING", 0.0, 0.0, 0.0, ring_devices);
      if (csv_out) {
        write_csv_row(*csv_out, world_size, spec, options, "SKIP_TILING", 0.0, 0.0, 0.0, ring_devices);
      }
      if (options.verify) {
        all_ok = false;
      }
      continue;
    }

    uint32_t flags_len = uint32_t(world_size) * tiling.num_tiles_total;

    constexpr int kThreads = 256;
    int blocks = static_cast<int>((flags_len + kThreads - 1) / kThreads);
    if (blocks == 0) {
      blocks = 1;
    }

    std::vector<RingAllreduceRankAtomics> atomics(world_size);
    std::vector<float*> device_data(world_size, nullptr);
    std::vector<uint32_t*> device_status(world_size, nullptr);

    bool alloc_ok = true;

    for (int rank = 0; rank < world_size; ++rank) {
      CUDA_CHECK(cudaSetDevice(ring_devices[rank]));

      atomics[rank].flags_len = flags_len;

      cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&device_data[rank]), spec.bytes_per_rank);
      if (st != cudaSuccess) {
        alloc_ok = false;
        break;
      }

      st = cudaMalloc(reinterpret_cast<void**>(&device_status[rank]), sizeof(uint32_t));
      if (st != cudaSuccess) {
        alloc_ok = false;
        break;
      }

      st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_rs_ready), sizeof(RingAllreduceSystemAtomicU32) * flags_len);
      if (st != cudaSuccess) {
        alloc_ok = false;
        break;
      }
      st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_ag_ready), sizeof(RingAllreduceSystemAtomicU32) * flags_len);
      if (st != cudaSuccess) {
        alloc_ok = false;
        break;
      }

      st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_abort), sizeof(RingAllreduceSystemAtomicU32));
      if (st != cudaSuccess) {
        alloc_ok = false;
        break;
      }
      st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_error), sizeof(RingAllreduceSystemAtomicU32));
      if (st != cudaSuccess) {
        alloc_ok = false;
        break;
      }

      st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_tiles_finished), sizeof(RingAllreduceDeviceAtomicU32));
      if (st != cudaSuccess) {
        alloc_ok = false;
        break;
      }

      st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_barrier_gather_token), sizeof(RingAllreduceSystemAtomicU32));
      if (st != cudaSuccess) {
        alloc_ok = false;
        break;
      }
      st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_barrier_gather_status), sizeof(RingAllreduceSystemAtomicU32));
      if (st != cudaSuccess) {
        alloc_ok = false;
        break;
      }
      st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_barrier_release_token), sizeof(RingAllreduceSystemAtomicU32));
      if (st != cudaSuccess) {
        alloc_ok = false;
        break;
      }
      st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_barrier_release_status), sizeof(RingAllreduceSystemAtomicU32));
      if (st != cudaSuccess) {
        alloc_ok = false;
        break;
      }

      // Initialize payload to 0.
      CUDA_CHECK(cudaMemsetAsync(device_data[rank], 0, spec.bytes_per_rank, streams[rank]));
      CUDA_CHECK(cudaMemsetAsync(device_status[rank], 0xFF, sizeof(uint32_t), streams[rank]));

      construct_rank_atomics_kernel<<<blocks, kThreads, 0, streams[rank]>>>(atomics[rank]);
      st = cudaGetLastError();
      if (st != cudaSuccess) {
        alloc_ok = false;
        break;
      }

      reset_rank_atomics_init_kernel<<<blocks, kThreads, 0, streams[rank]>>>(atomics[rank], 1u);
      st = cudaGetLastError();
      if (st != cudaSuccess) {
        alloc_ok = false;
        break;
      }
    }

    if (!alloc_ok) {
      // Best-effort cleanup of partial allocations.
      for (int rank = 0; rank < world_size; ++rank) {
        CUDA_CHECK(cudaSetDevice(ring_devices[rank]));

        if (streams[rank]) {
          (void)cudaStreamSynchronize(streams[rank]);
        }

        if (atomics[rank].self_barrier_release_status) (void)cudaFree(atomics[rank].self_barrier_release_status);
        if (atomics[rank].self_barrier_release_token) (void)cudaFree(atomics[rank].self_barrier_release_token);
        if (atomics[rank].self_barrier_gather_status) (void)cudaFree(atomics[rank].self_barrier_gather_status);
        if (atomics[rank].self_barrier_gather_token) (void)cudaFree(atomics[rank].self_barrier_gather_token);
        if (atomics[rank].self_tiles_finished) (void)cudaFree(atomics[rank].self_tiles_finished);
        if (atomics[rank].self_error) (void)cudaFree(atomics[rank].self_error);
        if (atomics[rank].self_abort) (void)cudaFree(atomics[rank].self_abort);
        if (atomics[rank].self_ag_ready) (void)cudaFree(atomics[rank].self_ag_ready);
        if (atomics[rank].self_rs_ready) (void)cudaFree(atomics[rank].self_rs_ready);
        if (device_status[rank]) (void)cudaFree(device_status[rank]);
        if (device_data[rank]) (void)cudaFree(device_data[rank]);
      }

      print_row(world_size, spec, "SKIP_OOM", 0.0, 0.0, 0.0, ring_devices);
      if (csv_out) {
        write_csv_row(*csv_out, world_size, spec, options, "SKIP_OOM", 0.0, 0.0, 0.0, ring_devices);
      }
      if (options.verify) {
        all_ok = false;
      }
      continue;
    }

    // Ensure per-rank initialization is complete before any cross-device accesses.
    for (int rank = 0; rank < world_size; ++rank) {
      CUDA_CHECK(cudaSetDevice(ring_devices[rank]));
      CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
    }

    // Prepare params.
    std::vector<RingAllreduceParams<float, kMaxWorldSize>> params(world_size);

    for (int rank = 0; rank < world_size; ++rank) {
      RingAllreduceParams<float, kMaxWorldSize> p{};

      p.world_size = world_size;
      p.rank = rank;
      p.epoch = 1u;

      p.count = spec.count_elems;
      p.num_channels = options.num_channels;

      p.tile_elems = tiling.tile_elems;
      p.num_chunks_total = tiling.num_chunks_total;
      p.max_chunk_elems = tiling.max_chunk_elems;
      p.tiles_per_chunk = tiling.tiles_per_chunk;
      p.num_tiles_total = tiling.num_tiles_total;

      if (options.timeouts_enabled) {
        // Hang-resistant defaults (same as sample).
        p.timeout_iters = 1u << 18;
        p.timeout_cycles = 0;
        p.poll_sleep_start = 0;
        p.poll_sleep_ns = 0;
      }
      else {
        // Best-performance default: allow overlap/prefetch paths.
        p.timeout_iters = 0;
        p.timeout_cycles = 0;
        p.poll_sleep_start = 0;
        p.poll_sleep_ns = 0;
      }

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

      for (int peer = 0; peer < world_size; ++peer) {
        p.peer_data[peer] = device_data[peer];
        p.peer_rs_ready[peer] = atomics[peer].self_rs_ready;
        p.peer_ag_ready[peer] = atomics[peer].self_ag_ready;
        p.peer_abort[peer] = atomics[peer].self_abort;

        p.peer_barrier_gather_token[peer] = atomics[peer].self_barrier_gather_token;
        p.peer_barrier_gather_status[peer] = atomics[peer].self_barrier_gather_status;
        p.peer_barrier_release_token[peer] = atomics[peer].self_barrier_release_token;
        p.peer_barrier_release_status[peer] = atomics[peer].self_barrier_release_status;
      }

      // Debug hooks disabled.
      p.debug_abort_rank = 0xffff'ffffu;
      p.debug_abort_ag_step = 0u;
      p.debug_abort_before_ag_publish = 0u;
      p.debug_abort_after_ag_publish = 0u;

      p.debug_release_delay_rank = 0xffff'ffffu;
      p.debug_release_delay_iters = 0u;

      p.debug_jitter_seed = 0u;
      p.debug_jitter_max_iters = 0u;
      p.debug_jitter_mask = 0u;

      params[rank] = p;
    }

    auto reset_scalars = [&](int rank) {
      CUDA_CHECK(cudaSetDevice(ring_devices[rank]));

      CUDA_CHECK(cudaMemsetAsync(device_status[rank], 0xFF, sizeof(uint32_t), streams[rank]));

      CUDA_CHECK(cudaMemsetAsync(atomics[rank].self_abort, 0, sizeof(RingAllreduceSystemAtomicU32), streams[rank]));
      CUDA_CHECK(cudaMemsetAsync(atomics[rank].self_error, 0, sizeof(RingAllreduceSystemAtomicU32), streams[rank]));
      CUDA_CHECK(cudaMemsetAsync(atomics[rank].self_tiles_finished, 0, sizeof(RingAllreduceDeviceAtomicU32), streams[rank]));

      // Barrier state is epoch-tagged; reset to 0 for cleanliness.
      CUDA_CHECK(cudaMemsetAsync(atomics[rank].self_barrier_gather_token, 0, sizeof(RingAllreduceSystemAtomicU32), streams[rank]));
      CUDA_CHECK(cudaMemsetAsync(atomics[rank].self_barrier_gather_status, 0, sizeof(RingAllreduceSystemAtomicU32), streams[rank]));
      CUDA_CHECK(cudaMemsetAsync(atomics[rank].self_barrier_release_token, 0, sizeof(RingAllreduceSystemAtomicU32), streams[rank]));
      CUDA_CHECK(cudaMemsetAsync(atomics[rank].self_barrier_release_status, 0, sizeof(RingAllreduceSystemAtomicU32), streams[rank]));
    };

    uint32_t epoch = 1u;
    bool ok_status = true;
    bool verify_ok = true;

    auto make_verify_seed = [&]() -> uint64_t {
      uint64_t seed = options.verify_seed;
      seed ^= static_cast<uint64_t>(world_size) * 0x9e3779b97f4a7c15ull;
      seed ^= spec.bytes_per_rank * 0xbf58476d1ce4e5b9ull;
      seed ^= static_cast<uint64_t>(options.num_channels) * 0x94d049bb133111ebull;
      seed ^= static_cast<uint64_t>(options.tile_elems) << 32;
      return seed;
    };

    auto verify_payload = [&](uint32_t verify_epoch) -> bool {
      uint64_t seed = make_verify_seed();
      uint64_t count = spec.count_elems;

      int sample_count = options.verify_samples;
      if (static_cast<uint64_t>(sample_count) > count) {
        sample_count = static_cast<int>(count);
      }
      if (sample_count <= 0) {
        return true;
      }

      std::vector<uint64_t> sample_indices(sample_count);
      if (static_cast<uint64_t>(sample_count) == count) {
        for (int i = 0; i < sample_count; ++i) {
          sample_indices[i] = static_cast<uint64_t>(i);
        }
      }
      else {
        sample_indices[0] = 0;
        if (sample_count > 1) {
          sample_indices[1] = count - 1;
        }
        uint64_t state = seed ^ 0xd2b74407b1ce6e93ull;
        for (int i = 2; i < sample_count; ++i) {
          state = splitmix64(state + static_cast<uint64_t>(i));
          sample_indices[i] = state % count;
        }
      }

      std::vector<float> expected(sample_count);
      for (int i = 0; i < sample_count; ++i) {
        uint64_t idx = sample_indices[i];
        double acc = 0.0;
        for (int r = 0; r < world_size; ++r) {
          acc += static_cast<double>(random_value_float(seed, static_cast<uint32_t>(r), idx));
        }
        expected[i] = static_cast<float>(acc);
      }

      int threads = 256;
      uint64_t blocks_u64 = (count + static_cast<uint64_t>(threads) - 1) / static_cast<uint64_t>(threads);
      int blocks_fill = static_cast<int>(std::min<uint64_t>(blocks_u64, 4096ull));
      if (blocks_fill == 0) {
        blocks_fill = 1;
      }

      // Fill random inputs on each device.
      for (int rank = 0; rank < world_size; ++rank) {
        CUDA_CHECK(cudaSetDevice(ring_devices[rank]));
        fill_random_float_kernel<<<blocks_fill, threads, 0, streams[rank]>>>(
            device_data[rank], count, seed, static_cast<uint32_t>(rank));
        CUDA_CHECK(cudaGetLastError());
      }

      // Run a single allreduce.
      for (int rank = 0; rank < world_size; ++rank) {
        reset_scalars(rank);
        params[rank].epoch = verify_epoch;

        cutlass::distributed::collective::ring_allreduce_sm100<float><<<tiling.num_tiles_total, 256, 0, streams[rank]>>>(
            params[rank],
            device_status[rank]);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_events[rank], streams[rank]));
      }

      wait_for_events_or_die(ring_devices, stop_events, options.watchdog_ms, "verify");

      // Status must be OK.
      for (int rank = 0; rank < world_size; ++rank) {
        uint32_t st = 0;
        CUDA_CHECK(cudaSetDevice(ring_devices[rank]));
        CUDA_CHECK(cudaMemcpy(&st, device_status[rank], sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (static_cast<RingAllreduceError>(st) != RingAllreduceError::kOk) {
          if (options.verbose) {
            std::cerr << "verify status rank" << rank << ": "
                      << ring_allreduce_error_to_string(static_cast<RingAllreduceError>(st))
                      << " (" << st << ")\n";
          }
          return false;
        }
      }

      // Spot-check payload values.
      std::vector<float> got(sample_count);
      int blocks_gather = (sample_count + threads - 1) / threads;

      for (int rank = 0; rank < world_size; ++rank) {
        CUDA_CHECK(cudaSetDevice(ring_devices[rank]));

        uint64_t* d_indices = nullptr;
        float* d_got = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), sizeof(uint64_t) * sample_count));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_got), sizeof(float) * sample_count));

        CUDA_CHECK(cudaMemcpyAsync(d_indices, sample_indices.data(), sizeof(uint64_t) * sample_count,
                                   cudaMemcpyHostToDevice, streams[rank]));

        gather_samples_float_kernel<<<blocks_gather, threads, 0, streams[rank]>>>(
            device_data[rank], d_indices, d_got, static_cast<uint32_t>(sample_count));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(got.data(), d_got, sizeof(float) * sample_count,
                                   cudaMemcpyDeviceToHost, streams[rank]));
        CUDA_CHECK(cudaStreamSynchronize(streams[rank]));

        CUDA_CHECK(cudaFree(d_got));
        CUDA_CHECK(cudaFree(d_indices));

        for (int i = 0; i < sample_count; ++i) {
          float e = expected[i];
          float v = got[i];
          double tol = 1.0e-4 * (1.0 + std::fabs(static_cast<double>(e)));
          if (std::fabs(static_cast<double>(v) - static_cast<double>(e)) > tol) {
            if (options.verbose) {
              std::cerr << "verify mismatch: rank=" << rank
                        << " idx=" << sample_indices[i]
                        << " got=" << v
                        << " expected=" << e
                        << " tol=" << tol << "\n";
            }
            return false;
          }
        }
      }

      return true;
    };

    if (options.verify) {
      verify_ok = verify_payload(++epoch);
      ok_status = ok_status && verify_ok;

      // Restore 0 payload for stable bandwidth timing (no exponential growth).
      for (int rank = 0; rank < world_size; ++rank) {
        CUDA_CHECK(cudaSetDevice(ring_devices[rank]));
        CUDA_CHECK(cudaMemsetAsync(device_data[rank], 0, spec.bytes_per_rank, streams[rank]));
      }
      for (int rank = 0; rank < world_size; ++rank) {
        CUDA_CHECK(cudaSetDevice(ring_devices[rank]));
        CUDA_CHECK(cudaStreamSynchronize(streams[rank]));
      }
    }

    // Warmup.
    for (int it = 0; it < options.warmup_iters && ok_status; ++it) {
      ++epoch;

      for (int rank = 0; rank < world_size; ++rank) {
        reset_scalars(rank);
        params[rank].epoch = epoch;

        cutlass::distributed::collective::ring_allreduce_sm100<float><<<tiling.num_tiles_total, 256, 0, streams[rank]>>>(
            params[rank],
            device_status[rank]);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_events[rank], streams[rank]));
      }

      wait_for_events_or_die(ring_devices, stop_events, options.watchdog_ms, "warmup");

      // Check statuses.
      for (int rank = 0; rank < world_size; ++rank) {
        uint32_t st = 0;
        CUDA_CHECK(cudaSetDevice(ring_devices[rank]));
        CUDA_CHECK(cudaMemcpy(&st, device_status[rank], sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (static_cast<RingAllreduceError>(st) != RingAllreduceError::kOk) {
          ok_status = false;
          if (options.verbose) {
            std::cerr << "warmup status rank" << rank << ": "
                      << ring_allreduce_error_to_string(static_cast<RingAllreduceError>(st))
                      << " (" << st << ")\n";
          }
        }
      }

      if (!ok_status) {
        break;
      }
    }

    double avg_ms = 0.0;
    double algbw = 0.0;
    double busbw = 0.0;

    std::string result = "OK";

    if (!verify_ok) {
      result = "FAIL_VERIFY";
    }
    else if (!ok_status) {
      result = "FAIL";
    }
    else {
      // Measure.
      double sum_iter_ms = 0.0;

      for (int it = 0; it < options.measure_iters; ++it) {
        ++epoch;

        for (int rank = 0; rank < world_size; ++rank) {
          reset_scalars(rank);
          params[rank].epoch = epoch;

          CUDA_CHECK(cudaEventRecord(start_events[rank], streams[rank]));

          cutlass::distributed::collective::ring_allreduce_sm100<float><<<tiling.num_tiles_total, 256, 0, streams[rank]>>>(
              params[rank],
              device_status[rank]);

          CUDA_CHECK(cudaGetLastError());

          CUDA_CHECK(cudaEventRecord(stop_events[rank], streams[rank]));
        }

        wait_for_events_or_die(ring_devices, stop_events, options.watchdog_ms, "measure");

        // Timings.
        double iter_ms = 0.0;
        for (int rank = 0; rank < world_size; ++rank) {
          float ms = 0.0f;
          CUDA_CHECK(cudaSetDevice(ring_devices[rank]));
          CUDA_CHECK(cudaEventElapsedTime(&ms, start_events[rank], stop_events[rank]));
          iter_ms = std::max(iter_ms, static_cast<double>(ms));
        }

        sum_iter_ms += iter_ms;

        // Check statuses.
        for (int rank = 0; rank < world_size; ++rank) {
          uint32_t st = 0;
          CUDA_CHECK(cudaSetDevice(ring_devices[rank]));
          CUDA_CHECK(cudaMemcpy(&st, device_status[rank], sizeof(uint32_t), cudaMemcpyDeviceToHost));
          auto e = static_cast<RingAllreduceError>(st);
          if (e != RingAllreduceError::kOk) {
            result = "FAIL";
            if (options.verbose) {
              std::cerr << "measure status rank" << rank << ": " << ring_allreduce_error_to_string(e)
                        << " (" << st << ")\n";
            }
          }
        }

        if (result != "OK") {
          break;
        }
      }

      if (result == "OK") {
        avg_ms = sum_iter_ms / static_cast<double>(options.measure_iters);
        double t = avg_ms / 1.0e3;
        algbw = (static_cast<double>(spec.bytes_per_rank) / t) / 1.0e9;
        busbw = algbw * (2.0 * static_cast<double>(world_size - 1) / static_cast<double>(world_size));
      }
    }

    print_row(world_size, spec, result, avg_ms, algbw, busbw, ring_devices);
    if (csv_out) {
      write_csv_row(*csv_out, world_size, spec, options, result, avg_ms, algbw, busbw, ring_devices);
    }

    if (options.verify && result != "OK") {
      all_ok = false;
    }

    // Cleanup per-rank allocations.
    for (int rank = 0; rank < world_size; ++rank) {
      CUDA_CHECK(cudaSetDevice(ring_devices[rank]));

      destroy_rank_atomics_kernel<<<blocks, kThreads, 0, streams[rank]>>>(atomics[rank]);
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
    }
  }

  for (int rank = 0; rank < world_size; ++rank) {
    CUDA_CHECK(cudaSetDevice(ring_devices[rank]));
    CUDA_CHECK(cudaEventDestroy(stop_events[rank]));
    CUDA_CHECK(cudaEventDestroy(start_events[rank]));
    CUDA_CHECK(cudaStreamDestroy(streams[rank]));
  }

  return all_ok;
}

} // namespace

int main(int argc, char const** argv) {

  static_assert(sizeof(RingAllreduceSystemAtomicU32) == sizeof(uint32_t),
                "RingAllreduceSystemAtomicU32 must be 32-bit for memset resets");
  static_assert(sizeof(RingAllreduceDeviceAtomicU32) == sizeof(uint32_t),
                "RingAllreduceDeviceAtomicU32 must be 32-bit for memset resets");

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

  std::vector<SizeSpec> sizes;
  if (!parse_sizes_float(options, &sizes)) {
    options.print_usage(std::cerr);
    return 1;
  }

  // CSV output (optional).
  std::ofstream csv;
  bool csv_enabled = !options.csv_path.empty();
  if (csv_enabled) {
    bool need_header = true;
    {
      std::ifstream existing(options.csv_path);
      if (existing.good()) {
        int c = existing.peek();
        need_header = (c == std::char_traits<char>::eof());
      }
    }

    csv.open(options.csv_path, std::ios::out | std::ios::app);
    if (!csv.good()) {
      std::cerr << "failed to open csv path: " << options.csv_path << "\n";
      return 1;
    }
    if (need_header) {
      maybe_write_csv_header(csv);
      csv.flush();
    }
  }

  // Device selection.
  std::vector<int> eligible_devices = enumerate_eligible_devices();
  if (eligible_devices.empty()) {
    std::cerr << "no eligible SM100/SM103 devices found\n";
    return 0;
  }

  if (!options.devices.empty()) {
    if (!validate_devices_unique_and_in_range(options.devices)) {
      return 1;
    }
    for (int d : options.devices) {
      if (!device_is_eligible(d)) {
        std::cerr << "specified device is not eligible (need SM100/SM103 with kernel image): " << d << "\n";
        return 1;
      }
    }
  }

  std::cout << "dtype=float num_channels=" << options.num_channels
            << " tile_elems=" << options.tile_elems
            << " timeouts_enabled=" << (options.timeouts_enabled ? 1 : 0)
            << " warmup_iters=" << options.warmup_iters
            << " measure_iters=" << options.measure_iters
            << " verify=" << (options.verify ? 1 : 0)
            << " verify_samples=" << options.verify_samples
            << "\n";

  print_table_header();

  bool overall_ok = true;

  for (int world_size : options.world_sizes) {
    if (world_size <= 1) {
      // world_size==1 is allowed for debugging but not very interesting.
      if (options.verbose) {
        std::cerr << "skipping world_size=" << world_size << " (benchmark is intended for multi-GPU)\n";
      }
      continue;
    }

    std::vector<int> ring_devices;

    if (!options.devices.empty()) {
      if (static_cast<int>(options.devices.size()) < world_size) {
        std::cerr << "--devices list shorter than requested world_size=" << world_size << "\n";
        return 1;
      }
      ring_devices.assign(options.devices.begin(), options.devices.begin() + world_size);

      auto p2p = validate_ring_p2p_caps_and_enable_peer_access(world_size, ring_devices.data());
      if (!p2p.ok()) {
        print_host_result("validate_ring_p2p_caps_and_enable_peer_access", p2p);
        return 1;
      }
    }
    else {
      if (!find_ring_auto(world_size, eligible_devices, &ring_devices, options.verbose)) {
        // No ring found; skip this world size.
        SizeSpec dummy;
        dummy.bytes_per_rank = 0;
        dummy.count_elems = 0;
        dummy.token = "";

        // Print one skip row per size to keep output rectangular.
        for (auto const& spec : sizes) {
          print_row(world_size, spec, "SKIP_NO_RING", 0.0, 0.0, 0.0, {});
          if (csv_enabled) {
            write_csv_row(csv, world_size, spec, options, "SKIP_NO_RING", 0.0, 0.0, 0.0, {});
          }
        }
        if (options.verify) {
          overall_ok = false;
        }
        continue;
      }

      auto p2p = validate_ring_p2p_caps_and_enable_peer_access(world_size, ring_devices.data());
      if (!p2p.ok()) {
        // This should be rare since find_ring_auto used dry-run checks.
        if (options.verbose) {
          print_host_result("validate_ring_p2p_caps_and_enable_peer_access", p2p);
        }
        for (auto const& spec : sizes) {
          print_row(world_size, spec, "SKIP_NO_RING", 0.0, 0.0, 0.0, {});
          if (csv_enabled) {
            write_csv_row(csv, world_size, spec, options, "SKIP_NO_RING", 0.0, 0.0, 0.0, {});
          }
        }
        if (options.verify) {
          overall_ok = false;
        }
        continue;
      }
    }

    bool ok = run_world_size_float(options, world_size, ring_devices, sizes, csv_enabled ? &csv : nullptr);
    overall_ok = overall_ok && ok;
  }

  return overall_ok ? 0 : 1;
}
