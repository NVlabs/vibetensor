// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/rng/kernels_cuda.h"

#include "vbt/rng/philox_util.h"

#include <algorithm>
#include <cstdlib>
#include <limits>

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#include <cerrno>
#include <cuda_runtime_api.h>
#endif

namespace vbt {
namespace rng {
namespace cuda {

namespace {

#if VBT_WITH_CUDA
static inline std::uint32_t parse_env_u32(const char* name,
                                          std::uint32_t def) noexcept {
  const char* v = std::getenv(name);
  if (!v || *v == '\0') return def;
  char* end = nullptr;
  errno = 0;
  unsigned long long x = std::strtoull(v, &end, 10);
  if (end == v || errno != 0) return def;
  if (x > static_cast<unsigned long long>(std::numeric_limits<std::uint32_t>::max())) {
    return def;
  }
  return static_cast<std::uint32_t>(x);
}
#endif

} // anonymous namespace

ExecutionPolicy calc_execution_policy(std::int64_t numel,
                                      std::uint32_t outputs_per_block,
                                      std::uint32_t blocks_per_thread_default) noexcept {
  ExecutionPolicy pol{};

  if (numel <= 0 || outputs_per_block == 0u) {
    pol.grid_x = 1u;
    pol.block_x = 1u;
    pol.blocks_per_thread = 1u;
    return pol;
  }

  std::uint64_t N = static_cast<std::uint64_t>(numel);
  std::uint64_t total_blocks = ceil_div_u64(N, static_cast<std::uint64_t>(outputs_per_block));

#if VBT_WITH_CUDA
  int dev = 0;
  (void)cudaGetDevice(&dev); // best-effort; fall back to generic caps on error

  cudaDeviceProp prop{};
  std::uint32_t max_tpb = 1024u;
  std::uint64_t max_grid_x = 65535ull;
  if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
    if (prop.maxThreadsPerBlock > 0) {
      max_tpb = static_cast<std::uint32_t>(prop.maxThreadsPerBlock);
    }
    if (prop.maxGridSize[0] > 0) {
      max_grid_x = static_cast<std::uint64_t>(prop.maxGridSize[0]);
    }
  }

  std::uint32_t tpb_default = 256u;
  std::uint32_t tpb = parse_env_u32("VBT_RNG_CUDA_BLOCK", tpb_default);
  if (tpb < 64u) tpb = 64u;
  if (tpb > max_tpb) tpb = max_tpb;
  // Warp-align downward; keep at least one warp.
  tpb = (tpb / 32u) * 32u;
  if (tpb == 0u) tpb = 32u;

  std::uint32_t bpt_default = (blocks_per_thread_default == 0u) ? 4u : blocks_per_thread_default;
  std::uint32_t bpt = parse_env_u32("VBT_RNG_CUDA_BLOCKS_PER_THREAD", bpt_default);
  if (bpt < 1u) bpt = 1u;
  if (bpt > 16u) bpt = 16u;

  std::uint64_t threads_needed = ceil_div_u64(total_blocks, static_cast<std::uint64_t>(bpt));
  std::uint64_t blocks_needed = ceil_div_u64(threads_needed, static_cast<std::uint64_t>(tpb));

  std::uint64_t grid_cap = (max_grid_x == 0ull) ? 65535ull : max_grid_x;
  std::uint64_t gx = std::max<std::uint64_t>(1ull,
      std::min<std::uint64_t>(blocks_needed, grid_cap));

  pol.grid_x = static_cast<std::uint32_t>(gx);
  pol.block_x = tpb;
  pol.blocks_per_thread = bpt;
#else
  (void)outputs_per_block;
  (void)blocks_per_thread_default;
  pol.grid_x = 1u;
  pol.block_x = 1u;
  pol.blocks_per_thread = 1u;
#endif

  return pol;
}

} // namespace cuda
} // namespace rng
} // namespace vbt
