// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/rng/generator.h"

#include <limits>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <vector>

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

namespace vbt {
namespace rng {

PhiloxState Generator::reserve_blocks(std::uint64_t blocks) {
  if (blocks == 0) {
    return PhiloxState{ seed_.load(std::memory_order_relaxed),
                        offset_.load(std::memory_order_relaxed) };
  }
  while (true) {
    std::uint64_t cur = offset_.load(std::memory_order_relaxed);
    if (blocks > (std::numeric_limits<std::uint64_t>::max() - cur)) {
      throw std::runtime_error(std::string("rng: offset overflow: cannot reserve ") + std::to_string((unsigned long long)blocks) +
                               std::string(" blocks at offset=") + std::to_string((unsigned long long)cur));
    }
    std::uint64_t next = cur + blocks;
    if (offset_.compare_exchange_weak(cur, next,
                                      std::memory_order_relaxed,
                                      std::memory_order_relaxed)) {
      const std::uint64_t s = seed_.load(std::memory_order_relaxed);
      return PhiloxState{ s, cur };
    }
  }
}

PhiloxState Generator::reserve_u32(std::uint64_t num_u32) {
  std::uint64_t blocks = (num_u32 + 3ull) / 4ull;
  return reserve_blocks(blocks);
}

void Generator::manual_seed(std::uint64_t seed) noexcept {
  seed_.store(seed, std::memory_order_relaxed);
  offset_.store(0, std::memory_order_relaxed);
}

std::uint64_t Generator::initial_seed() const noexcept {
  return seed_.load(std::memory_order_relaxed);
}

PhiloxState Generator::get_state() const noexcept {
  return PhiloxState{ seed_.load(std::memory_order_relaxed),
                      offset_.load(std::memory_order_relaxed) };
}

void Generator::set_state(std::uint64_t seed, std::uint64_t offset) noexcept {
  seed_.store(seed, std::memory_order_relaxed);
  offset_.store(offset, std::memory_order_relaxed);
}

namespace {

class DefaultCpuGenerator final : public Generator {
 public:
  vbt::core::Device device() const noexcept override {
    return vbt::core::Device::cpu(0);
  }
};

#if VBT_WITH_CUDA
// Per-device CUDA generators; created lazily.
static std::mutex g_cuda_gen_mu;
static std::vector<std::unique_ptr<CudaGenerator>> g_cuda_gens;
#endif

} // anonymous namespace

Generator& default_cpu() {
  static DefaultCpuGenerator g;
  return g;
}

CudaGenerator::CudaGenerator(int device_index) noexcept
  : device_index_(device_index) {}

vbt::core::Device CudaGenerator::device() const noexcept {
  return vbt::core::Device::cuda(device_index_);
}

CudaGenerator& default_cuda(int device_index) {
#if VBT_WITH_CUDA
  if (device_index < 0) {
    throw std::runtime_error("default_cuda: device_index must be non-negative");
  }
  std::lock_guard<std::mutex> lock(g_cuda_gen_mu);
  const std::size_t idx = static_cast<std::size_t>(device_index);
  if (idx >= g_cuda_gens.size()) {
    g_cuda_gens.resize(idx + 1);
  }
  if (!g_cuda_gens[idx]) {
    g_cuda_gens[idx] = std::make_unique<CudaGenerator>(device_index);
  }
  return *g_cuda_gens[idx];
#else
  (void)device_index;
  throw std::runtime_error("default_cuda: CUDA support not built");
#endif
}

} // namespace rng
} // namespace vbt
