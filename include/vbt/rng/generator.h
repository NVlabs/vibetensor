// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>

#include "vbt/core/device.h"

namespace vbt {
namespace rng {

struct PhiloxState {
  std::uint64_t seed;
  std::uint64_t offset; // measured in Philox blocks (4x u32)
};

// Base RNG generator using Philox4x32-10 state {seed, offset}.
// Offset is measured in Philox blocks and advanced only via reserve_* APIs.
class Generator {
 public:
  Generator() = default;
  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;
  virtual ~Generator() = default;

  // Device tag for this generator (cpu:k or cuda:k).
  virtual vbt::core::Device device() const noexcept = 0;

  // Advance global offset by `blocks` and return {seed, base_offset}.
  PhiloxState reserve_blocks(std::uint64_t blocks);

  // Convenience: reserve ceil_div(num_u32, 4) Philox blocks.
  PhiloxState reserve_u32(std::uint64_t num_u32);

  // Seeding/state
  void manual_seed(std::uint64_t seed) noexcept;
  std::uint64_t initial_seed() const noexcept;
  PhiloxState get_state() const noexcept;
  void set_state(std::uint64_t seed, std::uint64_t offset) noexcept;

 protected:
  std::atomic<std::uint64_t> seed_{0};
  std::atomic<std::uint64_t> offset_{0};
};

// Default CPU generator singleton (process lifetime)
Generator& default_cpu();

// CUDA-specific generator (per device).
// State semantics match Generator (Philox {seed, offset} in blocks).
class CudaGenerator final : public Generator {
 public:
  explicit CudaGenerator(int device_index) noexcept;
  CudaGenerator(const CudaGenerator&) = delete;
  CudaGenerator& operator=(const CudaGenerator&) = delete;

  vbt::core::Device device() const noexcept override;

 private:
  int device_index_{0};
};

// Default CUDA generator singleton for a given CUDA device index.
// When built without CUDA support, calling this will throw at runtime.
CudaGenerator& default_cuda(int device_index);

} // namespace rng
} // namespace vbt
