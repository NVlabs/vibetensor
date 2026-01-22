// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "vbt/core/tensor.h"
#include "vbt/rng/generator.h"

namespace vbt {
namespace rng {
namespace cuda {

// Launch configuration for Philox-based RNG kernels on CUDA.
struct ExecutionPolicy {
  std::uint32_t grid_x{1};
  std::uint32_t block_x{1};
  std::uint32_t blocks_per_thread{1};
};

// Calculate a deterministic execution policy for a kernel that consumes
// total_blocks = ceil_div_u64(numel, outputs_per_block) Philox blocks.
// This helper chooses grid/block sizes only; CUDA RNG wrappers are
// responsible for computing total_blocks and outputs_per_block for
// Philox accounting.
//   VBT_RNG_CUDA_BLOCK             → threads-per-block
//   VBT_RNG_CUDA_BLOCKS_PER_THREAD → blocks-per-thread (grid-stride depth)
ExecutionPolicy calc_execution_policy(std::int64_t numel,
                                      std::uint32_t outputs_per_block = 4,
                                      std::uint32_t blocks_per_thread_default = 4) noexcept;

// In-place uniform float32 fill for CUDA tensors using Philox RNG.
// Preconditions:
//   - t.device().type == kDLCUDA
//   - t.dtype() == ScalarType::Float32 (guarded at caller in Python bindings
//     for TypeError parity; implementation also validates for C++ callers).
void uniform_(vbt::core::TensorImpl& t, float low, float high,
              vbt::rng::CudaGenerator& gen);

// In-place normal float32 fill for CUDA tensors using Box–Muller.
void normal_(vbt::core::TensorImpl& t, float mean, float std,
             vbt::rng::CudaGenerator& gen);

// In-place bernoulli float32 fill for CUDA tensors.
void bernoulli_(vbt::core::TensorImpl& t, float p,
                vbt::rng::CudaGenerator& gen);

// In-place randint int64 fill for CUDA tensors using unbiased Lemire mapping
// with per-element attempt dimension (ctr[2]).
void randint_(vbt::core::TensorImpl& t, std::int64_t low, std::int64_t high,
              vbt::rng::CudaGenerator& gen);

} // namespace cuda
} // namespace rng
} // namespace vbt
