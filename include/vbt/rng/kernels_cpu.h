// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "vbt/core/tensor.h"
#include "vbt/rng/generator.h"

namespace vbt {
namespace rng {
namespace cpu {

// Fill tensor in-place with uniform float32 in [low, high)
void uniform_(vbt::core::TensorImpl& t, float low, float high, vbt::rng::Generator& gen);

// Fill tensor in-place with normal float32 N(mean, std^2) using Box–Muller
// Preconditions: t.dtype()==Float32 else TypeError("expected floating dtype for normal_")
void normal_(vbt::core::TensorImpl& t, float mean, float std, vbt::rng::Generator& gen);

// Fill tensor in-place with Bernoulli(p) as float32 in {0.0, 1.0}
// Preconditions: t.dtype()==Float32 else TypeError("expected floating dtype for bernoulli_")
//                0 <= p <= 1 else ValueError("bernoulli_: p must be in [0, 1]")
void bernoulli_(vbt::core::TensorImpl& t, float p, vbt::rng::Generator& gen);

// Fill tensor in-place with randint in [low, high) as int64 using unbiased Lemire mapping
// Preconditions: t.dtype()==Int64 else TypeError("randint: output dtype must be int64")
//                low < high and (high-low) in [1, 2^63-1] else ValueError("randint: require low < high and (high - low) in [1, 2^63 - 1]")
void randint_(vbt::core::TensorImpl& t, std::int64_t low, std::int64_t high, vbt::rng::Generator& gen);

} // namespace cpu
} // namespace rng
} // namespace vbt
