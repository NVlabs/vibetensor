// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace vbt {
namespace dispatch {

// Dispatcher v2 key ordering: highest-priority first (lower enum value wins).
enum class DispatchKey : uint8_t {
  Autograd = 0,
  Python = 1,
  Fabric = 2,
  CUDA = 3,
  CPU = 4,
  NumKeys,
};

constexpr std::size_t kNumDispatchKeys =
    static_cast<std::size_t>(DispatchKey::NumKeys);

} // namespace dispatch
} // namespace vbt
