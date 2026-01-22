// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace vbt {
namespace core {

enum class MemoryFormat : uint8_t {
  Contiguous = 0,
  ChannelsLast = 1,
  Preserve = 2
};

} // namespace core
} // namespace vbt
