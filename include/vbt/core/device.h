// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

// Reuse DLPack enums for device type
#include <dlpack/dlpack.h>

namespace vbt {
namespace core {

struct Device {
  DLDeviceType type{kDLCPU};
  int32_t index{0};

  static constexpr Device cpu(int32_t idx = 0) { return Device{kDLCPU, idx}; }
  static constexpr Device cuda(int32_t idx = 0) { return Device{kDLCUDA, idx}; }

  std::string to_string() const {
    switch (type) {
      case kDLCPU:  return std::string("cpu:")  + std::to_string(index);
      case kDLCUDA: return std::string("cuda:") + std::to_string(index);
      default:      return std::string("unknown:") + std::to_string(index);
    }
  }
  friend inline bool operator==(const Device& a, const Device& b) {
    return a.type == b.type && a.index == b.index;
  }
  friend inline bool operator!=(const Device& a, const Device& b) { return !(a == b); }
};

} // namespace core
} // namespace vbt
