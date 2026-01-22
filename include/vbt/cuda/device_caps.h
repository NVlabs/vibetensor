// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "vbt/cuda/stream.h"

namespace vbt {
namespace cuda {

// Cached CUDA device capability snapshot used for launch parameter clamping.
// This is a host-only helper.
struct DeviceCaps {
  int          device_index{0};
  unsigned int max_threads_per_block{256};
  unsigned int max_grid_x{65535u};
  unsigned int max_grid_y{65535u};
};

// Return cached CUDA caps for a device.
// Semantics: when built without CUDA support or on error, returns conservative
// defaults that are safe for kernel launch parameter clamping.
DeviceCaps get_device_caps(DeviceIndex device);

} // namespace cuda
} // namespace vbt
