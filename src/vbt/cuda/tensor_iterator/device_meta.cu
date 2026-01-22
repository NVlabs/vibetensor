// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/tensor_iterator/cuda.h"

namespace vbt {
namespace core {

// Device-side offset calculator for TI-produced DeviceStrideMeta now lives
// inline in vbt/core/tensor_iterator/cuda.h to avoid cross-TU device
// references without requiring CUDA separable compilation.

}  // namespace core
}  // namespace vbt
