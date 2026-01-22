// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vbt/core/tensor.h"

namespace vbt {
namespace core {

// Clone arbitrary strided tensor into a new contiguous tensor on the same device.
// - CPU -> clone_cpu
// - CUDA -> clone_cuda
TensorImpl clone_contiguous_same_device(const TensorImpl& self);

// Clone arbitrary strided CPU tensor into a new contiguous CPU tensor.
// Throws std::invalid_argument if self is not a CPU tensor.
TensorImpl clone_cpu(const TensorImpl& self);

#if VBT_WITH_CUDA
// Clone arbitrary strided CUDA tensor into a new contiguous tensor on the same device
TensorImpl clone_cuda(const TensorImpl& self);
#endif

} // namespace core
} // namespace vbt
