// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "vbt/core/device.h"
#include "vbt/core/tensor.h"

namespace vbt { namespace autograd {

// Device-agnostic dense in-place add used by the autograd engine
// and AccumulateGrad. Supports CPU and (when enabled) single-device
// CUDA float32/float16 tensors. See design/cuda_ad/p2/README.md for
// invariants.
void autograd_add_inplace_dense(vbt::core::TensorImpl& acc,
                                const vbt::core::TensorImpl& addend,
                                const vbt::core::Device& autograd_device);

}} // namespace vbt::autograd
