// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "vbt/core/tensor.h"

struct DLManagedTensor; // fwd from DLPack

namespace vbt {
namespace interop {

std::unique_ptr<DLManagedTensor, void(*)(DLManagedTensor*)> to_dlpack(const vbt::core::TensorImpl& t);

vbt::core::TensorImpl from_dlpack(DLManagedTensor* /*managed*/);

// Accepts capsules with device_type in {kDLCUDA, kDLGPU, kDLCUDAHost} and copies
// into a newly allocated CUDA tensor on the current device using the current stream.
// Throws std::runtime_error on mixed-device or unsupported device types.
#if VBT_WITH_CUDA
vbt::core::TensorImpl from_dlpack_cuda_copy(DLManagedTensor* /*managed*/);
#endif

} // namespace interop
} // namespace vbt
