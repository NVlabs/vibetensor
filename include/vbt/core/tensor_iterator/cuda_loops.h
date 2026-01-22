// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vbt/core/tensor_iter.h"
#include "vbt/cuda/stream.h"
#include <cuda_runtime.h>

namespace vbt {
namespace core {

// TI-based GPU kernel helper.
// Automates DeviceStrideMeta construction, indexing choice, and kernel launch.

template <typename Launcher>
void ti_gpu_kernel(const ::vbt::core::TensorIter& iter, Launcher&& launcher) {
  if (iter.numel() == 0) return;

  if (!iter.can_use_32bit_indexing()) {
    // Reduction iterators do not support with_32bit_indexing(); fall back to a 64-bit launch.
    if (iter.is_reduction()) {
      int device_idx = iter.operand(0).device.index;
      auto stream = ::vbt::cuda::getCurrentStream(static_cast<::vbt::cuda::DeviceIndex>(device_idx));
      launcher(iter, /*use32=*/false, reinterpret_cast<void*>(stream.handle()));
      return;
    }

    bool any_split = false;
    iter.with_32bit_indexing([&](const ::vbt::core::TensorIter& sub) {
      any_split = true;
      ti_gpu_kernel(sub, launcher);
    });

    if (!any_split) {
      // Fallback to 64-bit kernel
      int device_idx = iter.operand(0).device.index;
      auto stream = ::vbt::cuda::getCurrentStream(static_cast<::vbt::cuda::DeviceIndex>(device_idx));
      launcher(iter, /*use32=*/false, reinterpret_cast<void*>(stream.handle()));
    }
    return;
  }

  int device_idx = iter.operand(0).device.index; 
  auto stream = ::vbt::cuda::getCurrentStream(static_cast<::vbt::cuda::DeviceIndex>(device_idx));

  launcher(iter, /*use32=*/true, reinterpret_cast<void*>(stream.handle()));
}

} // namespace core
} // namespace vbt
