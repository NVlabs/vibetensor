// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vbt/core/dtype.h"

namespace vbt { namespace autograd {

// Single source of truth for which CUDA dtypes are supported by the autograd
// engine (including custom Python Functions and gradient accumulation).
//
// Policy:
// - Always allow Float32 when CUDA is built.
// - Allow Float16 when CUDA is built.
[[nodiscard]] inline bool is_cuda_autograd_dtype_supported(
    vbt::core::ScalarType dt) noexcept {
#if VBT_WITH_CUDA
  switch (dt) {
    case vbt::core::ScalarType::Float32:
    case vbt::core::ScalarType::Float16:
      return true;
    default:
      return false;
  }
#else
  (void)dt;
  return false;
#endif
}

}} // namespace vbt::autograd
