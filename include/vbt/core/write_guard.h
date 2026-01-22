// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>

#include "vbt/core/tensor.h"
#include "vbt/core/overlap.h"

namespace vbt {
namespace core {

inline void check_writable(const TensorImpl& dst) {
  if (dst.numel() == 0) return;
  assert_no_internal_overlap(dst);
}

} // namespace core
} // namespace vbt
