// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/tensor_iterator/cpu.h"

namespace vbt {
namespace core {

void for_each_cpu(const TensorIterBase& iter,
                  loop1d_t loop,
                  void* ctx) {
  iter.for_each_cpu(loop, ctx);
}

} // namespace core
} // namespace vbt
