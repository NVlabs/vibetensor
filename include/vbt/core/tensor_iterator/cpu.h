// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vbt/core/tensor_iterator/core.h"

namespace vbt {
namespace core {

class TensorIterBase;
class TensorIter;

// Canonical external entry point for CPU elementwise drivers.
void for_each_cpu(const TensorIterBase& iter,
                  loop1d_t loop,
                  void* ctx);

// Canonical external entry point for CPU reductions.
void for_each_reduction_cpu(const TensorIter& iter,
                            reduce_loop1d_t loop,
                            void* ctx);

} // namespace core
} // namespace vbt
