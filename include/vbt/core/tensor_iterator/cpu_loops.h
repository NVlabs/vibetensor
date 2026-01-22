// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vbt/core/tensor_iter.h"
#include <functional>

namespace vbt {
namespace core {

// TI-based CPU kernel helper.
// Currently supports single-threaded execution over the iterator.
// Lambda F should accept arguments corresponding to the iterator's operands.
// For now, we expose a simpler interface that passes raw pointers to the lambda,
// assuming the user handles casting based on common_dtype.
//
// Future work: automatic casting and typed lambdas.

// A scalar loop signature: void(char** data, const int64_t* strides, int64_t n)
// But we want to provide a typed experience if possible.

template <typename Loop>
void ti_cpu_kernel(const TensorIterBase& iter, Loop&& loop) {
  static_assert(std::is_convertible_v<Loop, ::vbt::core::TensorIterBase::loop1d_t>,
      "Loop must be convertible to loop1d_t (non-capturing lambda or function pointer). "
      "Capturing lambdas are not supported in this low-level API.");
  iter.for_each_cpu(std::forward<Loop>(loop), nullptr);
}

} // namespace core
} // namespace vbt
