// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace vbt::core {

template <class T>
struct alignas(sizeof(T) * 2) Complex final {
  T re;
  T im;
};

using Complex64 = Complex<float>;   // sizeof=8,  alignof=8
using Complex128 = Complex<double>; // sizeof=16, alignof=16

} // namespace vbt::core
