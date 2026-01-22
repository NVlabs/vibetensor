// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if VBT_INTERNAL_TESTS
#include <cstdint>
#endif

namespace vbt {
namespace cuda {

// Return number of CUDA devices detected.
// Semantics: when built without CUDA support or on error, returns 0.
// When CUDA is enabled and the runtime reports success, returns a nonnegative count.
int device_count() noexcept;

#if VBT_INTERNAL_TESTS
std::uint64_t device_count_calls_for_tests() noexcept;
void reset_device_count_calls_for_tests() noexcept;
#endif

} // namespace cuda
} // namespace vbt
