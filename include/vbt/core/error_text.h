// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace vbt { namespace core {

inline constexpr const char* kCloneCudaDtypeAllowlistMsg =
    "clone: CUDA dtype not supported yet (allowed: float32, float64, complex64, complex128, int64)";

inline constexpr const char* kCloneKernelLaunchFailedPrefix =
    "clone: kernel launch failed: ";

}} // namespace vbt::core
