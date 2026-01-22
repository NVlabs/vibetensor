// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <cassert>
#include <absl/log/log.h>

namespace vbt {
// Initialize Abseil logging once; optionally set min log level.
void InitLogging(std::optional<int> min_level);
}

// Shorthand macros matching policy in design
#define VBT_LOG(level) LOG(level)
#define VBT_CHECK(cond) CHECK(cond)
#define VBT_ASSERT(cond) assert(cond)
