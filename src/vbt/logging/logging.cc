// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/logging/logging.h"
#include <mutex>
#include <absl/log/initialize.h>
#include <absl/log/globals.h>
#include <absl/base/log_severity.h>

namespace vbt {
namespace {
std::once_flag g_once;
}

void InitLogging(std::optional<int> min_level) {
  std::call_once(g_once, [] { absl::InitializeLog(); });
  if (min_level) {
    absl::SetMinLogLevel(static_cast<absl::LogSeverityAtLeast>(
        absl::NormalizeLogSeverity(*min_level)));
  }
}

}  // namespace vbt
