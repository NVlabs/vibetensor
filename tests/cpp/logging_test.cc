// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <absl/log/initialize.h>
#include "vbt/logging/logging.h"
#include <gtest/gtest.h>

TEST(Logging, InitOnce) {
  vbt::InitLogging(std::nullopt);
  vbt::InitLogging(2);
  VBT_LOG(INFO) << "ok";
  auto* fn = &absl::InitializeLog;
  (void)fn;
}
