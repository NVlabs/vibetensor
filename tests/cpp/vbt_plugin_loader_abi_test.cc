// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <string>
#include "vbt/dispatch/plugin_loader.h"

TEST(PluginLoaderAbiTest, MissingInitSymbol) {
#ifndef PLUGIN_MISSING_INIT_PATH
  GTEST_SKIP() << "No missing_init plugin path provided";
#else
  const char* so = PLUGIN_MISSING_INIT_PATH;
  auto st = vbt::dispatch::plugin::load_library(so);
  ASSERT_NE(st, VT_STATUS_OK);
  std::string msg = vbt::dispatch::plugin::get_last_error();
  ASSERT_NE(msg.find("missing symbol: vbt_plugin_init"), std::string::npos) << msg;
#endif
}

TEST(PluginLoaderAbiTest, BadAbiVersion) {
#ifndef PLUGIN_BAD_ABI_PATH
  GTEST_SKIP() << "No bad_abi plugin path provided";
#else
  const char* so = PLUGIN_BAD_ABI_PATH;
  auto st = vbt::dispatch::plugin::load_library(so);
  ASSERT_EQ(st, VT_STATUS_ABI_MISMATCH);
  std::string msg = vbt::dispatch::plugin::get_last_error();
  ASSERT_NE(msg.find("ABI mismatch"), std::string::npos) << msg;
#endif
}
