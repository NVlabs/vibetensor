// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <string>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include "vbt/dispatch/plugin_loader.h"

TEST(PluginInitFailureNoDlcloseTest, FailedInitKeepsLibraryLoaded) {
#ifndef PLUGIN_FAIL_ROLLBACK_PATH
  GTEST_SKIP() << "No fail_rollback plugin path provided";
#else
#if defined(__linux__) || defined(__APPLE__)
  const char* so = PLUGIN_FAIL_ROLLBACK_PATH;

  // Ensure the plugin is not already loaded in this process.
  (void)dlerror();
  void* pre = dlopen(so, RTLD_NOLOAD | RTLD_LAZY | RTLD_LOCAL);
  if (pre) {
    dlclose(pre);
    GTEST_SKIP() << "fail_rollback plugin already loaded";
  }

  const vt_status st = vbt::dispatch::plugin::load_library(so);
  ASSERT_EQ(st, VT_STATUS_INVALID_ARG) << "status=" << static_cast<int>(st)
                                       << " err=" << vbt::dispatch::plugin::get_last_error();

  // After init failure we must keep the .so loaded to avoid stale function pointers
  // calling into unmapped code.
  (void)dlerror();
  void* still = dlopen(so, RTLD_NOLOAD | RTLD_LAZY | RTLD_LOCAL);
  const char* em = dlerror();
  ASSERT_NE(still, nullptr) << (em ? em : "expected library still loaded after init failure");
  dlclose(still);
#else
  GTEST_SKIP() << "RTLD_NOLOAD not supported on this platform";
#endif
#endif
}
