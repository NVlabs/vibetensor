// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/plugin_loader.h"

namespace {
struct AtomicCommitEnvGuard {
  AtomicCommitEnvGuard() { setenv("VBT_PLUGIN_ATOMIC_COMMIT", "1", 1); }
};
static AtomicCommitEnvGuard g_atomic_commit_env_guard;
}  // namespace

#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS

TEST(PluginAtomicCommitFailInitTest, DlclosesLibraryOnInitFailure) {
#ifndef PLUGIN_P3_FAIL_INIT_PATH
  GTEST_SKIP() << "PLUGIN_P3_FAIL_INIT_PATH not provided";
#else
#if !(defined(__linux__) || defined(__APPLE__))
  GTEST_SKIP() << "dlopen/RTLD_NOLOAD not supported on this platform";
#else
  const char* so = PLUGIN_P3_FAIL_INIT_PATH;

  // Ensure the plugin is not already loaded in this process.
  (void)dlerror();
  void* pre = dlopen(so, RTLD_NOLOAD | RTLD_LAZY | RTLD_LOCAL);
  if (pre) {
    dlclose(pre);
    GTEST_SKIP() << "p3_fail_init plugin already loaded";
  }

  // Enable dispatcher v2 for this test binary.
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  // VBT_PLUGIN_ATOMIC_COMMIT is set by a global guard before main().

  auto& D = vbt::dispatch::Dispatcher::instance();
  ASSERT_FALSE(D.has("vt::p3_fail"));

  const vt_status st = vbt::dispatch::plugin::load_library(so);
  EXPECT_EQ(st, VT_STATUS_INTERNAL) << "err="
                                   << vbt::dispatch::plugin::get_last_error();
  EXPECT_NE(std::string(vbt::dispatch::plugin::get_last_error())
                .find("plugin init failed"),
            std::string::npos);

  // Atomic init failure: no new op should be registered.
  EXPECT_FALSE(D.has("vt::p3_fail"));

  // Loader should not mark the library as loaded.
  EXPECT_FALSE(vbt::dispatch::plugin::is_library_loaded(so));

  // dlclose proof: RTLD_NOLOAD should fail after init failure.
  (void)dlerror();
  void* post = dlopen(so, RTLD_NOLOAD | RTLD_LAZY | RTLD_LOCAL);
  const char* em = dlerror();
  EXPECT_EQ(post, nullptr)
      << (em ? em : "expected RTLD_NOLOAD to fail after init failure");
  if (post) dlclose(post);
#endif
#endif
}

#else

TEST(PluginAtomicCommitFailInitTest, Skipped) {
  GTEST_SKIP() << "requires VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS";
}

#endif  // VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
