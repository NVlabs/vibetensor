// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <cctype>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace vbt::cuda;

namespace {

std::string current_alloc_conf() {
  const char* env = std::getenv("VBT_CUDA_ALLOC_CONF");
  return env ? std::string(env) : std::string();
}

} // namespace

TEST(CudaAllocatorFractionEnvAndApiTest, EnvAndSetterBehavior) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  std::string conf = current_alloc_conf();

  const bool has_fraction_key =
      conf.find("per_process_memory_fraction") != std::string::npos;
  const bool expects_valid_fraction =
      conf.find("per_process_memory_fraction=0.25") != std::string::npos;
  const bool expects_invalid_fraction = has_fraction_key && !expects_valid_fraction;

  if (expects_invalid_fraction) {
    ::testing::internal::CaptureStderr();
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (expects_invalid_fraction) {
    std::string stderr_out = ::testing::internal::GetCapturedStderr();
    EXPECT_NE(stderr_out.find(
                  "[VBT_CUDA_ALLOC_CONF] warning: invalid per_process_memory_fraction='"),
              std::string::npos);
    EXPECT_NE(stderr_out.find(
                  "(must be in [0,1]); using default 1.0"),
              std::string::npos);
  }

  double f = A.getMemoryFraction();

  if (!has_fraction_key) {
    EXPECT_DOUBLE_EQ(f, 1.0);
  } else if (expects_valid_fraction) {
    EXPECT_NEAR(f, 0.25, 1e-9);
  } else {
    // Invalid env values fall back to 1.0.
    EXPECT_DOUBLE_EQ(f, 1.0);
  }

  // Setter API behavior (CUDA builds only).
  EXPECT_NO_THROW(A.setMemoryFraction(0.0));
  EXPECT_NO_THROW(A.setMemoryFraction(0.5));
  EXPECT_NO_THROW(A.setMemoryFraction(1.0));
  EXPECT_THROW(A.setMemoryFraction(-0.1), std::invalid_argument);
  EXPECT_THROW(A.setMemoryFraction(1.1), std::invalid_argument);
#endif
}

TEST(CudaAllocatorSplitTailTest, SplitTailScaffoldingAlwaysFalse) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  // Simple alloc/free workload to populate allocator state.
  void* p0 = A.raw_alloc(1 << 20);
  ASSERT_NE(p0, nullptr);
  A.raw_delete(p0);

  void* p1 = A.raw_alloc(1 << 18);
  ASSERT_NE(p1, nullptr);
  A.raw_delete(p1);

  // Drain any deferred frees and cached segments.
  A.process_events();
  A.emptyCache();

  auto ptrs = A.debug_tracked_block_ptrs();
  for (void* ptr : ptrs) {
    EXPECT_FALSE(A.debug_block_is_split_tail(ptr));
  }

  int stack_val = 42;
  EXPECT_FALSE(A.debug_block_is_split_tail(&stack_val));
  EXPECT_FALSE(A.debug_block_is_split_tail(nullptr));
#endif
}

TEST(CudaAllocatorSplitTailTest, EnvFlagAndSplitEnabledGate) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  std::string conf = current_alloc_conf();
  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  auto trim = [](std::string& t) {
    size_t a = 0;
    while (a < t.size() && (t[a] == ' ' || t[a] == '\t' || t[a] == '\n' || t[a] == ',')) ++a;
    size_t b = t.size();
    while (b > a && (t[b-1] == ' ' || t[b-1] == '\t' || t[b-1] == '\n')) --b;
    t = t.substr(a, b - a);
  };
  auto parse_bool = [](const std::string& t) {
    std::string u;
    u.reserve(t.size());
    for (char c : t) {
      u.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return (u == "1" || u == "true" || u == "yes" || u == "on");
  };

  bool has_flag_key = false;
  bool expected_cfg_flag = false;

  std::size_t i = 0;
  while (i < conf.size()) {
    while (i < conf.size() && (conf[i] == ' ' || conf[i] == '\t' || conf[i] == '\n' || conf[i] == ',')) ++i;
    if (i >= conf.size()) break;
    std::size_t k0 = i;
    while (i < conf.size() && conf[i] != '=' && conf[i] != ',') ++i;
    if (i >= conf.size() || conf[i] != '=') break;
    std::string key = conf.substr(k0, i - k0);
    ++i;
    std::size_t v0 = i;
    while (i < conf.size() && conf[i] != ',') ++i;
    std::string val = conf.substr(v0, i - v0);
    trim(key);
    trim(val);
    if (key == "enable_block_splitting") {
      has_flag_key = true;
      expected_cfg_flag = parse_bool(val);
    }
  }

  bool cfg_flag = A.debug_cfg_enable_block_splitting();
  if (!has_flag_key) {
    EXPECT_FALSE(cfg_flag);
  } else {
    EXPECT_EQ(cfg_flag, expected_cfg_flag);
  }

  // Determine backend kind from env string.
  std::string conf_lower = conf;
  for (char& c : conf_lower) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  bool is_async_backend =
      conf_lower.find("backend=cudamallocasync") != std::string::npos;

  bool split_enabled = A.debug_split_enabled();
  if (is_async_backend) {
    EXPECT_FALSE(split_enabled);
  } else {
    EXPECT_EQ(split_enabled, cfg_flag);
  }
#endif
}
