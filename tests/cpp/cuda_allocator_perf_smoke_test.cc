// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <sstream>
#include <string>

#include "include/cuda_allocator_perf.h"

using namespace vbt::cuda;
using namespace vbt::cuda::testonly;

TEST(CudaAllocatorPerfSmokeTest, EmitsJsonForB1SmokeScenario) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  const char* env = std::getenv("VBT_RUN_ALLOCATOR_GRAPHS_PERF");
  if (!env || std::string(env) != "1") {
    GTEST_SKIP() << "allocator+graphs perf wrappers disabled (set VBT_RUN_ALLOCATOR_GRAPHS_PERF=1)";
  }

  RunCounts overrides{};
  RunCounts counts = resolve_run_counts(ScenarioId::B1, RunMode::Smoke, overrides);

  PerfConfig cfg;
  cfg.scenario_id = ScenarioId::B1;
  cfg.runner = Runner::CppNative;
  cfg.run_mode = RunMode::Smoke;
  cfg.device_index = 0;
  cfg.counts = counts;
  cfg.num_replays = 0;
  cfg.notes = "smoke";

  PerfResult res = run_B1(cfg);

  std::ostringstream oss;
  write_perf_result_json(res, oss);
  std::string json = oss.str();

  ASSERT_FALSE(json.empty());
  EXPECT_EQ(json.front(), '{');
  EXPECT_NE(json.find("\"schema_version\""), std::string::npos);
  EXPECT_NE(json.find("\"scenario_id\""), std::string::npos);
  EXPECT_NE(json.find("\"metrics\""), std::string::npos);
#endif
}
