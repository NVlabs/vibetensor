// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <thread>
#include <vector>

#include "vbt/cuda/fabric_state.h"
#include "vbt/cuda/fabric_topology.h"

using vbt::cuda::fabric::FabricInitStatus;
using vbt::cuda::fabric::FabricInitStats;
using vbt::cuda::fabric::FabricMode;
using vbt::cuda::fabric::FabricState;
using vbt::cuda::fabric::FabricTopology;
using vbt::cuda::fabric::fabric_enabled_for_ops;
using vbt::cuda::fabric::fabric_state;
using vbt::cuda::fabric::in_same_fabric;
using vbt::cuda::fabric::is_fabric_usable_for_with_primary;

// and initialization/reset behavior via the test hooks described in

TEST(FabricTopologyHelpersTest, InSameFabricAndUsableWithPrimary) {
  FabricTopology topo;
  topo.device_count = 3;
  topo.can_access_peer.assign(3, std::vector<bool>(3, false));
  topo.p2p_enabled.assign(3, std::vector<bool>(3, false));

  // Devices 0 and 1 form a clique of size 2; device 2 is isolated.
  topo.clique_id = {0, 0, 1};
  topo.clique_size = {2, 1};

  // Basic clique invariants.
  EXPECT_EQ(topo.clique_id.size(), topo.device_count);
  EXPECT_EQ(topo.clique_size.size(), 2);
  EXPECT_EQ(topo.clique_size[0] + topo.clique_size[1], topo.device_count);
  for (int d = 0; d < topo.device_count; ++d) {
    EXPECT_GE(topo.clique_id[d], 0);
    EXPECT_LT(topo.clique_id[d],
              static_cast<int>(topo.clique_size.size()));
  }
  for (int sz : topo.clique_size) {
    EXPECT_GE(sz, 1);
  }

  EXPECT_TRUE(in_same_fabric(0, 1, topo));
  EXPECT_TRUE(in_same_fabric(1, 0, topo));
  EXPECT_TRUE(in_same_fabric(0, 0, topo));
  EXPECT_FALSE(in_same_fabric(0, 2, topo));
  EXPECT_FALSE(in_same_fabric(2, 0, topo));

  // Usable when all devices are within the primary's clique.
  {
    std::array<int, 2> devs{{1, 2}};
    EXPECT_FALSE(is_fabric_usable_for_with_primary(0, devs, topo));
  }

  {
    std::array<int, 1> devs{{1}};
    EXPECT_TRUE(is_fabric_usable_for_with_primary(0, devs, topo));
  }

  // Primary out of range is rejected.
  {
    std::array<int, 1> devs{{0}};
    EXPECT_FALSE(is_fabric_usable_for_with_primary(-1, devs, topo));
    EXPECT_FALSE(is_fabric_usable_for_with_primary(3, devs, topo));
  }
}

TEST(FabricGateTest, BasicScenarios) {
  // Helper to build a minimal topology with a single clique of given size.
  auto make_topology = [](int device_count, int clique_size) {
    FabricTopology topo;
    topo.device_count = device_count;
    topo.can_access_peer.assign(device_count,
                                std::vector<bool>(device_count, false));
    topo.p2p_enabled.assign(device_count,
                            std::vector<bool>(device_count, false));
    topo.clique_id.assign(device_count, 0);
    topo.clique_size.assign(1, clique_size);
    return topo;
  };

  // 0 GPUs: gate is always false regardless of mode.
  {
    FabricState fs{};
    fs.topology = make_topology(0, 0);
    fs.uva_ok = true;
    fs.init_status = FabricInitStatus::Ok;
    fs.config.mode.store(FabricMode::BestEffort, std::memory_order_relaxed);
    EXPECT_FALSE(fabric_enabled_for_ops(fs));
  }

  // 1 GPU: UVA Ok but no clique of size >= 2 => gate is false.
  {
    FabricState fs{};
    fs.topology = make_topology(1, 1);
    fs.uva_ok = true;
    fs.init_status = FabricInitStatus::Ok;
    fs.config.mode.store(FabricMode::BestEffort, std::memory_order_relaxed);
    EXPECT_FALSE(fabric_enabled_for_ops(fs));
  }

  // 2 GPUs, clique size 2, mode == Disabled => gate is false.
  {
    FabricState fs{};
    fs.topology = make_topology(2, 2);
    fs.uva_ok = true;
    fs.init_status = FabricInitStatus::Ok;
    fs.config.mode.store(FabricMode::Disabled, std::memory_order_relaxed);
    EXPECT_FALSE(fabric_enabled_for_ops(fs));
  }

  // 2 GPUs, clique size 2, UVA Ok, non-disabled mode => gate is true.
  {
    FabricState fs{};
    fs.topology = make_topology(2, 2);
    fs.uva_ok = true;
    fs.init_status = FabricInitStatus::Ok;
    fs.config.mode.store(FabricMode::BestEffort, std::memory_order_relaxed);
    EXPECT_TRUE(fabric_enabled_for_ops(fs));

    fs.config.mode.store(FabricMode::DryRun, std::memory_order_relaxed);
    EXPECT_TRUE(fabric_enabled_for_ops(fs));
  }

  // Any non-Ok init_status or uva_ok == false forces the gate closed.
  {
    FabricState fs{};
    fs.topology = make_topology(2, 2);
    fs.uva_ok = false;
    fs.init_status = FabricInitStatus::UvaFailed;
    fs.config.mode.store(FabricMode::BestEffort, std::memory_order_relaxed);
    EXPECT_FALSE(fabric_enabled_for_ops(fs));
  }
}

#if defined(VBT_INTERNAL_TESTS)

TEST(FabricStateInitTest, NoCudaAndSingleGpuViaHooks) {
  using vbt::cuda::fabric::fabric_test_hooks;
  using vbt::cuda::fabric::reset_fabric_state_for_tests;

  // 0-GPU synthetic topology => NoCuda status and canonical disable reason.
  {
    auto& hooks = fabric_test_hooks();
    hooks.fake_topology_builder = [](FabricTopology& topo) {
      topo.device_count = 0;
      topo.can_access_peer.clear();
      topo.p2p_enabled.clear();
      topo.clique_id.clear();
      topo.clique_size.clear();
    };

    reset_fabric_state_for_tests();
    const FabricState& fs = fabric_state();
    EXPECT_EQ(fs.topology.device_count, 0);
    EXPECT_EQ(fs.init_status, FabricInitStatus::NoCuda);
    EXPECT_FALSE(fs.uva_ok);
    EXPECT_FALSE(fs.disable_reason.empty());
    EXPECT_EQ(fs.disable_reason.rfind("[Fabric] Built without CUDA or no CUDA devices are available; Fabric is disabled", 0),
              0u);
    EXPECT_EQ(fs.init_stats.topology_build_attempts, 1u);
    EXPECT_EQ(fs.init_stats.topology_build_failures, 0u);
    EXPECT_EQ(fs.init_stats.uva_self_test_attempts, 0u);
    EXPECT_EQ(fs.init_stats.uva_self_test_failures, 0u);
  }

  // 1-GPU synthetic topology => Ok status, UVA Ok, empty disable_reason.
  {
    auto& hooks = fabric_test_hooks();
    hooks.fake_topology_builder = [](FabricTopology& topo) {
      topo.device_count = 1;
      topo.can_access_peer.assign(1, std::vector<bool>(1, false));
      topo.p2p_enabled.assign(1, std::vector<bool>(1, false));
      topo.clique_id = {0};
      topo.clique_size = {1};
    };

    reset_fabric_state_for_tests();
    const FabricState& fs = fabric_state();
    EXPECT_EQ(fs.topology.device_count, 1);
    EXPECT_EQ(fs.init_status, FabricInitStatus::Ok);
    EXPECT_TRUE(fs.uva_ok);
    EXPECT_TRUE(fs.disable_reason.empty());
    EXPECT_EQ(fs.init_stats.topology_build_attempts, 1u);
    EXPECT_EQ(fs.init_stats.topology_build_failures, 0u);
    EXPECT_EQ(fs.init_stats.uva_self_test_attempts, 1u);
    EXPECT_EQ(fs.init_stats.uva_self_test_failures, 0u);
  }
}

TEST(FabricStateInitTest, UvaFailureAndResetHooks) {
  using vbt::cuda::fabric::fabric_test_hooks;
  using vbt::cuda::fabric::reset_fabric_state_for_tests;

  auto& hooks = fabric_test_hooks();

  // Two-device clique with forced_uva_ok = false => UvaFailed status and
  // canonical UVA-disabled substring in disable_reason.
  hooks.fake_topology_builder = [](FabricTopology& topo) {
    topo.device_count = 2;
    topo.can_access_peer.assign(2, std::vector<bool>(2, false));
    topo.p2p_enabled.assign(2, std::vector<bool>(2, false));
    topo.clique_id = {0, 0};
    topo.clique_size = {2};
  };
  hooks.forced_uva_ok = false;

  reset_fabric_state_for_tests();
  const FabricState& fs1 = fabric_state();
  EXPECT_EQ(fs1.topology.device_count, 2);
  EXPECT_EQ(fs1.init_status, FabricInitStatus::UvaFailed);
  EXPECT_FALSE(fs1.uva_ok);
  EXPECT_NE(fs1.disable_reason.find("[Fabric] UVA invariant violated on this platform; Fabric is disabled"),
            std::string::npos);

  // After reset with forced_uva_ok = true, state should transition to Ok and
  // disable_reason should be cleared.
  hooks.forced_uva_ok = true;
  reset_fabric_state_for_tests();
  const FabricState& fs2 = fabric_state();
  EXPECT_EQ(fs2.topology.device_count, 2);
  EXPECT_EQ(fs2.init_status, FabricInitStatus::Ok);
  EXPECT_TRUE(fs2.uva_ok);
  EXPECT_TRUE(fs2.disable_reason.empty());
}

TEST(FabricStateInitTest, InitStatsIdempotenceAndThreadSafety) {
  using vbt::cuda::fabric::fabric_test_hooks;
  using vbt::cuda::fabric::reset_fabric_state_for_tests;

  auto& hooks = fabric_test_hooks();
  hooks.fake_topology_builder = [](FabricTopology& topo) {
    topo.device_count = 2;
    topo.can_access_peer.assign(2, std::vector<bool>(2, false));
    topo.p2p_enabled.assign(2, std::vector<bool>(2, false));
    topo.clique_id = {0, 0};
    topo.clique_size = {2};
  };
  hooks.forced_uva_ok = true;

  reset_fabric_state_for_tests();
  const FabricState& fs1 = fabric_state();
  const FabricState* ptr = &fs1;

  EXPECT_EQ(fs1.topology.device_count, 2);
  EXPECT_EQ(fs1.init_status, FabricInitStatus::Ok);
  EXPECT_TRUE(fs1.uva_ok);

  EXPECT_EQ(fs1.init_stats.topology_build_attempts, 1u);
  EXPECT_EQ(fs1.init_stats.topology_build_failures, 0u);
  EXPECT_EQ(fs1.init_stats.uva_self_test_attempts, 1u);
  EXPECT_EQ(fs1.init_stats.uva_self_test_failures, 0u);

  // Repeated calls must return the same pointer and leave stats unchanged.
  for (int i = 0; i < 8; ++i) {
    const FabricState& fs = fabric_state();
    EXPECT_EQ(&fs, ptr);
    EXPECT_EQ(fs.init_stats.topology_build_attempts,
              fs1.init_stats.topology_build_attempts);
    EXPECT_EQ(fs.init_stats.topology_build_failures,
              fs1.init_stats.topology_build_failures);
    EXPECT_EQ(fs.init_stats.uva_self_test_attempts,
              fs1.init_stats.uva_self_test_attempts);
    EXPECT_EQ(fs.init_stats.uva_self_test_failures,
              fs1.init_stats.uva_self_test_failures);
  }

  // Basic thread-safety: multiple threads calling fabric_state() see the same
  // pointer and do not mutate stats.
  std::vector<std::thread> threads;
  threads.reserve(8);
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([ptr]() {
      const FabricState& fs = fabric_state();
      EXPECT_EQ(&fs, ptr);
    });
  }
  for (auto& th : threads) {
    th.join();
  }
}

TEST(FabricUvaSelfTest, BasicBehavior) {
  using vbt::cuda::fabric::fabric_test_hooks;

  FabricInitStats stats{};
  std::string disable_reason;

  // 0-GPU topology: vacuously OK, no failures and no message.
  {
    FabricTopology topo;
    topo.device_count = 0;
    stats = FabricInitStats{};
    disable_reason.clear();
    bool ok = run_uva_self_test(topo, &stats, &disable_reason);
    EXPECT_TRUE(ok);
    EXPECT_EQ(stats.uva_self_test_attempts, 1u);
    EXPECT_EQ(stats.uva_self_test_failures, 0u);
    EXPECT_TRUE(disable_reason.empty());
  }

  // 1-GPU topology: vacuously OK, no failures and no message.
  {
    FabricTopology topo;
    topo.device_count = 1;
    stats = FabricInitStats{};
    disable_reason.clear();
    bool ok = run_uva_self_test(topo, &stats, &disable_reason);
    EXPECT_TRUE(ok);
    EXPECT_EQ(stats.uva_self_test_attempts, 1u);
    EXPECT_EQ(stats.uva_self_test_failures, 0u);
    EXPECT_TRUE(disable_reason.empty());
  }

  // 2-GPU clique with forced_uva_ok = false => failure with canonical message
  // and a single failure count.
  {
    auto& hooks = fabric_test_hooks();
    hooks.forced_uva_ok = false;

    FabricTopology topo;
    topo.device_count = 2;
    topo.clique_id = {0, 0};
    topo.clique_size = {2};

    stats = FabricInitStats{};
    disable_reason.clear();

    bool ok = run_uva_self_test(topo, &stats, &disable_reason);
    EXPECT_FALSE(ok);
    EXPECT_EQ(stats.uva_self_test_attempts, 1u);
    EXPECT_EQ(stats.uva_self_test_failures, 1u);
    EXPECT_NE(disable_reason.find("[Fabric] UVA invariant violated on this platform; Fabric is disabled"),
              std::string::npos);
  }

  // 2-GPU clique with forced_uva_ok = true => success with no failures or
  // disable_reason.
  {
    auto& hooks = fabric_test_hooks();
    hooks.forced_uva_ok = true;

    FabricTopology topo;
    topo.device_count = 2;
    topo.clique_id = {0, 0};
    topo.clique_size = {2};

    stats = FabricInitStats{};
    disable_reason.clear();

    bool ok = run_uva_self_test(topo, &stats, &disable_reason);
    EXPECT_TRUE(ok);
    EXPECT_EQ(stats.uva_self_test_attempts, 1u);
    EXPECT_EQ(stats.uva_self_test_failures, 0u);
    EXPECT_TRUE(disable_reason.empty());
  }

  // Reset hook for other tests.
  auto& hooks = fabric_test_hooks();
  hooks.forced_uva_ok.reset();
}

TEST(FabricStateInitTest, CudaErrorClassificationFromTopologyBuilderFailure) {
  using vbt::cuda::fabric::fabric_test_hooks;
  using vbt::cuda::fabric::reset_fabric_state_for_tests;

  auto& hooks = fabric_test_hooks();
  hooks.fake_topology_builder = [](FabricTopology&) {
    throw std::runtime_error("synthetic topology failure");
  };
  hooks.forced_uva_ok.reset();

  reset_fabric_state_for_tests();
  const FabricState& fs = fabric_state();

  EXPECT_EQ(fs.topology.device_count, 0);
  EXPECT_EQ(fs.init_status, FabricInitStatus::CudaError);
  EXPECT_FALSE(fs.uva_ok);
  EXPECT_FALSE(fs.disable_reason.empty());
}

#endif  // VBT_INTERNAL_TESTS
