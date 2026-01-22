// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/core/indexing.h"

using vbt::core::indexing::advanced_index_32bit_enabled;
using vbt::core::indexing::set_advanced_index_32bit_enabled_for_tests;
using vbt::core::indexing::detail::compute_mindex32_default_from_env_value_for_tests;
using vbt::core::indexing::detail::compute_adv_flag_from_env_raw;

#if VBT_INTERNAL_TESTS
using vbt::core::indexing::detail::AdvancedIndexEnvConfig;
using vbt::core::indexing::detail::AdvancedIndexEnvConfigGuard;
using vbt::core::indexing::detail::EnvProbeCounters;
using vbt::core::indexing::detail::get_advanced_index_env_config;
using vbt::core::indexing::detail::get_advanced_index_env_config_for_tests;
using vbt::core::indexing::detail::get_env_probe_counters_for_tests;
using vbt::core::indexing::detail::reset_advanced_index_env_config_for_tests;
using vbt::core::indexing::detail::clear_advanced_index_env_config_override_for_tests;
using vbt::core::indexing::detail::reset_env_probe_counters_for_tests;
#if VBT_WITH_CUDA
using vbt::core::indexing::detail::CudaBoundsMode;
using vbt::core::indexing::cuda_impl::get_effective_cuda_bounds_mode_for_tests;
#endif
#endif

namespace {

// Simple RAII helper mirroring the guards used in other indexing tests.
struct AdvancedIndex32BitGuard {
  bool prev;
  explicit AdvancedIndex32BitGuard(bool enabled)
      : prev(advanced_index_32bit_enabled()) {
    set_advanced_index_32bit_enabled_for_tests(enabled);
  }
  ~AdvancedIndex32BitGuard() {
    set_advanced_index_32bit_enabled_for_tests(prev);
  }
};

} // namespace

TEST(IndexingAdvancedFlagEnvTest, EnvTableConformanceHelperMatchesDesign) {
  // Unset / empty -> optimizations enabled.
  EXPECT_TRUE(compute_mindex32_default_from_env_value_for_tests(nullptr));
  EXPECT_TRUE(compute_mindex32_default_from_env_value_for_tests(""));

  // False-like values (case-insensitive) -> optimizations enabled.
  EXPECT_TRUE(compute_mindex32_default_from_env_value_for_tests("0"));
  EXPECT_TRUE(compute_mindex32_default_from_env_value_for_tests("false"));
  EXPECT_TRUE(compute_mindex32_default_from_env_value_for_tests("FALSE"));
  EXPECT_TRUE(compute_mindex32_default_from_env_value_for_tests("No"));
  EXPECT_TRUE(compute_mindex32_default_from_env_value_for_tests("OFF"));

  // Any other non-empty string disables optimizations (kill-switch engaged).
  EXPECT_FALSE(compute_mindex32_default_from_env_value_for_tests("1"));
  EXPECT_FALSE(compute_mindex32_default_from_env_value_for_tests("true"));
  EXPECT_FALSE(compute_mindex32_default_from_env_value_for_tests("YES"));
  EXPECT_FALSE(compute_mindex32_default_from_env_value_for_tests("on"));
  EXPECT_FALSE(compute_mindex32_default_from_env_value_for_tests("maybe"));
}

TEST(IndexingAdvancedFlagEnvTest, AdvancedIndexingEnvTableConformanceMatchesDesign) {
  // Unset / empty -> advanced indexing enabled.
  EXPECT_TRUE(compute_adv_flag_from_env_raw(nullptr));
  EXPECT_TRUE(compute_adv_flag_from_env_raw(""));

  // Explicit false-like values (case-insensitive, with trimming) -> disabled.
  EXPECT_FALSE(compute_adv_flag_from_env_raw("0"));
  EXPECT_FALSE(compute_adv_flag_from_env_raw("false"));
  EXPECT_FALSE(compute_adv_flag_from_env_raw("FALSE"));
  EXPECT_FALSE(compute_adv_flag_from_env_raw("no"));
  EXPECT_FALSE(compute_adv_flag_from_env_raw(" off "));

  // All other non-empty values enable advanced indexing.
  EXPECT_TRUE(compute_adv_flag_from_env_raw("1"));
  EXPECT_TRUE(compute_adv_flag_from_env_raw("true"));
  EXPECT_TRUE(compute_adv_flag_from_env_raw("True"));
  EXPECT_TRUE(compute_adv_flag_from_env_raw(" yes "));
  EXPECT_TRUE(compute_adv_flag_from_env_raw("garbage"));
}

TEST(IndexingAdvancedFlagEnvTest, SetterNestedGuardsRestorePreviousValue) {
  const bool initial = advanced_index_32bit_enabled();

  {
    AdvancedIndex32BitGuard outer(!initial);
    EXPECT_EQ(advanced_index_32bit_enabled(), !initial);

    {
      AdvancedIndex32BitGuard inner(initial);
      // Inner guard should temporarily restore the original value.
      EXPECT_EQ(advanced_index_32bit_enabled(), initial);
    }

    // After inner guard destruction, outer guard's value must be restored.
    EXPECT_EQ(advanced_index_32bit_enabled(), !initial);
  }

  // After all guards, we must be back to the original process-wide value.
  EXPECT_EQ(advanced_index_32bit_enabled(), initial);
}

#if VBT_INTERNAL_TESTS

TEST(IndexingAdvancedFlagEnvTest, EnvConfigSingleGetEnvPerVarAcrossCalls) {
  reset_env_probe_counters_for_tests();

  // First read may compute env defaults once.
  (void)get_advanced_index_env_config();
  EnvProbeCounters first = get_env_probe_counters_for_tests();

  // Second read must reuse the cached config without touching getenv.
  (void)get_advanced_index_env_config();
  EnvProbeCounters second = get_env_probe_counters_for_tests();

  EXPECT_EQ(second.num_getenv_calls_enable_adv,
            first.num_getenv_calls_enable_adv);
  EXPECT_EQ(second.num_getenv_calls_mindex32_disable,
            first.num_getenv_calls_mindex32_disable);
  EXPECT_EQ(second.num_getenv_calls_debug_adv_index,
            first.num_getenv_calls_debug_adv_index);
  EXPECT_EQ(second.num_getenv_calls_cuda_gpu_bounds_disable,
            first.num_getenv_calls_cuda_gpu_bounds_disable);
  EXPECT_EQ(second.num_getenv_calls_cuda_max_blocks,
            first.num_getenv_calls_cuda_max_blocks);
  EXPECT_EQ(second.num_getenv_calls_cuda_bool_mask_indices,
            first.num_getenv_calls_cuda_bool_mask_indices);
  EXPECT_EQ(second.num_getenv_calls_cuda_bool_mask_cub,
            first.num_getenv_calls_cuda_bool_mask_cub);
  EXPECT_EQ(second.num_getenv_calls_cuda_extended_dtypes,
            first.num_getenv_calls_cuda_extended_dtypes);

  // Within this test run, each env var should be consulted at most once.
  EXPECT_LE(first.num_getenv_calls_enable_adv, 1u);
  EXPECT_LE(first.num_getenv_calls_mindex32_disable, 1u);
  EXPECT_LE(first.num_getenv_calls_debug_adv_index, 1u);
  EXPECT_LE(first.num_getenv_calls_cuda_gpu_bounds_disable, 1u);
  EXPECT_LE(first.num_getenv_calls_cuda_max_blocks, 1u);
  EXPECT_LE(first.num_getenv_calls_cuda_bool_mask_indices, 2u);
  EXPECT_LE(first.num_getenv_calls_cuda_bool_mask_cub, 2u);
  EXPECT_LE(first.num_getenv_calls_cuda_extended_dtypes, 2u);
}

TEST(IndexingAdvancedFlagEnvTest, EnvConfigOverrideDoesNotTouchProcessEnv) {
  // Snapshot current config and then zero out the probe counters so we
  // only observe getenv calls made during this test body.
  AdvancedIndexEnvConfig base_cfg = get_advanced_index_env_config_for_tests();
  reset_env_probe_counters_for_tests();
  EnvProbeCounters before = get_env_probe_counters_for_tests();

  AdvancedIndexEnvConfig override_cfg = base_cfg;
  override_cfg.advanced_indexing_default = !base_cfg.advanced_indexing_default;
  override_cfg.mindex32_default = !base_cfg.mindex32_default;
  override_cfg.debug_adv_index = !base_cfg.debug_adv_index;
  override_cfg.cuda_gpu_bounds_disable = !base_cfg.cuda_gpu_bounds_disable;
  override_cfg.cuda_bounds_default =
      (base_cfg.cuda_bounds_default ==
       vbt::core::indexing::detail::CudaBoundsMode::LegacyHost)
          ? vbt::core::indexing::detail::CudaBoundsMode::DeviceNormalized
          : vbt::core::indexing::detail::CudaBoundsMode::LegacyHost;
  override_cfg.cuda_max_blocks_cap = base_cfg.cuda_max_blocks_cap + 1;

  reset_advanced_index_env_config_for_tests(override_cfg);

  const auto& cfg = get_advanced_index_env_config();
  EXPECT_EQ(cfg.advanced_indexing_default, override_cfg.advanced_indexing_default);
  EXPECT_EQ(cfg.mindex32_default, override_cfg.mindex32_default);
  EXPECT_EQ(cfg.debug_adv_index, override_cfg.debug_adv_index);
  EXPECT_EQ(cfg.cuda_bounds_default, override_cfg.cuda_bounds_default);
  EXPECT_EQ(cfg.cuda_gpu_bounds_disable, override_cfg.cuda_gpu_bounds_disable);
  EXPECT_EQ(cfg.cuda_max_blocks_cap, override_cfg.cuda_max_blocks_cap);

  EnvProbeCounters after = get_env_probe_counters_for_tests();
  EXPECT_EQ(after.num_getenv_calls_enable_adv,
            before.num_getenv_calls_enable_adv);
  EXPECT_EQ(after.num_getenv_calls_mindex32_disable,
            before.num_getenv_calls_mindex32_disable);
  EXPECT_EQ(after.num_getenv_calls_debug_adv_index,
            before.num_getenv_calls_debug_adv_index);
  EXPECT_EQ(after.num_getenv_calls_cuda_gpu_bounds_disable,
            before.num_getenv_calls_cuda_gpu_bounds_disable);
  EXPECT_EQ(after.num_getenv_calls_cuda_max_blocks,
            before.num_getenv_calls_cuda_max_blocks);
  EXPECT_EQ(after.num_getenv_calls_cuda_bool_mask_indices,
            before.num_getenv_calls_cuda_bool_mask_indices);
  EXPECT_EQ(after.num_getenv_calls_cuda_bool_mask_cub,
            before.num_getenv_calls_cuda_bool_mask_cub);
  EXPECT_EQ(after.num_getenv_calls_cuda_extended_dtypes,
            before.num_getenv_calls_cuda_extended_dtypes);

  // Clear the override so later tests see env-based defaults again.
  clear_advanced_index_env_config_override_for_tests();
  (void)base_cfg;  // suppress unused warning in non-internal builds
}

TEST(IndexingAdvancedFlagEnvTest, EnvConfigGuardRestoresPreviousConfig) {
  AdvancedIndexEnvConfig original = get_advanced_index_env_config_for_tests();

  AdvancedIndexEnvConfig tmp = original;
  tmp.advanced_indexing_default = !original.advanced_indexing_default;

  {
    AdvancedIndexEnvConfigGuard guard(tmp);
    const auto& inside = get_advanced_index_env_config();
    EXPECT_EQ(inside.advanced_indexing_default, tmp.advanced_indexing_default);
  }

  const auto& restored = get_advanced_index_env_config();
  EXPECT_EQ(restored.advanced_indexing_default,
            original.advanced_indexing_default);
  EXPECT_EQ(restored.mindex32_default, original.mindex32_default);
  EXPECT_EQ(restored.debug_adv_index, original.debug_adv_index);
  EXPECT_EQ(restored.cuda_bounds_default, original.cuda_bounds_default);
  EXPECT_EQ(restored.cuda_gpu_bounds_disable, original.cuda_gpu_bounds_disable);
  EXPECT_EQ(restored.cuda_max_blocks_cap, original.cuda_max_blocks_cap);
  EXPECT_EQ(restored.cuda_allow_bool_mask_indices, original.cuda_allow_bool_mask_indices);
  EXPECT_EQ(restored.cuda_bool_mask_use_cub, original.cuda_bool_mask_use_cub);
  EXPECT_EQ(restored.cuda_allow_extended_dtypes,
            original.cuda_allow_extended_dtypes);
}

#if VBT_WITH_CUDA
TEST(IndexingAdvancedFlagEnvTest, GpuBoundsKillSwitchForcesLegacyHostMode) {
  AdvancedIndexEnvConfig base_cfg = get_advanced_index_env_config_for_tests();

  // First, confirm that when the kill-switch is off and the default is
  // DeviceNormalized, the effective mode is DeviceNormalized.
  {
    AdvancedIndexEnvConfig cfg = base_cfg;
    cfg.cuda_bounds_default = CudaBoundsMode::DeviceNormalized;
    cfg.cuda_gpu_bounds_disable = false;

    AdvancedIndexEnvConfigGuard guard(cfg);
    auto mode = get_effective_cuda_bounds_mode_for_tests();
    EXPECT_EQ(mode, CudaBoundsMode::DeviceNormalized);
  }

  // When the kill-switch is truthy, LegacyHost must win even if the
  // default bounds mode is DeviceNormalized.
  {
    AdvancedIndexEnvConfig cfg = base_cfg;
    cfg.cuda_bounds_default = CudaBoundsMode::DeviceNormalized;
    cfg.cuda_gpu_bounds_disable = true;

    AdvancedIndexEnvConfigGuard guard(cfg);
    auto mode = get_effective_cuda_bounds_mode_for_tests();
    EXPECT_EQ(mode, CudaBoundsMode::LegacyHost);
  }
}
#endif  // VBT_WITH_CUDA

#endif  // VBT_INTERNAL_TESTS
