// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "vbt/plugin/vbt_plugin.h"

// Basic ABI and layout checks for the TI-backed plugin helpers.

namespace {

static_assert(sizeof(vt_iter_overlap_mode) == sizeof(std::int32_t),
              "vt_iter_overlap_mode must remain 32-bit for ABI");

static_assert(offsetof(vt_iter_config, check_mem_overlap) == sizeof(std::int64_t),
              "vt_iter_config::check_mem_overlap must follow max_rank");

static_assert(sizeof(vt_iter_config) >= sizeof(std::int64_t) + sizeof(std::int32_t),
              "vt_iter_config must be at least {int64_t, int32_t} in size");

static_assert(std::is_standard_layout<vbt_host_api>::value,
              "vbt_host_api must be standard-layout for C ABI");

static_assert(offsetof(vbt_host_api, vt_tensor_iter_unary_cpu) >
                  offsetof(vbt_host_api, register_kernel2),
              "vt_tensor_iter_unary_cpu must be appended after register_kernel2");

static_assert(offsetof(vbt_host_api, vt_tensor_iter_binary_cpu) >
                  offsetof(vbt_host_api, vt_tensor_iter_unary_cpu),
              "vt_tensor_iter_binary_cpu must follow vt_tensor_iter_unary_cpu");

static_assert(sizeof(vbt_host_api) >=
                  offsetof(vbt_host_api, vt_tensor_iter_binary_cpu) +
                      sizeof(((vbt_host_api*)0)->vt_tensor_iter_binary_cpu),
              "vbt_host_api must contain helper pointers at the end");

static_assert(offsetof(vbt_host_api, set_device_policy) >
                  offsetof(vbt_host_api, vt_tensor_iter_destroy),
              "set_device_policy must be appended after vt_tensor_iter_destroy");

}  // namespace

TEST(TensorIterPluginHelpersAbiLayoutTest, VersionAndIterConfigLayout) {
  // ABI version bump for the plugin helper + handle surfaces.
  EXPECT_EQ(VBT_PLUGIN_ABI_VERSION_MAJOR, 1u);
  EXPECT_EQ(VBT_PLUGIN_ABI_VERSION_MINOR, 4u);

  EXPECT_EQ(sizeof(vt_iter_overlap_mode), sizeof(std::int32_t));
  EXPECT_EQ(offsetof(vt_iter_config, check_mem_overlap), sizeof(std::int64_t));
  EXPECT_GE(sizeof(vt_iter_config),
            sizeof(std::int64_t) + sizeof(std::int32_t));
}
