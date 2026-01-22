// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "vbt/plugin/vbt_plugin.h"

// C++-side ABI and layout checks for the vt_tensor_iter handle & metadata APIs.

namespace {

static_assert(std::is_standard_layout<vbt_host_api>::value,
              "vbt_host_api must be standard-layout for C ABI");

static_assert(std::is_standard_layout<vt_tensor_iter_desc>::value,
              "vt_tensor_iter_desc must be standard-layout");

static_assert(std::is_standard_layout<vt_tensor_iter_alias_info>::value,
              "vt_tensor_iter_alias_info must be standard-layout");

static_assert(std::is_standard_layout<vt_tensor_iter_cuda_desc>::value,
              "vt_tensor_iter_cuda_desc must be standard-layout");

// Ensure new function pointers are appended after the helper pointers.
static_assert(offsetof(vbt_host_api, vt_tensor_iter_build_elementwise) >
                  offsetof(vbt_host_api, vt_tensor_iter_binary_cpu),
              "vt_tensor_iter_build_elementwise must follow vt_tensor_iter_binary_cpu");

static_assert(offsetof(vbt_host_api, vt_tensor_iter_destroy) >=
                  offsetof(vbt_host_api, vt_tensor_iter_build_elementwise),
              "vt_tensor_iter_destroy must be appended after handle builders");

static_assert(offsetof(vbt_host_api, set_device_policy) >
                  offsetof(vbt_host_api, vt_tensor_iter_destroy),
              "set_device_policy must be appended after vt_tensor_iter_destroy");

static_assert(sizeof(vbt_host_api) >=
                  offsetof(vbt_host_api, set_device_policy) +
                      sizeof(((vbt_host_api*)0)->set_device_policy),
              "vbt_host_api must contain handle function pointers at the end");

}  // namespace

TEST(TensorIterPluginHandleAbiLayoutTest, VersionAndConstants) {
  EXPECT_EQ(VBT_PLUGIN_ABI_VERSION_MAJOR, 1u);
  EXPECT_EQ(VBT_PLUGIN_ABI_VERSION_MINOR, 4u);

  EXPECT_EQ(VT_TENSOR_ITER_MAX_RANK, 64);
  EXPECT_EQ(VT_TENSOR_ITER_MAX_OPERANDS, 64);
  EXPECT_LE(VT_TENSOR_ITER_CUDA_MAX_NDIM, VT_TENSOR_ITER_MAX_RANK);
}
