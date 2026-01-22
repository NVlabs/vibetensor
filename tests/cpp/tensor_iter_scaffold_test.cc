// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "vbt/core/tensor_iter.h"

namespace vbt {
namespace core {

// Basic existence checks for core types.
static_assert(std::is_default_constructible_v<IterOperand>,
              "IterOperand should be default constructible");

static_assert(std::is_trivially_copyable_v<DeviceStrideMeta>,
              "DeviceStrideMeta should be trivially copyable");
static_assert(std::is_standard_layout_v<DeviceStrideMeta>,
              "DeviceStrideMeta should have standard layout");

static_assert(std::is_default_constructible_v<TensorIterConfig>,
              "TensorIterConfig should be default constructible");
static_assert(std::is_move_constructible_v<TensorIterConfig>,
              "TensorIterConfig should be move constructible");
static_assert(!std::is_copy_constructible_v<TensorIterConfig>,
              "TensorIterConfig should not be copy constructible");

static_assert(std::is_default_constructible_v<TensorIter>,
              "TensorIter should be default constructible");
static_assert(std::is_move_constructible_v<TensorIter>,
              "TensorIter should be move constructible");
static_assert(!std::is_copy_constructible_v<TensorIter>,
              "TensorIter should not be copy constructible");

// Signature checks for helper types.
using Loop1D = TensorIterBase::loop1d_t;
static_assert(std::is_pointer_v<Loop1D>,
              "TensorIterBase::loop1d_t should be a pointer type");

TEST(TensorIterScaffoldTest, TypesAreAvailable) {
  // This test intentionally performs no runtime work. The primary value of
  // this translation unit is the compile-time static_asserts above, which
  SUCCEED();
}

}  // namespace core
}  // namespace vbt
