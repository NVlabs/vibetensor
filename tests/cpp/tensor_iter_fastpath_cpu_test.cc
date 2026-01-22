// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "vbt/core/tensor_iter.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"

#include "vbt/core/tensor_iterator/core.h"

#ifndef VBT_TI_ENABLE_TEST_HOOKS
#error "VBT_TI_ENABLE_TEST_HOOKS must be defined for tensor_iter_fastpath_cpu_test"
#endif

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::TensorIter;
using vbt::core::TensorIterConfig;
using vbt::core::OptionalTensorImplRef;
using vbt::core::testing::TensorIterTestHelper;

namespace {

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  void* base = nullptr;
  if (nbytes > 0) {
    base = ::operator new(nbytes);
  }
  return vbt::core::make_intrusive<Storage>(
      DataPtr(base, [](void* p) noexcept { ::operator delete(p); }), nbytes);
}

static TensorImpl make_cpu_contiguous_tensor(const std::vector<std::int64_t>& sizes,
                                             ScalarType dtype = ScalarType::Float32) {
  const std::size_t nd = sizes.size();
  std::vector<std::int64_t> strides(nd, 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(nd) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  std::int64_t ne = 1;
  bool any_zero = false;
  for (auto s : sizes) {
    if (s == 0) {
      any_zero = true;
      break;
    }
    ne *= s;
  }
  if (any_zero) {
    ne = 0;
  }

  const std::size_t item_b = static_cast<std::size_t>(vbt::core::itemsize(dtype));
  const std::size_t nbytes = static_cast<std::size_t>(ne) * item_b;
  auto storage = make_storage_bytes(nbytes);
  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0, dtype,
                    Device::cpu());
}

}  // namespace

TEST(TensorIterFastpathCpuTest, FastpathEnabledFor1DNonBroadcastNoD) {
  TensorImpl out = make_cpu_contiguous_tensor({8}, ScalarType::Float32);
  TensorImpl in  = make_cpu_contiguous_tensor({8}, ScalarType::Float32);

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(in);
  cfg.check_mem_overlap(false);
  TensorIter iter = cfg.build();

  EXPECT_EQ(iter.ndim(), 1);
  EXPECT_TRUE(TensorIterTestHelper::cpu_nod_contig_fastpath(iter));

  int tiles = 0;
  auto loop = [](char** /*data*/, const std::int64_t* /*strides*/, std::int64_t /*size*/, void* ctx) {
    auto* count = static_cast<int*>(ctx);
    ++(*count);
  };
  vbt::core::for_each_cpu(iter, loop, &tiles);
  EXPECT_EQ(tiles, 1);
}

TEST(TensorIterFastpathCpuTest, FastpathDisabledForBroadcastingOperand) {
  TensorImpl out = make_cpu_contiguous_tensor({4, 5}, ScalarType::Float32);
  TensorImpl in  = make_cpu_contiguous_tensor({1, 5}, ScalarType::Float32);

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(in);
  cfg.check_mem_overlap(false);
  TensorIter iter = cfg.build();

  EXPECT_GE(iter.ndim(), 1);
  EXPECT_FALSE(TensorIterTestHelper::cpu_nod_contig_fastpath(iter));
}
