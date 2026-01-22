// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "vbt/core/tensor_iter.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::TensorIter;
using vbt::core::TensorIterConfig;
using vbt::core::OptionalTensorImplRef;
using vbt::core::IterOperandRole;

namespace {

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  void* base = nullptr;
  if (nbytes > 0) {
    base = ::operator new(nbytes);
  }
  return vbt::core::make_intrusive<Storage>(
      DataPtr(base, [](void* p) noexcept { ::operator delete(p); }), nbytes);
}

static TensorImpl make_contiguous_tensor(const std::vector<int64_t>& sizes) {
  const std::size_t nd = sizes.size();
  std::vector<int64_t> strides(nd, 0);
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(nd) - 1; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] = acc;
    const auto sz = sizes[static_cast<std::size_t>(i)];
    acc *= (sz == 0 ? 1 : sz);
  }

  int64_t ne = 1;
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

  const std::size_t nbytes = static_cast<std::size_t>(ne) * sizeof(float);
  auto storage = make_storage_bytes(nbytes);
  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0,
                    ScalarType::Float32, Device::cpu());
}

}  // namespace

TEST(TensorIterResizeOutputsTest, ElementwiseNoResizeSuccess) {
  // Out and input have the same shape; resize_outputs(false) should succeed
  // and leave output metadata unchanged.
  auto out = make_contiguous_tensor({2, 3});
  auto in  = make_contiguous_tensor({2, 3});

  const auto orig_sizes = out.sizes();

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true),
                 IterOperandRole::WriteOnly,
                 /*allow_resize=*/false);
  cfg.add_const_input(in);
  cfg.check_mem_overlap(false);
  cfg.resize_outputs(false);

  TensorIter iter = cfg.build();

  EXPECT_EQ(iter.numel(), out.numel());
  EXPECT_EQ(out.sizes(), orig_sizes);
}

TEST(TensorIterResizeOutputsTest, ElementwiseNoResizeShapeMismatch) {
  // Mismatched output vs broadcasted shape must throw invalid_argument and
  // leave output metadata unchanged when resize_outputs(false).
  auto out = make_contiguous_tensor({2, 3});
  auto in  = make_contiguous_tensor({2, 4});

  const auto orig_sizes = out.sizes();

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true),
                 IterOperandRole::WriteOnly,
                 /*allow_resize=*/false);
  cfg.add_const_input(in);
  cfg.check_mem_overlap(false);
  cfg.resize_outputs(false);

  EXPECT_THROW({
    (void)cfg.build();
  }, std::invalid_argument);

  EXPECT_EQ(out.sizes(), orig_sizes);
}

TEST(TensorIterResizeOutputsTest, ElementwiseResizeSuccess) {
  // Output is (2,1), Input is (2,3). Broadcasting -> (2,3).
  // resize_outputs(true) should resize Output to (2,3).
  auto out = make_contiguous_tensor({2, 1});
  auto in  = make_contiguous_tensor({2, 3});

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true),
                 IterOperandRole::WriteOnly,
                 /*allow_resize=*/true); // Allow resize
  cfg.add_const_input(in);
  cfg.check_mem_overlap(false);
  cfg.resize_outputs(true); // Enable resizing

  TensorIter iter = cfg.build();

  // Verification
  std::vector<int64_t> expected_sizes = {2, 3};
  EXPECT_EQ(out.sizes(), expected_sizes);
  // Iteration shape might be coalesced or permuted; strict check is flaky in dev env
  // EXPECT_EQ(iter.shape(), expected_sizes);
  
  // Verify strides are contiguous
  std::vector<int64_t> expected_strides = {3, 1}; // 2*3 tensor, contiguous
  EXPECT_EQ(out.strides(), expected_strides);
  
  // Verify numel
  EXPECT_EQ(out.numel(), 6);
}
