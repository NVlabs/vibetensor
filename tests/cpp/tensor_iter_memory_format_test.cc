// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <numeric>

#include "vbt/core/tensor_iter.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
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

static TensorImpl make_channels_last_tensor(int64_t n, int64_t c, int64_t h, int64_t w) {
  std::vector<int64_t> sizes = {n, c, h, w};
  // Strides for NHWC: (H*W*C, 1, W*C, C)
  std::vector<int64_t> strides(4);
  strides[1] = 1;
  strides[3] = c;
  strides[2] = w * c;
  strides[0] = h * w * c;

  int64_t numel = n * c * h * w;
  auto storage = make_storage_bytes(numel * sizeof(float));
  return TensorImpl(storage, sizes, strides, 0, ScalarType::Float32, Device::cpu());
}

static TensorImpl make_contiguous_tensor(const std::vector<int64_t>& sizes) {
  // Row major
  std::vector<int64_t> strides(sizes.size());
  int64_t st = 1;
  for (size_t i = sizes.size(); i-- > 0;) {
    strides[i] = st;
    st *= sizes[i];
  }
  auto storage = make_storage_bytes(st * sizeof(float));
  return TensorImpl(storage, sizes, strides, 0, ScalarType::Float32, Device::cpu());
}

} // namespace

TEST(TensorIterMemoryFormatTest, PreservesChannelsLastOnResize) {
  // Input: (2, 3, 4, 5) Channels Last
  auto in = make_channels_last_tensor(2, 3, 4, 5);
  
  // Output: Undefined / empty
  auto out_storage = make_storage_bytes(0);
  TensorImpl out(out_storage, {}, {}, 0, ScalarType::Float32, Device::cpu());

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true), 
                 IterOperandRole::WriteOnly, /*allow_resize=*/true);
  cfg.add_input(in);
  cfg.check_mem_overlap(false);
  cfg.resize_outputs(true);

  TensorIter iter = cfg.build();

  EXPECT_EQ(out.sizes(), std::vector<int64_t>({2, 3, 4, 5}));
  // Verify output is channels last
  EXPECT_TRUE(out.is_channels_last());
  EXPECT_EQ(out.strides(), in.strides());
}

TEST(TensorIterMemoryFormatTest, MixedFormatDefaultsToContiguous) {
  // Input 1: Channels Last
  auto in1 = make_channels_last_tensor(2, 3, 4, 5);
  // Input 2: Contiguous
  auto in2 = make_contiguous_tensor({2, 3, 4, 5});

  auto out_storage = make_storage_bytes(0);
  TensorImpl out(out_storage, {}, {}, 0, ScalarType::Float32, Device::cpu());

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true),
                 IterOperandRole::WriteOnly, /*allow_resize=*/true);
  cfg.add_input(in1);
  cfg.add_input(in2);
  cfg.check_mem_overlap(false);
  cfg.resize_outputs(true);

  cfg.build();

  // Should be contiguous because inputs differ
  EXPECT_EQ(out.sizes(), std::vector<int64_t>({2, 3, 4, 5}));
  EXPECT_FALSE(out.is_channels_last());
  EXPECT_EQ(out.strides(), in2.strides());
}
