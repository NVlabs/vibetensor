// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "vbt/core/tensor_iter.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
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

namespace {
static StoragePtr make_storage_bytes(std::size_t nbytes) {
  void* base = nullptr;
  if (nbytes > 0) {
    base = ::operator new(nbytes);
  }
  return vbt::core::make_intrusive<Storage>(
      DataPtr(base, [](void* p) noexcept { ::operator delete(p); }), nbytes);
}

static TensorImpl make_tensor(std::vector<int64_t> sizes, ScalarType dtype) {
  int64_t n = 1;
  for(auto s : sizes) n *= s;
  auto storage = make_storage_bytes(n * 4); // Enough bytes
  std::vector<int64_t> strides(sizes.size(), 1); // dummy strides
  return TensorImpl(storage, sizes, strides, 0, dtype, Device::cpu());
}
}

TEST(TensorIterStaticDeclTest, StaticShapeValidation) {
  auto t = make_tensor({10, 10}, ScalarType::Float32);
  auto out = make_tensor({10, 10}, ScalarType::Float32);

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, true));
  cfg.add_input(t);
  cfg.check_mem_overlap(false);
  
  // Correct declaration
  cfg.declare_static_shape(std::vector<int64_t>{10, 10});
  EXPECT_NO_THROW(cfg.build());
  
  // Incorrect declaration
  TensorIterConfig cfg2;
  cfg2.add_output(OptionalTensorImplRef(&out, true));
  cfg2.add_input(t);
  cfg2.check_mem_overlap(false);
  cfg2.declare_static_shape(std::vector<int64_t>{10, 20}); // Mismatch
  EXPECT_THROW(cfg2.build(), std::invalid_argument);
}

TEST(TensorIterStaticDeclTest, StaticDtypeValidation) {
  auto t = make_tensor({10}, ScalarType::Float32);
  auto out = make_tensor({10}, ScalarType::Float32);

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, true));
  cfg.add_input(t);
  cfg.check_mem_overlap(false);
  
  // Correct
  cfg.declare_static_dtype(ScalarType::Float32);
  EXPECT_NO_THROW(cfg.build());
  
  // Incorrect
  TensorIterConfig cfg2;
  cfg2.add_output(OptionalTensorImplRef(&out, true));
  cfg2.add_input(t);
  cfg2.check_mem_overlap(false);
  cfg2.declare_static_dtype(ScalarType::Int32); // Mismatch
  EXPECT_THROW(cfg2.build(), std::invalid_argument);
}
