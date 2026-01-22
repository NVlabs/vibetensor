// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cstring> // for memset

#include "vbt/core/tensor_iter.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/tensor_iterator/cpu.h"

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
  // Zero init for reduction accumulation
  if (base) std::memset(base, 0, nbytes);
  
  return vbt::core::make_intrusive<Storage>(
      DataPtr(base, [](void* p) noexcept { ::operator delete(p); }), nbytes);
}

static TensorImpl make_tensor(const std::vector<int64_t>& sizes) {
  int64_t n = 1;
  for(auto s : sizes) n *= s;
  auto storage = make_storage_bytes(n * sizeof(float));
  std::vector<int64_t> strides(sizes.size());
  int64_t st = 1;
  for (size_t i = sizes.size(); i-- > 0;) {
    strides[i] = st;
    st *= sizes[i];
  }
  return TensorImpl(storage, sizes, strides, 0, ScalarType::Float32, Device::cpu());
}
}

TEST(TensorIterReductionMultidimTest, Reduce2Dims) {
  // Reduce (2, 3, 4) over dims (0, 2) -> (3)
  auto in = make_tensor({2, 3, 4});
  auto out = make_tensor({3});

  // Fill input with 1.0
  float* din = static_cast<float*>(in.data());
  for(int i=0; i<2*3*4; ++i) din[i] = 1.0f;

  // Output initialized to 0.0 by make_storage_bytes
  
  std::vector<int64_t> dims = {0, 2};
  auto iter = TensorIter::reduce_op(out, in, dims);

  // Check iter shape/dims
  // Expected: reduce_dims has 2 entries.
  EXPECT_EQ(iter.num_reduce_dims(), 2);
  EXPECT_TRUE(iter.is_reduction());

  // Run reduction loop
  vbt::core::for_each_reduction_cpu(iter, [](char** data, const int64_t* strides, int64_t size, void*) {
      float* out_ptr = (float*)data[0];
      float* in_ptr = (float*)data[1];
      int64_t s_out = strides[0] / sizeof(float);
      int64_t s_in = strides[1] / sizeof(float);
      
      for(int64_t i=0; i<size; ++i) {
          out_ptr[i * s_out] += in_ptr[i * s_in];
      }
  }, nullptr);
  
  // Verify result
  // Each output element corresponds to reduction over dim 0 (size 2) and dim 2 (size 4).
  // Total elements reduced per output: 2 * 4 = 8.
  // Value should be 8.0.
  float* dout = static_cast<float*>(out.data());
  for(int i=0; i<3; ++i) {
      EXPECT_FLOAT_EQ(dout[i], 8.0f);
  }
}
