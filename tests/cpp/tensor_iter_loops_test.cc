// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>

#include "vbt/core/tensor_iter.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/tensor_iterator/cpu_loops.h"

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

TEST(TensorIterLoopsTest, CpuKernelRuns) {
  auto a = make_tensor({10});
  auto b = make_tensor({10});
  auto out = make_tensor({10});
  
  // Fill a, b
  float* pa = static_cast<float*>(a.data());
  float* pb = static_cast<float*>(b.data());
  for(int i=0; i<10; ++i) { pa[i] = i; pb[i] = 2*i; }
  
  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, true));
  cfg.add_input(a);
  cfg.add_input(b);
  cfg.check_mem_overlap(false);
  auto iter = cfg.build();
  
  vbt::core::ti_cpu_kernel(iter, [](char** data, const int64_t* strides, int64_t n, void*) {
      float* out_ptr = (float*)data[0];
      float* a_ptr = (float*)data[1];
      float* b_ptr = (float*)data[2];
      int64_t s_out = strides[0] / sizeof(float);
      int64_t s_a = strides[1] / sizeof(float);
      int64_t s_b = strides[2] / sizeof(float);
      
      for(int64_t i=0; i<n; ++i) {
          out_ptr[i * s_out] = a_ptr[i * s_a] + b_ptr[i * s_b];
      }
  });
  
  float* pout = static_cast<float*>(out.data());
  for(int i=0; i<10; ++i) {
      EXPECT_EQ(pout[i], 3*i);
  }
}
