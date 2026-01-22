// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "vbt/core/tensor.h"
#include "vbt/core/tensor_ops.h"
#include "vbt/core/storage.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/device.h"

#if !VBT_WITH_CUDA
#error "This test requires CUDA build"
#endif

#include <cuda_runtime_api.h>

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::ScalarType;
using vbt::core::Device;

static inline std::vector<int64_t> contig_strides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> st(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    st[static_cast<std::size_t>(i)] = acc;
    int64_t d = sizes[static_cast<std::size_t>(i)];
    if (d == 0) d = 1;
    acc *= d;
  }
  return st;
}

TEST(CudaCloneTest, Float32Transposed2D) {
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  const int dev = 0;
  const int64_t R = 2, C = 3;
  std::size_t nbytes = static_cast<std::size_t>(R*C) * sizeof(float);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  TensorImpl base(storage, {R,C}, contig_strides({R,C}), 0, ScalarType::Float32, Device{kDLCUDA, dev});
  // Fill base with 0..5 from host
  std::vector<float> h(R*C);
  for (int i=0;i<R*C;++i) h[i] = static_cast<float>(i);
  cudaMemcpy(base.data(), h.data(), nbytes, cudaMemcpyHostToDevice);

  // Transposed view: sizes [3,2], strides [1,3]
  TensorImpl t = base.as_strided({C,R}, {1,C}, 0);
  auto out = vbt::core::clone_cuda(t);
  ASSERT_TRUE(out.is_contiguous());
  ASSERT_EQ(out.sizes(), (std::vector<int64_t>{C,R}));
  std::vector<float> hout(C*R, -1.0f);
  cudaMemcpy(hout.data(), out.data(), static_cast<std::size_t>(C*R)*sizeof(float), cudaMemcpyDeviceToHost);
  for (int64_t i=0;i<C;++i) {
    for (int64_t j=0;j<R;++j) {
      float expect = h[static_cast<std::size_t>(j*C + i)];
      float got = hout[static_cast<std::size_t>(i*R + j)];
      EXPECT_EQ(got, expect);
    }
  }
}

TEST(CudaCloneTest, Float32NegativeStride1D) {
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  const int dev = 0;
  const int64_t N = 6;
  std::size_t nbytes = static_cast<std::size_t>(N) * sizeof(float);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  TensorImpl base(storage, {N}, contig_strides({N}), 0, ScalarType::Float32, Device{kDLCUDA, dev});
  std::vector<float> h(N); for (int i=0;i<N;++i) h[i] = static_cast<float>(i);
  cudaMemcpy(base.data(), h.data(), nbytes, cudaMemcpyHostToDevice);

  // Reverse view: sizes [6], strides [-1], offset 5
  TensorImpl rev = base.as_strided({N}, {-1}, /*storage_offset=*/N-1);
  auto out = vbt::core::clone_cuda(rev);
  ASSERT_TRUE(out.is_contiguous());
  std::vector<float> hout(N, -1.0f);
  cudaMemcpy(hout.data(), out.data(), nbytes, cudaMemcpyDeviceToHost);
  for (int i=0;i<N;++i) {
    EXPECT_EQ(hout[static_cast<std::size_t>(i)], h[static_cast<std::size_t>(N-1 - i)]);
  }
}

TEST(CudaCloneTest, ZeroSizeOkay) {
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  const int dev = 0;
  auto storage = vbt::cuda::new_cuda_storage(0, dev);
  TensorImpl empty(storage, {0,3}, contig_strides({0,3}), 0, ScalarType::Float32, Device{kDLCUDA, dev});
  auto out = vbt::core::clone_cuda(empty);
  EXPECT_EQ(out.numel(), 0);
  EXPECT_TRUE(out.is_contiguous());
}

TEST(CudaCloneTest, UnsupportedDtypeBool) {
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  const int dev = 0;
  const int64_t N = 4;
  std::size_t nbytes = static_cast<std::size_t>(N) * sizeof(bool);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  TensorImpl tb(storage, {N}, contig_strides({N}), 0, ScalarType::Bool, Device{kDLCUDA, dev});
  EXPECT_THROW({ (void)vbt::core::clone_cuda(tb); }, std::invalid_argument);
}
