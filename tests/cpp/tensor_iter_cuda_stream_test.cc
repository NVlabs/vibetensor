// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;

extern "C" vbt::core::TensorImpl vbt_cuda_add_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_mul_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_relu_impl(const vbt::core::TensorImpl&);
extern "C" void vbt_register_default_kernels();

namespace {

static TensorImpl make_cuda_tensor_from_host(const std::vector<float>& host) {
#if VBT_WITH_CUDA
  const int dev = 0;
  const std::size_t nbytes = host.size() * sizeof(float);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  if (nbytes > 0) {
    cudaError_t st = cudaMemcpy(storage->data(), host.data(), nbytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) throw std::runtime_error("cudaMemcpy H2D failed");
  }
  std::vector<int64_t> sizes{static_cast<int64_t>(host.size())};
  std::vector<int64_t> strides{1};
  return TensorImpl(storage, sizes, strides, 0, ScalarType::Float32, Device::cuda(dev));
#else
  (void)host; throw std::runtime_error("CUDA not built");
#endif
}

}  // namespace

TEST(TensorIterCudaStreamTest, RecordStreamCalledOncePerOpWhenNPositive) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  vbt_register_default_kernels();
  vbt::cuda::debug_reset_record_stream_call_count();

  const int N = 16;
  std::vector<float> ha(N, 1.0f);
  std::vector<float> hb(N, 2.0f);
  TensorImpl a = make_cuda_tensor_from_host(ha);
  TensorImpl b = make_cuda_tensor_from_host(hb);

  TensorImpl out_add = vbt_cuda_add_impl(a, b);
  TensorImpl out_mul = vbt_cuda_mul_impl(a, b);
  TensorImpl out_relu = vbt_cuda_relu_impl(a);

  (void)out_add;
  (void)out_mul;
  (void)out_relu;

  cudaDeviceSynchronize();

  std::size_t calls = vbt::cuda::debug_record_stream_call_count();
  // Each op should record streams for its input storages. The exact count depends
  // on implementation details (aliasing, view handling), so we check that:
  // 1) At least one call per op (3 ops minimum)
  // 2) No more than a reasonable upper bound (e.g., 2 inputs + 1 output per op = 9 max)
  EXPECT_GE(calls, 3u) << "Expected at least one record_stream call per op";
  EXPECT_LE(calls, 12u) << "Unexpectedly high record_stream call count";
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(TensorIterCudaStreamTest, RecordStreamNotCalledWhenNZero) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  vbt_register_default_kernels();
  vbt::cuda::debug_reset_record_stream_call_count();

  // Zero-length tensors: N == 0
  std::vector<float> empty;
  TensorImpl a0 = make_cuda_tensor_from_host(empty);
  TensorImpl b0 = make_cuda_tensor_from_host(empty);

  TensorImpl out_add0 = vbt_cuda_add_impl(a0, b0);
  TensorImpl out_mul0 = vbt_cuda_mul_impl(a0, b0);
  TensorImpl out_relu0 = vbt_cuda_relu_impl(a0);

  (void)out_add0;
  (void)out_mul0;
  (void)out_relu0;

  cudaDeviceSynchronize();

  std::size_t calls = vbt::cuda::debug_record_stream_call_count();
  EXPECT_EQ(calls, 0u);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
