// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/tensor.h"
#include "vbt/cuda/allocator.h"
#include "vbt/cuda/cub.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::TensorImpl;

extern "C" TensorImpl vbt_cuda_sum_impl(const TensorImpl& self, std::vector<int64_t> dims, bool keepdim);

#if VBT_INTERNAL_TESTS
extern "C" std::uint64_t vbt_cuda_debug_get_cub_reduce_all_sum_fastpath_calls();
extern "C" void vbt_cuda_debug_reset_cub_reduce_all_sum_fastpath_calls();
#endif

namespace {

static TensorImpl make_cuda_f32_1d_from_host(const std::vector<float>& host) {
#if VBT_WITH_CUDA
  const int dev = 0;
  vbt::cuda::DeviceGuard dg(dev);
  const std::size_t nbytes = host.size() * sizeof(float);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  cudaError_t st = cudaMemcpy(storage->data(), host.data(), nbytes, cudaMemcpyHostToDevice);
  if (st != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy H2D failed");
  }
  std::vector<int64_t> sizes{static_cast<int64_t>(host.size())};
  std::vector<int64_t> strides{1};
  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0, ScalarType::Float32, Device::cuda(dev));
#else
  (void)host;
  throw std::runtime_error("CUDA not built");
#endif
}

static TensorImpl make_cuda_i64_1d_from_host(const std::vector<long long>& host) {
#if VBT_WITH_CUDA
  const int dev = 0;
  vbt::cuda::DeviceGuard dg(dev);
  const std::size_t nbytes = host.size() * sizeof(long long);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  cudaError_t st = cudaMemcpy(storage->data(), host.data(), nbytes, cudaMemcpyHostToDevice);
  if (st != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy H2D failed");
  }
  std::vector<int64_t> sizes{static_cast<int64_t>(host.size())};
  std::vector<int64_t> strides{1};
  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0, ScalarType::Int64, Device::cuda(dev));
#else
  (void)host;
  throw std::runtime_error("CUDA not built");
#endif
}

static TensorImpl make_cuda_f32_scalar(float v) {
#if VBT_WITH_CUDA
  const int dev = 0;
  vbt::cuda::DeviceGuard dg(dev);
  auto storage = vbt::cuda::new_cuda_storage(sizeof(float), dev);
  cudaError_t st = cudaMemcpy(storage->data(), &v, sizeof(float), cudaMemcpyHostToDevice);
  if (st != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy H2D failed");
  }
  return TensorImpl(storage, /*sizes=*/{}, /*strides=*/{}, /*storage_offset=*/0, ScalarType::Float32, Device::cuda(dev));
#else
  (void)v;
  throw std::runtime_error("CUDA not built");
#endif
}

TEST(CudaCubReduceAllSumTest, ReduceAllContigSumF32MatchesHost) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  auto& alloc = vbt::cuda::Allocator::get(0);
  auto stream = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/0);
  vbt::cuda::DeviceGuard dg(0);
  vbt::cuda::CUDAStreamGuard sg(stream);
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  const int n = 1024;
  std::vector<float> h_in(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    h_in[static_cast<std::size_t>(i)] = static_cast<float>(i % 13) - 6.0f;
  }
  float expected = std::accumulate(h_in.begin(), h_in.end(), 0.0f);

  void* d_in_void = alloc.raw_alloc(static_cast<std::size_t>(n) * sizeof(float), stream);
  ASSERT_NE(d_in_void, nullptr);
  auto* d_in = static_cast<float*>(d_in_void);

  void* d_out_void = alloc.raw_alloc(sizeof(float), stream);
  ASSERT_NE(d_out_void, nullptr);
  auto* d_out = static_cast<float*>(d_out_void);

  cudaError_t st = cudaMemcpyAsync(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice, cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  vbt::cuda::cub::reduce_all_contig_sum_f32(alloc, stream, d_in, n, d_out);

  float h_out = 0.0f;
  st = cudaMemcpyAsync(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost, cu_stream);
  ASSERT_EQ(st, cudaSuccess);
  st = cudaStreamSynchronize(cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  EXPECT_FLOAT_EQ(h_out, expected);

  alloc.raw_delete(d_in);
  alloc.raw_delete(d_out);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaCubReduceAllSumTest, ReduceAllContigSumI64MatchesHost) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  auto& alloc = vbt::cuda::Allocator::get(0);
  auto stream = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/0);
  vbt::cuda::DeviceGuard dg(0);
  vbt::cuda::CUDAStreamGuard sg(stream);
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  const int n = 2048;
  std::vector<long long> h_in(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    h_in[static_cast<std::size_t>(i)] = static_cast<long long>((i % 7) - 3);
  }
  long long expected = 0;
  for (auto v : h_in) expected += v;

  void* d_in_void = alloc.raw_alloc(static_cast<std::size_t>(n) * sizeof(long long), stream);
  ASSERT_NE(d_in_void, nullptr);
  auto* d_in = static_cast<long long*>(d_in_void);

  void* d_out_void = alloc.raw_alloc(sizeof(long long), stream);
  ASSERT_NE(d_out_void, nullptr);
  auto* d_out = static_cast<long long*>(d_out_void);

  cudaError_t st = cudaMemcpyAsync(d_in, h_in.data(), h_in.size() * sizeof(long long), cudaMemcpyHostToDevice, cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  vbt::cuda::cub::reduce_all_contig_sum_i64(alloc, stream, d_in, n, d_out);

  long long h_out = 0;
  st = cudaMemcpyAsync(&h_out, d_out, sizeof(long long), cudaMemcpyDeviceToHost, cu_stream);
  ASSERT_EQ(st, cudaSuccess);
  st = cudaStreamSynchronize(cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  EXPECT_EQ(h_out, expected);

  alloc.raw_delete(d_in);
  alloc.raw_delete(d_out);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaCubReduceAllSumTest, EmptyInputsWriteZero) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  auto& alloc = vbt::cuda::Allocator::get(0);
  auto stream = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/0);
  vbt::cuda::DeviceGuard dg(0);
  vbt::cuda::CUDAStreamGuard sg(stream);
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  void* d_out_f_void = alloc.raw_alloc(sizeof(float), stream);
  ASSERT_NE(d_out_f_void, nullptr);
  auto* d_out_f = static_cast<float*>(d_out_f_void);

  void* d_out_i_void = alloc.raw_alloc(sizeof(long long), stream);
  ASSERT_NE(d_out_i_void, nullptr);
  auto* d_out_i = static_cast<long long*>(d_out_i_void);

  vbt::cuda::cub::reduce_all_contig_sum_f32(alloc, stream, /*d_in=*/nullptr, /*n=*/0, d_out_f);
  vbt::cuda::cub::reduce_all_contig_sum_i64(alloc, stream, /*d_in=*/nullptr, /*n=*/0, d_out_i);

  float h_f = 123.0f;
  long long h_i = 123;
  cudaError_t st = cudaMemcpyAsync(&h_f, d_out_f, sizeof(float), cudaMemcpyDeviceToHost, cu_stream);
  ASSERT_EQ(st, cudaSuccess);
  st = cudaMemcpyAsync(&h_i, d_out_i, sizeof(long long), cudaMemcpyDeviceToHost, cu_stream);
  ASSERT_EQ(st, cudaSuccess);
  st = cudaStreamSynchronize(cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  EXPECT_EQ(h_f, 0.0f);
  EXPECT_EQ(h_i, 0);

  alloc.raw_delete(d_out_f);
  alloc.raw_delete(d_out_i);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaReductionTest, ScalarReduceAllSumIsIdentity) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  // Regression test: sum(dim=None) on rank-0 tensors must not throw.
  TensorImpl a = make_cuda_f32_scalar(3.5f);
  TensorImpl out = vbt_cuda_sum_impl(a, /*dims=*/{}, /*keepdim=*/false);

  float h = 0.0f;
  cudaError_t st = cudaMemcpy(&h, out.data(), sizeof(float), cudaMemcpyDeviceToHost);
  ASSERT_EQ(st, cudaSuccess);
  EXPECT_FLOAT_EQ(h, 3.5f);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaReductionTest, CubFastpathTakenWhenEnvEnabled) {
#if VBT_WITH_CUDA && VBT_INTERNAL_TESTS
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  // Enable the internal fast path.
  ASSERT_EQ(::setenv("VBT_INTERNAL_CUDA_CUB_REDUCE_ALL_SUM", "1", /*overwrite=*/1), 0);
  vbt_cuda_debug_reset_cub_reduce_all_sum_fastpath_calls();

  const int n = 8192;
  std::vector<float> h_in(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    h_in[static_cast<std::size_t>(i)] = static_cast<float>(i % 5);
  }
  float expected = std::accumulate(h_in.begin(), h_in.end(), 0.0f);

  TensorImpl a = make_cuda_f32_1d_from_host(h_in);
  TensorImpl out = vbt_cuda_sum_impl(a, /*dims=*/{}, /*keepdim=*/false);

  float h_out = 0.0f;
  cudaError_t st = cudaMemcpy(&h_out, out.data(), sizeof(float), cudaMemcpyDeviceToHost);
  ASSERT_EQ(st, cudaSuccess);
  EXPECT_FLOAT_EQ(h_out, expected);

  EXPECT_EQ(vbt_cuda_debug_get_cub_reduce_all_sum_fastpath_calls(), 1u);
#else
  GTEST_SKIP() << "Requires internal CUDA build";
#endif
}

TEST(CudaReductionTest, CubFastpathTakenForInt64WhenEnvEnabled) {
#if VBT_WITH_CUDA && VBT_INTERNAL_TESTS
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  ASSERT_EQ(::setenv("VBT_INTERNAL_CUDA_CUB_REDUCE_ALL_SUM", "1", /*overwrite=*/1), 0);
  vbt_cuda_debug_reset_cub_reduce_all_sum_fastpath_calls();

  const int n = 4096;
  std::vector<long long> h_in(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    h_in[static_cast<std::size_t>(i)] = static_cast<long long>((i % 11) - 5);
  }
  long long expected = 0;
  for (auto v : h_in) expected += v;

  TensorImpl a = make_cuda_i64_1d_from_host(h_in);
  TensorImpl out = vbt_cuda_sum_impl(a, /*dims=*/{}, /*keepdim=*/false);

  long long h_out = 0;
  cudaError_t st = cudaMemcpy(&h_out, out.data(), sizeof(long long), cudaMemcpyDeviceToHost);
  ASSERT_EQ(st, cudaSuccess);
  EXPECT_EQ(h_out, expected);

  EXPECT_EQ(vbt_cuda_debug_get_cub_reduce_all_sum_fastpath_calls(), 1u);
#else
  GTEST_SKIP() << "Requires internal CUDA build";
#endif
}

} // namespace
