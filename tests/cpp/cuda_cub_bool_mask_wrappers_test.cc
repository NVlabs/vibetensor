// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/cub.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/stream.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {

TEST(CudaCubWrappersTest, ReduceSumU8AsI32MatchesHost) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  auto& alloc = vbt::cuda::Allocator::get(0);
  auto stream = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/0);
  vbt::cuda::DeviceGuard dg(0);
  vbt::cuda::CUDAStreamGuard sg(stream);
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  const std::vector<std::uint8_t> h_flags{0, 1, 0, 1, 1, 0, 1};
  const int n = static_cast<int>(h_flags.size());

  void* d_flags_void = alloc.raw_alloc(static_cast<std::size_t>(n) * sizeof(std::uint8_t), stream);
  ASSERT_NE(d_flags_void, nullptr);
  auto* d_flags = static_cast<std::uint8_t*>(d_flags_void);

  void* d_sum_void = alloc.raw_alloc(sizeof(int), stream);
  ASSERT_NE(d_sum_void, nullptr);
  auto* d_sum = static_cast<int*>(d_sum_void);

  cudaError_t st = cudaMemcpyAsync(d_flags, h_flags.data(), static_cast<std::size_t>(n), cudaMemcpyHostToDevice, cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  vbt::cuda::cub::reduce_sum_u8_as_i32(alloc, stream, d_flags, n, d_sum);

  int h_sum = -1;
  st = cudaMemcpyAsync(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost, cu_stream);
  ASSERT_EQ(st, cudaSuccess);
  st = cudaStreamSynchronize(cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  int expected = 0;
  for (std::uint8_t v : h_flags) expected += (v != 0) ? 1 : 0;
  EXPECT_EQ(h_sum, expected);

  alloc.raw_delete(d_flags);
  alloc.raw_delete(d_sum);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaCubWrappersTest, ReduceSumU8AsI32LargeNAllOnes) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  auto& alloc = vbt::cuda::Allocator::get(0);
  auto stream = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/0);
  vbt::cuda::DeviceGuard dg(0);
  vbt::cuda::CUDAStreamGuard sg(stream);
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  const int n = 1024;
  const std::vector<std::uint8_t> h_flags(static_cast<std::size_t>(n), 1);

  void* d_flags_void = alloc.raw_alloc(static_cast<std::size_t>(n) * sizeof(std::uint8_t), stream);
  ASSERT_NE(d_flags_void, nullptr);
  auto* d_flags = static_cast<std::uint8_t*>(d_flags_void);

  void* d_sum_void = alloc.raw_alloc(sizeof(int), stream);
  ASSERT_NE(d_sum_void, nullptr);
  auto* d_sum = static_cast<int*>(d_sum_void);

  cudaError_t st =
      cudaMemcpyAsync(d_flags, h_flags.data(), static_cast<std::size_t>(n), cudaMemcpyHostToDevice, cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  vbt::cuda::cub::reduce_sum_u8_as_i32(alloc, stream, d_flags, n, d_sum);

  int h_sum = -1;
  st = cudaMemcpyAsync(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost, cu_stream);
  ASSERT_EQ(st, cudaSuccess);
  st = cudaStreamSynchronize(cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  EXPECT_EQ(h_sum, n);

  alloc.raw_delete(d_flags);
  alloc.raw_delete(d_sum);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaCubWrappersTest, SelectIndicesFromU8FlagsI64MatchesHost) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  auto& alloc = vbt::cuda::Allocator::get(0);
  auto stream = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/0);
  vbt::cuda::DeviceGuard dg(0);
  vbt::cuda::CUDAStreamGuard sg(stream);
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  const std::vector<std::uint8_t> h_flags{0, 1, 0, 1, 1};
  const int n = static_cast<int>(h_flags.size());

  void* d_flags_void = alloc.raw_alloc(static_cast<std::size_t>(n) * sizeof(std::uint8_t), stream);
  ASSERT_NE(d_flags_void, nullptr);
  auto* d_flags = static_cast<std::uint8_t*>(d_flags_void);

  void* d_indices_void = alloc.raw_alloc(static_cast<std::size_t>(n) * sizeof(std::int64_t), stream);
  ASSERT_NE(d_indices_void, nullptr);
  auto* d_indices = static_cast<std::int64_t*>(d_indices_void);

  void* d_count_void = alloc.raw_alloc(sizeof(int), stream);
  ASSERT_NE(d_count_void, nullptr);
  auto* d_count = static_cast<int*>(d_count_void);

  cudaError_t st = cudaMemcpyAsync(d_flags, h_flags.data(), static_cast<std::size_t>(n), cudaMemcpyHostToDevice, cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  vbt::cuda::cub::select_indices_from_u8_flags_i64(alloc, stream, d_flags, n, d_indices, d_count);

  int h_count = -1;
  std::vector<std::int64_t> h_indices(static_cast<std::size_t>(n), -1);

  st = cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, cu_stream);
  ASSERT_EQ(st, cudaSuccess);
  st = cudaMemcpyAsync(
      h_indices.data(), d_indices, h_indices.size() * sizeof(std::int64_t), cudaMemcpyDeviceToHost, cu_stream);
  ASSERT_EQ(st, cudaSuccess);
  st = cudaStreamSynchronize(cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  std::vector<std::int64_t> expected;
  for (int i = 0; i < n; ++i) {
    if (h_flags[static_cast<std::size_t>(i)] != 0) expected.push_back(i);
  }
  ASSERT_EQ(h_count, static_cast<int>(expected.size()));
  for (int i = 0; i < h_count; ++i) {
    EXPECT_EQ(h_indices[static_cast<std::size_t>(i)], expected[static_cast<std::size_t>(i)]);
  }

  alloc.raw_delete(d_flags);
  alloc.raw_delete(d_indices);
  alloc.raw_delete(d_count);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaCubWrappersTest, EmptyInputsWriteZero) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  auto& alloc = vbt::cuda::Allocator::get(0);
  auto stream = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/0);
  vbt::cuda::DeviceGuard dg(0);
  vbt::cuda::CUDAStreamGuard sg(stream);
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  void* d_sum_void = alloc.raw_alloc(sizeof(int), stream);
  ASSERT_NE(d_sum_void, nullptr);
  auto* d_sum = static_cast<int*>(d_sum_void);

  void* d_count_void = alloc.raw_alloc(sizeof(int), stream);
  ASSERT_NE(d_count_void, nullptr);
  auto* d_count = static_cast<int*>(d_count_void);

  vbt::cuda::cub::reduce_sum_u8_as_i32(alloc, stream, /*d_flags01=*/nullptr, /*n=*/0, d_sum);
  vbt::cuda::cub::select_indices_from_u8_flags_i64(
      alloc, stream, /*d_flags01=*/nullptr, /*n=*/0, /*d_out_indices=*/nullptr, d_count);

  int h_sum = -1;
  int h_count = -1;
  cudaError_t st = cudaMemcpyAsync(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost, cu_stream);
  ASSERT_EQ(st, cudaSuccess);
  st = cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, cu_stream);
  ASSERT_EQ(st, cudaSuccess);
  st = cudaStreamSynchronize(cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  EXPECT_EQ(h_sum, 0);
  EXPECT_EQ(h_count, 0);

  alloc.raw_delete(d_sum);
  alloc.raw_delete(d_count);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

} // namespace
