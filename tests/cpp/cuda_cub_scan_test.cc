// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdlib>
#include <numeric>
#include <string>
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

static std::vector<int> host_exclusive_scan(const std::vector<int>& in) {
  std::vector<int> out(in.size());
  int running = 0;
  for (std::size_t i = 0; i < in.size(); ++i) {
    out[i] = running;
    running += in[i];
  }
  return out;
}

class ScopedEnvRestore final {
 public:
  explicit ScopedEnvRestore(const char* key) : key_(key) {
    const char* v = std::getenv(key_);
    had_prev_ = (v != nullptr);
    if (had_prev_) {
      prev_ = v;
    }
  }

  ScopedEnvRestore(const ScopedEnvRestore&) = delete;
  ScopedEnvRestore& operator=(const ScopedEnvRestore&) = delete;

  ~ScopedEnvRestore() {
    if (had_prev_) {
      (void)::setenv(key_, prev_.c_str(), 1);
    } else {
      (void)::unsetenv(key_);
    }
  }

 private:
  const char* key_;
  bool had_prev_{false};
  std::string prev_;
};

TEST(CudaCubScanTest, ExclusiveScanI32MatchesHost) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  auto& alloc = vbt::cuda::Allocator::get(0);
  auto stream = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/0);
  vbt::cuda::DeviceGuard dg(0);
  vbt::cuda::CUDAStreamGuard sg(stream);
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  const std::vector<int> h_in{1, 2, 3, 4, 5, 6, 7};
  const int n = static_cast<int>(h_in.size());

  void* d_in_void = alloc.raw_alloc(static_cast<std::size_t>(n) * sizeof(int), stream);
  ASSERT_NE(d_in_void, nullptr);
  auto* d_in = static_cast<int*>(d_in_void);

  void* d_out_void = alloc.raw_alloc(static_cast<std::size_t>(n) * sizeof(int), stream);
  ASSERT_NE(d_out_void, nullptr);
  auto* d_out = static_cast<int*>(d_out_void);

  cudaError_t st = cudaMemcpyAsync(
      d_in,
      h_in.data(),
      static_cast<std::size_t>(n) * sizeof(int),
      cudaMemcpyHostToDevice,
      cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  vbt::cuda::cub::exclusive_scan_i32(alloc, stream, d_in, n, d_out);

  std::vector<int> h_out(static_cast<std::size_t>(n), -1);
  st = cudaMemcpyAsync(
      h_out.data(),
      d_out,
      static_cast<std::size_t>(n) * sizeof(int),
      cudaMemcpyDeviceToHost,
      cu_stream);
  ASSERT_EQ(st, cudaSuccess);
  st = cudaStreamSynchronize(cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  const std::vector<int> expected = host_exclusive_scan(h_in);
  EXPECT_EQ(h_out, expected);

  alloc.raw_delete(d_in);
  alloc.raw_delete(d_out);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaCubScanTest, ExclusiveScanI32ChunkedMatchesHost) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  const char* kEnv = "VBT_INTERNAL_CUDA_CUB_SCAN_MAX_ITEMS";
  ScopedEnvRestore env_restore(kEnv);

  // Force a tiny chunk size to exercise the multi-chunk path.
  ASSERT_EQ(::setenv(kEnv, "4", 1), 0);

  auto& alloc = vbt::cuda::Allocator::get(0);
  auto stream = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/0);
  vbt::cuda::DeviceGuard dg(0);
  vbt::cuda::CUDAStreamGuard sg(stream);
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream.handle());

  std::vector<int> h_in;
  for (int i = 0; i < 17; ++i) {
    h_in.push_back(i % 3);
  }
  const int n = static_cast<int>(h_in.size());

  void* d_in_void = alloc.raw_alloc(static_cast<std::size_t>(n) * sizeof(int), stream);
  ASSERT_NE(d_in_void, nullptr);
  auto* d_in = static_cast<int*>(d_in_void);

  void* d_out_void = alloc.raw_alloc(static_cast<std::size_t>(n) * sizeof(int), stream);
  ASSERT_NE(d_out_void, nullptr);
  auto* d_out = static_cast<int*>(d_out_void);

  cudaError_t st = cudaMemcpyAsync(
      d_in,
      h_in.data(),
      static_cast<std::size_t>(n) * sizeof(int),
      cudaMemcpyHostToDevice,
      cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  vbt::cuda::cub::exclusive_scan_i32(alloc, stream, d_in, n, d_out);

  std::vector<int> h_out(static_cast<std::size_t>(n), -1);
  st = cudaMemcpyAsync(
      h_out.data(),
      d_out,
      static_cast<std::size_t>(n) * sizeof(int),
      cudaMemcpyDeviceToHost,
      cu_stream);
  ASSERT_EQ(st, cudaSuccess);
  st = cudaStreamSynchronize(cu_stream);
  ASSERT_EQ(st, cudaSuccess);

  const std::vector<int> expected = host_exclusive_scan(h_in);
  EXPECT_EQ(h_out, expected);

  alloc.raw_delete(d_in);
  alloc.raw_delete(d_out);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CudaCubScanTest, EmptyInputsNoThrow) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  auto& alloc = vbt::cuda::Allocator::get(0);
  auto stream = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/0);

  EXPECT_NO_THROW(vbt::cuda::cub::exclusive_scan_i32(alloc, stream, /*d_in=*/nullptr, /*n=*/0, /*d_out=*/nullptr));
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

} // namespace
