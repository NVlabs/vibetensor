// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/checked_math.h"

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/reduction_env.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using vbt::core::TensorImpl;
using vbt::core::StoragePtr;
using vbt::core::ScalarType;
using vbt::core::Device;

extern "C" vbt::core::TensorImpl vbt_cuda_sum_impl(const vbt::core::TensorImpl&,
                                                    std::vector<int64_t> dims,
                                                    bool keepdim);

namespace {

static TensorImpl make_cuda_contiguous_2d_f32_from_host(vbt::cuda::DeviceIndex dev,
                                                       std::int64_t rows,
                                                       std::int64_t cols,
                                                       const std::vector<float>& host) {
#if VBT_WITH_CUDA
  if (rows < 0 || cols < 0) throw std::invalid_argument("rows/cols must be >=0");

  std::int64_t numel = 0;
  if (!vbt::core::checked_mul_i64(rows, cols, numel)) {
    throw std::invalid_argument("numel overflow");
  }

  if (host.size() != static_cast<std::size_t>(numel)) {
    throw std::invalid_argument("host size mismatch");
  }

  const std::size_t nbytes = host.size() * sizeof(float);
  StoragePtr storage = vbt::cuda::new_cuda_storage(nbytes, static_cast<int>(dev));

  if (nbytes > 0) {
    cudaError_t st = cudaMemcpy(storage->data(), host.data(), nbytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy H2D failed");
    }
  }

  std::vector<int64_t> sizes{rows, cols};
  std::vector<int64_t> strides{cols, 1};
  return TensorImpl(storage,
                    sizes,
                    strides,
                    /*storage_offset=*/0,
                    ScalarType::Float32,
                    Device::cuda(static_cast<int>(dev)));
#else
  (void)dev;
  (void)rows;
  (void)cols;
  (void)host;
  throw std::runtime_error("CUDA not built");
#endif
}

struct ReductionTestOverridesGuard {
#if VBT_INTERNAL_TESTS
  ~ReductionTestOverridesGuard() {
    vbt::cuda::reduction::clear_cuda_reduction_kernel_policy_override_for_tests();
    vbt::cuda::reduction::clear_cuda_reduction_grid_x_cap_override_for_tests();
    vbt::cuda::reduction::clear_cuda_reduction_env_config_override_for_tests();
    vbt::cuda::reduction::clear_cuda_reduction_k2multi_ctas_per_output_override_for_tests();
    vbt::cuda::reduction::clear_cuda_reduction_k2multi_fault_mode_override_for_tests();
    vbt::cuda::reduction::clear_cuda_reduction_k2multi_stream_mismatch_injection_override_for_tests();
  }
#else
  ~ReductionTestOverridesGuard() = default;
#endif
};

}  // namespace

TEST(CudaReductionRecordStreamHazardTest, K2MultiWorkspaceStreamMismatchDoesNotCorruptOutput) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "Built without CUDA";
#else
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  constexpr vbt::cuda::DeviceIndex dev = 0;

  // Start from a clean allocator state so the workspace reuse behavior is
  // deterministic.
  auto& alloc = vbt::cuda::Allocator::get(dev);
  alloc.emptyCache();

  // Keep allocations on the default stream.
  vbt::cuda::Stream default_stream = vbt::cuda::getDefaultStream(dev);
  vbt::cuda::setCurrentStream(default_stream);

#if !VBT_INTERNAL_TESTS
  GTEST_SKIP() << "Reduction hazard test requires VBT_INTERNAL_TESTS";
#else
  using vbt::cuda::reduction::CudaReductionEnvConfig;
  using vbt::cuda::reduction::CudaReductionKernel;
  using vbt::cuda::reduction::CudaReductionKernelPolicy;
  using vbt::cuda::reduction::CudaReductionLastStats;

  ReductionTestOverridesGuard guard;

  vbt::cuda::reduction::reset_cuda_reduction_env_config_for_tests(
      CudaReductionEnvConfig{.staged_default = false, .cuda_max_blocks_cap = 0});

  vbt::cuda::reduction::set_cuda_reduction_k2multi_ctas_per_output_for_tests(2u);
  vbt::cuda::reduction::set_cuda_reduction_kernel_policy_for_tests(
      CudaReductionKernelPolicy::ForceK2MultiStrict);
  vbt::cuda::reduction::set_cuda_reduction_k2multi_stream_mismatch_injection_enabled_for_tests(
      true);

  // Large enough that the K2-multi kernel is still pending when we attempt to
  // allocate and clobber a same-sized buffer on the default stream.
  constexpr std::int64_t rows = 1;
  constexpr std::int64_t cols = (1ll << 22);  // 4,194,304 (exactly representable in float)

  const std::size_t numel = static_cast<std::size_t>(rows * cols);
  std::vector<float> host(numel, 1.0f);

  TensorImpl x = make_cuda_contiguous_2d_f32_from_host(dev, rows, cols, host);

  vbt::cuda::reduction::reset_cuda_reduction_last_stats_for_tests();
  TensorImpl out = vbt_cuda_sum_impl(x, /*dims=*/{1}, /*keepdim=*/false);

  const CudaReductionLastStats stats =
      vbt::cuda::reduction::get_cuda_reduction_last_stats_for_tests();
  ASSERT_EQ(stats.selected_kernel, CudaReductionKernel::K2Multi);
  ASSERT_NE(stats.launch_stream_id, 0u);

  // Ensure the launch stream is still busy. If it isn't, the clobber below
  // becomes a no-op (even if the allocator incorrectly reuses the workspace).
  vbt::cuda::Stream launch_stream(vbt::cuda::Stream::UNCHECKED,
                                 stats.launch_stream_id,
                                 dev);
  ASSERT_FALSE(launch_stream.query());

  const std::size_t ws_bytes =
      static_cast<std::size_t>(stats.k2multi_workspace_total_bytes);
  ASSERT_GT(ws_bytes, 0u);

  StoragePtr clobber = vbt::cuda::new_cuda_storage(ws_bytes, static_cast<int>(dev));

  // Corrupt the buffer with 0xFF. If it aliases the workspace (because
  // record_stream was missing/wrong), the K2-multi semaphore protocol will be
  // violated and output values will be incorrect.
  cudaError_t st_memset = cudaMemsetAsync(
      clobber->data(),
      0xFF,
      ws_bytes,
      reinterpret_cast<cudaStream_t>(default_stream.handle()));
  ASSERT_EQ(st_memset, cudaSuccess);

  cudaError_t st_sync = cudaDeviceSynchronize();
  ASSERT_EQ(st_sync, cudaSuccess);

  ASSERT_EQ(out.numel(), rows);
  std::vector<float> host_out(static_cast<std::size_t>(out.numel()));
  cudaError_t st_d2h = cudaMemcpy(host_out.data(),
                                 out.data(),
                                 host_out.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost);
  ASSERT_EQ(st_d2h, cudaSuccess);

  const float expected = static_cast<float>(cols);
  EXPECT_EQ(host_out[0], expected);
#endif  // VBT_INTERNAL_TESTS
#endif  // VBT_WITH_CUDA
}
