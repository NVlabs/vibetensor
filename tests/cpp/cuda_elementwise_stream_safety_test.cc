// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"
#include "vbt/cuda/allocator.h"
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

extern "C" vbt::core::TensorImpl vbt_cuda_add_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" void vbt_register_default_kernels();
extern "C" void vbt_register_cuda_elementwise_kernels();

namespace {

static TensorImpl make_cuda_tensor_from_host(const std::vector<float>& host, int dev) {
#if VBT_WITH_CUDA
  const std::size_t nbytes = host.size() * sizeof(float);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  if (nbytes > 0) {
    cudaError_t st = cudaMemcpy(storage->data(), host.data(), nbytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) throw std::runtime_error("cudaMemcpy H2D failed");
  }
  std::vector<int64_t> sizes{static_cast<int64_t>(host.size())};
  std::vector<int64_t> strides{1};
  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0, ScalarType::Float32, Device::cuda(dev));
#else
  (void)host;
  (void)dev;
  throw std::runtime_error("CUDA not built");
#endif
}

#if VBT_WITH_CUDA
struct CudaFreeHostDeleter final {
  void operator()(float* p) const noexcept {
    if (p) {
      (void)cudaFreeHost(p);
    }
  }
};
#endif

}  // namespace

TEST(CUDAElementwiseStreamSafety, AddRespectsStreamAndAllocatorFences) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "Built without CUDA";
#else
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  const int dev = 0;
  vbt::cuda::DeviceGuard dg(static_cast<vbt::cuda::DeviceIndex>(dev));

  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();

  auto& alloc = vbt::cuda::Allocator::get(static_cast<vbt::cuda::DeviceIndex>(dev));
  alloc.emptyCache();

  const std::size_t N = 1u << 16;  // 64K
  std::vector<float> ha(N), hb(N);
  for (std::size_t i = 0; i < N; ++i) {
    ha[i] = static_cast<float>(i);
    hb[i] = static_cast<float>(2 * i);
  }

  float* pinned_raw = nullptr;
  ASSERT_EQ(cudaMallocHost(reinterpret_cast<void**>(&pinned_raw), N * sizeof(float)), cudaSuccess);
  std::unique_ptr<float, CudaFreeHostDeleter> pinned(pinned_raw);

  TensorImpl out;
  void* a_ptr = nullptr;
  void* b_ptr = nullptr;

  vbt::cuda::Stream s1 = vbt::cuda::getStreamFromPool(false, static_cast<vbt::cuda::DeviceIndex>(dev));

  {
    TensorImpl a = make_cuda_tensor_from_host(ha, dev);
    TensorImpl b = make_cuda_tensor_from_host(hb, dev);
    a_ptr = a.data();
    b_ptr = b.data();
    ASSERT_NE(a_ptr, nullptr);
    ASSERT_NE(b_ptr, nullptr);

    {
      vbt::cuda::CUDAStreamGuard g(s1);

      // Ensure S1 has non-trivial work after the D2H copy so that allocator
      // events recorded on S1 cannot be ready before we inspect free lists.
      constexpr std::size_t kScratchBytes = 64ull << 20;  // 64 MiB
      auto scratch = vbt::cuda::new_cuda_storage(kScratchBytes, dev);

      out = vbt_cuda_add_impl(a, b);
      ASSERT_NE(out.data(), nullptr);

      cudaStream_t s1h = reinterpret_cast<cudaStream_t>(s1.handle());
      ASSERT_EQ(
          cudaMemcpyAsync(pinned.get(), out.data(), N * sizeof(float), cudaMemcpyDeviceToHost, s1h),
          cudaSuccess);
      ASSERT_EQ(cudaMemsetAsync(scratch->data(), 0xA5, kScratchBytes, s1h), cudaSuccess);
    }
    // a and b destruct here under the default stream.
  }

  ASSERT_TRUE(alloc.owns(a_ptr));
  ASSERT_TRUE(alloc.owns(b_ptr));

  // If the add used S1 correctly, record_stream must have recorded S1 usage
  // for the inputs, so freeing under the default stream should place these
  // blocks into limbo (not free lists) until S1 work completes.
  EXPECT_FALSE(alloc.debug_is_in_free_list_for_testing(a_ptr));
  EXPECT_FALSE(alloc.debug_is_in_free_list_for_testing(b_ptr));

  s1.synchronize();
  alloc.process_events();

  EXPECT_TRUE(alloc.debug_is_in_free_list_for_testing(a_ptr));
  EXPECT_TRUE(alloc.debug_is_in_free_list_for_testing(b_ptr));

  // Validate a few points from the async D2H copy queued on S1.
  auto expect_at = [&](std::size_t i) {
    ASSERT_LT(i, N);
    EXPECT_FLOAT_EQ(pinned.get()[i], ha[i] + hb[i]) << "at index " << i;
  };
  expect_at(0);
  expect_at(1);
  expect_at(2);
  expect_at(N / 2);
  expect_at(N - 1);
#endif
}
