// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/rng/generator.h"
#include "vbt/rng/kernels_cpu.h"
#include "vbt/rng/kernels_cuda.h"
#include "vbt/cpu/storage.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/device.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;

static vbt::core::TensorImpl make_cpu_i64(std::int64_t N) {
  std::vector<int64_t> sizes{N};
  std::vector<int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>((N > 0 ? N : 0)) * sizeof(long long);
  auto storage = vbt::cpu::new_cpu_storage(nbytes, /*pinned=*/false);
  return TensorImpl(storage, std::move(sizes), std::move(strides), 0, ScalarType::Int64, Device::cpu());
}

static vbt::core::TensorImpl make_cuda_i64(std::int64_t N, int dev) {
#if VBT_WITH_CUDA
  std::vector<int64_t> sizes{N};
  std::vector<int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>((N > 0 ? N : 0)) * sizeof(long long);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  return TensorImpl(storage, std::move(sizes), std::move(strides), 0, ScalarType::Int64, Device::cuda(dev));
#else
  (void)N; (void)dev; throw std::runtime_error("CUDA not built");
#endif
}

static std::vector<long long> copy_cuda_tensor_to_host_i64(const TensorImpl& t) {
#if VBT_WITH_CUDA
  const std::int64_t N = t.numel();
  std::vector<long long> out(static_cast<std::size_t>(N > 0 ? N : 0));
  if (N > 0) {
    cudaError_t st = cudaMemcpy(out.data(), t.data(), out.size() * sizeof(long long), cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) throw std::runtime_error("cudaMemcpy D2H failed");
  }
  return out;
#else
  (void)t; throw std::runtime_error("CUDA not built");
#endif
}

TEST(RngCudaRandint, CpuCudaParityContiguous) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  const int dev = 0;
  const std::uint64_t seeds[] = {0ull, 1ull, 123ull, 9999ull};
  const std::int64_t sizes[] = {0, 1, 2, 3, 4, 31, 32, 33, 513};
  const std::pair<long long,long long> ranges[] = {{0,1}, {0,3}, {10, 100000}, {0, (1ll<<16)}, {0, (1ll<<16)-1}};
  for (std::uint64_t seed : seeds) {
    for (auto rg : ranges) {
      for (std::int64_t N : sizes) {
        TensorImpl t_cpu = make_cpu_i64(N);
        TensorImpl t_cuda = make_cuda_i64(N, dev);
        vbt::rng::default_cpu().set_state(seed, 0);
        vbt::rng::default_cuda(dev).set_state(seed, 0);
        vbt::rng::cpu::randint_(t_cpu, rg.first, rg.second, vbt::rng::default_cpu());
        vbt::rng::cuda::randint_(t_cuda, rg.first, rg.second, vbt::rng::default_cuda(dev));
        std::vector<long long> host_cpu;
        if (N > 0) { host_cpu.resize(static_cast<std::size_t>(N)); std::memcpy(host_cpu.data(), t_cpu.data(), static_cast<std::size_t>(N) * sizeof(long long)); }
        std::vector<long long> host_cuda = copy_cuda_tensor_to_host_i64(t_cuda);
        ASSERT_EQ(host_cpu.size(), host_cuda.size());
        for (std::size_t i = 0; i < host_cpu.size(); ++i) {
          EXPECT_EQ(host_cpu[i], host_cuda[i]);
        }
      }
    }
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
