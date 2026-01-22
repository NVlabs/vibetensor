// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/rng/generator.h"
#include "vbt/rng/kernels_cpu.h"
#include "vbt/rng/kernels_cuda.h"
#include "vbt/rng/philox_util.h"
#include "vbt/cpu/storage.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;

static TensorImpl make_cpu_tensor(std::int64_t N) {
  std::vector<int64_t> sizes{N};
  std::vector<int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>((N > 0 ? N : 0)) * sizeof(float);
  auto storage = vbt::cpu::new_cpu_storage(nbytes, /*pinned=*/false);
  return TensorImpl(storage, std::move(sizes), std::move(strides), /*storage_offset=*/0,
                    ScalarType::Float32, Device::cpu());
}

static TensorImpl make_cuda_tensor(std::int64_t N, int dev) {
#if VBT_WITH_CUDA
  std::vector<int64_t> sizes{N};
  std::vector<int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>((N > 0 ? N : 0)) * sizeof(float);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  return TensorImpl(storage, std::move(sizes), std::move(strides), /*storage_offset=*/0,
                    ScalarType::Float32, Device::cuda(dev));
#else
  (void)N; (void)dev; throw std::runtime_error("CUDA not built");
#endif
}

static std::vector<float> copy_cuda_tensor_to_host(const TensorImpl& t) {
#if VBT_WITH_CUDA
  const std::int64_t N = t.numel();
  std::vector<float> out(static_cast<std::size_t>(N > 0 ? N : 0));
  if (N > 0) {
    cudaError_t st = cudaMemcpy(out.data(), t.data(), out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) throw std::runtime_error("cudaMemcpy D2H failed");
  }
  return out;
#else
  (void)t; throw std::runtime_error("CUDA not built");
#endif
}

TEST(RngCudaUniform, CpuCudaParityContiguous) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  const int dev = 0;
  const std::uint64_t seeds[] = {0ull, 1ull, 123ull, 9999ull};
  const std::int64_t sizes[] = {0, 1, 2, 3, 4, 31, 32, 33, 513};
  const float lows[] = {0.0f, -1.0f};
  const float highs[] = {1.0f,  2.0f};

  for (std::uint64_t seed : seeds) {
    for (std::size_t li = 0; li < 2; ++li) {
      const float low = lows[li];
      const float high = highs[li];
      for (std::int64_t N : sizes) {
        TensorImpl t_cpu = make_cpu_tensor(N);
        TensorImpl t_cuda = make_cuda_tensor(N, dev);

        vbt::rng::default_cpu().set_state(seed, 0);
        vbt::rng::default_cuda(dev).set_state(seed, 0);

        vbt::rng::cpu::uniform_(t_cpu, low, high, vbt::rng::default_cpu());
        vbt::rng::cuda::uniform_(t_cuda, low, high, vbt::rng::default_cuda(dev));

        // Copy CPU tensor to host
        std::vector<float> host_cpu;
        const std::int64_t n64 = t_cpu.numel();
        if (n64 > 0) {
          const std::size_t nbytes = static_cast<std::size_t>(n64) * sizeof(float);
          host_cpu.resize(static_cast<std::size_t>(n64));
          std::memcpy(host_cpu.data(), t_cpu.data(), nbytes);
        }

        // Copy CUDA tensor to host
        std::vector<float> host_cuda = copy_cuda_tensor_to_host(t_cuda);

        ASSERT_EQ(host_cpu.size(), host_cuda.size());
        for (std::size_t i = 0; i < host_cpu.size(); ++i) {
          EXPECT_NEAR(host_cpu[i], host_cuda[i], 1e-6f);
        }

        // Offsets should advance identically by ceil_div(N,4)
        auto st_cpu = vbt::rng::default_cpu().get_state();
        auto st_cuda = vbt::rng::default_cuda(dev).get_state();
        EXPECT_EQ(st_cpu.seed, seed);
        EXPECT_EQ(st_cuda.seed, seed);
        const std::uint64_t expected_off = vbt::rng::ceil_div_u64(static_cast<std::uint64_t>(N > 0 ? N : 0), 4ull);
        EXPECT_EQ(st_cpu.offset, expected_off);
        EXPECT_EQ(st_cuda.offset, expected_off);
      }
    }
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(RngCudaUniform, ScheduleIndependenceEnvKnobs) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  const int dev = 0;
  const std::int64_t N = 4097; // cover multiple blocks and tail
  const std::uint64_t seed = 1234ull;

  auto run_with_env = [&](const char* tpb, const char* bpt) {
    if (tpb) setenv("VBT_RNG_CUDA_BLOCK", tpb, 1); else unsetenv("VBT_RNG_CUDA_BLOCK");
    if (bpt) setenv("VBT_RNG_CUDA_BLOCKS_PER_THREAD", bpt, 1); else unsetenv("VBT_RNG_CUDA_BLOCKS_PER_THREAD");

    TensorImpl t = make_cuda_tensor(N, dev);
    vbt::rng::default_cuda(dev).set_state(seed, 0);
    vbt::rng::cuda::uniform_(t, 0.0f, 1.0f, vbt::rng::default_cuda(dev));
    return copy_cuda_tensor_to_host(t);
  };

  // Baseline
  auto ref = run_with_env("256", "4");

  // Vary threads-per-block and blocks-per-thread; outputs must remain identical
  auto a = run_with_env("128", "4");
  auto b = run_with_env("512", "4");
  auto c = run_with_env("256", "1");
  auto d = run_with_env("256", "8");

  ASSERT_EQ(ref.size(), a.size());
  ASSERT_EQ(ref.size(), b.size());
  ASSERT_EQ(ref.size(), c.size());
  ASSERT_EQ(ref.size(), d.size());

  for (std::size_t i = 0; i < ref.size(); ++i) {
    EXPECT_FLOAT_EQ(ref[i], a[i]);
    EXPECT_FLOAT_EQ(ref[i], b[i]);
    EXPECT_FLOAT_EQ(ref[i], c[i]);
    EXPECT_FLOAT_EQ(ref[i], d[i]);
  }

  auto st_ref = vbt::rng::default_cuda(dev).get_state();
  const std::uint64_t expected_off = vbt::rng::ceil_div_u64(static_cast<std::uint64_t>(N), 4ull);
  EXPECT_EQ(st_ref.seed, seed);
  EXPECT_EQ(st_ref.offset, expected_off);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
