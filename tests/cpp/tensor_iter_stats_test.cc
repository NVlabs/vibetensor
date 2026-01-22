// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "vbt/core/tensor_iterator/core.h"
#include "vbt/core/tensor_iterator/cpu.h"
#include "vbt/core/tensor_iter.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"

#include "vbt/cuda/storage.h"
#include "vbt/cuda/device.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

#ifndef VBT_TI_STATS
#error "VBT_TI_STATS must be defined for tensor_iter_stats_test"
#endif

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::TensorIter;
using vbt::core::TensorIterConfig;
using vbt::core::OptionalTensorImplRef;
using vbt::core::TensorIterStats;
using vbt::core::get_tensor_iter_stats;
using vbt::core::reset_tensor_iter_stats;
using vbt::core::DeviceStrideMeta;
using vbt::core::testing::TensorIterTestHelper;

extern "C" vbt::core::TensorImpl vbt_cuda_add_impl(const vbt::core::TensorImpl&,
                                                   const vbt::core::TensorImpl&);

namespace {

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  void* base = nullptr;
  if (nbytes > 0) {
    base = ::operator new(nbytes);
  }
  return vbt::core::make_intrusive<Storage>(
      DataPtr(base, [](void* p) noexcept { ::operator delete(p); }), nbytes);
}

static TensorImpl make_cpu_contiguous_tensor(const std::vector<std::int64_t>& sizes,
                                             ScalarType dtype = ScalarType::Float32) {
  const std::size_t nd = sizes.size();
  std::vector<std::int64_t> strides(nd, 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(nd) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  std::int64_t ne = 1;
  bool any_zero = false;
  for (auto s : sizes) {
    if (s == 0) {
      any_zero = true;
      break;
    }
    ne *= s;
  }
  if (any_zero) {
    ne = 0;
  }

  const std::size_t item_b = static_cast<std::size_t>(vbt::core::itemsize(dtype));
  const std::size_t nbytes = static_cast<std::size_t>(ne) * item_b;
  auto storage = make_storage_bytes(nbytes);
  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0, dtype,
                    Device::cpu());
}

#if VBT_WITH_CUDA
static TensorImpl make_cuda_contiguous_tensor(const std::vector<std::int64_t>& sizes,
                                              ScalarType dtype = ScalarType::Float32,
                                              int dev = 0) {
  std::int64_t ne = 1;
  bool any_zero = false;
  for (auto s : sizes) {
    if (s == 0) {
      any_zero = true;
      break;
    }
    ne *= s;
  }
  if (any_zero) {
    ne = 0;
  }

  const std::size_t item_b = static_cast<std::size_t>(vbt::core::itemsize(dtype));
  const std::size_t nbytes = static_cast<std::size_t>(ne) * item_b;
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);

  std::vector<std::int64_t> strides(sizes.size(), 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0, dtype,
                    Device::cuda(dev));
}
#endif

}  // namespace

TEST(TensorIterStatsTest, ResetClearsAllCounters) {
  reset_tensor_iter_stats();
  const TensorIterStats& stats = get_tensor_iter_stats();
  EXPECT_EQ(stats.cpu_invocations.load(), 0u);
  EXPECT_EQ(stats.cpu_reduction_invocations.load(), 0u);
  EXPECT_EQ(stats.cuda_meta_exports.load(), 0u);
  EXPECT_EQ(stats.cuda_ti_kernel_launches.load(), 0u);
  EXPECT_EQ(stats.num_32bit_splits.load(), 0u);
}

TEST(TensorIterStatsTest, CpuAndReductionInvocationsCountCalls) {
  reset_tensor_iter_stats();

  // Elementwise CPU iterator with non-zero numel.
  TensorImpl out = make_cpu_contiguous_tensor({4}, ScalarType::Float32);
  TensorImpl in  = make_cpu_contiguous_tensor({4}, ScalarType::Float32);

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(in);
  cfg.check_mem_overlap(false);
  TensorIter iter = cfg.build();

  int elt_tiles = 0;
  auto loop = [](char** /*data*/,
                 const std::int64_t* /*strides*/,
                 std::int64_t /*size*/,
                 void* ctx) {
    auto* count = static_cast<int*>(ctx);
    ++(*count);
  };
  vbt::core::for_each_cpu(iter, loop, &elt_tiles);
  EXPECT_GE(elt_tiles, 1);

  // Reduction iterator.
  TensorImpl in_red = make_cpu_contiguous_tensor({2, 3}, ScalarType::Float32);
  TensorImpl out_red = make_cpu_contiguous_tensor({2}, ScalarType::Float32);
  std::int64_t dim = 1;
  TensorIter red_iter = TensorIter::reduce_op(
      out_red, in_red, std::span<const std::int64_t>(&dim, 1));

  int red_tiles = 0;
  auto red_loop = [](char** /*data*/,
                     const std::int64_t* /*strides*/,
                     std::int64_t /*size*/,
                     void* ctx) {
    auto* count = static_cast<int*>(ctx);
    ++(*count);
  };
  vbt::core::for_each_reduction_cpu(red_iter, red_loop, &red_tiles);
  EXPECT_GE(red_tiles, 1);

  const TensorIterStats& stats = get_tensor_iter_stats();
  EXPECT_EQ(stats.cpu_invocations.load(), 1u);
  EXPECT_EQ(stats.cpu_reduction_invocations.load(), 1u);
}

TEST(TensorIterStatsTest, CpuInvocationsCountZeroNumelIterators) {
  reset_tensor_iter_stats();

  TensorImpl out = make_cpu_contiguous_tensor({0}, ScalarType::Float32);
  TensorImpl in  = make_cpu_contiguous_tensor({0}, ScalarType::Float32);

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(in);
  cfg.check_mem_overlap(false);
  TensorIter iter = cfg.build();

  int calls = 0;
  auto loop = [](char** /*data*/,
                 const std::int64_t* /*strides*/,
                 std::int64_t /*size*/,
                 void* ctx) {
    auto* count = static_cast<int*>(ctx);
    ++(*count);
  };
  vbt::core::for_each_cpu(iter, loop, &calls);
  EXPECT_EQ(calls, 0);

  const TensorIterStats& stats = get_tensor_iter_stats();
  EXPECT_EQ(stats.cpu_invocations.load(), 1u);
}

TEST(TensorIterStatsTest, Num32BitSplitsTracksSplitting) {
  reset_tensor_iter_stats();

  const std::int64_t big =
      static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 10;
  std::int64_t sz[1] = {big};
  TensorIter iter = TensorIterTestHelper::make_iterator_for_shape_with_dummy_operand(
      std::span<const std::int64_t>(sz, 1));

  int tiles = 0;
  iter.with_32bit_indexing([&](const TensorIter& sub) {
    ++tiles;
    EXPECT_TRUE(sub.can_use_32bit_indexing());
  });
  EXPECT_GE(tiles, 1);

  const TensorIterStats& stats = get_tensor_iter_stats();
  EXPECT_GT(stats.num_32bit_splits.load(), 0u);
}

TEST(TensorIterStatsTest, CudaMetaExportsIncrementOnCudaIterators) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  reset_tensor_iter_stats();

  const int dev = 0;
  const std::int64_t N = 16;

  TensorImpl out = make_cuda_contiguous_tensor({N}, ScalarType::Float32, dev);
  TensorImpl a   = make_cuda_contiguous_tensor({N}, ScalarType::Float32, dev);
  TensorImpl b   = make_cuda_contiguous_tensor({N}, ScalarType::Float32, dev);

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.add_input(b);
  TensorIter iter = cfg.build();

  DeviceStrideMeta mo{};
  DeviceStrideMeta ma{};
  DeviceStrideMeta mb{};
  iter.export_device_meta(0, &mo);
  iter.export_device_meta(1, &ma);
  iter.export_device_meta(2, &mb);

  const TensorIterStats& stats = get_tensor_iter_stats();
  EXPECT_EQ(stats.cuda_meta_exports.load(), 3u);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(TensorIterStatsTest, CudaTiKernelLaunchesIncrementOnTiBackedAdd) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  reset_tensor_iter_stats();

  const int dev = 0;
  const int rows = 2;
  const int cols = 16;

  TensorImpl a = make_cuda_contiguous_tensor({rows, cols}, ScalarType::Float32, dev);
  TensorImpl b = make_cuda_contiguous_tensor({cols}, ScalarType::Float32, dev);

  TensorImpl out = vbt_cuda_add_impl(a, b);
  (void)out;
  cudaDeviceSynchronize();

  const TensorIterStats& stats = get_tensor_iter_stats();
  EXPECT_EQ(stats.cuda_ti_kernel_launches.load(), 1u);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
