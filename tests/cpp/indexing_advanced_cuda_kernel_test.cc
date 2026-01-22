// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>
#include <string>
#include <limits>
#include <optional>

#include "vbt/core/indexing.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"
#include "vbt/core/checked_math.h"
#include "vbt/core/indexing_advanced_stats.h"
#include "vbt/core/indexing/index_errors.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::indexing::IndexSpec;
using vbt::core::indexing::TensorIndex;
using vbt::core::indexing::index;
using vbt::core::indexing::index_put_;
using vbt::core::indexing::advanced_index_32bit_enabled;
using vbt::core::indexing::set_advanced_index_32bit_enabled_for_tests;
using vbt::core::indexing::cuda_impl::make_advanced_index_cuda;
using vbt::core::indexing::cuda_impl::AdvancedIndexCudaMode;
using vbt::core::indexing::get_m_index_advanced_stats;
using vbt::core::indexing::reset_m_index_advanced_stats_for_tests;
using vbt::core::indexing::Slice;
namespace idx_errors = vbt::core::indexing::errors;

#if VBT_WITH_CUDA && VBT_INTERNAL_TESTS
using vbt::core::indexing::detail::AdvancedIndexEnvConfig;
using vbt::core::indexing::detail::AdvancedIndexEnvConfigGuard;
using vbt::core::indexing::detail::CudaBoundsMode;
using vbt::core::indexing::detail::get_advanced_index_env_config_for_tests;
using vbt::core::indexing::cuda_impl::get_effective_cuda_bounds_mode_for_tests;
using vbt::core::indexing::cuda_impl::get_1d_grid_x_for_tests;
using vbt::core::indexing::cuda_impl::set_device_max_grid_x_override_for_tests;
#endif

static StoragePtr make_cpu_storage_bytes(std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(
      DataPtr(::operator new(nbytes), [](void* p) noexcept { ::operator delete(p); }),
      nbytes);
}

struct AdvancedIndex32BitCudaGuard {
  bool prev;
  explicit AdvancedIndex32BitCudaGuard(bool enabled)
      : prev(advanced_index_32bit_enabled()) {
    set_advanced_index_32bit_enabled_for_tests(enabled);
  }
  ~AdvancedIndex32BitCudaGuard() {
    set_advanced_index_32bit_enabled_for_tests(prev);
  }
};

TEST(IndexingAdvancedCudaKernelTest, GatherSimple1DMatchesCpu) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  // CPU base tensor: [10, 11, 12, 13, 14]
  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage_cpu = make_cpu_storage_bytes(nbytes);
  TensorImpl base_cpu(storage_cpu, sizes, strides, /*storage_offset=*/0,
                      ScalarType::Float32, Device::cpu());
  auto* base_data_cpu = static_cast<float*>(base_cpu.data());
  for (int i = 0; i < 5; ++i) {
    base_data_cpu[i] = static_cast<float>(10 + i);
  }

  // CPU index tensor: positions [0, 4, 2].
  const std::vector<std::int64_t> idx_sizes{3};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto idx_storage_cpu = make_cpu_storage_bytes(idx_nbytes);
  TensorImpl idx_cpu(idx_storage_cpu, idx_sizes, idx_strides, 0,
                     ScalarType::Int64, Device::cpu());
  auto* idx_data_cpu = static_cast<std::int64_t*>(idx_cpu.data());
  idx_data_cpu[0] = 0;
  idx_data_cpu[1] = 4;
  idx_data_cpu[2] = 2;

  IndexSpec spec_cpu;
  spec_cpu.items.emplace_back(TensorIndex(idx_cpu));

  // Reference result via core CPU path.
  TensorImpl expected = vbt::core::indexing::index(base_cpu, spec_cpu);

  // CUDA base tensor with the same contents on device 0.
  const int dev = 0;
  auto storage_cuda = vbt::cuda::new_cuda_storage(nbytes, dev);
  TensorImpl base_cuda(storage_cuda, sizes, strides, /*storage_offset=*/0,
                       ScalarType::Float32, Device::cuda(dev));

  // Copy CPU base data to CUDA.
  {
    cudaError_t st = cudaMemcpy(
        base_cuda.data(), base_cpu.data(), nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for base failed";
  }

  // CUDA index tensor with the same indices on device 0.
  auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
  TensorImpl idx_cuda(idx_storage_cuda, idx_sizes, idx_strides, 0,
                      ScalarType::Int64, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        idx_cuda.data(), idx_cpu.data(), idx_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for idx failed";
  }

  IndexSpec spec_cuda;
  spec_cuda.items.emplace_back(TensorIndex(idx_cuda));

  // CUDA result via unified index() entrypoint.
  TensorImpl out_cuda = vbt::core::indexing::index(base_cuda, spec_cuda);

  ASSERT_EQ(out_cuda.device().type, kDLCUDA);
  ASSERT_EQ(out_cuda.sizes().size(), 1u);
  EXPECT_EQ(out_cuda.sizes()[0], 3);

  // Copy CUDA result back to CPU and compare values.
  std::vector<float> out_host(3, 0.0f);
  {
    const std::size_t out_nbytes = static_cast<std::size_t>(3 * sizeof(float));
    cudaError_t st = cudaMemcpy(
        out_host.data(), out_cuda.data(), out_nbytes,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for out failed";
  }

  const float* expected_data = static_cast<const float*>(expected.data());
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(expected_data[i], out_host[i]);
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(IndexingAdvancedCudaKernelTest, GatherWithStorageOffsetMatchesCpu) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  // CPU base tensor: [10, 11, ..., 19]
  const std::vector<std::int64_t> base_sizes{10};
  const std::vector<std::int64_t> base_strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(10 * sizeof(float));
  auto storage_cpu = make_cpu_storage_bytes(nbytes);
  TensorImpl base_cpu(storage_cpu, base_sizes, base_strides, /*storage_offset=*/0,
                      ScalarType::Float32, Device::cpu());
  auto* base_data_cpu = static_cast<float*>(base_cpu.data());
  for (int i = 0; i < 10; ++i) {
    base_data_cpu[i] = static_cast<float>(10 + i);
  }

  // View into the base tensor with a non-zero storage_offset: [12, 13, 14, 15, 16]
  const std::vector<std::int64_t> view_sizes{5};
  const std::vector<std::int64_t> view_strides{1};
  TensorImpl view_cpu(storage_cpu, view_sizes, view_strides, /*storage_offset=*/2,
                      ScalarType::Float32, Device::cpu());

  // CPU index tensor: positions [0, 4, 2] within the view.
  const std::vector<std::int64_t> idx_sizes{3};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto idx_storage_cpu = make_cpu_storage_bytes(idx_nbytes);
  TensorImpl idx_cpu(idx_storage_cpu, idx_sizes, idx_strides, 0,
                     ScalarType::Int64, Device::cpu());
  auto* idx_data_cpu = static_cast<std::int64_t*>(idx_cpu.data());
  idx_data_cpu[0] = 0;
  idx_data_cpu[1] = 4;
  idx_data_cpu[2] = 2;

  IndexSpec spec_cpu;
  spec_cpu.items.emplace_back(TensorIndex(idx_cpu));

  // Reference result via core CPU path.
  TensorImpl expected = vbt::core::indexing::index(view_cpu, spec_cpu);

  // CUDA base tensor with the same contents on device 0.
  const int dev = 0;
  auto storage_cuda = vbt::cuda::new_cuda_storage(nbytes, dev);
  TensorImpl base_cuda(storage_cuda, base_sizes, base_strides, /*storage_offset=*/0,
                       ScalarType::Float32, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        base_cuda.data(), base_cpu.data(), nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for base failed";
  }

  TensorImpl view_cuda(storage_cuda, view_sizes, view_strides, /*storage_offset=*/2,
                       ScalarType::Float32, Device::cuda(dev));

  // CUDA index tensor with the same indices on device 0.
  auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
  TensorImpl idx_cuda(idx_storage_cuda, idx_sizes, idx_strides, 0,
                      ScalarType::Int64, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        idx_cuda.data(), idx_cpu.data(), idx_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for idx failed";
  }

  IndexSpec spec_cuda;
  spec_cuda.items.emplace_back(TensorIndex(idx_cuda));

  // CUDA result via unified index() entrypoint.
  TensorImpl out_cuda = vbt::core::indexing::index(view_cuda, spec_cuda);

  std::vector<float> out_host(3, 0.0f);
  {
    const std::size_t out_nbytes = static_cast<std::size_t>(3 * sizeof(float));
    cudaError_t st = cudaMemcpy(
        out_host.data(), out_cuda.data(), out_nbytes,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for out failed";
  }

  const float* expected_data = static_cast<const float*>(expected.data());
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(expected_data[i], out_host[i]);
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(IndexingAdvancedCudaKernelTest, ScatterOverwrite1DMatchesCpu) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage_cpu = make_cpu_storage_bytes(nbytes);
  TensorImpl base_cpu(storage_cpu, sizes, strides, /*storage_offset=*/0,
                      ScalarType::Float32, Device::cpu());
  auto* base_data_cpu = static_cast<float*>(base_cpu.data());
  for (int i = 0; i < 5; ++i) {
    base_data_cpu[i] = 0.0f;
  }

  const std::vector<std::int64_t> idx_sizes{3};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto idx_storage_cpu = make_cpu_storage_bytes(idx_nbytes);
  TensorImpl idx_cpu(idx_storage_cpu, idx_sizes, idx_strides, 0,
                     ScalarType::Int64, Device::cpu());
  auto* idx_data_cpu = static_cast<std::int64_t*>(idx_cpu.data());
  idx_data_cpu[0] = 0;
  idx_data_cpu[1] = 4;
  idx_data_cpu[2] = 2;

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx_cpu));

  const std::vector<std::int64_t> val_sizes{3};
  const std::vector<std::int64_t> val_strides{1};
  const std::size_t val_nbytes = static_cast<std::size_t>(3 * sizeof(float));
  auto val_storage_cpu = make_cpu_storage_bytes(val_nbytes);
  TensorImpl values_cpu(val_storage_cpu, val_sizes, val_strides, 0,
                        ScalarType::Float32, Device::cpu());
  auto* val_data_cpu = static_cast<float*>(values_cpu.data());
  val_data_cpu[0] = 5.0f;
  val_data_cpu[1] = 7.0f;
  val_data_cpu[2] = 9.0f;

  // Set up CUDA base and copy the initial CPU contents before CPU writes.
  const int dev = 0;
  auto storage_cuda = vbt::cuda::new_cuda_storage(nbytes, dev);
  TensorImpl base_cuda(storage_cuda, sizes, strides, /*storage_offset=*/0,
                       ScalarType::Float32, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        base_cuda.data(), base_cpu.data(), nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for base failed";
  }

  auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
  TensorImpl idx_cuda(idx_storage_cuda, idx_sizes, idx_strides, 0,
                      ScalarType::Int64, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        idx_cuda.data(), idx_cpu.data(), idx_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for idx failed";
  }

  auto val_storage_cuda = vbt::cuda::new_cuda_storage(val_nbytes, dev);
  TensorImpl values_cuda(val_storage_cuda, val_sizes, val_strides, 0,
                         ScalarType::Float32, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        values_cuda.data(), values_cpu.data(), val_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for values failed";
  }

  IndexSpec spec_cuda;
  spec_cuda.items.emplace_back(TensorIndex(idx_cuda));

  // CPU baseline via index_put_ (overwrite semantics).
  index_put_(base_cpu, spec, values_cpu, /*accumulate=*/false);

  // CUDA advanced write via index_put_ -> advanced_index_put_cuda.
  index_put_(base_cuda, spec_cuda, values_cuda, /*accumulate=*/false);

  std::vector<float> out_host(5, 0.0f);
  {
    cudaError_t st = cudaMemcpy(
        out_host.data(), base_cuda.data(), nbytes,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for base failed";
  }

  for (std::size_t i = 0; i < out_host.size(); ++i) {
    EXPECT_FLOAT_EQ(base_data_cpu[i], out_host[i]);
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(IndexingAdvancedCudaKernelTest, ScatterAccumulate1DMatchesCpu) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage_cpu = make_cpu_storage_bytes(nbytes);
  TensorImpl base_cpu(storage_cpu, sizes, strides, /*storage_offset=*/0,
                      ScalarType::Float32, Device::cpu());
  auto* base_data_cpu = static_cast<float*>(base_cpu.data());
  for (int i = 0; i < 5; ++i) {
    base_data_cpu[i] = 0.0f;
  }

  const std::vector<std::int64_t> idx_sizes{4};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(4 * sizeof(std::int64_t));
  auto idx_storage_cpu = make_cpu_storage_bytes(idx_nbytes);
  TensorImpl idx_cpu(idx_storage_cpu, idx_sizes, idx_strides, 0,
                     ScalarType::Int64, Device::cpu());
  auto* idx_data_cpu = static_cast<std::int64_t*>(idx_cpu.data());
  // Indices [0, 1, 0, 4] introduce a duplicate at position 0.
  idx_data_cpu[0] = 0;
  idx_data_cpu[1] = 1;
  idx_data_cpu[2] = 0;
  idx_data_cpu[3] = 4;

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx_cpu));

  const std::vector<std::int64_t> val_sizes{4};
  const std::vector<std::int64_t> val_strides{1};
  const std::size_t val_nbytes = static_cast<std::size_t>(4 * sizeof(float));
  auto val_storage_cpu = make_cpu_storage_bytes(val_nbytes);
  TensorImpl values_cpu(val_storage_cpu, val_sizes, val_strides, 0,
                        ScalarType::Float32, Device::cpu());
  auto* val_data_cpu = static_cast<float*>(values_cpu.data());
  val_data_cpu[0] = 1.0f;
  val_data_cpu[1] = 2.0f;
  val_data_cpu[2] = 3.0f;
  val_data_cpu[3] = 4.0f;

  const int dev = 0;
  auto storage_cuda = vbt::cuda::new_cuda_storage(nbytes, dev);
  TensorImpl base_cuda(storage_cuda, sizes, strides, /*storage_offset=*/0,
                       ScalarType::Float32, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        base_cuda.data(), base_cpu.data(), nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for base failed";
  }

  auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
  TensorImpl idx_cuda(idx_storage_cuda, idx_sizes, idx_strides, 0,
                      ScalarType::Int64, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        idx_cuda.data(), idx_cpu.data(), idx_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for idx failed";
  }

  auto val_storage_cuda = vbt::cuda::new_cuda_storage(val_nbytes, dev);
  TensorImpl values_cuda(val_storage_cuda, val_sizes, val_strides, 0,
                         ScalarType::Float32, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        values_cuda.data(), values_cpu.data(), val_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for values failed";
  }

  IndexSpec spec_cuda;
  spec_cuda.items.emplace_back(TensorIndex(idx_cuda));

  // CPU baseline with accumulate=true semantics.
  index_put_(base_cpu, spec, values_cpu, /*accumulate=*/true);

  // CUDA advanced write with accumulate=true (float32 only).
  index_put_(base_cuda, spec_cuda, values_cuda, /*accumulate=*/true);

  std::vector<float> out_host(5, 0.0f);
  {
    cudaError_t st = cudaMemcpy(
        out_host.data(), base_cuda.data(), nbytes,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for base failed";
  }

  for (std::size_t i = 0; i < out_host.size(); ++i) {
    EXPECT_NEAR(base_data_cpu[i], out_host[i], 1e-5f);
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(IndexingAdvancedCudaKernelTest, ScatterAccumulate1DMatchesCpu_CubPrototype) {
#if VBT_WITH_CUDA && VBT_INTERNAL_TESTS
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  AdvancedIndexEnvConfig cfg = get_advanced_index_env_config_for_tests();
  cfg.cuda_cub_index_put_accumulate = true;
  AdvancedIndexEnvConfigGuard cfg_guard(cfg);

  // --- Float32 accumulate ---
  {
    const std::vector<std::int64_t> sizes{5};
    const std::vector<std::int64_t> strides{1};
    const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
    auto storage_cpu = make_cpu_storage_bytes(nbytes);
    TensorImpl base_cpu(storage_cpu, sizes, strides, /*storage_offset=*/0,
                        ScalarType::Float32, Device::cpu());
    auto* base_data_cpu = static_cast<float*>(base_cpu.data());
    for (int i = 0; i < 5; ++i) {
      base_data_cpu[i] = 0.0f;
    }

    const std::vector<std::int64_t> idx_sizes{4};
    const std::vector<std::int64_t> idx_strides{1};
    const std::size_t idx_nbytes =
        static_cast<std::size_t>(4 * sizeof(std::int64_t));
    auto idx_storage_cpu = make_cpu_storage_bytes(idx_nbytes);
    TensorImpl idx_cpu(idx_storage_cpu, idx_sizes, idx_strides, 0,
                       ScalarType::Int64, Device::cpu());
    auto* idx_data_cpu = static_cast<std::int64_t*>(idx_cpu.data());
    idx_data_cpu[0] = 0;
    idx_data_cpu[1] = 1;
    idx_data_cpu[2] = 0;
    idx_data_cpu[3] = 4;

    IndexSpec spec;
    spec.items.emplace_back(TensorIndex(idx_cpu));

    const std::vector<std::int64_t> val_sizes{4};
    const std::vector<std::int64_t> val_strides{1};
    const std::size_t val_nbytes = static_cast<std::size_t>(4 * sizeof(float));
    auto val_storage_cpu = make_cpu_storage_bytes(val_nbytes);
    TensorImpl values_cpu(val_storage_cpu, val_sizes, val_strides, 0,
                          ScalarType::Float32, Device::cpu());
    auto* val_data_cpu = static_cast<float*>(values_cpu.data());
    val_data_cpu[0] = 1.0f;
    val_data_cpu[1] = 2.0f;
    val_data_cpu[2] = 3.0f;
    val_data_cpu[3] = 4.0f;

    const int dev = 0;
    auto storage_cuda = vbt::cuda::new_cuda_storage(nbytes, dev);
    TensorImpl base_cuda(storage_cuda, sizes, strides, /*storage_offset=*/0,
                         ScalarType::Float32, Device::cuda(dev));
    {
      cudaError_t st = cudaMemcpy(
          base_cuda.data(), base_cpu.data(), nbytes,
          cudaMemcpyHostToDevice);
      ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for base failed";
    }

    auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
    TensorImpl idx_cuda(idx_storage_cuda, idx_sizes, idx_strides, 0,
                        ScalarType::Int64, Device::cuda(dev));
    {
      cudaError_t st = cudaMemcpy(
          idx_cuda.data(), idx_cpu.data(), idx_nbytes,
          cudaMemcpyHostToDevice);
      ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for idx failed";
    }

    auto val_storage_cuda = vbt::cuda::new_cuda_storage(val_nbytes, dev);
    TensorImpl values_cuda(val_storage_cuda, val_sizes, val_strides, 0,
                           ScalarType::Float32, Device::cuda(dev));
    {
      cudaError_t st = cudaMemcpy(
          values_cuda.data(), values_cpu.data(), val_nbytes,
          cudaMemcpyHostToDevice);
      ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for values failed";
    }

    IndexSpec spec_cuda;
    spec_cuda.items.emplace_back(TensorIndex(idx_cuda));

    // CPU baseline.
    index_put_(base_cpu, spec, values_cpu, /*accumulate=*/true);

    // CUDA accumulate under the CUB flag.
    index_put_(base_cuda, spec_cuda, values_cuda, /*accumulate=*/true);

    std::vector<float> out_host(5, 0.0f);
    {
      cudaError_t st = cudaMemcpy(
          out_host.data(), base_cuda.data(), nbytes,
          cudaMemcpyDeviceToHost);
      ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for base failed";
    }

    for (std::size_t i = 0; i < out_host.size(); ++i) {
      EXPECT_NEAR(base_data_cpu[i], out_host[i], 1e-5f);
    }
  }

  // --- Int64 accumulate ---
  {
    const std::vector<std::int64_t> sizes{5};
    const std::vector<std::int64_t> strides{1};
    const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(std::int64_t));
    auto storage_cpu = make_cpu_storage_bytes(nbytes);
    TensorImpl base_cpu(storage_cpu, sizes, strides, /*storage_offset=*/0,
                        ScalarType::Int64, Device::cpu());
    auto* base_data_cpu = static_cast<std::int64_t*>(base_cpu.data());
    for (int i = 0; i < 5; ++i) {
      base_data_cpu[i] = 0;
    }

    const std::vector<std::int64_t> idx_sizes{4};
    const std::vector<std::int64_t> idx_strides{1};
    const std::size_t idx_nbytes =
        static_cast<std::size_t>(4 * sizeof(std::int64_t));
    auto idx_storage_cpu = make_cpu_storage_bytes(idx_nbytes);
    TensorImpl idx_cpu(idx_storage_cpu, idx_sizes, idx_strides, 0,
                       ScalarType::Int64, Device::cpu());
    auto* idx_data_cpu = static_cast<std::int64_t*>(idx_cpu.data());
    idx_data_cpu[0] = 0;
    idx_data_cpu[1] = 1;
    idx_data_cpu[2] = 0;
    idx_data_cpu[3] = 4;

    IndexSpec spec;
    spec.items.emplace_back(TensorIndex(idx_cpu));

    const std::vector<std::int64_t> val_sizes{4};
    const std::vector<std::int64_t> val_strides{1};
    const std::size_t val_nbytes =
        static_cast<std::size_t>(4 * sizeof(std::int64_t));
    auto val_storage_cpu = make_cpu_storage_bytes(val_nbytes);
    TensorImpl values_cpu(val_storage_cpu, val_sizes, val_strides, 0,
                          ScalarType::Int64, Device::cpu());
    auto* val_data_cpu = static_cast<std::int64_t*>(values_cpu.data());
    val_data_cpu[0] = 1;
    val_data_cpu[1] = 2;
    val_data_cpu[2] = 3;
    val_data_cpu[3] = 4;

    const int dev = 0;
    auto storage_cuda = vbt::cuda::new_cuda_storage(nbytes, dev);
    TensorImpl base_cuda(storage_cuda, sizes, strides, /*storage_offset=*/0,
                         ScalarType::Int64, Device::cuda(dev));
    {
      cudaError_t st = cudaMemcpy(
          base_cuda.data(), base_cpu.data(), nbytes,
          cudaMemcpyHostToDevice);
      ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for base failed";
    }

    auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
    TensorImpl idx_cuda(idx_storage_cuda, idx_sizes, idx_strides, 0,
                        ScalarType::Int64, Device::cuda(dev));
    {
      cudaError_t st = cudaMemcpy(
          idx_cuda.data(), idx_cpu.data(), idx_nbytes,
          cudaMemcpyHostToDevice);
      ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for idx failed";
    }

    auto val_storage_cuda = vbt::cuda::new_cuda_storage(val_nbytes, dev);
    TensorImpl values_cuda(val_storage_cuda, val_sizes, val_strides, 0,
                           ScalarType::Int64, Device::cuda(dev));
    {
      cudaError_t st = cudaMemcpy(
          values_cuda.data(), values_cpu.data(), val_nbytes,
          cudaMemcpyHostToDevice);
      ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for values failed";
    }

    IndexSpec spec_cuda;
    spec_cuda.items.emplace_back(TensorIndex(idx_cuda));

    // CPU baseline.
    index_put_(base_cpu, spec, values_cpu, /*accumulate=*/true);

    // CUDA accumulate (only supported via CUB path).
    index_put_(base_cuda, spec_cuda, values_cuda, /*accumulate=*/true);

    std::vector<std::int64_t> out_host(5, 0);
    {
      cudaError_t st = cudaMemcpy(
          out_host.data(), base_cuda.data(), nbytes,
          cudaMemcpyDeviceToHost);
      ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for base failed";
    }

    for (std::size_t i = 0; i < out_host.size(); ++i) {
      EXPECT_EQ(base_data_cpu[i], out_host[i]);
    }
  }

  // --- Int64 accumulate with non-zero storage_offset ---
  {
    const std::vector<std::int64_t> base_sizes{10};
    const std::vector<std::int64_t> base_strides{1};
    const std::size_t nbytes =
        static_cast<std::size_t>(10 * sizeof(std::int64_t));

    auto storage_cpu = make_cpu_storage_bytes(nbytes);
    TensorImpl base_cpu(storage_cpu, base_sizes, base_strides, /*storage_offset=*/0,
                        ScalarType::Int64, Device::cpu());
    auto* base_data_cpu = static_cast<std::int64_t*>(base_cpu.data());
    for (int i = 0; i < 10; ++i) {
      base_data_cpu[i] = 0;
    }

    const std::vector<std::int64_t> view_sizes{5};
    const std::vector<std::int64_t> view_strides{1};
    TensorImpl view_cpu(storage_cpu, view_sizes, view_strides, /*storage_offset=*/2,
                        ScalarType::Int64, Device::cpu());

    const std::vector<std::int64_t> idx_sizes{4};
    const std::vector<std::int64_t> idx_strides{1};
    const std::size_t idx_nbytes =
        static_cast<std::size_t>(4 * sizeof(std::int64_t));
    auto idx_storage_cpu = make_cpu_storage_bytes(idx_nbytes);
    TensorImpl idx_cpu(idx_storage_cpu, idx_sizes, idx_strides, 0,
                       ScalarType::Int64, Device::cpu());
    auto* idx_data_cpu = static_cast<std::int64_t*>(idx_cpu.data());
    idx_data_cpu[0] = 0;
    idx_data_cpu[1] = 1;
    idx_data_cpu[2] = 0;
    idx_data_cpu[3] = 4;

    IndexSpec spec;
    spec.items.emplace_back(TensorIndex(idx_cpu));

    const std::vector<std::int64_t> val_sizes{4};
    const std::vector<std::int64_t> val_strides{1};
    const std::size_t val_nbytes =
        static_cast<std::size_t>(4 * sizeof(std::int64_t));
    auto val_storage_cpu = make_cpu_storage_bytes(val_nbytes);
    TensorImpl values_cpu(val_storage_cpu, val_sizes, val_strides, 0,
                          ScalarType::Int64, Device::cpu());
    auto* val_data_cpu = static_cast<std::int64_t*>(values_cpu.data());
    val_data_cpu[0] = 1;
    val_data_cpu[1] = 2;
    val_data_cpu[2] = 3;
    val_data_cpu[3] = 4;

    const int dev = 0;
    auto storage_cuda = vbt::cuda::new_cuda_storage(nbytes, dev);
    TensorImpl base_cuda(storage_cuda, base_sizes, base_strides, /*storage_offset=*/0,
                         ScalarType::Int64, Device::cuda(dev));
    {
      cudaError_t st = cudaMemcpy(
          base_cuda.data(), base_cpu.data(), nbytes,
          cudaMemcpyHostToDevice);
      ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for base failed";
    }

    // CPU baseline (updates view, writes into base storage).
    index_put_(view_cpu, spec, values_cpu, /*accumulate=*/true);

    TensorImpl view_cuda(storage_cuda, view_sizes, view_strides, /*storage_offset=*/2,
                         ScalarType::Int64, Device::cuda(dev));

    auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
    TensorImpl idx_cuda(idx_storage_cuda, idx_sizes, idx_strides, 0,
                        ScalarType::Int64, Device::cuda(dev));
    {
      cudaError_t st = cudaMemcpy(
          idx_cuda.data(), idx_cpu.data(), idx_nbytes,
          cudaMemcpyHostToDevice);
      ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for idx failed";
    }

    auto val_storage_cuda = vbt::cuda::new_cuda_storage(val_nbytes, dev);
    TensorImpl values_cuda(val_storage_cuda, val_sizes, val_strides, 0,
                           ScalarType::Int64, Device::cuda(dev));
    {
      cudaError_t st = cudaMemcpy(
          values_cuda.data(), values_cpu.data(), val_nbytes,
          cudaMemcpyHostToDevice);
      ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for values failed";
    }

    IndexSpec spec_cuda;
    spec_cuda.items.emplace_back(TensorIndex(idx_cuda));

    // CUDA accumulate on the view under the CUB flag.
    index_put_(view_cuda, spec_cuda, values_cuda, /*accumulate=*/true);

    std::vector<std::int64_t> out_host(10, 0);
    {
      cudaError_t st = cudaMemcpy(
          out_host.data(), base_cuda.data(), nbytes,
          cudaMemcpyDeviceToHost);
      ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for base failed";
    }

    for (std::size_t i = 0; i < out_host.size(); ++i) {
      EXPECT_EQ(base_data_cpu[i], out_host[i]);
    }
  }
#else
  GTEST_SKIP() << "Built without CUDA or internal tests";
#endif
}

TEST(IndexingAdvancedCudaKernelTest, ResultTooLargeTriggersDosCap) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  // Base CUDA tensor of shape (D0, D1). The prefix dimension D0 is chosen
  // so that D0 * N exceeds the CUDA advanced-index result-numel cap while
  // keeping the actual allocation size modest.
  const std::int64_t D0 = 200'000;  // prefix dimension size
  const std::int64_t D1 = 2;        // advanced dimension size
  const std::vector<std::int64_t> base_sizes{D0, D1};
  const std::vector<std::int64_t> base_strides{D1, 1};
  const std::size_t nbytes =
      static_cast<std::size_t>(D0 * D1 * sizeof(float));

  const int dev = 0;
  auto storage_cuda = vbt::cuda::new_cuda_storage(nbytes, dev);
  TensorImpl base(storage_cuda, base_sizes, base_strides,
                  /*storage_offset=*/0,
                  ScalarType::Float32, Device::cuda(dev));

  // Index tensor on the last dimension with small cardinality N such that
  // D0 * N > kAdvIndexMaxResultNumel while index_numel stays well below the
  // index-numel DoS cap.
  const std::int64_t N = 2000;  // 2e5 * 2e3 = 4e8 > 1e8
  const std::vector<std::int64_t> idx_sizes{N};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes =
      static_cast<std::size_t>(N) * sizeof(std::int64_t);
  auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
  TensorImpl idx(idx_storage_cuda, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cuda(dev));

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(Slice{}));  // prefix full-range slice
  spec.items.emplace_back(TensorIndex(idx));

  try {
    (void)make_advanced_index_cuda(base, spec,
                                   AdvancedIndexCudaMode::Read);
    FAIL() << "expected std::runtime_error for CUDA advanced indexing result DoS cap";
  } catch (const std::runtime_error& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find(idx_errors::kErrCudaAdvResultTooLarge),
              std::string::npos);
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
TEST(IndexingAdvancedCudaKernelTest, GatherParityWith32BitFlag) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  // CPU base tensor: [10, 11, 12, 13, 14]
  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage_cpu = make_cpu_storage_bytes(nbytes);
  TensorImpl base_cpu(storage_cpu, sizes, strides, /*storage_offset=*/0,
                      ScalarType::Float32, Device::cpu());
  auto* base_data_cpu = static_cast<float*>(base_cpu.data());
  for (int i = 0; i < 5; ++i) {
    base_data_cpu[i] = static_cast<float>(10 + i);
  }

  const std::vector<std::int64_t> idx_sizes{3};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto idx_storage_cpu = make_cpu_storage_bytes(idx_nbytes);
  TensorImpl idx_cpu(idx_storage_cpu, idx_sizes, idx_strides, 0,
                     ScalarType::Int64, Device::cpu());
  auto* idx_data_cpu = static_cast<std::int64_t*>(idx_cpu.data());
  idx_data_cpu[0] = 0;
  idx_data_cpu[1] = 4;
  idx_data_cpu[2] = 2;

  IndexSpec spec_cpu;
  spec_cpu.items.emplace_back(TensorIndex(idx_cpu));

  // CUDA base tensor with the same contents on device 0.
  const int dev = 0;
  auto storage_cuda = vbt::cuda::new_cuda_storage(nbytes, dev);
  TensorImpl base_cuda(storage_cuda, sizes, strides, /*storage_offset=*/0,
                       ScalarType::Float32, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        base_cuda.data(), base_cpu.data(), nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for base failed";
  }

  auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
  TensorImpl idx_cuda(idx_storage_cuda, idx_sizes, idx_strides, 0,
                      ScalarType::Int64, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        idx_cuda.data(), idx_cpu.data(), idx_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for idx failed";
  }

  IndexSpec spec_cuda;
  spec_cuda.items.emplace_back(TensorIndex(idx_cuda));

  std::vector<float> baseline(3, 0.0f);
  {
    AdvancedIndex32BitCudaGuard guard(false);
    TensorImpl out_cuda = index(base_cuda, spec_cuda);

    ASSERT_EQ(out_cuda.device().type, kDLCUDA);
    ASSERT_EQ(out_cuda.sizes().size(), 1u);
    EXPECT_EQ(out_cuda.sizes()[0], 3);

    const std::size_t out_nbytes = static_cast<std::size_t>(3 * sizeof(float));
    cudaError_t st = cudaMemcpy(
        baseline.data(), out_cuda.data(), out_nbytes,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for out (flag off) failed";
  }

  std::vector<float> out_host(3, 0.0f);
  {
    AdvancedIndex32BitCudaGuard guard(true);
    TensorImpl out_cuda = index(base_cuda, spec_cuda);

    ASSERT_EQ(out_cuda.device().type, kDLCUDA);
    ASSERT_EQ(out_cuda.sizes().size(), 1u);
    EXPECT_EQ(out_cuda.sizes()[0], 3);

    const std::size_t out_nbytes = static_cast<std::size_t>(3 * sizeof(float));
    cudaError_t st = cudaMemcpy(
        out_host.data(), out_cuda.data(), out_nbytes,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for out (flag on) failed";
  }

  for (std::size_t i = 0; i < out_host.size(); ++i) {
    EXPECT_FLOAT_EQ(baseline[i], out_host[i]);
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(IndexingAdvancedCudaKernelTest, ScatterOverwriteParityWith32BitFlag) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));

  // Two identical CUDA bases so we can compare flag off vs on.
  const int dev = 0;
  auto storage_cuda1 = vbt::cuda::new_cuda_storage(nbytes, dev);
  auto storage_cuda2 = vbt::cuda::new_cuda_storage(nbytes, dev);

  TensorImpl base1_cuda(storage_cuda1, sizes, strides, /*storage_offset=*/0,
                        ScalarType::Float32, Device::cuda(dev));
  TensorImpl base2_cuda(storage_cuda2, sizes, strides, /*storage_offset=*/0,
                        ScalarType::Float32, Device::cuda(dev));

  // Initialize both bases to zeros.
  {
    std::vector<float> zeros(5, 0.0f);
    cudaError_t st = cudaMemcpy(
        base1_cuda.data(), zeros.data(), nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for base1 failed";
    st = cudaMemcpy(
        base2_cuda.data(), zeros.data(), nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for base2 failed";
  }

  const std::vector<std::int64_t> idx_sizes{3};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto idx_storage = make_cpu_storage_bytes(idx_nbytes);
  TensorImpl idx_cpu(idx_storage, idx_sizes, idx_strides, 0,
                     ScalarType::Int64, Device::cpu());
  auto* idx_data_cpu = static_cast<std::int64_t*>(idx_cpu.data());
  idx_data_cpu[0] = 0;
  idx_data_cpu[1] = 4;
  idx_data_cpu[2] = 2;

  IndexSpec spec_cpu;
  spec_cpu.items.emplace_back(TensorIndex(idx_cpu));

  auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
  TensorImpl idx_cuda(idx_storage_cuda, idx_sizes, idx_strides, 0,
                      ScalarType::Int64, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        idx_cuda.data(), idx_cpu.data(), idx_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for idx failed";
  }

  const std::vector<std::int64_t> val_sizes{3};
  const std::vector<std::int64_t> val_strides{1};
  const std::size_t val_nbytes = static_cast<std::size_t>(3 * sizeof(float));
  auto val_storage = make_cpu_storage_bytes(val_nbytes);
  TensorImpl values_cpu(val_storage, val_sizes, val_strides, 0,
                        ScalarType::Float32, Device::cpu());
  auto* val_data_cpu = static_cast<float*>(values_cpu.data());
  val_data_cpu[0] = 5.0f;
  val_data_cpu[1] = 7.0f;
  val_data_cpu[2] = 9.0f;

  auto val_storage_cuda = vbt::cuda::new_cuda_storage(val_nbytes, dev);
  TensorImpl values_cuda(val_storage_cuda, val_sizes, val_strides, 0,
                         ScalarType::Float32, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        values_cuda.data(), val_data_cpu, val_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for values failed";
  }

  IndexSpec spec_cuda;
  spec_cuda.items.emplace_back(TensorIndex(idx_cuda));

  {
    AdvancedIndex32BitCudaGuard guard(false);
    index_put_(base1_cuda, spec_cuda, values_cuda, /*accumulate=*/false);
  }

  {
    AdvancedIndex32BitCudaGuard guard(true);
    index_put_(base2_cuda, spec_cuda, values_cuda, /*accumulate=*/false);
  }

  std::vector<float> out1(5, 0.0f);
  std::vector<float> out2(5, 0.0f);
  {
    cudaError_t st = cudaMemcpy(
        out1.data(), base1_cuda.data(), nbytes,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for base1 failed";
    st = cudaMemcpy(
        out2.data(), base2_cuda.data(), nbytes,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for base2 failed";
  }

  for (std::size_t i = 0; i < out1.size(); ++i) {
    EXPECT_FLOAT_EQ(out1[i], out2[i]);
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(IndexingAdvancedCudaKernelTest, Fast1DPathIncrementsStatsAndRespectsFlag) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  // CPU base tensor: [0, 1, 2, 3, 4].
  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage_cpu = make_cpu_storage_bytes(nbytes);
  TensorImpl base_cpu(storage_cpu, sizes, strides, /*storage_offset=*/0,
                      ScalarType::Float32, Device::cpu());
  auto* base_data_cpu = static_cast<float*>(base_cpu.data());
  for (int i = 0; i < 5; ++i) {
    base_data_cpu[i] = static_cast<float>(i);
  }

  // CPU index tensor: positions [0, 2, 4].
  const std::vector<std::int64_t> idx_sizes{3};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto idx_storage_cpu = make_cpu_storage_bytes(idx_nbytes);
  TensorImpl idx_cpu(idx_storage_cpu, idx_sizes, idx_strides, 0,
                     ScalarType::Int64, Device::cpu());
  auto* idx_data_cpu = static_cast<std::int64_t*>(idx_cpu.data());
  idx_data_cpu[0] = 0;
  idx_data_cpu[1] = 2;
  idx_data_cpu[2] = 4;

  IndexSpec spec_cpu;
  spec_cpu.items.emplace_back(TensorIndex(idx_cpu));

  // Reference result via core CPU path.
  TensorImpl expected = vbt::core::indexing::index(base_cpu, spec_cpu);

  // CUDA base tensor with the same contents on device 0.
  const int dev = 0;
  auto storage_cuda = vbt::cuda::new_cuda_storage(nbytes, dev);
  TensorImpl base_cuda(storage_cuda, sizes, strides, /*storage_offset=*/0,
                       ScalarType::Float32, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        base_cuda.data(), base_cpu.data(), nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for base failed";
  }

  auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
  TensorImpl idx_cuda(idx_storage_cuda, idx_sizes, idx_strides, 0,
                      ScalarType::Int64, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        idx_cuda.data(), idx_cpu.data(), idx_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for idx failed";
  }

  IndexSpec spec_cuda;
  spec_cuda.items.emplace_back(TensorIndex(idx_cuda));

  // With flag enabled, 1D fast path should be allowed and stats should record a hit.
  reset_m_index_advanced_stats_for_tests();
  {
    AdvancedIndex32BitCudaGuard guard(true);
    TensorImpl out_cuda = vbt::core::indexing::index(base_cuda, spec_cuda);
    ASSERT_EQ(out_cuda.device().type, kDLCUDA);

    // Copy result back and compare with CPU reference.
    std::vector<float> out_host(3, 0.0f);
    cudaError_t st = cudaMemcpy(
        out_host.data(), out_cuda.data(),
        static_cast<std::size_t>(3 * sizeof(float)),
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for out failed";

    const float* expected_data = static_cast<const float*>(expected.data());
    for (std::size_t i = 0; i < out_host.size(); ++i) {
      EXPECT_FLOAT_EQ(expected_data[i], out_host[i]);
    }
  }

  const auto& stats_after_fast = get_m_index_advanced_stats();
  EXPECT_EQ(stats_after_fast.cuda_fast1d_forward_hits.load(std::memory_order_relaxed), 1u);

  // With flag disabled, fast path must not run and the hit counter must not change.
  reset_m_index_advanced_stats_for_tests();
  {
    AdvancedIndex32BitCudaGuard guard(false);
    TensorImpl out_cuda = vbt::core::indexing::index(base_cuda, spec_cuda);

    std::vector<float> out_host(3, 0.0f);
    cudaError_t st = cudaMemcpy(
        out_host.data(), out_cuda.data(),
        static_cast<std::size_t>(3 * sizeof(float)),
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for out (flag off) failed";

    const float* expected_data = static_cast<const float*>(expected.data());
    for (std::size_t i = 0; i < out_host.size(); ++i) {
      EXPECT_FLOAT_EQ(expected_data[i], out_host[i]);
    }
  }

  const auto& stats_after_disabled = get_m_index_advanced_stats();
  EXPECT_EQ(stats_after_disabled.cuda_fast1d_forward_hits.load(std::memory_order_relaxed), 0u);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

#if VBT_WITH_CUDA && VBT_INTERNAL_TESTS

TEST(IndexingAdvancedCudaKernelTest, BoundsModeInRangeParity) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  // CPU base tensor: [10, 11, 12, 13, 14].
  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage_cpu = make_cpu_storage_bytes(nbytes);
  TensorImpl base_cpu(storage_cpu, sizes, strides, /*storage_offset=*/0,
                      ScalarType::Float32, Device::cpu());
  auto* base_data_cpu = static_cast<float*>(base_cpu.data());
  for (int i = 0; i < 5; ++i) {
    base_data_cpu[i] = static_cast<float>(10 + i);
  }

  // Indices [0, -1, 2, 3] are all in range once negative indices are
  // normalized via v += D.
  const std::vector<std::int64_t> idx_sizes{4};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(4 * sizeof(std::int64_t));
  auto idx_storage_cpu = make_cpu_storage_bytes(idx_nbytes);
  TensorImpl idx_cpu(idx_storage_cpu, idx_sizes, idx_strides, 0,
                     ScalarType::Int64, Device::cpu());
  auto* idx_data_cpu = static_cast<std::int64_t*>(idx_cpu.data());
  idx_data_cpu[0] = 0;
  idx_data_cpu[1] = -1;  // maps to 4
  idx_data_cpu[2] = 2;
  idx_data_cpu[3] = 3;

  IndexSpec spec_cpu;
  spec_cpu.items.emplace_back(TensorIndex(idx_cpu));

  const int dev = 0;
  auto storage_cuda = vbt::cuda::new_cuda_storage(nbytes, dev);
  TensorImpl base_cuda(storage_cuda, sizes, strides, /*storage_offset=*/0,
                       ScalarType::Float32, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        base_cuda.data(), base_cpu.data(), nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for base failed";
  }

  auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
  TensorImpl idx_cuda(idx_storage_cuda, idx_sizes, idx_strides, 0,
                      ScalarType::Int64, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        idx_cuda.data(), idx_cpu.data(), idx_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for idx failed";
  }

  IndexSpec spec_cuda;
  spec_cuda.items.emplace_back(TensorIndex(idx_cuda));

  AdvancedIndexEnvConfig base_cfg = get_advanced_index_env_config_for_tests();

  std::vector<float> legacy_out(4, 0.0f);
  {
    AdvancedIndexEnvConfig cfg = base_cfg;
    cfg.cuda_bounds_default = CudaBoundsMode::LegacyHost;
    cfg.cuda_gpu_bounds_disable = false;

    AdvancedIndexEnvConfigGuard guard(cfg);
    EXPECT_EQ(get_effective_cuda_bounds_mode_for_tests(),
              CudaBoundsMode::LegacyHost);

    TensorImpl out = index(base_cuda, spec_cuda);
    ASSERT_EQ(out.device().type, kDLCUDA);
    ASSERT_EQ(out.sizes().size(), 1u);
    EXPECT_EQ(out.sizes()[0], 4);

    cudaError_t st = cudaMemcpy(
        legacy_out.data(), out.data(),
        static_cast<std::size_t>(4 * sizeof(float)),
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H (LegacyHost) failed";
  }

  std::vector<float> device_out(4, 0.0f);
  {
    AdvancedIndexEnvConfig cfg = base_cfg;
    cfg.cuda_bounds_default = CudaBoundsMode::DeviceNormalized;
    cfg.cuda_gpu_bounds_disable = false;

    AdvancedIndexEnvConfigGuard guard(cfg);
    EXPECT_EQ(get_effective_cuda_bounds_mode_for_tests(),
              CudaBoundsMode::DeviceNormalized);

    TensorImpl out = index(base_cuda, spec_cuda);
    ASSERT_EQ(out.device().type, kDLCUDA);
    ASSERT_EQ(out.sizes().size(), 1u);
    EXPECT_EQ(out.sizes()[0], 4);

    cudaError_t st = cudaMemcpy(
        device_out.data(), out.data(),
        static_cast<std::size_t>(4 * sizeof(float)),
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H (DeviceNormalized) failed";
  }

  for (std::size_t i = 0; i < legacy_out.size(); ++i) {
    EXPECT_FLOAT_EQ(legacy_out[i], device_out[i]);
  }
}

TEST(IndexingAdvancedCudaKernelTest, BoundsModeOobParity) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage_cpu = make_cpu_storage_bytes(nbytes);
  TensorImpl base_cpu(storage_cpu, sizes, strides, /*storage_offset=*/0,
                      ScalarType::Float32, Device::cpu());

  // Indices [0, 5] include an out-of-range value for D=5.
  const std::vector<std::int64_t> idx_sizes{2};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(2 * sizeof(std::int64_t));
  auto idx_storage_cpu = make_cpu_storage_bytes(idx_nbytes);
  TensorImpl idx_cpu(idx_storage_cpu, idx_sizes, idx_strides, 0,
                     ScalarType::Int64, Device::cpu());
  auto* idx_data_cpu = static_cast<std::int64_t*>(idx_cpu.data());
  idx_data_cpu[0] = 0;
  idx_data_cpu[1] = 5;  // OOB

  const int dev = 0;
  auto storage_cuda = vbt::cuda::new_cuda_storage(nbytes, dev);
  TensorImpl base_cuda(storage_cuda, sizes, strides, /*storage_offset=*/0,
                       ScalarType::Float32, Device::cuda(dev));

  auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
  TensorImpl idx_cuda(idx_storage_cuda, idx_sizes, idx_strides, 0,
                      ScalarType::Int64, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        idx_cuda.data(), idx_cpu.data(), idx_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for idx failed";
  }

  IndexSpec spec_cuda;
  spec_cuda.items.emplace_back(TensorIndex(idx_cuda));

  AdvancedIndexEnvConfig base_cfg = get_advanced_index_env_config_for_tests();

  std::string legacy_msg;
  {
    AdvancedIndexEnvConfig cfg = base_cfg;
    cfg.cuda_bounds_default = CudaBoundsMode::LegacyHost;
    cfg.cuda_gpu_bounds_disable = false;

    AdvancedIndexEnvConfigGuard guard(cfg);
    EXPECT_EQ(get_effective_cuda_bounds_mode_for_tests(),
              CudaBoundsMode::LegacyHost);

    try {
      (void)index(base_cuda, spec_cuda);
      FAIL() << "expected std::out_of_range in LegacyHost mode";
    } catch (const std::out_of_range& ex) {
      legacy_msg = ex.what();
      EXPECT_NE(legacy_msg.find(idx_errors::kErrIndexOutOfRange),
                std::string::npos);
    }
  }

  std::string device_msg;
  {
    AdvancedIndexEnvConfig cfg = base_cfg;
    cfg.cuda_bounds_default = CudaBoundsMode::DeviceNormalized;
    cfg.cuda_gpu_bounds_disable = false;

    AdvancedIndexEnvConfigGuard guard(cfg);
    EXPECT_EQ(get_effective_cuda_bounds_mode_for_tests(),
              CudaBoundsMode::DeviceNormalized);

    try {
      (void)index(base_cuda, spec_cuda);
      FAIL() << "expected std::out_of_range in DeviceNormalized mode";
    } catch (const std::out_of_range& ex) {
      device_msg = ex.what();
      EXPECT_NE(device_msg.find(idx_errors::kErrIndexOutOfRange),
                std::string::npos);
    }
  }

  EXPECT_EQ(legacy_msg, device_msg);
}

TEST(IndexingAdvancedCudaKernelTest, BoolMaskCubBackendAvoidsMaskD2H) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  const int dev = 0;

  // Base CUDA tensor: shape (2, 3), values [0..5].
  const std::vector<std::int64_t> base_sizes{2, 3};
  const std::vector<std::int64_t> base_strides{3, 1};
  const std::size_t base_nbytes = static_cast<std::size_t>(6 * sizeof(float));
  auto base_storage_cuda = vbt::cuda::new_cuda_storage(base_nbytes, dev);
  TensorImpl base_cuda(base_storage_cuda, base_sizes, base_strides,
                       /*storage_offset=*/0,
                       ScalarType::Float32,
                       Device::cuda(dev));
  {
    const std::vector<float> h_base{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    cudaError_t st = cudaMemcpy(
        base_cuda.data(), h_base.data(), base_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for base failed";
  }

  // Bool mask base: length 6, then take a stride-2 view of length 3.
  // Base bytes: [1, 0, 0, 1, 1, 0] -> view [1, 0, 1].
  const std::vector<std::int64_t> mask_base_sizes{6};
  const std::vector<std::int64_t> mask_base_strides{1};
  const std::size_t mask_base_nbytes = static_cast<std::size_t>(6 * sizeof(std::uint8_t));
  auto mask_storage_cuda = vbt::cuda::new_cuda_storage(mask_base_nbytes, dev);
  TensorImpl mask_base_cuda(mask_storage_cuda,
                            mask_base_sizes,
                            mask_base_strides,
                            /*storage_offset=*/0,
                            ScalarType::Bool,
                            Device::cuda(dev));
  {
    const std::vector<std::uint8_t> h_mask_base{1, 0, 0, 1, 1, 0};
    cudaError_t st = cudaMemcpy(
        mask_base_cuda.data(), h_mask_base.data(), mask_base_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for mask failed";
  }

  TensorImpl mask_view(mask_storage_cuda,
                      std::vector<std::int64_t>{3},
                      std::vector<std::int64_t>{2},
                      /*storage_offset=*/0,
                      ScalarType::Bool,
                      Device::cuda(dev));

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(Slice{}));  // prefix full-range slice
  spec.items.emplace_back(TensorIndex(mask_view));

  auto check_out = [](const TensorImpl& out) {
    ASSERT_EQ(out.sizes().size(), 2u);
    EXPECT_EQ(out.sizes()[0], 2);
    EXPECT_EQ(out.sizes()[1], 2);

    std::vector<float> out_host(4);
    cudaError_t st = cudaMemcpy(
        out_host.data(), out.data(), 4 * sizeof(float),
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for out failed";

    // Expected values: columns {0, 2}.
    EXPECT_FLOAT_EQ(out_host[0], 0.f);
    EXPECT_FLOAT_EQ(out_host[1], 2.f);
    EXPECT_FLOAT_EQ(out_host[2], 3.f);
    EXPECT_FLOAT_EQ(out_host[3], 5.f);
  };

  AdvancedIndexEnvConfig base_cfg = get_advanced_index_env_config_for_tests();
  base_cfg.cuda_allow_bool_mask_indices = true;

  // Legacy path should perform a full-mask D2H copy.
  {
    AdvancedIndexEnvConfig cfg = base_cfg;
    cfg.cuda_bool_mask_use_cub = false;
    AdvancedIndexEnvConfigGuard guard(cfg);

    reset_m_index_advanced_stats_for_tests();
    TensorImpl out = index(base_cuda, spec);
    check_out(out);

    const auto& stats = get_m_index_advanced_stats();
    EXPECT_EQ(stats.cuda_bool_mask_d2h_bytes.load(std::memory_order_relaxed), 3u);
  }

  // CUB backend should avoid full-mask D2H and only sync a scalar count.
  {
    AdvancedIndexEnvConfig cfg = base_cfg;
    cfg.cuda_bool_mask_use_cub = true;
    AdvancedIndexEnvConfigGuard guard(cfg);

    reset_m_index_advanced_stats_for_tests();
    TensorImpl out = index(base_cuda, spec);
    check_out(out);

    const auto& stats = get_m_index_advanced_stats();
    EXPECT_EQ(stats.cuda_bool_mask_d2h_bytes.load(std::memory_order_relaxed), 0u);
  }
}

TEST(IndexingAdvancedCudaKernelTest, Make1DGridRespectsDeviceAndEnvCaps) {
  const int dev = 0;

  // Force a deterministic device cap so tests do not depend on the
  // real hardware limit.
  set_device_max_grid_x_override_for_tests(dev, 128u);

  AdvancedIndexEnvConfig base_cfg = get_advanced_index_env_config_for_tests();
  const int threads = 256;

  // G1: Env unset/effective cap <= 0; small N -> ceil_div(N, threads).
  {
    AdvancedIndexEnvConfig cfg = base_cfg;
    cfg.cuda_max_blocks_cap = 0;
    AdvancedIndexEnvConfigGuard guard(cfg);

    const std::int64_t N = 1000;
    const unsigned int grid_x =
        get_1d_grid_x_for_tests(N, threads, dev);
    const unsigned int expected =
        static_cast<unsigned int>((N + threads - 1) / threads);
    EXPECT_EQ(grid_x, expected);
    EXPECT_GE(grid_x, 1u);
    EXPECT_LE(grid_x, 128u);
  }

  // G2: Env unset; large N near result cap -> clamp to device_max.
  {
    AdvancedIndexEnvConfig cfg = base_cfg;
    cfg.cuda_max_blocks_cap = 0;
    AdvancedIndexEnvConfigGuard guard(cfg);

    const std::int64_t N = 128LL * threads * 10;
    const unsigned int grid_x =
        get_1d_grid_x_for_tests(N, threads, dev);
    EXPECT_EQ(grid_x, 128u);
  }

  // G3: Env smaller than device_max -> env cap wins.
  {
    AdvancedIndexEnvConfig cfg = base_cfg;
    cfg.cuda_max_blocks_cap = 4;
    AdvancedIndexEnvConfigGuard guard(cfg);

    const std::int64_t N = 128LL * threads * 10;
    const unsigned int grid_x =
        get_1d_grid_x_for_tests(N, threads, dev);
    EXPECT_EQ(grid_x, 4u);
  }

  // G4: Env larger than device_max -> device_max wins.
  {
    AdvancedIndexEnvConfig cfg = base_cfg;
    cfg.cuda_max_blocks_cap = 1000;
    AdvancedIndexEnvConfigGuard guard(cfg);

    const std::int64_t N = 128LL * threads * 10;
    const unsigned int grid_x =
        get_1d_grid_x_for_tests(N, threads, dev);
    EXPECT_EQ(grid_x, 128u);
  }

  // G5: Negative env cap behaves like unset.
  {
    AdvancedIndexEnvConfig cfg = base_cfg;
    cfg.cuda_max_blocks_cap = -5;
    AdvancedIndexEnvConfigGuard guard(cfg);

    const std::int64_t N = 128LL * threads * 10;
    const unsigned int grid_x =
        get_1d_grid_x_for_tests(N, threads, dev);
    EXPECT_EQ(grid_x, 128u);
  }

  set_device_max_grid_x_override_for_tests(dev, std::nullopt);
}

TEST(IndexingAdvancedCudaKernelTest, Make1DGridDeviceCapsOverrideSingleBlockMatchesCpu) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  const int dev = 0;
  set_device_max_grid_x_override_for_tests(dev, 1u);

  AdvancedIndexEnvConfig base_cfg = get_advanced_index_env_config_for_tests();
  AdvancedIndexEnvConfig cfg = base_cfg;
  cfg.cuda_max_blocks_cap = 0;
  AdvancedIndexEnvConfigGuard guard(cfg);

  const std::vector<std::int64_t> sizes{1024};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(1024 * sizeof(float));
  auto storage_cpu = make_cpu_storage_bytes(nbytes);
  TensorImpl base_cpu(storage_cpu, sizes, strides, /*storage_offset=*/0,
                      ScalarType::Float32, Device::cpu());
  auto* base_data_cpu = static_cast<float*>(base_cpu.data());
  for (int i = 0; i < 1024; ++i) {
    base_data_cpu[i] = static_cast<float>(i);
  }

  const std::vector<std::int64_t> idx_sizes{1024};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(1024 * sizeof(std::int64_t));
  auto idx_storage_cpu = make_cpu_storage_bytes(idx_nbytes);
  TensorImpl idx_cpu(idx_storage_cpu, idx_sizes, idx_strides, 0,
                     ScalarType::Int64, Device::cpu());
  auto* idx_data_cpu = static_cast<std::int64_t*>(idx_cpu.data());
  for (int i = 0; i < 1024; ++i) {
    idx_data_cpu[i] = i;
  }

  IndexSpec spec_cpu;
  spec_cpu.items.emplace_back(TensorIndex(idx_cpu));

  // CPU baseline.
  TensorImpl expected = vbt::core::indexing::index(base_cpu, spec_cpu);

  auto storage_cuda = vbt::cuda::new_cuda_storage(nbytes, dev);
  TensorImpl base_cuda(storage_cuda, sizes, strides, /*storage_offset=*/0,
                       ScalarType::Float32, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        base_cuda.data(), base_cpu.data(), nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for base failed";
  }

  auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
  TensorImpl idx_cuda(idx_storage_cuda, idx_sizes, idx_strides, 0,
                      ScalarType::Int64, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        idx_cuda.data(), idx_cpu.data(), idx_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for idx failed";
  }

  IndexSpec spec_cuda;
  spec_cuda.items.emplace_back(TensorIndex(idx_cuda));

  TensorImpl out_cuda = vbt::core::indexing::index(base_cuda, spec_cuda);

  // With the override, make_1d_grid must choose a single block.
  const unsigned int grid_x = get_1d_grid_x_for_tests(
      static_cast<std::int64_t>(expected.numel()), 256, dev);
  EXPECT_EQ(grid_x, 1u);

  std::vector<float> out_host(1024);
  {
    cudaError_t st = cudaMemcpy(
        out_host.data(), out_cuda.data(), nbytes,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for out failed";
  }

  const float* expected_data = static_cast<const float*>(expected.data());
  for (int i = 0; i < 1024; ++i) {
    EXPECT_FLOAT_EQ(expected_data[i], out_host[static_cast<std::size_t>(i)]);
  }

  set_device_max_grid_x_override_for_tests(dev, std::nullopt);
}

#endif  // VBT_WITH_CUDA && VBT_INTERNAL_TESTS
