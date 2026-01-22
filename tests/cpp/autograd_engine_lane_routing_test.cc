// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vbt/autograd/lane_routing.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"

#if VBT_WITH_CUDA
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"
#include <cuda_runtime_api.h>
#endif

using vbt::autograd::AccumulateGrad;
using vbt::autograd::AutogradMeta;
using vbt::autograd::Lane;
using vbt::autograd::LaneMode;
using vbt::autograd::Node;
using vbt::autograd::OptionalTensor;
using vbt::autograd::StreamKind;
using vbt::autograd::lane_for_node;
using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::TensorImpl;

namespace {

static TensorImpl make_cpu_dense_f32(const std::vector<int64_t>& sizes, float fill) {
  std::size_t ne = 1;
  bool has_zero_dim = false;
  for (auto s : sizes) {
    if (s == 0) {
      has_zero_dim = true;
    } else {
      ne *= static_cast<std::size_t>(s);
    }
  }
  if (has_zero_dim) ne = 0;
  const std::size_t nbytes = ne * sizeof(float);

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  StoragePtr st = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    acc *= (sizes[idx] == 0 ? 1 : sizes[idx]);
  }

  TensorImpl t(st, sizes, strides, /*storage_offset=*/0, ScalarType::Float32, Device::cpu());
  if (ne > 0) {
    float* p = static_cast<float*>(t.data());
    for (std::size_t i = 0; i < ne; ++i) {
      p[i] = fill;
    }
  }
  return t;
}

#if VBT_WITH_CUDA
static TensorImpl make_cuda_dense_f32(const std::vector<int64_t>& sizes, float fill, int dev = 0) {
  std::size_t ne = 1;
  bool has_zero_dim = false;
  for (auto s : sizes) {
    if (s == 0) {
      has_zero_dim = true;
    } else {
      ne *= static_cast<std::size_t>(s);
    }
  }
  if (has_zero_dim) ne = 0;
  const std::size_t nbytes = ne * sizeof(float);

  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  if (nbytes > 0) {
    std::vector<float> host(ne, fill);
    cudaError_t st = cudaMemcpy(storage->data(), host.data(), nbytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy H2D failed in make_cuda_dense_f32");
    }
  }

  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    acc *= (sizes[idx] == 0 ? 1 : sizes[idx]);
  }

  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0, ScalarType::Float32, Device::cuda(dev));
}
#endif

struct CpuOnlyNode final : Node {
  uint32_t num_inputs() const noexcept override { return 1; }
  StreamKind stream_kind() const noexcept override { return StreamKind::CpuOnly; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    (void)grads_in;
    return std::vector<OptionalTensor>(1);
  }
};

struct CudaAllowlistedNode final : Node {
  uint32_t num_inputs() const noexcept override { return 1; }
  StreamKind stream_kind() const noexcept override { return StreamKind::CudaAllowlisted; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    (void)grads_in;
    return std::vector<OptionalTensor>(1);
  }
};

} // namespace

TEST(AutogradLaneRoutingTest, CpuAutogradDeviceAlwaysCpuLane) {
  Device autograd_dev = Device::cpu(0);

  CpuOnlyNode cpu_only;
  CudaAllowlistedNode cuda_allowlisted;

  EXPECT_EQ(lane_for_node(autograd_dev, cpu_only), Lane::CPU);
  EXPECT_EQ(lane_for_node(autograd_dev, cuda_allowlisted), Lane::CPU);
}

TEST(AutogradLaneRoutingTest, CudaAutogradDeviceRoutesByStreamKind) {
  Device autograd_dev = Device::cuda(0);

  CpuOnlyNode cpu_only;
  CudaAllowlistedNode cuda_allowlisted;

  EXPECT_EQ(lane_for_node(autograd_dev, cpu_only), Lane::CPU);
  EXPECT_EQ(lane_for_node(autograd_dev, cuda_allowlisted), Lane::CUDA);
}

TEST(AutogradLaneRoutingTest, SingleLaneCpuOverrideForcesCpuLane) {
  Device autograd_dev = Device::cuda(0);
  CudaAllowlistedNode cuda_allowlisted;

  EXPECT_EQ(lane_for_node(autograd_dev, cuda_allowlisted, LaneMode::SingleLaneCPU), Lane::CPU);
}

TEST(AutogradLaneRoutingTest, AccumulateGradAlwaysCpuLane) {
  TensorImpl leaf = make_cpu_dense_f32({2}, 0.0f);
  AutogradMeta* meta = vbt::autograd::get_autograd_meta(leaf, /*create_if_missing=*/true);
  ASSERT_NE(meta, nullptr);

  AccumulateGrad acc(meta);

  EXPECT_EQ(acc.stream_kind(), StreamKind::CpuOnly);
  EXPECT_EQ(lane_for_node(Device::cpu(0), acc), Lane::CPU);
  EXPECT_EQ(lane_for_node(Device::cuda(0), acc), Lane::CPU);
}

#if VBT_WITH_CUDA
TEST(AutogradLaneRoutingTest, AccumulateGradCudaLeafStillCpuLane) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  TensorImpl leaf = make_cuda_dense_f32({2}, 0.0f, /*dev=*/0);
  AutogradMeta* meta = vbt::autograd::get_autograd_meta(leaf, /*create_if_missing=*/true);
  ASSERT_NE(meta, nullptr);

  AccumulateGrad acc(meta);
  _tag_accumulategrad_cuda_leaf(acc, leaf);

  ASSERT_EQ(acc.stream_kind(), StreamKind::CudaAllowlisted);
  EXPECT_EQ(lane_for_node(Device::cuda(0), acc), Lane::CPU);
}
#endif
