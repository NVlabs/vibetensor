// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>
#include <stdexcept>

#include "vbt/core/tensor_iter.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"

#ifndef VBT_TI_ENABLE_TEST_HOOKS
#error "VBT_TI_ENABLE_TEST_HOOKS must be defined for TI Fabric tests"
#endif

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::TensorIter;
using vbt::core::testing::TensorIterTestHelper;

namespace {

static TensorImpl make_cuda_contiguous_tensor(std::vector<std::int64_t> sizes,
                                             int device_index) {
#if VBT_WITH_CUDA
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

  const std::size_t nbytes = static_cast<std::size_t>(ne) * sizeof(float);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, device_index);

  return TensorImpl(storage,
                    std::move(sizes),
                    std::move(strides),
                    /*storage_offset=*/0,
                    ScalarType::Float32,
                    Device::cuda(device_index));
#else
  (void)sizes;
  (void)device_index;
  throw std::runtime_error("CUDA not built");
#endif
}

static TensorImpl make_cpu_scalar_float(float v) {
  // 0-dim scalar: sizes == {}, strides == {}
  auto storage = vbt::core::make_intrusive<vbt::core::Storage>(
      vbt::core::DataPtr(::operator new(sizeof(float)),
                         [](void* p) noexcept { ::operator delete(p); }),
      sizeof(float));
  auto t = TensorImpl(storage,
                      /*sizes=*/{},
                      /*strides=*/{},
                      /*storage_offset=*/0,
                      ScalarType::Float32,
                      Device::cpu());
  *static_cast<float*>(t.data()) = v;
  return t;
}

}  // namespace

TEST(TensorIterFabricTest, BuildSucceedsForTwoCudaDevicesWithRemoteReadOnlyInput) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 CUDA devices";
  }

  TensorImpl out = make_cuda_contiguous_tensor({4}, /*device_index=*/0);
  TensorImpl a   = make_cuda_contiguous_tensor({4}, /*device_index=*/0);
  TensorImpl b   = make_cuda_contiguous_tensor({4}, /*device_index=*/1);

  TensorIter iter = vbt::core::make_fabric_elementwise_2gpu_iter(
      out, a, b, Device::cuda(0));

  EXPECT_EQ(TensorIterTestHelper::common_device(iter), Device::cuda(0));
  ASSERT_EQ(iter.ntensors(), 3);
  EXPECT_EQ(iter.operand(0).device, Device::cuda(0));
  EXPECT_EQ(iter.operand(1).device, Device::cuda(0));
  EXPECT_EQ(iter.operand(2).device, Device::cuda(1));
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(TensorIterFabricTest, BuildSucceedsWhenEitherInputIsRemote) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 CUDA devices";
  }

  TensorImpl out = make_cuda_contiguous_tensor({4}, /*device_index=*/0);
  TensorImpl a   = make_cuda_contiguous_tensor({4}, /*device_index=*/1);
  TensorImpl b   = make_cuda_contiguous_tensor({4}, /*device_index=*/0);

  TensorIter iter = vbt::core::make_fabric_elementwise_2gpu_iter(
      out, a, b, Device::cuda(0));

  EXPECT_EQ(TensorIterTestHelper::common_device(iter), Device::cuda(0));
  EXPECT_EQ(iter.operand(0).device, Device::cuda(0));
  EXPECT_EQ(iter.operand(1).device, Device::cuda(1));
  EXPECT_EQ(iter.operand(2).device, Device::cuda(0));
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(TensorIterFabricTest, RejectsOutputNotOnPrimaryDevice) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 CUDA devices";
  }

  TensorImpl out = make_cuda_contiguous_tensor({4}, /*device_index=*/0);
  TensorImpl a   = make_cuda_contiguous_tensor({4}, /*device_index=*/0);
  TensorImpl b   = make_cuda_contiguous_tensor({4}, /*device_index=*/1);

  vbt::core::TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true),
                 vbt::core::IterOperandRole::WriteOnly,
                 /*allow_resize=*/false)
      .add_input(a)
      .add_input(b)
      .check_all_same_device(false)
      .enable_fabric_2gpu_elementwise(/*primary_device=*/2);

  EXPECT_THROW((void)cfg.build(), std::invalid_argument);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(TensorIterFabricTest, RejectsCpuScalarOperand) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() < 1) {
    GTEST_SKIP() << "Need >= 1 CUDA device";
  }

  TensorImpl out = make_cuda_contiguous_tensor({4}, /*device_index=*/0);
  TensorImpl a   = make_cuda_contiguous_tensor({4}, /*device_index=*/0);
  TensorImpl s   = make_cpu_scalar_float(1.0f);

  vbt::core::TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true),
                 vbt::core::IterOperandRole::WriteOnly,
                 /*allow_resize=*/false)
      .add_input(a)
      .add_input(s)
      .check_all_same_device(false)
      .enable_fabric_2gpu_elementwise(/*primary_device=*/0);

  EXPECT_THROW((void)cfg.build(), std::invalid_argument);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(TensorIterFabricTest, RejectsRemoteOutput) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 CUDA devices";
  }

  TensorImpl out = make_cuda_contiguous_tensor({4}, /*device_index=*/1);
  TensorImpl a   = make_cuda_contiguous_tensor({4}, /*device_index=*/0);
  TensorImpl b   = make_cuda_contiguous_tensor({4}, /*device_index=*/1);

  EXPECT_THROW((void)vbt::core::make_fabric_elementwise_2gpu_iter(
                   out, a, b, Device::cuda(0)),
               std::invalid_argument);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
