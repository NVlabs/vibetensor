// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/dispatch/dispatcher.h"
#if VBT_WITH_CUDA
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"
#endif

using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::TensorImpl;
using vbt::dispatch::BoxedStack;
using vbt::dispatch::Dispatcher;

#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS

namespace {

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(
      DataPtr(::operator new(nbytes),
              [](void* p) noexcept { ::operator delete(p); }),
      nbytes);
}

static TensorImpl cpu_tensor_f32_1d(std::size_t n) {
  auto st = make_storage_bytes(n * sizeof(float));
  TensorImpl t(st, /*sizes=*/{static_cast<std::int64_t>(n)}, /*strides=*/{1},
               /*storage_offset=*/0, ScalarType::Float32, Device::cpu());
  return t;
}

#if VBT_WITH_CUDA
static TensorImpl cuda_tensor_f32_1d(std::size_t n, int dev) {
  auto st = vbt::cuda::new_cuda_storage(
      n * vbt::core::itemsize(ScalarType::Float32), dev);
  TensorImpl t(st, /*sizes=*/{static_cast<std::int64_t>(n)}, /*strides=*/{1},
               /*storage_offset=*/0, ScalarType::Float32, Device::cuda(dev));
  return t;
}
#endif  // VBT_WITH_CUDA

}  // namespace

TEST(DispatchV2NoNameAllowlistsTest, AllSameDeviceRejectsMixedCpuCuda) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA not available";
#else
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& D = Dispatcher::instance();
  const std::string fqname = "test_dispatch_v2::no_name_allowlists_mixed";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_dispatch_v2");
  D.def(fqname + "(Tensor, Tensor) -> Tensor");

  BoxedStack stack;
  stack.emplace_back(cpu_tensor_f32_1d(/*n=*/1));
  stack.emplace_back(cuda_tensor_f32_1d(/*n=*/1, /*dev=*/0));

  try {
    D.callBoxed(fqname, stack);
    FAIL() << "expected mixed-device error";
  } catch (const std::runtime_error& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("Expected all tensors to be on the same device"),
              std::string::npos)
        << msg;
    EXPECT_NE(msg.find("cpu:0"), std::string::npos) << msg;
    EXPECT_NE(msg.find("cuda:0"), std::string::npos) << msg;
  }
#endif
}

#else

TEST(DispatchV2NoNameAllowlistsTest, SkippedWhenDispatchV2Disabled) {
  GTEST_SKIP() << "dispatch v2 or internal tests disabled";
}

#endif  // VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
