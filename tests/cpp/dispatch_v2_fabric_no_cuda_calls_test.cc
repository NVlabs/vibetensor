// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/guard.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/kernel_function.h"

using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::TensorImpl;
using vbt::dispatch::BoxedStack;
using vbt::dispatch::Dispatcher;
using vbt::dispatch::KernelFunction;

#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
namespace {

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(
      DataPtr(::operator new(nbytes),
              [](void* p) noexcept { ::operator delete(p); }),
      nbytes);
}

static TensorImpl cpu_scalar_i64(std::int64_t v) {
  auto st = make_storage_bytes(sizeof(std::int64_t));
  TensorImpl t(st, /*sizes=*/{}, /*strides=*/{}, /*storage_offset=*/0,
               ScalarType::Int64, Device::cpu());
  auto* p = static_cast<std::int64_t*>(t.data());
  if (p) *p = v;
  return t;
}

static TensorImpl fake_cuda_tensor_f32_1d(std::size_t n, int dev) {
  auto st = make_storage_bytes(n * sizeof(float));
  return TensorImpl(st,
                    /*sizes=*/{static_cast<std::int64_t>(n)},
                    /*strides=*/{1},
                    /*storage_offset=*/0,
                    ScalarType::Float32,
                    Device::cuda(dev));
}

static std::atomic<int> g_boxed_calls{0};

static void boxed_echo_first(BoxedStack& s) {
  g_boxed_calls.fetch_add(1, std::memory_order_relaxed);
  TensorImpl out = s[0];
  s.clear();
  s.push_back(out);
}

}  // namespace
#endif

TEST(DispatchV2FabricNoCudaCallsTest, HelperSkipsCudaCallsWhenFlagEnabled) {
#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
  vbt::dispatch::DispatchV2ModeGuard v2(true);

  auto& D = Dispatcher::instance();
  const std::string fqname = "test_v2_fabric::dummy";

  if (!D.has(fqname)) {
    D.registerLibrary("test_v2_fabric");
    D.def(fqname + "(Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");
    D.registerCudaKernelFunction(fqname,
                                 KernelFunction::makeBoxed(/*arity=*/5,
                                                          &boxed_echo_first));
    D.mark_fabric_op(fqname, /*is_fabric_op=*/true,
                     /*allow_multi_device_fabric=*/true);
  }

  struct TestCase {
    const char* name;
    bool flag_on;
    std::int64_t compute_device;
    bool expect_throw;
    const char* expect_substr;
    std::uint64_t dc_delta;
    std::uint64_t dg_delta;
    int boxed_delta;
  };

  const std::int64_t devindex_max_plus_1 =
      static_cast<std::int64_t>(
          std::numeric_limits<vbt::cuda::DeviceIndex>::max()) +
      1;

  const std::vector<TestCase> cases = {
      {"match_success_no_cuda", true, 0, false, nullptr, 0, 0, 1},
      {"negative_invalid", true, -1, true, "is invalid", 0, 0, 0},
      {"overflow_invalid", true, (static_cast<std::int64_t>(1) << 40) + 1, true,
       "is invalid", 0, 0, 0},
      {"devindex_max_plus_1_invalid", true, devindex_max_plus_1, true,
       "is invalid", 0, 0, 0},
      {"mismatch_in_range", true, 999, true,
       "must match one of the operand devices", 0, 0, 0},
      {"out_of_range_max_i64", false, std::numeric_limits<std::int64_t>::max(),
       true, "out of range for visible CUDA devices", 1, 0, 0},
  };

  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.name);

    // These counters are process-global; reset per-row so that the deltas below
    // reflect only the single Dispatcher::callBoxed invocation.
    vbt::cuda::reset_device_count_calls_for_tests();
    vbt::cuda::reset_device_guard_ctor_calls_for_tests();
    g_boxed_calls.store(0, std::memory_order_relaxed);

    vbt::dispatch::DispatchV2FabricNoCudaCallsGuard no_cuda(tc.flag_on);
    ASSERT_EQ(vbt::dispatch::dispatch_v2_fabric_no_cuda_calls(), tc.flag_on);

    TensorImpl a0 = fake_cuda_tensor_f32_1d(/*n=*/4, /*dev=*/0);
    TensorImpl b1 = fake_cuda_tensor_f32_1d(/*n=*/4, /*dev=*/1);

    TensorImpl compute = cpu_scalar_i64(tc.compute_device);
    TensorImpl require1 = cpu_scalar_i64(1);
    TensorImpl fallback1 = cpu_scalar_i64(1);

    BoxedStack stack{a0, b1, compute, require1, fallback1};

    const std::uint64_t dc0 = vbt::cuda::device_count_calls_for_tests();
    const std::uint64_t dg0 = vbt::cuda::device_guard_ctor_calls_for_tests();
    const int boxed0 = g_boxed_calls.load(std::memory_order_relaxed);
    ASSERT_EQ(dc0, 0u);
    ASSERT_EQ(dg0, 0u);
    ASSERT_EQ(boxed0, 0);

    if (tc.expect_throw) {
      try {
        D.callBoxed(fqname, stack);
        FAIL() << "expected std::runtime_error";
      } catch (const std::runtime_error& e) {
        const std::string msg(e.what());
        EXPECT_NE(msg.find(tc.expect_substr), std::string::npos) << msg;
      }
    } else {
      D.callBoxed(fqname, stack);
      ASSERT_EQ(stack.size(), 1u);
      EXPECT_EQ(stack[0].device(), a0.device());
    }

    const std::uint64_t dc1 = vbt::cuda::device_count_calls_for_tests();
    const std::uint64_t dg1 = vbt::cuda::device_guard_ctor_calls_for_tests();
    const int boxed1 = g_boxed_calls.load(std::memory_order_relaxed);
    EXPECT_EQ(dc1 - dc0, tc.dc_delta);
    EXPECT_EQ(dg1 - dg0, tc.dg_delta);
    EXPECT_EQ(boxed1 - boxed0, tc.boxed_delta);
  }
#else
  GTEST_SKIP() << "dispatch v2 or internal tests disabled";
#endif
}
