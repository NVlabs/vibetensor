// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>

#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/kernel_function.h"

using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::TensorImpl;
using vbt::dispatch::BoxedStack;
using vbt::dispatch::Dispatcher;
using vbt::dispatch::KernelFunction;

static TensorImpl make_cpu_scalar_i64(std::int64_t v) {
  void* buf = ::operator new(sizeof(std::int64_t));
  *static_cast<std::int64_t*>(buf) = v;
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto st = vbt::core::make_intrusive<Storage>(std::move(dp), sizeof(std::int64_t));
  return TensorImpl(st, /*sizes=*/{}, /*strides=*/{}, /*storage_offset=*/0, ScalarType::Int64, Device::cpu());
}

#if VBT_WITH_CUDA
static TensorImpl make_cuda_tensor_f32(int dev) {
  auto st = vbt::cuda::new_cuda_storage(4 * vbt::core::itemsize(ScalarType::Float32), dev);
  return TensorImpl(st, /*sizes=*/{4}, /*strides=*/{1}, /*storage_offset=*/0, ScalarType::Float32, Device::cuda(dev));
}
#endif

static std::atomic<int> g_boxed_calls{0};

static void boxed_echo_first(BoxedStack& s) {
  g_boxed_calls.fetch_add(1, std::memory_order_relaxed);
  TensorImpl out = s[0];
  s.clear();
  s.push_back(out);
}

TEST(DispatcherFabricTest, DefaultsAreFalse) {
  auto& D = Dispatcher::instance();
  const char* op = "fabric_testlib::defaults";
  if (!D.has(op)) {
    D.registerLibrary("fabric_testlib");
    D.def("fabric_testlib::defaults(Tensor) -> Tensor");

    // minimal CPU kernel
    auto id = [](const TensorImpl& a) { return a; };
    D.registerCpuKernel(op, +id);
  }

  auto h = D.find(op);
  EXPECT_FALSE(h.get().is_fabric_op);
  EXPECT_FALSE(h.get().allow_multi_device_fabric);
}

TEST(DispatcherFabricTest, MarkFabricOpInvariants) {
  auto& D = Dispatcher::instance();

  // Unknown op
  EXPECT_THROW(D.mark_fabric_op("__missing__::op", true, true), std::invalid_argument);

  // allow_multi_device_fabric implies is_fabric_op
  const char* op_bad = "fabric_testlib::mark_bad";
  if (!D.has(op_bad)) {
    D.registerLibrary("fabric_testlib");
    D.def("fabric_testlib::mark_bad(Tensor) -> Tensor");
    auto id = [](const TensorImpl& a) { return a; };
    D.registerCpuKernel(op_bad, +id);
  }
  EXPECT_THROW(D.mark_fabric_op(op_bad, /*is_fabric_op=*/false, /*allow_multi_device_fabric=*/true), std::runtime_error);

  // mark_fabric_op must reject existing autograd fallback
  const char* op_has_ag = "fabric_testlib::mark_has_autograd";
  if (!D.has(op_has_ag)) {
    D.registerLibrary("fabric_testlib");
    D.def("fabric_testlib::mark_has_autograd(Tensor) -> Tensor");
    auto id = [](const TensorImpl& a) { return a; };
    D.registerCpuKernel(op_has_ag, +id);

    auto boxed_ctx = [](void* /*ctx*/, BoxedStack& s) {
      // Never called in this test; if it is, just return the first arg.
      TensorImpl out = s[0];
      s.clear();
      s.push_back(out);
    };
    KernelFunction kf = KernelFunction::makeBoxedCtx(/*arity=*/1, +boxed_ctx, /*ctx=*/nullptr);
    D.registerAutogradFallback(op_has_ag, kf);
  }
  EXPECT_THROW(D.mark_fabric_op(op_has_ag, /*is_fabric_op=*/true, /*allow_multi_device_fabric=*/true), std::runtime_error);

  // registerAutogradFallback must reject Fabric ops
  const char* op_marked = "fabric_testlib::mark_then_autograd";
  if (!D.has(op_marked)) {
    D.registerLibrary("fabric_testlib");
    D.def("fabric_testlib::mark_then_autograd(Tensor) -> Tensor");
    auto id = [](const TensorImpl& a) { return a; };
    D.registerCpuKernel(op_marked, +id);
    D.mark_fabric_op(op_marked, /*is_fabric_op=*/true, /*allow_multi_device_fabric=*/false);
  }
  auto boxed_ctx2 = [](void* /*ctx*/, BoxedStack& s) {
    TensorImpl out = s[0];
    s.clear();
    s.push_back(out);
  };
  KernelFunction kf2 = KernelFunction::makeBoxedCtx(/*arity=*/1, +boxed_ctx2, /*ctx=*/nullptr);
  EXPECT_THROW(D.registerAutogradFallback(op_marked, kf2), std::runtime_error);
}

TEST(DispatcherFabricTest, CallBoxedBypassIsGuarded) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA not built";
#else
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& D = Dispatcher::instance();

  const char* op_non = "fabric_testlib::dummy_non";
  if (!D.has(op_non)) {
    D.registerLibrary("fabric_testlib");
    D.def("fabric_testlib::dummy_non(Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");
    D.registerCudaKernelFunction(op_non, KernelFunction::makeBoxed(/*arity=*/5, &boxed_echo_first));
  }

  const char* op_fab = "fabric_testlib::dummy_fabric";
  if (!D.has(op_fab)) {
    D.registerLibrary("fabric_testlib");
    D.def("fabric_testlib::dummy_fabric(Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");
    D.registerCudaKernelFunction(op_fab, KernelFunction::makeBoxed(/*arity=*/5, &boxed_echo_first));
    D.mark_fabric_op(op_fab, /*is_fabric_op=*/true, /*allow_multi_device_fabric=*/true);
  }

  TensorImpl a0 = make_cuda_tensor_f32(/*dev=*/0);
  TensorImpl b0 = make_cuda_tensor_f32(/*dev=*/0);

  TensorImpl compute0 = make_cpu_scalar_i64(0);
  TensorImpl require1 = make_cpu_scalar_i64(1);
  TensorImpl fallback1 = make_cpu_scalar_i64(1);

  // Non-fabric op must still enforce same-device invariant (CPU scalars are a mismatch).
  {
    BoxedStack s{a0, b0, compute0, require1, fallback1};
    EXPECT_THROW(D.callBoxed(op_non, s), std::runtime_error);
  }

  // Fabric-marked op bypasses same-device invariant and dispatches to CUDA base.
  {
    const int before = g_boxed_calls.load(std::memory_order_relaxed);
    BoxedStack s{a0, b0, compute0, require1, fallback1};
    D.callBoxed(op_fab, s);
    EXPECT_EQ(s.size(), 1u);
    EXPECT_EQ(s[0].device(), a0.device());
    EXPECT_EQ(g_boxed_calls.load(std::memory_order_relaxed), before + 1);
  }

  // Bad compute_device must be rejected before kernel execution.
  {
    const bool no_cuda_calls = vbt::dispatch::dispatch_v2_fabric_no_cuda_calls();
    TensorImpl bad_compute = make_cpu_scalar_i64(999);
    BoxedStack s{a0, b0, bad_compute, require1, fallback1};
    try {
      D.callBoxed(op_fab, s);
      FAIL() << "expected compute_device error";
    } catch (const std::runtime_error& e) {
      std::string msg = e.what();
      EXPECT_NE(msg.find("[Fabric]"), std::string::npos);
      if (no_cuda_calls) {
        EXPECT_NE(msg.find("must match one of the operand devices"),
                  std::string::npos);
      } else {
        EXPECT_NE(msg.find("out of range"), std::string::npos);
      }
    }
  }

  // Two-device placement allowed only when primary matches an operand.
  if (vbt::cuda::device_count() >= 2) {
    TensorImpl b1 = make_cuda_tensor_f32(/*dev=*/1);

    // Valid: primary=0, operand on cuda:0 exists.
    {
      BoxedStack s{a0, b1, compute0, require1, fallback1};
      D.callBoxed(op_fab, s);
      EXPECT_EQ(s.size(), 1u);
      EXPECT_EQ(s[0].device(), a0.device());
    }

    // Invalid: primary=1 but no operand on cuda:1.
    {
      TensorImpl compute1 = make_cpu_scalar_i64(1);
      BoxedStack s{a0, b0, compute1, require1, fallback1};
      try {
        D.callBoxed(op_fab, s);
        FAIL() << "expected primary-must-match-operand error";
      } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("must match one of the operand devices"), std::string::npos);
      }
    }
  }
#endif
}
