// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdexcept>
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/registration.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/autograd/wrapper.h"
#if VBT_WITH_CUDA
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"
#endif

using vbt::dispatch::Dispatcher;
using vbt::dispatch::OperatorHandle;
using vbt::dispatch::BoxedStack;
using vbt::dispatch::KernelFunction;
using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;

static vbt::core::StoragePtr make_cpu_storage(void* base, std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(DataPtr(base, [](void*) noexcept {}), nbytes);
}

static int g_fallback_counter = 0;
static void fallback_ctx_count(void* ctx, BoxedStack& s) {
  ++g_fallback_counter;
  auto* entry = static_cast<vbt::dispatch::OperatorEntry*>(ctx);
  vbt::autograd::SkipAutogradGuard g;
  Dispatcher::instance().redispatchBoxed(vbt::dispatch::OperatorHandle(entry), s);
}

// Incorrect fallback that calls callBoxed under RedispatchGuard to test safety-net
static int g_incorrect_invocations = 0;
static void fallback_ctx_incorrect(void* ctx, BoxedStack& s) {
  ++g_incorrect_invocations;
  auto* entry = static_cast<vbt::dispatch::OperatorEntry*>(ctx);
  vbt::dispatch::Dispatcher::RedispatchGuard rg(entry);
  Dispatcher::instance().callBoxed(vbt::dispatch::OperatorHandle(entry), s);
}

static TensorImpl id_kernel(const TensorImpl& a) { return a; }

TEST(SkipAutogradGuardRAII, NestingAndExceptionSafety) {
#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
  vbt::dispatch::DispatchV2ModeGuard v2_guard{true};
#endif
  auto& D = Dispatcher::instance();
  const char* op = "vt::id_guard";
  if (!D.has(op)) {
    D.registerLibrary("vt");
    D.def(std::string(op) + "(Tensor) -> Tensor");
    D.registerCpuKernel(op, &id_kernel);
  }
  KernelFunction kf = KernelFunction::makeBoxedCtx(/*arity=*/1, &fallback_ctx_count, /*ctx=*/nullptr);
  (void)D.tryRegisterAutogradFallback(op, kf);

  auto st = make_cpu_storage(reinterpret_cast<void*>(0xABCD), 64);
  TensorImpl a(st, {2}, {1}, 0, ScalarType::Float32, Device::cpu());

  g_fallback_counter = 0;
  // no guard → runs
  {
    BoxedStack s{a};
    D.callBoxed(op, s);
  }
  EXPECT_EQ(g_fallback_counter, 1);

  // guard → skip
  {
    vbt::autograd::SkipAutogradGuard g1;
    BoxedStack s{a};
    D.callBoxed(op, s);
    EXPECT_EQ(g_fallback_counter, 1);
    // nested guard → still skip
    {
      vbt::autograd::SkipAutogradGuard g2;
      BoxedStack s2{a};
      D.callBoxed(op, s2);
      EXPECT_EQ(g_fallback_counter, 1);
    }
    // still under g1 → still skip
    BoxedStack s3{a};
    D.callBoxed(op, s3);
    EXPECT_EQ(g_fallback_counter, 1);
  }

  // after guards pop → runs again
  {
    BoxedStack s{a};
    D.callBoxed(op, s);
    EXPECT_EQ(g_fallback_counter, 2);
  }

#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() > 0) {
    // Exception safety: call mixed-device under guard; after catch, counter not incremented; after guard, counter increments
    const char* two = "vt::two_guard";
    if (!D.has(two)) {
      D.def(std::string(two) + "(Tensor, Tensor) -> Tensor");
      D.registerCpuKernel(two, +[](const TensorImpl& x, const TensorImpl& /*y*/){ return x; });
      D.registerCudaKernel(two, +[](const TensorImpl& x, const TensorImpl& /*y*/){ return x; });
    }
    KernelFunction kf2 = KernelFunction::makeBoxedCtx(/*arity=*/2, &fallback_ctx_count, /*ctx=*/nullptr);
    (void)D.tryRegisterAutogradFallback(two, kf2);

    auto st_cpu = make_cpu_storage(reinterpret_cast<void*>(0xBEEF), 64);
    TensorImpl a_cpu(st_cpu, {2}, {1}, 0, ScalarType::Float32, Device::cpu());
    auto st_cuda = vbt::cuda::new_cuda_storage(2 * vbt::core::itemsize(ScalarType::Float32), 0);
    TensorImpl a_cuda(st_cuda, {2}, {1}, 0, ScalarType::Float32, Device::cuda(0));

    {
      vbt::autograd::SkipAutogradGuard g;
      BoxedStack mixed{a_cpu, a_cuda};
      EXPECT_THROW(D.callBoxed(two, mixed), std::runtime_error);
      EXPECT_LE(g_fallback_counter, 2); // unchanged within guard
    }
    // Now outside guard
    BoxedStack s{a_cpu, a_cpu};
    D.callBoxed(two, s);
    EXPECT_GE(g_fallback_counter, 3);
  }
#endif
}

TEST(SkipAutogradGuardRAII, SafetyNetPreventsReenteringStage1) {
#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
  vbt::dispatch::DispatchV2ModeGuard v2_guard{true};
#endif
  auto& D = Dispatcher::instance();
  const char* op = "vt::id_incorrect";
  if (!D.has(op)) {
    D.registerLibrary("vt");
    D.def(std::string(op) + "(Tensor) -> Tensor");
    D.registerCpuKernel(op, &id_kernel);
  }
  KernelFunction kf = KernelFunction::makeBoxedCtx(/*arity=*/1, &fallback_ctx_incorrect, /*ctx=*/nullptr);
  (void)D.tryRegisterAutogradFallback(op, kf);

  auto st = make_cpu_storage(reinterpret_cast<void*>(0xFEED), 64);
  TensorImpl a(st, {2}, {1}, 0, ScalarType::Float32, Device::cpu());
  g_incorrect_invocations = 0;
  BoxedStack s{a};
  D.callBoxed(op, s);
  // Should complete without recursion; counter increments exactly once
  EXPECT_EQ(g_incorrect_invocations, 1);
}
