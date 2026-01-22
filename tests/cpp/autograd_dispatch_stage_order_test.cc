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

// Base kernels
static int g_base_count = 0;
static TensorImpl vt_id_impl(const TensorImpl& a) { ++g_base_count; return a; }
static TensorImpl vt_two_impl(const TensorImpl& a, const TensorImpl& /*b*/) { ++g_base_count; return a; }

// Boxed override that redispatches to base
static int g_override_count = 0;
static void vt_id_override(BoxedStack& s) {
  ++g_override_count;
  Dispatcher::instance().redispatchBoxedCurrent(s);
}

// Autograd fallback (ctx=&OperatorEntry) that redispatches to base under SkipAutogradGuard
static int g_fallback_count = 0;
static void vt_autograd_fallback_ctx(void* ctx, BoxedStack& s) {
  ++g_fallback_count;
  auto* entry = static_cast<vbt::dispatch::OperatorEntry*>(ctx);
  OperatorHandle op(entry);
  vbt::autograd::SkipAutogradGuard g;
  Dispatcher::instance().redispatchBoxed(op, s);
}

TEST(AutogradDispatchStageOrder, Stage1BeforeStage2AndStage3) {
#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
  vbt::dispatch::DispatchV2ModeGuard v2_guard{true};
#endif
  auto& D = Dispatcher::instance();
  // Setup vt::id(Tensor)->Tensor
  if (!D.has("vt::id")) {
    D.registerLibrary("vt");
    D.def("vt::id(Tensor) -> Tensor");
    D.registerCpuKernel("vt::id", &vt_id_impl);
  }
  // Register boxed override and autograd fallback
  D.tryRegisterBoxedOverride("vt::id", &vt_id_override);
  KernelFunction kf = KernelFunction::makeBoxedCtx(/*arity=*/1, &vt_autograd_fallback_ctx, /*ctx=*/nullptr);
  try { D.registerAutogradFallback("vt::id", kf); } catch (...) {}

  // Build a CPU tensor
  auto st = make_cpu_storage(reinterpret_cast<void*>(0x1111), 64);
  TensorImpl a(st, {4}, {1}, 0, ScalarType::Float32, Device::cpu());

  // Case: fallback:on, override:on, skip=false => fallback runs, override not, base runs
  g_base_count = g_override_count = g_fallback_count = 0;
  BoxedStack s{a};
  D.callBoxed("vt::id", s);
  ASSERT_EQ(s.size(), 1u);
  EXPECT_EQ(g_fallback_count, 1);
  EXPECT_EQ(g_override_count, 0);
  EXPECT_EQ(g_base_count, 1);

  // Case: with SkipAutogradGuard => Stage 1 skipped; Stage 2 override executes then base
  g_base_count = g_override_count = g_fallback_count = 0;
  {
    vbt::autograd::SkipAutogradGuard guard;
    BoxedStack s2{a};
    D.callBoxed("vt::id", s2);
  }
  EXPECT_EQ(g_fallback_count, 0);
  EXPECT_EQ(g_override_count, 1);
  EXPECT_EQ(g_base_count, 1);
}

TEST(AutogradDispatchStageOrder, FallbackAbsentUsesOverrideThenBase) {
#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
  vbt::dispatch::DispatchV2ModeGuard v2_guard{true};
#endif
  auto& D = Dispatcher::instance();
  const char* op = "vt::id2";
  if (!D.has(op)) {
    D.registerLibrary("vt");
    D.def(std::string(op) + "(Tensor) -> Tensor");
    D.registerCpuKernel(op, &vt_id_impl);
  }
  // Only boxed override, no fallback
  D.tryRegisterBoxedOverride(op, &vt_id_override);

  auto st = make_cpu_storage(reinterpret_cast<void*>(0x2222), 64);
  TensorImpl a(st, {2}, {1}, 0, ScalarType::Float32, Device::cpu());
  g_base_count = g_override_count = g_fallback_count = 0;
  BoxedStack s{a};
  D.callBoxed(op, s);
  ASSERT_EQ(s.size(), 1u);
  EXPECT_EQ(g_fallback_count, 0);
  EXPECT_EQ(g_override_count, 1);
  EXPECT_EQ(g_base_count, 1);
}

TEST(AutogradDispatchStageOrder, NullaryFallbackRegistrationRejected) {
#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
  vbt::dispatch::DispatchV2ModeGuard v2_guard{true};
#endif
  auto& D = Dispatcher::instance();
  const char* op = "vt::unit";
  if (!D.has(op)) {
    D.registerLibrary("vt");
    D.def(std::string(op) + "() -> Tensor");
    auto unit = []() -> TensorImpl {
      auto st = make_cpu_storage(reinterpret_cast<void*>(0x3333), 4);
      return TensorImpl(st, {}, {}, 0, ScalarType::Float32, Device::cpu());
    };
    D.registerCpuKernel(op, +unit);
  }
  KernelFunction kf = KernelFunction::makeBoxedCtx(/*arity=*/0, &vt_autograd_fallback_ctx, /*ctx=*/nullptr);
  EXPECT_THROW(D.registerAutogradFallback(op, kf), std::runtime_error);
}

TEST(AutogradDispatchStageOrder, ArityMismatchMessageParityViaFallback) {
#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
  vbt::dispatch::DispatchV2ModeGuard v2_guard{true};
#endif
  auto& D = Dispatcher::instance();
  // Ensure vt::id has fallback registered from previous test; if not, register once
  if (D.has("vt::id")) {
    KernelFunction kf = KernelFunction::makeBoxedCtx(/*arity=*/1, &vt_autograd_fallback_ctx, /*ctx=*/nullptr);
    (void)D.tryRegisterAutogradFallback("vt::id", kf);
  }
  BoxedStack empty{};
  try {
    D.callBoxed("vt::id", empty);
    FAIL() << "expected arity mismatch";
  } catch (const std::invalid_argument& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("arity mismatch vt::id: expected 1, got 0"), std::string::npos);
  }
}

TEST(AutogradDispatchStageOrder, MixedDeviceErrorParityThroughFallback) {
#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
  vbt::dispatch::DispatchV2ModeGuard v2_guard{true};
#endif
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  auto& D = Dispatcher::instance();
  const char* op = "vt::two";
  if (!D.has(op)) {
    D.registerLibrary("vt");
    D.def(std::string(op) + "(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel(op, &vt_two_impl);
#if VBT_WITH_CUDA
    D.registerCudaKernel(op, &vt_two_impl);
#endif
  }
  // Register autograd fallback for vt::two
  KernelFunction kf = KernelFunction::makeBoxedCtx(/*arity=*/2, &vt_autograd_fallback_ctx, /*ctx=*/nullptr);
  (void)D.tryRegisterAutogradFallback(op, kf);

  auto st_cpu = make_cpu_storage(reinterpret_cast<void*>(0x4444), 64);
  TensorImpl a_cpu(st_cpu, {4}, {1}, 0, ScalarType::Float32, Device::cpu());
  auto st_cuda = vbt::cuda::new_cuda_storage(4 * vbt::core::itemsize(ScalarType::Float32), 0);
  TensorImpl a_cuda(st_cuda, {4}, {1}, 0, ScalarType::Float32, Device::cuda(0));

  BoxedStack mixed{a_cpu, a_cuda};
  try {
    D.callBoxed(op, mixed);
    FAIL() << "expected mixed-device error";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("Expected all tensors to be on the same device"), std::string::npos);
    EXPECT_NE(msg.find("cpu:0"), std::string::npos);
    EXPECT_NE(msg.find("cuda:0"), std::string::npos);
  }
#else
  GTEST_SKIP() << "CUDA disabled";
#endif
}

TEST(DispatchV2RedispatchTest, RedispatchBoxedUsesStateV2InV2Mode) {
#if !(VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "dispatch v2/internal tests disabled";
#else
  vbt::dispatch::DispatchV2ModeGuard v2_guard{true};

  auto& D = Dispatcher::instance();
  static int uniq = 0;
  const std::string op =
      std::string("test_dispatch_v2::redispatch_boxed_uses_state_v2_") +
      std::to_string(uniq++);
  ASSERT_FALSE(D.has(op)) << "test op already registered: " << op;

  D.registerLibrary("test_dispatch_v2");
  OperatorHandle h = D.def(op + "(Tensor) -> Tensor");
  D.registerCpuKernel(op, &vt_id_impl);

  // Corrupt v1 storage after snapshot publication; redispatch in v2 must use state_v2.
  auto& entry = h.get();
  auto saved_cpu_base = entry.cpu_base;
  entry.cpu_base.reset();

  auto st = make_cpu_storage(reinterpret_cast<void*>(0x5555), 64);
  TensorImpl a(st, {4}, {1}, 0, ScalarType::Float32, Device::cpu());

  g_base_count = 0;
  BoxedStack s{a};
  D.redispatchBoxed(h, s);
  ASSERT_EQ(s.size(), 1u);
  EXPECT_EQ(g_base_count, 1);

  entry.cpu_base = saved_cpu_base;
#endif
}

TEST(DispatchV2RedispatchTest, RedispatchBoxedCurrentUsesStateV2InV2Mode) {
#if !(VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "dispatch v2/internal tests disabled";
#else
  vbt::dispatch::DispatchV2ModeGuard v2_guard{true};

  auto& D = Dispatcher::instance();
  static int uniq = 0;
  const std::string op =
      std::string("test_dispatch_v2::redispatch_boxed_current_uses_state_v2_") +
      std::to_string(uniq++);
  ASSERT_FALSE(D.has(op)) << "test op already registered: " << op;

  D.registerLibrary("test_dispatch_v2");
  OperatorHandle h = D.def(op + "(Tensor) -> Tensor");
  D.registerCpuKernel(op, &vt_id_impl);
  D.registerBoxedOverride(op, &vt_id_override);

  // Corrupt v1 storage after snapshot publication; redispatchBoxedCurrent in v2 must use state_v2.
  auto& entry = h.get();
  auto saved_cpu_base = entry.cpu_base;
  entry.cpu_base.reset();

  auto st = make_cpu_storage(reinterpret_cast<void*>(0x6666), 64);
  TensorImpl a(st, {4}, {1}, 0, ScalarType::Float32, Device::cpu());

  g_base_count = g_override_count = g_fallback_count = 0;
  BoxedStack s{a};
  D.callBoxed(op, s);
  ASSERT_EQ(s.size(), 1u);
  EXPECT_EQ(g_override_count, 1);
  EXPECT_EQ(g_base_count, 1);

  entry.cpu_base = saved_cpu_base;
#endif
}

TEST(DispatchV2RedispatchTest, RedispatchBoxedCurrentIsStrictEvenForAllowMixedOps) {
#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
  vbt::dispatch::DispatchV2ModeGuard v2_guard{true};
#endif
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  auto& D = Dispatcher::instance();
  const char* op = "vt::check_stream";
  if (!D.has(op)) {
    D.registerLibrary("vt");
    D.def(std::string(op) + "(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel(op, &vt_two_impl);
    D.registerCudaKernel(op, &vt_two_impl);
  }

  auto st_tpl = make_cpu_storage(reinterpret_cast<void*>(0x7777), sizeof(std::int64_t));
  TensorImpl tpl_cpu(st_tpl, /*sizes=*/{}, /*strides=*/{}, 0, ScalarType::Int64, Device::cpu());
  auto st_cuda = vbt::cuda::new_cuda_storage(
      4 * vbt::core::itemsize(ScalarType::Float32), 0);
  TensorImpl a_cuda(st_cuda, {4}, {1}, 0, ScalarType::Float32, Device::cuda(0));

  // Base dispatch allows mixed devices for allow-mixed ops.
  g_base_count = 0;
  BoxedStack mixed{a_cuda, tpl_cpu};
  D.redispatchBoxed(op, mixed);
  ASSERT_EQ(mixed.size(), 1u);
  EXPECT_EQ(g_base_count, 1);

  // But redispatchBoxedCurrent enforces strict same-device validation.
  (void)D.tryRegisterBoxedOverride(op, &vt_id_override);
  BoxedStack mixed2{a_cuda, tpl_cpu};
  try {
    D.callBoxed(op, mixed2);
    FAIL() << "expected mixed-device error via redispatchBoxedCurrent";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("Expected all tensors to be on the same device"), std::string::npos);
    EXPECT_NE(msg.find("cpu:0"), std::string::npos);
    EXPECT_NE(msg.find("cuda:0"), std::string::npos);
  }
#else
  GTEST_SKIP() << "CUDA disabled";
#endif
}
