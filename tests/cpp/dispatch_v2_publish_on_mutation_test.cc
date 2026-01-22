// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <string>

#include "vbt/dispatch/dispatcher.h"

using vbt::dispatch::BoxedStack;
using vbt::dispatch::Dispatcher;
using vbt::dispatch::KernelFunction;
using vbt::dispatch::OperatorEntry;
using vbt::dispatch::OperatorHandle;
using vbt::core::TensorImpl;

namespace {

static void boxed_noop(BoxedStack&) {}

static void boxed_ctx_noop(void*, BoxedStack&) {}

static TensorImpl unboxed_id(const TensorImpl& a) { return a; }

struct MutationCase {
  const char* suffix;
  void (*apply)(Dispatcher&, const std::string& fqname);
};

static void apply_register_boxed_override(Dispatcher& D, const std::string& fqname) {
  D.registerBoxedOverride(fqname, &boxed_noop);
}

static void apply_try_register_boxed_override(Dispatcher& D, const std::string& fqname) {
  EXPECT_TRUE(D.tryRegisterBoxedOverride(fqname, &boxed_noop));
}

static void apply_register_cpu_kernelfunction(Dispatcher& D, const std::string& fqname) {
  D.registerCpuKernelFunction(fqname, KernelFunction::makeBoxed(/*arity=*/1, &boxed_noop));
}

static void apply_register_cuda_kernelfunction(Dispatcher& D, const std::string& fqname) {
  D.registerCudaKernelFunction(fqname, KernelFunction::makeBoxed(/*arity=*/1, &boxed_noop));
}

static void apply_replace_cpu_kernelfunction(Dispatcher& D, const std::string& fqname) {
  (void)D.replaceCpuKernelFunction(fqname, KernelFunction::makeBoxed(/*arity=*/1, &boxed_noop));
}

static void apply_replace_cuda_kernelfunction(Dispatcher& D, const std::string& fqname) {
  (void)D.replaceCudaKernelFunction(fqname, KernelFunction::makeBoxed(/*arity=*/1, &boxed_noop));
}

static void apply_uninstall_cpu_kernelfunction(Dispatcher& D, const std::string& fqname) {
  (void)D.uninstallCpuKernelFunction(fqname);
}

static void apply_uninstall_cuda_kernelfunction(Dispatcher& D, const std::string& fqname) {
  (void)D.uninstallCudaKernelFunction(fqname);
}

static void apply_register_autograd_fallback(Dispatcher& D, const std::string& fqname) {
  D.registerAutogradFallback(fqname, KernelFunction::makeBoxedCtx(/*arity=*/1, &boxed_ctx_noop, /*ctx=*/nullptr));
}

static void apply_try_register_autograd_fallback(Dispatcher& D, const std::string& fqname) {
  EXPECT_TRUE(D.tryRegisterAutogradFallback(
      fqname, KernelFunction::makeBoxedCtx(/*arity=*/1, &boxed_ctx_noop, /*ctx=*/nullptr)));
}

static void apply_mark_fabric_op(Dispatcher& D, const std::string& fqname) {
  D.mark_fabric_op(fqname, /*is_fabric_op=*/true, /*allow_multi_device_fabric=*/false);
}

static void apply_register_cpu_kernel_template(Dispatcher& D, const std::string& fqname) {
  D.registerCpuKernel(fqname, &unboxed_id);
}

static void apply_register_cuda_kernel_template(Dispatcher& D, const std::string& fqname) {
  D.registerCudaKernel(fqname, &unboxed_id);
}

}  // namespace

TEST(DispatchV2SnapshotTest, PublishesOnEveryMutationEntryPoint) {
#if !VBT_WITH_DISPATCH_V2
  GTEST_SKIP() << "dispatch v2 disabled";
#else
  auto& D = Dispatcher::instance();
  D.registerLibrary("test_dispatch_v2");

  static constexpr MutationCase cases[] = {
      {"register_boxed_override", &apply_register_boxed_override},
      {"try_register_boxed_override", &apply_try_register_boxed_override},
      {"register_cpu_kernelfunction", &apply_register_cpu_kernelfunction},
      {"register_cuda_kernelfunction", &apply_register_cuda_kernelfunction},
      {"replace_cpu_kernelfunction", &apply_replace_cpu_kernelfunction},
      {"replace_cuda_kernelfunction", &apply_replace_cuda_kernelfunction},
      {"uninstall_cpu_kernelfunction", &apply_uninstall_cpu_kernelfunction},
      {"uninstall_cuda_kernelfunction", &apply_uninstall_cuda_kernelfunction},
      {"register_autograd_fallback", &apply_register_autograd_fallback},
      {"try_register_autograd_fallback", &apply_try_register_autograd_fallback},
      {"mark_fabric_op", &apply_mark_fabric_op},
      {"register_cpu_kernel_template", &apply_register_cpu_kernel_template},
      {"register_cuda_kernel_template", &apply_register_cuda_kernel_template},
  };

  for (const auto& c : cases) {
    const std::string fqname = std::string("test_dispatch_v2::publish_") + c.suffix;
    ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

    OperatorHandle h = D.def(fqname + "(Tensor) -> Tensor");

    const OperatorEntry& entry = h.get();
    const auto* st0 = entry.state_v2.load(std::memory_order_acquire);
    ASSERT_NE(st0, nullptr);

    c.apply(D, fqname);

    const auto* st1 = entry.state_v2.load(std::memory_order_acquire);
    ASSERT_NE(st1, nullptr);
    EXPECT_NE(st1, st0) << "case=" << c.suffix;
    EXPECT_NE(st1->version, st0->version) << "case=" << c.suffix;
  }
#endif
}
