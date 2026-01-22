// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <string>

#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/kernel_function.h"
#include "vbt/dispatch/plugin_loader.h"

using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::TensorImpl;
using vbt::dispatch::BoxedStack;
using vbt::dispatch::Dispatcher;
using vbt::dispatch::KernelFunction;

namespace {

static TensorImpl make_cpu_scalar_i64(std::int64_t v) {
  void* buf = ::operator new(sizeof(std::int64_t));
  *static_cast<std::int64_t*>(buf) = v;
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto st = vbt::core::make_intrusive<Storage>(std::move(dp), sizeof(std::int64_t));
  return TensorImpl(st, /*sizes=*/{}, /*strides=*/{}, /*storage_offset=*/0,
                    ScalarType::Int64, Device::cpu());
}

#if VBT_WITH_CUDA
static TensorImpl make_cuda_tensor_f32(int dev) {
  auto st = vbt::cuda::new_cuda_storage(4 * vbt::core::itemsize(ScalarType::Float32), dev);
  return TensorImpl(st, /*sizes=*/{4}, /*strides=*/{1}, /*storage_offset=*/0,
                    ScalarType::Float32, Device::cuda(dev));
}
#endif

static void boxed_echo_first(BoxedStack& s) {
  TensorImpl out = s[0];
  s.clear();
  s.push_back(out);
}

}  // namespace

TEST(PluginDevicePolicyAndStreamTest, SetDevicePolicyHostThunkValidation) {
#if !VBT_WITH_DISPATCH_V2
  GTEST_SKIP() << "dispatch v2 disabled";
#endif
#ifndef PLUGIN_SET_DEVICE_POLICY_VALIDATE_PATH
  GTEST_SKIP() << "No set_device_policy_validate plugin path provided";
#else
  const char* so = PLUGIN_SET_DEVICE_POLICY_VALIDATE_PATH;
  vt_status st = vbt::dispatch::plugin::load_library(so);
  ASSERT_EQ(st, VT_STATUS_OK) << "status=" << static_cast<int>(st)
                              << " err=" << vbt::dispatch::plugin::get_last_error();
#endif
}

TEST(PluginDevicePolicyAndStreamTest, FabricPluginTrampolineUsesComputeDeviceStream) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA not built";
#else
#if !VBT_WITH_DISPATCH_V2
  GTEST_SKIP() << "dispatch v2 disabled";
#endif
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need >=2 CUDA devices";
  }
#ifndef PLUGIN_FABRIC_STREAM_CHECK_PATH
  GTEST_SKIP() << "No fabric_stream_check plugin path provided";
#else
  auto& D = Dispatcher::instance();
  const char* op = "fabric_testlib::plugin_stream_check";

  if (!D.has(op)) {
    D.registerLibrary("fabric_testlib");
    D.def("fabric_testlib::plugin_stream_check(Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");
    D.registerCudaKernelFunction(op,
                                 KernelFunction::makeBoxed(/*arity=*/5, &boxed_echo_first));
  }

  // Ensure the plugin registers while Fabric bypass is disabled.
  D.mark_fabric_op(op, /*is_fabric_op=*/true, /*allow_multi_device_fabric=*/false);

  // Load the plugin that replaces the CUDA kernel for this op.
  const char* so = PLUGIN_FABRIC_STREAM_CHECK_PATH;
  vt_status st = vbt::dispatch::plugin::load_library(so);
  ASSERT_EQ(st, VT_STATUS_OK) << "status=" << static_cast<int>(st)
                              << " err=" << vbt::dispatch::plugin::get_last_error();

  // Late-enable Fabric bypass after plugin registration; the trampoline must
  // consult the current op metadata, not a registration-time snapshot.
  D.mark_fabric_op(op, /*is_fabric_op=*/true, /*allow_multi_device_fabric=*/true);

  // Force a non-default stream on device 1 so that passing the stream from
  // device 0 would be observable (default streams are all 0). Persist the
  // selection in TLS so it remains visible when the current device differs.
  vbt::cuda::Stream prev1 = vbt::cuda::getCurrentStream(/*device=*/1);
  vbt::cuda::Stream s1 = vbt::cuda::getStreamFromPool(/*high_priority=*/false,
                                                      /*device=*/1);
  ASSERT_NE(s1.handle(), 0u) << "expected non-default stream";
  vbt::cuda::setCurrentStream(s1);
  struct RestoreStream {
    vbt::cuda::Stream prev;
    ~RestoreStream() { vbt::cuda::setCurrentStream(prev); }
  } restore{prev1};

  // Ensure the call-time current device differs from compute_device.
  vbt::cuda::DeviceGuard dg0(static_cast<vbt::cuda::DeviceIndex>(0));

  TensorImpl a0 = make_cuda_tensor_f32(/*dev=*/0);
  TensorImpl b1 = make_cuda_tensor_f32(/*dev=*/1);

  TensorImpl compute1 = make_cpu_scalar_i64(1);
  TensorImpl require1 = make_cpu_scalar_i64(1);
  TensorImpl fallback1 = make_cpu_scalar_i64(1);

  // Call the CUDA base kernel directly to avoid depending on dispatcher Fabric
  // pinning semantics.
  auto h = D.find(op);
  auto& entry = h.get();
  ASSERT_TRUE(entry.cuda_base.has_value());
  ASSERT_EQ(entry.cuda_base->mode, KernelFunction::Mode::BoxedWithCtx);
  ASSERT_NE(entry.cuda_base->boxed_ctx, nullptr);

  BoxedStack stack{a0, b1, compute1, require1, fallback1};
  entry.cuda_base->callBoxed(op, stack);

  ASSERT_EQ(stack.size(), 1u);
  EXPECT_EQ(stack[0].device(), Device::cuda(1));

  // Deterministic arity contract: bypass KernelFunction's arity checks and call
  // the trampoline directly with the wrong number of args.
  BoxedStack bad_stack{a0, b1, compute1, require1};
  try {
    entry.cuda_base->boxed_ctx(entry.cuda_base->ctx, bad_stack);
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("[Fabric] plugin trampoline: expected nargs==5"),
              std::string::npos)
        << msg;
  }
#endif
#endif
}
