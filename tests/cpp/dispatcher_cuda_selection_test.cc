// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <atomic>
#include <stdexcept>

#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/registration.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"

using vbt::dispatch::Dispatcher;
using vbt::dispatch::BoxedStack;
using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;

static vbt::core::StoragePtr make_cpu_storage(void* base, std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(DataPtr(base, [](void*) noexcept {}), nbytes);
}

static std::atomic<int> g_who_am_i_cpu_calls{0};
static std::atomic<int> g_who_am_i_cuda_calls{0};

static TensorImpl who_am_i_cpu() {
  g_who_am_i_cpu_calls.fetch_add(1, std::memory_order_relaxed);
  auto st = make_cpu_storage(reinterpret_cast<void*>(0x2222), 4);
  return TensorImpl(st, {}, {}, 0, ScalarType::Float32, Device::cpu());
}

// Should never be called: nullary ops are CPU-only in the dispatcher.
static TensorImpl who_am_i_cuda() {
  g_who_am_i_cuda_calls.fetch_add(1, std::memory_order_relaxed);
  auto st = make_cpu_storage(reinterpret_cast<void*>(0x3333), 4);
  return TensorImpl(st, {}, {}, 0, ScalarType::Float32, Device::cpu());
}

// vt::where_am_i(Tensor) -> Tensor
static TensorImpl where_am_i_cpu(const TensorImpl& a) { return a; }
static TensorImpl where_am_i_cuda(const TensorImpl& a) { return a; }

// vt::two(Tensor, Tensor) -> Tensor
static TensorImpl two_impl(const TensorImpl& a, const TensorImpl& /*b*/) { return a; }

TEST(DispatcherCuda, SelectionAndMixedDevice) {
  auto& D = Dispatcher::instance();
  if (!D.has("vt::where_am_i")) {
    D.registerLibrary("vt");
    D.def("vt::where_am_i(Tensor) -> Tensor");
    D.registerCpuKernel("vt::where_am_i", &where_am_i_cpu);
#if VBT_WITH_CUDA
    if (vbt::cuda::device_count() > 0) {
      D.registerCudaKernel("vt::where_am_i", &where_am_i_cuda);
    }
#endif
  }
  if (!D.has("vt::two")) {
    D.def("vt::two(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel("vt::two", &two_impl);
#if VBT_WITH_CUDA
    if (vbt::cuda::device_count() > 0) {
      D.registerCudaKernel("vt::two", &two_impl);
    }
#endif
  }

  // Build a CPU tensor
  auto st_cpu = make_cpu_storage(reinterpret_cast<void*>(0x1234), /*nbytes=*/64);
  TensorImpl a_cpu(st_cpu, {4}, {1}, 0, ScalarType::Float32, Device::cpu());

  // CPU dispatch
  BoxedStack s_cpu{a_cpu};
  D.callBoxed("vt::where_am_i", s_cpu);
  ASSERT_EQ(s_cpu.size(), 1u);
  EXPECT_EQ(s_cpu[0].device(), Device::cpu());

#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  // Build a CUDA tensor
  auto st_cuda = vbt::cuda::new_cuda_storage(4 * vbt::core::itemsize(ScalarType::Float32), 0);
  TensorImpl a_cuda(st_cuda, {4}, {1}, 0, ScalarType::Float32, Device::cuda(0));
  BoxedStack s_cuda{a_cuda};
  D.callBoxed("vt::where_am_i", s_cuda);
  ASSERT_EQ(s_cuda.size(), 1u);
  EXPECT_EQ(s_cuda[0].device(), Device::cuda(0));

  // Mixed-device error substring check
  BoxedStack mixed{a_cpu, a_cuda};
  try {
    D.callBoxed("vt::two", mixed);
    FAIL() << "expected mixed-device error";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("Expected all tensors to be on the same device"), std::string::npos);
    EXPECT_NE(msg.find("cpu:0"), std::string::npos);
    EXPECT_NE(msg.find("cuda:0"), std::string::npos);
  }
#endif
}

TEST(DispatcherCuda, NullaryRemainsCPU) {
#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);
#endif

  g_who_am_i_cpu_calls.store(0, std::memory_order_relaxed);
  g_who_am_i_cuda_calls.store(0, std::memory_order_relaxed);

  auto& D = Dispatcher::instance();
  const char* opname = "vt::who_am_i_nullary_cpu_only_test";
  ASSERT_FALSE(D.has(opname)) << "test op already registered: " << opname;

  D.registerLibrary("vt");
  D.def(std::string(opname) + "() -> Tensor");
  D.registerCpuKernel(opname, &who_am_i_cpu);
#if VBT_WITH_CUDA
  D.registerCudaKernel(opname, &who_am_i_cuda);
#endif

  BoxedStack s{};
  D.callBoxed(opname, s);
  ASSERT_EQ(s.size(), 1u);
  EXPECT_EQ(s[0].device(), Device::cpu());

  EXPECT_EQ(g_who_am_i_cpu_calls.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(g_who_am_i_cuda_calls.load(std::memory_order_relaxed), 0);
}
