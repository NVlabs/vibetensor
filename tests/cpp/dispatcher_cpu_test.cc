// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdexcept>

#include "vbt/dispatch/registration.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/write_guard.h"
#include "vbt/core/intrusive_ptr.h"

using vbt::dispatch::Dispatcher;
using vbt::dispatch::BoxedStack;
using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::check_writable;

static vbt::core::StoragePtr make_storage(void* base, std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(DataPtr(base, [](void*) noexcept {}), nbytes);
}

static TensorImpl vt_id_impl(const TensorImpl& a) { return a; }
static int g_override_count = 0;
static void vt_id_override(BoxedStack& s) {
  ++g_override_count;
  Dispatcher::instance().redispatchBoxed("vt::id", s);
}

TEST(DispatcherCpuTest, BoxedRedispatchOverride) {
  Dispatcher::instance().registerLibrary("vt");
  Dispatcher::instance().def("vt::id(Tensor) -> Tensor");
  Dispatcher::instance().registerCpuKernel("vt::id", &vt_id_impl);
  Dispatcher::instance().registerBoxedOverride("vt::id", &vt_id_override);

  // Call with one arg
  auto storage = make_storage(reinterpret_cast<void*>(0x1111), /*nbytes=*/128);
  TensorImpl a(storage, {2}, {1}, 0, ScalarType::Float32, Device::cpu());
  BoxedStack s{a};
  g_override_count = 0;
  Dispatcher::instance().callBoxed("vt::id", s);
  ASSERT_EQ(s.size(), 1u);
  EXPECT_EQ(g_override_count, 1);
}

TEST(DispatcherCpuTest, FaultyBoxedOverrideRestoresStack) {
  auto& D = Dispatcher::instance();
  const char* opname = "vt::id2";
  if (!D.has(opname)) {
    D.registerLibrary("vt");
    D.def("vt::id2(Tensor) -> Tensor");
    D.registerCpuKernel(opname, &vt_id_impl);
  }
  auto bad = [](BoxedStack& stack) {
    if (!stack.empty()) stack.push_back(stack.back());
  };
  try { D.registerBoxedOverride(opname, bad); } catch (...) {}

  auto storage = make_storage(reinterpret_cast<void*>(0x5555), /*nbytes=*/64);
  TensorImpl a(storage, {1}, {1}, 0, ScalarType::Float32, Device::cpu());
  BoxedStack s{a};
  EXPECT_THROW(D.callBoxed(opname, s), std::runtime_error);
  ASSERT_EQ(s.size(), 1u);
}

// Minimal CPU add kernel (unboxed, arity=2) operating on contiguous tensors of equal shape
static TensorImpl vt_add_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.dtype() != ScalarType::Float32 || b.dtype() != ScalarType::Float32) throw std::invalid_argument("dtype must be float32");
  if (a.device() != Device::cpu() || b.device() != Device::cpu()) throw std::invalid_argument("device must be cpu");
  if (a.sizes() != b.sizes()) throw std::invalid_argument("size mismatch");
  if (!a.is_contiguous() || !b.is_contiguous()) throw std::invalid_argument("contiguous only test kernel");
  auto out = a; // shallow copy of metadata and version counter
  return out;
}

static TensorImpl vt_relu_impl(const TensorImpl& a) {
  if (!a.is_contiguous()) throw std::invalid_argument("contiguous only");
  return a;
}

static TensorImpl vt_unit_impl() { // 0-arg op
  auto storage = make_storage(reinterpret_cast<void*>(0x2222), /*nbytes=*/4);
  return TensorImpl(storage, {}, {}, 0, ScalarType::Float32, Device::cpu());
}

TEST(DispatcherCpuTest, UnboxedBridgingAndArityChecks) {
  auto& D = Dispatcher::instance();
  if (!D.has("vt::unit")) {
    D.registerLibrary("vt");
    D.def("vt::unit() -> Tensor");
    D.registerCpuKernel("vt::unit", &vt_unit_impl);
  }
  if (!D.has("vt::relu")) {
    D.def("vt::relu(Tensor) -> Tensor");
    D.registerCpuKernel("vt::relu", &vt_relu_impl);
  }
  if (!D.has("vt::add")) {
    D.def("vt::add(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel("vt::add", &vt_add_impl);
  }

  auto storage = make_storage(reinterpret_cast<void*>(0x3333), /*nbytes=*/64);
  TensorImpl a(storage, {4}, {1}, 0, ScalarType::Float32, Device::cpu());
  TensorImpl b(storage, {4}, {1}, 4, ScalarType::Float32, Device::cpu());

  // unit(): 0 args
  BoxedStack s0{};
  D.callBoxed("vt::unit", s0);
  ASSERT_EQ(s0.size(), 1u);

  // relu(a): 1 arg
  BoxedStack s1{a};
  D.callBoxed("vt::relu", s1);
  ASSERT_EQ(s1.size(), 1u);

  // add(a,b): 2 args
  BoxedStack s2{a, b};
  D.callBoxed("vt::add", s2);
  ASSERT_EQ(s2.size(), 1u);

  // Arity mismatches keep stack unchanged
  BoxedStack bad{a};
  EXPECT_THROW(D.callBoxed("vt::add", bad), std::invalid_argument);
  ASSERT_EQ(bad.size(), 1u); // unchanged
}

TEST(DispatcherCpuTest, InPlaceGuardIntegration) {
  auto storage = make_storage(reinterpret_cast<void*>(0x4444), /*nbytes=*/64);
  TensorImpl t_neg(storage, {2}, {-1}, 1, ScalarType::Float32, Device::cpu());
  EXPECT_NO_THROW(check_writable(t_neg));
}
