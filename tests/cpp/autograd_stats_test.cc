// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "vbt/autograd/stats.h"
#include "vbt/autograd/wrapper.h"
#include "vbt/autograd/engine.h"
#include "vbt/autograd/function.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"

using vbt::autograd::AutogradStatsSnapshot;
using vbt::autograd::stats;
using vbt::autograd::reset_stats;

static vbt::core::TensorImpl make_cpu_dense_f32(const std::vector<int64_t>& sizes, float fill) {
  using namespace vbt::core;
  std::size_t ne = 1; for (auto s : sizes) ne *= static_cast<std::size_t>(s == 0 ? 1 : s);
  std::size_t nbytes = ne * sizeof(float);
  void* buf = nullptr; if (nbytes > 0) buf = ::operator new(nbytes);
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  StoragePtr st = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1; for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) { strides[static_cast<std::size_t>(i)] = acc; acc *= (sizes[static_cast<std::size_t>(i)] == 0 ? 1 : sizes[static_cast<std::size_t>(i)]); }
  TensorImpl t(st, sizes, strides, 0, ScalarType::Float32, Device::cpu());
  float* p = static_cast<float*>(t.data());
  for (std::size_t i = 0; i < ne; ++i) p[i] = fill;
  return t;
}

TEST(AutogradStats, WrapperInvocationAndGuardSkip) {
  reset_stats();
  (void)vbt::autograd::register_autograd_fallbacks();
  // Prepare arguments to call vt::add
  auto a = make_cpu_dense_f32({2,2}, 1.0f);
  auto b = make_cpu_dense_f32({2,2}, 2.0f);
  vbt::dispatch::BoxedStack s{a, b};
  AutogradStatsSnapshot before = stats();
  vbt::dispatch::Dispatcher::instance().callBoxed("vt::add", s);
  AutogradStatsSnapshot mid = stats();
  EXPECT_EQ(mid.wrapper_invocations, before.wrapper_invocations + 1);
  // Guard skip path
  {
    vbt::autograd::SkipAutogradGuard g;
    vbt::dispatch::BoxedStack s2{a, b};
    AutogradStatsSnapshot pre2 = stats();
    vbt::dispatch::Dispatcher::instance().callBoxed("vt::add", s2);
    AutogradStatsSnapshot post2 = stats();
    EXPECT_EQ(post2.wrapper_guard_skips, pre2.wrapper_guard_skips + 1);
  }
}

TEST(AutogradStats, EngineRunCounters) {
  reset_stats();
  using vbt::autograd::OptionalTensor;
  using vbt::autograd::FunctionNode;
  using vbt::autograd::InputMeta;
  using vbt::autograd::ensure_next_edges_sized;
  using vbt::autograd::run_backward;
  using vbt::core::TensorImpl;

  // Single-input FunctionNode that passes grad through
  std::vector<InputMeta> metas = { InputMeta{vbt::core::ScalarType::Float32, vbt::core::Device::cpu(), {2}, true} };
  auto backward = [](std::vector<OptionalTensor>&& gin) {
    std::vector<OptionalTensor> out(1);
    if (!gin.empty()) out[0] = std::move(gin[0]);
    return out;
  };
  auto node = vbt::core::make_intrusive<FunctionNode>("Pass", metas, backward);
  ensure_next_edges_sized(*node);
  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);
  AutogradStatsSnapshot before = stats();
  std::size_t cb_called = 0;
  std::vector<std::function<void()>> cbs = { [&](){ cb_called++; }, [&](){ cb_called++; } };
  run_backward(vbt::core::intrusive_ptr<vbt::autograd::Node>(node.get()), seed, cbs);
  AutogradStatsSnapshot after = stats();
  EXPECT_EQ(after.engine_runs, before.engine_runs + 1);
  EXPECT_GE(after.engine_nodes_processed, before.engine_nodes_processed + 1);
  EXPECT_GE(after.engine_edges_processed, before.engine_edges_processed);
  EXPECT_EQ(cb_called, 2u);
  EXPECT_EQ(after.engine_callbacks_run, before.engine_callbacks_run + 2);
}

TEST(AutogradStats, CountersZeroByDefault) {
  reset_stats();
  AutogradStatsSnapshot s = stats();
  EXPECT_EQ(s.graph_nodes_exposed, 0u);
  EXPECT_EQ(s.graph_edges_exposed, 0u);
  EXPECT_EQ(s.saved_tensors_packed, 0u);
  EXPECT_EQ(s.saved_tensors_unpacked, 0u);
  EXPECT_EQ(s.saved_tensors_hook_violations, 0u);
  EXPECT_EQ(s.multi_grad_hooks_registered, 0u);
  EXPECT_EQ(s.multi_grad_hooks_fired_all, 0u);
  EXPECT_EQ(s.multi_grad_hooks_fired_any, 0u);
}
