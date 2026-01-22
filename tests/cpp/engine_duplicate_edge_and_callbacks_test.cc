// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>

#include <absl/base/config.h>

#if defined(ABSL_HAVE_LEAK_SANITIZER)
extern "C" void __lsan_disable();
extern "C" void __lsan_enable();

struct LsanScope {
  LsanScope() { __lsan_disable(); }
  ~LsanScope() { __lsan_enable(); }
};
#endif

#include "vbt/autograd/engine.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"

using vbt::autograd::Node;
using vbt::autograd::OptionalTensor;
using vbt::autograd::AccumulateGrad;
using vbt::autograd::run_backward;
using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;

namespace {

static TensorImpl make_cpu_dense_f32(const std::vector<int64_t>& sizes, float fill) {
  std::size_t ne = 1;
  for (auto s : sizes) ne *= static_cast<std::size_t>(s == 0 ? 1 : s);
  std::size_t nbytes = ne * sizeof(float);
  void* buf = nullptr; if (nbytes > 0) buf = ::operator new(nbytes);
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  StoragePtr st = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1; for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) { strides[static_cast<std::size_t>(i)] = acc; acc *= (sizes[static_cast<std::size_t>(i)] == 0 ? 1 : sizes[static_cast<std::size_t>(i)]); }
  TensorImpl t(st, sizes, strides, /*offset=*/0, vbt::core::ScalarType::Float32, vbt::core::Device::cpu());
  float* p = static_cast<float*>(t.data());
  for (std::size_t i = 0; i < ne; ++i) p[i] = fill;
  return t;
}

struct FakeAddBackward final : Node {
  uint32_t num_inputs() const noexcept override { return 2; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    // Pass-through: produce two grads routed to two inputs
    std::vector<OptionalTensor> out(2);
    out[0] = std::move(grads_in[0]);
    out[1] = std::move(grads_in[1]);
    return out;
  }
};

} // anonymous

TEST(AutogradEngineEdgeCasesTest, DuplicateEdgeAccumulationAndCallbacks) {
#if defined(ABSL_HAVE_LEAK_SANITIZER)
  LsanScope lsan_scope;
#endif
  // Prepare upstream gradient g (Float32)
  TensorImpl g_impl = make_cpu_dense_f32({4}, 1.0f);

  // AccumulateGrad sink: use a dummy leaf's meta.grad
  TensorImpl leaf = make_cpu_dense_f32({4}, 0.0f);
  auto* meta = vbt::autograd::get_autograd_meta(leaf, /*create_if_missing=*/true);
  auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);

  // Root node: FakeAddBackward
  auto addb = vbt::core::make_intrusive<FakeAddBackward>();
  // Two edges from addb to the same AccumulateGrad slot 0
  addb->next_edges.push_back(vbt::autograd::Edge{vbt::core::intrusive_ptr<Node>(acc.get()), 0});
  addb->next_edges.push_back(vbt::autograd::Edge{vbt::core::intrusive_ptr<Node>(acc.get()), 0});

  // Seed initial grads (two identical grads)
  std::vector<OptionalTensor> seed(2);
  seed[0] = g_impl;
  seed[1] = g_impl;

  // Callbacks order tracking
  std::vector<int> cb_order;
  auto cb1 = [&]() { cb_order.push_back(1); };
  auto cb2 = [&]() { cb_order.push_back(2); };
  auto cb3 = [&]() { cb_order.push_back(3); };

  run_backward(vbt::core::intrusive_ptr<Node>(addb.get()), seed, {cb1, cb2, cb3});

  ASSERT_TRUE(meta->grad_ptr != nullptr && meta->grad_has);
  const auto& grad = *(meta->grad_ptr);
  const float* pdata = static_cast<const float*>(grad.data());
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(pdata[i], 2.0f) << "index " << i;
  }
  ASSERT_EQ(cb_order.size(), 3u);
  EXPECT_EQ(cb_order[0], 1);
  EXPECT_EQ(cb_order[1], 2);
  EXPECT_EQ(cb_order[2], 3);
}

TEST(AutogradEngineEdgeCasesTest, CallbackExceptionStopsSequence) {
#if defined(ABSL_HAVE_LEAK_SANITIZER)
  LsanScope lsan_scope;
#endif
  TensorImpl g_impl = make_cpu_dense_f32({2}, 1.0f);
  vbt::autograd::OptionalTensor sink; (void)sink; // not used
  auto* meta = vbt::autograd::get_autograd_meta(g_impl, /*create_if_missing=*/true);
  auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);
  auto addb = vbt::core::make_intrusive<FakeAddBackward>();
  addb->next_edges.push_back(vbt::autograd::Edge{vbt::core::intrusive_ptr<Node>(acc.get()), 0});
  addb->next_edges.push_back(vbt::autograd::Edge{vbt::core::intrusive_ptr<Node>(acc.get()), 0});
  std::vector<OptionalTensor> seed(2);
  seed[0] = g_impl; seed[1] = g_impl;

  int ran = 0;
  auto cb1 = [&]() { ran = ran * 10 + 1; };
  auto cb2 = [&]() { ran = ran * 10 + 2; throw std::runtime_error("boom"); };
  auto cb3 = [&]() { ran = ran * 10 + 3; };

  EXPECT_THROW(run_backward(vbt::core::intrusive_ptr<Node>(addb.get()), seed, {cb1, cb2, cb3}), std::runtime_error);
  EXPECT_EQ(ran, 12);
}
