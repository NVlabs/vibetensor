// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "vbt/autograd/meta.h"
#include "vbt/autograd/wrapper.h"
#include "vbt/autograd/node.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/autograd/engine.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"
#include "vbt/core/device.h"

using vbt::autograd::AutogradMeta;
using vbt::autograd::Node;
using vbt::autograd::OptionalTensor;
using vbt::autograd::AccumulateGrad;
using vbt::autograd::get_grad_fn;
using vbt::autograd::gradient_root;
using vbt::autograd::register_leaf_hook;
using vbt::autograd::get_leaf_hooks;
using vbt::autograd::clear_tensor_grad;
using vbt::autograd::detach_copy;
using vbt::autograd::detach_inplace;
using vbt::autograd::is_leaf;
using vbt::autograd::is_view;
using vbt::autograd::requires_grad;
using vbt::autograd::set_requires_grad;
using vbt::autograd::get_autograd_meta;
using vbt::autograd::TensorHook;
using vbt::autograd::run_backward;
using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;

namespace {

static TensorImpl make_cpu_dense_f32(const std::vector<int64_t>& sizes, float fill) {
  std::size_t ne = 1;
  for (auto s : sizes) ne *= static_cast<std::size_t>(s == 0 ? 1 : s);
  const std::size_t nbytes = ne * sizeof(float);
  void* buf = nullptr;
  if (nbytes > 0) buf = ::operator new(nbytes);
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  StoragePtr st = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] = acc;
    acc *= (sizes[static_cast<std::size_t>(i)] == 0 ? 1 : sizes[static_cast<std::size_t>(i)]);
  }
  TensorImpl t(st, sizes, strides, 0, vbt::core::ScalarType::Float32, vbt::core::Device::cpu());
  float* p = static_cast<float*>(t.data());
  for (std::size_t i = 0; i < ne; ++i) p[i] = fill;
  return t;
}

struct DummyNode final : Node {
  uint32_t num_inputs() const noexcept override { return 1; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    return std::move(grads_in);
  }
};

struct PassThrough1 final : Node {
  uint32_t num_inputs() const noexcept override { return 1; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    return std::move(grads_in);
  }
};

struct RecordingHook final : TensorHook {
  std::vector<TensorImpl> seen;
  bool throw_on_call{false};

  void call(const TensorImpl& grad) override {
    seen.push_back(grad);
    if (throw_on_call) {
      throw std::runtime_error("hook boom");
    }
  }
};

} // anonymous namespace

TEST(AutogradTensorSurfaceTest, GetGradFnHelper) {
  // Tensor without AutogradMeta
  TensorImpl t = make_cpu_dense_f32({2}, 1.0f);
  auto fn0 = get_grad_fn(t);
  EXPECT_FALSE(static_cast<bool>(fn0));

  // Tensor with AutogradMeta but no grad_fn
  AutogradMeta* m = get_autograd_meta(t, /*create_if_missing=*/true);
  ASSERT_NE(m, nullptr);
  m->grad_fn.reset();
  auto fn1 = get_grad_fn(t);
  EXPECT_FALSE(static_cast<bool>(fn1));

  // Tensor with grad_fn installed
  auto ptr = vbt::core::intrusive_ptr<Node>(new DummyNode(), /*add_ref=*/true);
  m->grad_fn = ptr;
  auto fn2 = get_grad_fn(t);
  EXPECT_TRUE(static_cast<bool>(fn2));
  EXPECT_EQ(fn2.get(), ptr.get());
}

TEST(AutogradTensorSurfaceTest, GradientRootResolvesViewsToBase) {
  TensorImpl base = make_cpu_dense_f32({2}, 1.0f);
  // Mark base as requiring grad and a leaf
  set_requires_grad(base, true);
  EXPECT_TRUE(requires_grad(base));
  EXPECT_TRUE(is_leaf(base));
  EXPECT_FALSE(is_view(base));

  // Create a view and wire view metadata
  std::vector<int64_t> sizes = base.sizes();
  std::vector<int64_t> strides = base.strides();
  TensorImpl view = base.as_strided(sizes, strides, base.storage_offset());
  vbt::autograd::as_view(base, view);
  EXPECT_TRUE(is_view(view));

  const TensorImpl* r_base = gradient_root(base);
  const TensorImpl* r_view = gradient_root(view);
  ASSERT_NE(r_base, nullptr);
  ASSERT_NE(r_view, nullptr);
  EXPECT_EQ(r_base, &base);
  // gradient_root(view) may return an internal snapshot handle; validate it
  // resolves to the same AutogradMeta as the base leaf.
  EXPECT_EQ(get_autograd_meta(*r_view), get_autograd_meta(base));
}

TEST(AutogradTensorSurfaceTest, AsViewOnSharedMetaIsNoop) {
  TensorImpl base = make_cpu_dense_f32({2}, 1.0f);
  set_requires_grad(base, true);

  // TensorImpl is a shallow value type; copies share AutogradMeta.
  TensorImpl out = base;
  vbt::autograd::as_view(base, out);

  EXPECT_FALSE(is_view(base));
  EXPECT_FALSE(is_view(out));

  const TensorImpl* r = gradient_root(out);
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(r, &out);
  EXPECT_EQ(get_autograd_meta(*r), get_autograd_meta(base));
}

TEST(AutogradTensorSurfaceTest, GradientRootNonDiffViewReturnsSelf) {
  TensorImpl base = make_cpu_dense_f32({2}, 1.0f);
  EXPECT_FALSE(requires_grad(base));

  // Create a real view and mark it as a (non-differentiable) view.
  TensorImpl view =
      base.as_strided(base.sizes(), base.strides(), base.storage_offset());
  vbt::autograd::as_view(base, view);

  EXPECT_TRUE(is_view(view));
  EXPECT_FALSE(requires_grad(view));

  const TensorImpl* r = gradient_root(view);
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(r, &view);

  AutogradMeta* mv = get_autograd_meta(view, /*create_if_missing=*/false);
  ASSERT_NE(mv, nullptr);
  EXPECT_FALSE(mv->view.base_root);
}

TEST(AutogradTensorSurfaceTest, AsViewIsDifferentiableOnlyWhenGraphEnabled) {
  struct ModeRestoreGuard {
    bool prev_grad;
    bool prev_inf;
    ModeRestoreGuard() noexcept
        : prev_grad(vbt::autograd::GradMode::is_enabled()),
          prev_inf(vbt::autograd::InferenceMode::is_enabled()) {}
    ~ModeRestoreGuard() noexcept {
      vbt::autograd::GradMode::set_enabled(prev_grad);
      vbt::autograd::InferenceMode::set_enabled(prev_inf);
    }
  } restore;

  auto run_case = [&](bool grad_enabled, bool inf_enabled, bool expect_diff) {
    vbt::autograd::GradMode::set_enabled(grad_enabled);
    vbt::autograd::InferenceMode::set_enabled(inf_enabled);

    TensorImpl base = make_cpu_dense_f32({2}, 1.0f);
    set_requires_grad(base, true);

    TensorImpl view = base.as_strided(std::vector<int64_t>{1},
                                      std::vector<int64_t>{1},
                                      base.storage_offset() + 1);
    vbt::autograd::as_view(base, view);

    EXPECT_TRUE(is_view(view));
    EXPECT_EQ(requires_grad(view), expect_diff);
    EXPECT_EQ(is_leaf(view), !expect_diff);

    AutogradMeta* mv = get_autograd_meta(view, /*create_if_missing=*/false);
    ASSERT_NE(mv, nullptr);
    EXPECT_EQ(static_cast<bool>(mv->view.base_root), expect_diff);

    const TensorImpl* r = gradient_root(view);
    ASSERT_NE(r, nullptr);
    if (expect_diff) {
      EXPECT_NE(r, &view);
      EXPECT_EQ(get_autograd_meta(*r), get_autograd_meta(base));
    } else {
      EXPECT_EQ(r, &view);
    }
  };

  run_case(/*grad_enabled=*/true, /*inf_enabled=*/false, /*expect_diff=*/true);
  run_case(/*grad_enabled=*/false, /*inf_enabled=*/false, /*expect_diff=*/false);
  run_case(/*grad_enabled=*/true, /*inf_enabled=*/true, /*expect_diff=*/false);
  run_case(/*grad_enabled=*/false, /*inf_enabled=*/true, /*expect_diff=*/false);
}

TEST(AutogradTensorSurfaceTest, AsViewClearsStaleHistoryOnOut) {
  TensorImpl base = make_cpu_dense_f32({2}, 1.0f);  // non-grad
  TensorImpl out = make_cpu_dense_f32({2}, 2.0f);

  // Install bogus history on out.
  AutogradMeta* m = get_autograd_meta(out, /*create_if_missing=*/true);
  ASSERT_NE(m, nullptr);
  set_requires_grad(out, true);
  m->is_leaf = false;
  m->output_nr = 3;
  m->grad_fn = vbt::core::intrusive_ptr<Node>(new DummyNode(), /*add_ref=*/true);

  ASSERT_TRUE(requires_grad(out));
  ASSERT_TRUE(static_cast<bool>(get_grad_fn(out)));

  vbt::autograd::as_view(base, out);

  EXPECT_TRUE(is_view(out));
  EXPECT_FALSE(requires_grad(out));
  EXPECT_TRUE(is_leaf(out));
  EXPECT_FALSE(static_cast<bool>(get_grad_fn(out)));
  EXPECT_EQ(m->output_nr, 0);
  EXPECT_FALSE(m->view.base_root);
}

TEST(AutogradTensorSurfaceTest, DetachCopyClearsAutogradMetaAndSharesStorage) {
  TensorImpl t = make_cpu_dense_f32({2}, 3.0f);
  AutogradMeta* m = get_autograd_meta(t, /*create_if_missing=*/true);
  ASSERT_NE(m, nullptr);
  set_requires_grad(t, true);
  EXPECT_TRUE(requires_grad(t));

  TensorImpl detached = detach_copy(t);

  // Shallow copy: same storage and version counter
  EXPECT_NE(&detached, &t);
  EXPECT_EQ(detached.storage().get(), t.storage().get());
  EXPECT_EQ(detached.sizes(), t.sizes());
  EXPECT_EQ(detached.strides(), t.strides());
  EXPECT_EQ(detached.storage_offset(), t.storage_offset());
  EXPECT_EQ(detached.dtype(), t.dtype());
  EXPECT_EQ(detached.device(), t.device());
  EXPECT_EQ(detached.version(), t.version());

  // Autograd metadata cleared
  EXPECT_FALSE(requires_grad(detached));
  EXPECT_TRUE(is_leaf(detached));
  EXPECT_FALSE(is_view(detached));
  EXPECT_FALSE(static_cast<bool>(get_grad_fn(detached)));
}

TEST(AutogradTensorSurfaceTest, DetachInplaceClearsHistoryButKeepsGradBuffer) {
  TensorImpl t = make_cpu_dense_f32({2}, 2.0f);
  AutogradMeta* m = get_autograd_meta(t, /*create_if_missing=*/true);
  ASSERT_NE(m, nullptr);
  // Pretend t is a non-leaf with history and a gradient buffer
  set_requires_grad(t, true);
  m->is_leaf = false;
  m->output_nr = 5;
  m->view.is_view = true;
  m->view.base_root = std::make_shared<TensorImpl>(t);  // arbitrary non-null handle
  auto grad_buf = make_cpu_dense_f32({2}, 4.0f);
  m->grad_ptr = std::make_unique<TensorImpl>(grad_buf);
  m->grad_has = true;

  detach_inplace(t);

  EXPECT_FALSE(requires_grad(t));
  EXPECT_TRUE(is_leaf(t));
  EXPECT_FALSE(is_view(t));
  EXPECT_FALSE(m->view.base_root);
  EXPECT_FALSE(static_cast<bool>(get_grad_fn(t)));
  EXPECT_EQ(m->output_nr, 0);
  // Gradient buffer is preserved for inspection
  ASSERT_TRUE(m->grad_ptr != nullptr && m->grad_has);
  const float* pdata = static_cast<const float*>(m->grad_ptr->data());
  EXPECT_FLOAT_EQ(pdata[0], 4.0f);
  EXPECT_FLOAT_EQ(pdata[1], 4.0f);
}

TEST(AutogradTensorSurfaceTest, ClearTensorGradResetsOnlyGradFields) {
  TensorImpl t = make_cpu_dense_f32({2}, 1.0f);
  AutogradMeta* m = get_autograd_meta(t, /*create_if_missing=*/true);
  ASSERT_NE(m, nullptr);
  set_requires_grad(t, true);
  m->is_leaf = true;
  m->view.is_view = false;
  auto grad_buf = make_cpu_dense_f32({2}, 7.0f);
  m->grad_ptr = std::make_unique<TensorImpl>(grad_buf);
  m->grad_has = true;

  clear_tensor_grad(t);

  EXPECT_TRUE(requires_grad(t));  // unchanged
  EXPECT_TRUE(is_leaf(t));
  EXPECT_FALSE(is_view(t));
  EXPECT_EQ(m->grad_ptr, nullptr);
  EXPECT_FALSE(m->grad_has);
}

TEST(AutogradTensorSurfaceTest, AccumulateGradNoHooksUpdatesLeafGrad) {
  TensorImpl g = make_cpu_dense_f32({4}, 1.0f);
  TensorImpl leaf = make_cpu_dense_f32({4}, 0.0f);
  AutogradMeta* meta = get_autograd_meta(leaf, /*create_if_missing=*/true);
  ASSERT_NE(meta, nullptr);

  auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);
  auto root = vbt::core::make_intrusive<PassThrough1>();
  root->next_edges.push_back(vbt::autograd::Edge{vbt::core::intrusive_ptr<Node>(acc.get()), 0});

  std::vector<OptionalTensor> seed(1);
  seed[0] = g;

  run_backward(vbt::core::intrusive_ptr<Node>(root.get()), seed, {});

  ASSERT_TRUE(meta->grad_ptr != nullptr && meta->grad_has);
  const TensorImpl& grad = *meta->grad_ptr;
  const float* pdata = static_cast<const float*>(grad.data());
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(pdata[i], 1.0f) << "index " << i;
  }
}

TEST(AutogradTensorSurfaceTest, AccumulateGradHooksSeeDetachedCloneAndPropagate) {
  TensorImpl g = make_cpu_dense_f32({2}, 3.0f);
  TensorImpl leaf = make_cpu_dense_f32({2}, 0.0f);
  AutogradMeta* meta = get_autograd_meta(leaf, /*create_if_missing=*/true);
  ASSERT_NE(meta, nullptr);

  auto hook_impl = vbt::core::make_intrusive<RecordingHook>();
  register_leaf_hook(leaf, vbt::core::intrusive_ptr<TensorHook>(hook_impl.get(), /*add_ref=*/true));

  auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);
  auto root = vbt::core::make_intrusive<PassThrough1>();
  root->next_edges.push_back(vbt::autograd::Edge{vbt::core::intrusive_ptr<Node>(acc.get()), 0});

  std::vector<OptionalTensor> seed(1);
  seed[0] = g;

  run_backward(vbt::core::intrusive_ptr<Node>(root.get()), seed, {});

  ASSERT_TRUE(meta->grad_ptr != nullptr && meta->grad_has);
  const TensorImpl& grad = *meta->grad_ptr;
  const float* pdata = static_cast<const float*>(grad.data());
  EXPECT_FLOAT_EQ(pdata[0], 3.0f);
  EXPECT_FLOAT_EQ(pdata[1], 3.0f);

  // Hook sees exactly one detached clone
  ASSERT_EQ(hook_impl->seen.size(), 1u);
  const TensorImpl& seen = hook_impl->seen[0];
  EXPECT_EQ(seen.sizes(), grad.sizes());
  EXPECT_EQ(seen.strides(), grad.strides());
  const float* hdata = static_cast<const float*>(seen.data());
  EXPECT_FLOAT_EQ(hdata[0], 3.0f);
  EXPECT_FLOAT_EQ(hdata[1], 3.0f);
  // Storage is different (clone), so mutating hook argument would not affect .grad
  EXPECT_NE(seen.storage().get(), grad.storage().get());
}

TEST(AutogradTensorSurfaceTest, AccumulateGradHookExceptionsPropagateAndStopSequence) {
  TensorImpl g = make_cpu_dense_f32({2}, 1.0f);
  TensorImpl leaf = make_cpu_dense_f32({2}, 0.0f);
  AutogradMeta* meta = get_autograd_meta(leaf, /*create_if_missing=*/true);
  ASSERT_NE(meta, nullptr);

  auto hook_impl = vbt::core::make_intrusive<RecordingHook>();
  hook_impl->throw_on_call = true;
  register_leaf_hook(leaf, vbt::core::intrusive_ptr<TensorHook>(hook_impl.get(), /*add_ref=*/true));

  auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);
  auto root = vbt::core::make_intrusive<PassThrough1>();
  root->next_edges.push_back(vbt::autograd::Edge{vbt::core::intrusive_ptr<Node>(acc.get()), 0});

  std::vector<OptionalTensor> seed(1);
  seed[0] = g;

  EXPECT_THROW(run_backward(vbt::core::intrusive_ptr<Node>(root.get()), seed, {}), std::runtime_error);
  // Even when hooks throw, the gradient buffer has been updated.
  ASSERT_TRUE(meta->grad_ptr != nullptr && meta->grad_has);
}
