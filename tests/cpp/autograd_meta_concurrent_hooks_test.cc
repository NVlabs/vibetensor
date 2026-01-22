// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "vbt/autograd/meta.h"
#include "vbt/autograd/node.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/core/tensor.h"

using vbt::autograd::AccumulateGrad;
using vbt::autograd::AutogradMeta;
using vbt::autograd::OptionalTensor;
using vbt::autograd::TensorHook;
using vbt::autograd::get_autograd_meta;
using vbt::autograd::register_leaf_hook;
using vbt::autograd::set_requires_grad;
using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;

namespace {

static TensorImpl make_cpu_dense_f32_1d(std::int64_t n, float fill) {
  const std::size_t ne = static_cast<std::size_t>(n < 0 ? 0 : n);
  const std::size_t nbytes = ne * sizeof(float);

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  StoragePtr st = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<std::int64_t> sizes{n};
  std::vector<std::int64_t> strides{1};
  TensorImpl t(st, sizes, strides, 0, vbt::core::ScalarType::Float32, vbt::core::Device::cpu());

  float* p = static_cast<float*>(t.data());
  for (std::size_t i = 0; i < ne; ++i) {
    p[i] = fill;
  }
  return t;
}

struct TryLockGradMutexHook final : TensorHook {
  AutogradMeta* meta{nullptr};
  std::atomic<bool> could_lock{false};

  explicit TryLockGradMutexHook(AutogradMeta* m) : meta(m) {}

  void call(const TensorImpl&) override {
    if (!meta) return;
    std::unique_lock<std::mutex> lk(meta->grad_mutex, std::try_to_lock);
    could_lock.store(lk.owns_lock(), std::memory_order_relaxed);
  }
};

struct RemovedNoopHook final : TensorHook {
  RemovedNoopHook() { set_removed(true); }
  void call(const TensorImpl&) override {}
};

} // namespace

#if VBT_WITH_AUTOGRAD
TEST(AutogradMetaConcurrentHooks, HookInvocationDoesNotHoldGradMutex) {
  TensorImpl leaf = make_cpu_dense_f32_1d(/*n=*/4, 0.0f);
  set_requires_grad(leaf, true);
  AutogradMeta* meta = get_autograd_meta(leaf, /*create_if_missing=*/false);
  ASSERT_NE(meta, nullptr);

  auto hook_impl = vbt::core::make_intrusive<TryLockGradMutexHook>(meta);
  auto hook = vbt::core::intrusive_ptr<TensorHook>(hook_impl.get(), /*add_ref=*/true);
  register_leaf_hook(leaf, hook);

  AccumulateGrad ag(meta);
  std::vector<OptionalTensor> in(1);
  in[0] = make_cpu_dense_f32_1d(/*n=*/4, 1.0f);
  ag.apply(std::move(in));

  EXPECT_TRUE(hook_impl->could_lock.load(std::memory_order_relaxed));
}

TEST(AutogradMetaConcurrentHooks, ConcurrentRegistrationAndAccumulateGradCompletes) {
  constexpr int kRegisterIters = 2000;
  constexpr int kApplyIters = 400;

  TensorImpl leaf = make_cpu_dense_f32_1d(/*n=*/16, 0.0f);
  set_requires_grad(leaf, true);
  AutogradMeta* meta = get_autograd_meta(leaf, /*create_if_missing=*/false);
  ASSERT_NE(meta, nullptr);

  AccumulateGrad ag(meta);

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};

  std::thread registrar([&]() {
    ready.fetch_add(1, std::memory_order_acq_rel);
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (int i = 0; i < kRegisterIters; ++i) {
      auto h_impl = vbt::core::make_intrusive<RemovedNoopHook>();
      auto h = vbt::core::intrusive_ptr<TensorHook>(h_impl.get(), /*add_ref=*/true);
      register_leaf_hook(leaf, std::move(h));
    }
  });

  std::thread applier([&]() {
    ready.fetch_add(1, std::memory_order_acq_rel);
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (int i = 0; i < kApplyIters; ++i) {
      std::vector<OptionalTensor> in(1);
      in[0] = make_cpu_dense_f32_1d(/*n=*/16, 1.0f);
      ag.apply(std::move(in));
    }
  });

  while (ready.load(std::memory_order_acquire) < 2) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);

  registrar.join();
  applier.join();

  // Basic sanity: some hooks were registered.
  {
    std::lock_guard<std::mutex> lk(meta->grad_mutex);
    EXPECT_GT(meta->hooks.size(), 0u);
  }
}
#endif
