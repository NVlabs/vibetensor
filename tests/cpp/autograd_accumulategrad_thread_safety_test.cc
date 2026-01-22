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
using vbt::autograd::get_autograd_meta;
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

} // namespace

#if VBT_WITH_AUTOGRAD
TEST(AutogradAccumulateGradThreadSafety, ConcurrentApplyAccumulatesDeterministically) {
  constexpr int kThreads = 8;
  constexpr int kItersPerThread = 200;
  constexpr std::int64_t kN = 128;

  TensorImpl leaf = make_cpu_dense_f32_1d(kN, 0.0f);
  set_requires_grad(leaf, true);
  AutogradMeta* meta = get_autograd_meta(leaf, /*create_if_missing=*/false);
  ASSERT_NE(meta, nullptr);

  AccumulateGrad ag(meta);

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&]() {
      ready.fetch_add(1, std::memory_order_acq_rel);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      for (int i = 0; i < kItersPerThread; ++i) {
        TensorImpl g = make_cpu_dense_f32_1d(kN, 1.0f);
        std::vector<OptionalTensor> in(1);
        in[0] = std::move(g);
        ag.apply(std::move(in));
      }
    });
  }

  while (ready.load(std::memory_order_acquire) < kThreads) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);

  for (auto& th : threads) {
    th.join();
  }

  // Verify leaf .grad was accumulated deterministically.
  {
    std::lock_guard<std::mutex> lk(meta->grad_mutex);
    ASSERT_TRUE(meta->grad_ptr);
    ASSERT_TRUE(meta->grad_has);

    const float* p = static_cast<const float*>(meta->grad_ptr->data());
    const float expected = static_cast<float>(kThreads * kItersPerThread);
    for (std::int64_t i = 0; i < kN; ++i) {
      EXPECT_FLOAT_EQ(p[i], expected);
    }
  }
}
#endif
