// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "vbt/autograd/engine.h"
#include "vbt/autograd/engine_toggles.h"
#include "vbt/autograd/function.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"

using vbt::autograd::AccumulateGrad;
using vbt::autograd::FunctionNode;
using vbt::autograd::InputMeta;
using vbt::autograd::OptionalTensor;
using vbt::autograd::TensorHook;
using vbt::autograd::ensure_next_edges_sized;
using vbt::autograd::register_leaf_hook;
using vbt::autograd::run_backward;
using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::TensorImpl;
using vbt::core::intrusive_ptr;

namespace {

static TensorImpl make_cpu_dense_f32(const std::vector<int64_t>& sizes, float fill) {
  std::size_t ne = 1;
  for (auto s : sizes) {
    if (s == 0) {
      ne = 0;
      break;
    }
    ne *= static_cast<std::size_t>(s);
  }
  const std::size_t nbytes = ne * sizeof(float);

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  StoragePtr st = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    acc *= (sizes[idx] == 0 ? 1 : sizes[idx]);
  }

  TensorImpl t(st, sizes, strides, /*offset=*/0, ScalarType::Float32, Device::cpu());
  float* p = static_cast<float*>(t.data());
  for (std::size_t i = 0; i < ne; ++i) p[i] = fill;
  return t;
}

struct MtToggleRestore {
  bool prev;
  MtToggleRestore() : prev(vbt::autograd::is_multithreading_enabled()) {
    vbt::autograd::set_multithreading_enabled(true);
  }
  ~MtToggleRestore() {
    vbt::autograd::set_multithreading_enabled(prev);
  }
};

struct MtToggleDisableRestore {
  bool prev;
  MtToggleDisableRestore() : prev(vbt::autograd::is_multithreading_enabled()) {
    vbt::autograd::set_multithreading_enabled(false);
  }
  ~MtToggleDisableRestore() {
    vbt::autograd::set_multithreading_enabled(prev);
  }
};

struct NestedBackwardHook final : TensorHook {
  std::atomic<bool>* inner_called;
  std::atomic<bool>* inner_threw;
  std::atomic<bool>* inner_msg_ok;

  NestedBackwardHook(std::atomic<bool>* c, std::atomic<bool>* t, std::atomic<bool>* ok)
      : inner_called(c), inner_threw(t), inner_msg_ok(ok) {}

  void call(const TensorImpl&) override {
    inner_called->store(true, std::memory_order_release);

    // Build a tiny independent graph and attempt to run backward while this
    // backward is still active. This must be rejected.
    TensorImpl leaf2 = make_cpu_dense_f32({2}, 0.0f);
    auto* meta2 = vbt::autograd::get_autograd_meta(leaf2, /*create_if_missing=*/true);
    auto acc2 = vbt::core::make_intrusive<AccumulateGrad>(meta2);

    std::vector<InputMeta> meta = {
        InputMeta{ScalarType::Float32, Device::cpu(), {2}, /*is_strided_dense=*/true}};

    auto Root2 = vbt::core::make_intrusive<FunctionNode>(
        "Inner",
        meta,
        [](std::vector<OptionalTensor>&& gin) {
          std::vector<OptionalTensor> out(1);
          if (!gin.empty()) out[0] = std::move(gin[0]);
          return out;
        });
    ensure_next_edges_sized(*Root2);
    Root2->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(acc2.get()), 0};

    std::vector<OptionalTensor> seed(1);
    seed[0] = make_cpu_dense_f32({2}, 1.0f);

    try {
      run_backward(intrusive_ptr<vbt::autograd::Node>(Root2.get()), seed, {});
    } catch (const std::exception& e) {
      inner_threw->store(true, std::memory_order_release);
      const std::string msg = e.what();
      inner_msg_ok->store(msg.find("nested backward") != std::string::npos,
                          std::memory_order_release);
      return;
    } catch (...) {
      inner_threw->store(true, std::memory_order_release);
      inner_msg_ok->store(false, std::memory_order_release);
      return;
    }

    // No exception means the engine allowed nested backward (not supported).
    inner_threw->store(false, std::memory_order_release);
    inner_msg_ok->store(false, std::memory_order_release);
  }
};

static void run_leaf_hook_nested_backward_scenario() {
  std::atomic<bool> inner_called{false};
  std::atomic<bool> inner_threw{false};
  std::atomic<bool> inner_msg_ok{false};

  // Outer backward graph: Root -> AccumulateGrad(leaf)
  TensorImpl leaf = make_cpu_dense_f32({4}, 0.0f);
  auto* meta = vbt::autograd::get_autograd_meta(leaf, /*create_if_missing=*/true);
  ASSERT_NE(meta, nullptr);
  auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);

  auto hook = vbt::core::make_intrusive<NestedBackwardHook>(&inner_called, &inner_threw, &inner_msg_ok);
  register_leaf_hook(leaf, intrusive_ptr<TensorHook>(hook.get(), /*add_ref=*/true));

  std::vector<InputMeta> meta1 = {
      InputMeta{ScalarType::Float32, Device::cpu(), {4}, /*is_strided_dense=*/true}};

  auto Root = vbt::core::make_intrusive<FunctionNode>(
      "Outer",
      meta1,
      [](std::vector<OptionalTensor>&& gin) {
        std::vector<OptionalTensor> out(1);
        if (!gin.empty()) out[0] = std::move(gin[0]);
        return out;
      });
  ensure_next_edges_sized(*Root);
  Root->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(acc.get()), 0};

  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cpu_dense_f32({4}, 1.0f);

  EXPECT_NO_THROW(run_backward(intrusive_ptr<vbt::autograd::Node>(Root.get()), seed, {}));

  EXPECT_TRUE(inner_called.load(std::memory_order_acquire));
  EXPECT_TRUE(inner_threw.load(std::memory_order_acquire));
  EXPECT_TRUE(inner_msg_ok.load(std::memory_order_acquire));

  ASSERT_TRUE(meta->grad_ptr != nullptr && meta->grad_has);
  const TensorImpl& g = *meta->grad_ptr;
  const float* p = static_cast<const float*>(g.data());
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(p[i], 1.0f) << "index " << i;
  }
}

} // namespace

TEST(AutogradHardeningTest, LeafHookCallingBackwardIsRejectedMtEnabled) {
  MtToggleRestore mt_guard;
  run_leaf_hook_nested_backward_scenario();
}

TEST(AutogradHardeningTest, LeafHookCallingBackwardIsRejectedMtDisabled) {
  MtToggleDisableRestore mt_guard;
  run_leaf_hook_nested_backward_scenario();
}

TEST(AutogradHardeningTest, ConcurrentBackwardFromWorkerThreadDoesNotDeadlock) {
  MtToggleRestore mt_guard;

  std::atomic<bool> inner_attempted{false};
  std::atomic<bool> inner_threw{false};
  std::atomic<bool> inner_msg_ok{false};

  // Inner graph that the spawned thread will attempt to run.
  TensorImpl inner_leaf = make_cpu_dense_f32({2}, 0.0f);
  auto* inner_meta = vbt::autograd::get_autograd_meta(inner_leaf, /*create_if_missing=*/true);
  ASSERT_NE(inner_meta, nullptr);
  auto inner_acc = vbt::core::make_intrusive<AccumulateGrad>(inner_meta);

  std::vector<InputMeta> inner_meta1 = {
      InputMeta{ScalarType::Float32, Device::cpu(), {2}, /*is_strided_dense=*/true}};

  auto InnerRoot = vbt::core::make_intrusive<FunctionNode>(
      "InnerRoot",
      inner_meta1,
      [](std::vector<OptionalTensor>&& gin) {
        std::vector<OptionalTensor> out(1);
        if (!gin.empty()) out[0] = std::move(gin[0]);
        return out;
      });
  ensure_next_edges_sized(*InnerRoot);
  InnerRoot->next_edges[0] =
      vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(inner_acc.get()), 0};

  std::vector<OptionalTensor> inner_seed(1);
  inner_seed[0] = make_cpu_dense_f32({2}, 1.0f);

  // Outer graph: Root spawns a thread that attempts a second backward.
  TensorImpl leaf = make_cpu_dense_f32({4}, 0.0f);
  auto* meta = vbt::autograd::get_autograd_meta(leaf, /*create_if_missing=*/true);
  ASSERT_NE(meta, nullptr);
  auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);

  std::vector<InputMeta> meta1 = {
      InputMeta{ScalarType::Float32, Device::cpu(), {4}, /*is_strided_dense=*/true}};

  auto Root = vbt::core::make_intrusive<FunctionNode>(
      "OuterRoot",
      meta1,
      [&, InnerRoot, inner_seed](std::vector<OptionalTensor>&& gin) mutable {
        std::thread th([&, InnerRoot, inner_seed]() mutable {
          inner_attempted.store(true, std::memory_order_release);
          try {
            run_backward(intrusive_ptr<vbt::autograd::Node>(InnerRoot.get()), inner_seed, {});
          } catch (const std::exception& e) {
            inner_threw.store(true, std::memory_order_release);
            const std::string msg = e.what();
            inner_msg_ok.store(msg.find("already in progress") != std::string::npos,
                               std::memory_order_release);
            return;
          } catch (...) {
            inner_threw.store(true, std::memory_order_release);
            inner_msg_ok.store(false, std::memory_order_release);
            return;
          }

          inner_threw.store(false, std::memory_order_release);
          inner_msg_ok.store(false, std::memory_order_release);
        });
        th.join();

        std::vector<OptionalTensor> out(1);
        if (!gin.empty()) out[0] = std::move(gin[0]);
        return out;
      });
  ensure_next_edges_sized(*Root);
  Root->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(acc.get()), 0};

  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cpu_dense_f32({4}, 1.0f);

  // Watchdog to turn a deadlock into a fast failure.
  std::atomic<bool> finished{false};
  std::thread watchdog([&]() {
    using namespace std::chrono_literals;
    for (int i = 0; i < 150 && !finished.load(std::memory_order_acquire); ++i) {
      std::this_thread::sleep_for(100ms);
    }
    if (!finished.load(std::memory_order_acquire)) {
      std::fprintf(stderr, "AutogradHardeningTest watchdog timeout: likely deadlock\n");
      std::fflush(stderr);
      std::abort();
    }
  });

  EXPECT_NO_THROW(run_backward(intrusive_ptr<vbt::autograd::Node>(Root.get()), seed, {}));

  finished.store(true, std::memory_order_release);
  watchdog.join();

  EXPECT_TRUE(inner_attempted.load(std::memory_order_acquire));
  EXPECT_TRUE(inner_threw.load(std::memory_order_acquire));
  EXPECT_TRUE(inner_msg_ok.load(std::memory_order_acquire));

  ASSERT_TRUE(meta->grad_ptr != nullptr && meta->grad_has);
  const TensorImpl& g = *meta->grad_ptr;
  const float* p = static_cast<const float*>(g.data());
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(p[i], 1.0f) << "index " << i;
  }
}
