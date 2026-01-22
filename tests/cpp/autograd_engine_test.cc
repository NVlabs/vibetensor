// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <type_traits>

#include "vbt/autograd/engine.h"
#include "vbt/autograd/stats.h"
#include "vbt/autograd/function.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"

using vbt::autograd::AccumulateGrad;
using vbt::autograd::AutogradStatsSnapshot;
using vbt::autograd::Engine;
using vbt::autograd::FunctionNode;
using vbt::autograd::InputMeta;
using vbt::autograd::OptionalTensor;
using vbt::autograd::ensure_next_edges_sized;
using vbt::autograd::reset_stats;
using vbt::autograd::run_backward;
using vbt::autograd::stats;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;

namespace {

static TensorImpl make_cpu_dense_f32(const std::vector<int64_t>& sizes, float fill) {
  std::size_t ne = 1;
  for (auto s : sizes) ne *= static_cast<std::size_t>(s == 0 ? 1 : s);
  std::size_t nbytes = ne * sizeof(float);
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
  TensorImpl t(st, sizes, strides, /*offset=*/0, ScalarType::Float32, Device::cpu());
  float* p = static_cast<float*>(t.data());
  for (std::size_t i = 0; i < ne; ++i) p[i] = fill;
  return t;
}

} // anonymous namespace

static_assert(std::is_empty<Engine>::value,
              "Basic: Engine must remain stateless (no data members)");

TEST(AutogradEngineTest, EngineSingletonStatelessness) {
  Engine& e1 = Engine::get_default_engine();
  Engine& e2 = Engine::get_default_engine();
  EXPECT_EQ(&e1, &e2);
}

TEST(AutogradEngineTest, EngineVsFreeRunBackwardEquivalent) {
  // Simple 1-input FunctionNode that passes grad through to an AccumulateGrad sink.
  auto run_once = [](bool via_engine, std::vector<float>& grad_values,
                     AutogradStatsSnapshot& after_stats) {
    reset_stats();
    AutogradStatsSnapshot before = stats();

    // Leaf and its meta/AccumulateGrad sink
    TensorImpl leaf = make_cpu_dense_f32({2}, 0.0f);
    auto* meta = vbt::autograd::get_autograd_meta(leaf, /*create_if_missing=*/true);
    auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);

    // Root FunctionNode that simply forwards its single input gradient.
    std::vector<InputMeta> metas = {
        InputMeta{ScalarType::Float32, Device::cpu(), {2}, /*is_strided_dense=*/true}};
    auto backward = [](std::vector<OptionalTensor>&& gin) {
      std::vector<OptionalTensor> out(1);
      if (!gin.empty()) {
        out[0] = std::move(gin[0]);
      }
      return out;
    };
    auto node = vbt::core::make_intrusive<FunctionNode>("Pass", metas, backward);
    ensure_next_edges_sized(*node);
    node->next_edges[0] = vbt::autograd::Edge{vbt::core::intrusive_ptr<vbt::autograd::Node>(acc.get()), 0};

    // Seed gradient into root.
    std::vector<OptionalTensor> seed(1);
    seed[0] = make_cpu_dense_f32({2}, 1.0f);

    if (via_engine) {
      Engine::get_default_engine().run_backward(
          vbt::core::intrusive_ptr<vbt::autograd::Node>(node.get()), seed, {});
    } else {
      run_backward(vbt::core::intrusive_ptr<vbt::autograd::Node>(node.get()), seed, {});
    }

    after_stats = stats();

    ASSERT_TRUE(meta->grad_ptr != nullptr && meta->grad_has);
    const TensorImpl& grad = *meta->grad_ptr;
    const float* p = static_cast<const float*>(grad.data());
    grad_values.assign(p, p + grad.numel());

    // Sanity: at least one engine run happened.
    EXPECT_EQ(after_stats.engine_runs, before.engine_runs + 1);
  };

  std::vector<float> grad_free;
  std::vector<float> grad_engine;
  AutogradStatsSnapshot stats_free{};
  AutogradStatsSnapshot stats_engine{};

  run_once(/*via_engine=*/false, grad_free, stats_free);
  run_once(/*via_engine=*/true, grad_engine, stats_engine);

  ASSERT_EQ(grad_free.size(), grad_engine.size());
  for (std::size_t i = 0; i < grad_free.size(); ++i) {
    EXPECT_FLOAT_EQ(grad_free[i], grad_engine[i]) << "index " << i;
  }

  EXPECT_EQ(stats_free.engine_runs, stats_engine.engine_runs);
  EXPECT_EQ(stats_free.engine_nodes_processed, stats_engine.engine_nodes_processed);
  EXPECT_EQ(stats_free.engine_edges_processed, stats_engine.engine_edges_processed);
  EXPECT_EQ(stats_free.engine_callbacks_run, stats_engine.engine_callbacks_run);
  EXPECT_EQ(stats_free.engine_duplicates_coalesced, stats_engine.engine_duplicates_coalesced);
}

TEST(AutogradEngineTest, LifecycleMethodsAreNoOps) {
  reset_stats();
  AutogradStatsSnapshot before = stats();

  Engine& engine = Engine::get_default_engine();
  EXPECT_NO_THROW(engine.start_device_threads_once());
  EXPECT_NO_THROW(engine.stop());
  EXPECT_NO_THROW(engine.release_workers());

  AutogradStatsSnapshot mid = stats();
  EXPECT_EQ(mid.engine_runs, before.engine_runs);
  EXPECT_EQ(mid.engine_nodes_processed, before.engine_nodes_processed);
  EXPECT_EQ(mid.engine_edges_processed, before.engine_edges_processed);
  EXPECT_EQ(mid.engine_callbacks_run, before.engine_callbacks_run);
  EXPECT_EQ(mid.engine_duplicates_coalesced, before.engine_duplicates_coalesced);

  // Run a simple backward to ensure Engine still works after lifecycle calls.
  TensorImpl leaf = make_cpu_dense_f32({2}, 0.0f);
  auto* meta = vbt::autograd::get_autograd_meta(leaf, /*create_if_missing=*/true);
  auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);

  std::vector<InputMeta> metas = {
      InputMeta{ScalarType::Float32, Device::cpu(), {2}, /*is_strided_dense=*/true}};
  auto backward = [](std::vector<OptionalTensor>&& gin) {
    std::vector<OptionalTensor> out(1);
    if (!gin.empty()) {
      out[0] = std::move(gin[0]);
    }
    return out;
  };
  auto node = vbt::core::make_intrusive<FunctionNode>("PassLifecycle", metas, backward);
  ensure_next_edges_sized(*node);
  node->next_edges[0] =
      vbt::autograd::Edge{vbt::core::intrusive_ptr<vbt::autograd::Node>(acc.get()), 0};

  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);
  run_backward(vbt::core::intrusive_ptr<vbt::autograd::Node>(node.get()), seed, {});

  AutogradStatsSnapshot after = stats();
  EXPECT_EQ(after.engine_runs, mid.engine_runs + 1);
  ASSERT_TRUE(meta->grad_ptr != nullptr && meta->grad_has);
}
