// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>

#include "vbt/autograd/engine.h"
#include "vbt/autograd/function.h"
#include "vbt/autograd/meta.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"

using vbt::autograd::GraphTask;
using vbt::autograd::InputMeta;
using vbt::autograd::Node;
using vbt::autograd::OptionalTensor;
using vbt::autograd::ensure_next_edges_sized;
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

// Simple consumer node for InputBuffer slot behavior tests.
struct DummyConsumer final : Node {
  uint32_t num_inputs() const noexcept override { return 1; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    std::vector<OptionalTensor> out(1);
    if (!grads_in.empty()) {
      out[0] = std::move(grads_in[0]);
    }
    return out;
  }
};

struct CountingConsumer final : Node {
  int apply_calls{0};

  uint32_t num_inputs() const noexcept override { return 1; }
  uint32_t num_incoming_grad_slots() const noexcept override { return 2; }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    (void)grads_in;
    ++apply_calls;
    return std::vector<OptionalTensor>(1);
  }
};

} // anonymous namespace

// Test-only helpers implemented in src/vbt/autograd/engine.cc
namespace vbt { namespace autograd {
void _test_route_edge_with_coalesce(GraphTask& gt,
                                    Node* consumer,
                                    std::size_t pos,
                                    OptionalTensor&& grad,
                                    vbt::core::intrusive_ptr<Node> consumer_keep);
void _test_add_gradient(GraphTask& gt,
                        Node* consumer,
                        std::size_t pos,
                        OptionalTensor&& grad,
                        vbt::core::intrusive_ptr<Node> consumer_keep);
void _test_compute_dependencies(GraphTask& gt,
                                vbt::core::intrusive_ptr<Node> root);
void _test_seed_root_buffer(GraphTask& gt,
                            vbt::core::intrusive_ptr<Node> root,
                            const std::vector<OptionalTensor>& initial_grads);
void _test_validate_graph_task_structure(GraphTask& gt,
                                         vbt::core::intrusive_ptr<Node> root);
}} // namespace vbt::autograd

using vbt::autograd::_test_route_edge_with_coalesce;
using vbt::autograd::_test_add_gradient;
using vbt::autograd::_test_compute_dependencies;
using vbt::autograd::_test_seed_root_buffer;
using vbt::autograd::_test_validate_graph_task_structure;

TEST(AutogradEngineGraphTopology, ChainDependenciesAndNodesInGraph) {
  GraphTask gt;

  auto root = vbt::core::make_intrusive<DummyConsumer>();
  auto mid = vbt::core::make_intrusive<DummyConsumer>();
  auto leaf = vbt::core::make_intrusive<DummyConsumer>();

  root->next_edges.resize(1);
  root->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<Node>(mid.get()), 0};

  mid->next_edges.resize(1);
  mid->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<Node>(leaf.get()), 0};

  _test_compute_dependencies(gt, intrusive_ptr<Node>(root.get()));

  // All nodes are discovered.
  EXPECT_EQ(gt.nodes_in_graph.size(), 3u);
  EXPECT_TRUE(gt.nodes_in_graph.count(root.get()));
  EXPECT_TRUE(gt.nodes_in_graph.count(mid.get()));
  EXPECT_TRUE(gt.nodes_in_graph.count(leaf.get()));

  // Dependencies reflect structural in-degree.
  ASSERT_TRUE(gt.dependencies.count(root.get()));
  ASSERT_TRUE(gt.dependencies.count(mid.get()));
  ASSERT_TRUE(gt.dependencies.count(leaf.get()));

  EXPECT_EQ(gt.dependencies[root.get()], 0);
  EXPECT_EQ(gt.dependencies[mid.get()], 1);
  EXPECT_EQ(gt.dependencies[leaf.get()], 1);
}

TEST(AutogradEngineGraphTopology, DiamondGraphDependencies) {
  GraphTask gt;

  auto root = vbt::core::make_intrusive<DummyConsumer>();
  auto left = vbt::core::make_intrusive<DummyConsumer>();
  auto right = vbt::core::make_intrusive<DummyConsumer>();
  auto join = vbt::core::make_intrusive<DummyConsumer>();

  // root -> left, root -> right
  root->next_edges.resize(2);
  root->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<Node>(left.get()), 0};
  root->next_edges[1] = vbt::autograd::Edge{intrusive_ptr<Node>(right.get()), 0};

  // left -> join, right -> join
  left->next_edges.resize(1);
  left->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<Node>(join.get()), 0};
  right->next_edges.resize(1);
  right->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<Node>(join.get()), 0};

  _test_compute_dependencies(gt, intrusive_ptr<Node>(root.get()));

  EXPECT_EQ(gt.nodes_in_graph.size(), 4u);
  EXPECT_TRUE(gt.nodes_in_graph.count(root.get()));
  EXPECT_TRUE(gt.nodes_in_graph.count(left.get()));
  EXPECT_TRUE(gt.nodes_in_graph.count(right.get()));
  EXPECT_TRUE(gt.nodes_in_graph.count(join.get()));

  ASSERT_TRUE(gt.dependencies.count(root.get()));
  ASSERT_TRUE(gt.dependencies.count(left.get()));
  ASSERT_TRUE(gt.dependencies.count(right.get()));
  ASSERT_TRUE(gt.dependencies.count(join.get()));

  EXPECT_EQ(gt.dependencies[root.get()], 0);
  EXPECT_EQ(gt.dependencies[left.get()], 1);
  EXPECT_EQ(gt.dependencies[right.get()], 1);
  EXPECT_EQ(gt.dependencies[join.get()], 2);
}

TEST(AutogradEngineGraphTopology, CycleDetectionThrowsLogicError) {
  GraphTask gt;

  auto a = vbt::core::make_intrusive<DummyConsumer>();
  auto b = vbt::core::make_intrusive<DummyConsumer>();

  a->next_edges.resize(1);
  a->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<Node>(b.get()), 0};
  b->next_edges.resize(1);
  b->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<Node>(a.get()), 0};

  EXPECT_THROW(_test_compute_dependencies(gt, intrusive_ptr<Node>(a.get())), std::logic_error);
}

TEST(AutogradEngineGraphTopology, NullEdgesDoNotContributeDependenciesOrNodes) {
  GraphTask gt;

  auto root = vbt::core::make_intrusive<DummyConsumer>();

  // Two outgoing edges, both null.
  root->next_edges.resize(2);
  root->next_edges[0] = vbt::autograd::Edge{};
  root->next_edges[1] = vbt::autograd::Edge{};

  _test_compute_dependencies(gt, intrusive_ptr<Node>(root.get()));

  // Only the root should be present.
  EXPECT_EQ(gt.nodes_in_graph.size(), 1u);
  EXPECT_TRUE(gt.nodes_in_graph.count(root.get()));

  ASSERT_TRUE(gt.dependencies.count(root.get()));
  EXPECT_EQ(gt.dependencies[root.get()], 0);
  EXPECT_EQ(gt.dependencies.size(), 1u);
}

TEST(AutogradEngineGraphTopology, UnreachableNodesAreNotAddedToGraph) {
  GraphTask gt;

  auto root = vbt::core::make_intrusive<DummyConsumer>();
  auto orphan = vbt::core::make_intrusive<DummyConsumer>();

  // Root has no outgoing edges; orphan is never referenced.
  _test_compute_dependencies(gt, intrusive_ptr<Node>(root.get()));

  EXPECT_EQ(gt.nodes_in_graph.size(), 1u);
  EXPECT_TRUE(gt.nodes_in_graph.count(root.get()));
  EXPECT_FALSE(gt.nodes_in_graph.count(orphan.get()));

  ASSERT_TRUE(gt.dependencies.count(root.get()));
  EXPECT_FALSE(gt.dependencies.count(orphan.get()));
}

TEST(AutogradEngineInputBuffer, RootSeedingInitialGradsPopulateFreshBufferAndEnqueueRoot) {
  GraphTask gt;
  auto root = vbt::core::make_intrusive<DummyConsumer>();

  // initial_grads.size() == root->num_inputs() == 1; this is the
  // happy-path contract for root seeding.
  std::vector<OptionalTensor> seed(1);
  OptionalTensor g0;
  g0 = make_cpu_dense_f32({1}, 7.0f);
  seed[0] = g0;  // defined gradient

  _test_seed_root_buffer(gt, intrusive_ptr<Node>(root.get()), seed);

  auto it = gt.inputs.find(root.get());
  ASSERT_NE(it, gt.inputs.end());
  const GraphTask::InputBuffer& ib = it->second;

  EXPECT_EQ(ib.expected, seed.size());
  EXPECT_EQ(ib.grads_in.size(), seed.size());
  EXPECT_EQ(ib.present.size(), seed.size());
  EXPECT_EQ(ib.received, seed.size());
  EXPECT_TRUE(ib.enqueued);

  ASSERT_EQ(ib.present.size(), 1u);
  EXPECT_EQ(ib.present[0], 1u);
  ASSERT_TRUE(ib.grads_in[0].has_value());

  ASSERT_FALSE(gt.ready.empty());
  auto queued = gt.ready.pop_front();
  EXPECT_EQ(queued.get(), root.get());
  EXPECT_TRUE(gt.ready.empty());
}

TEST(AutogradEngineInputBuffer, RootSeedingMismatchedInitialGradsSizeThrows) {
  GraphTask gt;
  auto root = vbt::core::make_intrusive<DummyConsumer>();

  // root->num_inputs() == 1; provide two seeds to trigger the
  // size-equality guard in seed_root_input_buffer.
  std::vector<OptionalTensor> seed(2);

  EXPECT_THROW(_test_seed_root_buffer(gt, intrusive_ptr<Node>(root.get()), seed),
               std::invalid_argument);

  // Helper must not have created a buffer or enqueued the root.
  EXPECT_TRUE(gt.inputs.empty());
  EXPECT_TRUE(gt.ready.empty());
}

TEST(AutogradEngineInputBuffer, SingleDefinedArrivalPopulatesSlotAndEnqueues) {
  GraphTask gt;
  auto consumer = vbt::core::make_intrusive<DummyConsumer>();

  OptionalTensor g;
  g = make_cpu_dense_f32({2}, 1.0f);

  _test_route_edge_with_coalesce(gt, consumer.get(), 0, std::move(g), intrusive_ptr<Node>(consumer.get()));

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  const GraphTask::InputBuffer& ib = it->second;

  EXPECT_EQ(ib.expected, 1u);
  ASSERT_EQ(ib.grads_in.size(), 1u);
  ASSERT_EQ(ib.present.size(), 1u);
  EXPECT_EQ(ib.present[0], 1u);
  EXPECT_EQ(ib.received, 1u);
  ASSERT_TRUE(ib.grads_in[0].has_value());
  const TensorImpl& stored = ib.grads_in[0].value();
  ASSERT_EQ(stored.numel(), 2);
  const float* p = static_cast<const float*>(stored.data());
  EXPECT_FLOAT_EQ(p[0], 1.0f);
  EXPECT_FLOAT_EQ(p[1], 1.0f);
  EXPECT_TRUE(ib.enqueued);
  ASSERT_FALSE(gt.ready.empty());
  EXPECT_EQ(gt.ready.pop_front().get(), consumer.get());
}

TEST(AutogradEngineInputBuffer, DuplicateDefinedArrivalAccumulatesAndCountsDuplicate) {
  GraphTask gt;
  auto consumer = vbt::core::make_intrusive<DummyConsumer>();

  OptionalTensor g1;
  OptionalTensor g2;
  g1 = make_cpu_dense_f32({2}, 1.0f);
  g2 = make_cpu_dense_f32({2}, 2.0f);

  _test_route_edge_with_coalesce(gt, consumer.get(), 0, std::move(g1), intrusive_ptr<Node>(consumer.get()));
  EXPECT_EQ(gt.duplicates_coalesced, 0u);
  _test_route_edge_with_coalesce(gt, consumer.get(), 0, std::move(g2), intrusive_ptr<Node>(consumer.get()));
  EXPECT_EQ(gt.duplicates_coalesced, 1u);

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  const GraphTask::InputBuffer& ib = it->second;
  ASSERT_TRUE(ib.grads_in[0].has_value());
  const TensorImpl& stored = ib.grads_in[0].value();
  ASSERT_EQ(stored.numel(), 2);
  const float* p = static_cast<const float*>(stored.data());
  EXPECT_FLOAT_EQ(p[0], 3.0f);
  EXPECT_FLOAT_EQ(p[1], 3.0f);
  EXPECT_EQ(ib.received, 1u);
  EXPECT_TRUE(ib.enqueued);
}

TEST(AutogradEngineInputBuffer, NulloptSequencesDoNotCreateGradTensors) {
  GraphTask gt;
  auto consumer = vbt::core::make_intrusive<DummyConsumer>();

  OptionalTensor g_null;
  _test_route_edge_with_coalesce(gt, consumer.get(), 0, std::move(g_null), intrusive_ptr<Node>(consumer.get()));

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  GraphTask::InputBuffer& ib = it->second;
  EXPECT_EQ(ib.received, 1u);
  EXPECT_EQ(ib.present[0], 1u);
  EXPECT_FALSE(ib.grads_in[0].has_value());
  EXPECT_TRUE(ib.enqueued);

  OptionalTensor g_def;
  g_def = make_cpu_dense_f32({2}, 4.0f);
  _test_route_edge_with_coalesce(gt, consumer.get(), 0, std::move(g_def), intrusive_ptr<Node>(consumer.get()));

  EXPECT_EQ(gt.duplicates_coalesced, 0u);
  ASSERT_TRUE(ib.grads_in[0].has_value());
  const TensorImpl& stored = ib.grads_in[0].value();
  const float* p = static_cast<const float*>(stored.data());
  EXPECT_FLOAT_EQ(p[0], 4.0f);
  EXPECT_FLOAT_EQ(p[1], 4.0f);
  EXPECT_EQ(ib.received, 1u);
}

TEST(AutogradEngineInputBuffer, DefinedThenNulloptKeepsOriginalGrad) {
  GraphTask gt;
  auto consumer = vbt::core::make_intrusive<DummyConsumer>();

  OptionalTensor g_def;
  g_def = make_cpu_dense_f32({1}, 5.0f);
  _test_route_edge_with_coalesce(gt, consumer.get(), 0, std::move(g_def), intrusive_ptr<Node>(consumer.get()));

  OptionalTensor g_null;
  _test_route_edge_with_coalesce(gt, consumer.get(), 0, std::move(g_null), intrusive_ptr<Node>(consumer.get()));

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  const GraphTask::InputBuffer& ib = it->second;
  ASSERT_TRUE(ib.grads_in[0].has_value());
  const TensorImpl& stored = ib.grads_in[0].value();
  ASSERT_EQ(stored.numel(), 1);
  const float* p = static_cast<const float*>(stored.data());
  EXPECT_FLOAT_EQ(p[0], 5.0f);
  EXPECT_EQ(gt.duplicates_coalesced, 0u);
  EXPECT_EQ(ib.received, 1u);
}

TEST(AutogradEngineInputBuffer, PartialSlotsDoNotEnqueueConsumer) {
  GraphTask gt;
  auto consumer = vbt::core::make_intrusive<CountingConsumer>();

  OptionalTensor g;
  g = make_cpu_dense_f32({1}, 1.0f);
  _test_route_edge_with_coalesce(gt, consumer.get(), 0, std::move(g), intrusive_ptr<Node>(consumer.get()));

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  const GraphTask::InputBuffer& ib = it->second;
  EXPECT_EQ(ib.expected, 2u);  // CountingConsumer::num_incoming_grad_slots
  EXPECT_EQ(ib.received, 1u);
  ASSERT_EQ(ib.present.size(), 2u);
  EXPECT_EQ(ib.present[0], 1u);
  EXPECT_EQ(ib.present[1], 0u);
  EXPECT_FALSE(ib.enqueued);
  EXPECT_TRUE(gt.ready.empty());
}

TEST(AutogradEngineInputBuffer, SafetyNetRunsPartiallyFilledBuffers) {
  auto consumer = vbt::core::make_intrusive<CountingConsumer>();
  ensure_next_edges_sized(*consumer);

  std::vector<InputMeta> metas = {
      InputMeta{ScalarType::Float32, Device::cpu(), {1}, /*is_strided_dense=*/true}};
  auto backward = [](std::vector<OptionalTensor>&& gin) {
    std::vector<OptionalTensor> out(1);
    if (!gin.empty()) {
      out[0] = std::move(gin[0]);
    }
    return out;
  };
  auto root = vbt::core::make_intrusive<vbt::autograd::FunctionNode>("Root", metas, backward);
  ensure_next_edges_sized(*root);
  root->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<Node>(consumer.get()), 0};

  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cpu_dense_f32({1}, 1.0f);

  run_backward(intrusive_ptr<Node>(root.get()), seed, {});

  EXPECT_EQ(consumer->apply_calls, 1);
}

// --- add_gradient parity tests vs coalesce_incoming ---

namespace {

static void run_sequence_with_helpers(const std::vector<int>& pattern) {
  GraphTask gt_coalesce;
  GraphTask gt_add;

  auto consumer_coalesce = vbt::core::make_intrusive<DummyConsumer>();
  auto consumer_add = vbt::core::make_intrusive<DummyConsumer>();

  for (std::size_t i = 0; i < pattern.size(); ++i) {
    const int kind = pattern[i];

    OptionalTensor g_coalesce;
    OptionalTensor g_add;
    if (kind == 1) {
      g_coalesce = make_cpu_dense_f32({1}, static_cast<float>(i + 1));
      g_add = make_cpu_dense_f32({1}, static_cast<float>(i + 1));
    }

    _test_route_edge_with_coalesce(gt_coalesce,
                                   consumer_coalesce.get(),
                                   0,
                                   std::move(g_coalesce),
                                   intrusive_ptr<Node>(consumer_coalesce.get()));

    _test_add_gradient(gt_add,
                       consumer_add.get(),
                       0,
                       std::move(g_add),
                       intrusive_ptr<Node>(consumer_add.get()));
  }

  auto it_coalesce = gt_coalesce.inputs.find(consumer_coalesce.get());
  auto it_add = gt_add.inputs.find(consumer_add.get());
  ASSERT_NE(it_coalesce, gt_coalesce.inputs.end());
  ASSERT_NE(it_add, gt_add.inputs.end());

  const GraphTask::InputBuffer& ib_c = it_coalesce->second;
  const GraphTask::InputBuffer& ib_a = it_add->second;

  EXPECT_EQ(ib_c.expected, ib_a.expected);
  EXPECT_EQ(ib_c.received, ib_a.received);
  EXPECT_EQ(ib_c.enqueued, ib_a.enqueued);
  ASSERT_EQ(ib_c.present.size(), ib_a.present.size());
  ASSERT_EQ(ib_c.grads_in.size(), ib_a.grads_in.size());

  for (std::size_t i = 0; i < ib_c.present.size(); ++i) {
    EXPECT_EQ(ib_c.present[i], ib_a.present[i]) << "present mismatch at slot " << i;
    const OptionalTensor& sc = ib_c.grads_in[i];
    const OptionalTensor& sa = ib_a.grads_in[i];
    EXPECT_EQ(sc.has_value(), sa.has_value()) << "has_value mismatch at slot " << i;
    if (sc.has_value()) {
      const TensorImpl& tc = sc.value();
      const TensorImpl& ta = sa.value();
      ASSERT_EQ(tc.numel(), ta.numel());
      const float* pc = static_cast<const float*>(tc.data());
      const float* pa = static_cast<const float*>(ta.data());
      for (int64_t j = 0; j < tc.numel(); ++j) {
        EXPECT_FLOAT_EQ(pc[j], pa[j]) << "value mismatch at element " << j;
      }
    }
  }

  EXPECT_EQ(gt_coalesce.duplicates_coalesced, gt_add.duplicates_coalesced);
  EXPECT_EQ(gt_coalesce.ready.empty(), gt_add.ready.empty());
}

} // anonymous namespace

TEST(AutogradEngineAddGradient, ParityWithCoalesceForBasicSequences) {
  // Single defined arrival.
  run_sequence_with_helpers({1});

  // Two defined arrivals (duplicate) should accumulate and bump duplicates_coalesced.
  run_sequence_with_helpers({1, 1});

  // Nullopt first, then defined: promotion behavior.
  run_sequence_with_helpers({0, 1});

  // Defined, then nullopt: keep original grad.
  run_sequence_with_helpers({1, 0});
}

TEST(AutogradEngineAddGradient, OutOfRangeSlotThrowsSameAsCoalesce) {
  GraphTask gt_coalesce;
  GraphTask gt_add;

  auto consumer_coalesce = vbt::core::make_intrusive<DummyConsumer>();
  auto consumer_add = vbt::core::make_intrusive<DummyConsumer>();

  OptionalTensor g1;
  g1 = make_cpu_dense_f32({1}, 1.0f);

  // First arrival to create buffers.
  _test_route_edge_with_coalesce(gt_coalesce,
                                 consumer_coalesce.get(),
                                 0,
                                 std::move(g1),
                                 intrusive_ptr<Node>(consumer_coalesce.get()));

  OptionalTensor g2;
  g2 = make_cpu_dense_f32({1}, 1.0f);

  // Second helper should throw on out-of-range slot index.
  EXPECT_THROW(
      _test_route_edge_with_coalesce(gt_coalesce,
                                     consumer_coalesce.get(),
                                     1,
                                     std::move(g2),
                                     intrusive_ptr<Node>(consumer_coalesce.get())),
      std::out_of_range);

  OptionalTensor g3;
  g3 = make_cpu_dense_f32({1}, 1.0f);

  EXPECT_THROW(
      _test_add_gradient(gt_add,
                         consumer_add.get(),
                         1,
                         std::move(g3),
                         intrusive_ptr<Node>(consumer_add.get())),
      std::out_of_range);
}

TEST(AutogradEngineValidation, ValidGraphPassesStructureChecks) {
  GraphTask gt;

  auto root = vbt::core::make_intrusive<DummyConsumer>();
  auto consumer = vbt::core::make_intrusive<DummyConsumer>();

  // root -> consumer (single edge into slot 0)
  root->next_edges.resize(1);
  root->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<Node>(consumer.get()), 0};

  _test_compute_dependencies(gt, intrusive_ptr<Node>(root.get()));

  // Seed root buffer with one defined gradient.
  std::vector<OptionalTensor> seed(1);
  OptionalTensor g0;
  g0 = make_cpu_dense_f32({1}, 1.0f);
  seed[0] = g0;
  _test_seed_root_buffer(gt, intrusive_ptr<Node>(root.get()), seed);

  // Route a single defined grad into the consumer slot via coalesce helper.
  OptionalTensor g_consumer;
  g_consumer = make_cpu_dense_f32({1}, 2.0f);
  _test_route_edge_with_coalesce(gt,
                                 consumer.get(),
                                 0,
                                 std::move(g_consumer),
                                 intrusive_ptr<Node>(consumer.get()));

  EXPECT_NO_THROW(_test_validate_graph_task_structure(gt, intrusive_ptr<Node>(root.get())));
}

TEST(AutogradEngineValidation, InputBufferPresentVsGradsInvariantIsEnforced) {
  GraphTask gt;
  auto node = vbt::core::make_intrusive<DummyConsumer>();

  // Populate structural accounting to satisfy membership checks.
  gt.nodes_in_graph.insert(node.get());
  gt.dependencies.emplace(node.get(), 0);

  GraphTask::InputBuffer ib;
  ib.ensure_cpu_capacity(1);
  // Violate invariant: present[0] == 0 but grads_in[0] engaged.
  ib.grads_in[0] = make_cpu_dense_f32({1}, 3.0f);
  ib.present[0] = 0u;
  ib.received = 0u;

  gt.inputs.emplace(node.get(), std::move(ib));

  EXPECT_THROW(
      _test_validate_graph_task_structure(gt, intrusive_ptr<Node>(node.get())),
      std::logic_error);
}

TEST(AutogradEngineValidation, DuplicatesCoalescedBoundIsEnforced) {
  GraphTask gt;

  auto root = vbt::core::make_intrusive<DummyConsumer>();
  auto consumer = vbt::core::make_intrusive<DummyConsumer>();

  // root has two structural edges into the same consumer slot 0.
  root->next_edges.resize(2);
  root->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<Node>(consumer.get()), 0};
  root->next_edges[1] = vbt::autograd::Edge{intrusive_ptr<Node>(consumer.get()), 0};

  _test_compute_dependencies(gt, intrusive_ptr<Node>(root.get()));

  std::vector<OptionalTensor> seed(1);
  OptionalTensor g0;
  g0 = make_cpu_dense_f32({1}, 1.0f);
  seed[0] = g0;
  _test_seed_root_buffer(gt, intrusive_ptr<Node>(root.get()), seed);

  // Two defined arrivals into the same consumer slot create one duplicate.
  OptionalTensor g1;
  OptionalTensor g2;
  g1 = make_cpu_dense_f32({1}, 1.0f);
  g2 = make_cpu_dense_f32({1}, 1.0f);

  _test_route_edge_with_coalesce(gt,
                                 consumer.get(),
                                 0,
                                 std::move(g1),
                                 intrusive_ptr<Node>(consumer.get()));
  _test_route_edge_with_coalesce(gt,
                                 consumer.get(),
                                 0,
                                 std::move(g2),
                                 intrusive_ptr<Node>(consumer.get()));

  // Sanity: validation should pass for the real accounting produced above.
  EXPECT_NO_THROW(_test_validate_graph_task_structure(gt, intrusive_ptr<Node>(root.get())));

  // Now deliberately violate the global bound duplicates_coalesced <= Σ(E(n)-R(n)).
  gt.duplicates_coalesced = gt.duplicates_coalesced + 1;
  EXPECT_THROW(
      _test_validate_graph_task_structure(gt, intrusive_ptr<Node>(root.get())),
      std::logic_error);
}
