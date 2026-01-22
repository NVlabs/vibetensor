// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

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

} // namespace

TEST(AutogradDepsSingleLaneTest, DepsSchedulingPreventsEarlyEnqueueOnDuplicateEdges) {
  // model, node A can be enqueued and executed after receiving only ONE of its
  // two structural incoming edges (both edges map to the same input_nr slot).
  //
  // must not run until BOTH incoming edges have been routed.

  MtToggleRestore mt_guard;

  // Leaf and its AccumulateGrad sink.
  TensorImpl leaf = make_cpu_dense_f32({4}, 0.0f);
  auto* meta = vbt::autograd::get_autograd_meta(leaf, /*create_if_missing=*/true);
  ASSERT_NE(meta, nullptr);
  auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);

  // A forwards its (coalesced) grad to the leaf.
  std::vector<InputMeta> meta1 = {
      InputMeta{ScalarType::Float32, Device::cpu(), {4}, /*is_strided_dense=*/true}};
  auto A = vbt::core::make_intrusive<FunctionNode>(
      "A",
      meta1,
      [](std::vector<OptionalTensor>&& gin) {
        std::vector<OptionalTensor> out(1);
        if (!gin.empty()) out[0] = std::move(gin[0]);
        return out;
      });
  ensure_next_edges_sized(*A);
  A->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(acc.get()), 0};

  // P2 forwards its grad to A (duplicate edge to same slot 0).
  auto P2 = vbt::core::make_intrusive<FunctionNode>(
      "P2",
      meta1,
      [](std::vector<OptionalTensor>&& gin) {
        std::vector<OptionalTensor> out(1);
        if (!gin.empty()) out[0] = std::move(gin[0]);
        return out;
      });
  ensure_next_edges_sized(*P2);
  P2->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(A.get()), 0};

  // X forwards its grad to P2 (delays P2 until after A would have been enqueued).
  auto X = vbt::core::make_intrusive<FunctionNode>(
      "X",
      meta1,
      [](std::vector<OptionalTensor>&& gin) {
        std::vector<OptionalTensor> out(1);
        if (!gin.empty()) out[0] = std::move(gin[0]);
        return out;
      });
  ensure_next_edges_sized(*X);
  X->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(P2.get()), 0};

  // P1 forwards its grad to A (duplicate edge to same slot 0).
  auto P1 = vbt::core::make_intrusive<FunctionNode>(
      "P1",
      meta1,
      [](std::vector<OptionalTensor>&& gin) {
        std::vector<OptionalTensor> out(1);
        if (!gin.empty()) out[0] = std::move(gin[0]);
        return out;
      });
  ensure_next_edges_sized(*P1);
  P1->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(A.get()), 0};

  // Root produces grads to P1 and X.
  std::vector<InputMeta> meta2 = {
      InputMeta{ScalarType::Float32, Device::cpu(), {4}, /*is_strided_dense=*/true},
      InputMeta{ScalarType::Float32, Device::cpu(), {4}, /*is_strided_dense=*/true}};
  auto Root = vbt::core::make_intrusive<FunctionNode>(
      "Root",
      meta2,
      [](std::vector<OptionalTensor>&& gin) {
        std::vector<OptionalTensor> out(2);
        if (gin.size() > 0) out[0] = std::move(gin[0]);
        if (gin.size() > 1) out[1] = std::move(gin[1]);
        return out;
      });
  ensure_next_edges_sized(*Root);
  Root->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(P1.get()), 0};
  Root->next_edges[1] = vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(X.get()), 0};

  // Seed: two distinct grads.
  std::vector<OptionalTensor> seed(2);
  seed[0] = make_cpu_dense_f32({4}, 1.0f);  // via P1
  seed[1] = make_cpu_dense_f32({4}, 2.0f);  // via X -> P2

  EXPECT_NO_THROW(run_backward(intrusive_ptr<vbt::autograd::Node>(Root.get()), seed, {}));

  ASSERT_TRUE(meta->grad_ptr != nullptr && meta->grad_has);
  const TensorImpl& g = *meta->grad_ptr;
  const float* p = static_cast<const float*>(g.data());
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(p[i], 3.0f) << "index " << i;
  }
}
