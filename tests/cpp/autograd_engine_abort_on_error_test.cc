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

TEST(AutogradAbortOnErrorTest, ThrowsWithoutDeadlocking) {
  MtToggleRestore mt_guard;

  TensorImpl leaf = make_cpu_dense_f32({4}, 0.0f);
  auto* meta = vbt::autograd::get_autograd_meta(leaf, /*create_if_missing=*/true);
  ASSERT_NE(meta, nullptr);
  auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);

  std::vector<InputMeta> meta1 = {
      InputMeta{ScalarType::Float32, Device::cpu(), {4}, /*is_strided_dense=*/true}};

  auto Boom = vbt::core::make_intrusive<FunctionNode>(
      "Boom",
      meta1,
      [](std::vector<OptionalTensor>&&) -> std::vector<OptionalTensor> {
        throw std::runtime_error("boom");
      });
  ensure_next_edges_sized(*Boom);
  Boom->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(acc.get()), 0};

  auto Root = vbt::core::make_intrusive<FunctionNode>(
      "Root",
      meta1,
      [](std::vector<OptionalTensor>&& gin) {
        std::vector<OptionalTensor> out(1);
        if (!gin.empty()) out[0] = std::move(gin[0]);
        return out;
      });
  ensure_next_edges_sized(*Root);
  Root->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(Boom.get()), 0};

  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cpu_dense_f32({4}, 1.0f);

  EXPECT_ANY_THROW(run_backward(intrusive_ptr<vbt::autograd::Node>(Root.get()), seed, {}));
}
