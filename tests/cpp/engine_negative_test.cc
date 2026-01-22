// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>
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
static TensorImpl make_cpu_dense_i64(const std::vector<int64_t>& sizes, long long fill) {
  std::size_t ne = 1; for (auto s : sizes) ne *= static_cast<std::size_t>(s == 0 ? 1 : s);
  std::size_t nbytes = ne * sizeof(long long);
  void* buf = nullptr; if (nbytes > 0) buf = ::operator new(nbytes);
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  StoragePtr st = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1; for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) { strides[static_cast<std::size_t>(i)] = acc; acc *= (sizes[static_cast<std::size_t>(i)] == 0 ? 1 : sizes[static_cast<std::size_t>(i)]); }
  TensorImpl t(st, sizes, strides, 0, vbt::core::ScalarType::Int64, vbt::core::Device::cpu());
  auto* p = static_cast<long long*>(t.data());
  for (std::size_t i = 0; i < ne; ++i) p[i] = fill;
  return t;
}

struct PassThrough2 final : Node {
  uint32_t num_inputs() const noexcept override { return 2; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    return std::move(grads_in);
  }
};
} // anonymous

TEST(AutogradEngineEdgeCasesTest, RejectsInt64Accumulation) {
  TensorImpl g = make_cpu_dense_i64({2}, 1);
  auto* meta = vbt::autograd::get_autograd_meta(g, /*create_if_missing=*/true);
  auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);
  auto root = vbt::core::make_intrusive<PassThrough2>();
  root->next_edges.push_back(vbt::autograd::Edge{vbt::core::intrusive_ptr<Node>(acc.get()), 0});
  root->next_edges.push_back(vbt::autograd::Edge{vbt::core::intrusive_ptr<Node>(acc.get()), 0});
  std::vector<OptionalTensor> seed(2);
  seed[0] = g; seed[1] = g;
  EXPECT_THROW(run_backward(vbt::core::intrusive_ptr<Node>(root.get()), seed, {}), std::runtime_error);
}
