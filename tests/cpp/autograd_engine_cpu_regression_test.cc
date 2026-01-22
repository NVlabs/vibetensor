// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
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

#if VBT_AUTOGRAD_TESTING
namespace vbt { namespace autograd {
void _test_clear_last_backward_snapshot() noexcept;
bool _test_last_backward_has_value() noexcept;
vbt::core::Device _test_last_backward_autograd_device() noexcept;
bool _test_last_backward_streaming_enabled_snapshot() noexcept;
void _test_last_backward_cuda_counters(std::uint64_t* recorded,
                                      std::uint64_t* waited,
                                      std::uint64_t* cross) noexcept;
}} // namespace vbt::autograd
#endif

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

struct ToggleRestore {
  bool prev;
  ToggleRestore() : prev(vbt::autograd::is_streaming_backwards_enabled()) {}
  ~ToggleRestore() { vbt::autograd::set_streaming_backwards_enabled(prev); }
};

} // namespace

#if VBT_AUTOGRAD_TESTING
TEST(AutogradCpuRegressionTest, CPUBackwardDoesNotTouchCudaCounters) {
  ToggleRestore restore;
  vbt::autograd::set_streaming_backwards_enabled(true);
  vbt::autograd::_test_clear_last_backward_snapshot();

  // Leaf and its AccumulateGrad sink.
  TensorImpl leaf = make_cpu_dense_f32({2}, 0.0f);
  auto* meta = vbt::autograd::get_autograd_meta(leaf, /*create_if_missing=*/true);
  auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);

  // Root FunctionNode that forwards its single input gradient.
  std::vector<InputMeta> metas = {
      InputMeta{ScalarType::Float32, Device::cpu(), {2}, /*is_strided_dense=*/true}};
  auto backward = [](std::vector<OptionalTensor>&& gin) {
    std::vector<OptionalTensor> out(1);
    if (!gin.empty()) {
      out[0] = std::move(gin[0]);
    }
    return out;
  };
  auto node = vbt::core::make_intrusive<FunctionNode>("PassCpuRegression", metas, backward);
  ensure_next_edges_sized(*node);
  node->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(acc.get()), 0};

  // Seed gradient into root.
  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);

  run_backward(vbt::core::intrusive_ptr<vbt::autograd::Node>(node.get()), seed, {});

  ASSERT_TRUE(vbt::autograd::_test_last_backward_has_value());
  EXPECT_EQ(vbt::autograd::_test_last_backward_autograd_device(), Device::cpu(0));
  EXPECT_TRUE(vbt::autograd::_test_last_backward_streaming_enabled_snapshot());

  std::uint64_t recorded = 0;
  std::uint64_t waited = 0;
  std::uint64_t cross = 0;
  vbt::autograd::_test_last_backward_cuda_counters(&recorded, &waited, &cross);
  EXPECT_EQ(recorded, 0u);
  EXPECT_EQ(waited, 0u);
  EXPECT_EQ(cross, 0u);
}
#else
TEST(AutogradCpuRegressionTest, CPUBackwardDoesNotTouchCudaCounters) {
  GTEST_SKIP() << "VBT_AUTOGRAD_TESTING is disabled";
}
#endif
