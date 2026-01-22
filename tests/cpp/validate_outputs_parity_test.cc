// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>
#include <string>

#include "vbt/autograd/engine.h"
#include "vbt/autograd/function.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/device.h"

using vbt::autograd::OptionalTensor;
using vbt::autograd::FunctionNode;
using vbt::autograd::InputMeta;
using vbt::autograd::run_backward;
using vbt::autograd::ensure_next_edges_sized;
using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;

namespace {
static TensorImpl make_cpu_dense_f32(const std::vector<int64_t>& sizes, float fill) {
  std::size_t ne = 1; for (auto s : sizes) ne *= static_cast<std::size_t>(s == 0 ? 1 : s);
  std::size_t nbytes = ne * sizeof(float);
  void* buf = nullptr; if (nbytes > 0) buf = ::operator new(nbytes);
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  StoragePtr st = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1; for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) { strides[static_cast<std::size_t>(i)] = acc; acc *= (sizes[static_cast<std::size_t>(i)] == 0 ? 1 : sizes[static_cast<std::size_t>(i)]); }
  TensorImpl t(st, sizes, strides, 0, vbt::core::ScalarType::Float32, vbt::core::Device::cpu());
  float* p = static_cast<float*>(t.data());
  for (std::size_t i = 0; i < ne; ++i) p[i] = fill;
  return t;
}

static TensorImpl make_cpu_nondense_f32_view_like(const TensorImpl& base) {
  // Create a simple as_strided view with different strides if possible
  std::vector<int64_t> sizes = base.sizes();
  std::vector<int64_t> strides = base.strides();
  if (sizes.empty()) return base; // scalar remains dense by definition
  if (sizes.size() == 1) {
    // 1D: use stride 2 and extend storage
    std::vector<int64_t> ns = sizes; ns[0] = 1; // still 1 elem
    std::vector<int64_t> st = strides; st[0] = strides[0] + 1; // wrong stride
    return base.as_strided(ns, st, base.storage_offset());
  }
  // Multi-d: swap last two strides
  std::swap(strides[strides.size()-1], strides[strides.size()-2]);
  return base.as_strided(sizes, strides, base.storage_offset());
}
}

TEST(ValidateOutputs, WrongNumberOfGradients) {
  // Meta for two inputs
  std::vector<InputMeta> metas = {
    InputMeta{vbt::core::ScalarType::Float32, vbt::core::Device::cpu(), {2}, true},
    InputMeta{vbt::core::ScalarType::Float32, vbt::core::Device::cpu(), {2}, true},
  };
  // Backward returns only one gradient to trigger count mismatch
  auto backward = [](std::vector<OptionalTensor>&& /*gin*/) {
    std::vector<OptionalTensor> out(1);
    out[0] = make_cpu_dense_f32({2}, 1.0f);
    return out;
  };
  auto node = vbt::core::make_intrusive<FunctionNode>("TestFn", metas, backward);
  ensure_next_edges_sized(*node);

  // Seed grads (size must match num_inputs for root)
  std::vector<OptionalTensor> seed(2);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);
  seed[1] = make_cpu_dense_f32({2}, 1.0f);

  try {
    run_backward(vbt::core::intrusive_ptr<vbt::autograd::Node>(node.get()), seed, {});
    FAIL() << "Expected runtime_error";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("wrong number of gradients"), std::string::npos) << msg;
  }
}

TEST(ValidateOutputs, DtypeMismatch) {
  std::vector<InputMeta> metas = {
    InputMeta{vbt::core::ScalarType::Float32, vbt::core::Device::cpu(), {2}, true},
  };
  auto backward = [](std::vector<OptionalTensor>&& /*gin*/) {
    std::vector<OptionalTensor> out(1);
    // Intentionally wrong dtype
    // Build Int64 tensor
    std::size_t ne = 2; std::size_t nbytes = ne * sizeof(long long);
    void* buf = ::operator new(nbytes);
    DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
    StoragePtr st = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
    std::vector<int64_t> sizes = {2}; std::vector<int64_t> strides = {1};
    TensorImpl t(st, sizes, strides, 0, vbt::core::ScalarType::Int64, vbt::core::Device::cpu());
    out[0] = t;
    return out;
  };
  auto node = vbt::core::make_intrusive<FunctionNode>("TestFn", metas, backward);
  ensure_next_edges_sized(*node);
  std::vector<OptionalTensor> seed(1); seed[0] = make_cpu_dense_f32({2}, 1.0f);
  try {
    run_backward(vbt::core::intrusive_ptr<vbt::autograd::Node>(node.get()), seed, {});
    FAIL() << "Expected runtime_error";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("dtype mismatch"), std::string::npos) << msg;
  }
}

TEST(ValidateOutputs, SizesMismatch) {
  std::vector<InputMeta> metas = {
    InputMeta{vbt::core::ScalarType::Float32, vbt::core::Device::cpu(), {2,2}, true},
  };
  auto backward = [](std::vector<OptionalTensor>&& /*gin*/) {
    std::vector<OptionalTensor> out(1);
    out[0] = make_cpu_dense_f32({4}, 1.0f); // wrong shape
    return out;
  };
  auto node = vbt::core::make_intrusive<FunctionNode>("TestFn", metas, backward);
  ensure_next_edges_sized(*node);
  std::vector<OptionalTensor> seed(1); seed[0] = make_cpu_dense_f32({2,2}, 1.0f);
  try {
    run_backward(vbt::core::intrusive_ptr<vbt::autograd::Node>(node.get()), seed, {});
    FAIL() << "Expected runtime_error";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("sizes mismatch"), std::string::npos) << msg;
  }
}

TEST(ValidateOutputs, LayoutMismatchWhenDense) {
  TensorImpl base = make_cpu_dense_f32({4,4}, 1.0f);
  std::vector<InputMeta> metas = {
    InputMeta{vbt::core::ScalarType::Float32, vbt::core::Device::cpu(), {2,2}, /*dense*/true},
  };
  auto backward = [base](std::vector<OptionalTensor>&& /*gin*/) {
    std::vector<OptionalTensor> out(1);
    // Create a non-dense 2x2 view into a 4x4 base by skipping rows/cols
    std::vector<int64_t> sizes{2,2};
    std::vector<int64_t> strides{2,4}; // step over elements
    TensorImpl v = base.as_strided(sizes, strides, /*storage_offset=*/0);
    out[0] = v;
    return out;
  };
  auto node = vbt::core::make_intrusive<FunctionNode>("TestFn", metas, backward);
  ensure_next_edges_sized(*node);
  std::vector<OptionalTensor> seed(1); seed[0] = base;
  try {
    run_backward(vbt::core::intrusive_ptr<vbt::autograd::Node>(node.get()), seed, {});
    FAIL() << "Expected runtime_error";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("layout mismatch"), std::string::npos) << msg;
  }
}

TEST(ValidateOutputs, DeviceMismatchAndZeroDimAllowance) {
  // Expect CUDA device in meta, but return CPU grads
  std::vector<InputMeta> metas = {
    InputMeta{vbt::core::ScalarType::Float32, vbt::core::Device::cuda(0), /*0-d*/{}, true},
    InputMeta{vbt::core::ScalarType::Float32, vbt::core::Device::cuda(0), {2}, true},
  };
  auto backward = [](std::vector<OptionalTensor>&& /*gin*/) {
    std::vector<OptionalTensor> out(2);
    // Slot 0: 0-d CPU grad allowed
    TensorImpl scalar_cpu = make_cpu_dense_f32(/*sizes=*/{}, 1.0f);
    out[0] = scalar_cpu;
    // Slot 1: non-scalar CPU grad -> should trigger device mismatch
    out[1] = make_cpu_dense_f32({2}, 1.0f);
    return out;
  };
  auto node = vbt::core::make_intrusive<FunctionNode>("TestFn", metas, backward);
  ensure_next_edges_sized(*node);
  std::vector<OptionalTensor> seed(2);
  seed[0] = make_cpu_dense_f32({}, 1.0f);
  seed[1] = make_cpu_dense_f32({2}, 1.0f);
  try {
    run_backward(vbt::core::intrusive_ptr<vbt::autograd::Node>(node.get()), seed, {});
    FAIL() << "Expected runtime_error";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("device mismatch"), std::string::npos) << msg;
    // Should mention both cuda:0 and cpu:0
    EXPECT_NE(msg.find("cuda:0"), std::string::npos) << msg;
    EXPECT_NE(msg.find("cpu:0"), std::string::npos) << msg;
  }
}
