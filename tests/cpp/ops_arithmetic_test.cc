// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cstring>

#include "vbt/dispatch/dispatcher.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/intrusive_ptr.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::dispatch::Dispatcher;
using vbt::dispatch::BoxedStack;

extern "C" void vbt_register_default_kernels();

// Declarations of kernels to test directly (must match src/vbt/ops/default_kernels.cc)
// Defined in global namespace in default_kernels.cc (non-static)
using vbt::core::TensorImpl;
extern TensorImpl vbt_default_sum_impl(const TensorImpl& self, std::vector<int64_t> dims, bool keepdim);
extern TensorImpl vbt_default_mean_impl(const TensorImpl& self, std::vector<int64_t> dims, bool keepdim);

namespace {

TensorImpl make_float_tensor(const std::vector<int64_t>& sizes, const std::vector<float>& data) {
  int64_t numel = 1;
  for (auto s : sizes) numel *= s;
  
  if (static_cast<size_t>(numel) != data.size()) {
    throw std::runtime_error("Data size mismatch");
  }

  size_t nbytes = numel * sizeof(float);
  void* raw = ::operator new(nbytes);
  std::memcpy(raw, data.data(), nbytes);
  
  DataPtr dp(raw, [](void* p) noexcept { ::operator delete(p); });
  auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
  
  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
    strides[i] = acc;
    acc *= sizes[i];
  }

  return TensorImpl(storage, sizes, strides, 0, ScalarType::Float32, Device::cpu());
}

std::vector<float> get_float_data(const TensorImpl& t) {
  if (t.dtype() != ScalarType::Float32) throw std::runtime_error("Not float");
  const float* ptr = static_cast<const float*>(t.data());
  int64_t numel = t.numel();
  return std::vector<float>(ptr, ptr + numel);
}

TensorImpl call_binary_op(const std::string& op, const TensorImpl& a, const TensorImpl& b) {
  BoxedStack stack;
  stack.push_back(a);
  stack.push_back(b);
  Dispatcher::instance().callBoxed(op, stack);
  return stack.back();
}

TensorImpl call_unary_op(const std::string& op, const TensorImpl& a) {
  BoxedStack stack;
  stack.push_back(a);
  Dispatcher::instance().callBoxed(op, stack);
  return stack.back();
}

class OpsArithmeticTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    vbt_register_default_kernels();
  }
};

TEST_F(OpsArithmeticTest, AddBroadcast) {
  // [2, 3] + [3] -> [2, 3]
  auto a = make_float_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  auto b = make_float_tensor({3}, {10, 20, 30});
  
  auto out = call_binary_op("vt::add", a, b);
  auto data = get_float_data(out);
  
  std::vector<float> expected = {11, 22, 33, 14, 25, 36};
  EXPECT_EQ(data, expected);
}

TEST_F(OpsArithmeticTest, Sub) {
  auto a = make_float_tensor({2, 2}, {10, 20, 30, 40});
  auto b = make_float_tensor({2, 2}, {1, 2, 3, 4});
  
  auto out = call_binary_op("vt::sub", a, b);
  auto data = get_float_data(out);
  
  std::vector<float> expected = {9, 18, 27, 36};
  EXPECT_EQ(data, expected);
}

TEST_F(OpsArithmeticTest, MulBroadcast) {
  // [2, 1] * [3] -> [2, 3]
  auto a = make_float_tensor({2, 1}, {2, 3});
  auto b = make_float_tensor({3}, {10, 20, 30});
  
  auto out = call_binary_op("vt::mul", a, b);
  auto data = get_float_data(out);
  
  // Row 0: 2 * {10, 20, 30} = {20, 40, 60}
  // Row 1: 3 * {10, 20, 30} = {30, 60, 90}
  std::vector<float> expected = {20, 40, 60, 30, 60, 90};
  EXPECT_EQ(data, expected);
}

TEST_F(OpsArithmeticTest, Div) {
  auto a = make_float_tensor({4}, {10, 20, 30, 40});
  auto b = make_float_tensor({4}, {2, 4, 5, 8});
  
  auto out = call_binary_op("vt::div", a, b);
  auto data = get_float_data(out);
  
  std::vector<float> expected = {5, 5, 6, 5};
  EXPECT_EQ(data, expected);
}

TEST_F(OpsArithmeticTest, Abs) {
  auto a = make_float_tensor({3}, {-1, 0, 1});
  auto out = call_unary_op("vt::abs", a);
  auto data = get_float_data(out);
  
  std::vector<float> expected = {1, 0, 1};
  EXPECT_EQ(data, expected);
}

TEST_F(OpsArithmeticTest, Neg) {
  auto a = make_float_tensor({3}, {-1, 0, 1});
  auto out = call_unary_op("vt::neg", a);
  auto data = get_float_data(out);
  
  std::vector<float> expected = {1, 0, -1};
  EXPECT_EQ(data, expected); // 0 can be -0 but float eq check usually handles it
  EXPECT_EQ(data[1], 0.0f); // explicitly check 0
}

TEST_F(OpsArithmeticTest, Reciprocal) {
  auto a = make_float_tensor({2}, {2, 4});
  auto out = call_unary_op("vt::reciprocal", a);
  auto data = get_float_data(out);
  
  std::vector<float> expected = {0.5, 0.25};
  EXPECT_EQ(data, expected);
}

TEST_F(OpsArithmeticTest, SubScalarBroadcast) {
  // [2, 2] - [1]
  auto a = make_float_tensor({2, 2}, {10, 20, 30, 40});
  auto b = make_float_tensor({1}, {5});
  
  auto out = call_binary_op("vt::sub", a, b);
  auto data = get_float_data(out);
  
  std::vector<float> expected = {5, 15, 25, 35};
  EXPECT_EQ(data, expected);
}

TensorImpl make_int_tensor(const std::vector<int64_t>& sizes, const std::vector<int64_t>& data) {
  int64_t numel = 1;
  for (auto s : sizes) numel *= s;
  
  if (static_cast<size_t>(numel) != data.size()) {
    throw std::runtime_error("Data size mismatch");
  }

  size_t nbytes = numel * sizeof(int64_t);
  void* raw = ::operator new(nbytes);
  std::memcpy(raw, data.data(), nbytes);
  
  DataPtr dp(raw, [](void* p) noexcept { ::operator delete(p); });
  auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
  
  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
    strides[i] = acc;
    acc *= sizes[i];
  }

  return TensorImpl(storage, sizes, strides, 0, ScalarType::Int64, Device::cpu());
}

TEST_F(OpsArithmeticTest, Int64DivByZero) {
  auto a = make_int_tensor({1}, {10});
  auto b = make_int_tensor({1}, {0});
  
  EXPECT_THROW({
    call_binary_op("vt::div", a, b);
  }, std::runtime_error);
}

TEST_F(OpsArithmeticTest, SumAll) {
  auto a = make_float_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  
  TensorImpl out = vbt_default_sum_impl(a, {}, false);
  
  auto data = get_float_data(out);
  EXPECT_EQ(data.size(), 1);
  EXPECT_FLOAT_EQ(data[0], 21.0f);
}

TEST_F(OpsArithmeticTest, MeanAll) {
  auto a = make_float_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  
  TensorImpl out = vbt_default_mean_impl(a, {}, false);
  
  auto data = get_float_data(out);
  EXPECT_EQ(data.size(), 1);
  EXPECT_FLOAT_EQ(data[0], 3.5f); // 21 / 6
}

TEST_F(OpsArithmeticTest, MeanEmpty) {
  // [2, 0] tensor
  auto a = make_float_tensor({2, 0}, {}); 
  
  // Mean over dim 1 (size 0) -> should be NaN for float
  TensorImpl out = vbt_default_mean_impl(a, std::vector<int64_t>{1}, false);
  
  auto data = get_float_data(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::isnan(data[0]));
  EXPECT_TRUE(std::isnan(data[1]));
}


} // namespace
