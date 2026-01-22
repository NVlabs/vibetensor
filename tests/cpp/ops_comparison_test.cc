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
// CUDA registration is handled internally or via dlsym usually, but here we might need to link it.
// In tests, usually `vbt_register_default_kernels` is enough for CPU.
// For CUDA, we rely on the build system linking the CUDA library or the test setup.

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

std::vector<bool> get_bool_data(const TensorImpl& t) {
  if (t.dtype() != ScalarType::Bool) throw std::runtime_error("Not bool");
  const bool* ptr = static_cast<const bool*>(t.data());
  int64_t numel = t.numel();
  return std::vector<bool>(ptr, ptr + numel);
}

TensorImpl call_binary_op(const std::string& op, const TensorImpl& a, const TensorImpl& b) {
  BoxedStack stack;
  stack.push_back(a);
  stack.push_back(b);
  Dispatcher::instance().callBoxed(op, stack);
  return stack.back();
}

class OpsComparisonTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    vbt_register_default_kernels();
  }
};

TEST_F(OpsComparisonTest, Eq) {
  auto a = make_float_tensor({4}, {1, 2, 3, 4});
  auto b = make_float_tensor({4}, {1, 3, 3, 5});
  
  auto out = call_binary_op("vt::eq", a, b);
  auto data = get_bool_data(out);
  
  std::vector<bool> expected = {true, false, true, false};
  EXPECT_EQ(data, expected);
}

TEST_F(OpsComparisonTest, Ne) {
  auto a = make_float_tensor({4}, {1, 2, 3, 4});
  auto b = make_float_tensor({4}, {1, 3, 3, 5});
  
  auto out = call_binary_op("vt::ne", a, b);
  auto data = get_bool_data(out);
  
  std::vector<bool> expected = {false, true, false, true};
  EXPECT_EQ(data, expected);
}

TEST_F(OpsComparisonTest, Lt) {
  auto a = make_float_tensor({4}, {1, 2, 3, 4});
  auto b = make_float_tensor({4}, {2, 2, 2, 2});
  
  auto out = call_binary_op("vt::lt", a, b);
  auto data = get_bool_data(out);
  
  std::vector<bool> expected = {true, false, false, false};
  EXPECT_EQ(data, expected);
}

TEST_F(OpsComparisonTest, Gt) {
  auto a = make_float_tensor({4}, {1, 2, 3, 4});
  auto b = make_float_tensor({4}, {2, 2, 2, 2});
  
  auto out = call_binary_op("vt::gt", a, b);
  auto data = get_bool_data(out);
  
  std::vector<bool> expected = {false, false, true, true};
  EXPECT_EQ(data, expected);
}

TEST_F(OpsComparisonTest, Le) {
  auto a = make_float_tensor({4}, {1, 2, 3, 4});
  auto b = make_float_tensor({4}, {2, 2, 2, 2});
  
  auto out = call_binary_op("vt::le", a, b);
  auto data = get_bool_data(out);
  
  std::vector<bool> expected = {true, true, false, false};
  EXPECT_EQ(data, expected);
}

TEST_F(OpsComparisonTest, Ge) {
  auto a = make_float_tensor({4}, {1, 2, 3, 4});
  auto b = make_float_tensor({4}, {2, 2, 2, 2});
  
  auto out = call_binary_op("vt::ge", a, b);
  auto data = get_bool_data(out);
  
  std::vector<bool> expected = {false, true, true, true};
  EXPECT_EQ(data, expected);
}

TEST_F(OpsComparisonTest, Broadcast) {
  // [2, 2] vs [2]
  // 1 2   vs  2 3
  // 3 4
  
  auto a = make_float_tensor({2, 2}, {1, 2, 3, 4});
  auto b = make_float_tensor({2}, {2, 3});
  
  // Row 0: 1vs2, 2vs3 -> T, T (lt)
  // Row 1: 3vs2, 4vs3 -> F, F (lt)
  
  auto out = call_binary_op("vt::lt", a, b);
  auto data = get_bool_data(out);
  
  std::vector<bool> expected = {true, true, false, false};
  EXPECT_EQ(data, expected);
}

TEST_F(OpsComparisonTest, Int64) {
  auto a = make_int_tensor({3}, {1, 2, 3});
  auto b = make_int_tensor({3}, {1, 3, 2});
  
  auto out = call_binary_op("vt::lt", a, b);
  auto data = get_bool_data(out);
  // 1<1 (F), 2<3 (T), 3<2 (F)
  std::vector<bool> expected = {false, true, false};
  EXPECT_EQ(data, expected);
}

TEST_F(OpsComparisonTest, UnsupportedDtypeThrows) {
  // Create Int8 tensors (assuming Int8 is not supported by default_kernels comparison)
  std::vector<int64_t> sizes = {1};
  std::vector<int64_t> strides = {1};
  size_t nbytes = 1;
  void* raw1 = ::operator new(nbytes);
  void* raw2 = ::operator new(nbytes);
  DataPtr dp1(raw1, [](void* p) noexcept { ::operator delete(p); });
  DataPtr dp2(raw2, [](void* p) noexcept { ::operator delete(p); });
  auto s1 = vbt::core::make_intrusive<Storage>(std::move(dp1), nbytes);
  auto s2 = vbt::core::make_intrusive<Storage>(std::move(dp2), nbytes);
  
  // Int32 is not supported in default kernels (only Float32/Int64)
  TensorImpl a(s1, sizes, strides, 0, ScalarType::Int32, Device::cpu());
  TensorImpl b(s2, sizes, strides, 0, ScalarType::Int32, Device::cpu());

  EXPECT_THROW({
    call_binary_op("vt::eq", a, b);
  }, std::invalid_argument);
}

} // namespace
