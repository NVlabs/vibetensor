// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/view_ops.h"
#include "vbt/core/intrusive_ptr.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;

static vbt::core::StoragePtr S(void* base) {
  return vbt::core::make_intrusive<Storage>(DataPtr(base, [](void*) noexcept {}), 4096);
}

TEST(ViewOpsTest, SelectGeometryAndOOB) {
  auto st = S(reinterpret_cast<void*>(0x6000));
  TensorImpl t(st, {2,3}, {3,1}, 0, ScalarType::Float32, Device::cpu());
  // select dim 1, index 2 -> sizes [2], strides [3], offset += 2
  auto v = vbt::core::select(t, 1, 2);
  EXPECT_EQ(v.sizes(), std::vector<int64_t>({2}));
  EXPECT_EQ(v.strides(), std::vector<int64_t>({3}));
  EXPECT_EQ(v.storage_offset(), 2);

  // OOB message
  std::string msg;
  try { (void)vbt::core::select(t, 1, 3); FAIL() << "expected"; } catch (const std::out_of_range& e) { msg = e.what(); }
  EXPECT_NE(msg.find("select(): index 3 out of range"), std::string::npos);
}

TEST(ViewOpsTest, NarrowGeometryAndErrors) {
  auto st = S(reinterpret_cast<void*>(0x6100));
  TensorImpl t(st, {4}, {1}, 0, ScalarType::Float32, Device::cpu());
  auto v = vbt::core::narrow(t, 0, 1, 2);
  EXPECT_EQ(v.sizes(), std::vector<int64_t>({2}));
  EXPECT_EQ(v.strides(), std::vector<int64_t>({1}));
  EXPECT_EQ(v.storage_offset(), 1);

  // length negative
  std::string msg;
  try { (void)vbt::core::narrow(t, 0, 0, -1); FAIL() << "expected"; } catch (const std::invalid_argument& e) { msg = e.what(); }
  EXPECT_EQ(msg, std::string("narrow(): length must be non-negative."));
  // start out of range
  msg.clear();
  try { (void)vbt::core::narrow(t, 0, 5, 1); FAIL() << "expected"; } catch (const std::out_of_range& e) { msg = e.what(); }
  EXPECT_NE(msg.find("start out of range (expected to be in range of ["), std::string::npos);
}

TEST(ViewOpsTest, SqueezeUnsqueezePermuteTranspose) {
  auto st = S(reinterpret_cast<void*>(0x6200));
  TensorImpl t(st, {2,1,3}, {3,3,1}, 0, ScalarType::Float32, Device::cpu());
  auto s = vbt::core::squeeze(t);
  EXPECT_EQ(s.sizes(), (std::vector<int64_t>{2,3}));
  auto s1 = vbt::core::squeeze(t, 1);
  EXPECT_EQ(s1.sizes(), (std::vector<int64_t>{2,3}));
  auto u = vbt::core::unsqueeze(s1, 1);
  EXPECT_EQ(u.sizes(), (std::vector<int64_t>{2,1,3}));

  // permute dims
  auto p = vbt::core::permute(s1, std::vector<int64_t>{1,0});
  EXPECT_EQ(p.sizes(), (std::vector<int64_t>{3,2}));
  // transpose
  auto tr = vbt::core::transpose(s1, 0, 1);
  EXPECT_EQ(tr.sizes(), (std::vector<int64_t>{3,2}));
}
