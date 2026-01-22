// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/overlap.h"
#include "vbt/core/intrusive_ptr.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::MemOverlap;
using vbt::core::MemOverlapStatus;
using vbt::core::has_internal_overlap;
using vbt::core::get_overlap_status;
using vbt::core::assert_no_internal_overlap;
using vbt::core::assert_no_partial_overlap;
using vbt::core::assert_no_overlap;

static vbt::core::StoragePtr S(void* base) {
  return vbt::core::make_intrusive<Storage>(DataPtr(base, [](void*) noexcept {}), 4096);
}

TEST(OverlapPolicyTest, InternalOverlapMessage) {
  auto st = S(reinterpret_cast<void*>(0x7000));
  // size>1 with stride==0 triggers internal overlap
  TensorImpl t(st, {3}, {0}, 0, ScalarType::Float32, Device::cpu());
  std::string msg;
  try {
    assert_no_internal_overlap(t);
    FAIL() << "expected throw";
  } catch (const std::invalid_argument& e) {
    msg = e.what();
  }
  EXPECT_EQ(msg, std::string("unsupported operation: more than one element of the written-to tensor refers to a single memory location. Please clone() the tensor before performing the operation."));
}

TEST(OverlapPolicyTest, GetOverlapStatusMatrix) {
  auto st = S(reinterpret_cast<void*>(0x7100));
  TensorImpl a(st, {4}, {1}, 0, ScalarType::Float32, Device::cpu());
  TensorImpl b(st, {2,2}, {2,1}, 0, ScalarType::Float32, Device::cpu());
  TensorImpl c(st, {2,2}, {2,1}, 0, ScalarType::Float32, Device::cpu());
  TensorImpl d(st, {4,1}, {1,1}, 4, ScalarType::Float32, Device::cpu()); // adjacent [16,32)

  // Partial: same span, different strides
  EXPECT_EQ(get_overlap_status(a, b), MemOverlapStatus::Partial);
  // Full: exact alias (equal span and equal strides)
  EXPECT_EQ(get_overlap_status(b, c), MemOverlapStatus::Full);
  // No: adjacent
  EXPECT_EQ(get_overlap_status(a, d), MemOverlapStatus::No);

  // TooHard: non-NO&D input
  TensorImpl nd(st, {2,3}, {3,-1}, 0, ScalarType::Float32, Device::cpu());
  EXPECT_EQ(get_overlap_status(nd, a), MemOverlapStatus::TooHard);
}

TEST(OverlapPolicyTest, PartialAndFullMessages) {
  auto st = S(reinterpret_cast<void*>(0x7200));
  TensorImpl a(st, {4}, {1}, 0, ScalarType::Float32, Device::cpu());
  TensorImpl b(st, {2,2}, {2,1}, 0, ScalarType::Float32, Device::cpu());

  // Partial overlap message
  std::string msg;
  try {
    assert_no_partial_overlap(a, b);
    FAIL() << "expected throw";
  } catch (const std::invalid_argument& e) {
    msg = e.what();
  }
  EXPECT_EQ(msg, std::string("unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location. Please clone() the tensor before performing the operation."));

  // Full overlap message through assert_no_overlap
  TensorImpl c(st, {2,2}, {2,1}, 0, ScalarType::Float32, Device::cpu());
  msg.clear();
  try {
    assert_no_overlap(b, c);
    FAIL() << "expected throw";
  } catch (const std::invalid_argument& e) {
    msg = e.what();
  }
  EXPECT_EQ(msg, std::string("unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location. Please clone() the tensor before performing the operation."));
}
