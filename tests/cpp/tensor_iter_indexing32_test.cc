// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <array>

#include "vbt/core/tensor_iter.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::TensorIter;
using vbt::core::TensorIterConfig;
using vbt::core::OptionalTensorImplRef;
using vbt::core::kTensorIterMaxRank;
using vbt::core::testing::TensorIterTestHelper;

namespace {

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  void* base = nullptr;
  if (nbytes > 0) {
    base = ::operator new(nbytes);
  }
  return vbt::core::make_intrusive<Storage>(
      DataPtr(base, [](void* p) noexcept { ::operator delete(p); }), nbytes);
}

static TensorImpl make_contiguous_tensor(
    const std::vector<std::int64_t>& sizes,
    ScalarType dtype) {
  const std::size_t nd = sizes.size();
  std::vector<std::int64_t> strides(nd, 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(nd) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  std::int64_t ne = 1;
  bool any_zero = false;
  for (auto s : sizes) {
    if (s == 0) {
      any_zero = true;
      break;
    }
    ne *= s;
  }
  if (any_zero) {
    ne = 0;
  }

  const std::size_t item_b = vbt::core::itemsize(dtype);
  const std::size_t nbytes = static_cast<std::size_t>(ne) * item_b;
  auto storage = make_storage_bytes(nbytes);
  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0,
                    dtype, Device::cpu());
}

} // namespace

TEST(TensorIterIndexing32Test, Contiguous1DWithinInt32IsSafe) {
  std::vector<std::int64_t> sizes{128};
  TensorImpl out = make_contiguous_tensor(sizes, ScalarType::Float32);
  TensorImpl in  = make_contiguous_tensor(sizes, ScalarType::Float32);

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(in);
  cfg.check_mem_overlap(false);
  TensorIter iter = cfg.build();

  EXPECT_TRUE(iter.can_use_32bit_indexing());

  int calls = 0;
  iter.with_32bit_indexing([&](const TensorIter& sub) {
    ++calls;
    EXPECT_TRUE(sub.can_use_32bit_indexing());
    EXPECT_EQ(sub.numel(), iter.numel());
  });
  EXPECT_EQ(calls, 1);
}

TEST(TensorIterIndexing32Test, ZeroNumelIteratorsAreTriviallySafe) {
  std::int64_t sz[3] = {2, 0, 3};
  TensorIter iter = TensorIterTestHelper::make_iterator_for_shape_with_dummy_operand(
      std::span<const std::int64_t>(sz, 3));

  EXPECT_EQ(iter.numel(), 0);
  EXPECT_TRUE(iter.can_use_32bit_indexing());

  int calls = 0;
  iter.with_32bit_indexing([&](const TensorIter& sub) {
    ++calls;
    EXPECT_EQ(sub.numel(), 0);
    EXPECT_TRUE(sub.can_use_32bit_indexing());
  });
  EXPECT_EQ(calls, 1);
}

TEST(TensorIterIndexing32Test, NumelExceedingInt32IsNot32BitSafe) {
  const std::int64_t i32_max =
      static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
  std::int64_t sz[2] = {i32_max, 2};
  TensorIter iter = TensorIterTestHelper::make_iterator_for_shape(
      std::span<const std::int64_t>(sz, 2));

  EXPECT_FALSE(iter.can_use_32bit_indexing());

  int calls = 0;
  iter.with_32bit_indexing([&](const TensorIter&) { ++calls; });
  EXPECT_EQ(calls, 0);
}

TEST(TensorIterIndexing32Test, With32BitIndexingOnReductionsThrows) {
  std::vector<std::int64_t> in_sizes{2, 3};
  TensorImpl in = make_contiguous_tensor(in_sizes, ScalarType::Float32);

  std::vector<std::int64_t> out_sizes{2};  // reduce over dim=1, keepdim=false
  TensorImpl out = make_contiguous_tensor(out_sizes, ScalarType::Float32);

  std::array<std::int64_t, 1> dims{{1}};
  TensorIter iter = TensorIter::reduce_op(
      out, in, std::span<const std::int64_t>(dims.data(), dims.size()));

  EXPECT_TRUE(iter.is_reduction());
  EXPECT_THROW(
      iter.with_32bit_indexing([](const TensorIter&) {}),
      std::logic_error);
}
