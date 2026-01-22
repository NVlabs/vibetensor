// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "vbt/core/tensor_iter.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/checked_math.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::TensorIter;
using vbt::core::TensorIterConfig;
using vbt::core::OptionalTensorImplRef;
using vbt::core::DeviceStrideMeta;
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

static TensorImpl make_contiguous_tensor(const std::vector<int64_t>& sizes) {
  const std::size_t nd = sizes.size();
  std::vector<int64_t> strides(nd, 0);

  // Build strides from the back, treating zero-size dims as having extent 1
  // and clamping on overflow to avoid UB in this test-only helper.
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(nd) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes[idx];
    const std::int64_t step = (sz == 0 ? 1 : sz);
    std::int64_t tmp = 0;
    if (!vbt::core::checked_mul_i64(acc, step, tmp)) {
      // On overflow, stop growing strides; remaining entries keep the last
      // representable value. This is sufficient for rank/shape tests.
      break;
    }
    acc = tmp;
  }

  std::int64_t ne = 1;
  bool any_zero = false;
  bool overflow = false;
  for (auto s : sizes) {
    if (s == 0) {
      any_zero = true;
      break;
    }
    std::int64_t tmp = 0;
    if (!vbt::core::checked_mul_i64(ne, s, tmp)) {
      overflow = true;
      break;
    }
    ne = tmp;
  }
  if (any_zero || overflow) {
    ne = 0;
  }

  const std::size_t nbytes = static_cast<std::size_t>(ne) * sizeof(float);
  auto storage = make_storage_bytes(nbytes);
  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0,
                    ScalarType::Float32, Device::cpu());
}

}  // namespace

TEST(TensorIterNumelMaxRankTest, NumelZeroAndOverflowSentinel) {
  // Zero-size dim => numel() == 0.
  {
    std::int64_t sz[3] = {2, 0, 3};
    TensorIter iter = TensorIterTestHelper::make_iterator_for_shape(
        std::span<const std::int64_t>(sz, 3));
    EXPECT_EQ(iter.ndim(), 3);
    EXPECT_EQ(iter.numel(), 0);
  }

  // Overflow in product => numel() == 0 sentinel.
  {
    const std::int64_t big =
        std::numeric_limits<std::int64_t>::max() / 2 + 1;
    std::int64_t sz[2] = {big, 4};
    TensorIter iter = TensorIterTestHelper::make_iterator_for_shape(
        std::span<const std::int64_t>(sz, 2));
    EXPECT_EQ(iter.ndim(), 2);
    EXPECT_EQ(iter.numel(), 0);
  }
}

TEST(TensorIterNumelMaxRankTest, ExportDeviceMetaZeroNumelShapes) {
  // Zero-size dim case.
  {
    std::int64_t sz[3] = {2, 0, 3};
    TensorIter iter = TensorIterTestHelper::make_iterator_for_shape_with_dummy_operand(
        std::span<const std::int64_t>(sz, 3));

    DeviceStrideMeta meta{};
    EXPECT_NO_THROW(iter.export_device_meta(0, &meta));
    EXPECT_EQ(meta.ndim, iter.ndim());
    for (int d = 0; d < meta.ndim; ++d) {
      EXPECT_EQ(meta.sizes[d], iter.shape()[static_cast<std::size_t>(d)]);
    }
  }

  // Overflow sentinel case: numel()==0 but sizes are non-zero.
  {
    const std::int64_t big =
        std::numeric_limits<std::int64_t>::max() / 2 + 1;
    std::int64_t sz[2] = {big, 4};
    TensorIter iter = TensorIterTestHelper::make_iterator_for_shape_with_dummy_operand(
        std::span<const std::int64_t>(sz, 2));

    DeviceStrideMeta meta{};
    EXPECT_NO_THROW(iter.export_device_meta(0, &meta));
    EXPECT_EQ(meta.ndim, iter.ndim());
    for (int d = 0; d < meta.ndim; ++d) {
      EXPECT_EQ(meta.sizes[d], iter.shape()[static_cast<std::size_t>(d)]);
    }
  }
}

TEST(TensorIterNumelMaxRankTest, BuildIteratorRankBeyondMaxRankThrows) {
  // Shape with rank kTensorIterMaxRank + 1 but zero-numel so storage is small.
  std::vector<std::int64_t> sizes(static_cast<std::size_t>(kTensorIterMaxRank) + 1,
                                  2);
  sizes[0] = 0;  // zero-size dim keeps numel()==0 but contributes to rank.

  TensorImpl out = make_contiguous_tensor(sizes);
  TensorImpl in  = make_contiguous_tensor(sizes);

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(in);
  cfg.check_mem_overlap(false);
  cfg.set_max_rank(kTensorIterMaxRank);

  bool threw = false;
  try {
    (void)cfg.build();
  } catch (const std::invalid_argument& e) {
    threw = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("iteration rank exceeds max_rank"), std::string::npos);
  }
  EXPECT_TRUE(threw);
}

TEST(TensorIterNumelMaxRankTest, SetMaxRankOutOfRangeRejects) {
  std::vector<std::int64_t> sizes{2, 3};
  TensorImpl out = make_contiguous_tensor(sizes);
  TensorImpl in  = make_contiguous_tensor(sizes);

  // max_rank < 1
  {
    TensorIterConfig cfg;
    cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
    cfg.add_input(in);
    cfg.check_mem_overlap(false);
    cfg.set_max_rank(0);
    EXPECT_THROW((void)cfg.build(), std::invalid_argument);
  }

  // max_rank > kTensorIterMaxRank
  {
    TensorIterConfig cfg;
    cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
    cfg.add_input(in);
    cfg.check_mem_overlap(false);
    cfg.set_max_rank(static_cast<std::int64_t>(kTensorIterMaxRank) + 1);
    EXPECT_THROW((void)cfg.build(), std::invalid_argument);
  }
}
