// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vbt/core/indexing.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/indexing/index_errors.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::indexing::IndexKind;
using vbt::core::indexing::IndexSpec;
using vbt::core::indexing::TensorIndex;
using vbt::core::indexing::Slice;
using vbt::core::indexing::NormalizedSlice;
using vbt::core::indexing::normalize_slice;
using vbt::core::indexing::has_any_advanced;
using vbt::core::indexing::expand_ellipsis_and_validate;
using vbt::core::indexing::basic_index;
using vbt::core::indexing::basic_index_put;
using vbt::core::indexing::errors::kErrInvalidZeroDim;

static vbt::core::StoragePtr make_storage_bytes(std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(
      DataPtr(::operator new(nbytes), [](void* p) noexcept { ::operator delete(p); }),
      nbytes);
}

TEST(IndexingBasicTest, HasAnyAdvancedDetectsBooleanAndTensorKinds) {
  IndexSpec spec;
  EXPECT_FALSE(has_any_advanced(spec));

  spec.items.emplace_back(TensorIndex(nullptr));  // None
  EXPECT_FALSE(has_any_advanced(spec));

  spec.items.emplace_back(TensorIndex(true));  // Boolean advanced
  EXPECT_TRUE(has_any_advanced(spec));

  IndexSpec spec_tensor;
  TensorImpl dummy;  // default-constructed; kind is Tensor
  spec_tensor.items.emplace_back(TensorIndex(dummy));
  EXPECT_TRUE(has_any_advanced(spec_tensor));
}

TEST(IndexingBasicTest, NormalizeSlicePositiveStepBasicCases) {
  // D = 5, slice(1, 4, 2) -> indices 1,3
  Slice s;
  s.start = 1;
  s.stop = 4;
  s.step = 2;
  NormalizedSlice ns = normalize_slice(s, /*dim_size=*/5);
  EXPECT_EQ(ns.start, 1);
  EXPECT_EQ(ns.step, 2);
  EXPECT_EQ(ns.length, 2);

  // Default start/stop with positive step
  Slice s2;
  s2.start.reset();
  s2.stop.reset();
  s2.step.reset();
  NormalizedSlice ns2 = normalize_slice(s2, /*dim_size=*/3);
  EXPECT_EQ(ns2.start, 0);
  EXPECT_EQ(ns2.step, 1);
  EXPECT_EQ(ns2.length, 3);
}

TEST(IndexingBasicTest, NormalizeSliceNegativeStepBasicCases) {
  // D = 5, slice(None, None, -1) -> indices 4,3,2,1,0
  Slice s;
  s.start.reset();
  s.stop.reset();
  s.step = -1;
  NormalizedSlice ns = normalize_slice(s, /*dim_size=*/5);
  EXPECT_EQ(ns.step, -1);
  EXPECT_EQ(ns.length, 5);

  // D = 5, slice(3, 1, -2) -> indices 3
  Slice s2;
  s2.start = 3;
  s2.stop = 1;
  s2.step = -2;
  NormalizedSlice ns2 = normalize_slice(s2, /*dim_size=*/5);
  EXPECT_EQ(ns2.start, 3);
  EXPECT_EQ(ns2.step, -2);
  EXPECT_EQ(ns2.length, 1);
}

TEST(IndexingBasicTest, ExpandEllipsisAndValidateBasic) {
  IndexSpec raw;
  raw.items.emplace_back(TensorIndex(std::int64_t{0}));  // consumes 1 dim
  raw.items.emplace_back(TensorIndex(TensorIndex::EllipsisTag{}));
  raw.items.emplace_back(TensorIndex(std::int64_t{1}));

  IndexSpec spec = expand_ellipsis_and_validate(raw, /*self_dim=*/3);
  // Expect [Integer, Slice, Integer] after expansion.
  ASSERT_EQ(spec.items.size(), 3u);
  EXPECT_EQ(spec.items[0].kind, IndexKind::Integer);
  EXPECT_EQ(spec.items[1].kind, IndexKind::Slice);
  EXPECT_EQ(spec.items[2].kind, IndexKind::Integer);

  // Too many indices
  IndexSpec too_many;
  too_many.items.emplace_back(TensorIndex(std::int64_t{0}));
  too_many.items.emplace_back(TensorIndex(std::int64_t{1}));
  too_many.items.emplace_back(TensorIndex(std::int64_t{2}));
  EXPECT_THROW(expand_ellipsis_and_validate(too_many, /*self_dim=*/2), std::invalid_argument);
}

TEST(IndexingBasicTest, BasicIndexIntegerAndSliceViews) {
  // Create a simple 2x3 tensor with contiguous layout.
  const std::vector<std::int64_t> sizes{2, 3};
  const std::vector<std::int64_t> strides{3, 1};
  const std::size_t nbytes = static_cast<std::size_t>(2 * 3 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // spec: x[1] -> shape (3,)
  IndexSpec spec_row;
  spec_row.items.emplace_back(TensorIndex(std::int64_t{1}));
  TensorImpl row = basic_index(base, spec_row);
  ASSERT_EQ(row.sizes().size(), 1u);
  EXPECT_EQ(row.sizes()[0], 3);
  EXPECT_EQ(row.storage_offset(), strides[0]);  // offset for row 1

  // spec: x[:, 1:] -> shape (2, 2)
  IndexSpec spec_slice;
  spec_slice.items.emplace_back(TensorIndex(Slice{}));  // full slice dim 0
  Slice s_col;
  s_col.start = 1;
  s_col.stop.reset();
  s_col.step.reset();
  spec_slice.items.emplace_back(TensorIndex(s_col));

  TensorImpl view = basic_index(base, spec_slice);
  ASSERT_EQ(view.sizes().size(), 2u);
  EXPECT_EQ(view.sizes()[0], 2);
  EXPECT_EQ(view.sizes()[1], 2);
}

TEST(IndexingBasicTest, BasicIndexZeroDimBehavior) {
  // 0-d tensor
  const std::vector<std::int64_t> sizes{};
  const std::vector<std::int64_t> strides{};
  const std::size_t nbytes = static_cast<std::size_t>(sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl scalar(storage, sizes, strides, /*storage_offset=*/0,
                    ScalarType::Float32, Device::cpu());

  // x[()] -> returns self
  IndexSpec empty;
  TensorImpl same = basic_index(scalar, empty);
  EXPECT_EQ(same.sizes().size(), 0u);

  // x[None] -> unsqueezed (1,)
  IndexSpec one_none;
  one_none.items.emplace_back(TensorIndex(nullptr));
  TensorImpl expanded = basic_index(scalar, one_none);
  ASSERT_EQ(expanded.sizes().size(), 1u);
  EXPECT_EQ(expanded.sizes()[0], 1);

  // x[0] -> invalid index of a 0-dim tensor
  IndexSpec invalid;
  invalid.items.emplace_back(TensorIndex(std::int64_t{0}));
  try {
    (void)basic_index(scalar, invalid);
    FAIL() << "expected std::out_of_range for invalid 0-d index";
  } catch (const std::out_of_range& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find(kErrInvalidZeroDim), std::string::npos);
  }
}

TEST(IndexingBasicTest, BasicIndexPutCpuScalarBroadcastsAndWrites) {
  // 2x3 tensor with contiguous layout; initialize all zeros.
  const std::vector<std::int64_t> sizes{2, 3};
  const std::vector<std::int64_t> strides{3, 1};
  const std::size_t nbytes = static_cast<std::size_t>(2 * 3 * sizeof(float));
  auto buf = static_cast<float*>(::operator new(nbytes));
  for (int i = 0; i < 6; ++i) buf[i] = 0.0f;

  auto storage = vbt::core::make_intrusive<Storage>(
      DataPtr(buf, [](void* p) noexcept { ::operator delete(p); }), nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // value tensor: scalar 2.0f
  const std::vector<std::int64_t> val_sizes{};
  const std::vector<std::int64_t> val_strides{};
  auto val_storage = make_storage_bytes(sizeof(float));
  TensorImpl value(val_storage, val_sizes, val_strides, 0,
                   ScalarType::Float32, Device::cpu());
  *static_cast<float*>(value.data()) = 2.0f;

  // spec: x[:, 1:] -> should write into last two columns of each row.
  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(Slice{}));  // dim 0 full
  Slice s_col;
  s_col.start = 1;
  s_col.stop.reset();
  s_col.step.reset();
  spec.items.emplace_back(TensorIndex(s_col));

  EXPECT_NO_THROW(basic_index_put(base, spec, value));

  // Check that rows have pattern [0, 2, 2].
  ASSERT_NE(base.data(), nullptr);
  float* p = static_cast<float*>(base.data());
  EXPECT_FLOAT_EQ(p[0], 0.0f);
  EXPECT_FLOAT_EQ(p[1], 2.0f);
  EXPECT_FLOAT_EQ(p[2], 2.0f);
  EXPECT_FLOAT_EQ(p[3], 0.0f);
  EXPECT_FLOAT_EQ(p[4], 2.0f);
  EXPECT_FLOAT_EQ(p[5], 2.0f);
}
