// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

#include "vbt/core/indexing.h"
#include "vbt/core/indexing_advanced_stats.h"
#include "vbt/core/indexing/index_errors.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/tensor_ops.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::indexing::IndexKind;
using vbt::core::indexing::IndexSpec;
using vbt::core::indexing::TensorIndex;
using vbt::core::indexing::Slice;
using vbt::core::indexing::AdvancedIndex;
using vbt::core::indexing::make_advanced_index;
using vbt::core::indexing::get_m_index_advanced_stats;
using vbt::core::indexing::reset_m_index_advanced_stats_for_tests;
namespace idx_errors = vbt::core::indexing::errors;

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(
      DataPtr(::operator new(nbytes),
              [](void* p) noexcept { ::operator delete(p); }),
      nbytes);
}

TEST(IndexingAdvancedMakeTest, Simple1DIntegerTensorIndex) {
  // Base: 1D contiguous float32 tensor of size 5.
  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // Index tensor: Int64 [3] with values [0, 4, 2].
  const std::vector<std::int64_t> idx_sizes{3};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx_tensor(idx_storage, idx_sizes, idx_strides, 0,
                        ScalarType::Int64, Device::cpu());
  auto* idx_data = static_cast<std::int64_t*>(idx_tensor.data());
  idx_data[0] = 0;
  idx_data[1] = 4;
  idx_data[2] = 2;

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx_tensor));

  AdvancedIndex info = make_advanced_index(base, spec);

  // Single index tensor.
  ASSERT_EQ(info.indices.size(), 1u);
  ASSERT_EQ(info.indexed_sizes.size(), 1u);
  ASSERT_EQ(info.indexed_strides_elems.size(), 1u);

  EXPECT_EQ(info.indexed_sizes[0], 5);
  EXPECT_EQ(info.dims_before, 0);
  EXPECT_EQ(info.dims_after, 0);

  // Result shape should match index shape for 1D base.
  ASSERT_EQ(info.index_shape.size(), 1u);
  EXPECT_EQ(info.index_shape[0], 3);
  ASSERT_EQ(info.result_shape.size(), 1u);
  EXPECT_EQ(info.result_shape[0], 3);

  // src view must have result_shape.
  const auto& src_sizes = info.src.sizes();
  ASSERT_EQ(src_sizes.size(), info.result_shape.size());
  EXPECT_EQ(src_sizes[0], info.result_shape[0]);

  // Indices were normalized into the 0..size-1 range without change here.
  TensorImpl idx_view_clone = vbt::core::clone_cpu(info.indices[0]);
  auto* norm = static_cast<std::int64_t*>(idx_view_clone.data());
  ASSERT_EQ(idx_view_clone.numel(), 3);
  EXPECT_EQ(norm[0], 0);
  EXPECT_EQ(norm[1], 4);
  EXPECT_EQ(norm[2], 2);
}

TEST(IndexingAdvancedMakeTest, NegativeIndicesNormalizedAndBoundsChecked) {
  // Base: 1D contiguous float32 tensor of size 4.
  const std::vector<std::int64_t> sizes{4};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(4 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // Index tensor with negative indices in range.
  const std::vector<std::int64_t> idx_sizes{3};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx_tensor(idx_storage, idx_sizes, idx_strides, 0,
                        ScalarType::Int64, Device::cpu());
  auto* idx_data = static_cast<std::int64_t*>(idx_tensor.data());
  idx_data[0] = -1;  // -> 3
  idx_data[1] = -4;  // -> 0
  idx_data[2] = 1;   // stays 1

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx_tensor));

  AdvancedIndex info = make_advanced_index(base, spec);

  TensorImpl idx_view_clone = vbt::core::clone_cpu(info.indices[0]);
  auto* norm = static_cast<std::int64_t*>(idx_view_clone.data());
  ASSERT_EQ(idx_view_clone.numel(), 3);
  EXPECT_EQ(norm[0], 3);
  EXPECT_EQ(norm[1], 0);
  EXPECT_EQ(norm[2], 1);

  // Out-of-range negative index should throw with canonical substring.
  auto bad_idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl bad_idx(bad_idx_storage, idx_sizes, idx_strides, 0,
                     ScalarType::Int64, Device::cpu());
  auto* bad_data = static_cast<std::int64_t*>(bad_idx.data());
  bad_data[0] = -5;  // < -size
  bad_data[1] = 0;
  bad_data[2] = 1;

  IndexSpec bad_spec;
  bad_spec.items.emplace_back(TensorIndex(bad_idx));

  try {
    (void)make_advanced_index(base, bad_spec);
    FAIL() << "expected std::out_of_range for out-of-range advanced index";
  } catch (const std::out_of_range& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find(idx_errors::kErrIndexOutOfRange),
              std::string::npos);
  }
}

TEST(IndexingAdvancedMakeTest, ZeroDimSelfUnsupported) {
  // 0-d base tensor.
  const std::vector<std::int64_t> sizes0{};
  const std::vector<std::int64_t> strides0{};
  const std::size_t nbytes0 = static_cast<std::size_t>(sizeof(float));
  auto storage0 = make_storage_bytes(nbytes0);
  TensorImpl scalar(storage0, sizes0, strides0, /*storage_offset=*/0,
                    ScalarType::Float32, Device::cpu());

  const std::vector<std::int64_t> idx_sizes{1};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(1 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx_tensor(idx_storage, idx_sizes, idx_strides, 0,
                        ScalarType::Int64, Device::cpu());
  auto* idx_data = static_cast<std::int64_t*>(idx_tensor.data());
  idx_data[0] = 0;

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx_tensor));

  EXPECT_THROW({
    (void)make_advanced_index(scalar, spec);
  }, std::invalid_argument);
}

TEST(IndexingAdvancedMakeTest, BoolMaskConvertedToPerDimIndices2D) {
  // Base: 2D contiguous float32 tensor of shape (2, 2).
  const std::vector<std::int64_t> base_sizes{2, 2};
  const std::vector<std::int64_t> base_strides{2, 1};
  const std::size_t nbytes = static_cast<std::size_t>(4 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, base_sizes, base_strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // Bool mask:
  // [[true, false],
  //  [false, true]]
  const std::vector<std::int64_t> mask_sizes{2, 2};
  const std::vector<std::int64_t> mask_strides{2, 1};
  const std::size_t mask_nbytes = static_cast<std::size_t>(4 * sizeof(std::uint8_t));
  auto mask_storage = make_storage_bytes(mask_nbytes);
  TensorImpl mask(mask_storage, mask_sizes, mask_strides, 0,
                  ScalarType::Bool, Device::cpu());
  auto* m = static_cast<std::uint8_t*>(mask.data());
  m[0] = 1;
  m[1] = 0;
  m[2] = 0;
  m[3] = 1;

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(mask));

  AdvancedIndex info = make_advanced_index(base, spec);

  // Two index tensors, one per original dim.
  ASSERT_EQ(info.indices.size(), 2u);
  ASSERT_EQ(info.indexed_sizes.size(), 2u);
  ASSERT_EQ(info.indexed_strides_elems.size(), 2u);

  EXPECT_EQ(info.indexed_sizes[0], 2);
  EXPECT_EQ(info.indexed_sizes[1], 2);

  // index_shape is (n_true,) with n_true == 2.
  ASSERT_EQ(info.index_shape.size(), 1u);
  EXPECT_EQ(info.index_shape[0], 2);

  // result_shape should also be (2,).
  ASSERT_EQ(info.result_shape.size(), 1u);
  EXPECT_EQ(info.result_shape[0], 2);

  // Indices[0] == [0, 1], indices[1] == [0, 1].
  for (std::size_t dim = 0; dim < 2; ++dim) {
    TensorImpl idx_clone = vbt::core::clone_cpu(info.indices[dim]);
    auto* vals = static_cast<std::int64_t*>(idx_clone.data());
    ASSERT_EQ(idx_clone.numel(), 2);
    EXPECT_EQ(vals[0], 0);
    EXPECT_EQ(vals[1], 1);
  }
}

TEST(IndexingAdvancedMakeTest, ScalarBoolSelectsAllOrNone) {
  // Base: simple 2x2 tensor.
  const std::vector<std::int64_t> base_sizes{2, 2};
  const std::vector<std::int64_t> base_strides{2, 1};
  const std::size_t nbytes = static_cast<std::size_t>(4 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, base_sizes, base_strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());
  ASSERT_EQ(base.storage()->nbytes(), nbytes);

  // [True] -> selects all elements.
  {
    IndexSpec spec_true;
    spec_true.items.emplace_back(TensorIndex(true));
    AdvancedIndex info_true = make_advanced_index(base, spec_true);

    ASSERT_EQ(info_true.index_shape.size(), 1u);
    EXPECT_EQ(info_true.index_shape[0], 4);
    ASSERT_EQ(info_true.result_shape.size(), 1u);
    EXPECT_EQ(info_true.result_shape[0], 4);
    // For scalar-bool [True], all elements are selected.
    // We only verify metadata here; kernel behavior is
    // covered by separate CPU kernel tests.
    ASSERT_EQ(info_true.indices.size(), 2u);
  }

  // [False] -> selects no elements.
  {
    IndexSpec spec_false;
    spec_false.items.emplace_back(TensorIndex(false));
    AdvancedIndex info_false = make_advanced_index(base, spec_false);

    ASSERT_EQ(info_false.index_shape.size(), 1u);
    EXPECT_EQ(info_false.index_shape[0], 0);
    ASSERT_EQ(info_false.result_shape.size(), 1u);
    EXPECT_EQ(info_false.result_shape[0], 0);
  }
}

TEST(IndexingAdvancedMakeTest, MixingScalarBoolAndTensorAdvancedRejected) {
  // Base: 1D contiguous float32 tensor of size 4.
  const std::vector<std::int64_t> sizes{4};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(4 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // Integer index tensor of shape [2].
  const std::vector<std::int64_t> idx_sizes{2};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(2 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cpu());

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(true));
  spec.items.emplace_back(TensorIndex(idx));

  EXPECT_THROW({
    (void)make_advanced_index(base, spec);
  }, std::invalid_argument);
}

TEST(IndexingAdvancedMakeTest, CpuStatsHintCountsNonEmptyDomain) {
  reset_m_index_advanced_stats_for_tests();

  // Simple non-empty advanced index: same setup as Simple1DIntegerTensorIndex.
  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  const std::vector<std::int64_t> idx_sizes{3};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx_tensor(idx_storage, idx_sizes, idx_strides, 0,
                        ScalarType::Int64, Device::cpu());
  auto* idx_data = static_cast<std::int64_t*>(idx_tensor.data());
  idx_data[0] = 0;
  idx_data[1] = 4;
  idx_data[2] = 2;

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx_tensor));

  (void)make_advanced_index(base, spec);

  const auto& stats = get_m_index_advanced_stats();
  // Under current DoS caps, any non-empty result domain has hint==true.
  EXPECT_EQ(stats.cpu_32bit_hint_true.load(std::memory_order_relaxed), 1u);
  EXPECT_EQ(stats.cpu_32bit_hint_false.load(std::memory_order_relaxed), 0u);
}

TEST(IndexingAdvancedMakeTest, CpuStatsZeroNumelDoesNotIncrement) {
  reset_m_index_advanced_stats_for_tests();

  // Base: 1D contiguous float32 tensor of size 3.
  const std::vector<std::int64_t> sizes{3};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(3 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // Empty integer index tensor => result numel == 0.
  const std::vector<std::int64_t> idx_sizes{0};
  const std::vector<std::int64_t> idx_strides{1};
  auto idx_storage = make_storage_bytes(0);
  TensorImpl idx_tensor(idx_storage, idx_sizes, idx_strides, 0,
                        ScalarType::Int64, Device::cpu());

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx_tensor));

  (void)make_advanced_index(base, spec);

  const auto& stats = get_m_index_advanced_stats();
  EXPECT_EQ(stats.cpu_32bit_hint_true.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(stats.cpu_32bit_hint_false.load(std::memory_order_relaxed), 0u);
}

TEST(IndexingAdvancedMakeTest, SuffixIndicesAfterAdvancedBlockRejected) {
  // Base: 3D contiguous float32 tensor of shape (2, 2, 2).
  const std::vector<std::int64_t> base_sizes{2, 2, 2};
  const std::vector<std::int64_t> base_strides{4, 2, 1};
  const std::size_t nbytes = static_cast<std::size_t>(8 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, base_sizes, base_strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // Boolean mask followed by an integer suffix index: x[mask, 1].
  const std::vector<std::int64_t> mask_sizes{2, 2};
  const std::vector<std::int64_t> mask_strides{2, 1};
  const std::size_t mask_nbytes = static_cast<std::size_t>(4 * sizeof(std::uint8_t));
  auto mask_storage = make_storage_bytes(mask_nbytes);
  TensorImpl mask(mask_storage, mask_sizes, mask_strides, 0,
                  ScalarType::Bool, Device::cpu());

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(mask));
  spec.items.emplace_back(TensorIndex(std::int64_t{1}));

  try {
    (void)make_advanced_index(base, spec);
    FAIL() << "expected std::invalid_argument for suffix indices after the advanced block";
  } catch (const std::invalid_argument& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find("suffix indices after the advanced block are not supported"),
              std::string::npos);
  }
}

TEST(IndexingAdvancedMakeTest, PrefixOnlyPatternsSupported) {
  // Base: 2D contiguous float32 tensor of shape (2, 3).
  const std::vector<std::int64_t> base_sizes{2, 3};
  const std::vector<std::int64_t> base_strides{3, 1};
  const std::size_t nbytes = static_cast<std::size_t>(6 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, base_sizes, base_strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // Index tensor: Int64 [2].
  const std::vector<std::int64_t> idx_sizes{2};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(2 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cpu());
  auto* idx_data = static_cast<std::int64_t*>(idx.data());
  idx_data[0] = 0;
  idx_data[1] = 1;

  // Pattern x[1, idx].
  {
    IndexSpec spec;
    spec.items.emplace_back(TensorIndex(std::int64_t{1}));
    spec.items.emplace_back(TensorIndex(idx));

    AdvancedIndex info = make_advanced_index(base, spec);
    ASSERT_EQ(info.index_shape.size(), 1u);
    EXPECT_EQ(info.index_shape[0], 2);
    ASSERT_EQ(info.result_shape.size(), 1u);
    EXPECT_EQ(info.result_shape[0], 2);
  }

  // Pattern x[idx, 1] (2D normalization path).
  {
    IndexSpec spec;
    spec.items.emplace_back(TensorIndex(idx));
    spec.items.emplace_back(TensorIndex(std::int64_t{1}));

    AdvancedIndex info = make_advanced_index(base, spec);
    ASSERT_EQ(info.index_shape.size(), 1u);
    EXPECT_EQ(info.index_shape[0], 2);
    ASSERT_EQ(info.result_shape.size(), 1u);
    EXPECT_EQ(info.result_shape[0], 2);
  }
}

TEST(IndexingAdvancedMakeTest, ZeroDimAdvancedTensorIndexUnsupported) {
  // Base: 1D contiguous float32 tensor of size 4.
  const std::vector<std::int64_t> base_sizes{4};
  const std::vector<std::int64_t> base_strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(4 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, base_sizes, base_strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // 0-d index tensor.
  const std::vector<std::int64_t> idx_sizes{};
  const std::vector<std::int64_t> idx_strides{};
  const std::size_t idx_nbytes = static_cast<std::size_t>(1 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cpu());

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx));

  try {
    (void)make_advanced_index(base, spec);
    FAIL() << "expected std::invalid_argument for 0-d advanced tensor index";
  } catch (const std::invalid_argument& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find("advanced tensor indices must have dim() > 0"),
              std::string::npos);
  }
}

TEST(IndexingAdvancedMakeTest, ScalarBoolTooLargeTriggersDosCap) {
  // Choose a size just above the internal scalar-bool DoS cap (1e7).
  const std::int64_t N = 10'000'001;
  const std::vector<std::int64_t> base_sizes{N};
  const std::vector<std::int64_t> base_strides{1};
  auto storage = make_storage_bytes(0);
  TensorImpl base(storage, base_sizes, base_strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(true));

  try {
    (void)make_advanced_index(base, spec);
    FAIL() << "expected std::runtime_error for scalar-bool DoS cap";
  } catch (const std::runtime_error& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find(idx_errors::kErrAdvIndexTooLarge),
              std::string::npos);
  }
}

TEST(IndexingAdvancedMakeTest, ResultTooLargeTriggersDosCap) {
  // Base: 2D tensor where the prefix dimension is large enough that the
  // logical result domain exceeds the advanced-indexing result cap.
  const std::int64_t D0 = 200'000;  // prefix dimension size
  const std::int64_t D1 = 2;        // advanced dimension size
  const std::vector<std::int64_t> base_sizes{D0, D1};
  const std::vector<std::int64_t> base_strides{D1, 1};
  const std::size_t nbytes =
      static_cast<std::size_t>(D0 * D1 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, base_sizes, base_strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // Index tensor on the last dimension with small cardinality N such that
  // D0 * N > kAdvIndexMaxResultNumel while index_numel stays well below the
  // index-numel DoS cap.
  const std::int64_t N = 2000;  // 2e5 * 2e3 = 4e8 > 1e8
  const std::vector<std::int64_t> idx_sizes{N};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes =
      static_cast<std::size_t>(N) * sizeof(std::int64_t);
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cpu());
  auto* idx_data = static_cast<std::int64_t*>(idx.data());
  for (std::int64_t i = 0; i < N; ++i) {
    idx_data[i] = 0;  // all indices in-range for the advanced dim
  }

  IndexSpec spec;
  // Explicit prefix full slice so the tensor index targets the last dim.
  spec.items.emplace_back(TensorIndex(Slice{}));
  spec.items.emplace_back(TensorIndex(idx));

  try {
    (void)make_advanced_index(base, spec);
    FAIL() << "expected std::runtime_error for advanced indexing result DoS cap";
  } catch (const std::runtime_error& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find(idx_errors::kErrAdvResultTooLarge),
              std::string::npos);
  }
}

TEST(IndexingAdvancedMakeTest, TooManyIndexDimsTriggersDosCap) {
  // Base: small 2D tensor; exact contents are irrelevant for this test.
  const std::vector<std::int64_t> base_sizes{2, 2};
  const std::vector<std::int64_t> base_strides{2, 1};
  const std::size_t nbytes = static_cast<std::size_t>(4 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, base_sizes, base_strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // Index tensor with rank greater than the allowed kAdvIndexMaxNdim
  // (currently 25). Each dimension has size 1 so index_numel stays tiny.
  const std::int64_t ndims = 26;
  std::vector<std::int64_t> idx_sizes(static_cast<std::size_t>(ndims), 1);
  std::vector<std::int64_t> idx_strides(static_cast<std::size_t>(ndims), 1);
  const std::size_t idx_nbytes = sizeof(std::int64_t);
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cpu());

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx));

  try {
    (void)make_advanced_index(base, spec);
    FAIL() << "expected std::runtime_error for too many advanced index dims";
  } catch (const std::runtime_error& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find(idx_errors::kErrAdvTooManyIndexDims),
              std::string::npos);
  }
}