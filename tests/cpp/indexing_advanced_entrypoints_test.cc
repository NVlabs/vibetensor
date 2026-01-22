// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

#include "vbt/core/indexing.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::indexing::IndexSpec;
using vbt::core::indexing::TensorIndex;
using vbt::core::indexing::AdvancedIndex;
using vbt::core::indexing::make_advanced_index;
using vbt::core::indexing::index;
using vbt::core::indexing::index_put_;
using vbt::core::indexing::advanced_indexing_enabled;
using vbt::core::indexing::set_advanced_indexing_enabled_for_tests;

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(
      DataPtr(::operator new(nbytes),
              [](void* p) noexcept { ::operator delete(p); }),
      nbytes);
}

struct AdvancedIndexingGuard {
  bool prev;
  explicit AdvancedIndexingGuard(bool enabled)
      : prev(advanced_indexing_enabled()) {
    set_advanced_indexing_enabled_for_tests(enabled);
  }
  ~AdvancedIndexingGuard() {
    set_advanced_indexing_enabled_for_tests(prev);
  }
};

TEST(IndexingAdvancedEntryPointsTest, IndexDelegatesToBasicForBasicSpec) {
  const std::vector<std::int64_t> sizes{2, 3};
  const std::vector<std::int64_t> strides{3, 1};
  const std::size_t nbytes = static_cast<std::size_t>(6 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // Basic index spec: x[1] -> IndexKind::Integer only.
  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(std::int64_t{1}));

  TensorImpl out_basic = vbt::core::indexing::basic_index(base, spec);
  TensorImpl out_index = index(base, spec);

  ASSERT_EQ(out_basic.sizes().size(), out_index.sizes().size());
  for (std::size_t i = 0; i < out_basic.sizes().size(); ++i) {
    EXPECT_EQ(out_basic.sizes()[i], out_index.sizes()[i]);
  }
  EXPECT_EQ(out_basic.storage().get(), out_index.storage().get());
  EXPECT_EQ(out_basic.storage_offset(), out_index.storage_offset());

  // Feature flag disabled should still allow basic indexing.
  {
    AdvancedIndexingGuard guard(false);
    TensorImpl out_index_disabled = index(base, spec);
    ASSERT_EQ(out_basic.sizes().size(), out_index_disabled.sizes().size());
    for (std::size_t i = 0; i < out_basic.sizes().size(); ++i) {
      EXPECT_EQ(out_basic.sizes()[i], out_index_disabled.sizes()[i]);
    }
    EXPECT_EQ(out_basic.storage().get(), out_index_disabled.storage().get());
    EXPECT_EQ(out_basic.storage_offset(), out_index_disabled.storage_offset());
  }
}

TEST(IndexingAdvancedEntryPointsTest, IndexAdvancedCpuOnlyAndFeatureFlag) {
  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // Index tensor: Int64 [3].
  const std::vector<std::int64_t> idx_sizes{3};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cpu());
  auto* idx_data = static_cast<std::int64_t*>(idx.data());
  idx_data[0] = 0;
  idx_data[1] = 4;
  idx_data[2] = 2;

  IndexSpec spec_adv;
  spec_adv.items.emplace_back(TensorIndex(idx));

  // Feature flag enabled (default): index should succeed.
  AdvancedIndex info = make_advanced_index(base, spec_adv);
  TensorImpl expected = vbt::core::indexing::advanced_index_cpu(info);
  TensorImpl actual = index(base, spec_adv);

  ASSERT_EQ(expected.sizes().size(), actual.sizes().size());
  for (std::size_t i = 0; i < expected.sizes().size(); ++i) {
    EXPECT_EQ(expected.sizes()[i], actual.sizes()[i]);
  }

  // Feature flag disabled: index should throw with a clear message.
  {
    AdvancedIndexingGuard guard(false);
    try {
      (void)index(base, spec_adv);
      FAIL() << "expected std::runtime_error when advanced indexing is disabled";
    } catch (const std::runtime_error& ex) {
      const std::string msg = ex.what();
      EXPECT_NE(msg.find("advanced indexing disabled"), std::string::npos);
    }
  }

#if VBT_WITH_CUDA
  // CUDA tensor: mismatched CUDA index tensor should throw before launching kernels.
  auto cuda_storage = make_storage_bytes(nbytes);
  TensorImpl cuda_base(cuda_storage, sizes, strides, /*storage_offset=*/0,
                       ScalarType::Float32, Device::cuda(/*idx=*/0));

  // a different (logical) device to exercise the kernel-owned device mismatch error.
  TensorImpl idx_cuda_wrong(idx_storage, idx_sizes, idx_strides, 0,
                            ScalarType::Int64, Device::cuda(/*idx=*/1));

  IndexSpec spec_adv_cuda;
  spec_adv_cuda.items.emplace_back(TensorIndex(idx_cuda_wrong));

  try {
    (void)index(cuda_base, spec_adv_cuda);
    FAIL() << "expected std::invalid_argument for CUDA advanced indexing with mismatched CUDA index device";
  } catch (const std::invalid_argument& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find("index: advanced index tensor must be on the same CUDA device as self"),
              std::string::npos);
  }
#endif
}

TEST(IndexingAdvancedEntryPointsTest, IndexPutAdvancedUpdatesBaseAndHonorsFlag) {
  const std::vector<std::int64_t> sizes{4};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(4 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());
  auto* base_data = static_cast<float*>(base.data());
  for (int i = 0; i < 4; ++i) {
    base_data[i] = 0.0f;
  }

  // Index tensor: [1, 3].
  const std::vector<std::int64_t> idx_sizes{2};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(2 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cpu());
  auto* idx_data = static_cast<std::int64_t*>(idx.data());
  idx_data[0] = 1;
  idx_data[1] = 3;

  IndexSpec spec_adv;
  spec_adv.items.emplace_back(TensorIndex(idx));

  // Values: [5, 7].
  const std::vector<std::int64_t> val_sizes{2};
  const std::vector<std::int64_t> val_strides{1};
  const std::size_t val_nbytes = static_cast<std::size_t>(2 * sizeof(float));
  auto val_storage = make_storage_bytes(val_nbytes);
  TensorImpl values(val_storage, val_sizes, val_strides, 0,
                    ScalarType::Float32, Device::cpu());
  auto* val_data = static_cast<float*>(values.data());
  val_data[0] = 5.0f;
  val_data[1] = 7.0f;

  index_put_(base, spec_adv, values, /*accumulate=*/false);

  EXPECT_FLOAT_EQ(base_data[0], 0.0f);
  EXPECT_FLOAT_EQ(base_data[1], 5.0f);
  EXPECT_FLOAT_EQ(base_data[2], 0.0f);
  EXPECT_FLOAT_EQ(base_data[3], 7.0f);

  // Feature flag disabled should prevent advanced index_put_.
  {
    AdvancedIndexingGuard guard(false);
    try {
      index_put_(base, spec_adv, values, /*accumulate=*/false);
      FAIL() << "expected std::runtime_error when advanced indexing is disabled for index_put_";
    } catch (const std::runtime_error& ex) {
      const std::string msg = ex.what();
      EXPECT_NE(msg.find("advanced indexing disabled"), std::string::npos);
    }
  }
}
