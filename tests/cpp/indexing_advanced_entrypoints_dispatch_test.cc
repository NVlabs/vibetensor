// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

#include "vbt/core/indexing.h"
#include "vbt/core/indexing/index_errors.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/boxed.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::indexing::IndexSpec;
using vbt::core::indexing::TensorIndex;
using vbt::core::indexing::advanced_indexing_enabled;
using vbt::core::indexing::set_advanced_indexing_enabled_for_tests;
using vbt::dispatch::BoxedStack;
using vbt::dispatch::Dispatcher;
namespace idx_errors = vbt::core::indexing::errors;

extern "C" void vbt_register_indexing_kernels();

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

TEST(IndexingAdvancedEntryPointsDispatchTest, VtIndexMatchesCoreForTensorIndex) {
  AdvancedIndexingGuard flag_guard(true);

  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());
  auto* base_data = static_cast<float*>(base.data());
  for (int i = 0; i < 5; ++i) {
    base_data[i] = static_cast<float>(i);
  }

  // Index tensor: Int64 [3] = {0, 4, 2}.
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

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx));

  // Reference result via core entrypoint.
  TensorImpl expected = vbt::core::indexing::index(base, spec);

  vbt_register_indexing_kernels();

  // Build meta tensor matching encode_index_meta_tensor() layout.
  const std::vector<std::int64_t> meta_sizes{4};
  const std::vector<std::int64_t> meta_strides{1};
  const std::size_t meta_nbytes = static_cast<std::size_t>(4 * sizeof(std::int64_t));
  auto meta_storage = make_storage_bytes(meta_nbytes);
  TensorImpl meta(meta_storage, meta_sizes, meta_strides, 0,
                  ScalarType::Int64, Device::cpu());
  auto* m = static_cast<std::int64_t*>(meta.data());
  m[0] = 0;  // version
  m[1] = 1;  // advanced_kind = Tensor
  m[2] = 0;  // advanced_param (reserved)
  m[3] = 0;

  BoxedStack stack;
  stack.push_back(base);
  stack.push_back(idx);
  stack.push_back(meta);

  Dispatcher::instance().callBoxed("vt::index", stack);
  ASSERT_EQ(stack.size(), 1u);
  TensorImpl actual = stack[0];

  ASSERT_EQ(expected.sizes().size(), actual.sizes().size());
  for (std::size_t i = 0; i < expected.sizes().size(); ++i) {
    EXPECT_EQ(expected.sizes()[i], actual.sizes()[i]);
  }

  const float* expected_data = static_cast<const float*>(expected.data());
  const float* actual_data = static_cast<const float*>(actual.data());
  const std::size_t ne = static_cast<std::size_t>(expected.numel());
  for (std::size_t i = 0; i < ne; ++i) {
    EXPECT_FLOAT_EQ(expected_data[i], actual_data[i]);
  }
}

TEST(IndexingAdvancedEntryPointsDispatchTest, VtIndexPutMatchesCoreForTensorIndex) {
  AdvancedIndexingGuard flag_guard(true);

  const std::vector<std::int64_t> sizes{4};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(4 * sizeof(float));
  auto storage1 = make_storage_bytes(nbytes);
  auto storage2 = make_storage_bytes(nbytes);

  TensorImpl base_vt(storage1, sizes, strides, /*storage_offset=*/0,
                     ScalarType::Float32, Device::cpu());
  TensorImpl base_ref(storage2, sizes, strides, /*storage_offset=*/0,
                      ScalarType::Float32, Device::cpu());

  auto* data_vt = static_cast<float*>(base_vt.data());
  auto* data_ref = static_cast<float*>(base_ref.data());
  for (int i = 0; i < 4; ++i) {
    data_vt[i] = 0.0f;
    data_ref[i] = 0.0f;
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

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx));

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

  // Reference behavior via core index_put_.
  vbt::core::indexing::index_put_(base_ref, spec, values, /*accumulate=*/false);

  vbt_register_indexing_kernels();

  // Meta tensor as above.
  const std::vector<std::int64_t> meta_sizes{4};
  const std::vector<std::int64_t> meta_strides{1};
  const std::size_t meta_nbytes = static_cast<std::size_t>(4 * sizeof(std::int64_t));
  auto meta_storage = make_storage_bytes(meta_nbytes);
  TensorImpl meta(meta_storage, meta_sizes, meta_strides, 0,
                  ScalarType::Int64, Device::cpu());
  auto* m = static_cast<std::int64_t*>(meta.data());
  m[0] = 0;  // version
  m[1] = 1;  // advanced_kind = Tensor
  m[2] = 0;  // advanced_param
  m[3] = 0;  // prefix_len

  // 0-d Bool CPU tensor accumulate = False.
  const std::size_t acc_nbytes = sizeof(std::uint8_t);
  auto acc_storage = make_storage_bytes(acc_nbytes);
  std::vector<std::int64_t> acc_sizes;  // 0-d
  std::vector<std::int64_t> acc_strides;  // empty for 0-d
  TensorImpl acc(acc_storage, acc_sizes, acc_strides, 0,
                 ScalarType::Bool, Device::cpu());
  auto* acc_data = static_cast<std::uint8_t*>(acc.data());
  if (acc_data) *acc_data = 0u;

  BoxedStack stack;
  stack.push_back(base_vt);
  stack.push_back(idx);
  stack.push_back(values);
  stack.push_back(meta);
  stack.push_back(acc);

  Dispatcher::instance().callBoxed("vt::index_put", stack);
  ASSERT_EQ(stack.size(), 1u);

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(data_ref[i], data_vt[i]);
  }
}

TEST(IndexingAdvancedEntryPointsDispatchTest, VtIndexRejectsNonCpuSelf) {
  AdvancedIndexingGuard flag_guard(true);

  const std::vector<std::int64_t> sizes{2};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(2 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl self(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cuda());

  // Dummy index/meta; error should be raised before they are inspected.
  const std::vector<std::int64_t> idx_sizes{1};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes =
      static_cast<std::size_t>(1 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cuda());

  const std::vector<std::int64_t> meta_sizes{4};
  const std::vector<std::int64_t> meta_strides{1};
  const std::size_t meta_nbytes =
      static_cast<std::size_t>(4 * sizeof(std::int64_t));
  auto meta_storage = make_storage_bytes(meta_nbytes);
  TensorImpl meta(meta_storage, meta_sizes, meta_strides, 0,
                  ScalarType::Int64, Device::cuda());
  auto* m = static_cast<std::int64_t*>(meta.data());
  m[0] = 0;
  m[1] = 1;
  m[2] = 0;
  m[3] = 0;

  vbt_register_indexing_kernels();

  BoxedStack stack;
  stack.push_back(self);
  stack.push_back(idx);
  stack.push_back(meta);

  try {
    Dispatcher::instance().callBoxed("vt::index", stack);
    FAIL() << "expected std::invalid_argument for vt::index on non-CPU tensor";
  } catch (const std::invalid_argument& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find(idx_errors::kErrMetaInvalidShape),
              std::string::npos);
  }
}

TEST(IndexingAdvancedEntryPointsDispatchTest, VtIndexRejectsUnsupportedMetaVersion) {
  AdvancedIndexingGuard flag_guard(true);

  const std::vector<std::int64_t> sizes{3};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(3 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl self(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  const std::vector<std::int64_t> idx_sizes{1};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes =
      static_cast<std::size_t>(1 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cpu());
  auto* idx_data = static_cast<std::int64_t*>(idx.data());
  idx_data[0] = 0;

  const std::vector<std::int64_t> meta_sizes{4};
  const std::vector<std::int64_t> meta_strides{1};
  const std::size_t meta_nbytes =
      static_cast<std::size_t>(4 * sizeof(std::int64_t));
  auto meta_storage = make_storage_bytes(meta_nbytes);
  TensorImpl meta(meta_storage, meta_sizes, meta_strides, 0,
                  ScalarType::Int64, Device::cpu());
  auto* m = static_cast<std::int64_t*>(meta.data());
  m[0] = 1;  // unsupported meta version
  m[1] = 1;  // adv_kind = Tensor
  m[2] = 0;  // adv_param (reserved)
  m[3] = 0;  // prefix_len = 0

  vbt_register_indexing_kernels();

  BoxedStack stack;
  stack.push_back(self);
  stack.push_back(idx);
  stack.push_back(meta);

  try {
    Dispatcher::instance().callBoxed("vt::index", stack);
    FAIL() << "expected std::invalid_argument for unsupported meta version";
  } catch (const std::invalid_argument& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find(idx_errors::kErrMetaUnsupportedVersion),
              std::string::npos);
  }
}

TEST(IndexingAdvancedEntryPointsDispatchTest, VtIndexPutValidatesIndexDtypeAndMeta) {
  AdvancedIndexingGuard flag_guard(true);

  const std::vector<std::int64_t> sizes{3};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(3 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());

  // Index tensor with wrong dtype (Float32 instead of Int32/Int64/Bool).
  const std::vector<std::int64_t> idx_sizes{2};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(2 * sizeof(float));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Float32, Device::cpu());

  // Value tensor matching base.
  auto val_storage = make_storage_bytes(nbytes);
  TensorImpl value(val_storage, sizes, strides, 0,
                   ScalarType::Float32, Device::cpu());

  // Meta tensor with too few elements (invalid shape).
  const std::vector<std::int64_t> meta_sizes_bad{3};
  const std::vector<std::int64_t> meta_strides_bad{1};
  const std::size_t meta_bad_nbytes = static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto meta_storage_bad = make_storage_bytes(meta_bad_nbytes);
  TensorImpl meta_bad(meta_storage_bad, meta_sizes_bad, meta_strides_bad, 0,
                      ScalarType::Int64, Device::cpu());

  // 0-d Bool accumulate tensor but with wrong shape (1-d).
  const std::vector<std::int64_t> acc_sizes_bad{1};
  const std::vector<std::int64_t> acc_strides_bad{1};
  const std::size_t acc_bad_nbytes = sizeof(std::uint8_t);
  auto acc_storage_bad = make_storage_bytes(acc_bad_nbytes);
  TensorImpl acc_bad(acc_storage_bad, acc_sizes_bad, acc_strides_bad, 0,
                     ScalarType::Bool, Device::cpu());

  vbt_register_indexing_kernels();

  // First, invalid index dtype.
  {
    const std::vector<std::int64_t> meta_sizes{4};
    const std::vector<std::int64_t> meta_strides{1};
    const std::size_t meta_nbytes = static_cast<std::size_t>(4 * sizeof(std::int64_t));
    auto meta_storage = make_storage_bytes(meta_nbytes);
    TensorImpl meta(meta_storage, meta_sizes, meta_strides, 0,
                    ScalarType::Int64, Device::cpu());
    auto* m = static_cast<std::int64_t*>(meta.data());
    m[0] = 0;
    m[1] = 1;
    m[2] = 0;
    m[3] = 0;

    BoxedStack stack;
    stack.push_back(base);
    stack.push_back(idx);
    stack.push_back(value);
    stack.push_back(meta);
    stack.push_back(acc_bad);

    try {
      Dispatcher::instance().callBoxed("vt::index_put", stack);
      FAIL() << "expected std::invalid_argument for bad index dtype";
    } catch (const std::invalid_argument& ex) {
      const std::string msg = ex.what();
      EXPECT_NE(msg.find("vt::index_put: index tensor must be int32, int64, or bool"),
                std::string::npos);
    }
  }

  // Next, invalid meta shape.
  {
    BoxedStack stack;
    stack.push_back(base);
    // Provide a valid Int64 index this time.
    const std::vector<std::int64_t> idx_sizes_ok{2};
    const std::vector<std::int64_t> idx_strides_ok{1};
    const std::size_t idx_ok_nbytes = static_cast<std::size_t>(2 * sizeof(std::int64_t));
    auto idx_storage_ok = make_storage_bytes(idx_ok_nbytes);
    TensorImpl idx_ok(idx_storage_ok, idx_sizes_ok, idx_strides_ok, 0,
                      ScalarType::Int64, Device::cpu());

    stack.push_back(idx_ok);
    stack.push_back(value);
    stack.push_back(meta_bad);
    stack.push_back(acc_bad);

    try {
      Dispatcher::instance().callBoxed("vt::index_put", stack);
      FAIL() << "expected std::invalid_argument for bad meta tensor";
    } catch (const std::invalid_argument& ex) {
      const std::string msg = ex.what();
      EXPECT_NE(msg.find(idx_errors::kErrMetaInvalidShape),
                std::string::npos);
    }
  }
}
