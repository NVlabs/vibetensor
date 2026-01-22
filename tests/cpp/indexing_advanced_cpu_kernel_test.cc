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
using vbt::core::indexing::advanced_index_cpu;
using vbt::core::indexing::advanced_index_put_cpu;
using vbt::core::indexing::advanced_index_32bit_enabled;
using vbt::core::indexing::set_advanced_index_32bit_enabled_for_tests;

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(
      DataPtr(::operator new(nbytes),
              [](void* p) noexcept { ::operator delete(p); }),
      nbytes);
}

struct AdvancedIndex32BitGuard {
  bool prev;
  explicit AdvancedIndex32BitGuard(bool enabled)
      : prev(advanced_index_32bit_enabled()) {
    set_advanced_index_32bit_enabled_for_tests(enabled);
  }
  ~AdvancedIndex32BitGuard() {
    set_advanced_index_32bit_enabled_for_tests(prev);
  }
};

TEST(IndexingAdvancedCpuKernelTest, AdvancedIndex32BitFlagDefaultsTrue) {
  // in a fresh process. Later tests are free to toggle it via the
  // test-only setter or RAII guard.
  EXPECT_TRUE(advanced_index_32bit_enabled());
}

TEST(IndexingAdvancedCpuKernelTest, GatherSimple1DMatchesManual) {
  // Base: 1D contiguous float32 tensor of size 5 with known values.
  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());
  auto* base_data = static_cast<float*>(base.data());
  for (int i = 0; i < 5; ++i) {
    base_data[i] = static_cast<float>(10 + i);
  }

  // Index tensor: Int64 [3] with values [0, 4, 2].
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

  AdvancedIndex info = make_advanced_index(base, spec);
  TensorImpl out = advanced_index_cpu(info);

  ASSERT_EQ(out.sizes().size(), 1u);
  EXPECT_EQ(out.sizes()[0], 3);

  auto* out_data = static_cast<float*>(out.data());
  ASSERT_NE(out_data, nullptr);
  EXPECT_FLOAT_EQ(out_data[0], base_data[0]);
  EXPECT_FLOAT_EQ(out_data[1], base_data[4]);
  EXPECT_FLOAT_EQ(out_data[2], base_data[2]);
}

TEST(IndexingAdvancedCpuKernelTest, Gather32BitVs64BitParity) {
  // Reuse the simple 1D gather setup but run with 32-bit and 64-bit
  // loop counters to ensure behavior parity.
  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());
  auto* base_data = static_cast<float*>(base.data());
  for (int i = 0; i < 5; ++i) {
    base_data[i] = static_cast<float>(10 + i);
  }

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

  AdvancedIndex info = make_advanced_index(base, spec);

  // Force 32-bit path.
  AdvancedIndex32BitGuard guard32(/*enabled=*/true);
  TensorImpl out32 = advanced_index_cpu(info);

  // Force 64-bit path inside a nested guard.
  AdvancedIndex32BitGuard guard64(/*enabled=*/false);
  TensorImpl out64 = advanced_index_cpu(info);

  ASSERT_EQ(out32.sizes().size(), out64.sizes().size());
  ASSERT_EQ(out32.sizes()[0], out64.sizes()[0]);

  auto* out32_data = static_cast<float*>(out32.data());
  auto* out64_data = static_cast<float*>(out64.data());
  ASSERT_NE(out32_data, nullptr);
  ASSERT_NE(out64_data, nullptr);

  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(out32_data[i], out64_data[i]);
  }
}

TEST(IndexingAdvancedCpuKernelTest, ScatterOverwriteAndAccumulate) {
  // Base: 1D contiguous float32 tensor of size 4, initialized to zeros.
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

  // Index tensor: positions [1, 3].
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

  AdvancedIndex info = make_advanced_index(base, spec);

  // Overwrite: values [5.0, 7.0].
  const std::vector<std::int64_t> val_sizes{2};
  const std::vector<std::int64_t> val_strides{1};
  const std::size_t val_nbytes = static_cast<std::size_t>(2 * sizeof(float));
  auto val_storage = make_storage_bytes(val_nbytes);
  TensorImpl values(val_storage, val_sizes, val_strides, 0,
                    ScalarType::Float32, Device::cpu());
  auto* val_data = static_cast<float*>(values.data());
  val_data[0] = 5.0f;
  val_data[1] = 7.0f;

  advanced_index_put_cpu(info, values, /*accumulate=*/false);

  EXPECT_FLOAT_EQ(base_data[0], 0.0f);
  EXPECT_FLOAT_EQ(base_data[1], 5.0f);
  EXPECT_FLOAT_EQ(base_data[2], 0.0f);
  EXPECT_FLOAT_EQ(base_data[3], 7.0f);

  // Accumulate with duplicate indices [1, 1].
  const std::vector<std::int64_t> idx2_sizes{2};
  const std::vector<std::int64_t> idx2_strides{1};
  const std::size_t idx2_nbytes = static_cast<std::size_t>(2 * sizeof(std::int64_t));
  auto idx2_storage = make_storage_bytes(idx2_nbytes);
  TensorImpl idx2(idx2_storage, idx2_sizes, idx2_strides, 0,
                  ScalarType::Int64, Device::cpu());
  auto* idx2_data = static_cast<std::int64_t*>(idx2.data());
  idx2_data[0] = 1;
  idx2_data[1] = 1;

  IndexSpec spec2;
  spec2.items.emplace_back(TensorIndex(idx2));

  AdvancedIndex info2 = make_advanced_index(base, spec2);

  const std::vector<std::int64_t> val2_sizes{2};
  const std::vector<std::int64_t> val2_strides{1};
  const std::size_t val2_nbytes = static_cast<std::size_t>(2 * sizeof(float));
  auto val2_storage = make_storage_bytes(val2_nbytes);
  TensorImpl values2(val2_storage, val2_sizes, val2_strides, 0,
                     ScalarType::Float32, Device::cpu());
  auto* val2_data = static_cast<float*>(values2.data());
  val2_data[0] = 1.0f;
  val2_data[1] = 2.0f;

  advanced_index_put_cpu(info2, values2, /*accumulate=*/true);

  // Base[1] was 5.0, then +1.0, then +2.0 => 8.0.
  EXPECT_FLOAT_EQ(base_data[1], 8.0f);
}

TEST(IndexingAdvancedCpuKernelTest, ZeroSizedResultIsNoOp) {
  // Base: 1D contiguous float32 tensor of size 3.
  const std::vector<std::int64_t> sizes{3};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(3 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());
  ASSERT_EQ(base.storage()->nbytes(), nbytes);
  auto* base_data = static_cast<float*>(base.data());
  base_data[0] = 1.0f;
  base_data[1] = 2.0f;
  base_data[2] = 3.0f;

  // Empty integer index tensor.
  const std::vector<std::int64_t> idx_sizes{0};
  const std::vector<std::int64_t> idx_strides{1};
  auto idx_storage = make_storage_bytes(0);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cpu());

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx));

  AdvancedIndex info = make_advanced_index(base, spec);

  TensorImpl out = advanced_index_cpu(info);
  EXPECT_EQ(out.numel(), 0);

  const std::vector<std::int64_t> val_sizes{0};
  const std::vector<std::int64_t> val_strides{1};
  auto val_storage = make_storage_bytes(0);
  TensorImpl values(val_storage, val_sizes, val_strides, 0,
                    ScalarType::Float32, Device::cpu());

  advanced_index_put_cpu(info, values, /*accumulate=*/false);

  EXPECT_FLOAT_EQ(base_data[0], 1.0f);
  EXPECT_FLOAT_EQ(base_data[1], 2.0f);
  EXPECT_FLOAT_EQ(base_data[2], 3.0f);
}

TEST(IndexingAdvancedCpuKernelTest, AccumulateUnsupportedDtypeThrows) {
  // Base: 1D contiguous Float16 tensor of size 3.
  const std::vector<std::int64_t> sizes{3};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(3 * sizeof(std::uint16_t));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float16, Device::cpu());

  // Index tensor selecting positions [0, 1, 2].
  const std::vector<std::int64_t> idx_sizes{3};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cpu());
  auto* idx_data = static_cast<std::int64_t*>(idx.data());
  idx_data[0] = 0;
  idx_data[1] = 1;
  idx_data[2] = 2;

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx));

  AdvancedIndex info = make_advanced_index(base, spec);

  // Values tensor with matching Float16 dtype.
  const std::vector<std::int64_t> val_sizes{3};
  const std::vector<std::int64_t> val_strides{1};
  const std::size_t val_nbytes = static_cast<std::size_t>(3 * sizeof(std::uint16_t));
  auto val_storage = make_storage_bytes(val_nbytes);
  TensorImpl values(val_storage, val_sizes, val_strides, 0,
                    ScalarType::Float16, Device::cpu());

  try {
    advanced_index_put_cpu(info, values, /*accumulate=*/true);
    FAIL() << "expected std::invalid_argument for unsupported accumulate dtype";
  } catch (const std::invalid_argument& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find("advanced_index_put_cpu: accumulate unsupported for dtype"),
              std::string::npos);
  }
}

TEST(IndexingAdvancedCpuKernelTest, GatherParityWith32BitFlag) {
  // Base and index as in GatherSimple1DMatchesManual.
  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl base(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());
  auto* base_data = static_cast<float*>(base.data());
  for (int i = 0; i < 5; ++i) {
    base_data[i] = static_cast<float>(10 + i);
  }

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

  AdvancedIndex info = make_advanced_index(base, spec);

  std::vector<float> baseline(3);
  {
    AdvancedIndex32BitGuard guard(false);
    TensorImpl out = advanced_index_cpu(info);
    ASSERT_EQ(out.sizes().size(), 1u);
    EXPECT_EQ(out.sizes()[0], 3);
    auto* out_data = static_cast<float*>(out.data());
    for (int i = 0; i < 3; ++i) {
      baseline[static_cast<std::size_t>(i)] = out_data[i];
    }
  }

  {
    AdvancedIndex32BitGuard guard(true);
    TensorImpl out = advanced_index_cpu(info);
    ASSERT_EQ(out.sizes().size(), 1u);
    EXPECT_EQ(out.sizes()[0], 3);
    auto* out_data = static_cast<float*>(out.data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_FLOAT_EQ(out_data[i], baseline[static_cast<std::size_t>(i)]);
    }
  }
}

TEST(IndexingAdvancedCpuKernelTest, ScatterParityWith32BitFlag) {
  // Two identical bases so we can compare flag off vs on.
  const std::vector<std::int64_t> sizes{4};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(4 * sizeof(float));

  auto storage1 = make_storage_bytes(nbytes);
  TensorImpl base1(storage1, sizes, strides, /*storage_offset=*/0,
                   ScalarType::Float32, Device::cpu());
  auto* base1_data = static_cast<float*>(base1.data());
  for (int i = 0; i < 4; ++i) {
    base1_data[i] = 0.0f;
  }

  auto storage2 = make_storage_bytes(nbytes);
  TensorImpl base2(storage2, sizes, strides, /*storage_offset=*/0,
                   ScalarType::Float32, Device::cpu());
  auto* base2_data = static_cast<float*>(base2.data());
  for (int i = 0; i < 4; ++i) {
    base2_data[i] = 0.0f;
  }

  const std::vector<std::int64_t> idx_sizes{3};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cpu());
  auto* idx_data = static_cast<std::int64_t*>(idx.data());
  idx_data[0] = 1;
  idx_data[1] = 1;
  idx_data[2] = 3;

  const std::vector<std::int64_t> val_sizes{3};
  const std::vector<std::int64_t> val_strides{1};
  const std::size_t val_nbytes = static_cast<std::size_t>(3 * sizeof(float));
  auto val_storage = make_storage_bytes(val_nbytes);
  TensorImpl values(val_storage, val_sizes, val_strides, 0,
                    ScalarType::Float32, Device::cpu());
  auto* val_data = static_cast<float*>(values.data());
  val_data[0] = 1.0f;
  val_data[1] = 2.0f;
  val_data[2] = 3.0f;

  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx));

  AdvancedIndex info1 = make_advanced_index(base1, spec);
  AdvancedIndex info2 = make_advanced_index(base2, spec);

  {
    AdvancedIndex32BitGuard guard(false);
    advanced_index_put_cpu(info1, values, /*accumulate=*/true);
  }

  {
    AdvancedIndex32BitGuard guard(true);
    advanced_index_put_cpu(info2, values, /*accumulate=*/true);
  }

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(base1_data[i], base2_data[i]);
  }
}
