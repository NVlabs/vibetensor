// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <stdexcept>

#include "vbt/core/tensor_iter.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/overlap.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::TensorIter;
using vbt::core::TensorIterConfig;
using vbt::core::IterAliasInfo;
using vbt::core::IterOpSignature;
using vbt::core::IterOperandRole;
using vbt::core::MemOverlapStatus;

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
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(nd) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  int64_t ne = 1;
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

  const std::size_t nbytes = static_cast<std::size_t>(ne) * sizeof(float);
  auto storage = make_storage_bytes(nbytes);
  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0,
                    ScalarType::Float32, Device::cpu());
}

}  // namespace

TEST(TensorIterOverlapTest, OutOfPlaceNoAliasHasNoOutputInputAlias) {
  auto out = make_contiguous_tensor({4});
  auto a   = make_contiguous_tensor({4});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.check_mem_overlap(true);
  TensorIter iter = cfg.build();

  EXPECT_TRUE(iter.mem_overlap_checked());
  EXPECT_FALSE(iter.has_any_output_input_alias());
  ASSERT_EQ(iter.noutputs(), 1);
  ASSERT_EQ(iter.ninputs(), 1);
  EXPECT_EQ(iter.alias_status(0, 0), MemOverlapStatus::No);
}

TEST(TensorIterOverlapTest, InplaceAliasAllowedWhenDeclared) {
  auto self = make_contiguous_tensor({4});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&self, /*defined=*/true),
                 IterOperandRole::ReadWrite);
  cfg.add_input(self);
  cfg.check_mem_overlap(true);

  static const IterAliasInfo kAliases[] = {
      {0, 0, /*is_inplace=*/true, /*is_view=*/false},
  };
  static const IterOpSignature kSig{"vt::add_", kAliases,
                                    sizeof(kAliases) / sizeof(kAliases[0])};
  cfg.set_op_signature(&kSig);

  TensorIter iter = cfg.build();

  EXPECT_TRUE(iter.mem_overlap_checked());
  EXPECT_TRUE(iter.has_any_output_input_alias());
  ASSERT_EQ(iter.noutputs(), 1);
  ASSERT_EQ(iter.ninputs(), 1);
  EXPECT_EQ(iter.alias_status(0, 0), MemOverlapStatus::Full);
}

TEST(TensorIterOverlapTest, AliasIndicesOutOfRangeCauseError) {
  auto out = make_contiguous_tensor({4});
  auto a   = make_contiguous_tensor({4});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.check_mem_overlap(true);

  static const IterAliasInfo kBadAliases[] = {
      {1, 0, /*is_inplace=*/true, /*is_view=*/false},  // output index out of range
  };
  static const IterOpSignature kBadSig{"vt::add_", kBadAliases,
                                       sizeof(kBadAliases) / sizeof(kBadAliases[0])};
  cfg.set_op_signature(&kBadSig);

  EXPECT_THROW((void)cfg.build(), std::invalid_argument);
}

TEST(TensorIterOverlapTest, ReductionNoAliasHasNoOutputInputAlias) {
  auto in = make_contiguous_tensor({4});
  auto out = make_contiguous_tensor({});  // scalar value output

  TensorIterConfig cfg;
  cfg.is_reduction(true);
  cfg.check_mem_overlap(true);
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true),
                 IterOperandRole::ReduceOutput);
  cfg.add_input(in);
  std::int64_t dim = 0;
  cfg.set_reduce_dims(std::span<const std::int64_t>(&dim, 1), /*keepdim=*/false);

  TensorIter iter = cfg.build();
  EXPECT_TRUE(iter.is_reduction());
  EXPECT_TRUE(iter.mem_overlap_checked());
  EXPECT_FALSE(iter.has_any_output_input_alias());
  ASSERT_EQ(iter.noutputs(), 1);
  ASSERT_EQ(iter.ninputs(), 1);
  EXPECT_EQ(iter.alias_status(0, 0), MemOverlapStatus::No);
}

TEST(TensorIterOverlapTest, ReductionRejectsInputOutputAlias) {
  auto storage = make_storage_bytes(4 * sizeof(float));
  // Input: size [4], stride [1]
  TensorImpl in(storage, {4}, {1}, /*storage_offset=*/0,
                ScalarType::Float32, Device::cpu());
  // Output: scalar sharing the same storage
  TensorImpl out(storage, {}, {}, /*storage_offset=*/0,
                 ScalarType::Float32, Device::cpu());

  TensorIterConfig cfg;
  cfg.is_reduction(true);
  cfg.check_mem_overlap(true);
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true),
                 IterOperandRole::ReduceOutput);
  cfg.add_input(in);
  std::int64_t dim = 0;
  cfg.set_reduce_dims(std::span<const std::int64_t>(&dim, 1), /*keepdim=*/false);

  bool threw = false;
  try {
    (void)cfg.build();
  } catch (const std::invalid_argument& e) {
    threw = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("must not alias inputs"), std::string::npos);
  }
  EXPECT_TRUE(threw);
}
