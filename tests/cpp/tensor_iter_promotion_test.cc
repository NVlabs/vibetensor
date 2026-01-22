// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vbt/core/tensor_iter.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/type_promotion.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::TensorIter;
using vbt::core::TensorIterConfig;
using vbt::core::OptionalTensorImplRef;
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

struct PromotionCase {
  ScalarType out_dtype;
  std::vector<ScalarType> input_dtypes;
  bool promote_integer_to_float;
  ScalarType expected_common;
};

static TensorIter build_promotion_iter(const PromotionCase& pc) {
  std::vector<TensorImpl> inputs;
  inputs.reserve(pc.input_dtypes.size());
  const std::vector<std::int64_t> sizes{4};
  for (ScalarType dt : pc.input_dtypes) {
    inputs.push_back(make_contiguous_tensor(sizes, dt));
  }
  TensorImpl out = make_contiguous_tensor(sizes, pc.out_dtype);

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
  for (const TensorImpl& in : inputs) {
    cfg.add_input(in);
  }
  cfg.check_mem_overlap(false);
  cfg.promote_inputs_to_common_dtype(true);
  cfg.promote_integer_inputs_to_float(pc.promote_integer_to_float);
  TensorIter iter = cfg.build();
  return iter;
}

} // namespace

TEST(TensorIterPromotionTest, ResultTypeMatchesExamples) {
  const std::vector<PromotionCase> cases = {
      // Lattice examples from the design doc.
      {ScalarType::Bool,
       {ScalarType::Bool, ScalarType::Bool},
       false,
       ScalarType::Bool},
      {ScalarType::Int32,
       {ScalarType::Bool, ScalarType::Int32},
       false,
       ScalarType::Int32},
      {ScalarType::Int64,
       {ScalarType::Int32, ScalarType::Int64},
       false,
       ScalarType::Int64},
      {ScalarType::Float16,
       {ScalarType::Int64, ScalarType::Float16},
       false,
       ScalarType::Float16},
      {ScalarType::BFloat16,
       {ScalarType::Int64, ScalarType::BFloat16},
       false,
       ScalarType::BFloat16},
      {ScalarType::Float32,
       {ScalarType::Int64, ScalarType::Float32},
       false,
       ScalarType::Float32},
      {ScalarType::Float32,
       {ScalarType::Float16, ScalarType::BFloat16},
       false,
       ScalarType::Float32},
      {ScalarType::Float64,
       {ScalarType::Int64, ScalarType::Float64},
       false,
       ScalarType::Float64},
      {ScalarType::Complex64,
       {ScalarType::Float32, ScalarType::Complex64},
       false,
       ScalarType::Complex64},
      {ScalarType::Complex128,
       {ScalarType::Float64, ScalarType::Complex64},
       false,
       ScalarType::Complex128},
  };

  for (const PromotionCase& pc : cases) {
    SCOPED_TRACE("out_dtype=" + std::to_string(static_cast<int>(pc.out_dtype)));
    TensorIter iter = build_promotion_iter(pc);
    EXPECT_EQ(TensorIterTestHelper::common_dtype(iter), pc.expected_common);
  }
}

TEST(TensorIterPromotionTest, PromoteIntegerInputsToFloatUpgradesIntegralCommon) {
  PromotionCase pc;
  pc.out_dtype = ScalarType::Float32;  // Equals expected common after upgrade.
  pc.input_dtypes = {ScalarType::Int32, ScalarType::Int64};
  pc.promote_integer_to_float = true;
  pc.expected_common = ScalarType::Float32;

  TensorIter iter = build_promotion_iter(pc);
  EXPECT_EQ(TensorIterTestHelper::common_dtype(iter), ScalarType::Float32);
}

TEST(TensorIterPromotionTest, PromoteIntegerInputsToFloatUpgradesBoolOnlyToFloat32) {
  PromotionCase pc;
  pc.out_dtype = ScalarType::Float32;
  pc.input_dtypes = {ScalarType::Bool, ScalarType::Bool};
  pc.promote_integer_to_float = true;
  pc.expected_common = ScalarType::Float32;

  TensorIter iter = build_promotion_iter(pc);
  EXPECT_EQ(TensorIterTestHelper::common_dtype(iter), ScalarType::Float32);
}

TEST(TensorIterPromotionTest, PromoteIntegerInputsToFloatDoesNotTouchFloatCommon) {
  PromotionCase pc;
  pc.out_dtype = ScalarType::Float16;
  pc.input_dtypes = {ScalarType::Bool, ScalarType::Int32, ScalarType::Float16};
  pc.promote_integer_to_float = true;
  pc.expected_common = ScalarType::Float16;

  TensorIter iter = build_promotion_iter(pc);
  EXPECT_EQ(TensorIterTestHelper::common_dtype(iter), ScalarType::Float16);
}

TEST(TensorIterPromotionTest, FlagMisuseThrowsLogicError) {
  auto out_f = make_contiguous_tensor({4}, ScalarType::Float32);
  auto a_i   = make_contiguous_tensor({4}, ScalarType::Int32);

  // promote_integer_inputs_to_float without base promotion.
  {
    TensorIterConfig cfg;
    cfg.add_output(OptionalTensorImplRef(&out_f, /*defined=*/true));
    cfg.add_input(a_i);
    cfg.promote_integer_inputs_to_float(true);
    EXPECT_THROW((void)cfg.build(), std::logic_error);
  }

  // cast_common_dtype_to_outputs without base promotion.
  {
    TensorIterConfig cfg;
    cfg.add_output(OptionalTensorImplRef(&out_f, /*defined=*/true));
    cfg.add_input(a_i);
    cfg.cast_common_dtype_to_outputs(true);
    EXPECT_THROW((void)cfg.build(), std::logic_error);
  }

  // Promotion flags on reductions.
  {
    TensorIterConfig cfg;
    cfg.is_reduction(true);
    cfg.add_output(OptionalTensorImplRef(&out_f, /*defined=*/true),
                   vbt::core::IterOperandRole::ReduceOutput);
    cfg.add_input(a_i);
    const std::int64_t dim = 0;
    cfg.set_reduce_dims(std::span<const std::int64_t>(&dim, 1), /*keepdim=*/false);
    cfg.promote_inputs_to_common_dtype(true);
    EXPECT_THROW((void)cfg.build(), std::logic_error);
  }

  // Promotion flags on nullary / zero-input iterator.
  {
    TensorIterConfig cfg;
    cfg.add_output(OptionalTensorImplRef(&out_f, /*defined=*/true));
    cfg.check_mem_overlap(false);
    cfg.promote_inputs_to_common_dtype(true);
    bool threw = false;
    try {
      (void)cfg.build();
    } catch (const std::logic_error& e) {
      threw = true;
      std::string msg = e.what();
      EXPECT_NE(msg.find("promotion flags are not supported for nullary"), std::string::npos);
    }
    EXPECT_TRUE(threw);
  }
}

TEST(TensorIterPromotionTest, CastCommonDtypeValidationOnly) {
  // Inputs promote to Float32; outputs are Float32 or BFloat16 depending on case.
  auto make_inputs = [](ScalarType a_dt, ScalarType b_dt) {
    std::vector<std::int64_t> sizes{4};
    TensorImpl a = make_contiguous_tensor(sizes, a_dt);
    TensorImpl b = make_contiguous_tensor(sizes, b_dt);
    return std::make_pair(std::move(a), std::move(b));
  };

  // Allowed: D_common=Float32, outputs Float32.
  {
    auto [a, b] = make_inputs(ScalarType::Float32, ScalarType::Float32);
    TensorImpl out = make_contiguous_tensor({4}, ScalarType::Float32);

    TensorIterConfig cfg;
    cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
    cfg.add_input(a);
    cfg.add_input(b);
    cfg.check_mem_overlap(false);
    cfg.promote_inputs_to_common_dtype(true);
    cfg.promote_integer_inputs_to_float(true);
    cfg.cast_common_dtype_to_outputs(true);

    TensorIter iter = cfg.build();
    EXPECT_EQ(TensorIterTestHelper::common_dtype(iter), ScalarType::Float32);
  }

  // Allowed: D_common=Float32, outputs BFloat16 (cast validation).
  {
    auto [a, b] = make_inputs(ScalarType::Float32, ScalarType::Float32);
    TensorImpl out = make_contiguous_tensor({4}, ScalarType::BFloat16);

    TensorIterConfig cfg;
    cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
    cfg.add_input(a);
    cfg.add_input(b);
    cfg.check_mem_overlap(false);
    cfg.promote_inputs_to_common_dtype(true);
    cfg.promote_integer_inputs_to_float(true);
    cfg.cast_common_dtype_to_outputs(true);

    TensorIter iter = cfg.build();
    EXPECT_EQ(TensorIterTestHelper::common_dtype(iter), ScalarType::Float32);
  }

  // Disallowed: D_common=Float32, outputs Int32; cannot cast.
  {
    auto [a, b] = make_inputs(ScalarType::Float32, ScalarType::Float32);
    TensorImpl out = make_contiguous_tensor({4}, ScalarType::Int32);

    TensorIterConfig cfg;
    cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
    cfg.add_input(a);
    cfg.add_input(b);
    cfg.check_mem_overlap(false);
    cfg.promote_inputs_to_common_dtype(true);
    cfg.promote_integer_inputs_to_float(true);
    cfg.cast_common_dtype_to_outputs(true);

    bool threw = false;
    try {
      (void)cfg.build();
    } catch (const std::invalid_argument& e) {
      threw = true;
      std::string msg = e.what();
      EXPECT_NE(msg.find("cannot cast common dtype to output dtype"), std::string::npos);
    }
    EXPECT_TRUE(threw);
  }

  // Disallowed: cast_common_dtype_to_outputs on in-place iterator.
  {
    std::vector<std::int64_t> sizes{4};
    TensorImpl self = make_contiguous_tensor(sizes, ScalarType::Float32);
    TensorImpl other = make_contiguous_tensor(sizes, ScalarType::Float32);

    TensorIterConfig cfg;
    cfg.add_output(OptionalTensorImplRef(&self, /*defined=*/true),
                   vbt::core::IterOperandRole::ReadWrite);
    cfg.add_input(other);
    cfg.check_mem_overlap(true);
    cfg.promote_inputs_to_common_dtype(true);
    cfg.promote_integer_inputs_to_float(true);
    cfg.cast_common_dtype_to_outputs(true);

    bool threw = false;
    try {
      (void)cfg.build();
    } catch (const std::logic_error& e) {
      threw = true;
      std::string msg = e.what();
      EXPECT_NE(msg.find("cast_common_dtype_to_outputs is not allowed for in-place"),
                std::string::npos);
    }
    EXPECT_TRUE(threw);
  }
}

TEST(TensorIterPromotionTest, CheckAllSameDtypeFlagAllowsMismatchedTypesWithoutPromotion) {
  auto out = make_contiguous_tensor({4}, ScalarType::Float32);
  auto a   = make_contiguous_tensor({4}, ScalarType::Int32);

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.check_all_same_dtype(false);

  EXPECT_NO_THROW((void)cfg.build());
}

TEST(TypePromotionTest, NoImagDropCastIsRejected) {
  EXPECT_FALSE(vbt::core::can_cast(ScalarType::Complex64, ScalarType::Float64));
}
