// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "vbt/core/tensor_iter.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::TensorIter;
using vbt::core::TensorIterConfig;
using vbt::core::IterOperandRole;
using vbt::core::MemOverlapStatus;
using vbt::core::for_each_reduction_cpu;

namespace {

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  void* base = nullptr;
  if (nbytes > 0) {
    base = ::operator new(nbytes);
  }
  return vbt::core::make_intrusive<Storage>(
      DataPtr(base, [](void* p) noexcept { ::operator delete(p); }), nbytes);
}

static TensorImpl make_contiguous_tensor(const std::vector<int64_t>& sizes,
                                         ScalarType dtype = ScalarType::Float32) {
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

  const std::size_t item_b = static_cast<std::size_t>(vbt::core::itemsize(dtype));
  const std::size_t nbytes = static_cast<std::size_t>(ne) * item_b;
  auto storage = make_storage_bytes(nbytes);
  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0, dtype,
                    Device::cpu());
}

struct SumLoopCtx {
  int64_t tiles{0};
};

static void sum_loop_f32(char** data,
                         const std::int64_t* strides,
                         std::int64_t size,
                         void* ctx_void) {
  auto* ctx = static_cast<SumLoopCtx*>(ctx_void);
  ++ctx->tiles;
  char* out_base = data[0];
  char* in_base  = data[1];
  const std::int64_t in_stride = strides[1];
  auto* out = reinterpret_cast<float*>(out_base);
  float acc = 0.0f;
  for (std::int64_t i = 0; i < size; ++i) {
    const auto* pi =
        reinterpret_cast<const float*>(in_base + i * in_stride);
    acc += *pi;
  }
  *out = acc;
}

static void sum_loop_i64(char** data,
                         const std::int64_t* strides,
                         std::int64_t size,
                         void* ctx_void) {
  auto* ctx = static_cast<SumLoopCtx*>(ctx_void);
  ++ctx->tiles;
  char* out_base = data[0];
  char* in_base  = data[1];
  const std::int64_t in_stride = strides[1];
  auto* out = reinterpret_cast<long long*>(out_base);
  long long acc = 0;
  for (std::int64_t i = 0; i < size; ++i) {
    const auto* pi =
        reinterpret_cast<const long long*>(in_base + i * in_stride);
    acc += *pi;
  }
  *out = acc;
}

static void binary_two_output_loop(char** data,
                                   const std::int64_t* strides,
                                   std::int64_t size,
                                   void* /*ctx_void*/) {
  // data[0] and data[1] are outputs, data[2] is input
  char* out0_base = data[0];
  char* out1_base = data[1];
  char* in_base   = data[2];
  const std::int64_t out0_stride = strides[0];
  const std::int64_t out1_stride = strides[1];
  const std::int64_t in_stride   = strides[2];
  for (std::int64_t i = 0; i < size; ++i) {
    const auto* pin =
        reinterpret_cast<const float*>(in_base + i * in_stride);
    auto* p0 = reinterpret_cast<float*>(out0_base + i * out0_stride);
    auto* p1 = reinterpret_cast<float*>(out1_base + i * out1_stride);
    *p0 = *pin + 1.0f;
    *p1 = *pin - 1.0f;
  }
}

}  // namespace

TEST(TensorIterReductionTest, BuildSingleDimReductionMetadata) {
  auto in = make_contiguous_tensor({2, 3});

  // keepdim = false
  auto out_drop = make_contiguous_tensor({2});
  std::int64_t dim = 1;
  TensorIter iter_drop = TensorIter::reduce_op(out_drop, in,
                                               std::span<const std::int64_t>(&dim, 1));

  EXPECT_TRUE(iter_drop.is_reduction());
  EXPECT_EQ(iter_drop.ndim(), 2);
  ASSERT_EQ(iter_drop.shape().size(), 2u);
  EXPECT_EQ(iter_drop.shape()[0], 2);
  EXPECT_EQ(iter_drop.shape()[1], 3);
  ASSERT_EQ(iter_drop.reduce_dims().size(), 1u);
  EXPECT_EQ(iter_drop.reduce_dims()[0], 1);
  EXPECT_EQ(iter_drop.num_reduce_dims(), 1);
  EXPECT_EQ(iter_drop.ninputs(), 1);
  EXPECT_EQ(iter_drop.noutputs(), 1);

  // keepdim = true
  auto out_keep = make_contiguous_tensor({2, 1});
  TensorIter iter_keep = TensorIter::reduce_op(out_keep, in,
                                               std::span<const std::int64_t>(&dim, 1));
  EXPECT_TRUE(iter_keep.is_reduction());
  ASSERT_EQ(iter_keep.reduce_dims().size(), 1u);
  EXPECT_EQ(iter_keep.reduce_dims()[0], 1);
}

TEST(TensorIterReductionTest, SetReduceDimsPreconditions) {
  auto in = make_contiguous_tensor({2, 3});
  auto out = make_contiguous_tensor({2});

  // set_reduce_dims requires is_reduction(true)
  TensorIterConfig cfg1;
  cfg1.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg1.add_input(in);
  std::int64_t dim = 1;
  EXPECT_THROW(cfg1.set_reduce_dims(std::span<const std::int64_t>(&dim, 1), false),
               std::logic_error);

  // set_reduce_dims requires at least one input
  TensorIterConfig cfg2;
  cfg2.is_reduction(true);
  EXPECT_THROW(cfg2.set_reduce_dims(std::span<const std::int64_t>(&dim, 1), false),
               std::logic_error);

  // Multiple reduction dims are accepted and normalized (no throw).
  TensorIterConfig cfg3;
  cfg3.is_reduction(true);
  cfg3.add_input(in);
  std::int64_t dims2[2] = {1, 0};
  EXPECT_NO_THROW(
      cfg3.set_reduce_dims(std::span<const std::int64_t>(dims2, 2), false));

  // Negative dim normalization and out-of-range handling via factory
  std::int64_t neg_dim = -1;
  auto out2 = make_contiguous_tensor({2});
  TensorIter iter = TensorIter::reduce_op(out2, in,
                                          std::span<const std::int64_t>(&neg_dim, 1));
  ASSERT_EQ(iter.reduce_dims().size(), 1u);
  EXPECT_EQ(iter.reduce_dims()[0], 1);

  std::int64_t bad_dim = 2;  // out of range for {2,3}
  EXPECT_THROW((void)TensorIter::reduce_op(out2, in,
                                           std::span<const std::int64_t>(&bad_dim, 1)),
               std::invalid_argument);
}

TEST(TensorIterReductionTest, ForEachReductionCpuSumRowsFloatAndInt) {
  // Float32 path
  {
    auto in = make_contiguous_tensor({2, 3}, ScalarType::Float32);
    auto out = make_contiguous_tensor({2}, ScalarType::Float32);

    auto* in_data = static_cast<float*>(in.data());
    for (int i = 0; i < 6; ++i) {
      in_data[i] = static_cast<float>(i + 1);  // 1..6
    }

    std::int64_t dim = 1;
    TensorIter iter = TensorIter::reduce_op(out, in,
                                            std::span<const std::int64_t>(&dim, 1));

    SumLoopCtx ctx{};
    for_each_reduction_cpu(iter, &sum_loop_f32, &ctx);

    auto* out_data = static_cast<float*>(out.data());
    EXPECT_EQ(ctx.tiles, 2);
    EXPECT_FLOAT_EQ(out_data[0], 1.0f + 2.0f + 3.0f);
    EXPECT_FLOAT_EQ(out_data[1], 4.0f + 5.0f + 6.0f);
  }

  // Int64 path
  {
    auto in = make_contiguous_tensor({2, 3}, ScalarType::Int64);
    auto out = make_contiguous_tensor({2}, ScalarType::Int64);

    auto* in_data = static_cast<long long*>(in.data());
    for (int i = 0; i < 6; ++i) {
      in_data[i] = static_cast<long long>(i + 1);
    }

    std::int64_t dim = 1;
    TensorIter iter = TensorIter::reduce_op(out, in,
                                            std::span<const std::int64_t>(&dim, 1));

    SumLoopCtx ctx{};
    for_each_reduction_cpu(iter, &sum_loop_i64, &ctx);

    auto* out_data = static_cast<long long*>(out.data());
    EXPECT_EQ(ctx.tiles, 2);
    EXPECT_EQ(out_data[0], 1 + 2 + 3);
    EXPECT_EQ(out_data[1], 4 + 5 + 6);
  }
}

TEST(TensorIterReductionTest, ForEachReductionCpuPreconditionFailures) {
  // Non-reduction iterator should fail preconditions
  auto out = make_contiguous_tensor({4});
  auto in  = make_contiguous_tensor({4});
  TensorIter elem_iter = TensorIter::unary_op(out, in);
  EXPECT_THROW(for_each_reduction_cpu(elem_iter, &sum_loop_f32, nullptr),
               std::logic_error);

  // Null callback for a valid reduction iterator also fails
  auto in2 = make_contiguous_tensor({2, 3});
  auto out2 = make_contiguous_tensor({2});
  std::int64_t dim = 1;
  TensorIter red_iter = TensorIter::reduce_op(out2, in2,
                                              std::span<const std::int64_t>(&dim, 1));
  EXPECT_THROW(for_each_reduction_cpu(red_iter, nullptr, nullptr),
               std::invalid_argument);
}

TEST(TensorIterReductionTest, ForEachCpuOnReductionIteratorThrows) {
  auto in = make_contiguous_tensor({2, 3});
  auto out = make_contiguous_tensor({2});
  std::int64_t dim = 1;
  TensorIter iter = TensorIter::reduce_op(out, in,
                                          std::span<const std::int64_t>(&dim, 1));

  bool threw = false;
  try {
    iter.for_each_cpu(nullptr, nullptr);
  } catch (const std::logic_error& e) {
    threw = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("for_each_cpu is only valid for non-reduction"),
              std::string::npos);
  }
  EXPECT_TRUE(threw);
}

TEST(TensorIterReductionTest, MultiOutputElementwiseWritesBothOutputs) {
  auto out0 = make_contiguous_tensor({4});
  auto out1 = make_contiguous_tensor({4});
  auto in   = make_contiguous_tensor({4});

  auto* in_data = static_cast<float*>(in.data());
  for (int i = 0; i < 4; ++i) {
    in_data[i] = static_cast<float>(i);
  }

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out0, /*defined=*/true),
                 IterOperandRole::WriteOnly);
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out1, /*defined=*/true),
                 IterOperandRole::WriteOnly);
  cfg.add_input(in);
  cfg.check_mem_overlap(true);
  TensorIter iter = cfg.build();

  EXPECT_FALSE(iter.is_reduction());
  EXPECT_EQ(iter.noutputs(), 2);
  EXPECT_EQ(iter.ninputs(), 1);

  iter.for_each_cpu(&binary_two_output_loop, nullptr);

  auto* out0_data = static_cast<float*>(out0.data());
  auto* out1_data = static_cast<float*>(out1.data());
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(out0_data[i], in_data[i] + 1.0f);
    EXPECT_FLOAT_EQ(out1_data[i], in_data[i] - 1.0f);
  }
}

TEST(TensorIterReductionTest, MultiOutputElementwiseRejectsOverlappingOutputs) {
  auto storage = make_storage_bytes(4 * sizeof(float));
  // Two overlapping views into the same storage
  TensorImpl out0(storage, {4}, {1}, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());
  TensorImpl out1(storage, {4}, {1}, /*storage_offset=*/1,
                  ScalarType::Float32, Device::cpu());
  auto in = make_contiguous_tensor({4});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out0, /*defined=*/true),
                 IterOperandRole::WriteOnly);
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out1, /*defined=*/true),
                 IterOperandRole::WriteOnly);
  cfg.add_input(in);

  bool threw = false;
  try {
    (void)cfg.build();
  } catch (const std::invalid_argument& e) {
    threw = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("overlapping outputs"), std::string::npos);
  }
  EXPECT_TRUE(threw);
}
