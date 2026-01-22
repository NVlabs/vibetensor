// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>

#include "vbt/core/tensor_iter.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/strided_loop.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::TensorIter;
using vbt::core::TensorIterConfig;
using vbt::core::LoopSpec;
using vbt::core::build_unary_spec;

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
    strides[static_cast<std::size_t>(i)] = acc;
    const auto sz = sizes[static_cast<std::size_t>(i)];
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

struct UnaryLoopCtx {
  int64_t tiles{0};
};

static void unary_add_one_loop(char** data,
                               const std::int64_t* strides,
                               std::int64_t size,
                               void* ctx_void) {
  auto* ctx = static_cast<UnaryLoopCtx*>(ctx_void);
  ++ctx->tiles;
  char* out_base = data[0];
  char* a_base   = data[1];
  const std::int64_t out_stride = strides[0];
  const std::int64_t a_stride   = strides[1];
  for (std::int64_t i = 0; i < size; ++i) {
    auto* out = reinterpret_cast<float*>(out_base + i * out_stride);
    const auto* a = reinterpret_cast<const float*>(a_base + i * a_stride);
    *out = *a + 1.0f;
  }
}

struct BinaryLoopCtx {
  int64_t tiles{0};
};

static void binary_add_loop(char** data,
                            const std::int64_t* strides,
                            std::int64_t size,
                            void* ctx_void) {
  auto* ctx = static_cast<BinaryLoopCtx*>(ctx_void);
  ++ctx->tiles;
  char* out_base = data[0];
  char* a_base   = data[1];
  char* b_base   = data[2];
  const std::int64_t out_stride = strides[0];
  const std::int64_t a_stride   = strides[1];
  const std::int64_t b_stride   = strides[2];
  for (std::int64_t i = 0; i < size; ++i) {
    auto* out = reinterpret_cast<float*>(out_base + i * out_stride);
    const auto* a = reinterpret_cast<const float*>(a_base + i * a_stride);
    const auto* b = reinterpret_cast<const float*>(b_base + i * b_stride);
    *out = *a + *b;
  }
}

}  // namespace

TEST(TensorIterBasicTest, BuilderUnary1DContiguousShapeAndStrides) {
  auto out = make_contiguous_tensor({5});
  auto in  = make_contiguous_tensor({5});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(in);
  TensorIter iter = cfg.build();

  EXPECT_EQ(iter.ndim(), 1);
  ASSERT_EQ(iter.shape().size(), 1u);
  EXPECT_EQ(iter.shape()[0], 5);

  const auto& op_out = iter.operand(0);
  const auto& op_in  = iter.operand(1);
  ASSERT_EQ(op_out.dim_stride_bytes.size(), 1u);
  ASSERT_EQ(op_in.dim_stride_bytes.size(), 1u);
  const std::int64_t item_b = static_cast<std::int64_t>(out.itemsize());
  EXPECT_EQ(op_out.dim_stride_bytes[0], item_b);
  EXPECT_EQ(op_in.dim_stride_bytes[0], item_b);

  EXPECT_FALSE(op_out.is_read_write);
  EXPECT_TRUE(op_out.is_output);
  EXPECT_FALSE(op_in.is_output);

  EXPECT_FALSE(iter.is_reduction());
  EXPECT_EQ(iter.num_reduce_dims(), 0);
  EXPECT_TRUE(iter.reduce_dims().empty());
}

TEST(TensorIterBasicTest, BuilderBinary2DMatchesStridedLoopPermutation) {
  auto out = make_contiguous_tensor({2, 3});
  auto a   = make_contiguous_tensor({2, 3});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  TensorIter iter = cfg.build();

  // Iteration rank matches non-size-1 dims; contiguous 2D keeps both dims in a permuted order.
  EXPECT_EQ(iter.ndim(), 2);
  ASSERT_EQ(iter.shape().size(), 2u);
  EXPECT_EQ(iter.shape()[0], 3);
  EXPECT_EQ(iter.shape()[1], 2);

  const auto& op_out = iter.operand(0);
  ASSERT_EQ(op_out.dim_stride_bytes.size(), 2u);
  const std::int64_t item_b = static_cast<std::int64_t>(out.itemsize());
  // Fast-moving cols dim first, then rows.
  EXPECT_EQ(op_out.dim_stride_bytes[0], item_b);
  EXPECT_EQ(op_out.dim_stride_bytes[1], 3 * item_b);
}

TEST(TensorIterBasicTest, ForEachCpuUnary1DAddsOne) {
  auto out = make_contiguous_tensor({8});
  auto in  = make_contiguous_tensor({8});

  auto* out_data = static_cast<float*>(out.data());
  auto* in_data  = static_cast<float*>(in.data());
  for (int i = 0; i < 8; ++i) {
    in_data[i] = static_cast<float>(i * 2);
    out_data[i] = 0.0f;
  }

  TensorIter iter = TensorIter::unary_op(out, in);
  UnaryLoopCtx ctx;
  iter.for_each_cpu(&unary_add_one_loop, &ctx);

  EXPECT_EQ(ctx.tiles, 1);
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(out_data[i], in_data[i] + 1.0f);
  }
}

TEST(TensorIterBasicTest, ForEachCpuBinary2DAddsTensors) {
  auto out = make_contiguous_tensor({2, 3});
  auto a   = make_contiguous_tensor({2, 3});
  auto b   = make_contiguous_tensor({2, 3});

  auto* out_data = static_cast<float*>(out.data());
  auto* a_data   = static_cast<float*>(a.data());
  auto* b_data   = static_cast<float*>(b.data());

  int idx = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j, ++idx) {
      a_data[idx] = static_cast<float>(idx);
      b_data[idx] = static_cast<float>(idx * 10);
      out_data[idx] = 0.0f;
    }
  }

  TensorIter iter = TensorIter::binary_op(out, a, b);
  BinaryLoopCtx ctx;
  iter.for_each_cpu(&binary_add_loop, &ctx);

  const std::int64_t expected_tiles =
      iter.shape().empty() ? 1 : (iter.numel() / iter.shape().back());
  EXPECT_EQ(ctx.tiles, expected_tiles);

  idx = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j, ++idx) {
      EXPECT_FLOAT_EQ(out_data[idx], a_data[idx] + b_data[idx]);
    }
  }
}

TEST(TensorIterBasicTest, ForEachCpuZeroSizeNoCallback) {
  auto out = make_contiguous_tensor({2, 0, 3});
  auto a   = make_contiguous_tensor({2, 0, 3});

  // TensorIter::binary_op enforces shape/dtype/device but allows zero-size.
  TensorIter iter = TensorIter::binary_op(out, a, a);
  BinaryLoopCtx ctx;
  iter.for_each_cpu(&binary_add_loop, &ctx);
  EXPECT_EQ(iter.numel(), 0);
  EXPECT_EQ(ctx.tiles, 0);
}

TEST(TensorIterBasicTest, ForEachCpuScalarRank0SingleCall) {
  // Rank-0 scalar: sizes={}, strides={}
  std::vector<int64_t> sizes;   // empty
  std::vector<int64_t> strides; // empty
  auto storage_out = make_storage_bytes(sizeof(float));
  auto storage_a   = make_storage_bytes(sizeof(float));
  TensorImpl out(storage_out, sizes, strides, /*storage_offset=*/0,
                 ScalarType::Float32, Device::cpu());
  TensorImpl a(storage_a, sizes, strides, /*storage_offset=*/0,
               ScalarType::Float32, Device::cpu());

  auto* out_data = static_cast<float*>(out.data());
  auto* a_data   = static_cast<float*>(a.data());
  *a_data = 3.0f;
  *out_data = 0.0f;

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.check_mem_overlap(false);
  TensorIter iter = cfg.build();
  UnaryLoopCtx ctx;
  iter.for_each_cpu(&unary_add_one_loop, &ctx);

  EXPECT_EQ(iter.ndim(), 0);
  EXPECT_EQ(iter.numel(), 1);
  EXPECT_EQ(ctx.tiles, 1);
  EXPECT_TRUE(iter.is_trivial_1d());
  EXPECT_FALSE(iter.is_reduction());
  EXPECT_EQ(iter.num_reduce_dims(), 0);
  EXPECT_TRUE(iter.reduce_dims().empty());
  EXPECT_FLOAT_EQ(*out_data, *a_data + 1.0f);
}

TEST(TensorIterBasicTest, NegativeStrideUnaryProducesReversedCopy) {
  // Allocate storage for 6 float elements.
  auto storage = make_storage_bytes(6 * sizeof(float));
  // Base tensor: contiguous [6]
  TensorImpl base(storage, {6}, {1}, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());
  auto* base_data = static_cast<float*>(base.data());
  for (int i = 0; i < 6; ++i) {
    base_data[i] = static_cast<float>(i);
  }

  // View with sizes=[6], strides=[-1], storage_offset=5 (reversed)
  TensorImpl flipped(storage, {6}, {-1}, /*storage_offset=*/5,
                     ScalarType::Float32, Device::cpu());
  auto out = make_contiguous_tensor({6});

  TensorIter iter = TensorIter::unary_op(out, flipped);
  UnaryLoopCtx ctx;
  iter.for_each_cpu(&unary_add_one_loop, &ctx);

  auto* out_data = static_cast<float*>(out.data());
  // out[i] = flipped[i] + 1  => base[5-i] + 1
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(out_data[i], base_data[5 - i] + 1.0f);
  }
}

TEST(TensorIterBasicTest, NullCallbackThrows) {
  auto out = make_contiguous_tensor({4});
  auto a   = make_contiguous_tensor({4});
  TensorIter iter = TensorIter::unary_op(out, a);
  EXPECT_THROW(iter.for_each_cpu(nullptr, nullptr), std::invalid_argument);
}

TEST(TensorIterBasicTest, ResizeOutputsFalseIsAcceptedForElementwise) {
  auto out = make_contiguous_tensor({4});
  auto a   = make_contiguous_tensor({4});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.resize_outputs(false);

  // behaves like the default (outputs are never resized and shapes must
  // already match the broadcasted logical shape).
  EXPECT_NO_THROW((void)cfg.build());
}

TEST(TensorIterBasicTest, FactoriesEnforceShapeAndDtypeAndDevice) {
  auto out = make_contiguous_tensor({4});
  auto a   = make_contiguous_tensor({3});  // shape mismatch

  EXPECT_THROW((void)TensorIter::unary_op(out, a), std::invalid_argument);

  // Dtype mismatch: reuse storage but fake dtype via new TensorImpl
  auto storage = make_storage_bytes(4 * sizeof(float));
  TensorImpl out_i(storage, {4}, {1}, /*storage_offset=*/0,
                   ScalarType::Int32, Device::cpu());
  TensorImpl a_f(storage, {4}, {1}, /*storage_offset=*/0,
                 ScalarType::Float32, Device::cpu());
  EXPECT_THROW((void)TensorIter::unary_op(out_i, a_f), std::invalid_argument);
}

TEST(TensorIterBasicTest, CheckAllSameFlagsDoNotRelaxInvariants) {
  auto out = make_contiguous_tensor({4});
  auto a   = make_contiguous_tensor({3});  // shape mismatch

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.check_all_same_dtype(false).check_all_same_device(false);

  EXPECT_THROW((void)cfg.build(), std::invalid_argument);
}

TEST(TensorIterBasicTest, IsTrivial1DStatus) {
  auto out1 = make_contiguous_tensor({4});
  auto a1   = make_contiguous_tensor({4});
  TensorIter it1 = TensorIter::unary_op(out1, a1);
  EXPECT_TRUE(it1.is_trivial_1d());  // ndim==1

  auto out2 = make_contiguous_tensor({0});
  auto a2   = make_contiguous_tensor({0});
  TensorIter it2 = TensorIter::unary_op(out2, a2);
  EXPECT_EQ(it2.numel(), 0);
  EXPECT_TRUE(it2.is_trivial_1d());

  auto out3 = make_contiguous_tensor({2, 3});
  auto a3   = make_contiguous_tensor({2, 3});
  TensorIter it3 = TensorIter::unary_op(out3, a3);
  // 2D non-degenerate iterator is not trivial-1D.
  EXPECT_FALSE(it3.is_trivial_1d());
}

TEST(TensorIterBasicTest, InternalOverlapInOutputIsAlwaysRejected) {
  auto storage = make_storage_bytes(4 * sizeof(float));
  TensorImpl overlapped(storage, {2}, {0}, /*storage_offset=*/0,
                        ScalarType::Float32, Device::cpu());
  auto in = make_contiguous_tensor({2});

  {
    TensorIterConfig cfg;
    cfg.add_output(vbt::core::OptionalTensorImplRef(&overlapped, /*defined=*/true));
    cfg.add_input(in);
    EXPECT_THROW((void)cfg.build(), std::invalid_argument);
  }

  {
    TensorIterConfig cfg;
    cfg.add_output(vbt::core::OptionalTensorImplRef(&overlapped, /*defined=*/true));
    cfg.add_input(in);
    cfg.check_mem_overlap(false);
    EXPECT_THROW((void)cfg.build(), std::invalid_argument);
  }
}

TEST(TensorIterBasicTest, CrossTensorOverlapRejectedWhenChecksOn) {
  auto storage = make_storage_bytes(5 * sizeof(float));
  TensorImpl out(storage, {4}, {1}, /*storage_offset=*/0,
                 ScalarType::Float32, Device::cpu());
  TensorImpl a(storage, {4}, {1}, /*storage_offset=*/1,
               ScalarType::Float32, Device::cpu());

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  EXPECT_THROW((void)cfg.build(), std::invalid_argument);
}

TEST(TensorIterBasicTest, CrossTensorOverlapAllowedWhenChecksOff) {
  auto storage = make_storage_bytes(5 * sizeof(float));
  TensorImpl out(storage, {4}, {1}, /*storage_offset=*/0,
                 ScalarType::Float32, Device::cpu());
  TensorImpl a(storage, {4}, {1}, /*storage_offset=*/1,
               ScalarType::Float32, Device::cpu());

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.check_mem_overlap(false);
  EXPECT_NO_THROW((void)cfg.build());
}

struct NullaryLoopCtx {
  int64_t tiles{0};
};

static void nullary_fill_loop(char** data,
                              const std::int64_t* strides,
                              std::int64_t size,
                              void* ctx_void) {
  auto* ctx = static_cast<NullaryLoopCtx*>(ctx_void);
  ++ctx->tiles;
  char* out_base = data[0];
  const std::int64_t out_stride = strides[0];
  for (std::int64_t i = 0; i < size; ++i) {
    auto* out = reinterpret_cast<float*>(out_base + i * out_stride);
    *out = 7.0f;
  }
}

TEST(TensorIterBasicTest, NullaryOpFillsOutput) {
  auto out = make_contiguous_tensor({2, 3});
  auto* out_data = static_cast<float*>(out.data());
  for (int i = 0; i < 6; ++i) {
    out_data[i] = 0.0f;
  }

  TensorIter iter = TensorIter::nullary_op(out);
  NullaryLoopCtx ctx;
  iter.for_each_cpu(&nullary_fill_loop, &ctx);

  const std::int64_t expected_tiles =
      iter.shape().empty() ? 1 : (iter.numel() / iter.shape().back());
  EXPECT_EQ(ctx.tiles, expected_tiles);

  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(out_data[i], 7.0f);
  }
}

TEST(TensorIterBasicTest, NullaryOpZeroSizeHasNoCallbacks) {
  auto out = make_contiguous_tensor({0, 3});

  TensorIter iter = TensorIter::nullary_op(out);
  NullaryLoopCtx ctx;
  iter.for_each_cpu(&nullary_fill_loop, &ctx);

  EXPECT_EQ(iter.numel(), 0);
  EXPECT_EQ(ctx.tiles, 0);
}

TEST(TensorIterBasicTest, NullaryOpEnforcesCpuDevice) {
  auto storage = make_storage_bytes(4 * sizeof(float));
  TensorImpl out_cuda(storage, {4}, {1}, /*storage_offset=*/0,
                      ScalarType::Float32, Device::cuda());
  EXPECT_THROW((void)TensorIter::nullary_op(out_cuda), std::invalid_argument);
}

TEST(TensorIterBasicTest, EnforceLinearIterationIsNotSupported) {
  auto out = make_contiguous_tensor({4});
  auto a   = make_contiguous_tensor({4});
  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.enforce_linear_iteration(true);
  EXPECT_THROW((void)cfg.build(), std::logic_error);
}

TEST(TensorIterBasicTest, AllowCpuScalarsIsSupported) {
  auto out = make_contiguous_tensor({4});
  auto a   = make_contiguous_tensor({4});
  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.allow_cpu_scalars(true);
  EXPECT_NO_THROW((void)cfg.build());
}

TEST(TensorIterBasicTest, StaticDeclarationsDoNotThrowImmediately) {
  TensorIterConfig cfg;
  EXPECT_NO_THROW((void)cfg.declare_static_dtype_and_device(ScalarType::Float32, Device::cpu()));
  EXPECT_NO_THROW((void)cfg.declare_static_dtype(ScalarType::Float32));
  EXPECT_NO_THROW((void)cfg.declare_static_device(Device::cpu()));
  std::vector<int64_t> shape{2, 3};
  EXPECT_NO_THROW((void)cfg.declare_static_shape(shape));
}

TEST(TensorIterBasicTest, MaxRankExceededThrows) {
  auto out = make_contiguous_tensor({2, 3, 4});
  auto a   = make_contiguous_tensor({2, 3, 4});

  TensorIterConfig cfg;
  cfg.set_max_rank(2);
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  EXPECT_THROW((void)cfg.build(), std::invalid_argument);
}

TEST(TensorIterBasicTest, MaxRankWithinLimitAllowsBuild) {
  auto out = make_contiguous_tensor({2, 1, 3, 1});
  auto a   = make_contiguous_tensor({2, 1, 3, 1});

  TensorIterConfig cfg;
  cfg.set_max_rank(2);
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  TensorIter iter = cfg.build();
  // Rank is the number of non-size-1 dims; here {2,1,3,1} -> 2D.
  EXPECT_EQ(iter.ndim(), 2);
  EXPECT_EQ(iter.shape().size(), 2u);
}

TEST(TensorIterBasicTest, FactoriesRejectMismatchedCpuDeviceIndex) {
  auto storage = make_storage_bytes(4 * sizeof(float));
  TensorImpl out(storage, {4}, {1}, /*storage_offset=*/0,
                 ScalarType::Float32, Device::cpu(0));
  TensorImpl a(storage, {4}, {1}, /*storage_offset=*/0,
               ScalarType::Float32, Device::cpu(1));
  EXPECT_THROW((void)TensorIter::unary_op(out, a), std::invalid_argument);
}

TEST(TensorIterBasicTest, BuildRequiresAtLeastOneOutput) {
  auto a = make_contiguous_tensor({4});

  TensorIterConfig cfg;
  cfg.add_input(a);

  bool threw = false;
  try {
    (void)cfg.build();
  } catch (const std::invalid_argument& e) {
    threw = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("at least one output"), std::string::npos);
  }
  EXPECT_TRUE(threw);
}


TEST(TensorIterBasicTest, BuildRejectsShapeMismatch) {
  auto out = make_contiguous_tensor({4});
  auto a   = make_contiguous_tensor({3});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);

  bool threw = false;
  try {
    (void)cfg.build();
  } catch (const std::invalid_argument& e) {
    threw = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("same shape"), std::string::npos);
  }
  EXPECT_TRUE(threw);
}

TEST(TensorIterBasicTest, BuildRejectsDtypeMismatch) {
  auto storage = make_storage_bytes(4 * sizeof(float));
  TensorImpl out(storage, {4}, {1}, /*storage_offset=*/0,
                 ScalarType::Float32, Device::cpu());
  TensorImpl a(storage, {4}, {1}, /*storage_offset=*/0,
               ScalarType::Int32, Device::cpu());

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);

  bool threw = false;
  try {
    (void)cfg.build();
  } catch (const std::invalid_argument& e) {
    threw = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("same dtype"), std::string::npos);
  }
  EXPECT_TRUE(threw);
}

TEST(TensorIterBasicTest, BuildRejectsCpuDeviceMismatch) {
  auto storage = make_storage_bytes(4 * sizeof(float));
  TensorImpl out(storage, {4}, {1}, /*storage_offset=*/0,
                 ScalarType::Float32, Device::cpu(0));
  TensorImpl a(storage, {4}, {1}, /*storage_offset=*/0,
               ScalarType::Float32, Device::cpu(1));

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);

  bool threw = false;
  try {
    (void)cfg.build();
  } catch (const std::invalid_argument& e) {
    threw = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("same device"), std::string::npos);
  }
  EXPECT_TRUE(threw);
}
