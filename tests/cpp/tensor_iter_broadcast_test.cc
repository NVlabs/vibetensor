// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>

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

struct CountTilesCtx {
  int tiles{0};
};

static void count_tiles_loop(char** /*data*/, const std::int64_t* /*strides*/,
                             std::int64_t /*size*/, void* ctx_void) {
  auto* ctx = static_cast<CountTilesCtx*>(ctx_void);
  ++ctx->tiles;
}

}  // namespace

TEST(TensorIterBroadcastTest, ScalarAndVectorBroadcastStrides) {
  auto out = make_contiguous_tensor({4});
  // Scalar: rank-0 tensor
  auto scalar = make_contiguous_tensor({});
  auto vec = make_contiguous_tensor({4});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(scalar);
  cfg.add_input(vec);
  TensorIter iter = cfg.build();

  EXPECT_EQ(iter.ndim(), 1);
  ASSERT_EQ(iter.shape().size(), 1u);
  EXPECT_EQ(iter.shape()[0], 4);
  EXPECT_EQ(iter.noutputs(), 1);
  EXPECT_EQ(iter.ninputs(), 2);

  const auto& op_out    = iter.operand(0);
  const auto& op_scalar = iter.operand(1);
  const auto& op_vec    = iter.operand(2);

  ASSERT_EQ(op_out.dim_stride_bytes.size(), 1u);
  ASSERT_EQ(op_scalar.dim_stride_bytes.size(), 1u);
  ASSERT_EQ(op_vec.dim_stride_bytes.size(), 1u);

  const std::int64_t item_b = static_cast<std::int64_t>(out.itemsize());
  EXPECT_EQ(op_out.dim_stride_bytes[0], item_b);
  EXPECT_EQ(op_vec.dim_stride_bytes[0], item_b);
  // Broadcasted scalar: stride must be zero.
  EXPECT_EQ(op_scalar.dim_stride_bytes[0], 0);
}

TEST(TensorIterBroadcastTest, ZeroSizeBroadcastNoCallbacks) {
  auto out = make_contiguous_tensor({0, 3});
  auto a   = make_contiguous_tensor({0, 3});
  auto b   = make_contiguous_tensor({1, 3});  // broadcast over dim 0

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.add_input(b);
  TensorIter iter = cfg.build();

  ASSERT_EQ(iter.shape().size(), 2u);
  EXPECT_EQ(iter.shape()[0], 0);
  EXPECT_EQ(iter.shape()[1], 3);
  EXPECT_EQ(iter.numel(), 0);

  CountTilesCtx ctx;
  iter.for_each_cpu(&count_tiles_loop, &ctx);
  EXPECT_EQ(ctx.tiles, 0);
}

TEST(TensorIterBroadcastTest, InvalidZeroBroadcastThrows) {
  auto out = make_contiguous_tensor({0, 3});
  auto a   = make_contiguous_tensor({0, 3});
  auto b   = make_contiguous_tensor({2, 3});  // {0,3} vs {2,3} is invalid

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.add_input(b);

  bool threw = false;
  try {
    (void)cfg.build();
  } catch (const std::invalid_argument& e) {
    threw = true;
    std::string msg = e.what();
    // Message should mention broadcast shape mismatch.
    EXPECT_NE(msg.find("broadcast"), std::string::npos);
  }
  EXPECT_TRUE(threw);
}
