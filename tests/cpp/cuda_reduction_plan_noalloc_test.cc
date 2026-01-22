// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "vbt/core/tensor_iterator/core.h"
#include "vbt/cuda/reduction_env.h"
#include "vbt/cuda/reduction_plan.h"

using vbt::core::DeviceStrideMeta;
using vbt::cuda::reduction::CudaReduceIneligibleReason;
using vbt::cuda::reduction::build_cuda_reduce_plan_noalloc;

namespace {

static DeviceStrideMeta make_meta(std::int64_t ndim,
                                 std::initializer_list<std::int64_t> sizes,
                                 std::initializer_list<std::int64_t> strides) {
  DeviceStrideMeta m{};
  m.ndim = ndim;
  std::size_t i = 0;
  for (auto v : sizes) {
    m.sizes[i++] = v;
  }
  i = 0;
  for (auto v : strides) {
    m.strides[i++] = v;
  }
  return m;
}

}  // namespace

TEST(CudaReductionPlanNoallocTest, OutMetaReducedStrideZeroDoesNotTriggerRedStrideZero) {
  // OUTMETA0_OK: output reduced strides are 0 by TI design, but RedStrideZero
  // must consult the *input* meta only.
  DeviceStrideMeta in_meta = make_meta(/*ndim=*/2, /*sizes=*/{2, 3}, /*strides=*/{3, 1});
  DeviceStrideMeta out_meta = make_meta(/*ndim=*/2, /*sizes=*/{2, 3}, /*strides=*/{1, 0});
  const std::array<std::int64_t, 1> reduce_dims{1};

  auto r = build_cuda_reduce_plan_noalloc(out_meta, in_meta, reduce_dims,
                                         /*out_numel=*/2, /*slice_len=*/3);
  EXPECT_EQ(r.ineligible_reason, CudaReduceIneligibleReason::None);
  EXPECT_EQ(r.plan.out_numel, 2);
  EXPECT_EQ(r.plan.slice_len, 3);
  EXPECT_EQ(r.plan.iter_ndim, 2);
  EXPECT_EQ(r.plan.kept_ndim, 1);
  EXPECT_EQ(r.plan.kept_sizes[0], 2);
  EXPECT_EQ(r.plan.kept_in_strides[0], 3);
  EXPECT_EQ(r.plan.kept_out_strides[0], 1);
  EXPECT_EQ(r.plan.red_linear_stride, 1);
}

TEST(CudaReductionPlanNoallocTest, ActiveDimsSizeOneKeptNegativeStrideIgnored) {
  // ACTIVE1_KEPT_NEG_IGNORED: negative stride in a kept dim with size==1 must
  // not trigger KeptNegativeStride.
  DeviceStrideMeta in_meta = make_meta(/*ndim=*/3, /*sizes=*/{1, 3, 4}, /*strides=*/{-12, 4, 1});
  DeviceStrideMeta out_meta = make_meta(/*ndim=*/3, /*sizes=*/{1, 3, 4}, /*strides=*/{3, 1, 0});
  const std::array<std::int64_t, 1> reduce_dims{2};

  auto r = build_cuda_reduce_plan_noalloc(out_meta, in_meta, reduce_dims,
                                         /*out_numel=*/3, /*slice_len=*/4);
  EXPECT_EQ(r.ineligible_reason, CudaReduceIneligibleReason::None);
}

TEST(CudaReductionPlanNoallocTest, RedStrideZero) {
  DeviceStrideMeta in_meta = make_meta(/*ndim=*/2, /*sizes=*/{2, 3}, /*strides=*/{3, 0});
  DeviceStrideMeta out_meta = make_meta(/*ndim=*/2, /*sizes=*/{2, 3}, /*strides=*/{1, 0});
  const std::array<std::int64_t, 1> reduce_dims{1};

  auto r = build_cuda_reduce_plan_noalloc(out_meta, in_meta, reduce_dims,
                                         /*out_numel=*/2, /*slice_len=*/3);
  EXPECT_EQ(r.ineligible_reason, CudaReduceIneligibleReason::RedStrideZero);
}

TEST(CudaReductionPlanNoallocTest, RedMultiDimNegativeStride) {
  DeviceStrideMeta in_meta = make_meta(/*ndim=*/3, /*sizes=*/{2, 3, 4}, /*strides=*/{12, 4, -1});
  DeviceStrideMeta out_meta = make_meta(/*ndim=*/3, /*sizes=*/{2, 3, 4}, /*strides=*/{1, 0, 0});
  const std::array<std::int64_t, 2> reduce_dims{1, 2};

  auto r = build_cuda_reduce_plan_noalloc(out_meta, in_meta, reduce_dims,
                                         /*out_numel=*/2, /*slice_len=*/12);
  EXPECT_EQ(r.ineligible_reason, CudaReduceIneligibleReason::RedMultiDimNegativeStride);
}

TEST(CudaReductionPlanNoallocTest, KeptNegativeStride) {
  DeviceStrideMeta in_meta = make_meta(/*ndim=*/2, /*sizes=*/{2, 3}, /*strides=*/{-3, 1});
  DeviceStrideMeta out_meta = make_meta(/*ndim=*/2, /*sizes=*/{2, 3}, /*strides=*/{1, 0});
  const std::array<std::int64_t, 1> reduce_dims{1};

  auto r = build_cuda_reduce_plan_noalloc(out_meta, in_meta, reduce_dims,
                                         /*out_numel=*/2, /*slice_len=*/3);
  EXPECT_EQ(r.ineligible_reason, CudaReduceIneligibleReason::KeptNegativeStride);
}

TEST(CudaReductionPlanNoallocTest, RedNotLinearizable) {
  DeviceStrideMeta in_meta = make_meta(/*ndim=*/3, /*sizes=*/{2, 3, 4}, /*strides=*/{12, 4, 1});
  DeviceStrideMeta out_meta = make_meta(/*ndim=*/3, /*sizes=*/{2, 3, 4}, /*strides=*/{0, 1, 0});
  const std::array<std::int64_t, 2> reduce_dims{0, 2};

  auto r = build_cuda_reduce_plan_noalloc(out_meta, in_meta, reduce_dims,
                                         /*out_numel=*/3, /*slice_len=*/8);
  EXPECT_EQ(r.ineligible_reason, CudaReduceIneligibleReason::RedNotLinearizable);
}

TEST(CudaReductionPlanNoallocTest, OverflowInvalidReduceDimsOrdering) {
  DeviceStrideMeta in_meta = make_meta(/*ndim=*/2, /*sizes=*/{2, 3}, /*strides=*/{3, 1});
  DeviceStrideMeta out_meta = make_meta(/*ndim=*/2, /*sizes=*/{2, 3}, /*strides=*/{1, 0});
  const std::array<std::int64_t, 2> reduce_dims{1, 0};

  auto r = build_cuda_reduce_plan_noalloc(out_meta, in_meta, reduce_dims,
                                         /*out_numel=*/2, /*slice_len=*/3);
  EXPECT_EQ(r.ineligible_reason, CudaReduceIneligibleReason::Overflow);
}
