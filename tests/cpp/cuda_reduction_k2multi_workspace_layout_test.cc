// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <limits>

#include "vbt/cuda/reduction_workspace.h"

using vbt::cuda::reduction::K2MultiWorkspaceLayout;
using vbt::cuda::reduction::compute_k2multi_workspace_layout;
using vbt::cuda::reduction::kCudaReductionK2MultiAlign;

TEST(CudaReductionK2MultiWorkspaceLayoutTest, HappyPathAlignment) {
  K2MultiWorkspaceLayout layout;
  ASSERT_TRUE(compute_k2multi_workspace_layout(/*out_numel=*/2, /*ctas_per_output=*/4, /*itemsize=*/4, &layout));

  EXPECT_EQ(layout.partials_bytes, 32u);
  EXPECT_EQ(layout.sema_off, 256u);
  EXPECT_EQ(layout.semaphores_bytes, 8u);
  EXPECT_EQ(layout.total_bytes, 512u);

  EXPECT_EQ(layout.sema_off % kCudaReductionK2MultiAlign, 0u);
  EXPECT_EQ(layout.total_bytes % kCudaReductionK2MultiAlign, 0u);
}

TEST(CudaReductionK2MultiWorkspaceLayoutTest, RequiresAtLeastTwoCtas) {
  K2MultiWorkspaceLayout layout;
  EXPECT_FALSE(compute_k2multi_workspace_layout(/*out_numel=*/2, /*ctas_per_output=*/1, /*itemsize=*/4, &layout));
}

TEST(CudaReductionK2MultiWorkspaceLayoutTest, DetectsOverflow) {
  K2MultiWorkspaceLayout layout;
  const std::int64_t out_numel = std::numeric_limits<std::int64_t>::max();
  EXPECT_FALSE(compute_k2multi_workspace_layout(out_numel, /*ctas_per_output=*/2, /*itemsize=*/1, &layout));
}

TEST(CudaReductionK2MultiWorkspaceLayoutTest, AlreadyAlignedPartials) {
  K2MultiWorkspaceLayout layout;
  ASSERT_TRUE(compute_k2multi_workspace_layout(/*out_numel=*/32, /*ctas_per_output=*/2, /*itemsize=*/4, &layout));

  EXPECT_EQ(layout.partials_bytes, 256u);
  EXPECT_EQ(layout.sema_off, 256u);
  EXPECT_EQ(layout.semaphores_bytes, 128u);
  EXPECT_EQ(layout.total_bytes, 512u);

  EXPECT_EQ(layout.sema_off % kCudaReductionK2MultiAlign, 0u);
  EXPECT_EQ(layout.total_bytes % kCudaReductionK2MultiAlign, 0u);
}

TEST(CudaReductionK2MultiWorkspaceLayoutTest, RejectsNullOutputPointer) {
  EXPECT_FALSE(compute_k2multi_workspace_layout(/*out_numel=*/2, /*ctas_per_output=*/2, /*itemsize=*/4, nullptr));
}

TEST(CudaReductionK2MultiWorkspaceLayoutTest, RejectsNegativeOutNumel) {
  K2MultiWorkspaceLayout layout{123, 123, 123, 123};
  EXPECT_FALSE(compute_k2multi_workspace_layout(/*out_numel=*/-1, /*ctas_per_output=*/2, /*itemsize=*/4, &layout));
  EXPECT_EQ(layout.partials_bytes, 0u);
  EXPECT_EQ(layout.sema_off, 0u);
  EXPECT_EQ(layout.semaphores_bytes, 0u);
  EXPECT_EQ(layout.total_bytes, 0u);
}

TEST(CudaReductionK2MultiWorkspaceLayoutTest, RejectsZeroItemsize) {
  K2MultiWorkspaceLayout layout;
  EXPECT_FALSE(compute_k2multi_workspace_layout(/*out_numel=*/2, /*ctas_per_output=*/2, /*itemsize=*/0, &layout));
}
