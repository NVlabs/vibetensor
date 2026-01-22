// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/cub.h"
#include "vbt/cuda/stream.h"

#if !VBT_INTERNAL_TESTS
#  error "cuda_cub_temp_zero_test requires VBT_INTERNAL_TESTS"
#endif

namespace {

TEST(CudaCubWrappersTest, TempStorageBytesZeroUsesNonNullPointer) {
  auto& alloc = vbt::cuda::Allocator::get(0);
  auto stream = vbt::cuda::getCurrentStream(/*device=*/0);

  bool ok = vbt::cuda::cub::testonly::temp_storage_bytes0_requires_nonnull(alloc, stream);
  EXPECT_TRUE(ok);
}

} // namespace
