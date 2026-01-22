// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <string>
#include "vbt/cuda/graphs.h"

TEST(CUDAGraphsErrorTextTest, MessagesPinnedToDesign) {
  static constexpr const char* kDefaultMsg =
      "CUDA Graph capture on the default stream is not allowed. Please create a non-default CUDA stream and capture on it (e.g., vc.Stream()).";
  static constexpr const char* kNestedMsg =
      "nested CUDA graph capture is not allowed";
  static constexpr const char* kAllocDeniedMsg =
      "cuda allocator: allocations are forbidden during CUDA graph capture";
  static constexpr const char* kInstantiateInvalidMsg =
      "instantiate called in invalid state";
  static constexpr const char* kReplayInvalidMsg =
      "replay called in invalid state";
  static constexpr const char* kReplayNestedMsg =
      "Nested ReplayGuard not supported";
  static constexpr const char* kDeviceMismatchPrefix =
      "CUDA Graph device mismatch";
  static constexpr const char* kResetInvalidMsg =
      "reset called in invalid state";
  static constexpr const char* kResetInflightMsg =
      "reset called while replays are in flight";
  EXPECT_STREQ(vbt::cuda::kErrDefaultStreamCaptureBan, kDefaultMsg);
  EXPECT_STREQ(vbt::cuda::kErrNestedCaptureBan, kNestedMsg);
  EXPECT_STREQ(vbt::cuda::kErrAllocatorCaptureDenied, kAllocDeniedMsg);
  EXPECT_STREQ(vbt::cuda::kErrGraphInstantiateInvalidState, kInstantiateInvalidMsg);
  EXPECT_STREQ(vbt::cuda::kErrGraphReplayInvalidState, kReplayInvalidMsg);
  EXPECT_STREQ(vbt::cuda::kErrReplayNestedGuard, kReplayNestedMsg);
  EXPECT_STREQ(vbt::cuda::kErrGraphDeviceMismatchPrefix, kDeviceMismatchPrefix);
  EXPECT_STREQ(vbt::cuda::kErrGraphResetInvalidState, kResetInvalidMsg);
  EXPECT_STREQ(vbt::cuda::kErrGraphResetInflightDenied, kResetInflightMsg);
}

TEST(CUDAGraphsCountersTest, ResetCountersAreNonNegative) {
  vbt::cuda::GraphCounters c = vbt::cuda::cuda_graphs_counters();
  EXPECT_GE(c.graphs_reset, 0u);
  EXPECT_GE(c.reset_invalid_state, 0u);
  EXPECT_GE(c.reset_inflight_denied, 0u);
  EXPECT_GE(c.reset_errors, 0u);
}
