// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "vbt/cuda/graphs.h"
#include "vbt/cuda/stream.h"

using vbt::cuda::CaptureStatus;

TEST(CUDAGraphsStatusHelperTest, DefaultStreamIsNotCapturing) {
  // Default stream should not be capturing in a fresh test process
  auto ds = vbt::cuda::getDefaultStream(-1);
  EXPECT_EQ(vbt::cuda::streamCaptureStatus(ds), CaptureStatus::None);
}
