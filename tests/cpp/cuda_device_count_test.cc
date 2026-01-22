// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "vbt/cuda/device.h"

TEST(CudaBuild, DeviceCountNonNegative) {
  EXPECT_GE(vbt::cuda::device_count(), 0);
}
