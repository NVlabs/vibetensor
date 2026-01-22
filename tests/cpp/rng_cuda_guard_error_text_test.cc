// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/rng/graph_capture.h"

// Simple pin test to ensure the CUDA RNG mutation guard error text remains
// stable. Python tests rely on this exact string via the exported
// _ERR_CUDA_RNG_MUTATION_DURING_CAPTURE attribute on the vibetensor._C module.
TEST(RngCudaGraphCaptureCore, CudaRngMutationGuardErrorTextPinned) {
  EXPECT_STREQ(
      vbt::rng::graph_capture::kErrCudaRngMutationDuringCapture,
      "rng: generator state mutation is forbidden while CUDA Graph capture is active");
}
