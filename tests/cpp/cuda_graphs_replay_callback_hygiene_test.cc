// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "vbt/cuda/graphs.h"
#include "vbt/cuda/stream.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace vbt::cuda;

// Ensure that the CUDA Graphs replay callback remains callback-safe and
// does not attempt to call disallowed CUDA Runtime APIs while running
// under cudaLaunchHostFunc.
TEST(CUDAGraphReplayCallbackHygieneTest, NoCudaErrorNotPermittedFromCallback) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  Stream s = getStreamFromPool(false, dev);
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());

  constexpr int kIters = 8;
  for (int i = 0; i < kIters; ++i) {
    float* d = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d), sizeof(float)), cudaSuccess);

    {
      CUDAGraph g;

      g.capture_begin(s);

      int host_value = 1 + i;
      ASSERT_EQ(cudaMemcpyAsync(d,
                                &host_value,
                                sizeof(int),
                                cudaMemcpyHostToDevice,
                                raw),
                cudaSuccess);

      g.capture_end();
      g.instantiate();
      g.replay();
    }

    // Wait for the replay and callback to complete.
    ASSERT_EQ(cudaStreamSynchronize(raw), cudaSuccess);

    // The callback must not have called any CUDA Runtime APIs that result
    // in cudaErrorNotPermitted (or any other error).
    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << "Unexpected CUDA error after graph replay/callback";

    ASSERT_EQ(cudaFree(d), cudaSuccess);
  }
#endif
}
