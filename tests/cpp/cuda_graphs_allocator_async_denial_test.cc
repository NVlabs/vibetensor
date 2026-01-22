// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/graphs.h"
#include "vbt/cuda/stream.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace vbt::cuda;

namespace {
static inline bool has_substr(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}
}

TEST(CudaGraphsAllocatorAsyncBackendTest, DeniesAllocationsDuringCaptureWithCanonicalSubstring) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  // Force async backend for this process/binary
  setenv("VBT_CUDA_ALLOC_CONF", "backend=cudamallocasync", 1);

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);
  Stream s = getStreamFromPool(false, dev);

  GraphCounters before = cuda_graphs_counters();

  ASSERT_EQ(cudaStreamBeginCapture(reinterpret_cast<cudaStream_t>(s.handle()), cudaStreamCaptureModeThreadLocal), cudaSuccess);

  // Deny stream-overload
  try {
    (void)A.raw_alloc(4096, s);
    FAIL() << "Expected denial";
  } catch (const std::runtime_error& e) {
    EXPECT_TRUE(has_substr(e.what(), std::string(kErrAllocatorCaptureDenied)));
  }

  // Deny no-stream overload (set current to s)
  setCurrentStream(s);
  try {
    (void)A.raw_alloc(4096);
    FAIL() << "Expected denial";
  } catch (const std::runtime_error& e) {
    EXPECT_TRUE(has_substr(e.what(), std::string(kErrAllocatorCaptureDenied)));
  }

  cudaGraph_t g = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(reinterpret_cast<cudaStream_t>(s.handle()), &g), cudaSuccess);
  if (g) (void)cudaGraphDestroy(g);

  GraphCounters after = cuda_graphs_counters();
  EXPECT_EQ(after.allocator_capture_denied,
            before.allocator_capture_denied + 2);
#endif
}
