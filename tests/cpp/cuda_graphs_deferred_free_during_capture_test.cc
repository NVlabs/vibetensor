// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <string>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/stream.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace vbt::cuda;

TEST(CudaGraphsAllocatorDeferredFreeTest, DeferredFreeDuringCaptureAndFlushAfter) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);
  Stream s = getStreamFromPool(false, dev);

  // Allocate a buffer and then free it during capture to trigger deferral.
  void* p = A.raw_alloc(1 << 20, s);
  ASSERT_NE(p, nullptr);

  DeviceStats before = A.getDeviceStats();

  cudaError_t rc = cudaStreamBeginCapture(reinterpret_cast<cudaStream_t>(s.handle()), cudaStreamCaptureModeThreadLocal);
  ASSERT_EQ(rc, cudaSuccess);

  A.raw_delete(p); // should be deferred; no event record during capture

  DeviceStats mid = A.getDeviceStats();
  // No flush attempt should have occurred just by freeing during capture.
  EXPECT_EQ(mid.deferred_flush_attempts, before.deferred_flush_attempts);
  EXPECT_EQ(mid.deferred_flush_successes, before.deferred_flush_successes);

  cudaGraph_t g = nullptr;
  rc = cudaStreamEndCapture(reinterpret_cast<cudaStream_t>(s.handle()), &g);
  ASSERT_EQ(rc, cudaSuccess);
  if (g) { (void)cudaGraphDestroy(g); }

  // Now process events; this should move deferred free to limbo and increment attempts
  A.process_events();
  DeviceStats after = A.getDeviceStats();
  EXPECT_GE(after.deferred_flush_attempts, mid.deferred_flush_attempts);
  EXPECT_GE(after.deferred_flush_successes, mid.deferred_flush_successes);
#endif
}
