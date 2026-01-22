// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <string>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/graphs.h"
#include "vbt/cuda/stream.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace vbt::cuda;

TEST(CudaGraphsAllocatorRoutingAllowTagTest, GuardCaptureAllowsAndTagsBothOverloads) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);
  Stream s = getStreamFromPool(false, dev);

  // Pre-seed two free blocks to satisfy both allocations under capture
  void* seed1 = A.raw_alloc(1 << 20, s);
  ASSERT_NE(seed1, nullptr);
  setCurrentStream(s);
  A.raw_delete(seed1);
  void* seed2 = A.raw_alloc(1 << 20, s);
  ASSERT_NE(seed2, nullptr);
  setCurrentStream(s);
  A.raw_delete(seed2);

  MempoolId id = Allocator::create_pool_id(dev);
  Allocator::retain_pool(dev, id);

  // Begin routing then begin capture
  auto guard = Allocator::begin_allocate_to_pool(dev, id);
  ASSERT_EQ(cudaStreamBeginCapture(reinterpret_cast<cudaStream_t>(s.handle()), cudaStreamCaptureModeThreadLocal), cudaSuccess);

  // Stream-overload alloc should succeed and be tagged
  void* a = A.raw_alloc(1 << 20, s);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(A.debug_block_pool_id(a), id.id);

  // No-stream overload uses current stream; set to s and allocate
  setCurrentStream(s);
  void* b = A.raw_alloc(1 << 20);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(A.debug_block_pool_id(b), id.id);

  cudaGraph_t g = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(reinterpret_cast<cudaStream_t>(s.handle()), &g), cudaSuccess);
  if (g) (void)cudaGraphDestroy(g);

  // Cleanup
  A.raw_delete(a);
  A.raw_delete(b);
  guard.end();
  Allocator::release_pool(dev, id);
#endif
}
