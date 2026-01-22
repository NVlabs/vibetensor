// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
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

TEST(CudaGraphsAllocatorRoutingTest, GuardAllowsAllocDuringCaptureAndNoGuardDenies) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);
  Stream s = getStreamFromPool(false, dev);

  // Pre-seed a free block on this stream so capture-time allocation can reuse.
  void* seed = A.raw_alloc(1 << 20, s);
  ASSERT_NE(seed, nullptr);
  A.raw_delete(seed);

  // Create and retain a pool handle
  MempoolId id = Allocator::create_pool_id(dev);
  Allocator::retain_pool(dev, id);

  // Enable routing via guard; verify device routing flag, allocate (no capture) and ensure untagged
  auto guard = Allocator::begin_allocate_to_pool(dev, id);
  ASSERT_TRUE(guard.active());
  EXPECT_TRUE(A.debug_device_routing_active());

  void* p = A.raw_alloc(1 << 20, s);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(A.debug_block_pool_id(p), 0u);
  A.raw_delete(p);

  guard.end();
  Allocator::release_pool(dev, id);

  // Case: no guard + capture should deny
  cudaError_t rc = cudaStreamBeginCapture(reinterpret_cast<cudaStream_t>(s.handle()), cudaStreamCaptureModeThreadLocal);
  ASSERT_EQ(rc, cudaSuccess) << "BeginCapture failed (case 3)";
  try {
    (void)A.raw_alloc(4096, s);
    FAIL() << "Expected allocator capture denial";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_TRUE(has_substr(msg, std::string(kErrAllocatorCaptureDenied)));
  }
  cudaGraph_t g = nullptr;
  rc = cudaStreamEndCapture(reinterpret_cast<cudaStream_t>(s.handle()), &g);
  ASSERT_EQ(rc, cudaSuccess);
  if (g) { (void)cudaGraphDestroy(g); }
#endif
}
