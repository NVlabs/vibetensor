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

Allocator& alloc0() {
  return Allocator::get(0);
}

} // namespace

TEST(CudaGraphsAllocatorPerPoolFreeListsTest, GlobalAllocDoesNotReusePoolTaggedBlocks) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  Allocator& A = alloc0();
  Stream s = getStreamFromPool(false, dev);
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());

  const std::size_t kSize = 1 << 20;  // 1 MiB block

  // 1) Seed a global free block on this stream.
  void* global_seed = A.raw_alloc(kSize, s);
  ASSERT_NE(global_seed, nullptr);
  setCurrentStream(s);
  A.raw_delete(global_seed);
  EXPECT_EQ(A.debug_block_pool_id(global_seed), 0u);

  // 2) Create a graph-private pool and retain it so its blocks remain owned
  // by the pool while free.
  MempoolId id = Allocator::create_pool_id(dev);
  Allocator::retain_pool(dev, id);

  auto guard = Allocator::begin_allocate_to_pool(dev, id);
  ASSERT_TRUE(guard.active());

  // Begin capture on the stream so allocator routing is active.
  ASSERT_EQ(cudaStreamBeginCapture(raw, cudaStreamCaptureModeThreadLocal), cudaSuccess);

  // Allocate a block under capture; it should be tagged with the pool id and
  // must come from the pre-seeded global free list.
  void* pooled = A.raw_alloc(kSize, s);
  ASSERT_NE(pooled, nullptr);
  EXPECT_EQ(A.debug_block_pool_id(pooled), id.id);

  cudaGraph_t g = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(raw, &g), cudaSuccess);
  if (g) {
    (void)cudaGraphDestroy(g);
  }

  // End routing but keep the pool retained so its blocks remain pool-owned.
  guard.end();

  // Free the pooled allocation; the free lists now contain one global block
  // and one pool-tagged block for this size.
  setCurrentStream(s);
  A.raw_delete(pooled);
  EXPECT_EQ(A.debug_block_pool_id(pooled), id.id);

  // 3) A global allocation (no capture, no routing) must only reuse the
  // global block, never the pool-tagged block.
  void* global_again = A.raw_alloc(kSize, s);
  ASSERT_NE(global_again, nullptr);
  EXPECT_EQ(A.debug_block_pool_id(global_again), 0u);
  EXPECT_EQ(A.debug_block_pool_id(pooled), id.id);

  // Cleanup: free the global allocation and drop the pool ref so it can be GC'd.
  setCurrentStream(s);
  A.raw_delete(global_again);
  Allocator::release_pool(dev, id);
#endif
}
