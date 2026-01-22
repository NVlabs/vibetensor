// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <stdexcept>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/graphs.h"
#include "vbt/cuda/stream.h"
#include "cuda_graphs_allocator_test_helpers.h"

using namespace vbt::cuda;
namespace testonly = vbt::cuda::testonly;

// Pool GC helper should demote pool-owned segments back to the global pool
// without touching fraction/GC counters.
TEST(CudaGraphsAllocatorPoolInvariantsTest,
     PoolGCDemotesWithoutTouchingGlobalGCCounters) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
#if !defined(VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "VBT_INTERNAL_TESTS required";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (!testonly::has_native_backend(dev)) {
    GTEST_SKIP() << "Requires native backend";
  }

  testonly::quiesce_allocator_for_setup(dev);
  testonly::quiesce_graphs_for_snapshots(dev);

  // Create a pool and prewarm it so that pool-tagged segments exist.
  MempoolId id = Allocator::create_pool_id(dev);

  Stream s = getStreamFromPool(false, dev);
  const std::size_t kBytes = 1u << 20;  // 1 MiB

  try {
    Allocator::prewarm_graph_pool_for_stream(dev, id, s, kBytes, /*min_blocks=*/1);
  } catch (...) {
    GTEST_SKIP() << "prewarm_graph_pool_for_stream failed";
  }

  testonly::quiesce_graphs_for_snapshots(dev);

  testonly::CombinedSnapshot before_cs = testonly::take_snapshot(dev);

  bool           has_pool_before = false;
  std::uint64_t  pool_reserved_before = 0;
  for (const auto& gp : before_cs.pools) {
    if (gp.id.dev == dev && gp.id.id == id.id) {
      has_pool_before = true;
      pool_reserved_before = gp.bytes_reserved;
    }
  }

  if (!has_pool_before || pool_reserved_before == 0) {
    GTEST_SKIP() << "Prewarm did not create any pool-owned segments";
  }

  std::uint64_t gc_passes_before = before_cs.stats.gc_passes;
  std::uint64_t gc_bytes_before  = before_cs.stats.gc_reclaimed_bytes;

  // Force pool GC/demotion via test-only helper; this must not touch gc_*.
  testonly::allocator_debug_gc_pool_now(dev, id);

  testonly::CombinedSnapshot after_cs = testonly::take_snapshot(dev);

  bool           has_pool_after = false;
  std::uint64_t  pool_reserved_after = 0;
  for (const auto& gp : after_cs.pools) {
    if (gp.id.dev == dev && gp.id.id == id.id) {
      has_pool_after = true;
      pool_reserved_after = gp.bytes_reserved;
    }
  }

  // Pool should either disappear from snapshots or have zero reserved bytes.
  if (has_pool_after) {
    EXPECT_EQ(pool_reserved_after, 0u);
  }

  EXPECT_EQ(after_cs.stats.gc_passes, gc_passes_before);
  EXPECT_EQ(after_cs.stats.gc_reclaimed_bytes, gc_bytes_before);
#endif  // VBT_INTERNAL_TESTS
#endif  // VBT_WITH_CUDA
}
