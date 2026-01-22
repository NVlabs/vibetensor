// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/graphs.h"
#include "vbt/cuda/stream.h"
#include "cuda_graphs_allocator_test_helpers.h"

using namespace vbt::cuda;
namespace testonly = vbt::cuda::testonly;

class CudaAllocatorStatsSnapshotTest : public ::testing::Test {
 protected:
  void SetUp() override {
#if !VBT_WITH_CUDA
    GTEST_SKIP() << "CUDA required";
#else
    if (device_count() == 0) {
      GTEST_SKIP() << "No CUDA device";
    }
    dev_ = 0;
    testonly::quiesce_allocator_for_setup(dev_);
    testonly::quiesce_graphs_for_snapshots(dev_);
#endif
  }

  DeviceIndex dev_{0};
};

TEST_F(CudaAllocatorStatsSnapshotTest, GlobalInequalitiesNative) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
#if !defined(VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "VBT_INTERNAL_TESTS required";
#else
  if (!testonly::has_native_backend(dev_)) {
    GTEST_SKIP() << "Requires native backend";
  }

  Allocator& A = Allocator::get(dev_);
  Stream s = getStreamFromPool(false, dev_);

  // Small allocation pattern to exercise allocator without stressing memory.
  std::vector<void*> ptrs;
  auto try_alloc = [&](std::size_t nbytes) {
    void* p = nullptr;
    try {
      p = A.raw_alloc(nbytes, s);
    } catch (...) {
      p = nullptr;
    }
    if (p) ptrs.push_back(p);
  };

  try_alloc(1u << 20);  // 1 MiB
  try_alloc(2u << 20);  // 2 MiB

  if (ptrs.empty()) {
    GTEST_SKIP() << "Allocator could not satisfy small test allocations";
  }

  // Free one allocation to create some fragmentation.
  setCurrentStream(s);
  A.raw_delete(ptrs.back());
  ptrs.pop_back();

  testonly::drain_allocator_for_snapshots(dev_);

  testonly::CombinedSnapshot cs = testonly::take_snapshot(dev_);

  std::uint64_t seg_reserved = testonly::sum_segment_reserved(cs);
  std::uint64_t seg_active   = testonly::sum_segment_active(cs);

  EXPECT_LE(seg_reserved, cs.stats.reserved_bytes_all_current);
  EXPECT_LE(seg_active,   cs.stats.allocated_bytes_all_current);

  // Cleanup remaining allocations.
  setCurrentStream(s);
  for (void* p : ptrs) {
    A.raw_delete(p);
  }
#endif  // VBT_INTERNAL_TESTS
#endif  // VBT_WITH_CUDA
}

TEST_F(CudaAllocatorStatsSnapshotTest, PerPoolInequalities) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
#if !defined(VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "VBT_INTERNAL_TESTS required";
#else
  if (!testonly::has_native_backend(dev_)) {
    GTEST_SKIP() << "Requires native backend";
  }

  Allocator& A = Allocator::get(dev_);

  // Create a pool and prewarm it so that pool-tagged segments exist.
  MempoolId id = Allocator::create_pool_id(dev_);
  Allocator::retain_pool(dev_, id);

  Stream s = getStreamFromPool(false, dev_);
  const std::size_t kBytes = 1u << 20;  // 1 MiB

  try {
    Allocator::prewarm_graph_pool_for_stream(dev_, id, s, kBytes, /*min_blocks=*/1);
  } catch (...) {
    // Best-effort: if prewarm fails, skip the inequality check.
    Allocator::release_pool(dev_, id);
    GTEST_SKIP() << "prewarm_graph_pool_for_stream failed";
  }

  testonly::quiesce_graphs_for_snapshots(dev_);

  testonly::CombinedSnapshot cs = testonly::take_snapshot(dev_);

  bool found = false;
  for (const auto& gp : cs.pools) {
    if (gp.id.dev != dev_ || gp.id.id != id.id) {
      continue;
    }
    found = true;
    std::uint64_t A_snap = testonly::pool_active_bytes(cs, gp.id.id);
    std::uint64_t B_snap = testonly::pool_reserved_bytes(cs, gp.id.id);

    // Per-pool inequalities: active_snap <= bytes_reserved_pool <= reserved_snap.
    EXPECT_LE(A_snap, gp.bytes_reserved);
    EXPECT_LE(gp.bytes_reserved, B_snap);
  }

  if (!found) {
    Allocator::release_pool(dev_, id);
    GTEST_SKIP() << "No GraphPoolSnapshot for prewarmed pool";
  }

  Allocator::release_pool(dev_, id);
#endif  // VBT_INTERNAL_TESTS
#endif  // VBT_WITH_CUDA
}

TEST_F(CudaAllocatorStatsSnapshotTest, AsyncDeviceStatsRemainZeroAndNoSegments) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
#if !defined(VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "VBT_INTERNAL_TESTS required";
#else
  if (!testonly::has_async_backend(dev_)) {
    GTEST_SKIP() << "Requires async backend";
  }

  Allocator& A = Allocator::get(dev_);

  // Best-effort: exercise async backend with a tiny allocation.
  {
    Stream s = getStreamFromPool(false, dev_);
    void* p = nullptr;
    try {
      p = A.raw_alloc(1u << 20, s);
    } catch (...) {
      p = nullptr;
    }
    if (p) {
      setCurrentStream(s);
      A.raw_delete(p);
    }
  }

  testonly::drain_allocator_for_snapshots(dev_);

  testonly::CombinedSnapshot cs = testonly::take_snapshot(dev_);

  // Async backend should not report native snapshots or fraction/GC counters.
  EXPECT_TRUE(cs.segments.empty());
  EXPECT_EQ(cs.stats.fraction_cap_breaches, 0u);
  EXPECT_EQ(cs.stats.fraction_cap_misfires, 0u);
  EXPECT_EQ(cs.stats.gc_passes, 0u);
  EXPECT_EQ(cs.stats.gc_reclaimed_bytes, 0u);
#endif  // VBT_INTERNAL_TESTS
#endif  // VBT_WITH_CUDA
}
