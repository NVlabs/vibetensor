// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/graphs.h"
#include "vbt/cuda/stream.h"
#include "cuda_graphs_allocator_test_helpers.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace vbt::cuda;
namespace testonly = vbt::cuda::testonly;

namespace {
static inline bool has_substr(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}
}  // namespace

// GraphCounters::allocator_capture_denied without touching allocator
// fraction/GC counters or issuing cudaMalloc calls.
TEST(CudaGraphsAllocatorCaptureDenialTest,
     NativeCaptureDenialUpdatesGraphCountersOnly) {
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

  DeviceStats before_stats = A.getDeviceStats();
  GraphCounters before_gc = cuda_graphs_counters();

  Stream s = getStreamFromPool(false, dev);
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());

  const std::size_t kSize = 4096;

  setCurrentStream(s);
  ASSERT_EQ(cudaStreamBeginCapture(raw, cudaStreamCaptureModeThreadLocal),
            cudaSuccess);

  bool threw = false;
  try {
    (void)A.raw_alloc(kSize);
  } catch (const std::runtime_error& e) {
    threw = true;
    EXPECT_TRUE(has_substr(e.what(), std::string(kErrAllocatorCaptureDenied)));
  } catch (...) {
    threw = true;
  }
  EXPECT_TRUE(threw) << "Expected capture denial for no-stream overload";

  cudaGraph_t g = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(raw, &g), cudaSuccess);
  if (g) {
    (void)cudaGraphDestroy(g);
  }

  DeviceStats after_stats = A.getDeviceStats();
  GraphCounters after_gc = cuda_graphs_counters();

  EXPECT_EQ(after_stats.reserved_bytes_all_current,
            before_stats.reserved_bytes_all_current);
  EXPECT_EQ(after_stats.allocated_bytes_all_current,
            before_stats.allocated_bytes_all_current);
  EXPECT_EQ(after_stats.fraction_cap_breaches,
            before_stats.fraction_cap_breaches);
  EXPECT_EQ(after_stats.fraction_cap_misfires,
            before_stats.fraction_cap_misfires);
  EXPECT_EQ(after_stats.gc_passes, before_stats.gc_passes);
  EXPECT_EQ(after_stats.gc_reclaimed_bytes,
            before_stats.gc_reclaimed_bytes);

  // No growth via cudaMalloc should have occurred while capture was active.
  EXPECT_EQ(debug_get_cudaMalloc_call_count_for_testing(), 0u);

  // Graph counters must observe the allocator-capture-denied event.
  EXPECT_EQ(after_gc.allocator_capture_denied,
            before_gc.allocator_capture_denied + 1);
#endif  // VBT_INTERNAL_TESTS
#endif  // VBT_WITH_CUDA
}
