// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/graphs.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/device.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace vbt::cuda;

namespace {
static inline bool has_substr(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}
} // namespace

TEST(CudaGraphsAllocatorRawCaptureDenialTest, DeniesBothOverloadsWithoutRouting) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);
  Stream s = getStreamFromPool(false, dev);
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());

  const std::size_t kSize = 4096;

  // Case 1: no-stream overload uses the current stream; capture without
  // allocator routing must be denied.
  setCurrentStream(s);
  ASSERT_EQ(cudaStreamBeginCapture(raw, cudaStreamCaptureModeThreadLocal), cudaSuccess);
  try {
    (void)A.raw_alloc(kSize);
    FAIL() << "Expected allocator capture denial for no-stream overload";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_TRUE(has_substr(msg, std::string(kErrAllocatorCaptureDenied)));
  }
  cudaGraph_t g1 = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(raw, &g1), cudaSuccess);
  if (g1) {
    (void)cudaGraphDestroy(g1);
  }

  // Case 2: explicit stream overload under capture should also be denied when
  // no routing guard is active.
  ASSERT_EQ(cudaStreamBeginCapture(raw, cudaStreamCaptureModeThreadLocal), cudaSuccess);
  try {
    (void)A.raw_alloc(kSize, s);
    FAIL() << "Expected allocator capture denial for stream overload";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_TRUE(has_substr(msg, std::string(kErrAllocatorCaptureDenied)));
  }
  cudaGraph_t g2 = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(raw, &g2), cudaSuccess);
  if (g2) {
    (void)cudaGraphDestroy(g2);
  }
#endif
}

TEST(CudaGraphsAllocatorRawCaptureDenialTest,
     DenialDoesNotGrowOrTouchDiagnosticCounters) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
#ifndef VBT_INTERNAL_TESTS
  GTEST_SKIP() << "VBT_INTERNAL_TESTS required for debug counters";
#else
  if (device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceIndex dev = 0;
  Allocator& A = Allocator::get(dev);

  if (A.debug_backend_kind_for_testing() != BackendKind::Native) {
    GTEST_SKIP() << "Requires native backend";
  }

  // Quiesce allocator and reset stats/counters.
  A.process_events(-1);
  A.emptyCache();
  A.resetAccumulatedStats();
  debug_reset_cudaMalloc_calls_for_testing();

  DeviceStats before = A.getDeviceStats();

  Stream s = getStreamFromPool(false, dev);
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());
  const std::size_t kSize = 4096;

  setCurrentStream(s);
  ASSERT_EQ(cudaStreamBeginCapture(raw, cudaStreamCaptureModeThreadLocal), cudaSuccess);

  // No-stream overload under capture.
  try {
    (void)A.raw_alloc(kSize);
    FAIL() << "Expected allocator capture denial for no-stream overload";
  } catch (const std::runtime_error& e) {
    EXPECT_TRUE(has_substr(e.what(), std::string(kErrAllocatorCaptureDenied)));
  }

  // Stream overload under capture.
  try {
    (void)A.raw_alloc(kSize, s);
    FAIL() << "Expected allocator capture denial for stream overload";
  } catch (const std::runtime_error& e) {
    EXPECT_TRUE(has_substr(e.what(), std::string(kErrAllocatorCaptureDenied)));
  }

  cudaGraph_t g = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(raw, &g), cudaSuccess);
  if (g) {
    (void)cudaGraphDestroy(g);
  }

  DeviceStats after = A.getDeviceStats();

  // diagnostics on the native backend.
  EXPECT_EQ(after.reserved_bytes_all_current,
            before.reserved_bytes_all_current);
  EXPECT_EQ(after.num_device_alloc, before.num_device_alloc);
  EXPECT_EQ(after.fraction_cap_breaches, before.fraction_cap_breaches);
  EXPECT_EQ(after.fraction_cap_misfires, before.fraction_cap_misfires);
  EXPECT_EQ(after.gc_passes, before.gc_passes);
  EXPECT_EQ(after.gc_reclaimed_bytes, before.gc_reclaimed_bytes);

  // No growth via cudaMalloc should have occurred while capture was active.
  EXPECT_EQ(debug_get_cudaMalloc_call_count_for_testing(), 0u);
#endif  // VBT_INTERNAL_TESTS
#endif  // VBT_WITH_CUDA
}
