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

namespace {
static inline bool has_substr(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}
} // namespace

TEST(CUDAGraphStatusModeTest, DefaultStreamAndNestedCaptureRejection) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  GraphCounters before = cuda_graphs_counters();

  // Default-stream capture is banned before any CUDA calls.
  {
    CUDAGraph g;
    Stream ds = getDefaultStream(-1);
    try {
      g.capture_begin(ds);
      FAIL() << "Expected default-stream capture ban";
    } catch (const std::runtime_error& e) {
      std::string msg = e.what();
      EXPECT_TRUE(has_substr(msg, std::string(kErrDefaultStreamCaptureBan)));
    }
  }

  GraphCounters mid = cuda_graphs_counters();
  EXPECT_EQ(mid.denied_default_stream, before.denied_default_stream + 1);

  // Nested capture ban: raw CUDA capture followed by CUDAGraph capture_begin.
  DeviceIndex dev = 0;
  Stream s = getStreamFromPool(false, dev);
  CUDAGraph g2;

  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());
  ASSERT_EQ(cudaStreamBeginCapture(raw, cudaStreamCaptureModeThreadLocal), cudaSuccess);
  try {
    g2.capture_begin(s);
    FAIL() << "Expected nested capture ban";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_TRUE(has_substr(msg, std::string(kErrNestedCaptureBan)));
  }
  cudaGraph_t cg = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(raw, &cg), cudaSuccess);
  if (cg) {
    (void)cudaGraphDestroy(cg);
  }

  GraphCounters after = cuda_graphs_counters();
  EXPECT_EQ(after.nested_capture_denied, mid.nested_capture_denied + 1);
#endif
}

TEST(CUDAGraphStatusModeTest, UnsupportedCaptureModeRejected) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  Stream s = getStreamFromPool(false, dev);
  CUDAGraph g;

  try {
    g.capture_begin(s, std::nullopt, CaptureMode::Global);
    FAIL() << "Expected unsupported mode rejection";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_TRUE(has_substr(msg, std::string(kErrUnsupportedCaptureMode)));
  }
#endif
}

TEST(CUDAGraphStatusModeTest, PoolDeviceMismatchBumpsInvalidStateCounter) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  DeviceIndex dev = 0;
  Stream s = getStreamFromPool(false, dev);
  CUDAGraph g;

  GraphCounters before = cuda_graphs_counters();

  // Use a synthetic pool id whose device does not match the capture stream.
  MempoolId bad{static_cast<DeviceIndex>(dev + 1), 1u};

  try {
    g.capture_begin(s, bad, CaptureMode::ThreadLocal);
    FAIL() << "Expected device-mismatch rejection";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_TRUE(has_substr(msg, std::string(kErrGraphDeviceMismatchPrefix)));
  }

  GraphCounters after = cuda_graphs_counters();
  EXPECT_EQ(after.capture_begin_invalid_state,
            before.capture_begin_invalid_state + 1);
  EXPECT_EQ(after.captures_started, before.captures_started);
#endif
}
