// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/guard.h"
#include "vbt/cuda/stream.h"
#include <cassert>

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#  include <cuda_runtime_api.h>
#endif

namespace vbt { namespace cuda {

namespace {

} // anonymous

CUDAStreamGuard::CUDAStreamGuard(Stream s) noexcept {
#if VBT_WITH_CUDA
  int cur = -1; cudaGetDevice(&cur);
  assert(cur == static_cast<int>(s.device_index()));
  original_device_ = static_cast<DeviceIndex>(cur);
  original_stream_ = getCurrentStream(original_device_);
  // Switch
  DeviceGuard g(s.device_index());
  setCurrentStream(s);
  current_stream_ = s;
#else
  (void)s;
#endif
}

CUDAStreamGuard::~CUDAStreamGuard() noexcept {
#if VBT_WITH_CUDA
  // Restore stream then device
  {
    DeviceGuard g(original_device_);
    setCurrentStream(original_stream_);
  }
  if (original_device_ >= 0) {
    cudaSetDevice(static_cast<int>(original_device_));
  }
#endif
}

void CUDAStreamGuard::reset_stream(Stream s) noexcept {
#if VBT_WITH_CUDA
  // Guard discipline: CUDAStreamGuard can only be reset on its original device.
  assert(s.device_index() == original_device_);
  // Restore to original first, then set to new
  {
    DeviceGuard g(original_device_);
    setCurrentStream(original_stream_);
  }
  {
    DeviceGuard g2(s.device_index());
    setCurrentStream(s);
  }
  current_stream_ = s;
#else
  (void)s;
#endif
}

}} // namespace vbt::cuda
