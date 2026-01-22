// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/device_caps.h"

#include <mutex>
#include <vector>

#if VBT_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace vbt {
namespace cuda {

namespace {
struct CapsEntry {
  int       device_index{0};
  DeviceCaps caps{};
};

std::mutex              g_caps_mutex;
std::vector<CapsEntry>  g_caps_cache;
} // namespace

DeviceCaps get_device_caps(DeviceIndex device) {
#if !VBT_WITH_CUDA
  (void)device;
  return DeviceCaps{};
#else
  const int idx = static_cast<int>(device);

  std::lock_guard<std::mutex> lock(g_caps_mutex);
  for (const auto& e : g_caps_cache) {
    if (e.device_index == idx) {
      return e.caps;
    }
  }

  DeviceCaps caps{};
  caps.device_index = idx;

  cudaDeviceProp prop{};
  cudaError_t st = cudaGetDeviceProperties(&prop, idx);
  if (st == cudaSuccess) {
    if (prop.maxThreadsPerBlock > 0) {
      caps.max_threads_per_block = static_cast<unsigned int>(prop.maxThreadsPerBlock);
    }
    if (prop.maxGridSize[0] > 0) {
      caps.max_grid_x = static_cast<unsigned int>(prop.maxGridSize[0]);
    }
    if (prop.maxGridSize[1] > 0) {
      caps.max_grid_y = static_cast<unsigned int>(prop.maxGridSize[1]);
    }
  }

  g_caps_cache.push_back(CapsEntry{idx, caps});
  return caps;
#endif
}

} // namespace cuda
} // namespace vbt
