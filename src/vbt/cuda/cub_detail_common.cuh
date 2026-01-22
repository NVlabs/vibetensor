// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#include "vbt/cuda/allocator.h"
#include "vbt/cuda/graphs.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/stream.h"

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if !VBT_WITH_CUDA
#  error "cub_detail_common.cuh is CUDA-only"
#endif

#include <cuda_runtime_api.h>

namespace vbt { namespace cuda { namespace cub_detail {

inline void cudaCheck(cudaError_t st, const char* what) {
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    throw std::runtime_error(std::string(what) + ": " + (msg ? msg : ""));
  }
}

// CUDA-only helper defined in cub_wrappers.cu.
// Returns a non-null device pointer valid for the current device.
void* cub_dummy_temp_storage_ptr();

inline vbt::cuda::DeviceIndex checked_stream_device_index(vbt::cuda::Stream s) {
  if (s.device_index() < 0) {
    std::ostringstream oss;
    oss << "cub: stream.device_index must be >= 0 (got " << static_cast<int>(s.device_index()) << ")";
    throw std::invalid_argument(oss.str());
  }
  return s.device_index();
}

struct CubContext final {
  vbt::cuda::Allocator&   alloc;
  vbt::cuda::Stream       stream;
  vbt::cuda::DeviceGuard  dg;
  vbt::cuda::CUDAStreamGuard sg;

  CubContext(vbt::cuda::Allocator& a, vbt::cuda::Stream s)
      : alloc(a),
        stream(s),
        dg(checked_stream_device_index(s)),
        sg(s) {}

  CubContext(const CubContext&) = delete;
  CubContext& operator=(const CubContext&) = delete;
};

class StreamDeviceBuffer final {
 public:
  StreamDeviceBuffer(vbt::cuda::Allocator& alloc, vbt::cuda::Stream stream, std::size_t bytes)
      : alloc_(&alloc), dev_(stream.device_index()), stream_(stream), bytes_(bytes) {
    ptr_ = alloc_->raw_alloc(bytes_, stream_);
    if (!ptr_) {
      throw std::runtime_error("cub: raw_alloc returned null");
    }
  }

  StreamDeviceBuffer(const StreamDeviceBuffer&) = delete;
  StreamDeviceBuffer& operator=(const StreamDeviceBuffer&) = delete;

  StreamDeviceBuffer(StreamDeviceBuffer&& other) noexcept
      : alloc_(other.alloc_), dev_(other.dev_), stream_(other.stream_), ptr_(other.ptr_), bytes_(other.bytes_) {
    other.alloc_ = nullptr;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
    other.dev_ = -1;
  }

  StreamDeviceBuffer& operator=(StreamDeviceBuffer&& other) noexcept {
    if (this == &other) return *this;
    reset();
    alloc_ = other.alloc_;
    dev_ = other.dev_;
    stream_ = other.stream_;
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    other.alloc_ = nullptr;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
    other.dev_ = -1;
    return *this;
  }

  ~StreamDeviceBuffer() noexcept { reset(); }

  [[nodiscard]] void* data() const noexcept { return ptr_; }
  [[nodiscard]] std::size_t size_bytes() const noexcept { return bytes_; }

 private:
  void reset() noexcept {
    if (!ptr_ || !alloc_) return;
    // Ensure Allocator::raw_delete observes the correct TLS current stream.
    vbt::cuda::DeviceGuard dg(dev_);
    vbt::cuda::CUDAStreamGuard sg(stream_);
    alloc_->raw_delete(ptr_);
    ptr_ = nullptr;
    bytes_ = 0;
    alloc_ = nullptr;
    dev_ = -1;
  }

  vbt::cuda::Allocator* alloc_{nullptr};
  vbt::cuda::DeviceIndex dev_{-1};
  vbt::cuda::Stream stream_{vbt::cuda::Stream::UNCHECKED, 0u, 0};
  void* ptr_{nullptr};
  std::size_t bytes_{0};
};

inline void check_no_overlap_ranges(
    const void* a,
    std::size_t a_bytes,
    const void* b,
    std::size_t b_bytes,
    const char* overlap_msg) {
  if (!a || !b || a_bytes == 0 || b_bytes == 0) {
    return;
  }

  const std::uintptr_t a0 = reinterpret_cast<std::uintptr_t>(a);
  const std::uintptr_t b0 = reinterpret_cast<std::uintptr_t>(b);
  const std::uintptr_t a1 = a0 + a_bytes;
  const std::uintptr_t b1 = b0 + b_bytes;

  // Detect uintptr_t overflow in range computations.
  if (a1 < a0 || b1 < b0) {
    throw std::invalid_argument("cub: pointer range overflow");
  }

  if (a0 < b1 && b0 < a1) {
    throw std::invalid_argument(overlap_msg);
  }
}

inline void throw_capture_denied(vbt::cuda::DeviceIndex dev) {
  throw std::runtime_error(std::string(vbt::cuda::kErrAllocatorCaptureDenied) +
                           " on device " + std::to_string(static_cast<int>(dev)));
}

// Two-pass temp storage helper for CUB calls.
//
// CubCall signature:
//   cudaError_t call(void* temp_storage, std::size_t& temp_storage_bytes);
//
// The helper:
// - enforces device+stream guards (DC3)
// - rejects graph capture (allocations forbidden)
// - queries temp bytes then allocates temp storage
// - calls run pass with a non-null temp pointer even when bytes==0
// - checks CUDA errors on both passes and after launch
template <class CubCall>
inline void call_with_temp_storage_checked(
    vbt::cuda::Allocator& alloc,
    vbt::cuda::Stream stream,
    const char* what,
    CubCall&& call) {
  CubContext ctx(alloc, stream);

  if (vbt::cuda::streamCaptureStatus(stream) != vbt::cuda::CaptureStatus::None) {
    throw_capture_denied(stream.device_index());
  }

  // Pre-clear any sticky error so we don't attribute it to this wrapper.
  (void)cudaGetLastError();

  std::size_t temp_storage_bytes = 0;
  cudaError_t st = call(nullptr, temp_storage_bytes);
  cudaCheck(st, what);

  if (temp_storage_bytes > 0) {
    StreamDeviceBuffer tmp(ctx.alloc, ctx.stream, temp_storage_bytes);
    st = call(tmp.data(), temp_storage_bytes);
    cudaCheck(st, what);
  } else {
    void* dummy = cub_dummy_temp_storage_ptr();
    if (!dummy) {
      throw std::runtime_error("cub: dummy temp storage pointer is null");
    }
    st = call(dummy, temp_storage_bytes);
    cudaCheck(st, what);
  }

  cudaError_t lc = cudaGetLastError();
  cudaCheck(lc, what);
}

}}} // namespace vbt::cuda::cub_detail
