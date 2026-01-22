// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "vbt/autograd/engine.h"
#include "vbt/autograd/engine_toggles.h"
#include "vbt/autograd/node.h"
#include "vbt/autograd/types.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"

#if VBT_WITH_CUDA
#include "vbt/cuda/device.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#include <cuda_runtime_api.h>
#endif

using vbt::autograd::Node;
using vbt::autograd::OptionalTensor;
using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::TensorImpl;
using vbt::core::intrusive_ptr;

namespace {

static TensorImpl make_cpu_dense_f32(const std::vector<int64_t>& sizes, float fill) {
  std::size_t ne = 1;
  for (auto s : sizes) {
    ne *= static_cast<std::size_t>(s == 0 ? 1 : s);
  }
  const std::size_t nbytes = ne * sizeof(float);

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  StoragePtr st = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    acc *= (sizes[idx] == 0 ? 1 : sizes[idx]);
  }

  TensorImpl t(st, sizes, strides, /*storage_offset=*/0,
               ScalarType::Float32, Device::cpu());
  float* p = static_cast<float*>(t.data());
  for (std::size_t i = 0; i < ne; ++i) {
    p[i] = fill;
  }
  return t;
}

#if VBT_WITH_CUDA
static TensorImpl make_cuda_dense_f32(const std::vector<int64_t>& sizes,
                                      float fill,
                                      int dev = 0) {
  const std::size_t ne = [&]() {
    std::size_t n = 1;
    for (auto s : sizes) {
      n *= static_cast<std::size_t>(s == 0 ? 1 : s);
    }
    return n;
  }();
  const std::size_t nbytes = ne * sizeof(float);

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::DeviceGuard;
  const auto dev_idx = static_cast<DeviceIndex>(dev);
  DeviceGuard dg(dev_idx);

  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  if (nbytes > 0) {
    std::vector<float> host(ne, fill);
    cudaError_t st = cudaMemcpy(storage->data(), host.data(), nbytes,
                                cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy H2D failed in make_cuda_dense_f32");
    }
  }

  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    acc *= (sizes[idx] == 0 ? 1 : sizes[idx]);
  }

  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0,
                    ScalarType::Float32, Device::cuda(dev));
}
#endif

struct CpuSinkRoot final : Node {
  uint32_t num_inputs() const noexcept override { return 1; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    (void)grads_in;
    return {}; // sink
  }
};

#if VBT_WITH_CUDA
struct ToggleRestore {
  bool prev;
  ToggleRestore() : prev(vbt::autograd::is_streaming_backwards_enabled()) {}
  ~ToggleRestore() { vbt::autograd::set_streaming_backwards_enabled(prev); }
};

struct CaptureCleanup {
  cudaStream_t raw{};
  bool         done{false};

  explicit CaptureCleanup(cudaStream_t s) noexcept : raw(s) {}

  CaptureCleanup(const CaptureCleanup&) = delete;
  CaptureCleanup& operator=(const CaptureCleanup&) = delete;

  ~CaptureCleanup() noexcept {
    if (done) {
      return;
    }
    cudaGraph_t cg = nullptr;
    cudaError_t st = cudaStreamEndCapture(raw, &cg);
    if (st != cudaSuccess) {
      // Clear expected sticky error to avoid poisoning later CUDA calls.
      (void)cudaGetLastError();
    }
    if (cg) {
      (void)cudaGraphDestroy(cg);
    }
  }

  void mark_done() noexcept { done = true; }
};
#endif

} // namespace

TEST(AutogradCaptureGuardTest, PCAP1BackwardUnderCaptureForbiddenCuda) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  ToggleRestore restore;
  vbt::autograd::set_streaming_backwards_enabled(true);

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::DeviceGuard;
  using vbt::cuda::Stream;
  using vbt::cuda::CUDAStreamGuard;

  const auto dev_idx = static_cast<DeviceIndex>(0);
  DeviceGuard dg(dev_idx);

  auto root = vbt::core::make_intrusive<CpuSinkRoot>();
  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);

  // Begin raw capture on a non-default stream and keep it current.
  Stream cap = vbt::cuda::getStreamFromPool(/*priority=*/0, dev_idx);
  CUDAStreamGuard sg(cap);
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(cap.handle());
  ASSERT_EQ(cudaStreamBeginCapture(raw, cudaStreamCaptureModeThreadLocal), cudaSuccess);
  CaptureCleanup cleanup(raw);

  bool threw = false;
  try {
    vbt::autograd::run_backward(intrusive_ptr<Node>(root.get()), seed, {});
  } catch (const std::runtime_error& e) {
    threw = true;
    const std::string msg(e.what());
    EXPECT_NE(msg.find("CUDA Graph capture"), std::string::npos) << msg;
  } catch (...) {
    threw = true;
    ADD_FAILURE() << "expected std::runtime_error";
  }

  if (!threw) {
    ADD_FAILURE() << "expected runtime_error";
  }

  cudaGraph_t cg = nullptr;
  cudaError_t end_st = cudaStreamEndCapture(raw, &cg);
  EXPECT_EQ(end_st, cudaSuccess);
  if (cg) {
    (void)cudaGraphDestroy(cg);
  }
  if (end_st == cudaSuccess) {
    cleanup.mark_done();
  }
#endif
}

TEST(AutogradCaptureGuardTest, PCAP2CpuOnlyBackwardAllowedUnderCapture) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  // Leave CUDA autograd toggle alone; this test is CPU-only.
  using vbt::cuda::DeviceIndex;
  using vbt::cuda::DeviceGuard;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  DeviceGuard dg(dev_idx);

  Stream s = vbt::cuda::getStreamFromPool(/*priority=*/0, dev_idx);
  cudaStream_t raw = reinterpret_cast<cudaStream_t>(s.handle());

  ASSERT_EQ(cudaStreamBeginCapture(raw, cudaStreamCaptureModeThreadLocal), cudaSuccess);
  CaptureCleanup cleanup(raw);

  auto root = vbt::core::make_intrusive<CpuSinkRoot>();
  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);

  EXPECT_NO_THROW(vbt::autograd::run_backward(intrusive_ptr<Node>(root.get()), seed, {}));

  cudaGraph_t cg = nullptr;
  cudaError_t end_st = cudaStreamEndCapture(raw, &cg);
  EXPECT_EQ(end_st, cudaSuccess);
  if (cg) {
    (void)cudaGraphDestroy(cg);
  }
  if (end_st == cudaSuccess) {
    cleanup.mark_done();
  }
#endif
}
