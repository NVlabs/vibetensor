// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/dispatch/dispatcher.h"

#include "vbt/autograd/wrapper.h"
#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#include "vbt/cuda/allocator.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/fabric_state.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#endif

extern "C" void vbt_register_default_kernels();
#if VBT_WITH_CUDA
extern "C" void vbt_register_cuda_elementwise_kernels();
extern "C" void vbt_register_fabric_kernels();
#endif

using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::TensorImpl;
using vbt::dispatch::BoxedStack;
using vbt::dispatch::Dispatcher;

static TensorImpl make_cpu_scalar_i64(std::int64_t v) {
  void* buf = ::operator new(sizeof(std::int64_t));
  *static_cast<std::int64_t*>(buf) = v;
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto st = vbt::core::make_intrusive<Storage>(std::move(dp), sizeof(std::int64_t));
  return TensorImpl(st, /*sizes=*/{}, /*strides=*/{}, /*storage_offset=*/0, ScalarType::Int64, Device::cpu());
}

#if VBT_WITH_CUDA

static TensorImpl make_cuda_tensor_f32(int dev, const std::vector<float>& vals) {
  vbt::cuda::DeviceGuard dg(static_cast<vbt::cuda::DeviceIndex>(dev));

  const std::size_t nbytes = vals.size() * sizeof(float);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);

  TensorImpl t(storage,
              /*sizes=*/{static_cast<std::int64_t>(vals.size())},
              /*strides=*/{1},
              /*storage_offset=*/0,
              ScalarType::Float32,
              Device::cuda(dev));

  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(dev));
  vbt::cuda::Allocator& alloc = vbt::cuda::Allocator::get(static_cast<vbt::cuda::DeviceIndex>(dev));

  if (nbytes > 0) {
    cudaError_t st = alloc.memcpyAsync(t.data(), dev, vals.data(), -1, nbytes, stream, /*p2p_enabled=*/false);
    if (st != cudaSuccess) {
      const char* msg = cudaGetErrorString(st);
      throw std::runtime_error(std::string("memcpyAsync H2D failed: ") + (msg ? msg : ""));
    }
    cudaError_t st_sync = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.handle()));
    if (st_sync != cudaSuccess) {
      const char* msg = cudaGetErrorString(st_sync);
      throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + (msg ? msg : ""));
    }
  }

  return t;
}

static std::vector<float> cuda_tensor_to_host_f32(const TensorImpl& t) {
  if (t.device().type != kDLCUDA) throw std::runtime_error("expected CUDA tensor");
  if (t.dtype() != ScalarType::Float32) throw std::runtime_error("expected float32 tensor");

  const std::size_t n = static_cast<std::size_t>(t.numel());
  std::vector<float> host(n);
  const std::size_t nbytes = n * sizeof(float);

  int dev = static_cast<int>(t.device().index);
  vbt::cuda::DeviceGuard dg(static_cast<vbt::cuda::DeviceIndex>(dev));
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(dev));
  vbt::cuda::Allocator& alloc = vbt::cuda::Allocator::get(static_cast<vbt::cuda::DeviceIndex>(dev));

  if (nbytes > 0) {
    cudaError_t st = alloc.memcpyAsync(host.data(), -1, t.data(), dev, nbytes, stream, /*p2p_enabled=*/false);
    if (st != cudaSuccess) {
      const char* msg = cudaGetErrorString(st);
      throw std::runtime_error(std::string("memcpyAsync D2H failed: ") + (msg ? msg : ""));
    }
    cudaError_t st_sync = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.handle()));
    if (st_sync != cudaSuccess) {
      const char* msg = cudaGetErrorString(st_sync);
      throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + (msg ? msg : ""));
    }
  }

  return host;
}

#endif

TEST(FabricCorrectnessTest, CopyFallbackAddAndMulWorkWhenFabricDisabled) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA not built";
#else
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 CUDA devices";
  }

  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();
  vbt_register_fabric_kernels();

  // Force Fabric disabled so the op must take copy fallback.
  vbt::autograd::NoGradGuard ng;
  auto& fs = vbt::cuda::fabric::fabric_state();
  fs.config.mode.store(vbt::cuda::fabric::FabricMode::Disabled, std::memory_order_release);

  const std::vector<float> a_vals{1.f, 2.f, 3.f, 4.f, -5.f, 6.f, 7.f, 8.f};
  const std::vector<float> b_vals{10.f, -2.f, 0.5f, 1.f, 5.f, -6.f, 0.f, 2.f};

  TensorImpl a0 = make_cuda_tensor_f32(/*dev=*/0, a_vals);
  TensorImpl b1 = make_cuda_tensor_f32(/*dev=*/1, b_vals);

  TensorImpl compute0 = make_cpu_scalar_i64(0);
  TensorImpl require0 = make_cpu_scalar_i64(0);
  TensorImpl fallback1 = make_cpu_scalar_i64(1);

  // Add
  {
    BoxedStack s{a0, b1, compute0, require0, fallback1};
    Dispatcher::instance().callBoxed("vt::fabric_add", s);
    ASSERT_EQ(s.size(), 1u);
    const auto out = s[0];
    EXPECT_EQ(out.device(), Device::cuda(0));

    const auto host = cuda_tensor_to_host_f32(out);
    ASSERT_EQ(host.size(), a_vals.size());
    for (std::size_t i = 0; i < host.size(); ++i) {
      EXPECT_FLOAT_EQ(host[i], a_vals[i] + b_vals[i]);
    }
  }

  // Mul
  {
    BoxedStack s{a0, b1, compute0, require0, fallback1};
    Dispatcher::instance().callBoxed("vt::fabric_mul", s);
    ASSERT_EQ(s.size(), 1u);
    const auto out = s[0];
    EXPECT_EQ(out.device(), Device::cuda(0));

    const auto host = cuda_tensor_to_host_f32(out);
    ASSERT_EQ(host.size(), a_vals.size());
    for (std::size_t i = 0; i < host.size(); ++i) {
      EXPECT_FLOAT_EQ(host[i], a_vals[i] * b_vals[i]);
    }
  }

  // Add (non-contiguous remote input): exercises the temporary contiguous
  // clone on the remote device + cross-device copy lifetime.
  {
    std::vector<float> b_storage_vals(2 * b_vals.size(), 0.f);
    for (std::size_t i = 0; i < b_vals.size(); ++i) {
      b_storage_vals[2 * i] = b_vals[i];
    }
    TensorImpl b1_storage = make_cuda_tensor_f32(/*dev=*/1, b_storage_vals);
    auto st = b1_storage.storage();
    TensorImpl b1_strided(st,
                          /*sizes=*/{static_cast<std::int64_t>(b_vals.size())},
                          /*strides=*/{2},
                          /*storage_offset=*/0,
                          ScalarType::Float32,
                          Device::cuda(1));

    BoxedStack s{a0, b1_strided, compute0, require0, fallback1};
    Dispatcher::instance().callBoxed("vt::fabric_add", s);
    ASSERT_EQ(s.size(), 1u);
    const auto out = s[0];
    EXPECT_EQ(out.device(), Device::cuda(0));

    const auto host = cuda_tensor_to_host_f32(out);
    ASSERT_EQ(host.size(), a_vals.size());
    for (std::size_t i = 0; i < host.size(); ++i) {
      EXPECT_FLOAT_EQ(host[i], a_vals[i] + b_vals[i]);
    }
  }
#endif
}

TEST(FabricCorrectnessTest,
     FabricAddCopyFallbackWorksWhenCurrentDeviceMismatchUnderV2NoCudaCalls) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA not built";
#elif !VBT_WITH_DISPATCH_V2 || !VBT_INTERNAL_TESTS
  GTEST_SKIP() << "dispatch v2 or internal tests disabled";
#else
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 CUDA devices";
  }

  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();
  vbt_register_fabric_kernels();

  vbt::dispatch::DispatchV2ModeGuard v2(true);
  vbt::dispatch::DispatchV2FabricNoCudaCallsGuard no_cuda(true);
  ASSERT_TRUE(vbt::dispatch::dispatch_v2_fabric_no_cuda_calls());

  // Force Fabric disabled so the op must take copy fallback.
  vbt::autograd::NoGradGuard ng;
  auto& fs = vbt::cuda::fabric::fabric_state();
  const auto prev_mode = fs.config.mode.load(std::memory_order_acquire);
  struct FabricModeRestore {
    std::atomic<vbt::cuda::fabric::FabricMode>* mode;
    vbt::cuda::fabric::FabricMode prev;
    ~FabricModeRestore() { mode->store(prev, std::memory_order_release); }
  } mode_restore{&fs.config.mode, prev_mode};
  fs.config.mode.store(vbt::cuda::fabric::FabricMode::Disabled,
                       std::memory_order_release);

  const std::vector<float> a_vals{1.f, 2.f, 3.f, 4.f};
  const std::vector<float> b_vals{10.f, -2.f, 0.5f, 1.f};

  TensorImpl a0 = make_cuda_tensor_f32(/*dev=*/0, a_vals);
  TensorImpl b1 = make_cuda_tensor_f32(/*dev=*/1, b_vals);

  TensorImpl compute0 = make_cpu_scalar_i64(0);
  TensorImpl require0 = make_cpu_scalar_i64(0);
  TensorImpl fallback1 = make_cpu_scalar_i64(1);

  int prev_dev = -1;
  ASSERT_EQ(cudaGetDevice(&prev_dev), cudaSuccess);

  {
    constexpr int kCallerDevice = 1;

    // Set current_device != compute_device (compute=0, current=kCallerDevice) for the call.
    vbt::cuda::DeviceGuard dg(static_cast<vbt::cuda::DeviceIndex>(kCallerDevice));

    BoxedStack s{a0, b1, compute0, require0, fallback1};
    Dispatcher::instance().callBoxed("vt::fabric_add", s);

    // The op may temporarily pin to compute_device internally, but must restore the
    // caller's current device before returning.
    int inside_dev = -1;
    ASSERT_EQ(cudaGetDevice(&inside_dev), cudaSuccess);
    EXPECT_EQ(inside_dev, kCallerDevice);

    ASSERT_EQ(s.size(), 1u);
    const auto out = s[0];
    EXPECT_EQ(out.device(), Device::cuda(0));

    const auto host = cuda_tensor_to_host_f32(out);
    ASSERT_EQ(host.size(), a_vals.size());
    for (std::size_t i = 0; i < host.size(); ++i) {
      EXPECT_FLOAT_EQ(host[i], a_vals[i] + b_vals[i]);
    }
  }

  int after_dev = -1;
  ASSERT_EQ(cudaGetDevice(&after_dev), cudaSuccess);
  EXPECT_EQ(after_dev, prev_dev);
  if (after_dev != prev_dev && prev_dev >= 0) {
    (void)cudaSetDevice(prev_dev);
  }
#endif
}

TEST(FabricCorrectnessTest, FabricKernelWorksWhenPeerAccessAvailable) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA not built";
#else
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 CUDA devices";
  }

  int can01 = 0;
  int can10 = 0;
  cudaError_t st01 = cudaDeviceCanAccessPeer(&can01, 0, 1);
  cudaError_t st10 = cudaDeviceCanAccessPeer(&can10, 1, 0);
  if (st01 != cudaSuccess || st10 != cudaSuccess || !(can01 && can10)) {
    (void)cudaGetLastError();
    GTEST_SKIP() << "Peer access between devices 0 and 1 not available";
  }

  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();
  vbt_register_fabric_kernels();

  vbt::autograd::NoGradGuard ng;

  auto& fs = vbt::cuda::fabric::fabric_state();
  if (fs.init_status != vbt::cuda::fabric::FabricInitStatus::Ok || !fs.uva_ok) {
    GTEST_SKIP() << "UVA gate is disabled";
  }
  fs.config.mode.store(vbt::cuda::fabric::FabricMode::BestEffort, std::memory_order_release);

  const std::vector<float> a_vals{1.f, 2.f, 3.f, 4.f};
  const std::vector<float> b_vals{10.f, -2.f, 0.5f, 1.f};

  TensorImpl a0 = make_cuda_tensor_f32(/*dev=*/0, a_vals);
  TensorImpl b1 = make_cuda_tensor_f32(/*dev=*/1, b_vals);

  TensorImpl compute0 = make_cpu_scalar_i64(0);
  TensorImpl require1 = make_cpu_scalar_i64(1);
  TensorImpl fallback1 = make_cpu_scalar_i64(1);

  BoxedStack s{a0, b1, compute0, require1, fallback1};
  Dispatcher::instance().callBoxed("vt::fabric_add", s);
  ASSERT_EQ(s.size(), 1u);
  const auto out = s[0];
  EXPECT_EQ(out.device(), Device::cuda(0));

  const auto host = cuda_tensor_to_host_f32(out);
  ASSERT_EQ(host.size(), a_vals.size());
  for (std::size_t i = 0; i < host.size(); ++i) {
    EXPECT_FLOAT_EQ(host[i], a_vals[i] + b_vals[i]);
  }
#endif
}
