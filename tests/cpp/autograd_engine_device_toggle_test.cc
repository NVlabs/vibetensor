// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vbt/autograd/engine.h"
#include "vbt/autograd/engine_toggles.h"
#include "vbt/autograd/types.h"
#include "vbt/autograd/node.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"

#if VBT_WITH_CUDA
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/guard.h"
#include <cuda_runtime_api.h>
#endif

using vbt::autograd::GraphTask;
using vbt::autograd::Node;
using vbt::autograd::OptionalTensor;
using vbt::autograd::StreamKind;
using vbt::autograd::NodeStreamInfo;
using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::intrusive_ptr;

#if VBT_AUTOGRAD_TESTING
namespace vbt { namespace autograd {
void _test_seed_root_buffer(GraphTask& gt,
                            vbt::core::intrusive_ptr<Node> root,
                            const std::vector<OptionalTensor>& initial_grads);
}} // namespace vbt::autograd
#endif

namespace {

static TensorImpl make_cpu_dense_f32(const std::vector<int64_t>& sizes,
                                     float fill) {
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
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1;
       i >= 0; --i) {
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
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    acc *= (sizes[idx] == 0 ? 1 : sizes[idx]);
  }

  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0,
                    ScalarType::Float32, Device::cuda(dev));
}
#endif

struct CpuRoot final : Node {
  uint32_t num_inputs() const noexcept override { return 1; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    (void)grads_in;
    return {}; // sink
  }
};

struct TwoSlotRoot final : Node {
  uint32_t num_inputs() const noexcept override { return 2; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    (void)grads_in;
    return {}; // never reached in AD3 test
  }
};

#if VBT_WITH_CUDA
struct CudaAllowlistedSink final : Node {
  explicit CudaAllowlistedSink(Device dev, std::uint64_t stream_id) {
    NodeStreamInfo info;
    info.has_canonical_stream = true;
    info.device = dev;
    info.stream_id = stream_id;
    mutable_stream_info() = info;
    name = "CudaAllowlistedSink";
  }

  uint32_t num_inputs() const noexcept override { return 1; }
  StreamKind stream_kind() const noexcept override { return StreamKind::CudaAllowlisted; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    (void)grads_in;
    return {}; // sink
  }
};

struct ToggleFlippingProducer final : Node {
  explicit ToggleFlippingProducer(Device dev, std::uint64_t stream_id,
                                  intrusive_ptr<Node> next) {
    NodeStreamInfo info;
    info.has_canonical_stream = true;
    info.device = dev;
    info.stream_id = stream_id;
    mutable_stream_info() = info;
    name = "ToggleFlippingProducer";

    next_edges.resize(1);
    next_edges[0].fn = std::move(next);
    next_edges[0].input_nr = 0;
  }

  uint32_t num_inputs() const noexcept override { return 1; }
  StreamKind stream_kind() const noexcept override { return StreamKind::CudaAllowlisted; }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    // Flip global toggle mid-run. GraphTask should have snapshotted it already.
    vbt::autograd::set_streaming_backwards_enabled(false);

    std::vector<OptionalTensor> outs(1);
    if (!grads_in.empty()) {
      outs[0] = std::move(grads_in[0]);
    }
    return outs;
  }
};

struct ToggleRestore {
  bool prev;
  ToggleRestore() : prev(vbt::autograd::is_streaming_backwards_enabled()) {}
  ~ToggleRestore() { vbt::autograd::set_streaming_backwards_enabled(prev); }
};
#endif

} // namespace

#if VBT_AUTOGRAD_TESTING
TEST(AutogradDeviceDerivationTest, AD2AllNulloptNonValidatableDefaultsToCpu) {
  GraphTask gt;
  auto root = vbt::core::make_intrusive<CpuRoot>();

  std::vector<OptionalTensor> seed(1);
  seed[0] = OptionalTensor{};

  vbt::autograd::_test_seed_root_buffer(gt, intrusive_ptr<Node>(root.get()), seed);

  EXPECT_TRUE(gt.has_autograd_device);
  EXPECT_EQ(gt.autograd_device, Device::cpu(0));

  auto it = gt.inputs.find(root.get());
  ASSERT_NE(it, gt.inputs.end());
  const GraphTask::InputBuffer& ib = it->second;
  ASSERT_EQ(ib.expected, 1u);
  ASSERT_EQ(ib.present.size(), 1u);
  EXPECT_TRUE(ib.present[0]);
  EXPECT_FALSE(ib.grads_in[0].has_value());
}

#if VBT_WITH_CUDA
TEST(AutogradDeviceDerivationTest, AD3MultipleDevicesThrowSingleDeviceError) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  auto root = vbt::core::make_intrusive<TwoSlotRoot>();

  std::vector<OptionalTensor> seed(2);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);
  seed[1] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);

  try {
    vbt::autograd::_test_seed_root_buffer(gt, intrusive_ptr<Node>(root.get()), seed);
    FAIL() << "expected invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("single-device backward only"), std::string::npos) << msg;
  }
}
#else
TEST(AutogradDeviceDerivationTest, AD3MultipleDevicesThrowSingleDeviceError) {
  GTEST_SKIP() << "CUDA build disabled";
}
#endif
#endif  // VBT_AUTOGRAD_TESTING

#if VBT_WITH_CUDA
TEST(AutogradDeviceDerivationTest, AD1SingleCudaDeviceSetsAutogradDevice) {
#if VBT_AUTOGRAD_TESTING
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  using vbt::cuda::DeviceGuard;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  DeviceGuard dg(dev_idx);
  Stream cur = vbt::cuda::getCurrentStream(dev_idx);

  auto root = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), cur.id());

  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);

  vbt::autograd::_test_seed_root_buffer(gt, intrusive_ptr<Node>(root.get()), seed);

  EXPECT_TRUE(gt.has_autograd_device);
  EXPECT_EQ(gt.autograd_device, Device::cuda(0));
#else
  GTEST_SKIP() << "VBT_AUTOGRAD_TESTING disabled";
#endif
}

TEST(AutogradCudaToggleTest, TOGGLE1DisabledRejectsCudaBackwardAtEngineEntry) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  ToggleRestore restore;
  vbt::autograd::set_streaming_backwards_enabled(false);

  // Root node can be CPU-only; the engine rejects the run before draining.
  auto root = vbt::core::make_intrusive<CpuRoot>();

  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);

  try {
    vbt::autograd::run_backward(intrusive_ptr<Node>(root.get()), seed, {});
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("CUDA autograd is disabled"), std::string::npos) << msg;
  }
}

TEST(AutogradCudaToggleTest, TOGGLE2SnapshotSemanticsNotAffectedByMidRunToggleFlip) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  ToggleRestore restore;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  using vbt::cuda::DeviceGuard;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  DeviceGuard dg(dev_idx);
  Stream cur = vbt::cuda::getCurrentStream(dev_idx);

  // Enable CUDA autograd for the first run.
  vbt::autograd::set_streaming_backwards_enabled(true);

  auto sink = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), cur.id());
  auto root = vbt::core::make_intrusive<ToggleFlippingProducer>(Device::cuda(0), cur.id(),
                                                                intrusive_ptr<Node>(sink.get()));

  std::vector<OptionalTensor> seed1(1);
  seed1[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);

  // The producer flips the global toggle to false during apply(). The backward
  // should still complete because the engine snapshots the toggle at entry.
  ASSERT_NO_THROW(vbt::autograd::run_backward(intrusive_ptr<Node>(root.get()), seed1, {}));
  EXPECT_FALSE(vbt::autograd::is_streaming_backwards_enabled());

  // Second run should fail immediately because the toggle is now disabled.
  std::vector<OptionalTensor> seed2(1);
  seed2[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);

  try {
    vbt::autograd::run_backward(intrusive_ptr<Node>(root.get()), seed2, {});
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("CUDA autograd is disabled"), std::string::npos) << msg;
  }
}
#endif  // VBT_WITH_CUDA
