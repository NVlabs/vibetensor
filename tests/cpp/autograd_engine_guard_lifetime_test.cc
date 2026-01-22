// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
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

using vbt::autograd::Node;
using vbt::autograd::OptionalTensor;
using vbt::autograd::StreamKind;
using vbt::autograd::NodeStreamInfo;
using vbt::core::TensorImpl;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::intrusive_ptr;

namespace {

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

#if VBT_WITH_CUDA
struct StreamCheckedNode final : Node {
  explicit StreamCheckedNode(std::string nm,
                             Device dev,
                             std::uint64_t stream_id,
                             std::optional<TensorImpl> out0) : out0_(std::move(out0)) {
    name = std::move(nm);
    NodeStreamInfo info;
    info.has_canonical_stream = true;
    info.device = dev;
    info.stream_id = stream_id;
    mutable_stream_info() = info;
  }

  uint32_t num_inputs() const noexcept override { return 1; }
  StreamKind stream_kind() const noexcept override { return StreamKind::CudaAllowlisted; }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    (void)grads_in;

    const NodeStreamInfo& si = stream_info();
    const auto dev_idx = static_cast<vbt::cuda::DeviceIndex>(si.device.index);
    vbt::cuda::Stream cur = vbt::cuda::getCurrentStream(dev_idx);
    last_apply_stream_id_ = cur.id();
    if (cur.id() != si.stream_id) {
      throw std::runtime_error(
          "StreamCheckedNode: current stream does not match canonical stream");
    }

    std::vector<OptionalTensor> out(1);
    if (out0_.has_value()) {
      out[0] = out0_.value();
    }
    return out;
  }

  std::uint64_t last_apply_stream_id() const noexcept { return last_apply_stream_id_; }

 private:
  std::optional<TensorImpl> out0_;
  std::uint64_t last_apply_stream_id_{0};
};

struct StreamingToggleRestore {
  bool prev;
  StreamingToggleRestore() : prev(vbt::autograd::is_streaming_backwards_enabled()) {
    vbt::autograd::set_streaming_backwards_enabled(true);
  }
  ~StreamingToggleRestore() {
    vbt::autograd::set_streaming_backwards_enabled(prev);
  }
};

struct MtToggleRestore {
  bool prev;
  MtToggleRestore() : prev(vbt::autograd::is_multithreading_enabled()) {
    vbt::autograd::set_multithreading_enabled(true);
  }
  ~MtToggleRestore() {
    vbt::autograd::set_multithreading_enabled(prev);
  }
};

#if VBT_AUTOGRAD_TESTING
static std::uint64_t g_route_hook_calls = 0;
static void route_hook(const Node& producer,
                       const Node& consumer,
                       std::uint64_t current_stream_id) {
  (void)consumer;
  ++g_route_hook_calls;
  EXPECT_EQ(producer.stream_kind(), StreamKind::CudaAllowlisted);
  EXPECT_EQ(current_stream_id, producer.stream_info().stream_id)
      << "producer=" << producer.name;
}

struct RouteHookGuard {
  RouteHookGuard() {
    g_route_hook_calls = 0;
    vbt::autograd::_test_set_route_hook(&route_hook);
  }
  ~RouteHookGuard() {
    vbt::autograd::_test_set_route_hook(nullptr);
  }
};
#endif
#endif

} // namespace

#if VBT_WITH_CUDA && VBT_AUTOGRAD_TESTING
TEST(AutogradGuardLifetimeTest, GUARD1StreamGuardCoversApplyAndRouting) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  StreamingToggleRestore toggle_guard;
  RouteHookGuard hook_guard;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);

  Stream S_base = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  Stream S_root = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  Stream S_mid  = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  Stream S_sink = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);

  ASSERT_NE(S_base.id(), S_root.id());
  ASSERT_NE(S_root.id(), S_mid.id());
  ASSERT_NE(S_mid.id(), S_sink.id());

  // Graph: root -> mid -> sink
  auto sink = vbt::core::make_intrusive<StreamCheckedNode>(
      "sink", Device::cuda(0), S_sink.id(), /*out0=*/std::nullopt);
  sink->next_edges.resize(1);  // required size contract

  TensorImpl mid_out = make_cuda_dense_f32({4}, 3.0f, /*dev=*/0);
  auto mid = vbt::core::make_intrusive<StreamCheckedNode>(
      "mid", Device::cuda(0), S_mid.id(), /*out0=*/mid_out);
  mid->next_edges.resize(1);
  mid->next_edges[0].fn = intrusive_ptr<Node>(sink.get(), /*add_ref=*/true);
  mid->next_edges[0].input_nr = 0;

  TensorImpl root_out = make_cuda_dense_f32({4}, 2.0f, /*dev=*/0);
  auto root = vbt::core::make_intrusive<StreamCheckedNode>(
      "root", Device::cuda(0), S_root.id(), /*out0=*/root_out);
  root->next_edges.resize(1);
  root->next_edges[0].fn = intrusive_ptr<Node>(mid.get(), /*add_ref=*/true);
  root->next_edges[0].input_nr = 0;

  // Seed the root with a defined CUDA grad.
  std::vector<OptionalTensor> initial(1);
  initial[0] = make_cuda_dense_f32({4}, 1.0f, /*dev=*/0);

  // Ensure the engine restores the previous stream after each node.
  {
    vbt::cuda::CUDAStreamGuard base_guard(S_base);
    EXPECT_EQ(vbt::cuda::getCurrentStream(dev_idx).id(), S_base.id());

    vbt::autograd::run_backward(intrusive_ptr<Node>(root.get(), /*add_ref=*/true), initial, {});

    EXPECT_EQ(vbt::cuda::getCurrentStream(dev_idx).id(), S_base.id());
  }

  EXPECT_EQ(root->last_apply_stream_id(), S_root.id());
  EXPECT_EQ(mid->last_apply_stream_id(), S_mid.id());
  EXPECT_EQ(sink->last_apply_stream_id(), S_sink.id());

  // Two routed edges: root->mid and mid->sink.
  EXPECT_EQ(g_route_hook_calls, 2u);
}

TEST(AutogradGuardLifetimeTest, GUARD1StreamGuardCoversApplyAndRouting_MTEnabled) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  StreamingToggleRestore toggle_guard;
  MtToggleRestore mt_guard;
  RouteHookGuard hook_guard;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);

  Stream S_base = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  Stream S_root = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  Stream S_mid  = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  Stream S_sink = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);

  ASSERT_NE(S_base.id(), S_root.id());
  ASSERT_NE(S_root.id(), S_mid.id());
  ASSERT_NE(S_mid.id(), S_sink.id());

  // Graph: root -> mid -> sink
  auto sink = vbt::core::make_intrusive<StreamCheckedNode>(
      "sink", Device::cuda(0), S_sink.id(), /*out0=*/std::nullopt);
  sink->next_edges.resize(1);  // required size contract

  TensorImpl mid_out = make_cuda_dense_f32({4}, 3.0f, /*dev=*/0);
  auto mid = vbt::core::make_intrusive<StreamCheckedNode>(
      "mid", Device::cuda(0), S_mid.id(), /*out0=*/mid_out);
  mid->next_edges.resize(1);
  mid->next_edges[0].fn = intrusive_ptr<Node>(sink.get(), /*add_ref=*/true);
  mid->next_edges[0].input_nr = 0;

  TensorImpl root_out = make_cuda_dense_f32({4}, 2.0f, /*dev=*/0);
  auto root = vbt::core::make_intrusive<StreamCheckedNode>(
      "root", Device::cuda(0), S_root.id(), /*out0=*/root_out);
  root->next_edges.resize(1);
  root->next_edges[0].fn = intrusive_ptr<Node>(mid.get(), /*add_ref=*/true);
  root->next_edges[0].input_nr = 0;

  // Seed the root with a defined CUDA grad.
  std::vector<OptionalTensor> initial(1);
  initial[0] = make_cuda_dense_f32({4}, 1.0f, /*dev=*/0);

  // Ensure the engine restores the previous stream after each node.
  {
    vbt::cuda::CUDAStreamGuard base_guard(S_base);
    EXPECT_EQ(vbt::cuda::getCurrentStream(dev_idx).id(), S_base.id());

    vbt::autograd::run_backward(intrusive_ptr<Node>(root.get(), /*add_ref=*/true), initial, {});

    EXPECT_EQ(vbt::cuda::getCurrentStream(dev_idx).id(), S_base.id());
  }

  EXPECT_EQ(root->last_apply_stream_id(), S_root.id());
  EXPECT_EQ(mid->last_apply_stream_id(), S_mid.id());
  EXPECT_EQ(sink->last_apply_stream_id(), S_sink.id());

  // Two routed edges: root->mid and mid->sink.
  EXPECT_EQ(g_route_hook_calls, 2u);
}
#endif
