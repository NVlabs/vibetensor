// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <thread>
#include <stdexcept>
#include <string>
#include <vector>

#include "vbt/autograd/copy_like.h"
#include "vbt/autograd/engine.h"
#include "vbt/autograd/engine_toggles.h"
#include "vbt/autograd/function.h"
#include "vbt/autograd/node.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/types.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"

#if VBT_WITH_CUDA
#include "vbt/cuda/device.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/allocator.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#include <cuda_runtime_api.h>
#endif

using vbt::autograd::AutogradDeviceMode;
using vbt::autograd::CopyLikeNode;
using vbt::autograd::InputMeta;
using vbt::autograd::Node;
using vbt::autograd::NodeStreamInfo;
using vbt::autograd::OptionalTensor;
using vbt::autograd::StreamKind;
using vbt::autograd::run_backward;
using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::TensorImpl;
using vbt::core::intrusive_ptr;

#if VBT_AUTOGRAD_TESTING
namespace vbt { namespace autograd {
void _test_clear_last_backward_snapshot() noexcept;
vbt::core::Device _test_last_backward_autograd_device() noexcept;
std::uint64_t _test_last_backward_cuda_device_synchronizes() noexcept;
}} // namespace vbt::autograd
#endif

namespace {

struct DeviceModeRestore {
  AutogradDeviceMode prev;
  DeviceModeRestore() : prev(vbt::autograd::get_device_mode()) {}
  ~DeviceModeRestore() { vbt::autograd::set_device_mode(prev); }
};

struct MtToggleRestore {
  bool prev;
  explicit MtToggleRestore(bool v) : prev(vbt::autograd::is_multithreading_enabled()) {
    vbt::autograd::set_multithreading_enabled(v);
  }
  ~MtToggleRestore() { vbt::autograd::set_multithreading_enabled(prev); }
};

struct StreamingToggleRestore {
  bool prev;
  explicit StreamingToggleRestore(bool v) : prev(vbt::autograd::is_streaming_backwards_enabled()) {
    vbt::autograd::set_streaming_backwards_enabled(v);
  }
  ~StreamingToggleRestore() { vbt::autograd::set_streaming_backwards_enabled(prev); }
};

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

  TensorImpl t(st, sizes, strides, /*offset=*/0, ScalarType::Float32, Device::cpu());
  if (t.numel() == 0) {
    return t;
  }
  float* p = static_cast<float*>(t.data());
  for (std::size_t i = 0; i < ne; ++i) {
    p[i] = fill;
  }
  return t;
}

#if VBT_WITH_CUDA
static TensorImpl make_cuda_dense_f32(const std::vector<int64_t>& sizes, float fill, int dev = 0) {
  const std::size_t ne = [&]() {
    std::size_t n = 1;
    for (auto s : sizes) {
      n *= static_cast<std::size_t>(s == 0 ? 1 : s);
    }
    return n;
  }();
  const std::size_t nbytes = ne * sizeof(float);

  using vbt::cuda::DeviceGuard;
  using vbt::cuda::DeviceIndex;
  const auto dev_idx = static_cast<DeviceIndex>(dev);
  DeviceGuard dg(dev_idx);

  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  if (nbytes > 0) {
    std::vector<float> host(ne, fill);
    cudaError_t st = cudaMemcpy(storage->data(), host.data(), nbytes, cudaMemcpyHostToDevice);
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

  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0, ScalarType::Float32, Device::cuda(dev));
}
#endif

struct CpuSink final : Node {
  CpuSink() { name = "CpuSink"; }
  uint32_t num_inputs() const noexcept override { return 1; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    (void)grads_in;
    return {}; // sink
  }
};

struct CpuRoot1 final : Node {
  explicit CpuRoot1(intrusive_ptr<Node> next) {
    name = "CpuRoot1";
    next_edges.resize(1);
    next_edges[0].fn = std::move(next);
    next_edges[0].input_nr = 0;
  }
  uint32_t num_inputs() const noexcept override { return 1; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    std::vector<OptionalTensor> out(1);
    if (!grads_in.empty()) {
      out[0] = std::move(grads_in[0]);
    }
    return out;
  }
};

struct CpuCopyLikeRoot2 final : Node, CopyLikeNode {
  CpuCopyLikeRoot2(intrusive_ptr<Node> cpu_sink, intrusive_ptr<Node> cuda_sink) {
    name = "CpuCopyLikeRoot2";
    next_edges.resize(2);
    next_edges[0] = vbt::autograd::Edge{std::move(cpu_sink), 0};
    next_edges[1] = vbt::autograd::Edge{std::move(cuda_sink), 0};
  }

  uint32_t num_inputs() const noexcept override { return 2; }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    std::vector<OptionalTensor> out(2);
    if (!grads_in.empty()) {
      out[0] = std::move(grads_in[0]);
    }
#if VBT_WITH_CUDA
    out[1] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
#else
    (void)grads_in;
#endif
    return out;
  }
};

struct CpuCopyLikeRoot2Dev final : Node, CopyLikeNode {
  CpuCopyLikeRoot2Dev(intrusive_ptr<Node> cpu_sink, intrusive_ptr<Node> cuda_sink, int cuda_dev)
      : cuda_dev_(cuda_dev) {
    name = "CpuCopyLikeRoot2Dev";
    next_edges.resize(2);
    next_edges[0] = vbt::autograd::Edge{std::move(cpu_sink), 0};
    next_edges[1] = vbt::autograd::Edge{std::move(cuda_sink), 0};
  }

  uint32_t num_inputs() const noexcept override { return 2; }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    std::vector<OptionalTensor> out(2);
    if (!grads_in.empty()) {
      out[0] = std::move(grads_in[0]);
    }
#if VBT_WITH_CUDA
    out[1] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/cuda_dev_);
#else
    (void)grads_in;
#endif
    return out;
  }

 private:
  int cuda_dev_{0};
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

struct CudaAllowlistedValidatableSink final : Node, vbt::autograd::ValidatableNode {
  CudaAllowlistedValidatableSink(Device dev,
                                 std::uint64_t stream_id,
                                 std::vector<InputMeta> metas)
      : metas_(std::move(metas)) {
    NodeStreamInfo info;
    info.has_canonical_stream = true;
    info.device = dev;
    info.stream_id = stream_id;
    mutable_stream_info() = info;
    name = "CudaAllowlistedValidatableSink";
    // Keep validation happy: ValidatableNode expects grads_out.size() == input_metas.size().
    // Make this a sink by leaving edges null.
    next_edges.resize(static_cast<std::size_t>(metas_.size()));
  }

  uint32_t num_inputs() const noexcept override { return 1; }
  StreamKind stream_kind() const noexcept override { return StreamKind::CudaAllowlisted; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    (void)grads_in;
    return std::vector<OptionalTensor>(metas_.size());
  }

  const std::vector<InputMeta>& input_metas() const noexcept override { return metas_; }

 private:
  std::vector<InputMeta> metas_;
};

struct CpuNonCopyLikeCudaOutRoot1 final : Node {
  explicit CpuNonCopyLikeCudaOutRoot1(intrusive_ptr<Node> next) {
    name = "CpuNonCopyLikeCudaOutRoot1";
    next_edges.resize(1);
    next_edges[0].fn = std::move(next);
    next_edges[0].input_nr = 0;
  }
  uint32_t num_inputs() const noexcept override { return 1; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    (void)grads_in;
    std::vector<OptionalTensor> out(1);
    out[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
    return out;
  }
};

struct CudaProducerRoot1 final : Node {
  explicit CudaProducerRoot1(Device dev, std::uint64_t stream_id, intrusive_ptr<Node> next) {
    NodeStreamInfo info;
    info.has_canonical_stream = true;
    info.device = dev;
    info.stream_id = stream_id;
    mutable_stream_info() = info;
    name = "CudaProducerRoot1";

    next_edges.resize(1);
    next_edges[0].fn = std::move(next);
    next_edges[0].input_nr = 0;
  }

  uint32_t num_inputs() const noexcept override { return 1; }
  StreamKind stream_kind() const noexcept override { return StreamKind::CudaAllowlisted; }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    std::vector<OptionalTensor> out(1);
    if (!grads_in.empty()) {
      out[0] = std::move(grads_in[0]);
    }
    return out;
  }
};
#endif

#if VBT_AUTOGRAD_TESTING && VBT_WITH_CUDA
static std::uint64_t g_route_hook_sync_calls = 0;

static void route_hook_sync_dev0(const Node& producer,
                                 const Node& consumer,
                                 std::uint64_t current_stream_id) {
  (void)producer;
  (void)consumer;
  (void)current_stream_id;
  ++g_route_hook_sync_calls;
  const auto dev0 = static_cast<vbt::cuda::DeviceIndex>(0);
  vbt::cuda::Stream cur = vbt::cuda::getCurrentStream(dev0);
  cur.synchronize();
}

struct RouteHookSyncDev0Guard {
  RouteHookSyncDev0Guard() {
    g_route_hook_sync_calls = 0;
    vbt::autograd::_test_set_route_hook(&route_hook_sync_dev0);
  }
  ~RouteHookSyncDev0Guard() {
    vbt::autograd::_test_set_route_hook(nullptr);
  }
};

static const Node* g_route_watch_consumer = nullptr;
static int g_route_watch_device_index = -1;
static std::uint64_t g_route_watch_expected_stream_id = 0;
static std::uint64_t g_route_watch_calls = 0;

static void route_hook_watch_consumer_stream(const Node& producer,
                                             const Node& consumer,
                                             std::uint64_t current_stream_id) {
  if (!g_route_watch_consumer || &consumer != g_route_watch_consumer) {
    return;
  }
  ++g_route_watch_calls;

  if (g_route_watch_device_index < 0) {
    ADD_FAILURE() << "route_hook_watch_consumer_stream: missing watch device index";
    return;
  }

  const auto dev = static_cast<vbt::cuda::DeviceIndex>(g_route_watch_device_index);
  const std::uint64_t got = vbt::cuda::getCurrentStream(dev).id();
  EXPECT_EQ(current_stream_id, got)
      << "producer=" << producer.name << ", consumer=" << consumer.name;
  EXPECT_EQ(got, g_route_watch_expected_stream_id)
      << "producer=" << producer.name << ", consumer=" << consumer.name;
}

struct RouteHookWatchGuard {
  RouteHookWatchGuard() {
    g_route_watch_calls = 0;
    vbt::autograd::_test_set_route_hook(&route_hook_watch_consumer_stream);
  }
  ~RouteHookWatchGuard() {
    vbt::autograd::_test_set_route_hook(nullptr);
    g_route_watch_consumer = nullptr;
    g_route_watch_device_index = -1;
    g_route_watch_expected_stream_id = 0;
  }
};
#endif

} // namespace

#if VBT_WITH_CUDA
TEST(AutogradMultiDeviceSingleGpu, CpuAndCudaConsumersWorkWithExplicitCopyLikeProducer) {
#if !VBT_AUTOGRAD_TESTING
  GTEST_SKIP() << "VBT_AUTOGRAD_TESTING disabled";
#else
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceModeRestore mode_guard;
  MtToggleRestore mt_guard(true);
  StreamingToggleRestore stream_guard(true);

  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);
  vbt::autograd::_test_clear_last_backward_snapshot();

  using vbt::cuda::DeviceGuard;
  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  DeviceGuard dg(dev_idx);
  Stream cur = vbt::cuda::getCurrentStream(dev_idx);

  auto cpu_sink = vbt::core::make_intrusive<CpuSink>();
  auto cuda_sink = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), cur.id());
  auto root = vbt::core::make_intrusive<CpuCopyLikeRoot2>(
      intrusive_ptr<Node>(cpu_sink.get()), intrusive_ptr<Node>(cuda_sink.get()));

  std::vector<OptionalTensor> seed(2);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);
  seed[1] = make_cpu_dense_f32({2}, 1.0f);

  ASSERT_NO_THROW(run_backward(intrusive_ptr<Node>(root.get()), seed, {}));
  EXPECT_EQ(vbt::autograd::_test_last_backward_autograd_device(), Device::cuda(0));
  EXPECT_EQ(vbt::autograd::_test_last_backward_cuda_device_synchronizes(), 1u);
  EXPECT_EQ(vbt::autograd::_test_last_backward_cuda_devices_snapshot(), std::vector<int>({0}));
  EXPECT_EQ(vbt::autograd::_test_last_backward_cuda_lane_devices_snapshot(), std::vector<int>({0}));
#endif
}
#endif

#if VBT_WITH_CUDA
TEST(AutogradMultiDeviceSingleGpu, GuardDisciplineRestoresOwnerTlsStreamOnDevice1) {
#if !VBT_AUTOGRAD_TESTING
  GTEST_SKIP() << "VBT_AUTOGRAD_TESTING disabled";
#else
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need at least 2 CUDA devices";
  }

  DeviceModeRestore mode_guard;
  MtToggleRestore mt_guard(true);
  StreamingToggleRestore stream_guard(true);

  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);

  using vbt::cuda::DeviceGuard;
  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  using vbt::cuda::CUDAStreamGuard;

  const auto dev1 = static_cast<DeviceIndex>(1);
  DeviceGuard dg1(dev1);

  cudaStream_t raw_base = nullptr;
  cudaStream_t raw_node = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&raw_base, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&raw_node, cudaStreamNonBlocking), cudaSuccess);

  struct StreamCleanup {
    cudaStream_t s;
    ~StreamCleanup() {
      if (s) {
        (void)cudaStreamDestroy(s);
      }
    }
  };

  StreamCleanup cleanup_base{raw_base};
  StreamCleanup cleanup_node{raw_node};

  Stream S_base(Stream::UNCHECKED,
                static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(raw_base)),
                dev1);
  Stream S_node(Stream::UNCHECKED,
                static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(raw_node)),
                dev1);
  ASSERT_NE(S_base.id(), 0u);
  ASSERT_NE(S_node.id(), 0u);
  ASSERT_NE(S_base.id(), S_node.id());

  CUDAStreamGuard base_guard(S_base);
  EXPECT_EQ(vbt::cuda::getCurrentStream(dev1).id(), S_base.id());

  auto cpu_sink = vbt::core::make_intrusive<CpuSink>();
  auto cuda_sink = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(1), S_node.id());
  auto root = vbt::core::make_intrusive<CpuCopyLikeRoot2Dev>(
      intrusive_ptr<Node>(cpu_sink.get()),
      intrusive_ptr<Node>(cuda_sink.get()),
      /*cuda_dev=*/1);

  std::vector<OptionalTensor> seed(2);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);
  seed[1] = make_cpu_dense_f32({2}, 1.0f);

  // Ensure backward begins while current device != autograd device.
  {
    const auto dev0 = static_cast<DeviceIndex>(0);
    DeviceGuard dg0(dev0);
    ASSERT_NO_THROW(run_backward(intrusive_ptr<Node>(root.get()), seed, {}));
  }

  EXPECT_EQ(vbt::cuda::getCurrentStream(dev1).id(), S_base.id());

  // Drain allocator state before destroying the raw streams created by this
  // test. The caching allocator tracks stream ids for deferred frees and may
  // query their capture state during subsequent allocations.
  {
    (void)cudaDeviceSynchronize();
    vbt::cuda::Allocator::get(dev1).process_events();
    vbt::cuda::Allocator::get(dev1).emptyCache();
  }
#endif
}
#endif

#if VBT_WITH_CUDA
TEST(AutogradMultiDeviceSingleGpu, CpuWorkerExecutesAccumulateGradOnCanonicalDevice) {
#if !VBT_AUTOGRAD_TESTING
  GTEST_SKIP() << "VBT_AUTOGRAD_TESTING disabled";
#else
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need at least 2 CUDA devices";
  }

  DeviceModeRestore mode_guard;
  MtToggleRestore mt_guard(true);
  StreamingToggleRestore stream_guard(true);

  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);
  vbt::autograd::_test_clear_last_backward_snapshot();

  using vbt::cuda::DeviceGuard;
  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;

  const auto dev0 = static_cast<DeviceIndex>(0);
  DeviceGuard dg0(dev0);
  Stream cur0 = vbt::cuda::getCurrentStream(dev0);

  auto sink0 = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), cur0.id());

  // AccumulateGrad for a CUDA leaf on cuda:1 executes on the CPU lane. Ensure
  // the worker pins to the node's canonical device so prepare_consumer_stream_and_wait_mt
  // does not fail with:
  //   "missing or mismatched canonical stream on CUDA node".
  TensorImpl leaf1 = make_cuda_dense_f32({2}, 0.0f, /*dev=*/1);
  vbt::autograd::AutogradMeta* meta1 = vbt::autograd::get_autograd_meta(leaf1, /*create_if_missing=*/true);
  ASSERT_NE(meta1, nullptr);

  auto acc1 = vbt::core::make_intrusive<vbt::autograd::AccumulateGrad>(meta1);
  _tag_accumulategrad_cuda_leaf(*acc1, leaf1);

  struct CpuCopyLikeRootToCuda0AndAcc1 final : Node, CopyLikeNode {
    CpuCopyLikeRootToCuda0AndAcc1(intrusive_ptr<Node> cuda_sink, intrusive_ptr<Node> acc) {
      name = "CpuCopyLikeRootToCuda0AndAcc1";
      next_edges.resize(2);
      next_edges[0] = vbt::autograd::Edge{std::move(cuda_sink), 0};
      next_edges[1] = vbt::autograd::Edge{std::move(acc), 0};
    }

    uint32_t num_inputs() const noexcept override { return 2; }

    std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
      (void)grads_in;
      std::vector<OptionalTensor> out(2);
      out[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
      out[1] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/1);
      return out;
    }
  };

  auto root = vbt::core::make_intrusive<CpuCopyLikeRootToCuda0AndAcc1>(
      intrusive_ptr<Node>(sink0.get()), intrusive_ptr<Node>(acc1.get()));

  std::vector<OptionalTensor> seed(2);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);
  seed[1] = make_cpu_dense_f32({2}, 1.0f);

  ASSERT_NO_THROW(run_backward(intrusive_ptr<Node>(root.get()), seed, {}));
  EXPECT_EQ(vbt::autograd::_test_last_backward_autograd_device(), Device::cuda(0));
  EXPECT_EQ(vbt::autograd::_test_last_backward_cuda_devices_snapshot(), std::vector<int>({0, 1}));
  EXPECT_EQ(vbt::autograd::_test_last_backward_cuda_lane_devices_snapshot(), std::vector<int>({0}));
#endif
}
#endif

#if VBT_WITH_CUDA
TEST(AutogradMultiDeviceSingleGpu, PerDeviceCudaWorkersExecuteConcurrently) {
#if !VBT_AUTOGRAD_TESTING
  GTEST_SKIP() << "VBT_AUTOGRAD_TESTING disabled";
#else
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need at least 2 CUDA devices";
  }

  DeviceModeRestore mode_guard;
  MtToggleRestore mt_guard(true);
  StreamingToggleRestore stream_guard(true);

  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);

  struct Barrier {
    std::atomic<int> started{0};
  } barrier;

  struct CudaBarrierSink final : Node {
    explicit CudaBarrierSink(Barrier* barrier, int dev)
        : barrier_(barrier), expected_dev_(dev) {
      NodeStreamInfo info;
      info.has_canonical_stream = true;
      info.device = Device::cuda(dev);
      info.stream_id = 0; // default stream
      mutable_stream_info() = info;
      name = std::string("CudaBarrierSink") + std::to_string(dev);
    }

    uint32_t num_inputs() const noexcept override { return 1; }
    StreamKind stream_kind() const noexcept override { return StreamKind::CudaAllowlisted; }

    std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
      (void)grads_in;

      int dev = -1;
      const cudaError_t st = cudaGetDevice(&dev);
      if (st != cudaSuccess) {
        throw std::runtime_error("cudaGetDevice failed in CudaBarrierSink");
      }
      observed_dev_ = dev;

      barrier_->started.fetch_add(1, std::memory_order_relaxed);

      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (barrier_->started.load(std::memory_order_relaxed) < 2) {
        if (std::chrono::steady_clock::now() >= deadline) {
          throw std::runtime_error("CudaBarrierSink timed out waiting for peer CUDA worker");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      return {};
    }

    Barrier* barrier_{nullptr};
    int expected_dev_{-1};
    int observed_dev_{-1};
  };

  auto sink0 = vbt::core::make_intrusive<CudaBarrierSink>(&barrier, 0);
  auto sink1 = vbt::core::make_intrusive<CudaBarrierSink>(&barrier, 1);

  struct Root final : Node, CopyLikeNode {
    Root(intrusive_ptr<Node> a, intrusive_ptr<Node> b) {
      name = "RootToCudaBarrierSinks";
      next_edges.resize(2);
      next_edges[0] = vbt::autograd::Edge{std::move(a), 0};
      next_edges[1] = vbt::autograd::Edge{std::move(b), 0};
    }

    uint32_t num_inputs() const noexcept override { return 2; }

    std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
      (void)grads_in;
      std::vector<OptionalTensor> out(2);
      out[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
      out[1] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/1);
      return out;
    }
  };

  auto root = vbt::core::make_intrusive<Root>(
      intrusive_ptr<Node>(sink0.get()), intrusive_ptr<Node>(sink1.get()));

  std::vector<OptionalTensor> seed(2);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);
  seed[1] = make_cpu_dense_f32({2}, 1.0f);

  ASSERT_NO_THROW(run_backward(intrusive_ptr<Node>(root.get()), seed, {}));
  EXPECT_EQ(sink0->observed_dev_, 0);
  EXPECT_EQ(sink1->observed_dev_, 1);
#endif
}
#endif

#if VBT_WITH_CUDA
TEST(AutogradMultiDeviceSingleGpu, MultiQueueCancellationDoesNotHang) {
#if !VBT_AUTOGRAD_TESTING
  GTEST_SKIP() << "VBT_AUTOGRAD_TESTING disabled";
#else
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need at least 2 CUDA devices";
  }

  DeviceModeRestore mode_guard;
  MtToggleRestore mt_guard(true);
  StreamingToggleRestore stream_guard(true);

  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);

  struct Barrier {
    std::atomic<int> started{0};
  } barrier;

  auto cpu_sink = vbt::core::make_intrusive<CpuSink>();

  struct CudaSleepSink final : Node {
    CudaSleepSink(Barrier* barrier, int dev)
        : barrier_(barrier), expected_dev_(dev) {
      NodeStreamInfo info;
      info.has_canonical_stream = true;
      info.device = Device::cuda(dev);
      info.stream_id = 0; // default stream
      mutable_stream_info() = info;
      name = std::string("CudaSleepSink") + std::to_string(dev);
    }

    uint32_t num_inputs() const noexcept override { return 1; }
    StreamKind stream_kind() const noexcept override { return StreamKind::CudaAllowlisted; }

    std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
      (void)grads_in;

      int dev = -1;
      const cudaError_t st = cudaGetDevice(&dev);
      if (st != cudaSuccess) {
        throw std::runtime_error("cudaGetDevice failed in CudaSleepSink");
      }
      observed_dev_ = dev;
      if (dev != expected_dev_) {
        throw std::runtime_error("CudaSleepSink ran on wrong device");
      }

      barrier_->started.fetch_add(1, std::memory_order_relaxed);

      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (barrier_->started.load(std::memory_order_relaxed) < 2) {
        if (std::chrono::steady_clock::now() >= deadline) {
          throw std::runtime_error("CudaSleepSink timed out waiting for peer CUDA worker");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      // Keep one device in-flight while the other throws.
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      return {};
    }

    Barrier* barrier_{nullptr};
    int expected_dev_{-1};
    int observed_dev_{-1};
  };

  struct CudaBoomSink final : Node {
    CudaBoomSink(Barrier* barrier, int dev)
        : barrier_(barrier), expected_dev_(dev) {
      NodeStreamInfo info;
      info.has_canonical_stream = true;
      info.device = Device::cuda(dev);
      info.stream_id = 0; // default stream
      mutable_stream_info() = info;
      name = std::string("CudaBoomSink") + std::to_string(dev);
    }

    uint32_t num_inputs() const noexcept override { return 1; }
    StreamKind stream_kind() const noexcept override { return StreamKind::CudaAllowlisted; }

    std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
      (void)grads_in;

      int dev = -1;
      const cudaError_t st = cudaGetDevice(&dev);
      if (st != cudaSuccess) {
        throw std::runtime_error("cudaGetDevice failed in CudaBoomSink");
      }
      observed_dev_ = dev;
      if (dev != expected_dev_) {
        throw std::runtime_error("CudaBoomSink ran on wrong device");
      }

      barrier_->started.fetch_add(1, std::memory_order_relaxed);

      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (barrier_->started.load(std::memory_order_relaxed) < 2) {
        if (std::chrono::steady_clock::now() >= deadline) {
          throw std::runtime_error("CudaBoomSink timed out waiting for peer CUDA worker");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      throw std::runtime_error("boom");
    }

    Barrier* barrier_{nullptr};
    int expected_dev_{-1};
    int observed_dev_{-1};
  };

  auto cuda_sink0 = vbt::core::make_intrusive<CudaSleepSink>(&barrier, 0);
  auto cuda_sink1 = vbt::core::make_intrusive<CudaBoomSink>(&barrier, 1);

  struct Root final : Node, CopyLikeNode {
    Root(intrusive_ptr<Node> cpu, intrusive_ptr<Node> a, intrusive_ptr<Node> b) {
      name = "RootForCancellation";
      next_edges.resize(3);
      next_edges[0] = vbt::autograd::Edge{std::move(cpu), 0};
      next_edges[1] = vbt::autograd::Edge{std::move(a), 0};
      next_edges[2] = vbt::autograd::Edge{std::move(b), 0};
    }

    uint32_t num_inputs() const noexcept override { return 3; }

    std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
      (void)grads_in;
      std::vector<OptionalTensor> out(3);
      out[0] = make_cpu_dense_f32({2}, 1.0f);
      out[1] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
      out[2] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/1);
      return out;
    }
  };

  auto root = vbt::core::make_intrusive<Root>(
      intrusive_ptr<Node>(cpu_sink.get()),
      intrusive_ptr<Node>(cuda_sink0.get()),
      intrusive_ptr<Node>(cuda_sink1.get()));

  std::vector<OptionalTensor> seed(3);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);
  seed[1] = make_cpu_dense_f32({2}, 1.0f);
  seed[2] = make_cpu_dense_f32({2}, 1.0f);

  auto fut = std::async(
      std::launch::async,
      [root_keep = intrusive_ptr<Node>(root.get()), seed_keep = std::move(seed)]() mutable
          -> std::exception_ptr {
        try {
          run_backward(root_keep, std::move(seed_keep), {});
          return nullptr;
        } catch (...) {
          return std::current_exception();
        }
      });

  if (fut.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
    ADD_FAILURE() << "run_backward hung during multi-queue cancellation";
    std::abort();
  }

  std::exception_ptr ep = fut.get();
  ASSERT_NE(ep, nullptr);

  try {
    std::rethrow_exception(ep);
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()), "boom");
  } catch (...) {
    FAIL() << "Expected std::runtime_error";
  }

  EXPECT_EQ(cuda_sink0->observed_dev_, 0);
  EXPECT_EQ(cuda_sink1->observed_dev_, 1);
#endif
}
#endif

#if VBT_WITH_CUDA
TEST(AutogradMultiDeviceSingleGpu, RouteGuardUsesProducerDeviceIndex) {
#if !VBT_AUTOGRAD_TESTING
  GTEST_SKIP() << "VBT_AUTOGRAD_TESTING disabled";
#else
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need at least 2 CUDA devices";
  }

  DeviceModeRestore mode_guard;
  MtToggleRestore mt_guard(true);
  StreamingToggleRestore stream_guard(true);
  RouteHookSyncDev0Guard hook_guard;

  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);

  using vbt::cuda::DeviceGuard;
  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;

  // Create a non-default stream on cuda:1. If this stream handle is ever
  // paired with cuda:0, stream operations should fail.
  const auto dev1 = static_cast<DeviceIndex>(1);
  DeviceGuard dg1(dev1);

  cudaStream_t raw_prod = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&raw_prod, cudaStreamNonBlocking), cudaSuccess);

  struct StreamCleanup {
    cudaStream_t s;
    ~StreamCleanup() {
      if (s) {
        (void)cudaStreamDestroy(s);
      }
    }
  };

  StreamCleanup cleanup_prod{raw_prod};

  Stream S_prod(Stream::UNCHECKED,
                static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(raw_prod)),
                dev1);
  ASSERT_NE(S_prod.id(), 0u);

  auto sink0 = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), /*stream_id=*/0);

  // Allowlisted producer executes on cuda:1 but produces a CUDA grad on cuda:0.
  // This is allowed only for CopyLike nodes.
  struct CudaCopyLikeProducer1To0 final : Node, CopyLikeNode {
    CudaCopyLikeProducer1To0(intrusive_ptr<Node> next, Device exec_dev, std::uint64_t stream_id) {
      NodeStreamInfo info;
      info.has_canonical_stream = true;
      info.device = exec_dev;
      info.stream_id = stream_id;
      mutable_stream_info() = info;
      name = "CudaCopyLikeProducer1To0";

      next_edges.resize(1);
      next_edges[0].fn = std::move(next);
      next_edges[0].input_nr = 0;
    }

    uint32_t num_inputs() const noexcept override { return 1; }
    StreamKind stream_kind() const noexcept override { return StreamKind::CudaAllowlisted; }

    std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
      (void)grads_in;
      std::vector<OptionalTensor> out(1);
      out[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
      return out;
    }
  };

  auto producer = vbt::core::make_intrusive<CudaCopyLikeProducer1To0>(
      intrusive_ptr<Node>(sink0.get()),
      Device::cuda(1),
      S_prod.id());

  struct CpuCopyLikeRootToCudaDev final : Node, CopyLikeNode {
    CpuCopyLikeRootToCudaDev(intrusive_ptr<Node> next, int cuda_dev) : cuda_dev_(cuda_dev) {
      name = "CpuCopyLikeRootToCudaDev";
      next_edges.resize(1);
      next_edges[0].fn = std::move(next);
      next_edges[0].input_nr = 0;
    }

    uint32_t num_inputs() const noexcept override { return 1; }

    std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
      (void)grads_in;
      std::vector<OptionalTensor> out(1);
      out[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/cuda_dev_);
      return out;
    }

   private:
    int cuda_dev_{0};
  };

  auto root = vbt::core::make_intrusive<CpuCopyLikeRootToCudaDev>(
      intrusive_ptr<Node>(producer.get()),
      /*cuda_dev=*/1);

  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);

  ASSERT_NO_THROW(run_backward(intrusive_ptr<Node>(root.get()), seed, {}));
  EXPECT_EQ(g_route_hook_sync_calls, 2u);

  // Drain allocator state before destroying the raw stream created by this test.
  // The caching allocator tracks stream ids for deferred frees and may query their
  // capture state during subsequent allocations.
  {
    (void)cudaDeviceSynchronize();
    vbt::cuda::Allocator::get(dev1).process_events();
    vbt::cuda::Allocator::get(dev1).emptyCache();
  }
#endif
}

#endif

#if VBT_WITH_CUDA
TEST(AutogradMultiDeviceSingleGpu, OwnerRoutingSameDeviceUsesProducerStream) {
#if !VBT_AUTOGRAD_TESTING
  GTEST_SKIP() << "VBT_AUTOGRAD_TESTING disabled";
#else
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need at least 2 CUDA devices";
  }

  DeviceModeRestore mode_guard;
  MtToggleRestore mt_guard(true);
  StreamingToggleRestore stream_guard(true);
  RouteHookWatchGuard hook_guard;

  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);

  using vbt::cuda::DeviceGuard;
  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  using vbt::cuda::CUDAStreamGuard;

  const auto dev0 = static_cast<DeviceIndex>(0);
  const auto dev1 = static_cast<DeviceIndex>(1);

  Stream S_sink0 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev0);
  Stream S_prod1 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev1);
  Stream S_sink1 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev1);
  ASSERT_NE(S_prod1.id(), 0u);
  ASSERT_NE(S_sink1.id(), 0u);
  ASSERT_NE(S_prod1.id(), S_sink1.id());

  auto sink0 = vbt::core::make_intrusive<CudaAllowlistedSink>(
      Device::cuda(0), S_sink0.id());
  auto sink1 = vbt::core::make_intrusive<CudaAllowlistedSink>(
      Device::cuda(1), S_sink1.id());

  struct CudaProducer1 final : Node {
    CudaProducer1(intrusive_ptr<Node> next, std::uint64_t stream_id) {
      NodeStreamInfo info;
      info.has_canonical_stream = true;
      info.device = Device::cuda(1);
      info.stream_id = stream_id;
      mutable_stream_info() = info;
      name = "CudaProducer1";

      next_edges.resize(1);
      next_edges[0].fn = std::move(next);
      next_edges[0].input_nr = 0;
    }

    uint32_t num_inputs() const noexcept override { return 1; }
    StreamKind stream_kind() const noexcept override { return StreamKind::CudaAllowlisted; }

    std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
      std::vector<OptionalTensor> out(1);
      if (!grads_in.empty()) {
        out[0] = std::move(grads_in[0]);
      }
      return out;
    }
  };

  auto producer1 = vbt::core::make_intrusive<CudaProducer1>(
      intrusive_ptr<Node>(sink1.get()), S_prod1.id());

  struct CpuCopyLikeRootToCuda0AndProducer1 final : Node, CopyLikeNode {
    CpuCopyLikeRootToCuda0AndProducer1(intrusive_ptr<Node> cuda0_sink, intrusive_ptr<Node> prod1) {
      name = "CpuCopyLikeRootToCuda0AndProducer1";
      next_edges.resize(2);
      next_edges[0] = vbt::autograd::Edge{std::move(cuda0_sink), 0};
      next_edges[1] = vbt::autograd::Edge{std::move(prod1), 0};
    }

    uint32_t num_inputs() const noexcept override { return 2; }

    std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
      (void)grads_in;
      std::vector<OptionalTensor> out(2);
      out[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
      out[1] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/1);
      return out;
    }
  };

  auto root = vbt::core::make_intrusive<CpuCopyLikeRootToCuda0AndProducer1>(
      intrusive_ptr<Node>(sink0.get()), intrusive_ptr<Node>(producer1.get()));

  g_route_watch_consumer = sink1.get();
  g_route_watch_device_index = 1;
  g_route_watch_expected_stream_id = S_prod1.id();

  std::vector<OptionalTensor> seed(2);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);
  seed[1] = make_cpu_dense_f32({2}, 1.0f);

  {
    DeviceGuard dg1(dev1);
    CUDAStreamGuard base_guard(vbt::cuda::getDefaultStream(dev1));
    ASSERT_EQ(vbt::cuda::getCurrentStream(dev1).id(), 0u);

    {
      DeviceGuard dg0(dev0);
      ASSERT_NO_THROW(run_backward(intrusive_ptr<Node>(root.get()), seed, {}));
    }

    EXPECT_EQ(vbt::cuda::getCurrentStream(dev1).id(), 0u);
  }
  EXPECT_EQ(g_route_watch_calls, 1u);
#endif
}

TEST(AutogradMultiDeviceSingleGpu, OwnerRoutingCrossDeviceFencedUsesConsumerCanonicalStream) {
#if !VBT_AUTOGRAD_TESTING
  GTEST_SKIP() << "VBT_AUTOGRAD_TESTING disabled";
#else
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need at least 2 CUDA devices";
  }

  DeviceModeRestore mode_guard;
  MtToggleRestore mt_guard(true);
  StreamingToggleRestore stream_guard(true);
  RouteHookWatchGuard hook_guard;

  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);
  vbt::autograd::_test_clear_last_backward_snapshot();

  using vbt::cuda::DeviceGuard;
  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  using vbt::cuda::CUDAStreamGuard;

  const auto dev0 = static_cast<DeviceIndex>(0);
  const auto dev1 = static_cast<DeviceIndex>(1);

  Stream S_base0 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev0);
  Stream S_sink0 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev0);
  ASSERT_NE(S_base0.id(), 0u);
  ASSERT_NE(S_sink0.id(), 0u);
  ASSERT_NE(S_base0.id(), S_sink0.id());

  Stream S_root1 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev1);
  Stream S_prod1 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev1);
  ASSERT_NE(S_prod1.id(), 0u);

  auto sink0 = vbt::core::make_intrusive<CudaAllowlistedSink>(
      Device::cuda(0), S_sink0.id());

  struct CudaCopyLikeProducer1To0 final : Node, CopyLikeNode {
    CudaCopyLikeProducer1To0(intrusive_ptr<Node> next, std::uint64_t stream_id) {
      NodeStreamInfo info;
      info.has_canonical_stream = true;
      info.device = Device::cuda(1);
      info.stream_id = stream_id;
      mutable_stream_info() = info;
      name = "CudaCopyLikeProducer1To0";

      next_edges.resize(1);
      next_edges[0].fn = std::move(next);
      next_edges[0].input_nr = 0;
    }

    uint32_t num_inputs() const noexcept override { return 1; }
    StreamKind stream_kind() const noexcept override { return StreamKind::CudaAllowlisted; }

    std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
      (void)grads_in;
      std::vector<OptionalTensor> out(1);
      out[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
      return out;
    }
  };

  auto producer = vbt::core::make_intrusive<CudaCopyLikeProducer1To0>(
      intrusive_ptr<Node>(sink0.get()), S_prod1.id());

  auto root = vbt::core::make_intrusive<CudaProducerRoot1>(
      Device::cuda(1), S_root1.id(), intrusive_ptr<Node>(producer.get()));

  g_route_watch_consumer = sink0.get();
  g_route_watch_device_index = 0;
  g_route_watch_expected_stream_id = S_sink0.id();

  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/1);

  {
    DeviceGuard dg0(dev0);
    CUDAStreamGuard base_guard(S_base0);
    ASSERT_EQ(vbt::cuda::getCurrentStream(dev0).id(), S_base0.id());

    ASSERT_NO_THROW(run_backward(intrusive_ptr<Node>(root.get()), seed, {}));

    EXPECT_EQ(vbt::cuda::getCurrentStream(dev0).id(), S_base0.id());
  }

  EXPECT_EQ(g_route_watch_calls, 1u);
  EXPECT_EQ(vbt::autograd::_test_last_backward_cuda_device_synchronizes(), 1u);
#endif
}
#endif

TEST(AutogradMultiDeviceSingleGpu, CudaParticipationRequiresMt) {
  DeviceModeRestore mode_guard;
  MtToggleRestore mt_guard(false);
  StreamingToggleRestore stream_guard(true);

  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);

#if VBT_WITH_CUDA
  // Use stream_id=0 (default stream) to avoid requiring a live device.
  auto cuda_sink = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), /*stream_id=*/0);
  auto root = vbt::core::make_intrusive<CpuRoot1>(intrusive_ptr<Node>(cuda_sink.get()));

  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);

  try {
    run_backward(intrusive_ptr<Node>(root.get()), seed, {});
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("MultiDeviceExperimental requires multithreading"), std::string::npos) << msg;
  }
#else
  GTEST_SKIP() << "CUDA build disabled";
#endif
}

TEST(AutogradMultiDeviceSingleGpu, MultipleCudaIndicesInGraphMetadataNoLongerTriggerSingleGpuGuardrail) {
  DeviceModeRestore mode_guard;
  MtToggleRestore mt_guard(true);
  StreamingToggleRestore stream_guard(true);

  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);

#if VBT_WITH_CUDA
  auto sink0 = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), /*stream_id=*/0);
  auto sink1 = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(1), /*stream_id=*/0);

  struct Root final : Node {
    Root(intrusive_ptr<Node> a, intrusive_ptr<Node> b) {
      name = "RootTwoCudaSinks";
      next_edges.resize(2);
      next_edges[0] = vbt::autograd::Edge{std::move(a), 0};
      next_edges[1] = vbt::autograd::Edge{std::move(b), 0};
    }
    uint32_t num_inputs() const noexcept override { return 2; }
    std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
      std::vector<OptionalTensor> out(2);
      if (grads_in.size() > 0) out[0] = std::move(grads_in[0]);
      if (grads_in.size() > 1) out[1] = std::move(grads_in[1]);
      return out;
    }
  };

  auto root = vbt::core::make_intrusive<Root>(
      intrusive_ptr<Node>(sink0.get()), intrusive_ptr<Node>(sink1.get()));

  std::vector<OptionalTensor> seed(2);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);
  seed[1] = make_cpu_dense_f32({2}, 1.0f);

  const int ndev = vbt::cuda::device_count();
  if (ndev < 2) {
    try {
      run_backward(intrusive_ptr<Node>(root.get()), seed, {});
      FAIL() << "expected invalid_argument";
    } catch (const std::invalid_argument& e) {
      const std::string msg(e.what());
      EXPECT_NE(msg.find("invalid CUDA device index"), std::string::npos) << msg;
    }
    return;
  }

  try {
    run_backward(intrusive_ptr<Node>(root.get()), seed, {});
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("explicit copy node is required"), std::string::npos) << msg;
  }
#else
  GTEST_SKIP() << "CUDA build disabled";
#endif
}

#if VBT_WITH_CUDA
TEST(AutogradMultiDeviceSingleGpu, RejectsCpuProducerCudaOutputWithoutCopyLike) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceModeRestore mode_guard;
  MtToggleRestore mt_guard(true);
  StreamingToggleRestore stream_guard(true);

  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);

  using vbt::cuda::DeviceGuard;
  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  DeviceGuard dg(dev_idx);
  Stream cur = vbt::cuda::getCurrentStream(dev_idx);

  auto cuda_sink = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), cur.id());
  auto root = vbt::core::make_intrusive<CpuNonCopyLikeCudaOutRoot1>(intrusive_ptr<Node>(cuda_sink.get()));

  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);

  try {
    run_backward(intrusive_ptr<Node>(root.get()), seed, {});
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("explicit copy node is required"), std::string::npos) << msg;
  }
}

TEST(AutogradMultiDeviceSingleGpu, RejectsCudaGradRoutedToCpuConsumer) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceModeRestore mode_guard;
  MtToggleRestore mt_guard(true);
  StreamingToggleRestore stream_guard(true);

  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);

  using vbt::cuda::DeviceGuard;
  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  DeviceGuard dg(dev_idx);
  Stream cur = vbt::cuda::getCurrentStream(dev_idx);

  auto cpu_sink = vbt::core::make_intrusive<CpuSink>();
  auto root = vbt::core::make_intrusive<CudaProducerRoot1>(
      Device::cuda(0), cur.id(), intrusive_ptr<Node>(cpu_sink.get()));

  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);

  try {
    run_backward(intrusive_ptr<Node>(root.get()), seed, {});
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("explicit copy node is required"), std::string::npos) << msg;
  }
}

TEST(AutogradMultiDeviceSingleGpu, CpuEmptySentinelAcceptedForCudaSlot) {
#if !VBT_AUTOGRAD_TESTING
  GTEST_SKIP() << "VBT_AUTOGRAD_TESTING disabled";
#else
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  DeviceModeRestore mode_guard;
  MtToggleRestore mt_guard(true);
  StreamingToggleRestore stream_guard(true);

  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);
  vbt::autograd::_test_clear_last_backward_snapshot();

  using vbt::cuda::DeviceGuard;
  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  DeviceGuard dg(dev_idx);
  Stream cur = vbt::cuda::getCurrentStream(dev_idx);

  std::vector<InputMeta> metas = {
      InputMeta{ScalarType::Float32, Device::cuda(0), {0}, /*is_strided_dense=*/true}};
  auto cuda_consumer = vbt::core::make_intrusive<CudaAllowlistedValidatableSink>(
      Device::cuda(0), cur.id(), metas);

  struct Root final : Node {
    explicit Root(intrusive_ptr<Node> next) {
      name = "RootSentinel";
      next_edges.resize(1);
      next_edges[0].fn = std::move(next);
      next_edges[0].input_nr = 0;
    }
    uint32_t num_inputs() const noexcept override { return 1; }
    std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
      (void)grads_in;
      std::vector<OptionalTensor> out(1);
      out[0] = make_cpu_dense_f32({0}, 0.0f); // CPU empty sentinel
      return out;
    }
  };

  auto root = vbt::core::make_intrusive<Root>(intrusive_ptr<Node>(cuda_consumer.get()));

  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cpu_dense_f32({2}, 1.0f);

  ASSERT_NO_THROW(run_backward(intrusive_ptr<Node>(root.get()), seed, {}));
  EXPECT_EQ(vbt::autograd::_test_last_backward_cuda_device_synchronizes(), 0u);
#endif
}
#endif
