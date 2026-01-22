// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <stdexcept>

#include "vbt/autograd/node.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/autograd/engine.h"
#include "vbt/autograd/function.h"
#include "vbt/autograd/wrapper.h"
#include "vbt/autograd/meta.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#if VBT_WITH_CUDA
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/device.h"
#include <cuda_runtime_api.h>
#endif

using vbt::autograd::AccumulateGrad;
using vbt::autograd::AutogradMeta;
using vbt::autograd::GraphTask;
using vbt::autograd::Node;
using vbt::autograd::NodeStreamInfo;
using vbt::autograd::OptionalTensor;
using vbt::autograd::StreamKind;
using vbt::autograd::build_inplace_backward_node;
using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::intrusive_ptr;

#if VBT_AUTOGRAD_TESTING
namespace vbt { namespace autograd {
void _test_add_gradient(GraphTask& gt,
                        Node* consumer,
                        std::size_t pos,
                        OptionalTensor&& grad,
                        vbt::core::intrusive_ptr<Node> consumer_keep);

#if VBT_WITH_CUDA
void _test_prepare_consumer_stream_and_wait(GraphTask& gt,
                                           Node& consumer,
                                           GraphTask::InputBuffer* ib,
                                           std::vector<OptionalTensor>& grads_in);

bool _test_prepare_consumer_stream_and_wait_device(const vbt::core::Device& consumer_device,
                                                   Node& consumer,
                                                   GraphTask::InputBuffer* ib,
                                                   std::vector<OptionalTensor>& grads_in,
                                                   std::uint64_t* waited,
                                                   std::uint64_t* cross);

bool _test_prepare_consumer_stream_and_wait_mt_no_deltas(const vbt::core::Device& consumer_device,
                                                         Node& consumer,
                                                         GraphTask::InputBuffer* ib,
                                                         std::vector<OptionalTensor>& grads_in);

void _test_add_gradient_cuda_device(GraphTask& gt,
                                   Node* consumer,
                                   std::size_t pos,
                                   OptionalTensor&& grad,
                                   vbt::core::intrusive_ptr<Node> consumer_keep,
                                   const vbt::core::Device& slot_device);
#endif

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

static TensorImpl make_cpu_empty_numel0_f32() {
  DataPtr dp(nullptr, [](void*) noexcept {});
  StoragePtr st = vbt::core::make_intrusive<Storage>(std::move(dp), /*nbytes=*/0);
  std::vector<int64_t> sizes{0};
  std::vector<int64_t> strides{1};
  return TensorImpl(st, sizes, strides, /*storage_offset=*/0,
                    ScalarType::Float32, Device::cpu());
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

static std::vector<float> cuda_to_host_f32(const TensorImpl& t) {
  if (t.device().type != kDLCUDA) {
    throw std::invalid_argument("cuda_to_host_f32: expected CUDA tensor");
  }
  if (t.dtype() != ScalarType::Float32) {
    throw std::invalid_argument("cuda_to_host_f32: expected Float32 tensor");
  }

  using vbt::cuda::DeviceGuard;
  const auto dev_idx = static_cast<vbt::cuda::DeviceIndex>(t.device().index);
  DeviceGuard dg(dev_idx);

  const std::size_t ne = static_cast<std::size_t>(t.numel());
  std::vector<float> out(ne);
  if (ne == 0) return out;

  cudaError_t st = cudaMemcpy(
      out.data(),
      t.data(),
      ne * sizeof(float),
      cudaMemcpyDeviceToHost);
  if (st != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpy D2H failed in cuda_to_host_f32: ") +
                             cudaGetErrorString(st));
  }
  return out;
}
#endif

struct CpuOnlySink final : Node {
  uint32_t num_inputs() const noexcept override { return 1; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    (void)grads_in;
    return std::vector<OptionalTensor>(1);
  }
};

struct CudaAllowlistedSink final : Node {
  explicit CudaAllowlistedSink(Device dev, std::uint64_t stream_id) {
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
    return std::vector<OptionalTensor>(1);
  }
};

} // namespace

TEST(AutogradStreamsTest, WrapperNodeStreamInfoCpuOnlyByDefault) {
  using vbt::autograd::SavedVariable;

  // Build a simple CPU backward node for vt::add via the public helper.
  TensorImpl a = make_cpu_dense_f32({2}, 1.0f);
  TensorImpl b = make_cpu_dense_f32({2}, 2.0f);

  std::vector<SavedVariable> snaps;
  snaps.emplace_back(SavedVariable(a));
  snaps.emplace_back(SavedVariable(b));

  auto node = build_inplace_backward_node("vt::add", snaps);

  EXPECT_EQ(node->stream_kind(), StreamKind::CpuOnly);
  const NodeStreamInfo& si = node->stream_info();
  EXPECT_FALSE(si.has_canonical_stream);
}

#if VBT_WITH_CUDA
TEST(AutogradStreamsTest, WrapperNodeStreamInfoCudaAllowlisted) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  using vbt::autograd::SavedVariable;

  TensorImpl a = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
  TensorImpl b = make_cuda_dense_f32({2}, 2.0f, /*dev=*/0);

  std::vector<SavedVariable> snaps;
  snaps.emplace_back(SavedVariable(a));
  snaps.emplace_back(SavedVariable(b));

  auto node = build_inplace_backward_node("vt::add", snaps);

  EXPECT_EQ(node->stream_kind(), StreamKind::CudaAllowlisted);
  const NodeStreamInfo& si = node->stream_info();
  EXPECT_TRUE(si.has_canonical_stream);
  EXPECT_EQ(si.device.type, kDLCUDA);
  EXPECT_EQ(si.device.index, 0);

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  Stream current = vbt::cuda::getCurrentStream(dev_idx);
  EXPECT_EQ(si.stream_id, current.id());
}
#endif

TEST(AutogradInputBufferCudaMetaTest, IB1FreshBufferInvariants) {
  GraphTask::InputBuffer ib;
  const std::size_t n = 3;

  ib.ensure_cpu_capacity(n);
  ib.ensure_cuda_capacity(n);

  EXPECT_EQ(ib.expected, n);
  EXPECT_EQ(ib.received, 0u);
  EXPECT_FALSE(ib.enqueued);

  ASSERT_EQ(ib.grads_in.size(), n);
  ASSERT_EQ(ib.present.size(), n);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_FALSE(ib.grads_in[i].has_value());
    EXPECT_EQ(ib.present[i], 0u);
  }

  ASSERT_EQ(ib.is_accel.size(), n);
  ASSERT_EQ(ib.has_accum_stream.size(), n);
  ASSERT_EQ(ib.accum_device.size(), n);
  ASSERT_EQ(ib.accum_stream_id.size(), n);

  ASSERT_EQ(ib.has_ready_event.size(), n);
  ASSERT_EQ(ib.ready_device.size(), n);
  ASSERT_EQ(ib.ready_stream_id.size(), n);

  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(ib.is_accel[i], 0u);
    EXPECT_EQ(ib.has_accum_stream[i], 0u);
    EXPECT_EQ(ib.accum_device[i], Device{});
    EXPECT_EQ(ib.accum_stream_id[i], 0u);

    EXPECT_EQ(ib.has_ready_event[i], 0u);
    EXPECT_EQ(ib.ready_device[i], Device{});
    EXPECT_EQ(ib.ready_stream_id[i], 0u);
  }

#if VBT_WITH_CUDA
  ASSERT_EQ(ib.ready_events.size(), n);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_FALSE(ib.ready_events[i].is_created());
  }
#endif
}

#if VBT_WITH_CUDA && VBT_AUTOGRAD_TESTING
TEST(AutogradInputBufferCudaMetaTest, IB2NulloptArrivalsDoNotTouchCudaMetadata) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  gt.autograd_device = Device::cuda(0);
  gt.has_autograd_device = true;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  Stream S_acc = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);

  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), S_acc.id());

  // Route a nullopt grad into slot 0.
  vbt::autograd::_test_add_gradient(
      gt,
      consumer.get(),
      /*pos=*/0,
      OptionalTensor{},
      intrusive_ptr<Node>(consumer.get()));

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  const GraphTask::InputBuffer& ib = it->second;

  ASSERT_EQ(ib.expected, 1u);
  ASSERT_EQ(ib.present.size(), 1u);
  EXPECT_EQ(ib.present[0], 1u);
  EXPECT_EQ(ib.received, 1u);

  ASSERT_EQ(ib.grads_in.size(), 1u);
  EXPECT_FALSE(ib.grads_in[0].has_value());

  ASSERT_EQ(ib.is_accel.size(), 1u);
  EXPECT_EQ(ib.is_accel[0], 0u);
  EXPECT_EQ(ib.has_accum_stream[0], 0u);
  EXPECT_EQ(ib.has_ready_event[0], 0u);
  EXPECT_FALSE(ib.ready_events[0].is_created());

  EXPECT_EQ(gt.cuda_events_recorded, 0u);
  EXPECT_EQ(gt.cuda_events_waited, 0u);
  EXPECT_EQ(gt.cuda_cross_stream_routes, 0u);
}

TEST(AutogradInputBufferCudaMetaTest, IB3FirstCudaArrivalSetsPerSlotMetadata) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  gt.autograd_device = Device::cuda(0);
  gt.has_autograd_device = true;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  Stream S_acc = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);

  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), S_acc.id());
  TensorImpl grad = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);

  // Make producer stream match accumulation stream.
  {
    vbt::cuda::CUDAStreamGuard sg(S_acc);
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(grad),
        intrusive_ptr<Node>(consumer.get()));
  }

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  const GraphTask::InputBuffer& ib = it->second;

  ASSERT_EQ(ib.is_accel.size(), 1u);
  EXPECT_EQ(ib.is_accel[0], 1u);
  EXPECT_EQ(ib.has_accum_stream[0], 1u);
  EXPECT_EQ(ib.accum_device[0], Device::cuda(0));
  EXPECT_EQ(ib.accum_stream_id[0], S_acc.id());

  EXPECT_EQ(ib.has_ready_event[0], 1u);
  EXPECT_EQ(ib.ready_device[0], Device::cuda(0));
  EXPECT_EQ(ib.ready_stream_id[0], S_acc.id());
  EXPECT_TRUE(ib.ready_events[0].is_created());

  // Same-stream first arrival records only the ready event.
  EXPECT_GE(gt.cuda_events_recorded, 1u);
  EXPECT_EQ(gt.cuda_events_waited, 0u);
  EXPECT_EQ(gt.cuda_cross_stream_routes, 0u);
  EXPECT_LE(gt.cuda_events_waited, gt.cuda_events_recorded);
}

TEST(AutogradInputBufferCudaMetaTest, IB4DuplicateCudaArrivalAcrossStreamsUpdatesStatsAndMetadata) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  gt.autograd_device = Device::cuda(0);
  gt.has_autograd_device = true;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  Stream S_acc  = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  Stream S_prod = vbt::cuda::getStreamFromPool(/*high_priority=*/true, dev_idx);
  ASSERT_NE(S_acc.id(), S_prod.id());

  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), S_acc.id());

  TensorImpl g1 = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
  TensorImpl g2 = make_cuda_dense_f32({2}, 2.0f, /*dev=*/0);

  // First arrival on accumulation stream (no bridge event).
  {
    vbt::cuda::CUDAStreamGuard sg(S_acc);
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g1),
        intrusive_ptr<Node>(consumer.get()));
  }

  // Duplicate arrival on a different producer stream.
  {
    vbt::cuda::CUDAStreamGuard sg(S_prod);
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g2),
        intrusive_ptr<Node>(consumer.get()));
  }

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  const GraphTask::InputBuffer& ib = it->second;

  ASSERT_EQ(ib.is_accel.size(), 1u);
  EXPECT_EQ(ib.is_accel[0], 1u);
  EXPECT_EQ(ib.has_accum_stream[0], 1u);
  EXPECT_EQ(ib.accum_device[0], Device::cuda(0));
  EXPECT_EQ(ib.accum_stream_id[0], S_acc.id());

  EXPECT_EQ(ib.has_ready_event[0], 1u);
  EXPECT_EQ(ib.ready_device[0], Device::cuda(0));
  EXPECT_EQ(ib.ready_stream_id[0], S_acc.id());

  // First arrival: 1 recorded (ready event).
  // Duplicate arrival on different stream: 1 recorded (bridge) + 1 recorded (new ready).
  EXPECT_GE(gt.cuda_events_recorded, 3u);
  EXPECT_GE(gt.cuda_events_waited, 1u);
  EXPECT_GE(gt.cuda_cross_stream_routes, 1u);
  EXPECT_EQ(gt.duplicates_coalesced, 1u);
  EXPECT_LE(gt.cuda_events_waited, gt.cuda_events_recorded);

  // Accumulation result: slot == g1 + g2.
  ASSERT_EQ(ib.grads_in.size(), 1u);
  ASSERT_TRUE(ib.grads_in[0].has_value());

  S_acc.synchronize();
  std::vector<float> host = cuda_to_host_f32(ib.grads_in[0].value());
  ASSERT_EQ(host.size(), 2u);
  EXPECT_FLOAT_EQ(host[0], 3.0f);
  EXPECT_FLOAT_EQ(host[1], 3.0f);
}

TEST(AutogradWaitHardeningTest, ConsumerPrepRejectsCrossDeviceReadyEventDevice) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  gt.autograd_device = Device::cuda(0);
  gt.has_autograd_device = true;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);

  Stream S_cons  = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  Stream S_other = vbt::cuda::getStreamFromPool(/*high_priority=*/true, dev_idx);
  ASSERT_NE(S_cons.id(), S_other.id());

  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), S_cons.id());
  TensorImpl g1 = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);

  {
    vbt::cuda::CUDAStreamGuard sg(S_cons);
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g1),
        intrusive_ptr<Node>(consumer.get()));
  }

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  GraphTask::InputBuffer& ib = it->second;

  ASSERT_EQ(ib.expected, 1u);
  ASSERT_EQ(ib.has_ready_event.size(), 1u);
  ASSERT_EQ(ib.ready_device.size(), 1u);
  ASSERT_EQ(ib.ready_stream_id.size(), 1u);

  // Corrupt the metadata so the consumer prep path attempts a cross-device wait.
  ib.ready_stream_id[0] = S_other.id();
  ib.ready_device[0] = Device::cuda(1);

  std::vector<OptionalTensor> grads_in = ib.grads_in;
  try {
    vbt::autograd::_test_prepare_consumer_stream_and_wait(gt, *consumer, &ib, grads_in);
    FAIL() << "Expected exception";
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()),
              "VibeTensor CUDA autograd internal error: attempted cross-device CUDA event wait");
  }
}

TEST(AutogradWaitHardeningTest, AddGradientDuplicateRejectsCrossDeviceReadyEventDevice) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  gt.autograd_device = Device::cuda(0);
  gt.has_autograd_device = true;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);

  Stream S_acc   = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  Stream S_other = vbt::cuda::getStreamFromPool(/*high_priority=*/true, dev_idx);
  ASSERT_NE(S_acc.id(), S_other.id());

  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), S_acc.id());
  TensorImpl g1 = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
  TensorImpl g2 = make_cuda_dense_f32({2}, 2.0f, /*dev=*/0);

  {
    vbt::cuda::CUDAStreamGuard sg(S_acc);
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g1),
        intrusive_ptr<Node>(consumer.get()));
  }

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  GraphTask::InputBuffer& ib = it->second;

  // Corrupt the metadata so the duplicate-add path attempts a cross-device wait.
  ib.ready_stream_id[0] = S_other.id();
  ib.ready_device[0] = Device::cuda(1);

  {
    vbt::cuda::CUDAStreamGuard sg(S_acc);
    try {
      vbt::autograd::_test_add_gradient(
          gt,
          consumer.get(),
          /*pos=*/0,
          OptionalTensor(g2),
          intrusive_ptr<Node>(consumer.get()));
      FAIL() << "Expected exception";
    } catch (const std::runtime_error& e) {
      EXPECT_EQ(std::string(e.what()),
                "VibeTensor CUDA autograd internal error: attempted cross-device CUDA event wait");
    }
  }
}

TEST(AutogradAddGradientSequencesTest, AG2SingleCudaArrivalSameStreamRecordsOnlyReadyEvent) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  gt.autograd_device = Device::cuda(0);
  gt.has_autograd_device = true;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);

  Stream S_acc = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), S_acc.id());

  TensorImpl g1 = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);

  {
    vbt::cuda::CUDAStreamGuard sg(S_acc);
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g1),
        intrusive_ptr<Node>(consumer.get()));
  }

  EXPECT_EQ(gt.cuda_events_recorded, 1u);  // ready event only
  EXPECT_EQ(gt.cuda_events_waited, 0u);
  EXPECT_EQ(gt.cuda_cross_stream_routes, 0u);
  EXPECT_EQ(gt.duplicates_coalesced, 0u);

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  const GraphTask::InputBuffer& ib = it->second;
  ASSERT_EQ(ib.expected, 1u);
  EXPECT_TRUE(ib.grads_in[0].has_value());
  EXPECT_EQ(ib.accum_stream_id[0], S_acc.id());
  EXPECT_EQ(ib.ready_stream_id[0], S_acc.id());
}

TEST(AutogradAddGradientSequencesTest, AG3SingleCudaArrivalDifferentStreamRecordsBridgeAndReady) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  gt.autograd_device = Device::cuda(0);
  gt.has_autograd_device = true;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);

  Stream S_acc  = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  Stream S_prod = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  ASSERT_NE(S_acc.id(), S_prod.id());

  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), S_acc.id());
  TensorImpl g1 = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);

  {
    vbt::cuda::CUDAStreamGuard sg(S_prod);
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g1),
        intrusive_ptr<Node>(consumer.get()));
  }

  EXPECT_EQ(gt.cuda_events_recorded, 2u);  // bridge + ready
  EXPECT_EQ(gt.cuda_events_waited, 1u);
  EXPECT_EQ(gt.cuda_cross_stream_routes, 1u);
  EXPECT_EQ(gt.duplicates_coalesced, 0u);

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  const GraphTask::InputBuffer& ib = it->second;
  ASSERT_EQ(ib.expected, 1u);
  EXPECT_TRUE(ib.grads_in[0].has_value());
  EXPECT_EQ(ib.accum_stream_id[0], S_acc.id());
  EXPECT_EQ(ib.ready_stream_id[0], S_acc.id());
}

TEST(AutogradAddGradientSequencesTest, AG4MultipleCudaArrivalsAcrossStreamsAccumulatesAndUpdatesStats) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  gt.autograd_device = Device::cuda(0);
  gt.has_autograd_device = true;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);

  Stream S_acc   = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  Stream S_prod1 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  Stream S_prod2 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  ASSERT_NE(S_acc.id(), S_prod1.id());
  ASSERT_NE(S_acc.id(), S_prod2.id());
  ASSERT_NE(S_prod1.id(), S_prod2.id());

  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), S_acc.id());

  TensorImpl g1 = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
  TensorImpl g2 = make_cuda_dense_f32({2}, 2.0f, /*dev=*/0);
  TensorImpl g3 = make_cuda_dense_f32({2}, 3.0f, /*dev=*/0);

  // First arrival on S_prod1.
  {
    vbt::cuda::CUDAStreamGuard sg(S_prod1);
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g1),
        intrusive_ptr<Node>(consumer.get()));
  }

  // Duplicate arrival on S_prod2.
  {
    vbt::cuda::CUDAStreamGuard sg(S_prod2);
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g2),
        intrusive_ptr<Node>(consumer.get()));
  }

  // Third arrival on the accumulation stream (no bridge).
  {
    vbt::cuda::CUDAStreamGuard sg(S_acc);
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g3),
        intrusive_ptr<Node>(consumer.get()));
  }

  // 3 arrivals => 3 ready-event records.
  // 2 non-accum producer streams => 2 bridge records + 2 waits.
  EXPECT_EQ(gt.cuda_events_recorded, 5u);
  EXPECT_EQ(gt.cuda_events_waited, 2u);
  EXPECT_EQ(gt.cuda_cross_stream_routes, 2u);
  EXPECT_EQ(gt.duplicates_coalesced, 2u);

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  const GraphTask::InputBuffer& ib = it->second;
  ASSERT_EQ(ib.grads_in.size(), 1u);
  ASSERT_TRUE(ib.grads_in[0].has_value());

  S_acc.synchronize();
  std::vector<float> host = cuda_to_host_f32(ib.grads_in[0].value());
  ASSERT_EQ(host.size(), 2u);
  EXPECT_FLOAT_EQ(host[0], 6.0f);
  EXPECT_FLOAT_EQ(host[1], 6.0f);
}

TEST(AutogradAddGradientSequencesTest, AG5NulloptThenCudaArrivalPreservesReceivedAndSetsCudaState) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  gt.autograd_device = Device::cuda(0);
  gt.has_autograd_device = true;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);

  Stream S_acc = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), S_acc.id());

  // First arrival is nullopt.
  vbt::autograd::_test_add_gradient(
      gt,
      consumer.get(),
      /*pos=*/0,
      OptionalTensor{},
      intrusive_ptr<Node>(consumer.get()));

  {
    auto it = gt.inputs.find(consumer.get());
    ASSERT_NE(it, gt.inputs.end());
    const GraphTask::InputBuffer& ib = it->second;
    EXPECT_EQ(ib.received, 1u);
    EXPECT_TRUE(ib.present[0]);
    EXPECT_FALSE(ib.grads_in[0].has_value());
  }

  // Then a defined CUDA grad arrives.
  TensorImpl g1 = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
  {
    vbt::cuda::CUDAStreamGuard sg(S_acc);
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g1),
        intrusive_ptr<Node>(consumer.get()));
  }

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  const GraphTask::InputBuffer& ib = it->second;
  EXPECT_EQ(ib.received, 1u) << "received should count only first arrival";

  EXPECT_EQ(gt.cuda_events_recorded, 1u);
  EXPECT_EQ(gt.cuda_events_waited, 0u);
  EXPECT_EQ(gt.cuda_cross_stream_routes, 0u);

  EXPECT_EQ(ib.is_accel[0], 1u);
  EXPECT_EQ(ib.has_accum_stream[0], 1u);
  EXPECT_EQ(ib.accum_stream_id[0], S_acc.id());
  EXPECT_EQ(ib.has_ready_event[0], 1u);
  EXPECT_TRUE(ib.ready_events[0].is_created());
}

TEST(AutogradAddGradientSequencesTest, AG6CudaArrivalAfterBufferDrainedThrowsSizeMismatch) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  gt.autograd_device = Device::cuda(0);
  gt.has_autograd_device = true;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);

  Stream S_acc = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), S_acc.id());

  TensorImpl g1 = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
  {
    vbt::cuda::CUDAStreamGuard sg(S_acc);
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g1),
        intrusive_ptr<Node>(consumer.get()));
  }

  // Simulate the consumer being popped and its grads_in moved out.
  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  it->second.grads_in.clear();

  TensorImpl g2 = make_cuda_dense_f32({2}, 2.0f, /*dev=*/0);
  try {
    vbt::cuda::CUDAStreamGuard sg(S_acc);
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g2),
        intrusive_ptr<Node>(consumer.get()));
    FAIL() << "expected logic_error";
  } catch (const std::logic_error& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("InputBuffer size mismatch"), std::string::npos) << msg;
  }
}

TEST(AutogradAddGradientSequencesTest, STATS1EventCounterInvariantsHold) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  gt.autograd_device = Device::cuda(0);
  gt.has_autograd_device = true;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);

  Stream S_acc   = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  Stream S_prod1 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);
  ASSERT_NE(S_acc.id(), S_prod1.id());

  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), S_acc.id());

  TensorImpl g1 = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
  {
    vbt::cuda::CUDAStreamGuard sg(S_prod1);
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g1),
        intrusive_ptr<Node>(consumer.get()));
  }

  EXPECT_LE(gt.cuda_events_waited, gt.cuda_events_recorded);
  EXPECT_LE(gt.cuda_cross_stream_routes, gt.cuda_events_waited);
}

TEST(AutogradInputBufferCudaMetaTest, IB5RootSeedingWithSentinelLeavesCudaMetadataFresh) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  using vbt::autograd::FunctionNode;
  using vbt::autograd::InputMeta;

  GraphTask gt;

  std::vector<InputMeta> metas;
  metas.push_back(InputMeta{ScalarType::Float32, Device::cuda(0), /*sizes=*/{2}});
  metas.push_back(InputMeta{ScalarType::Float32, Device::cuda(0), /*sizes=*/{2}});
  metas.push_back(InputMeta{ScalarType::Float32, Device::cuda(0), /*sizes=*/{2}});

  auto root = vbt::core::make_intrusive<FunctionNode>(
      "RootFn",
      metas,
      [](std::vector<OptionalTensor>&& grads_in) {
        (void)grads_in;
        return std::vector<OptionalTensor>(3);
      });

  std::vector<OptionalTensor> seed(3);
  seed[0] = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
  seed[1] = OptionalTensor{};              // true nullopt
  seed[2] = make_cpu_empty_numel0_f32();   // CPU-empty CUDA sentinel

  vbt::autograd::_test_seed_root_buffer(gt, intrusive_ptr<Node>(root.get()), seed);

  EXPECT_TRUE(gt.has_autograd_device);
  EXPECT_EQ(gt.autograd_device, Device::cuda(0));

  auto it = gt.inputs.find(root.get());
  ASSERT_NE(it, gt.inputs.end());
  const GraphTask::InputBuffer& ib = it->second;

  ASSERT_EQ(ib.expected, 3u);
  ASSERT_EQ(ib.present.size(), 3u);
  EXPECT_EQ(ib.received, 3u);
  EXPECT_TRUE(ib.enqueued);

  EXPECT_EQ(ib.present[0], 1u);
  EXPECT_EQ(ib.present[1], 1u);
  EXPECT_EQ(ib.present[2], 1u);

  ASSERT_EQ(ib.grads_in.size(), 3u);
  EXPECT_TRUE(ib.grads_in[0].has_value());
  EXPECT_FALSE(ib.grads_in[1].has_value());
  EXPECT_FALSE(ib.grads_in[2].has_value());

  ASSERT_EQ(ib.is_accel.size(), 3u);
  for (std::size_t i = 0; i < 3u; ++i) {
    EXPECT_EQ(ib.is_accel[i], 0u);
    EXPECT_EQ(ib.has_accum_stream[i], 0u);
    EXPECT_EQ(ib.has_ready_event[i], 0u);
    EXPECT_FALSE(ib.ready_events[i].is_created());
  }

  EXPECT_EQ(gt.cuda_events_recorded, 0u);
  EXPECT_EQ(gt.cuda_events_waited, 0u);
  EXPECT_EQ(gt.cuda_cross_stream_routes, 0u);
}
#endif  // VBT_WITH_CUDA && VBT_AUTOGRAD_TESTING

#if VBT_WITH_CUDA
#if VBT_AUTOGRAD_TESTING
TEST(AutogradStreamsTest, AddGradientCudaRejectsCpuOnlyConsumer) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  gt.autograd_device = Device::cuda(0);
  gt.has_autograd_device = true;

  auto consumer = vbt::core::make_intrusive<CpuOnlySink>();
  TensorImpl grad = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);

  try {
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(grad),
        intrusive_ptr<Node>(consumer.get()));
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("CpuOnly node received CUDA gradient"), std::string::npos)
        << msg;
  }
}

TEST(AutogradStreamsTest, AddGradientCudaRejectsCanonicalStreamDeviceMismatch) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  gt.autograd_device = Device::cuda(0);
  gt.has_autograd_device = true;

  // Mark the consumer as CUDA-allowlisted, but claim its canonical stream is on cuda:1.
  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(1), /*stream_id=*/0);
  TensorImpl grad = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);

  try {
    vbt::autograd::_test_add_gradient(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(grad),
        intrusive_ptr<Node>(consumer.get()));
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()),
              "VibeTensor CUDA autograd internal error: missing or mismatched canonical stream on CUDA node");
  }
}

TEST(AutogradDeviceParametricHelpersTest, ConsumerCoreCanonicalMismatchUsesExplicitDevice) {
  std::uint64_t waited = 0;
  std::uint64_t cross = 0;

  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), /*stream_id=*/0);
  std::vector<OptionalTensor> grads_in;

  try {
    (void)vbt::autograd::_test_prepare_consumer_stream_and_wait_device(
        Device::cuda(1), *consumer, /*ib=*/nullptr, grads_in, &waited, &cross);
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()),
              "VibeTensor CUDA autograd internal error: missing or mismatched canonical stream on CUDA node");
  }

  EXPECT_EQ(waited, 0u);
  EXPECT_EQ(cross, 0u);
}

TEST(AutogradDeviceParametricHelpersTest, ConsumerCoreCrossDeviceWaitUsesExplicitDeviceAndPreservesCounters) {
  std::uint64_t waited = 0;
  std::uint64_t cross = 0;

  GraphTask::InputBuffer ib;
  ib.ensure_cpu_capacity(1);
  // ensure_cuda_capacity is runtime-free (sizes vectors, creates lazy Events),
  // so this test does not require a CUDA device.
  ib.ensure_cuda_capacity(1);

  ib.has_ready_event[0] = 1;
  ib.ready_stream_id[0] = 123u;
  ib.ready_device[0] = Device::cuda(1);

  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), /*stream_id=*/0);
  std::vector<OptionalTensor> grads_in;

  try {
    (void)vbt::autograd::_test_prepare_consumer_stream_and_wait_device(
        Device::cuda(0), *consumer, &ib, grads_in, &waited, &cross);
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()),
              "VibeTensor CUDA autograd internal error: attempted cross-device CUDA event wait");
  }

  EXPECT_EQ(waited, 0u);
  EXPECT_EQ(cross, 0u);
}

TEST(AutogradDeviceParametricHelpersTest, ConsumerCoreNullInputBufferNonAllowlistedReturnsNullopt) {
  std::uint64_t waited = 0;
  std::uint64_t cross = 0;

  auto consumer = vbt::core::make_intrusive<CpuOnlySink>();
  std::vector<OptionalTensor> grads_in;

  bool has_guard = vbt::autograd::_test_prepare_consumer_stream_and_wait_device(
      Device::cuda(0), *consumer, /*ib=*/nullptr, grads_in, &waited, &cross);

  EXPECT_FALSE(has_guard);
  EXPECT_EQ(waited, 0u);
  EXPECT_EQ(cross, 0u);
}

TEST(AutogradDeviceParametricHelpersTest, ConsumerCoreCpuOnlyAccelSlotCheckIsGatedByVectorSize) {
  std::uint64_t waited = 0;
  std::uint64_t cross = 0;

  auto consumer = vbt::core::make_intrusive<CpuOnlySink>();
  std::vector<OptionalTensor> grads_in;

  // Size mismatch: should not throw.
  GraphTask::InputBuffer ib_mismatch;
  ib_mismatch.ensure_cpu_capacity(1);
  bool has_guard1 = vbt::autograd::_test_prepare_consumer_stream_and_wait_device(
      Device::cuda(0), *consumer, &ib_mismatch, grads_in, &waited, &cross);
  EXPECT_FALSE(has_guard1);

  // Sized + accelerated: should throw.
  GraphTask::InputBuffer ib_sized;
  ib_sized.ensure_cpu_capacity(1);
  // ensure_cuda_capacity is runtime-free (sizes vectors, creates lazy Events).
  ib_sized.ensure_cuda_capacity(1);
  ib_sized.is_accel[0] = 1;

  try {
    (void)vbt::autograd::_test_prepare_consumer_stream_and_wait_device(
        Device::cuda(0), *consumer, &ib_sized, grads_in, &waited, &cross);
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()),
              "VibeTensor CUDA autograd internal error: CpuOnly consumer has accelerated slots");
  }

  EXPECT_EQ(waited, 0u);
  EXPECT_EQ(cross, 0u);
}

TEST(AutogradDeviceParametricHelpersTest, ConsumerCoreSkipsWaitLoopWhenReadyEventVectorsNotSized) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  Stream S_cons = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);

  std::uint64_t waited = 0;
  std::uint64_t cross = 0;

  GraphTask::InputBuffer ib;
  ib.ensure_cpu_capacity(1);  // no ensure_cuda_capacity -> has_ready_event.size() != expected

  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), S_cons.id());
  std::vector<OptionalTensor> grads_in;

  bool has_guard = vbt::autograd::_test_prepare_consumer_stream_and_wait_device(
      Device::cuda(0), *consumer, &ib, grads_in, &waited, &cross);

  EXPECT_TRUE(has_guard);
  EXPECT_EQ(waited, 0u);
  EXPECT_EQ(cross, 0u);
}

TEST(AutogradDeviceParametricHelpersTest, ConsumerMtWrapperAllowsDeltasNull) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  Stream S_cons = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);

  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), S_cons.id());
  std::vector<OptionalTensor> grads_in;

  bool has_guard = vbt::autograd::_test_prepare_consumer_stream_and_wait_mt_no_deltas(
      Device::cuda(0), *consumer, /*ib=*/nullptr, grads_in);

  EXPECT_TRUE(has_guard);
}

TEST(AutogradDeviceParametricHelpersTest, ProducerCoreUsesSlotDeviceNotGraphTaskAutogradDevice) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  GraphTask gt;
  gt.autograd_device = Device::cuda(static_cast<int>(vbt::cuda::device_count())); // poison
  gt.has_autograd_device = true;

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  Stream S_acc = vbt::cuda::getStreamFromPool(/*high_priority=*/false, dev_idx);

  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), S_acc.id());
  TensorImpl g1 = make_cuda_dense_f32({2}, 1.0f, /*dev=*/0);
  TensorImpl g2 = make_cuda_dense_f32({2}, 2.0f, /*dev=*/0);

  {
    vbt::cuda::CUDAStreamGuard sg(S_acc);
    vbt::autograd::_test_add_gradient_cuda_device(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g1),
        intrusive_ptr<Node>(consumer.get()),
        Device::cuda(0));
  }
  {
    vbt::cuda::CUDAStreamGuard sg(S_acc);
    vbt::autograd::_test_add_gradient_cuda_device(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(g2),
        intrusive_ptr<Node>(consumer.get()),
        Device::cuda(0));
  }

  auto it = gt.inputs.find(consumer.get());
  ASSERT_NE(it, gt.inputs.end());
  GraphTask::InputBuffer& ib = it->second;
  ASSERT_EQ(ib.expected, 1u);
  ASSERT_TRUE(ib.grads_in[0].has_value());

  S_acc.synchronize();
  std::vector<float> host = cuda_to_host_f32(ib.grads_in[0].value());
  ASSERT_EQ(host.size(), 2u);
  EXPECT_FLOAT_EQ(host[0], 3.0f);
  EXPECT_FLOAT_EQ(host[1], 3.0f);
}

TEST(AutogradDeviceParametricHelpersTest, ProducerCoreRejectsCrossDeviceAccumulation) {
  GraphTask gt;

  auto consumer = vbt::core::make_intrusive<CudaAllowlistedSink>(Device::cuda(0), /*stream_id=*/0);

  GraphTask::InputBuffer buf;
  buf.ensure_cpu_capacity(1);
  buf.ensure_cuda_capacity(1);

  buf.has_accum_stream[0] = 1;
  buf.accum_device[0] = Device::cuda(1);

  gt.inputs.emplace(consumer.get(), std::move(buf));

  // Defined grad required to reach the producer core (it will throw before using it).
  TensorImpl grad = make_cpu_dense_f32({2}, 1.0f);

  try {
    vbt::autograd::_test_add_gradient_cuda_device(
        gt,
        consumer.get(),
        /*pos=*/0,
        OptionalTensor(grad),
        intrusive_ptr<Node>(consumer.get()),
        Device::cuda(0));
    FAIL() << "expected runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()),
              "VibeTensor CUDA autograd internal error: attempted cross-device CUDA accumulation");
  }
}

TEST(AutogradStreamsTest, CpuEmptyCudaSentinelDerivesCudaAutogradDevice) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  using vbt::autograd::FunctionNode;
  using vbt::autograd::InputMeta;

  GraphTask gt;

  std::vector<InputMeta> metas;
  metas.push_back(InputMeta{ScalarType::Float32, Device::cuda(0), /*sizes=*/{2}});

  auto root = vbt::core::make_intrusive<FunctionNode>(
      "RootFn",
      metas,
      [](std::vector<OptionalTensor>&& grads_in) {
        (void)grads_in;
        return std::vector<OptionalTensor>(1);
      });

  std::vector<OptionalTensor> seed(1);
  seed[0] = make_cpu_empty_numel0_f32();

  vbt::autograd::_test_seed_root_buffer(gt, intrusive_ptr<Node>(root.get()), seed);

  EXPECT_TRUE(gt.has_autograd_device);
  EXPECT_EQ(gt.autograd_device.type, kDLCUDA);
  EXPECT_EQ(gt.autograd_device.index, 0);

  auto it = gt.inputs.find(root.get());
  ASSERT_NE(it, gt.inputs.end());
  ASSERT_EQ(it->second.grads_in.size(), 1u);
  ASSERT_EQ(it->second.present.size(), 1u);
  EXPECT_TRUE(it->second.present[0]);
  EXPECT_FALSE(it->second.grads_in[0].has_value());
}
#endif  // VBT_AUTOGRAD_TESTING
#endif  // VBT_WITH_CUDA

#if VBT_WITH_AUTOGRAD
TEST(AutogradStreamsTest, AccumulateGradCpuStreamKindCpuOnly) {
  TensorImpl leaf = make_cpu_dense_f32({2}, 0.0f);
  AutogradMeta* meta = vbt::autograd::get_autograd_meta(leaf, /*create_if_missing=*/true);
  ASSERT_NE(meta, nullptr);

  AccumulateGrad acc(meta);
  EXPECT_EQ(acc.stream_kind(), StreamKind::CpuOnly);
  const NodeStreamInfo& si = acc.stream_info();
  EXPECT_FALSE(si.has_canonical_stream);
}

#if VBT_WITH_CUDA
TEST(AutogradStreamsTest, AccumulateGradCudaTaggedHasCanonicalStream) {
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  TensorImpl leaf = make_cuda_dense_f32({2}, 0.0f, /*dev=*/0);
  AutogradMeta* meta = vbt::autograd::get_autograd_meta(leaf, /*create_if_missing=*/true);
  ASSERT_NE(meta, nullptr);

  AccumulateGrad acc(meta);
  _tag_accumulategrad_cuda_leaf(acc, leaf);

  EXPECT_EQ(acc.stream_kind(), StreamKind::CudaAllowlisted);
  const NodeStreamInfo& si = acc.stream_info();
  EXPECT_TRUE(si.has_canonical_stream);
  EXPECT_EQ(si.device.type, kDLCUDA);
  EXPECT_EQ(si.device.index, 0);

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(0);
  Stream current = vbt::cuda::getCurrentStream(dev_idx);
  EXPECT_EQ(si.stream_id, current.id());
}
#endif
#endif  // VBT_WITH_AUTOGRAD
