// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
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
#include "vbt/cuda/fabric_lifetime.h"
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

#endif

TEST(FabricLifetime, PlanBuildPrimaryAndRemoteWithProducers) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA not built";
#else
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 CUDA devices";
  }

  vbt::cuda::debug_clear_producer_metadata_for_testing();

  auto s0 = vbt::cuda::new_cuda_storage(/*nbytes=*/1024, /*device_index=*/0);
  auto s1 = vbt::cuda::new_cuda_storage(/*nbytes=*/1024, /*device_index=*/1);

  auto p0 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/0);
  auto p1 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/1);

  vbt::cuda::record_stream(s0, p0);
  vbt::cuda::record_stream(s1, p1);

  vbt::cuda::fabric::FabricStorageSets sets;
  sets.primary_storages = {s0};
  sets.remote_storages  = {s1};

  auto r = vbt::cuda::fabric::build_primary_remote_fence_plan(
      sets,
      /*primary_device=*/0,
      /*remote_device=*/1);

  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.plan.primary_device, 0);
  EXPECT_EQ(r.plan.remote_device, 1);
  EXPECT_EQ(r.plan.primary_producers.size(), 1u);
  EXPECT_EQ(r.plan.remote_producers.size(), 1u);
  EXPECT_EQ(r.plan.primary_producers[0].device, 0);
  EXPECT_EQ(r.plan.remote_producers[0].device, 1);
#endif
}

TEST(FabricLifetime, PlanDeduplicatesSharedProducerStreams) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA not built";
#else
  if (vbt::cuda::device_count() < 1) {
    GTEST_SKIP() << "Need >= 1 CUDA device";
  }

  vbt::cuda::debug_clear_producer_metadata_for_testing();

  auto s0a = vbt::cuda::new_cuda_storage(/*nbytes=*/1024, /*device_index=*/0);
  auto s0b = vbt::cuda::new_cuda_storage(/*nbytes=*/2048, /*device_index=*/0);

  auto p0 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/0);
  vbt::cuda::record_stream(s0a, p0);
  vbt::cuda::record_stream(s0b, p0);

  vbt::cuda::fabric::FabricStorageSets sets;
  sets.primary_storages = {s0a, s0b};

  auto r = vbt::cuda::fabric::build_primary_remote_fence_plan(
      sets,
      /*primary_device=*/0,
      /*remote_device=*/-1);

  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.plan.primary_producers.size(), 1u);
  EXPECT_EQ(r.plan.primary_producers[0].device, 0);
#endif
}

TEST(FabricLifetime, PlanMetadataMissingFailsClosed) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA not built";
#else
  vbt::cuda::debug_clear_producer_metadata_for_testing();

  void* buf = ::operator new(1024);
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto st = vbt::core::make_intrusive<Storage>(std::move(dp), 1024);

  vbt::cuda::fabric::FabricStorageSets sets;
  sets.primary_storages = {st};

  auto r = vbt::cuda::fabric::build_primary_remote_fence_plan(
      sets,
      /*primary_device=*/0,
      /*remote_device=*/-1);

  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.kind, vbt::cuda::fabric::FabricFailureKind::kMetadataFailure);
  ASSERT_TRUE(r.error.has_value());
  EXPECT_TRUE(r.error->is_metadata_failure);
#endif
}

TEST(FabricLifetime, ExecuteZeroProducersIsNoop) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA not built";
#else
  if (vbt::cuda::device_count() < 1) {
    GTEST_SKIP() << "Need >= 1 CUDA device";
  }

  vbt::cuda::fabric::debug_reset_fabric_fence_counters_for_testing();

  vbt::cuda::fabric::PrimaryRemoteFencePlan plan;
  plan.primary_device = 0;
  plan.remote_device  = -1;

  auto Sp = vbt::cuda::fabric::get_fabric_compute_stream(/*primary_device=*/0);
  auto r = vbt::cuda::fabric::execute_primary_remote_fence_plan(plan, Sp);

  ASSERT_TRUE(r.ok());
  auto c = vbt::cuda::fabric::debug_get_fabric_fence_counters_for_testing();
  EXPECT_EQ(c.num_event_record_calls, 0u);
  EXPECT_EQ(c.num_stream_wait_calls, 0u);
  EXPECT_EQ(c.num_producer_events_primary, 0u);
  EXPECT_EQ(c.num_producer_events_remote, 0u);
#endif
}

TEST(FabricLifetime, ExecuteSuccessPrimaryAndRemote) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA not built";
#else
  if (vbt::cuda::device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 CUDA devices";
  }

  vbt::cuda::debug_clear_producer_metadata_for_testing();
  vbt::cuda::fabric::debug_reset_fabric_fence_counters_for_testing();

  auto s0 = vbt::cuda::new_cuda_storage(/*nbytes=*/1024, /*device_index=*/0);
  auto s1 = vbt::cuda::new_cuda_storage(/*nbytes=*/1024, /*device_index=*/1);

  auto p0 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/0);
  auto p1 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/1);

  vbt::cuda::record_stream(s0, p0);
  vbt::cuda::record_stream(s1, p1);

  vbt::cuda::fabric::FabricStorageSets sets;
  sets.primary_storages = {s0};
  sets.remote_storages  = {s1};

  auto plan_r = vbt::cuda::fabric::build_primary_remote_fence_plan(sets, 0, 1);
  ASSERT_TRUE(plan_r.ok());

  auto Sp = vbt::cuda::fabric::get_fabric_compute_stream(/*primary_device=*/0);
  auto exec_r = vbt::cuda::fabric::execute_primary_remote_fence_plan(plan_r.plan, Sp);
  ASSERT_TRUE(exec_r.ok());

  auto c = vbt::cuda::fabric::debug_get_fabric_fence_counters_for_testing();
  EXPECT_EQ(c.num_event_record_calls, 2u);
  EXPECT_EQ(c.num_stream_wait_calls, 2u);
  EXPECT_EQ(c.num_producer_events_primary, 1u);
  EXPECT_EQ(c.num_producer_events_remote, 1u);
#endif
}

TEST(FabricLifetime, FabricAddUsesFencingWhenEnabled) {
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

  auto& fs = vbt::cuda::fabric::fabric_state();
  if (fs.init_status != vbt::cuda::fabric::FabricInitStatus::Ok || !fs.uva_ok) {
    GTEST_SKIP() << "UVA gate is disabled";
  }
  fs.config.mode.store(vbt::cuda::fabric::FabricMode::BestEffort, std::memory_order_release);

  vbt::autograd::NoGradGuard ng;

  const bool prev_lifetime = vbt::cuda::fabric::is_fabric_event_lifetime_enabled();
  struct LifetimeGuard {
    bool prev;
    ~LifetimeGuard() { vbt::cuda::fabric::set_fabric_event_lifetime_enabled(prev); }
  } lifetime_guard{prev_lifetime};

  vbt::cuda::fabric::set_fabric_event_lifetime_enabled(true);

  vbt::cuda::debug_clear_producer_metadata_for_testing();
  vbt::cuda::fabric::debug_reset_fabric_fence_counters_for_testing();

  TensorImpl a0 = make_cuda_tensor_f32(/*dev=*/0, {1.f, 2.f, 3.f, 4.f});
  TensorImpl b1 = make_cuda_tensor_f32(/*dev=*/1, {10.f, -2.f, 0.5f, 1.f});

  // the metadata producer set, not the current stream.
  auto prod1 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/1);
  vbt::cuda::record_stream(b1.storage(), prod1);

  TensorImpl compute0 = make_cpu_scalar_i64(0);
  TensorImpl require1 = make_cpu_scalar_i64(1);
  TensorImpl fallback1 = make_cpu_scalar_i64(1);

  vbt::dispatch::BoxedStack s{a0, b1, compute0, require1, fallback1};
  vbt::dispatch::Dispatcher::instance().callBoxed("vt::fabric_add", s);
  ASSERT_EQ(s.size(), 1u);

  auto c = vbt::cuda::fabric::debug_get_fabric_fence_counters_for_testing();
  EXPECT_EQ(c.num_producer_events_remote, 1u);
  EXPECT_EQ(c.num_event_record_calls, 1u);
  EXPECT_EQ(c.num_stream_wait_calls, 1u);
#endif
}

TEST(FabricLifetime, FabricAddRecordFailureFallsBackWhenAllowed) {
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

  auto& fs = vbt::cuda::fabric::fabric_state();
  if (fs.init_status != vbt::cuda::fabric::FabricInitStatus::Ok || !fs.uva_ok) {
    GTEST_SKIP() << "UVA gate is disabled";
  }
  fs.config.mode.store(vbt::cuda::fabric::FabricMode::BestEffort, std::memory_order_release);

  vbt::autograd::NoGradGuard ng;

  const bool prev_lifetime = vbt::cuda::fabric::is_fabric_event_lifetime_enabled();
  struct LifetimeGuard {
    bool prev;
    ~LifetimeGuard() { vbt::cuda::fabric::set_fabric_event_lifetime_enabled(prev); }
  } lifetime_guard{prev_lifetime};

  vbt::cuda::fabric::set_fabric_event_lifetime_enabled(true);

  vbt::cuda::debug_clear_producer_metadata_for_testing();
  vbt::cuda::fabric::debug_reset_fabric_fence_counters_for_testing();

  TensorImpl a0 = make_cuda_tensor_f32(/*dev=*/0, {1.f, 2.f, 3.f, 4.f});
  TensorImpl b1 = make_cuda_tensor_f32(/*dev=*/1, {10.f, -2.f, 0.5f, 1.f});

  auto prod1 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/1);
  vbt::cuda::record_stream(b1.storage(), prod1);

  vbt::cuda::fabric::debug_fail_fence_event_record_on_n_for_testing(1);

  TensorImpl compute0 = make_cpu_scalar_i64(0);
  TensorImpl require0 = make_cpu_scalar_i64(0);
  TensorImpl fallback1 = make_cpu_scalar_i64(1);

  vbt::dispatch::BoxedStack s{a0, b1, compute0, require0, fallback1};
  vbt::dispatch::Dispatcher::instance().callBoxed("vt::fabric_add", s);
  ASSERT_EQ(s.size(), 1u);

  auto c = vbt::cuda::fabric::debug_get_fabric_fence_counters_for_testing();
  EXPECT_EQ(c.num_event_record_calls, 0u);
  EXPECT_EQ(c.num_stream_wait_calls, 0u);

  // Clear hooks for any later tests in this binary.
  vbt::cuda::fabric::debug_fail_fence_event_record_on_n_for_testing(0);
#endif
}

TEST(FabricLifetime, FabricAddRecordFailureThrowsWhenRequired) {
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

  auto& fs = vbt::cuda::fabric::fabric_state();
  if (fs.init_status != vbt::cuda::fabric::FabricInitStatus::Ok || !fs.uva_ok) {
    GTEST_SKIP() << "UVA gate is disabled";
  }
  fs.config.mode.store(vbt::cuda::fabric::FabricMode::BestEffort, std::memory_order_release);

  vbt::autograd::NoGradGuard ng;

  const bool prev_lifetime = vbt::cuda::fabric::is_fabric_event_lifetime_enabled();
  struct LifetimeGuard {
    bool prev;
    ~LifetimeGuard() { vbt::cuda::fabric::set_fabric_event_lifetime_enabled(prev); }
  } lifetime_guard{prev_lifetime};

  vbt::cuda::fabric::set_fabric_event_lifetime_enabled(true);

  vbt::cuda::debug_clear_producer_metadata_for_testing();
  vbt::cuda::fabric::debug_reset_fabric_fence_counters_for_testing();

  TensorImpl a0 = make_cuda_tensor_f32(/*dev=*/0, {1.f, 2.f, 3.f, 4.f});
  TensorImpl b1 = make_cuda_tensor_f32(/*dev=*/1, {10.f, -2.f, 0.5f, 1.f});

  auto prod1 = vbt::cuda::getStreamFromPool(/*high_priority=*/false, /*device=*/1);
  vbt::cuda::record_stream(b1.storage(), prod1);

  vbt::cuda::fabric::debug_fail_fence_event_record_on_n_for_testing(1);

  TensorImpl compute0 = make_cpu_scalar_i64(0);
  TensorImpl require1 = make_cpu_scalar_i64(1);
  TensorImpl fallback1 = make_cpu_scalar_i64(1);

  try {
    vbt::dispatch::BoxedStack s{a0, b1, compute0, require1, fallback1};
    vbt::dispatch::Dispatcher::instance().callBoxed("vt::fabric_add", s);
    FAIL() << "expected Fabric error";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("injected cudaEventRecord failure"), std::string::npos);
  }

  vbt::cuda::fabric::debug_fail_fence_event_record_on_n_for_testing(0);
#endif
}
