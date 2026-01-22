// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "vbt/cuda/device.h"
#include "vbt/cuda/fabric_addmul_decision.h"
#include "vbt/cuda/fabric_state.h"
#include "vbt/cuda/fabric_topology.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/graphs.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/event.h"
#include "vbt/cpu/storage.h"

#if VBT_WITH_AUTOGRAD
#include "vbt/autograd/forward.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/wrapper.h"
#endif

using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::TensorImpl;
using vbt::cuda::DeviceGuard;
using vbt::cuda::CUDAStreamGuard;
using vbt::cuda::Stream;
using vbt::cuda::device_count;
using vbt::cuda::fabric::FabricMode;
using vbt::cuda::fabric::FabricState;
using vbt::cuda::fabric::FabricTopology;
using vbt::cuda::fabric::FabricAddMulDecision;
using vbt::cuda::fabric::FabricAddMulFallbackReason;
using vbt::cuda::fabric::decide_fabric_addmul_2gpu;
using vbt::cuda::fabric::fabric_state;

#if defined(VBT_INTERNAL_TESTS)
using vbt::cuda::fabric::fabric_test_hooks;
using vbt::cuda::fabric::reset_fabric_state_for_tests;
#endif

namespace {

std::vector<std::int64_t> contiguous_strides(const std::vector<std::int64_t>& sizes) {
  if (sizes.empty()) return {};
  std::vector<std::int64_t> strides(sizes.size());
  std::int64_t acc = 1;
  for (std::size_t i = sizes.size(); i-- > 0;) {
    strides[i] = acc;
    acc *= (sizes[i] == 0 ? 1 : sizes[i]);
  }
  return strides;
}

TensorImpl make_cuda_tensor(int dev,
                           const std::vector<std::int64_t>& sizes,
                           ScalarType dtype) {
  const std::vector<std::int64_t> strides = contiguous_strides(sizes);

  std::int64_t n = 1;
  if (sizes.empty()) {
    n = 1;
  } else {
    for (auto s : sizes) {
      if (s == 0) {
        n = 0;
        break;
      }
      n *= s;
    }
  }

  const std::size_t nbytes = static_cast<std::size_t>(n) * vbt::core::itemsize(dtype);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  return TensorImpl(std::move(storage), sizes, strides, /*storage_offset=*/0,
                    dtype, Device::cuda(dev));
}

TensorImpl make_cpu_tensor(const std::vector<std::int64_t>& sizes, ScalarType dtype) {
  const std::vector<std::int64_t> strides = contiguous_strides(sizes);

  std::int64_t n = 1;
  if (sizes.empty()) {
    n = 1;
  } else {
    for (auto s : sizes) {
      if (s == 0) {
        n = 0;
        break;
      }
      n *= s;
    }
  }

  const std::size_t nbytes = static_cast<std::size_t>(n) * vbt::core::itemsize(dtype);
  auto storage = vbt::cpu::new_cpu_storage(nbytes, /*pinned=*/false);
  return TensorImpl(std::move(storage), sizes, strides, /*storage_offset=*/0,
                    dtype, Device::cpu(0));
}

#if defined(VBT_INTERNAL_TESTS)
FabricState& init_fabric_state_with_topology(
    int device_count,
    std::vector<int> clique_id,
    std::vector<int> clique_size,
    FabricMode mode,
    std::optional<bool> forced_uva_ok = true) {
  auto& hooks = fabric_test_hooks();
  hooks.forced_uva_ok = forced_uva_ok;
  hooks.fake_topology_builder = [device_count,
                                clique_id = std::move(clique_id),
                                clique_size = std::move(clique_size)](FabricTopology& topo) {
    topo.device_count = device_count;
    topo.can_access_peer.assign(device_count,
                                std::vector<bool>(device_count, false));
    topo.p2p_enabled.assign(device_count,
                            std::vector<bool>(device_count, false));
    topo.clique_id = clique_id;
    topo.clique_size = clique_size;
  };

  reset_fabric_state_for_tests();
  FabricState& fs = fabric_state();
  fs.config.mode.store(mode, std::memory_order_relaxed);
  return fs;
}
#endif

}  // namespace

TEST(FabricDecideTest, InvalidComputeDeviceOutOfRange) {
#if !defined(VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "requires VBT_INTERNAL_TESTS";
#else
  (void)init_fabric_state_with_topology(
      /*device_count=*/2,
      /*clique_id=*/{0, 0},
      /*clique_size=*/{2},
      FabricMode::BestEffort);

  TensorImpl a = make_cuda_tensor(/*dev=*/0, /*sizes=*/{4}, ScalarType::Float32);
  TensorImpl b = make_cuda_tensor(/*dev=*/0, /*sizes=*/{4}, ScalarType::Float32);

  FabricAddMulDecision dec = decide_fabric_addmul_2gpu(
      /*compute_device=*/99, a, b,
      /*require_fabric=*/true,
      /*use_copy_fallback=*/true);

  EXPECT_EQ(dec.reason, FabricAddMulFallbackReason::kInvalidComputeDevice);
  EXPECT_FALSE(dec.use_fabric);
  EXPECT_FALSE(dec.use_copy_fallback);
  EXPECT_EQ(dec.primary_device, -1);
  EXPECT_EQ(dec.other_device, -1);
  EXPECT_EQ(dec.numel, 0);
#endif
}

TEST(FabricDecideTest, NotCudaIsArgumentError) {
#if !defined(VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "requires VBT_INTERNAL_TESTS";
#else
  (void)init_fabric_state_with_topology(
      /*device_count=*/1,
      /*clique_id=*/{0},
      /*clique_size=*/{1},
      FabricMode::BestEffort);

  TensorImpl a = make_cpu_tensor(/*sizes=*/{4}, ScalarType::Float32);
  TensorImpl b = make_cuda_tensor(/*dev=*/0, /*sizes=*/{4}, ScalarType::Float32);

  FabricAddMulDecision dec = decide_fabric_addmul_2gpu(
      /*compute_device=*/0, a, b,
      /*require_fabric=*/false,
      /*use_copy_fallback=*/true);

  EXPECT_EQ(dec.reason, FabricAddMulFallbackReason::kNotCuda);
  EXPECT_FALSE(dec.use_fabric);
  EXPECT_FALSE(dec.use_copy_fallback);
#endif
}

TEST(FabricDecideTest, InvalidShapesOrDtypesIsArgumentError) {
#if !defined(VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "requires VBT_INTERNAL_TESTS";
#else
  (void)init_fabric_state_with_topology(
      /*device_count=*/1,
      /*clique_id=*/{0},
      /*clique_size=*/{1},
      FabricMode::BestEffort);

  TensorImpl a = make_cuda_tensor(/*dev=*/0, /*sizes=*/{4}, ScalarType::Float32);
  TensorImpl b = make_cuda_tensor(/*dev=*/0, /*sizes=*/{5}, ScalarType::Float32);

  FabricAddMulDecision dec = decide_fabric_addmul_2gpu(
      /*compute_device=*/0, a, b,
      /*require_fabric=*/false,
      /*use_copy_fallback=*/true);

  EXPECT_EQ(dec.reason, FabricAddMulFallbackReason::kInvalidShapesOrDtypes);
  EXPECT_FALSE(dec.use_fabric);
  EXPECT_FALSE(dec.use_copy_fallback);
#endif
}

TEST(FabricDecideTest, FabricDisabledAllowsCopyFallbackWhenPermitted) {
#if !defined(VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "requires VBT_INTERNAL_TESTS";
#else
  (void)init_fabric_state_with_topology(
      /*device_count=*/2,
      /*clique_id=*/{0, 0},
      /*clique_size=*/{2},
      FabricMode::Disabled);

  if (device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 real CUDA devices for mixed-device tensors";
  }

  TensorImpl a = make_cuda_tensor(/*dev=*/0, /*sizes=*/{4}, ScalarType::Float32);
  TensorImpl b = make_cuda_tensor(/*dev=*/1, /*sizes=*/{4}, ScalarType::Float32);

#if VBT_WITH_AUTOGRAD
  vbt::autograd::NoGradGuard ng;
#endif

  FabricAddMulDecision dec = decide_fabric_addmul_2gpu(
      /*compute_device=*/0, a, b,
      /*require_fabric=*/false,
      /*use_copy_fallback=*/true);

  EXPECT_EQ(dec.reason, FabricAddMulFallbackReason::kFabricGloballyDisabled);
  EXPECT_FALSE(dec.use_fabric);
  EXPECT_TRUE(dec.use_copy_fallback);
  EXPECT_EQ(dec.primary_device, 0);
  EXPECT_EQ(dec.other_device, 1);
#endif
}

TEST(FabricDecideTest, NotInSameCliqueAllowsCopyFallbackWhenPermitted) {
#if !defined(VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "requires VBT_INTERNAL_TESTS";
#else
  // Note: clique_size values are intentionally >=2 to keep the global gate open,
  // while clique_id assigns devices to different cliques to make the pair unusable.
  (void)init_fabric_state_with_topology(
      /*device_count=*/2,
      /*clique_id=*/{0, 1},
      /*clique_size=*/{2, 2},
      FabricMode::BestEffort);

  if (device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 real CUDA devices for mixed-device tensors";
  }

  TensorImpl a = make_cuda_tensor(/*dev=*/0, /*sizes=*/{4}, ScalarType::Float32);
  TensorImpl b = make_cuda_tensor(/*dev=*/1, /*sizes=*/{4}, ScalarType::Float32);

#if VBT_WITH_AUTOGRAD
  vbt::autograd::NoGradGuard ng;
#endif

  FabricAddMulDecision dec = decide_fabric_addmul_2gpu(
      /*compute_device=*/0, a, b,
      /*require_fabric=*/false,
      /*use_copy_fallback=*/true);

  EXPECT_EQ(dec.reason, FabricAddMulFallbackReason::kNotInSameCliqueOrNoP2P);
  EXPECT_FALSE(dec.use_fabric);
  EXPECT_TRUE(dec.use_copy_fallback);
#endif
}

TEST(FabricDecideTest, SuccessRequiresGateOpenAndNoGrad) {
#if !defined(VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "requires VBT_INTERNAL_TESTS";
#else
  (void)init_fabric_state_with_topology(
      /*device_count=*/2,
      /*clique_id=*/{0, 0},
      /*clique_size=*/{2},
      FabricMode::BestEffort);

  if (device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 real CUDA devices for mixed-device tensors";
  }

  TensorImpl a = make_cuda_tensor(/*dev=*/0, /*sizes=*/{4}, ScalarType::Float32);
  TensorImpl b = make_cuda_tensor(/*dev=*/1, /*sizes=*/{4}, ScalarType::Float32);

#if VBT_WITH_AUTOGRAD
  vbt::autograd::NoGradGuard ng;
#endif

  FabricAddMulDecision dec = decide_fabric_addmul_2gpu(
      /*compute_device=*/0, a, b,
      /*require_fabric=*/true,
      /*use_copy_fallback=*/true);

  EXPECT_EQ(dec.reason, FabricAddMulFallbackReason::kNone);
  EXPECT_TRUE(dec.use_fabric);
  EXPECT_FALSE(dec.use_copy_fallback);
  EXPECT_EQ(dec.primary_device, 0);
  EXPECT_EQ(dec.other_device, 1);
  EXPECT_EQ(dec.numel, 4);
#endif
}

TEST(FabricDecideTest, RequiresGradTriggersCopyFallbackWhenPermitted) {
#if !defined(VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "requires VBT_INTERNAL_TESTS";
#elif !VBT_WITH_AUTOGRAD
  GTEST_SKIP() << "requires VBT_WITH_AUTOGRAD";
#else
  (void)init_fabric_state_with_topology(
      /*device_count=*/2,
      /*clique_id=*/{0, 0},
      /*clique_size=*/{2},
      FabricMode::BestEffort);

  if (device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 real CUDA devices for mixed-device tensors";
  }

  vbt::autograd::NoGradGuard ng;

  TensorImpl a = make_cuda_tensor(/*dev=*/0, /*sizes=*/{4}, ScalarType::Float32);
  TensorImpl b = make_cuda_tensor(/*dev=*/1, /*sizes=*/{4}, ScalarType::Float32);
  vbt::autograd::set_requires_grad(a, true);

  FabricAddMulDecision dec = decide_fabric_addmul_2gpu(
      /*compute_device=*/0, a, b,
      /*require_fabric=*/false,
      /*use_copy_fallback=*/true);

  EXPECT_EQ(dec.reason, FabricAddMulFallbackReason::kRequiresGrad);
  EXPECT_FALSE(dec.use_fabric);
  EXPECT_TRUE(dec.use_copy_fallback);
#endif
}

TEST(FabricDecideTest, InBackwardTriggersCopyFallbackWhenPermitted) {
#if !defined(VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "requires VBT_INTERNAL_TESTS";
#elif !VBT_WITH_AUTOGRAD
  GTEST_SKIP() << "requires VBT_WITH_AUTOGRAD";
#else
  (void)init_fabric_state_with_topology(
      /*device_count=*/2,
      /*clique_id=*/{0, 0},
      /*clique_size=*/{2},
      FabricMode::BestEffort);

  if (device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 real CUDA devices for mixed-device tensors";
  }

  vbt::autograd::NoGradGuard ng;
  vbt::autograd::BackwardGuard bg;

  TensorImpl a = make_cuda_tensor(/*dev=*/0, /*sizes=*/{4}, ScalarType::Float32);
  TensorImpl b = make_cuda_tensor(/*dev=*/1, /*sizes=*/{4}, ScalarType::Float32);

  FabricAddMulDecision dec = decide_fabric_addmul_2gpu(
      /*compute_device=*/0, a, b,
      /*require_fabric=*/false,
      /*use_copy_fallback=*/true);

  EXPECT_EQ(dec.reason, FabricAddMulFallbackReason::kInBackward);
  EXPECT_FALSE(dec.use_fabric);
  EXPECT_TRUE(dec.use_copy_fallback);
#endif
}

TEST(FabricDecideTest, GraphCaptureActiveIsHardError) {
#if !defined(VBT_INTERNAL_TESTS)
  GTEST_SKIP() << "requires VBT_INTERNAL_TESTS";
#else
  (void)init_fabric_state_with_topology(
      /*device_count=*/2,
      /*clique_id=*/{0, 0},
      /*clique_size=*/{2},
      FabricMode::BestEffort);

  if (device_count() < 2) {
    GTEST_SKIP() << "Need >= 2 real CUDA devices for mixed-device tensors";
  }

#if VBT_WITH_AUTOGRAD
  vbt::autograd::NoGradGuard ng;
#endif

  TensorImpl a = make_cuda_tensor(/*dev=*/0, /*sizes=*/{4}, ScalarType::Float32);
  TensorImpl b = make_cuda_tensor(/*dev=*/1, /*sizes=*/{4}, ScalarType::Float32);

  DeviceGuard dg(0);
  Stream s = vbt::cuda::getStreamFromPool(/*priority=*/0, /*device=*/0);
  CUDAStreamGuard sg(s);

  // Create a captureable CUDA op so capture_end has something to end.
  vbt::cuda::Event ev;

  vbt::cuda::CUDAGraph g;
  g.capture_begin(s);
  ev.record(s);

  FabricAddMulDecision dec = decide_fabric_addmul_2gpu(
      /*compute_device=*/0, a, b,
      /*require_fabric=*/false,
      /*use_copy_fallback=*/true);

  EXPECT_EQ(dec.reason, FabricAddMulFallbackReason::kGraphCaptureActive);
  EXPECT_FALSE(dec.use_fabric);
  EXPECT_FALSE(dec.use_copy_fallback);

  g.capture_end();
#endif
}
