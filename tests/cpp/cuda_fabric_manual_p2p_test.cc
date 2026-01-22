// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//

#include <gtest/gtest.h>

#if VBT_WITH_CUDA

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vbt/cuda/allocator.h"      // Allocator, enablePeerAccess, record_stream
#include "vbt/cuda/device.h"         // device_count
#include "vbt/cuda/event.h"          // Event
#include "vbt/cuda/guard.h"          // DeviceGuard, CUDAStreamGuard
#include "vbt/cuda/stream.h"         // Stream, getStreamFromPool
#include "vbt/cuda/fabric_state.h"   // FabricState, FabricInitStatus, FabricTopology, fabric_state
#include "vbt/cuda/fabric_topology.h" // FabricTopology, in_same_fabric

#include <cuda_runtime_api.h>

using vbt::cuda::Allocator;
using vbt::cuda::DeviceIndex;
using vbt::cuda::Stream;
using vbt::cuda::device_count;
using vbt::cuda::fabric::FabricInitStatus;
using vbt::cuda::fabric::FabricState;
using vbt::cuda::fabric::FabricTopology;
using vbt::cuda::fabric::fabric_state;
using vbt::cuda::fabric::in_same_fabric;

namespace vbt { namespace cuda { namespace fabric { namespace testonly {

// -----------------------------------------------------------------------------
// ManualP2P prereq classifier and helper APIs
// -----------------------------------------------------------------------------

enum class ManualP2PPrereqResult : std::uint8_t {
  kOk,        // All prereqs satisfied; tests may run.
  kSkip,      // Environment gap; tests must GTEST_SKIP.
  kHardFail,  // Suspected regression or programming error; tests must FAIL.
};

struct ManualP2PPrereqInfo {
  ManualP2PPrereqResult result{ManualP2PPrereqResult::kOk};
  std::string           message;  // Non-empty for kSkip / kHardFail.
};

// Peer-access classification helpers ------------------------------------------------

enum class PeerAccessOutcome {
  kOk,
  kCapabilityUnsupported,  // environment limitation (no peer access)
  kUnexpectedError,        // non-capability CUDA error (regression)
};

struct PeerAccessResult {
  PeerAccessOutcome outcome{PeerAccessOutcome::kOk};
  cudaError_t       st01{cudaSuccess};  // raw enablePeerAccess(0,1) status
  cudaError_t       st10{cudaSuccess};  // raw enablePeerAccess(1,0) status
};

inline const char* cuda_error_str(cudaError_t st) noexcept {
#if VBT_WITH_CUDA
  const char* s = cudaGetErrorString(st);
  return s ? s : "unknown";
#else
  (void)st;
  return "no_cuda";
#endif
}

inline bool is_capability_error(cudaError_t st) noexcept {
#if VBT_WITH_CUDA
  switch (st) {
    case cudaErrorPeerAccessUnsupported:
    case cudaErrorNotSupported:
    case cudaErrorNotPermitted:
      return true;  // capability / policy limitation -> Skip.
    default:
      return false;
  }
#else
  (void)st;
  return false;
#endif
}

// Treat "already enabled" as success so the helper is idempotent even when
// another test or production code has already enabled peer access.
inline cudaError_t normalize_peer_access_status(cudaError_t st) noexcept {
#if VBT_WITH_CUDA
  if (st == cudaErrorPeerAccessAlreadyEnabled) {
    (void)cudaGetLastError();
    return cudaSuccess;
  }
#endif
  return st;
}

inline PeerAccessResult enable_peer_access_or_classify(int dev0, int dev1) noexcept {
  PeerAccessResult res;

  res.st01 = Allocator::enablePeerAccess(dev0, dev1);
  res.st10 = Allocator::enablePeerAccess(dev1, dev0);

  cudaError_t st01 = normalize_peer_access_status(res.st01);
  cudaError_t st10 = normalize_peer_access_status(res.st10);

  if (st01 == cudaSuccess && st10 == cudaSuccess) {
    res.outcome = PeerAccessOutcome::kOk;
    return res;
  }

  if (is_capability_error(st01) || is_capability_error(st10)) {
    res.outcome = PeerAccessOutcome::kCapabilityUnsupported;
    return res;
  }

  res.outcome = PeerAccessOutcome::kUnexpectedError;
  return res;
}

// Ordered decision procedure S1–S8 ----------------------------------------------------

[[nodiscard]] ManualP2PPrereqResult
check_manual_p2p_prereqs(ManualP2PPrereqInfo* info) {
  // Caller must provide a valid info pointer.
  constexpr int kDev0 = 0;
  constexpr int kDev1 = 1;

  info->result = ManualP2PPrereqResult::kOk;
  info->message.clear();

  // S1: runtime device count fast path.
  int dc = device_count();
  if (dc < 2) {
    info->result  = ManualP2PPrereqResult::kSkip;
    info->message =
        "Fabric prereq: Need >= 2 CUDA devices (device_count=" +
        std::to_string(dc) + ")";
    return info->result;
  }

  // S2: topology shape and device-count invariants.
  FabricState& fs            = fabric_state();
  const FabricTopology& topo = fs.topology;

  if (topo.device_count < 2) {
    info->result  = ManualP2PPrereqResult::kHardFail;
    info->message =
        "Fabric prereq: FabricTopology.device_count < 2 despite"
        " runtime device_count >= 2 (suspected Fabric regression)";
    return info->result;
  }

  if (topo.device_count != dc) {
    info->result  = ManualP2PPrereqResult::kHardFail;
    info->message =
        "Fabric prereq: FabricTopology.device_count (" +
        std::to_string(topo.device_count) + ") disagrees with runtime"
        " device_count() (" + std::to_string(dc) + ")";
    return info->result;
  }

  if (topo.clique_id.size() != static_cast<std::size_t>(topo.device_count) ||
      topo.can_access_peer.size() != static_cast<std::size_t>(topo.device_count) ||
      topo.p2p_enabled.size() != static_cast<std::size_t>(topo.device_count) ||
      topo.clique_size.empty()) {
    info->result  = ManualP2PPrereqResult::kHardFail;
    info->message =
        "Fabric prereq: FabricTopology invariant violated (shape)";
    return info->result;
  }

  for (const auto& row : topo.can_access_peer) {
    if (row.size() != static_cast<std::size_t>(topo.device_count)) {
      info->result  = ManualP2PPrereqResult::kHardFail;
      info->message =
          "Fabric prereq: FabricTopology.can_access_peer row size mismatch";
      return info->result;
    }
  }

  for (const auto& row : topo.p2p_enabled) {
    if (row.size() != static_cast<std::size_t>(topo.device_count)) {
      info->result  = ManualP2PPrereqResult::kHardFail;
      info->message =
          "Fabric prereq: FabricTopology.p2p_enabled row size mismatch";
      return info->result;
    }
  }

  for (int d = 0; d < topo.device_count; ++d) {
    int cid = topo.clique_id[d];
    if (cid < 0 || cid >= static_cast<int>(topo.clique_size.size())) {
      info->result  = ManualP2PPrereqResult::kHardFail;
      info->message =
          "Fabric prereq: FabricTopology.clique_id out of range";
      return info->result;
    }
  }

  for (int c = 0; c < static_cast<int>(topo.clique_size.size()); ++c) {
    if (topo.clique_size[c] < 1) {
      info->result  = ManualP2PPrereqResult::kHardFail;
      info->message =
          "Fabric prereq: FabricTopology.clique_size entry < 1";
      return info->result;
    }
  }

  // S3: UVA gate status.
  if (fs.init_status != FabricInitStatus::Ok || !fs.uva_ok) {
    info->result = ManualP2PPrereqResult::kSkip;
    info->message =
        "Fabric prereq: UVA gate disabled (init_status=" +
        std::to_string(static_cast<int>(fs.init_status)) +
        ", uva_ok=" + (fs.uva_ok ? "true" : "false") +
        ", disable_reason=\"" + fs.disable_reason + "\")";
    return info->result;
  }

  // S4 / S5: clique and static P2P capability for devices 0 and 1.
  int cid0 = topo.clique_id[kDev0];
  int cid1 = topo.clique_id[kDev1];

  if (!in_same_fabric(kDev0, kDev1, topo) || topo.clique_size[cid0] < 2) {
    info->result  = ManualP2PPrereqResult::kSkip;
    info->message =
        "Fabric prereq: devices 0 and 1 are not in a clique of size >= 2";
    return info->result;
  }

  if (!topo.can_access_peer[kDev0][kDev1] ||
      !topo.can_access_peer[kDev1][kDev0]) {
    info->result  = ManualP2PPrereqResult::kSkip;
    info->message =
        "Fabric prereq: P2P unsupported between devices 0 and 1";
    return info->result;
  }

  // S6–S8: allocator-level peer-access classification.
  PeerAccessResult per = enable_peer_access_or_classify(kDev0, kDev1);
  switch (per.outcome) {
    case PeerAccessOutcome::kOk:
      info->result  = ManualP2PPrereqResult::kOk;
      info->message.clear();
      return info->result;  // S7 (happy path)

    case PeerAccessOutcome::kCapabilityUnsupported:
      info->result  = ManualP2PPrereqResult::kSkip;
      info->message =
          "Fabric prereq: peer access not supported between devices 0 and 1; "
          "dev0->1 status=" + std::to_string(static_cast<int>(per.st01)) +
          " (" + cuda_error_str(per.st01) + "), dev1->0 status=" +
          std::to_string(static_cast<int>(per.st10)) + " (" +
          cuda_error_str(per.st10) + ")";
      return info->result;  // S6

    case PeerAccessOutcome::kUnexpectedError:
    default:
      info->result  = ManualP2PPrereqResult::kHardFail;
      info->message =
          "Fabric prereq: unexpected CUDA error enabling peer access; "
          "dev0->1 status=" + std::to_string(static_cast<int>(per.st01)) +
          " (" + cuda_error_str(per.st01) + "), dev1->0 status=" +
          std::to_string(static_cast<int>(per.st10)) + " (" +
          cuda_error_str(per.st10) + ")";
      return info->result;  // S8
  }
}

// Optional compatibility wrapper that preserves earlier docs which referenced
// check_manual_p2p_prereqs_or_skip(). New tests should prefer the explicit
// switch pattern on ManualP2PPrereqResult.
inline void check_manual_p2p_prereqs_or_skip() {
  ManualP2PPrereqInfo info;
  auto res = check_manual_p2p_prereqs(&info);
  switch (res) {
    case ManualP2PPrereqResult::kOk:
      return;
    case ManualP2PPrereqResult::kSkip:
      GTEST_SKIP() << info.message;
      return;
    case ManualP2PPrereqResult::kHardFail:
    default:
      FAIL() << info.message;
      return;
  }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

struct ManualP2PConfig {
  std::size_t numel{0};
  float       dst_base{1000.0f};
  float       src_base{1.0f};
};

struct ManualP2PEnv final {
  DeviceIndex dev0{0};
  DeviceIndex dev1{1};

  Stream stream0;       // dev0 consumer stream.
  Stream stream1_fill;  // dev1 producer (H2D copies).
  Stream stream1_proxy; // dev1 proxy (waits on dev0 event).

  Allocator* alloc0{nullptr};
  Allocator* alloc1{nullptr};

  ManualP2PEnv() = default;

  ManualP2PEnv(const ManualP2PEnv&)            = delete;
  ManualP2PEnv& operator=(const ManualP2PEnv&) = delete;

  ManualP2PEnv(ManualP2PEnv&&) noexcept            = default;
  ManualP2PEnv& operator=(ManualP2PEnv&&) noexcept = default;
};

inline std::string sanitize_allocator_error(std::string_view what) {
  constexpr std::string_view kEnvToken = "VBT_CUDA_ALLOC_CONF";
  if (what.find(kEnvToken) != std::string_view::npos) {
    return "(allocator env redacted)";
  }
  if (what.size() > 200) {
    return std::string(what.substr(0, 200)) + "...";
  }
  return std::string(what);
}

inline ManualP2PEnv make_manual_p2p_env() {
  ManualP2PEnv env;

  env.dev0 = 0;
  env.dev1 = 1;

  // CONTRACT: make_manual_p2p_env is only called after
  // check_manual_p2p_prereqs(...) returned kOk in the same test.
  // It re-validates only device_count() >= 2 to guard obvious misuse.
  if (device_count() < 2) {
    ADD_FAILURE() << "make_manual_p2p_env called without satisfying prereqs (device_count < 2)";
    return env;
  }

  try {
    env.alloc0 = &Allocator::get(env.dev0);
    env.alloc1 = &Allocator::get(env.dev1);
  } catch (const std::exception& e) {
    ADD_FAILURE() << "Fabric: allocator configuration error: "
                  << sanitize_allocator_error(e.what());
    return env;
  }

  env.stream0       = vbt::cuda::getStreamFromPool(/*high_priority=*/false, env.dev0);
  env.stream1_fill  = vbt::cuda::getStreamFromPool(false, env.dev1);
  env.stream1_proxy = vbt::cuda::getStreamFromPool(false, env.dev1);

  return env;  // move-return.
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

namespace {

inline cudaStream_t as_cuda_stream(Stream s) noexcept {
  return reinterpret_cast<cudaStream_t>(s.handle());
}

struct AllocatedPtrDeleter {
  Allocator*  alloc{nullptr};
  DeviceIndex dev{0};
  Stream      stream;

  void operator()(void* p) const noexcept {
    if (!p || !alloc) return;
    vbt::cuda::DeviceGuard dg(dev);
    vbt::cuda::CUDAStreamGuard sg(stream);
    alloc->raw_delete(p);
  }
};

using UniqueDevicePtr = std::unique_ptr<void, AllocatedPtrDeleter>;

__global__ void manual_p2p_add_kernel(float* dst, const float* src, std::size_t n) {
  std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    static_cast<std::size_t>(threadIdx.x);
  if (idx < n) {
    dst[idx] += src[idx];
  }
}

inline void run_manual_p2p_add_once(const ManualP2PEnv& env,
                                   const ManualP2PConfig& cfg,
                                   bool exercise_lifetime) {
  ASSERT_NE(env.alloc0, nullptr);
  ASSERT_NE(env.alloc1, nullptr);
  ASSERT_GT(cfg.numel, 0u);

  const std::size_t N     = cfg.numel;
  const std::size_t bytes = N * sizeof(float);

  std::vector<float> host_dst(N);
  std::vector<float> host_src(N);
  for (std::size_t i = 0; i < N; ++i) {
    host_dst[i] = cfg.dst_base + static_cast<float>(i);
    host_src[i] = cfg.src_base + static_cast<float>(i);
  }

  UniqueDevicePtr dst0{nullptr, AllocatedPtrDeleter{env.alloc0, env.dev0, env.stream0}};
  UniqueDevicePtr src1{nullptr, AllocatedPtrDeleter{env.alloc1, env.dev1, env.stream1_fill}};

  ASSERT_NO_THROW(dst0.reset(env.alloc0->raw_alloc(bytes, env.stream0)));
  ASSERT_NO_THROW(src1.reset(env.alloc1->raw_alloc(bytes, env.stream1_fill)));

  ASSERT_NE(dst0.get(), nullptr);
  ASSERT_NE(src1.get(), nullptr);

  auto* dst0_f = static_cast<float*>(dst0.get());
  auto* src1_f = static_cast<float*>(src1.get());

  // H2D dst (device 0)
  {
    vbt::cuda::DeviceGuard dg(env.dev0);
    vbt::cuda::CUDAStreamGuard sg(env.stream0);
    ASSERT_EQ(cudaMemcpyAsync(dst0_f, host_dst.data(), bytes,
                              cudaMemcpyHostToDevice,
                              as_cuda_stream(env.stream0)),
              cudaSuccess);
  }

  // H2D src (device 1)
  {
    vbt::cuda::DeviceGuard dg(env.dev1);
    vbt::cuda::CUDAStreamGuard sg(env.stream1_fill);
    ASSERT_EQ(cudaMemcpyAsync(src1_f, host_src.data(), bytes,
                              cudaMemcpyHostToDevice,
                              as_cuda_stream(env.stream1_fill)),
              cudaSuccess);
    // Ensure src1 is initialized before launching the P2P consumer kernel.
    env.stream1_fill.synchronize();
  }

  // Launch kernel on device 0; it dereferences src1 (device 1 memory) via UVA+P2P.
  constexpr int kThreadsPerBlock = 128;
  int blocks = static_cast<int>((N + kThreadsPerBlock - 1) / kThreadsPerBlock);
  {
    vbt::cuda::DeviceGuard dg(env.dev0);
    vbt::cuda::CUDAStreamGuard sg(env.stream0);
    manual_p2p_add_kernel<<<blocks, kThreadsPerBlock, 0,
                            as_cuda_stream(env.stream0)>>>(dst0_f, src1_f, N);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "manual_p2p_add_kernel launch failed";
  }

  if (exercise_lifetime) {
    // Lifetime scenario:
    // - Ensure the allocator observes that src1 is in-use by a stream on dev1 that
    //   (via a cross-device event) depends on the dev0 consumer stream.
    // - Call raw_delete(src1) on the host before synchronizing the dev0 stream.
    // Under correct allocator semantics, this must be memcheck-clean.
    vbt::cuda::Event done;
    {
      vbt::cuda::DeviceGuard dg(env.dev0);
      ASSERT_NO_THROW(done.record(env.stream0));
    }
    {
      vbt::cuda::DeviceGuard dg(env.dev1);
      ASSERT_NO_THROW(done.wait(env.stream1_proxy));

#if defined(VBT_INTERNAL_TESTS)
      EXPECT_EQ(env.stream0.device_index(), env.dev0);
      EXPECT_EQ(env.stream1_proxy.device_index(), env.dev1);
#endif

      // Important: record the proxy stream (dev1) rather than the dev0 consumer.
      env.alloc1->record_stream(src1_f, env.stream1_proxy);

      // cross-device wait).
      vbt::cuda::CUDAStreamGuard sg(env.stream1_fill);
      env.alloc1->raw_delete(src1_f);

      // Prevent the RAII deleter from double-freeing.
      (void)src1.release();
    }
  }

  // D2H and validate.
  {
    vbt::cuda::DeviceGuard dg(env.dev0);
    vbt::cuda::CUDAStreamGuard sg(env.stream0);
    ASSERT_EQ(cudaMemcpyAsync(host_dst.data(), dst0_f, bytes,
                              cudaMemcpyDeviceToHost,
                              as_cuda_stream(env.stream0)),
              cudaSuccess);
    env.stream0.synchronize();
  }

  for (std::size_t i = 0; i < N; ++i) {
    float expected = (cfg.dst_base + static_cast<float>(i)) +
                     (cfg.src_base + static_cast<float>(i));
    EXPECT_EQ(host_dst[i], expected) << "mismatch at index " << i;
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

TEST(FabricManualP2PTest, FabricInitEnablesPeerAccessForDev0Dev1WhenSupported) {
  constexpr int kDev0 = 0;
  constexpr int kDev1 = 1;

  const int dc = device_count();
  if (dc < 2) {
    GTEST_SKIP() << "Need >= 2 CUDA devices (device_count=" << dc << ")";
  }

  FabricState& fs            = fabric_state();
  const FabricTopology& topo = fs.topology;

  if (fs.init_status != FabricInitStatus::Ok || !fs.uva_ok) {
    GTEST_SKIP() << "Fabric init/UVA gate not ok in this environment";
  }

  if (topo.device_count < 2 ||
      topo.can_access_peer.size() < 2 ||
      topo.p2p_enabled.size() < 2 ||
      topo.can_access_peer[kDev0].size() < 2 ||
      topo.can_access_peer[kDev1].size() < 2 ||
      topo.p2p_enabled[kDev0].size() < 2 ||
      topo.p2p_enabled[kDev1].size() < 2) {
    GTEST_SKIP() << "Fabric topology not populated for >=2 devices";
  }

  if (!topo.can_access_peer[kDev0][kDev1] ||
      !topo.can_access_peer[kDev1][kDev0]) {
    GTEST_SKIP() << "Peer access not supported between devices 0 and 1";
  }

  // The contract: when peer access is supported, Fabric initialization should
  // have already enabled it (and reflected it in topo.p2p_enabled).
  if (topo.p2p_enabled[kDev0][kDev1] && topo.p2p_enabled[kDev1][kDev0]) {
    return;
  }

  // If we can enable peer access manually, treat missing topo.p2p_enabled as a
  // regression. If enablement fails due to a capability/policy gap, skip.
  PeerAccessResult per = enable_peer_access_or_classify(kDev0, kDev1);
  switch (per.outcome) {
    case PeerAccessOutcome::kCapabilityUnsupported:
      GTEST_SKIP() << "Peer access enablement not supported in this environment";
      return;
    case PeerAccessOutcome::kUnexpectedError:
    default:
      FAIL() << "Unexpected CUDA error enabling peer access: dev0->1 status="
             << static_cast<int>(per.st01) << " (" << cuda_error_str(per.st01)
             << "), dev1->0 status=" << static_cast<int>(per.st10) << " ("
             << cuda_error_str(per.st10) << ")";
      return;
    case PeerAccessOutcome::kOk:
      FAIL() << "Fabric initialization did not enable peer access by default";
      return;
  }
}

TEST(FabricManualP2PTest, BasicManualP2PAdd) {
  ManualP2PPrereqInfo prereq;
  auto res = check_manual_p2p_prereqs(&prereq);
  switch (res) {
    case ManualP2PPrereqResult::kOk:
      break;
    case ManualP2PPrereqResult::kSkip:
      GTEST_SKIP() << prereq.message;
      return;
    case ManualP2PPrereqResult::kHardFail:
    default:
      FAIL() << prereq.message;
      return;
  }

  ManualP2PEnv env = make_manual_p2p_env();

  ManualP2PConfig cfg;
  cfg.numel    = 1024;
  cfg.dst_base = 1000.0f;
  cfg.src_base = 1.0f;

  run_manual_p2p_add_once(env, cfg, /*exercise_lifetime=*/false);
}

TEST(FabricManualP2PTest, AllocatorLifetimeRawDeleteBeforeSync) {
  ManualP2PPrereqInfo prereq;
  auto res = check_manual_p2p_prereqs(&prereq);
  switch (res) {
    case ManualP2PPrereqResult::kOk:
      break;
    case ManualP2PPrereqResult::kSkip:
      GTEST_SKIP() << prereq.message;
      return;
    case ManualP2PPrereqResult::kHardFail:
    default:
      FAIL() << prereq.message;
      return;
  }

  ManualP2PEnv env = make_manual_p2p_env();

  ManualP2PConfig cfg;
  cfg.numel    = 4096;
  cfg.dst_base = 1000.0f;
  cfg.src_base = 1.0f;

  run_manual_p2p_add_once(env, cfg, /*exercise_lifetime=*/true);
}

// -----------------------------------------------------------------------------
// Internal tests for prereq helper behavior. These only compile when
// VBT_INTERNAL_TESTS is enabled.
// -----------------------------------------------------------------------------

#if defined(VBT_INTERNAL_TESTS)

namespace {
struct FabricTestHooksGuard {
  FabricTestHooksGuard() = default;
  ~FabricTestHooksGuard() {
    using ::vbt::cuda::fabric::fabric_test_hooks;
    using ::vbt::cuda::fabric::reset_fabric_state_for_tests;
    auto& hooks = fabric_test_hooks();
    hooks.fake_topology_builder = {};
    hooks.forced_uva_ok.reset();
    reset_fabric_state_for_tests();
  }
};
}  // namespace

TEST(FabricManualP2PInternalTest, UvaDisabledYieldsSkipResult) {
  using ::vbt::cuda::fabric::FabricTopology;
  using ::vbt::cuda::fabric::fabric_test_hooks;
  using ::vbt::cuda::fabric::reset_fabric_state_for_tests;

  int dc = device_count();
  if (dc < 2) {
    GTEST_SKIP() << "Fabric internal prereq test requires >= 2 CUDA devices";
  }

  FabricTestHooksGuard guard;

  auto& hooks = fabric_test_hooks();
  hooks.fake_topology_builder = [dc](FabricTopology& topo) {
    topo.device_count = dc;
    topo.can_access_peer.assign(dc, std::vector<bool>(dc, false));
    topo.p2p_enabled.assign(dc, std::vector<bool>(dc, false));
    topo.clique_id.assign(dc, 0);
    topo.clique_size.assign(1, dc);
  };
  hooks.forced_uva_ok = false;

  reset_fabric_state_for_tests();

  ManualP2PPrereqInfo info;
  auto res = check_manual_p2p_prereqs(&info);
  EXPECT_EQ(res, ManualP2PPrereqResult::kSkip);
  EXPECT_NE(info.message.find("Fabric prereq: UVA gate disabled"),
            std::string::npos);
}

#endif  // VBT_INTERNAL_TESTS

}}}}  // namespace vbt::cuda::fabric::testonly

#else  // !VBT_WITH_CUDA

TEST(FabricManualP2PTest, BasicManualP2PAdd) {
  GTEST_SKIP() << "Built without CUDA";
}

TEST(FabricManualP2PTest, AllocatorLifetimeRawDeleteBeforeSync) {
  GTEST_SKIP() << "Built without CUDA";
}

#endif  // VBT_WITH_CUDA
