// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/graphs.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/allocator.h"
#include "vbt/rng/graph_capture.h"
#include "vbt/logging/logging.h"

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#  include <cuda_runtime_api.h>
#  include <new> // std::nothrow
#endif

#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace vbt { namespace cuda {

namespace {
thread_local bool t_replay_active = false;
} // anonymous

namespace detail {

std::atomic<std::uint64_t> g_captures_started{0};
std::atomic<std::uint64_t> g_captures_ended{0};
std::atomic<std::uint64_t> g_denied_default_stream{0};
std::atomic<std::uint64_t> g_nested_capture_denied{0};
std::atomic<std::uint64_t> g_end_in_dtor{0};
std::atomic<std::uint64_t> g_end_in_dtor_errors{0};
std::atomic<std::uint64_t> g_graphs_instantiated{0};
std::atomic<std::uint64_t> g_graphs_replayed{0};
std::atomic<std::uint64_t> g_replay_nesting_errors{0};
std::atomic<std::uint64_t> g_unsupported_capture_mode{0};
std::atomic<std::uint64_t> g_capture_begin_invalid_state{0};
std::atomic<std::uint64_t> g_capture_end_invalid_state{0};
std::atomic<std::uint64_t> g_instantiate_invalid_state{0};
std::atomic<std::uint64_t> g_instantiate_errors{0};
std::atomic<std::uint64_t> g_replay_invalid_state{0};
std::atomic<std::uint64_t> g_replay_device_mismatch{0};
std::atomic<std::uint64_t> g_replay_errors{0};
std::atomic<std::uint64_t> g_graphs_reset{0};
std::atomic<std::uint64_t> g_reset_invalid_state{0};
std::atomic<std::uint64_t> g_reset_inflight_denied{0};
std::atomic<std::uint64_t> g_reset_errors{0};
std::atomic<std::uint64_t> g_allocator_capture_denied{0};

inline void bump_captures_started() noexcept { g_captures_started.fetch_add(1, std::memory_order_relaxed); }
inline void bump_captures_ended() noexcept { g_captures_ended.fetch_add(1, std::memory_order_relaxed); }
inline void bump_denied_default_stream() noexcept { g_denied_default_stream.fetch_add(1, std::memory_order_relaxed); }
inline void bump_nested_capture_denied() noexcept { g_nested_capture_denied.fetch_add(1, std::memory_order_relaxed); }
inline void bump_end_in_dtor() noexcept { g_end_in_dtor.fetch_add(1, std::memory_order_relaxed); }
inline void bump_end_in_dtor_errors() noexcept { g_end_in_dtor_errors.fetch_add(1, std::memory_order_relaxed); }
inline void bump_graphs_instantiated() noexcept { g_graphs_instantiated.fetch_add(1, std::memory_order_relaxed); }
inline void bump_graphs_replayed() noexcept { g_graphs_replayed.fetch_add(1, std::memory_order_relaxed); }
inline void bump_replay_nesting_errors() noexcept { g_replay_nesting_errors.fetch_add(1, std::memory_order_relaxed); }
inline void bump_unsupported_capture_mode() noexcept { g_unsupported_capture_mode.fetch_add(1, std::memory_order_relaxed); }
inline void bump_capture_begin_invalid_state() noexcept { g_capture_begin_invalid_state.fetch_add(1, std::memory_order_relaxed); }
inline void bump_capture_end_invalid_state() noexcept { g_capture_end_invalid_state.fetch_add(1, std::memory_order_relaxed); }
inline void bump_instantiate_invalid_state() noexcept { g_instantiate_invalid_state.fetch_add(1, std::memory_order_relaxed); }
inline void bump_instantiate_errors() noexcept { g_instantiate_errors.fetch_add(1, std::memory_order_relaxed); }
inline void bump_replay_invalid_state() noexcept { g_replay_invalid_state.fetch_add(1, std::memory_order_relaxed); }
inline void bump_replay_device_mismatch() noexcept { g_replay_device_mismatch.fetch_add(1, std::memory_order_relaxed); }
inline void bump_replay_errors() noexcept { g_replay_errors.fetch_add(1, std::memory_order_relaxed); }
inline void bump_graphs_reset() noexcept { g_graphs_reset.fetch_add(1, std::memory_order_relaxed); }
inline void bump_reset_invalid_state() noexcept { g_reset_invalid_state.fetch_add(1, std::memory_order_relaxed); }
inline void bump_reset_inflight_denied() noexcept { g_reset_inflight_denied.fetch_add(1, std::memory_order_relaxed); }
inline void bump_reset_errors() noexcept { g_reset_errors.fetch_add(1, std::memory_order_relaxed); }
void bump_allocator_capture_denied() noexcept { g_allocator_capture_denied.fetch_add(1, std::memory_order_relaxed); }

} // namespace detail

// Definition of CUDAGraph inflight state used by host callbacks.
struct CUDAGraph::InflightState {
  // Cross-thread coordination between owner thread, callbacks, pollers.
  std::atomic<std::int64_t> counter{0};        // # replays in flight
  std::atomic<bool>         cleanup_requested{false};
  std::atomic<bool>         cleanup_enqueued{false};

  // Immutable after instantiate(); readable by callbacks and worker.
  DeviceIndex device{-1};
  MempoolId   pool{};

  // Written by instantiate() and, in the inflight-dtor branch, updated
  // by the destructor before setting cleanup_requested=true; read only
  // by the deferred-cleanup worker.
  void*       graph{nullptr};   // cudaGraph_t (opaque)
  void*       exec{nullptr};    // cudaGraphExec_t (opaque)
  bool        owns_pool{false}; // informational only; behavior keyed on pool.is_valid()
};

namespace {

#if VBT_WITH_CUDA
// Guard parity against CUDA enum values
static_assert(int(cudaStreamCaptureStatusNone) == 0, "unexpected int(cudaStreamCaptureStatusNone)");
static_assert(int(cudaStreamCaptureStatusActive) == 1, "unexpected int(cudaStreamCaptureStatusActive)");
static_assert(int(cudaStreamCaptureStatusInvalidated) == 2, "unexpected int(cudaStreamCaptureStatusInvalidated)");

static inline cudaStreamCaptureMode map_mode(CaptureMode m) noexcept {
  switch (m) {
    case CaptureMode::Global:      return cudaStreamCaptureModeGlobal;
    case CaptureMode::ThreadLocal: return cudaStreamCaptureModeThreadLocal;
    case CaptureMode::Relaxed:     return cudaStreamCaptureModeRelaxed;
    default:                       return cudaStreamCaptureModeThreadLocal;
  }
}
#endif

// Global counters live in vbt::cuda::detail (see above).

struct DeferredCleanupNode {
  std::shared_ptr<CUDAGraph::InflightState> inflight;
  DeferredCleanupNode* next{nullptr};
};

static std::atomic<DeferredCleanupNode*> g_deferred_head{nullptr};

#if defined(VBT_INTERNAL_TESTS)
std::atomic<std::uint64_t> g_debug_deferred_enqueue_failures{0};
std::atomic<std::uint64_t> g_debug_deferred_cleanup_errors{0};
#endif

bool enqueue_deferred_cleanup_node(
    std::shared_ptr<CUDAGraph::InflightState> inflight) noexcept {
#if VBT_WITH_CUDA
  if (!inflight) return false;
  auto* node = new (std::nothrow) DeferredCleanupNode{};
  if (!node) {
#if defined(VBT_INTERNAL_TESTS)
    g_debug_deferred_enqueue_failures.fetch_add(1, std::memory_order_relaxed);
#endif
    return false;  // leak-on-failure, but recorded in debug builds
  }
  node->inflight = std::move(inflight);

  DeferredCleanupNode* old_head = g_deferred_head.load(std::memory_order_relaxed);
  do {
    node->next = old_head;
  } while (!g_deferred_head.compare_exchange_weak(
      old_head, node, std::memory_order_release, std::memory_order_relaxed));
  return true;
#else
  (void)inflight;
  return false;
#endif
}

DeferredCleanupNode* take_deferred_list() noexcept {
#if VBT_WITH_CUDA
  return g_deferred_head.exchange(nullptr, std::memory_order_acq_rel);
#else
  return nullptr;
#endif
}

void perform_deferred_cleanup(DeferredCleanupNode* list) noexcept {
#if VBT_WITH_CUDA
  while (list) {
    auto* node = list;
    list = list->next;

    auto inflight = std::move(node->inflight);
    delete node;
    if (!inflight) continue;

    try {
      DeviceIndex dev = inflight->device;
      void*       g   = inflight->graph;
      void*       e   = inflight->exec;
      MempoolId   p   = inflight->pool;

      if (dev >= 0) {
        DeviceGuard dg(dev);
        if (e) (void)cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(e));
        if (g) (void)cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(g));
      }

      if (p.is_valid() && dev >= 0) {
        Allocator::release_pool(dev, p);
      }
    } catch (...) {
#if defined(VBT_INTERNAL_TESTS)
      g_debug_deferred_cleanup_errors.fetch_add(1, std::memory_order_relaxed);
#endif
      // Best-effort: swallow; leak-on-error.
    }
  }
#else
  (void)list;
#endif
}

void maybe_enqueue_cleanup(
    const std::shared_ptr<CUDAGraph::InflightState>& inflight) noexcept {
  if (!inflight) return;

  // Destructor must have requested cleanup.
  if (!inflight->cleanup_requested.load(std::memory_order_acquire)) return;

  // All replays must have completed.
  if (inflight->counter.load(std::memory_order_acquire) != 0) return;

  bool expected = false;
  if (!inflight->cleanup_enqueued.compare_exchange_strong(
          expected,
          true,
          std::memory_order_acq_rel,
          std::memory_order_relaxed)) {
    return;  // someone else already enqueued or is enqueuing
  }

  if (!enqueue_deferred_cleanup_node(inflight)) {
    // Enqueue failed (e.g., OOM). No automatic retry; permanent leak for
    // this graph instance, but g_debug_deferred_enqueue_failures was
    // incremented inside enqueue_deferred_cleanup_node.
    inflight->cleanup_enqueued.store(false, std::memory_order_release);
  }
}

#if VBT_WITH_CUDA
struct ReplayCallbackCtx {
  std::shared_ptr<CUDAGraph::InflightState> inflight;
};

static void CUDAGraphReplayHostFunc(void* data) noexcept {
  auto* ctx = static_cast<ReplayCallbackCtx*>(data);
  if (!ctx) return;

  std::shared_ptr<CUDAGraph::InflightState> inflight = std::move(ctx->inflight);
  delete ctx;
  if (!inflight) return;

  // 1. Mark this replay as complete.
  inflight->counter.fetch_sub(1, std::memory_order_acq_rel);

  // 2. End replay in allocator (host-only, CUDA-free).
  DeviceIndex dev  = inflight->device;
  MempoolId   pool = inflight->pool;
  if (pool.is_valid() && dev >= 0) {
    Allocator::mark_pool_replay_end(dev, pool);  // callback-safe, no CUDA
  }

  // 3. If cleanup has been requested and this was the last replay,
  //    enqueue deferred cleanup.
  maybe_enqueue_cleanup(inflight);
}
#endif

// Thread-local replay nesting guard handled via t_replay_active.

struct ReplayGuard {
  bool engaged{false};
  ReplayGuard() {
    if (t_replay_active) {
      detail::bump_replay_nesting_errors();
      throw std::runtime_error(kErrReplayNestedGuard);
    }
    t_replay_active = true;
    engaged = true;
  }
  ~ReplayGuard() noexcept {
    if (engaged) {
      t_replay_active = false;
    }
  }
};

struct PoolReplayGuard {
  DeviceIndex device{-1};
  MempoolId   pool{};
  bool        engaged{false};

  PoolReplayGuard(DeviceIndex dev, MempoolId p)
      : device(dev), pool(p) {
    if (pool.is_valid() && device >= 0) {
      Allocator::mark_pool_replay_begin(device, pool);  // may throw
      engaged = true;
    }
  }

  void dismiss() noexcept { engaged = false; }

  ~PoolReplayGuard() noexcept {
    if (engaged && pool.is_valid() && device >= 0) {
      Allocator::mark_pool_replay_end(device, pool);
    }
  }
};

#if VBT_WITH_CUDA

static void ensure_owner_thread(CUDAGraph& g) noexcept {
  (void)g;
}

#else  // !VBT_WITH_CUDA

static void ensure_owner_thread(CUDAGraph&) noexcept {}

#endif  // VBT_WITH_CUDA

} // anonymous

// CUDAStreamCaptureModeGuard PIMPL
struct CUDAStreamCaptureModeGuard::Impl {
#if VBT_WITH_CUDA
  cudaStreamCaptureMode prev{cudaStreamCaptureModeGlobal};
  bool have_prev{false};
#endif
};

CUDAStreamCaptureModeGuard::CUDAStreamCaptureModeGuard(Stream /*s*/, CaptureMode mode) noexcept {
  (void)mode;
#if VBT_WITH_CUDA
  pimpl_ = new (std::nothrow) Impl();
  if (!pimpl_) {
    VBT_LOG(WARNING) << "[cuda][graphs] failed to allocate CUDAStreamCaptureModeGuard::Impl; "
                     << "stream capture mode will not be restored for this scope";
    return;
  }
  auto desired = map_mode(mode);
  // Exchange; CUDA writes previous into desired
  auto st = cudaThreadExchangeStreamCaptureMode(&desired);
  if (st == cudaSuccess) {
    pimpl_->prev = desired;
    pimpl_->have_prev = true;
  } else {
    pimpl_->have_prev = false;
  }
#else
  (void)pimpl_;
#endif
}

CUDAStreamCaptureModeGuard::~CUDAStreamCaptureModeGuard() noexcept {
#if VBT_WITH_CUDA
  if (pimpl_) {
    if (pimpl_->have_prev) {
      auto prev = pimpl_->prev;
      (void)cudaThreadExchangeStreamCaptureMode(&prev);
    }
    delete pimpl_;
    pimpl_ = nullptr;
  }
#endif
}

CaptureStatus streamCaptureStatus(Stream s) {
#if VBT_WITH_CUDA
  // Treat legacy default stream (id==0) as not capturing in status helper fast-path.
  // Raw CUDA captures on default stream are still allowed; callers that need to
  // detect nested capture on default should use explicit non-default streams.
  if (s.id() == 0) {
    return CaptureStatus::None;
  }
  DeviceGuard g(s.device_index());
  cudaStreamCaptureStatus raw = cudaStreamCaptureStatusNone;
  auto st = cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(s.handle()), &raw);
  if (st != cudaSuccess) {
    // Do not clear sticky error; conservatively report Invalid
    return CaptureStatus::Invalid;
  }
  switch (raw) {
    case cudaStreamCaptureStatusNone:         return CaptureStatus::None;
    case cudaStreamCaptureStatusActive:       return CaptureStatus::Active;
    case cudaStreamCaptureStatusInvalidated:  return CaptureStatus::Invalid;
    default:                                  return CaptureStatus::Invalid;
  }
#else
  (void)s; return CaptureStatus::None;
#endif
}

CaptureStatus currentStreamCaptureStatus(DeviceIndex dev) {
  return streamCaptureStatus(getCurrentStream(dev));
}

GraphCounters cuda_graphs_counters() noexcept {
  GraphCounters c{};
  c.captures_started      = detail::g_captures_started.load(std::memory_order_relaxed);
  c.captures_ended        = detail::g_captures_ended.load(std::memory_order_relaxed);
  c.denied_default_stream = detail::g_denied_default_stream.load(std::memory_order_relaxed);
  c.nested_capture_denied = detail::g_nested_capture_denied.load(std::memory_order_relaxed);
  c.end_in_dtor           = detail::g_end_in_dtor.load(std::memory_order_relaxed);
  c.end_in_dtor_errors    = detail::g_end_in_dtor_errors.load(std::memory_order_relaxed);
  c.graphs_instantiated   = detail::g_graphs_instantiated.load(std::memory_order_relaxed);
  c.graphs_replayed       = detail::g_graphs_replayed.load(std::memory_order_relaxed);
  c.replay_nesting_errors = detail::g_replay_nesting_errors.load(std::memory_order_relaxed);
  c.unsupported_capture_mode    = detail::g_unsupported_capture_mode.load(std::memory_order_relaxed);
  c.capture_begin_invalid_state = detail::g_capture_begin_invalid_state.load(std::memory_order_relaxed);
  c.capture_end_invalid_state   = detail::g_capture_end_invalid_state.load(std::memory_order_relaxed);
  c.instantiate_invalid_state   = detail::g_instantiate_invalid_state.load(std::memory_order_relaxed);
  c.instantiate_errors          = detail::g_instantiate_errors.load(std::memory_order_relaxed);
  c.replay_invalid_state        = detail::g_replay_invalid_state.load(std::memory_order_relaxed);
  c.replay_device_mismatch      = detail::g_replay_device_mismatch.load(std::memory_order_relaxed);
  c.replay_errors               = detail::g_replay_errors.load(std::memory_order_relaxed);
  c.graphs_reset                = detail::g_graphs_reset.load(std::memory_order_relaxed);
  c.reset_invalid_state         = detail::g_reset_invalid_state.load(std::memory_order_relaxed);
  c.reset_inflight_denied       = detail::g_reset_inflight_denied.load(std::memory_order_relaxed);
  c.reset_errors                = detail::g_reset_errors.load(std::memory_order_relaxed);
  c.allocator_capture_denied    = detail::g_allocator_capture_denied.load(std::memory_order_relaxed);
  return c;
}

void assert_not_capturing_backward_stream(const vbt::core::Device& autograd_device) {
#if VBT_WITH_CUDA
  if (autograd_device.type != kDLCUDA) {
    return;
  }
  DeviceIndex dev_idx = static_cast<DeviceIndex>(autograd_device.index);
  CaptureStatus status = currentStreamCaptureStatus(dev_idx);
  if (status == CaptureStatus::Active) {
    throw std::runtime_error(
        "VibeTensor CUDA autograd: backward with CUDA gradients is not supported under CUDA Graph capture in this build");
  }
#else
  (void)autograd_device;
#endif
}

namespace detail {

void poll_deferred_graph_cleanup() noexcept {
  perform_deferred_cleanup(take_deferred_list());
}

} // namespace detail

// ----- CUDAGraph internals -----

struct CUDAGraph::GraphCaptureSession {
  CUDAGraph* parent_{};
  Stream      stream_;
  DeviceIndex dev_;
  CaptureMode mode_;
  bool        begun_{false};
  bool        finished_{false};

  GraphCaptureSession(CUDAGraph* parent,
                      Stream s,
                      DeviceIndex dev,
                      CaptureMode mode) noexcept
      : parent_(parent), stream_(s), dev_(dev), mode_(mode) {}

  void mark_begun() noexcept { begun_ = true; }
  void mark_finished() noexcept { finished_ = true; }
  [[nodiscard]] bool active() const noexcept { return begun_ && !finished_; }

  ~GraphCaptureSession() noexcept {
#if VBT_WITH_CUDA
    if (!active()) {
      return;
    }
    cudaGraph_t tmp = nullptr;
    {
      DeviceGuard dg(dev_);
      cudaError_t rc = cudaStreamEndCapture(
          reinterpret_cast<cudaStream_t>(stream_.handle()), &tmp);
      if (rc == cudaSuccess) {
        detail::bump_end_in_dtor();
        if (tmp != nullptr) {
          (void)cudaGraphDestroy(tmp);
        }
      } else {
        detail::bump_end_in_dtor_errors();
      }
    }

    if (parent_) {
      parent_->abort_rng_capture_for_graph();
    }
#endif
  }
};

CUDAGraph::~CUDAGraph() noexcept {
#if VBT_WITH_CUDA
#ifdef VBT_INTERNAL_TESTS
  ensure_owner_thread(*this);
#endif

  // Abort RNG capture before allocator/capture teardown.
  abort_rng_capture_for_graph();

  // Tear down allocator routing and capture session (unchanged).
  if (alloc_guard_ != nullptr) {
    auto* guard = static_cast<Allocator::AllocateToPoolGuard*>(alloc_guard_);
    guard->cancel();
    delete guard;
    alloc_guard_ = nullptr;
  }
  if (sess_ != nullptr) {
    delete sess_;
    sess_ = nullptr;
  }

  std::shared_ptr<InflightState> inflight = inflight_state_;
  std::int64_t inflight_count = 0;
  if (inflight) {
    inflight_count = inflight->counter.load(std::memory_order_acquire);
  }

  if (!inflight || inflight_count == 0) {
    // No replays in flight: immediate cleanup.
    try {
      reset_impl(/*from_dtor=*/true);
    } catch (...) {
      // noexcept destructor: swallow.
    }
    return;
  }

  // Replays in flight: defer cleanup via InflightState + pollers.
  inflight->graph     = graph_;
  inflight->exec      = exec_;
  inflight->owns_pool = owns_pool_;
  // device and pool were set in instantiate() and are immutable.

  inflight->cleanup_requested.store(true, std::memory_order_release);

  // If counter has already dropped to zero in a race, destructor may
  // itself enqueue cleanup.
  maybe_enqueue_cleanup(inflight);

  // Clear this object's handles; actual CUDA/allocator cleanup is owned
  // by the worker via inflight.
  graph_     = nullptr;
  exec_      = nullptr;
  pool_      = MempoolId{};
  owns_pool_ = false;
  device_    = -1;
  state_     = State::Destroyed;
#else
  (void)alloc_guard_;
  (void)sess_;
  (void)pool_;
  (void)owns_pool_;
  (void)device_;
  (void)graph_;
  (void)exec_;
  (void)inflight_state_;
#endif
}

void CUDAGraph::begin_rng_capture_for_graph(DeviceIndex dev) {
#if !VBT_WITH_CUDA
  (void)dev;
  return;
#else
#ifdef VBT_INTERNAL_TESTS
  if (dev < 0) {
    throw std::logic_error("CUDAGraph: negative device index in begin_rng_capture_for_graph");
  }
  if (rng_capture_active_ || rng_gen_ != nullptr) {
    throw std::logic_error("CUDAGraph: RNG capture already active at begin_rng_capture_for_graph entry");
  }
#endif

  if (dev < 0) {
    // In release builds, degrade gracefully: skip RNG integration rather than UB.
    return;
  }

  int dev_index = static_cast<int>(dev);
  vbt::rng::CudaGenerator& gen = vbt::rng::default_cuda(dev_index);

  // May throw (e.g., overlapping capture on this generator).
  vbt::rng::graph_capture::on_cuda_graph_capture_begin(gen);
  rng_gen_ = &gen;
  rng_capture_active_ = true;
#endif
}

void CUDAGraph::abort_rng_capture_for_graph() noexcept {
#if !VBT_WITH_CUDA
  return;
#else
  if (!rng_capture_active_ || rng_gen_ == nullptr) {
    return;  // idempotent
  }

  vbt::rng::CudaGenerator* gen = rng_gen_;
  rng_capture_active_ = false;
  rng_gen_ = nullptr;

  try {
    vbt::rng::graph_capture::on_cuda_graph_capture_abort(*gen);
  } catch (...) {
    // Step-1 promises noexcept here; swallow defensively to keep
    // destructors and error paths noexcept.
  }
#endif
}

void CUDAGraph::finalize_rng_capture_for_graph_success() {
#if !VBT_WITH_CUDA
  return;
#else
  if (!rng_capture_active_ || rng_gen_ == nullptr) {
    return;  // capture never started or already aborted
  }

  vbt::rng::CudaGenerator* gen = rng_gen_;

  try {
    (void)vbt::rng::graph_capture::on_cuda_graph_capture_end_success(*gen);
  } catch (...) {
    // Ensure capture is not leaked on any exception path from end_success.
    try {
      vbt::rng::graph_capture::on_cuda_graph_capture_abort(*gen);
    } catch (...) {
      // Abort is specified noexcept; swallow defensively to keep
      // destructors and error paths noexcept even if that changes.
    }
    rng_capture_active_ = false;
    rng_gen_ = nullptr;
    throw;  // treated as capture_end failure
  }

  rng_capture_active_ = false;
  rng_gen_ = nullptr;
#endif
}

void CUDAGraph::capture_begin(std::optional<Stream> stream,
                              std::optional<MempoolId> pool,
                              CaptureMode mode) {
#if !VBT_WITH_CUDA
  (void)stream; (void)pool; (void)mode;
  throw std::runtime_error(kErrCudaGraphsUnavailable);
#else
#ifdef VBT_INTERNAL_TESTS
  ensure_owner_thread(*this);
#endif
  if (state_ != State::None) {
    detail::bump_capture_begin_invalid_state();
    throw std::runtime_error(kErrGraphBeginInvalidState);
  }

  if (mode != CaptureMode::ThreadLocal) {
    detail::bump_unsupported_capture_mode();
    throw std::runtime_error(kErrUnsupportedCaptureMode);
  }

  Stream s = stream ? *stream : getCurrentStream(-1);

  // Nested capture on the same stream is banned (check first to prioritize nested over default-stream ban).
  if (streamCaptureStatus(s) == CaptureStatus::Active) {
    detail::bump_nested_capture_denied();
    throw std::runtime_error(kErrNestedCaptureBan);
  }

  // Default stream capture is banned.
  if (s.id() == 0) {
    detail::bump_denied_default_stream();
    throw std::runtime_error(kErrDefaultStreamCaptureBan);
  }

  DeviceIndex dev = s.device_index();
  device_ = dev;
  capture_stream_ = s;
  capture_id_ = CaptureId{};
  graph_ = nullptr;
  exec_ = nullptr;
  inflight_state_.reset();

  // Pool selection & retention
  if (pool && pool->is_valid()) {
    if (pool->dev != dev) {
      detail::bump_capture_begin_invalid_state();
      std::string msg = std::string(kErrGraphDeviceMismatchPrefix) +
                        ": graph_device=cuda:" +
                        std::to_string(static_cast<int>(dev)) +
                        " pool_device=cuda:" +
                        std::to_string(static_cast<int>(pool->dev));
      throw std::runtime_error(msg);
    }
    pool_ = *pool;
    owns_pool_ = false;
  } else {
    pool_ = Allocator::create_pool_id(dev);
    owns_pool_ = true;
  }

  bool retain_done = false;
  Allocator::AllocateToPoolGuard* guard_ptr = nullptr;

  try {
    Allocator::retain_pool(dev, pool_);
    retain_done = true;

    // Pre-warm the graph-private pool for this capture stream before routing
    // and capture begin. This primes the freelists so routed allocations during
    // capture can reuse blocks without calling cudaMalloc.
    {
      std::size_t prewarm_block_bytes = 1ull << 20;  // 1 MiB
      int         prewarm_blocks      = 4;
      std::size_t B_target            = prewarm_block_bytes * static_cast<std::size_t>(prewarm_blocks);
      Allocator::prewarm_graph_pool_for_stream(dev, pool_, s, B_target, prewarm_blocks);
    }

    // begin_allocate_to_pool returns a guard by value; keep it alive on the heap.
    Allocator::AllocateToPoolGuard guard_local =
        Allocator::begin_allocate_to_pool(dev, pool_);
    guard_ptr = new Allocator::AllocateToPoolGuard(std::move(guard_local));
    alloc_guard_ = guard_ptr;

    // Create capture session RAII and mark begun before BeginCapture.
    sess_ = new GraphCaptureSession(this, s, dev, mode);
    sess_->mark_begun();

    begin_rng_capture_for_graph(dev);

    {
      DeviceGuard dg(dev);
      cudaError_t rc = cudaStreamBeginCapture(
          reinterpret_cast<cudaStream_t>(s.handle()), map_mode(mode));
      if (rc != cudaSuccess) {
        // Roll back: mark session finished (no fallback), cancel routing, release pool.
        sess_->mark_finished();
        delete sess_;
        sess_ = nullptr;

        guard_ptr->cancel();
        delete guard_ptr;
        guard_ptr = nullptr;
        alloc_guard_ = nullptr;

        Allocator::release_pool(dev, pool_);
        pool_ = MempoolId{};
        owns_pool_ = false;

        abort_rng_capture_for_graph();

        throw std::runtime_error(std::string("cudaStreamBeginCapture failed: ") +
                                 cudaGetErrorString(rc));
      }

      // Best-effort query capture id
      cudaStreamCaptureStatus raw_status = cudaStreamCaptureStatusNone;
      unsigned long long cid = 0;
      cudaError_t rc_info = cudaStreamGetCaptureInfo(
          reinterpret_cast<cudaStream_t>(s.handle()), &raw_status, &cid);
      if (rc_info == cudaSuccess &&
          raw_status == cudaStreamCaptureStatusActive) {
        capture_id_.value = cid;
      } else {
        capture_id_.value = 0;
      }
    }

    state_ = State::Capturing;
    detail::bump_captures_started();
  } catch (...) {
    // Generic unwind for any failure after pool_ was set.
    abort_rng_capture_for_graph();
    if (guard_ptr) {
      guard_ptr->cancel();
      delete guard_ptr;
      alloc_guard_ = nullptr;
    }
    if (sess_) {
      sess_->mark_finished();
      delete sess_;
      sess_ = nullptr;
    }
    if (retain_done && pool_.is_valid()) {
      Allocator::release_pool(dev, pool_);
    }
    pool_ = MempoolId{};
    owns_pool_ = false;
    device_ = -1;
    capture_stream_ = Stream{Stream::UNCHECKED, 0u, 0};
    capture_id_.value = 0;
    state_ = State::None;
    throw;
  }
#endif
}

void CUDAGraph::capture_end() {
#if !VBT_WITH_CUDA
  throw std::runtime_error(kErrCudaGraphsUnavailable);
#else
#ifdef VBT_INTERNAL_TESTS
  ensure_owner_thread(*this);
#endif

  if (state_ != State::Capturing) {
    detail::bump_capture_end_invalid_state();
    throw std::runtime_error(kErrGraphEndInvalidState);
  }

  cudaGraph_t tmp = nullptr;
  {
    DeviceGuard dg(device_);
    cudaError_t rc = cudaStreamEndCapture(
        reinterpret_cast<cudaStream_t>(capture_stream_.handle()), &tmp);
    if (rc != cudaSuccess) {
      // Tear down routing but leave session active for dtor fallback.
      if (alloc_guard_ != nullptr) {
        auto* guard = static_cast<Allocator::AllocateToPoolGuard*>(alloc_guard_);
        guard->cancel();
        delete guard;
        alloc_guard_ = nullptr;
      }
      // RNG capture cannot be completed; abort immediately.
      abort_rng_capture_for_graph();
      throw std::runtime_error(std::string("cudaStreamEndCapture failed: ") +
                               cudaGetErrorString(rc));
    }
  }

  if (!tmp) {
    // Invalid capture; behave similarly to upstream TORCH_CHECK, but clean up.
    if (alloc_guard_ != nullptr) {
      auto* guard = static_cast<Allocator::AllocateToPoolGuard*>(alloc_guard_);
      guard->cancel();
      delete guard;
      alloc_guard_ = nullptr;
    }
    if (sess_ != nullptr) {
      sess_->mark_finished();
      delete sess_;
      sess_ = nullptr;
    }

    abort_rng_capture_for_graph();

    state_ = State::None;
    throw std::runtime_error("Invalid capture.");
  }

  graph_ = tmp;

  // Finalize RNG capture before tearing down allocator routing and session.
  try {
    finalize_rng_capture_for_graph_success();
  } catch (...) {
    // RNG finalize failed (e.g., generator state mutated during capture).
    // Best-effort cleanup of graph/exec/pool; leave this object empty.
    try {
      reset_impl(/*from_dtor=*/false);
    } catch (...) {
      // Avoid throwing a different exception than the RNG error.
    }
    state_ = State::None;
    throw;
  }

  if (alloc_guard_ != nullptr) {
    auto* guard = static_cast<Allocator::AllocateToPoolGuard*>(alloc_guard_);
    guard->end();
    delete guard;
    alloc_guard_ = nullptr;
  }

  if (sess_ != nullptr) {
    sess_->mark_finished();
    delete sess_;
    sess_ = nullptr;
  }

  state_ = State::Captured;
  detail::bump_captures_ended();
#endif
}

void CUDAGraph::instantiate() {
#if !VBT_WITH_CUDA
  throw std::runtime_error(kErrCudaGraphsUnavailable);
#else
#ifdef VBT_INTERNAL_TESTS
  ensure_owner_thread(*this);
#endif
  if (state_ != State::Captured || graph_ == nullptr || exec_ != nullptr ||
      device_ < 0) {
    detail::bump_instantiate_invalid_state();
    throw std::runtime_error(kErrGraphInstantiateInvalidState);
  }

  // Allocate inflight state first so we stay exception-safe even on OOM.
  auto inflight_sp = std::make_shared<InflightState>();
  inflight_sp->device = device_;
  inflight_sp->graph  = graph_;
  inflight_sp->exec   = nullptr;
  inflight_sp->pool   = pool_;
  inflight_sp->owns_pool = owns_pool_;

  DeviceGuard dg(device_);
  auto g = reinterpret_cast<cudaGraph_t>(graph_);
  cudaGraphExec_t exec_raw = nullptr;

  cudaError_t rc = cudaGraphInstantiate(
      &exec_raw,
      g,
      /*flags*/0);
  if (rc != cudaSuccess) {
    detail::bump_instantiate_errors();
    throw std::runtime_error(std::string("cudaGraphInstantiate failed: ") +
                             cudaGetErrorString(rc));
  }

  exec_ = exec_raw;
  inflight_sp->exec = exec_raw;
  inflight_state_ = std::move(inflight_sp);
  state_ = State::Instantiated;

  detail::bump_graphs_instantiated();
#endif
}

void CUDAGraph::reset() {
#if !VBT_WITH_CUDA
  throw std::runtime_error(kErrCudaGraphsUnavailable);
#else
#ifdef VBT_INTERNAL_TESTS
  ensure_owner_thread(*this);
#endif
  if ((state_ != State::Captured && state_ != State::Instantiated) ||
      device_ < 0) {
    detail::bump_reset_invalid_state();
    throw std::runtime_error(kErrGraphResetInvalidState);
  }

  if (inflight_state_) {
    std::int64_t in_flight = inflight_state_->counter.load(std::memory_order_acquire);
    if (in_flight > 0) {
      detail::bump_reset_inflight_denied();
      throw std::runtime_error(kErrGraphResetInflightDenied);
    }
  }

  try {
    reset_impl(/*from_dtor=*/false);
    detail::bump_graphs_reset();
  } catch (...) {
    detail::bump_reset_errors();
    throw;
  }
#endif
}

void CUDAGraph::reset_impl(bool from_dtor) {
#if !VBT_WITH_CUDA
  (void)from_dtor;
  return;
#else
  // Ensure RNG capture is not left active when resetting this graph.
  abort_rng_capture_for_graph();

  // Best-effort destruction of exec_ and graph_ under DeviceGuard.
  if (device_ >= 0) {
    DeviceGuard dg(device_);
    if (exec_ != nullptr) {
      (void)cudaGraphExecDestroy(
          reinterpret_cast<cudaGraphExec_t>(exec_));
      exec_ = nullptr;
    }
    if (graph_ != nullptr) {
      (void)cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(graph_));
      graph_ = nullptr;
    }
  }

  if (pool_.is_valid()) {
    if (from_dtor) {
      try {
        Allocator::release_pool(device_, pool_);
      } catch (...) {
        // Destructor path: swallow errors.
      }
    } else {
      Allocator::release_pool(device_, pool_);
    }
  }

  pool_ = MempoolId{};
  owns_pool_ = false;

  device_ = -1;
  state_ = State::None;
  capture_stream_ = Stream{Stream::UNCHECKED, 0u, 0};
  capture_id_ = CaptureId{};

  inflight_state_.reset();
#endif
}

void CUDAGraph::replay(std::optional<Stream> s) {
#if !VBT_WITH_CUDA
  (void)s;
  throw std::runtime_error(kErrCudaGraphsUnavailable);
#else
#ifdef VBT_INTERNAL_TESTS
  ensure_owner_thread(*this);
#endif
  ReplayGuard guard;  // may throw kErrReplayNestedGuard

  if (state_ != State::Instantiated || graph_ == nullptr || exec_ == nullptr ||
      device_ < 0 || !inflight_state_) {
    detail::bump_replay_invalid_state();
    throw std::runtime_error(kErrGraphReplayInvalidState);
  }

  Stream target = s.has_value() ? *s : capture_stream_;
  DeviceIndex stream_dev = target.device_index();
  if (stream_dev != device_) {
    detail::bump_replay_device_mismatch();
    std::string msg = std::string(kErrGraphDeviceMismatchPrefix) +
                      ": graph_device=cuda:" +
                      std::to_string(static_cast<int>(device_)) +
                      " stream_device=cuda:" +
                      std::to_string(static_cast<int>(stream_dev));
    throw std::runtime_error(msg);
  }

  auto inflight = inflight_state_;
  if (!inflight) {
    detail::bump_replay_invalid_state();
    throw std::runtime_error(kErrGraphReplayInvalidState);
  }

  PoolReplayGuard pool_guard(device_, pool_);

  // Allocate callback context and register this replay as in flight.
  auto* ctx = new ReplayCallbackCtx{inflight};
  inflight->counter.fetch_add(1, std::memory_order_acq_rel);

  DeviceGuard dg(device_);
  auto exec = reinterpret_cast<cudaGraphExec_t>(exec_);

  cudaError_t rc_launch = cudaGraphLaunch(
      exec, reinterpret_cast<cudaStream_t>(target.handle()));
  if (rc_launch != cudaSuccess) {
    detail::bump_replay_errors();
    inflight->counter.fetch_sub(1, std::memory_order_acq_rel);
    delete ctx;
    throw std::runtime_error(std::string("cudaGraphLaunch failed: ") +
                             cudaGetErrorString(rc_launch));
  }

  cudaError_t rc_cb = cudaLaunchHostFunc(
      reinterpret_cast<cudaStream_t>(target.handle()),
      &CUDAGraphReplayHostFunc,
      ctx);
  if (rc_cb != cudaSuccess) {
    detail::bump_replay_errors();
    // Best-effort: wait for this replay to complete, then fix counter.
    (void)cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(target.handle()));
    inflight->counter.fetch_sub(1, std::memory_order_acq_rel);
    delete ctx;
    throw std::runtime_error(std::string("cudaLaunchHostFunc failed: ") +
                             cudaGetErrorString(rc_cb));
  }

  // Success: hand off replay-end responsibilities to host callback.
  pool_guard.dismiss();

  detail::bump_graphs_replayed();
#endif
}

#ifdef VBT_INTERNAL_TESTS

ReplayGuardTestScope::ReplayGuardTestScope() {
  t_replay_active = true;
}

ReplayGuardTestScope::~ReplayGuardTestScope() {
  t_replay_active = false;
}

int CUDAGraph::debug_inflight() const noexcept {
  if (inflight_state_) {
    return static_cast<int>(
        inflight_state_->counter.load(std::memory_order_relaxed));
  }
  return 0;
}

bool CUDAGraph::debug_has_graph() const noexcept {
  return graph_ != nullptr;
}

bool CUDAGraph::debug_has_exec() const noexcept {
  return exec_ != nullptr;
}
#endif

}} // namespace vbt::cuda
