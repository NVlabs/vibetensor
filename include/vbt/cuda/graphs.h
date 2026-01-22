// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <optional>
#include <memory>

#include "vbt/cuda/stream.h"
#include "vbt/core/device.h"

namespace vbt { namespace rng { class CudaGenerator; }}

namespace vbt { namespace cuda {

// Capture status for a CUDA stream
// Mirrors cudaStreamCaptureStatus values but is independent of CUDA headers here.
enum class CaptureStatus : int {
  None    = 0,
  Active  = 1,
  Invalid = 2,
};

// Capture strictness modes. ThreadLocal is the only supported mode currently.
enum class CaptureMode : int {
  Global,
  ThreadLocal,
  Relaxed,
};

struct CaptureId {
  std::uint64_t value{0};
  [[nodiscard]] bool is_valid() const noexcept { return value != 0; }
};

struct MempoolId {
  DeviceIndex   dev{-1};
  std::uint64_t id{0};
  [[nodiscard]] bool is_valid() const noexcept { return dev >= 0 && id != 0; }
  [[nodiscard]] bool operator==(const MempoolId& o) const noexcept { return dev == o.dev && id == o.id; }
  [[nodiscard]] std::string to_string() const {
    return std::string("GraphPoolHandle(device=cuda:") +
           std::to_string(static_cast<int>(dev)) + ", id=" +
           std::to_string(id) + ")";
  }
};

// RAII guard for thread-local CUDA stream capture mode.
// When CUDA is disabled, this guard is a no-op.
struct CUDAStreamCaptureModeGuard final {
  explicit CUDAStreamCaptureModeGuard(Stream /*s*/, CaptureMode mode) noexcept;
  ~CUDAStreamCaptureModeGuard() noexcept;

  CUDAStreamCaptureModeGuard(const CUDAStreamCaptureModeGuard&) = delete;
  CUDAStreamCaptureModeGuard& operator=(const CUDAStreamCaptureModeGuard&) = delete;

 private:
  // Implementation details live in graphs.cc to avoid leaking CUDA headers
  struct Impl;
  Impl* pimpl_{nullptr};
};

// Helpers to query CUDA graph capture status for a stream or the current stream
CaptureStatus streamCaptureStatus(Stream s);
CaptureStatus currentStreamCaptureStatus(DeviceIndex dev = -1);

// Pinned error substrings (single source of truth)
inline constexpr const char* kErrDefaultStreamCaptureBan =
  "CUDA Graph capture on the default stream is not allowed. Please create a non-default CUDA stream and capture on it (e.g., vc.Stream()).";
inline constexpr const char* kErrNestedCaptureBan =
  "nested CUDA graph capture is not allowed";
inline constexpr const char* kErrAllocatorCaptureDenied =
  "cuda allocator: allocations are forbidden during CUDA graph capture";
inline constexpr const char* kErrGraphBeginInvalidState =
  "capture_begin called in invalid state";
inline constexpr const char* kErrGraphEndInvalidState =
  "capture_end called in invalid state";
inline constexpr const char* kErrUnsupportedCaptureMode =
  "only ThreadLocal capture mode is supported";
inline constexpr const char* kErrGraphInstantiateInvalidState =
  "instantiate called in invalid state";
inline constexpr const char* kErrGraphReplayInvalidState =
  "replay called in invalid state";
inline constexpr const char* kErrReplayNestedGuard =
  "Nested ReplayGuard not supported";
inline constexpr const char* kErrGraphDeviceMismatchPrefix =
  "CUDA Graph device mismatch"; // prefix; device labels follow
inline constexpr const char* kErrCudaGraphsUnavailable =
  "CUDA Graphs are only available when CUDA is enabled";
inline constexpr const char* kErrGraphResetInvalidState =
  "reset called in invalid state";
inline constexpr const char* kErrGraphResetInflightDenied =
  "reset called while replays are in flight";

// Aggregate counters for CUDA graph capture lifecycle.
// NOTE: This struct is append-only for ABI stability; new fields must be
// added at the end and old fields must never be reordered.
struct GraphCounters {
  // Capture lifecycle and legality
  std::uint64_t captures_started{0};
  std::uint64_t captures_ended{0};
  std::uint64_t denied_default_stream{0};
  std::uint64_t nested_capture_denied{0};
  std::uint64_t end_in_dtor{0};
  std::uint64_t end_in_dtor_errors{0};
  // Instantiate and replay
  std::uint64_t graphs_instantiated{0};
  std::uint64_t graphs_replayed{0};
  std::uint64_t replay_nesting_errors{0};
  // Additional counters (append-only)
  // Capture and capture-mode
  std::uint64_t unsupported_capture_mode{0};
  std::uint64_t capture_begin_invalid_state{0};
  std::uint64_t capture_end_invalid_state{0};
  // Instantiate and replay
  std::uint64_t instantiate_invalid_state{0};
  std::uint64_t instantiate_errors{0};
  std::uint64_t replay_invalid_state{0};
  std::uint64_t replay_device_mismatch{0};
  std::uint64_t replay_errors{0};
  // Forward-looking reset counters (currently unused)
  std::uint64_t graphs_reset{0};
  std::uint64_t reset_invalid_state{0};
  std::uint64_t reset_inflight_denied{0};
  std::uint64_t reset_errors{0};
  // Allocator observability bridge
  std::uint64_t allocator_capture_denied{0};
};

// Snapshot current CUDA graph counters (defined in graphs.cc).
GraphCounters cuda_graphs_counters() noexcept;

// Assert that backward for a given autograd device is not running under
// CUDA Graph capture on its current stream.
void assert_not_capturing_backward_stream(const vbt::core::Device& autograd_device);

// Internal-only helpers for cross-TU counter updates and deferred cleanup.
namespace detail {

// Bump the allocator-capture-denied counter by 1.
void bump_allocator_capture_denied() noexcept;

// Drain any pending CUDA Graph deferred-cleanup jobs.
// Safe to call from arbitrary host threads while CUDA is initialized.
// No-op on CPU-only builds.
void poll_deferred_graph_cleanup() noexcept;

} // namespace detail

#ifdef VBT_INTERNAL_TESTS
// Internal-only helper for tests to simulate a nested replay guard
// on the current host thread.
struct ReplayGuardTestScope {
  ReplayGuardTestScope();
  ~ReplayGuardTestScope();
};
#endif

// CUDA Graph wrapper (capture lifecycle).
class CUDAGraph final {
 public:
  CUDAGraph() = default;
  ~CUDAGraph() noexcept;

  // Begin a CUDA Graph capture on the given stream (or current stream if nullopt).
  // Optional pool selects/creates a private allocator pool. Only ThreadLocal mode
  // is supported; Global/Relaxed throw with kErrUnsupportedCaptureMode.
  void capture_begin(std::optional<Stream> stream = std::nullopt,
                     std::optional<MempoolId> pool = std::nullopt,
                     CaptureMode mode = CaptureMode::ThreadLocal);

  // End an in-progress capture. Valid only when is_capturing()==true.
  void capture_end();

  // Instantiate and replay a captured graph.
  void instantiate();
  void replay(std::optional<Stream> stream = std::nullopt);

  // Audit fix: reset captured/instantiated graphs and release their pools.
  void reset();

  [[nodiscard]] DeviceIndex device() const noexcept { return device_; }
  [[nodiscard]] Stream capture_stream() const noexcept { return capture_stream_; }
  [[nodiscard]] MempoolId pool() const noexcept { return pool_; }
  [[nodiscard]] bool is_capturing() const noexcept { return state_ == State::Capturing; }

#ifdef VBT_INTERNAL_TESTS
  // Debug/testing helpers; no ABI guarantees.
  int  debug_inflight() const noexcept;
  bool debug_has_graph() const noexcept;
  bool debug_has_exec() const noexcept;
#endif

  // Forward-declared inflight state (defined in graphs.cc).
  struct InflightState;

 private:
  enum class State : std::uint8_t { None, Capturing, Captured, Instantiated, Destroyed };

  DeviceIndex device_{-1};
  State       state_{State::None};
  Stream      capture_stream_{Stream::UNCHECKED, 0u, 0};
  CaptureId   capture_id_{};
  MempoolId   pool_{};
  bool        owns_pool_{false};
  void*       graph_{nullptr};  // cudaGraph_t
  void*       exec_{nullptr};   // cudaGraphExec_t

  // Internal helpers (defined in graphs.cc)
  struct GraphCaptureSession;
  GraphCaptureSession* sess_{nullptr};   // owned, may trigger fallback EndCapture in dtor
  void*                alloc_guard_{nullptr}; // opaque pointer to Allocator::AllocateToPoolGuard

  std::shared_ptr<InflightState> inflight_state_{};  // host-callback-backed inflight counter

  // Non-owning pointer to the per-device default CUDA generator.
  // default_cuda(device_) returns a process-lifetime singleton whose address
  // remains stable, so storing a raw pointer here is safe.
  vbt::rng::CudaGenerator* rng_gen_{nullptr};
  bool                     rng_capture_active_{false};

  void begin_rng_capture_for_graph(DeviceIndex dev);
  void abort_rng_capture_for_graph() noexcept;
  void finalize_rng_capture_for_graph_success();

  void reset_impl(bool from_dtor);
};

}} // namespace vbt::cuda
