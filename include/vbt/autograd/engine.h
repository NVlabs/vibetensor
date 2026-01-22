// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <deque>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstddef>
#include <cstdint>

#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/device.h"
#include "vbt/autograd/types.h"
#include "vbt/autograd/node.h"
#include "vbt/autograd/engine_toggles.h"
#if VBT_WITH_CUDA
#include "vbt/cuda/event.h"
#endif

namespace vbt { namespace autograd {

class ReadyQueue {
 public:
  void push_back(const vbt::core::intrusive_ptr<Node>& n) { q_.push_back(n); }
  vbt::core::intrusive_ptr<Node> pop_front() {
    auto n = q_.front(); q_.pop_front(); return n;
  }
  bool empty() const noexcept { return q_.empty(); }
 private:
  std::deque<vbt::core::intrusive_ptr<Node>> q_;
};

struct GraphTask {
  struct InputBuffer {
    std::vector<OptionalTensor> grads_in; // size == expected slots
    std::vector<uint8_t>        present;  // flags per slot (1==arrived), includes present-only nullopt
    std::size_t                 expected{0};              // required slot count (==grads_in.size())
    std::size_t                 received{0};              // number of slots with present[pos] == 1
    bool                        enqueued{false};         // whether consumer enqueued to ready

    std::vector<uint8_t>           is_accel;        // slot participates in CUDA accumulation
    std::vector<uint8_t>           has_accum_stream;
    std::vector<vbt::core::Device> accum_device;    // == autograd_device when used
    std::vector<std::uint64_t>     accum_stream_id; // accumulation stream id

    std::vector<uint8_t>           has_ready_event;
    std::vector<vbt::core::Device> ready_device;    // == autograd_device when used
    std::vector<std::uint64_t>     ready_stream_id; // stream where ready_events[pos] recorded

#if VBT_WITH_CUDA
    std::vector<vbt::cuda::Event>  ready_events;    // one per slot
#endif

    InputBuffer() = default;
    InputBuffer(const InputBuffer&) = delete;
    InputBuffer& operator=(const InputBuffer&) = delete;
    InputBuffer(InputBuffer&&) = default;
    InputBuffer& operator=(InputBuffer&&) = default;

    // CPU helper: (re)initialize logical slots for a fresh buffer.
    void ensure_cpu_capacity(std::size_t n);
    // CUDA helper: initialize per-slot CUDA metadata for a fresh buffer.
    void ensure_cuda_capacity(std::size_t n);
  };

  ReadyQueue ready;
  std::unordered_map<Node*, InputBuffer> inputs;
  std::unordered_map<Node*, vbt::core::intrusive_ptr<Node>> keepalive; // ensure reachability

  std::unordered_map<Node*, int>  dependencies;   // E(n): #non-null incoming edges
  std::unordered_set<Node*>       nodes_in_graph; // nodes reachable from root

  std::vector<std::function<void()>> callbacks;

  // Debug counters (optional)
  std::size_t nodes_processed{0};
  std::size_t edges_processed{0};
  std::size_t duplicates_coalesced{0};
  std::size_t callbacks_run{0};

  using ValidateHook = void(*)(const Node&, const std::vector<OptionalTensor>&);
  ValidateHook validate_hook{nullptr};

  bool             has_autograd_device{false};
  vbt::core::Device autograd_device{};         // CPU or single CUDA device
  bool             streaming_enabled_snapshot{false};

  AutogradDeviceMode device_mode_snapshot{AutogradDeviceMode::SingleDevice};

  std::uint64_t cuda_events_recorded{0};
  std::uint64_t cuda_events_waited{0};
  std::uint64_t cuda_cross_stream_routes{0};

  std::uint64_t cuda_device_synchronizes{0};

  // These are populated only for MultiDeviceExperimental.
  std::vector<int> cuda_devices_snapshot;      // sorted unique CUDA indices participating
  std::vector<int> cuda_lane_devices_snapshot; // sorted unique CUDA indices with CUDA-lane nodes

  std::uint64_t routing_device_switches{0};
};

// Per-thread callback invoked once after each successful backward run.
// The callback is cleared after it fires.
using BackwardCompleteCallback = std::function<void()>;
void set_backward_complete_callback(BackwardCompleteCallback cb);

// CPU-only autograd engine. External callers MUST use the free
// run_backward / run_backward_with_hook entrypoints below; Engine is
// intended for internal/testing use only.
class Engine {
 public:
  // Process-wide singleton accessor. Implemented via a function-local
  // static with no side effects in the constructor.
  static Engine& get_default_engine() noexcept;

  void run_backward(vbt::core::intrusive_ptr<Node> root,
                    const std::vector<OptionalTensor>& initial_grads,
                    const std::vector<std::function<void()>>& callbacks = {});

  // Test-only helper preserving existing semantics.
  void run_backward_with_hook(vbt::core::intrusive_ptr<Node> root,
                              const std::vector<OptionalTensor>& initial_grads,
                              const std::vector<std::function<void()>>& callbacks,
                              GraphTask::ValidateHook hook);

  void start_device_threads_once() noexcept;
  void stop() noexcept;
  void release_workers() noexcept;

 private:
  Engine() = default;
  ~Engine() = default;  // non-virtual; singleton lifetime is process-wide
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Core implementation moved mechanically from the existing free
  // run_backward_impl helper in engine.cc.
  void run_backward_impl(vbt::core::intrusive_ptr<Node> root,
                         const std::vector<OptionalTensor>& initial_grads,
                         const std::vector<std::function<void()>>& callbacks,
                         GraphTask::ValidateHook hook);

};

// Call sites must not refer to Engine directly outside tests.
void run_backward(vbt::core::intrusive_ptr<Node> root,
                  const std::vector<OptionalTensor>& initial_grads,
                  const std::vector<std::function<void()>>& callbacks = {});

// Test-only helper: run with an external validation hook (executed
// after built-in validate).
void run_backward_with_hook(vbt::core::intrusive_ptr<Node> root,
                            const std::vector<OptionalTensor>& initial_grads,
                            const std::vector<std::function<void()>>& callbacks,
                            GraphTask::ValidateHook hook);

#if VBT_AUTOGRAD_TESTING
// Test-only: access the per-backward snapshots from the last successful run.
std::vector<int> _test_last_backward_cuda_devices_snapshot();
std::vector<int> _test_last_backward_cuda_lane_devices_snapshot();
std::uint64_t _test_last_backward_routing_device_switches() noexcept;
#endif

#if VBT_AUTOGRAD_TESTING && VBT_WITH_CUDA
// Test-only hook: observe the current CUDA stream during gradient routing.
// Invoked from the engine drain loop immediately before each add_gradient() call.
using TestRouteHook = void(*)(const Node& producer,
                             const Node& consumer,
                             std::uint64_t current_stream_id);

// Set/clear the global routing hook. Pass nullptr to clear.
void _test_set_route_hook(TestRouteHook hook) noexcept;
#endif

}} // namespace vbt::autograd
