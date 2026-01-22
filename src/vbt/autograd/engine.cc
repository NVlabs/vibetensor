// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/autograd/engine.h"
#include "vbt/autograd/add_ops.h"
#include "vbt/autograd/concurrent_task_queue.h"
#include "vbt/autograd/owner_mailbox.h"
#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/autograd/function.h"
#include "vbt/autograd/wrapper.h"
#include "vbt/autograd/detail/stats_internal.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/forward.h"
#include "vbt/autograd/engine_toggles.h"
#include "vbt/autograd/copy_like.h"
#include "vbt/autograd/lane_routing.h"
#include "vbt/logging/logging.h"
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <thread>
#include <memory>

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <deque>
#include <exception>
#include <type_traits>
#include <cassert>
#include <set>
#include <optional>

#if VBT_WITH_CUDA
#include "vbt/cuda/stream.h"
#include "vbt/cuda/event.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/allocator.h"
#include "vbt/cuda/graphs.h"
#include "vbt/cuda/storage.h"
#endif


namespace vbt { namespace autograd {

using vbt::core::TensorImpl;
using vbt::core::intrusive_ptr;

void GraphTask::InputBuffer::ensure_cpu_capacity(std::size_t n) {
#if !defined(NDEBUG)
  assert(!enqueued);
  assert(received == 0);
  for (uint8_t p : present) {
    assert(p == 0);
  }
#endif
  grads_in.assign(n, OptionalTensor{});
  present.assign(n, 0);
  expected = n;
  // Reset runtime counters in both debug and release builds to keep
  // InputBuffer invariants simple even if a caller violates the
  // "fresh only" precondition.
  received = 0;
  enqueued = false;
}

void GraphTask::InputBuffer::ensure_cuda_capacity(std::size_t n) {
#if !defined(NDEBUG)
  // Must be called only on a structurally fresh buffer.
  assert(!enqueued);
  assert(received == 0);
  for (uint8_t p : present) {
    assert(p == 0);
  }
#endif

  is_accel.assign(n, 0);
  has_accum_stream.assign(n, 0);
  accum_device.assign(n, vbt::core::Device{});
  accum_stream_id.assign(n, 0);

  has_ready_event.assign(n, 0);
  ready_device.assign(n, vbt::core::Device{});
  ready_stream_id.assign(n, 0);

#if VBT_WITH_CUDA
  ready_events.clear();
  ready_events.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    ready_events.emplace_back(false);
  }
#endif
}

static_assert(std::is_empty<Engine>::value,
              "Engine must remain stateless (no data members)");

namespace {

static const bool kLogValidate = [](){ const char* v = std::getenv("VBT_LOG_AUTOGRAD_VALIDATE"); return v && std::strcmp(v, "1") == 0; }();
static const bool kLogEngine   = [](){ const char* v = std::getenv("VBT_LOG_AUTOGRAD_ENGINE");   return v && std::strcmp(v, "1") == 0; }();

//
// cross-thread callback mixing. Nested backward is rejected separately.
//
// Note: we take this gate in a non-blocking mode and throw if another backward
// is in progress to avoid deadlock patterns (e.g., a thread created inside a
// backward trying to start another backward while the owner waits for it).
static std::mutex g_backward_gate_mu;

static inline unsigned default_cpu_worker_threads() noexcept {
  unsigned hc = std::thread::hardware_concurrency();
  if (hc == 0) hc = 4;
  // Keep the pool modest to avoid oversubscription in unit tests.
  return std::max(1u, std::min(4u, hc));
}

// Use a heap-allocated std::function to avoid running nanobind::object
// destructors after the Python interpreter has shut down. This
// intentionally leaks a tiny amount of state at process exit but keeps
// test runs and interactive sessions stable.
static BackwardCompleteCallback* g_backward_complete_cb = nullptr;

static BackwardCompleteCallback& get_backward_complete_cb_storage() {
  if (!g_backward_complete_cb) {
    g_backward_complete_cb = new BackwardCompleteCallback();
  }
  return *g_backward_complete_cb;
}

// nodes_in_graph for a backward run without affecting scheduling.
struct GraphBuildResult {
  std::vector<Node*> topo; // reverse DFS post-order
};

enum class VisitState : std::uint8_t { kWhite, kGray, kBlack };

static GraphBuildResult build_graph_topology(GraphTask& gt,
                                             const intrusive_ptr<Node>& root) {
  GraphBuildResult result;

  gt.dependencies.clear();
  gt.nodes_in_graph.clear();

  Node* root_raw = root.get();
  if (!root_raw) {
    return result;
  }

  std::unordered_map<Node*, VisitState> visit;

  struct Frame {
    Node* node;
    std::size_t edge_idx;
  };

  std::vector<Frame> stack;
  stack.push_back({root_raw, 0});

  while (!stack.empty()) {
    Frame& f = stack.back();
    Node* n = f.node;

    auto it_state = visit.find(n);
    VisitState state = (it_state == visit.end()) ? VisitState::kWhite : it_state->second;

    if (state == VisitState::kBlack) {
      stack.pop_back();
      continue;
    }

    if (state == VisitState::kWhite) {
      visit[n] = VisitState::kGray;
      gt.nodes_in_graph.insert(n);
      gt.dependencies.emplace(n, 0);
    }

    auto& edges = n->next_edges;
    if (f.edge_idx < edges.size()) {
      const Edge& e = edges[f.edge_idx++];
      if (!e.fn) {
        continue;
      }
      Node* consumer = e.fn.get();

      auto it_consumer_state = visit.find(consumer);
      VisitState consumer_state = (it_consumer_state == visit.end()) ? VisitState::kWhite : it_consumer_state->second;

      if (consumer_state == VisitState::kGray) {
        throw std::logic_error("engine: detected cycle in autograd graph");
      }

      auto dep_it_pair = gt.dependencies.emplace(consumer, 0);
      ++dep_it_pair.first->second; // bump E(consumer)

      if (consumer_state == VisitState::kWhite) {
        stack.push_back({consumer, 0});
      }
      continue;
    }

    // All edges processed for this node.
    visit[n] = VisitState::kBlack;
    result.topo.push_back(n);
    stack.pop_back();
  }

  // Normalize root entry: ensure it exists and has E(root) == 0.
  auto it_root_dep = gt.dependencies.find(root_raw);
  if (it_root_dep == gt.dependencies.end()) {
    gt.dependencies.emplace(root_raw, 0);
  } else {
    it_root_dep->second = 0;
  }

  // Reverse DFS post-order to match design: topo is in
  // reverse post-order (root first for simple chains).
  std::reverse(result.topo.begin(), result.topo.end());

  return result;
}

static void exact_add_inplace(TensorImpl& acc, const TensorImpl& addend) {
  if (acc.device().type != addend.device().type || acc.device().index != addend.device().index) {
    throw std::logic_error("engine coalesce_add: device mismatch");
  }
  if (acc.dtype() != addend.dtype()) {
    throw std::logic_error("engine coalesce_add: dtype mismatch");
  }
  if (acc.sizes() != addend.sizes() || acc.strides() != addend.strides() || !acc.is_non_overlapping_and_dense() || !addend.is_non_overlapping_and_dense()) {
    throw std::logic_error("engine coalesce_add: metadata mismatch or non-dense");
  }
  const auto N = acc.numel();
  if (N == 0) return;
  if (acc.device().type != kDLCPU) {
    throw std::runtime_error("engine coalesce_add: CUDA stub");
  }
  if (acc.dtype() != vbt::core::ScalarType::Float32) {
    throw std::runtime_error("engine coalesce_add: only Float32 CPU supported");
  }
  float* a = static_cast<float*>(acc.data());
  const float* b = static_cast<const float*>(addend.data());
  for (int64_t i = 0; i < N; ++i) a[i] += b[i];
}


static GraphTask::InputBuffer& ensure_buffer(GraphTask& gt, Node* n, const intrusive_ptr<Node>& keep) {
  auto it = gt.inputs.find(n);
  if (it == gt.inputs.end()) {
    GraphTask::InputBuffer buf;
    std::size_t expected = static_cast<std::size_t>(n->num_incoming_grad_slots());
    // single incoming gradient slot even if they produce multiple input
    // gradients. Nodes that need this behavior override
    // num_incoming_grad_slots() to return 1, which keeps intermediate
    // nodes like MulBackward scheduled correctly when they sit under
    // another wrapper node (e.g., ReluBackward).
    buf.ensure_cpu_capacity(expected);
#if VBT_WITH_CUDA
    // If autograd_device has already been derived as CUDA, eagerly allocate
    // CUDA metadata while the buffer is structurally fresh.
    if (gt.has_autograd_device && gt.autograd_device.type == kDLCUDA) {
      buf.ensure_cuda_capacity(expected);
    }
#endif
    auto [it2, ok] = gt.inputs.emplace(n, std::move(buf));
    (void)ok;
    gt.keepalive.emplace(n, keep);
    return it2->second;
  }
  return it->second;
}

static void coalesce_incoming(GraphTask& gt, Node* dst, std::size_t input_nr, OptionalTensor grad, const intrusive_ptr<Node>& keep_dst) {
  GraphTask::InputBuffer& ib = ensure_buffer(gt, dst, keep_dst);
  if (input_nr >= ib.expected) {
    throw std::out_of_range("engine: incoming slot out of range");
  }
  OptionalTensor& slot = ib.grads_in[input_nr];
  if (ib.present[input_nr]) {
    // Duplicate arrival on the same slot
    if (grad.has_value()) {
      if (!slot.has_value()) {
        slot = std::move(grad); // promote defined over present-only
      } else {
        TensorImpl& acc = slot.value();
        const TensorImpl& add = grad.value();
        exact_add_inplace(acc, add);
        gt.duplicates_coalesced++;
      }
    }
    // present flag stays set
  } else {
    // First arrival
    ib.present[input_nr] = 1;
    ib.received += 1;
    if (grad.has_value()) {
      slot = std::move(grad);
    }
    if (!ib.enqueued && ib.received == ib.expected) {
      ib.enqueued = true;
      gt.ready.push_back(gt.keepalive[dst]);
    }
  }
}

static bool is_cpu_empty_cuda_sentinel_for_slot(const Node& node,
                                                std::size_t pos,
                                                const TensorImpl& g);

// MultiDeviceExperimental mode.
static vbt::core::Device expected_device_for_slot_multidevice(
    const Node& consumer,
    std::size_t pos);

[[noreturn]] static void throw_explicit_copy_required(
    const char* kind,
    const Node* producer,
    const Node& consumer,
    std::size_t pos,
    const vbt::core::Device& expected,
    const vbt::core::Device& got);

struct AddContext {
  GraphTask&              gt;            // owning GraphTask
  Node&                   consumer;      // consumer node (non-null)
  GraphTask::InputBuffer& buf;           // consumer buffer
  std::size_t             pos;           // slot index in [0, buf.expected)
  intrusive_ptr<Node>     consumer_keep; // used for ready-queue pushes
};

static void add_gradient_cpu(AddContext& ctx, OptionalTensor&& grad, bool schedule_by_slots);
#if VBT_WITH_CUDA
static void add_gradient_cuda_device(AddContext& ctx,
                                    OptionalTensor&& grad,
                                    const vbt::core::Device& slot_device,
                                    bool schedule_by_slots);
static void add_gradient_cuda(AddContext& ctx, OptionalTensor&& grad, bool schedule_by_slots);
#endif

// Gradient routing entrypoint: shared prelude then CPU/CUDA branches.
static void add_gradient(AddContext& ctx, OptionalTensor&& grad, bool schedule_by_slots) {
  GraphTask&              gt  = ctx.gt;
  GraphTask::InputBuffer& ib  = ctx.buf;
  const std::size_t       pos = ctx.pos;

  if (ib.grads_in.size() != ib.expected || ib.present.size() != ib.expected) {
    const std::string node_name = ctx.consumer.name.empty() ? std::string("<unnamed>")
                                                           : ctx.consumer.name;
    throw std::logic_error(
        "engine: InputBuffer size mismatch (node=" + node_name +
        ", pos=" + std::to_string(pos) +
        ", expected=" + std::to_string(ib.expected) +
        ", grads_in.size=" + std::to_string(ib.grads_in.size()) +
        ", present.size=" + std::to_string(ib.present.size()) + ")");
  }

  if (pos >= ib.expected) {
    throw std::out_of_range("engine: incoming slot out of range");
  }

  if (!gt.has_autograd_device) {
    throw std::logic_error(
        "add_gradient: autograd_device must be set before routing gradients");
  }

  if (gt.device_mode_snapshot == AutogradDeviceMode::SingleDevice) {
    if (grad.has_value()) {
      TensorImpl& g = grad.value();
      auto g_dev = g.device();

      if (gt.autograd_device.type == kDLCUDA &&
          g_dev.type == kDLCPU &&
          is_cpu_empty_cuda_sentinel_for_slot(ctx.consumer, pos, g)) {
        // Sentinel: counts for device derivation, but is treated as nullopt for
        // routing so we never store it in grads_in.
        grad.reset();
      } else if (gt.autograd_device.type == kDLCPU && g_dev.type == kDLCUDA) {
        throw std::runtime_error(
            "VibeTensor autograd: CUDA gradients are not allowed when autograd_device is CPU");
      } else if (gt.autograd_device.type == kDLCUDA &&
                 (g_dev.type != gt.autograd_device.type ||
                  g_dev.index != gt.autograd_device.index)) {
        throw std::runtime_error(
            "VibeTensor autograd: gradient device does not match this backward's autograd device");
      }
    }

    // First-arrival bookkeeping shared by CPU and CUDA paths.
    if (!ib.present[pos]) {
      ib.present[pos] = 1;
      ++ib.received;
    }

    if (gt.autograd_device.type != kDLCUDA) {
      return add_gradient_cpu(ctx, std::move(grad), schedule_by_slots);
    }

#if VBT_WITH_CUDA
    return add_gradient_cuda(ctx, std::move(grad), schedule_by_slots);
#else
    throw std::runtime_error(
        "VibeTensor CUDA autograd: autograd_device is CUDA but build has no CUDA support");
#endif
  }

  // gradient against the expected device for this consumer slot.
  const vbt::core::Device expected =
      expected_device_for_slot_multidevice(ctx.consumer, pos);

  if (grad.has_value()) {
    TensorImpl& g = grad.value();
    const vbt::core::Device got = g.device();

    if (expected.type == kDLCUDA &&
        got.type == kDLCPU &&
        is_cpu_empty_cuda_sentinel_for_slot(ctx.consumer, pos, g)) {
      // Sentinel: counts for device derivation, but is treated as nullopt for
      // routing so we never store it in grads_in.
      grad.reset();
    } else if (got != expected) {
      throw_explicit_copy_required(
          "consumer_mismatch", /*producer=*/nullptr, ctx.consumer, pos, expected, got);
    }
  }

  // First-arrival bookkeeping shared by CPU and CUDA paths.
  if (!ib.present[pos]) {
    ib.present[pos] = 1;
    ++ib.received;
  }

  if (expected.type != kDLCUDA) {
    return add_gradient_cpu(ctx, std::move(grad), schedule_by_slots);
  }

#if VBT_WITH_CUDA
  return add_gradient_cuda_device(ctx, std::move(grad), expected, schedule_by_slots);
#else
  throw std::runtime_error(
      "VibeTensor CUDA autograd: expected CUDA slot but build has no CUDA support");
#endif
}

using DeviceTypeCode = std::int32_t;              // underlying DLDeviceType
using DeviceKey      = std::pair<DeviceTypeCode, int>; // (type, index)

static vbt::core::Device device_from_key(const DeviceKey& key) {
  DLDeviceType t = static_cast<DLDeviceType>(key.first);
  int          i = key.second;
  if (t == kDLCUDA) {
    return vbt::core::Device::cuda(i);
  }
  return vbt::core::Device{t, i};
}

static bool is_cpu_empty_cuda_sentinel_for_slot(const Node& node,
                                                std::size_t pos,
                                                const TensorImpl& g) {
  auto* v = dynamic_cast<const ValidatableNode*>(&node);
  if (!v) return false;
  const auto& metas = v->input_metas();
  if (pos >= metas.size()) return false;
  const auto& m = metas[pos];
  if (m.device.type != kDLCUDA) return false;
  if (g.device().type != kDLCPU) return false;
  if (g.numel() != 0) return false;
  if (g.dtype() != m.dtype) return false;
  return true;
}

static vbt::core::Device expected_device_for_slot_multidevice(
    const Node& consumer,
    std::size_t pos) {
  (void)pos;

  if (consumer.stream_kind() != StreamKind::CudaAllowlisted) {
    return vbt::core::Device::cpu(0);
  }

  const NodeStreamInfo& si = consumer.stream_info();
  if (!si.has_canonical_stream || si.device.type != kDLCUDA) {
    throw std::runtime_error(
        "VibeTensor CUDA autograd internal error: missing or mismatched canonical stream on CUDA node");
  }

  return si.device;
}

[[noreturn]] static void throw_explicit_copy_required(
    const char* kind,
    const Node* producer,
    const Node& consumer,
    std::size_t pos,
    const vbt::core::Device& expected,
    const vbt::core::Device& got) {
  const std::string consumer_name =
      consumer.name.empty() ? std::string("<unnamed>") : consumer.name;
  const std::string producer_name = [&]() {
    if (!producer) {
      return std::string("<null>");
    }
    return producer->name.empty() ? std::string("<unnamed>") : producer->name;
  }();

  throw std::runtime_error(
      std::string("VibeTensor autograd: explicit copy node is required") +
      " (kind=" + std::string(kind) +
      ", producer=" + producer_name +
      ", consumer=" + consumer_name +
      ", pos=" + std::to_string(pos) +
      ", expected=" + expected.to_string() +
      ", got=" + got.to_string() + ")");
}

static vbt::core::Device derive_autograd_device(
    const intrusive_ptr<Node>& root,
    const std::vector<OptionalTensor>& initial_grads) {
  std::set<DeviceKey> logical_devices;

  Node* root_raw = root.get();
  auto* v = dynamic_cast<const ValidatableNode*>(root_raw);
  const auto* metas = v ? &v->input_metas() : nullptr;

  // 1. Collect logical devices from defined initial grads.
  for (std::size_t i = 0; i < initial_grads.size(); ++i) {
    if (!initial_grads[i].has_value()) continue;
    const TensorImpl& g = initial_grads[i].value();

    if (metas && is_cpu_empty_cuda_sentinel_for_slot(*root_raw, i, g)) {
      const auto& m = (*metas)[i];
      logical_devices.emplace(
          static_cast<DeviceTypeCode>(m.device.type), m.device.index);
      continue;
    }

    auto d = g.device();
    logical_devices.emplace(static_cast<DeviceTypeCode>(d.type), d.index);
  }

  if (logical_devices.size() > 1) {
    throw std::invalid_argument("VibeTensor autograd: single-device backward only");
  }

  if (logical_devices.empty()) {
    // Fallback to root metadata when available.
    if (metas) {
      std::set<DeviceKey> meta_devices;
      for (const auto& m : *metas) {
        meta_devices.emplace(
            static_cast<DeviceTypeCode>(m.device.type), m.device.index);
      }
      if (meta_devices.size() > 1) {
        throw std::invalid_argument(
            "VibeTensor autograd: single-device backward only");
      }
      if (!meta_devices.empty()) {
        return device_from_key(*meta_devices.begin());
      }
    }
    // Generic root: default to CPU:0.
    return vbt::core::Device::cpu(0);
  }

  return device_from_key(*logical_devices.begin());
}

static void seed_root_input_buffer(GraphTask& gt,
                                   const intrusive_ptr<Node>& root,
                                   const std::vector<OptionalTensor>& initial_grads,
                                   bool device_is_cuda) {
  Node* root_raw = root.get();
  if (!root_raw) {
    throw std::invalid_argument("engine: backward root must not be null");
  }

  const std::size_t expected_root = initial_grads.size();
  const std::size_t num_inputs = static_cast<std::size_t>(root_raw->num_inputs());
  if (expected_root != num_inputs) {
    throw std::invalid_argument("engine: initial_grads.size() must equal root->num_inputs()");
  }

  GraphTask::InputBuffer& root_ib = ensure_buffer(gt, root_raw, root);
#if !defined(NDEBUG)
  // Buffer must be fresh here: only ensure_buffer has touched it.
  assert(root_ib.received == 0);
  assert(!root_ib.enqueued);
  for (uint8_t p : root_ib.present) {
    assert(p == 0);
  }
#endif

  root_ib.ensure_cpu_capacity(expected_root);

#if VBT_WITH_CUDA
  if (device_is_cuda) {
    if (root_ib.is_accel.size() != expected_root) {
      root_ib.ensure_cuda_capacity(expected_root);
    }
  }
#else
  (void)device_is_cuda;
#endif

  if (!device_is_cuda) {
    for (std::size_t i = 0; i < expected_root; ++i) {
      root_ib.grads_in[i] = initial_grads[i];
      root_ib.present[i] = 1;
    }
    root_ib.received = expected_root;
  } else {
    Node* root_node = root_raw;
    auto* v = dynamic_cast<const ValidatableNode*>(root_node);
    const auto* metas = v ? &v->input_metas() : nullptr;

    for (std::size_t i = 0; i < expected_root; ++i) {
      root_ib.present[i] = 1;

      if (!initial_grads[i].has_value()) {
        root_ib.grads_in[i] = OptionalTensor{};
        continue;
      }

      const TensorImpl& g = initial_grads[i].value();
      if (metas && is_cpu_empty_cuda_sentinel_for_slot(*root_node, i, g)) {
        // Sentinel: logically present but treated as zero; do not store in grads_in.
        root_ib.grads_in[i] = OptionalTensor{};
        continue;
      }

      root_ib.grads_in[i] = initial_grads[i];
    }
    root_ib.received = expected_root;
  }

  root_ib.enqueued = true;
  gt.ready.push_back(root);

#if !defined(NDEBUG)
  // Root buffer invariants after seeding.
  assert(root_ib.expected == expected_root);
  assert(root_ib.grads_in.size() == expected_root);
  assert(root_ib.present.size() == expected_root);
  assert(root_ib.received == expected_root);
  assert(root_ib.enqueued);
#endif
}

static void seed_root_input_buffer_multidevice(
    GraphTask& gt,
    const intrusive_ptr<Node>& root,
    const std::vector<OptionalTensor>& initial_grads) {
  Node* root_raw = root.get();
  if (!root_raw) {
    throw std::invalid_argument("engine: backward root must not be null");
  }

  const std::size_t expected_root = initial_grads.size();
  const std::size_t num_inputs = static_cast<std::size_t>(root_raw->num_inputs());
  if (expected_root != num_inputs) {
    throw std::invalid_argument("engine: initial_grads.size() must equal root->num_inputs()");
  }

  GraphTask::InputBuffer& root_ib = ensure_buffer(gt, root_raw, root);
#if !defined(NDEBUG)
  // Buffer must be fresh here: only ensure_buffer has touched it.
  assert(root_ib.received == 0);
  assert(!root_ib.enqueued);
  for (uint8_t p : root_ib.present) {
    assert(p == 0);
  }
#endif

  root_ib.ensure_cpu_capacity(expected_root);

#if VBT_WITH_CUDA
  // If this backward participates in CUDA, seed a buffer with CUDA metadata
  // capacity so prepare_consumer_stream_and_wait_mt can use per-slot state.
  if (gt.autograd_device.type == kDLCUDA) {
    if (root_ib.is_accel.size() != expected_root) {
      root_ib.ensure_cuda_capacity(expected_root);
    }
  }
#endif

  for (std::size_t i = 0; i < expected_root; ++i) {
    root_ib.present[i] = 1;

    if (!initial_grads[i].has_value()) {
      root_ib.grads_in[i] = OptionalTensor{};
      continue;
    }

    const TensorImpl& g = initial_grads[i].value();
    const vbt::core::Device expected = expected_device_for_slot_multidevice(*root_raw, i);

    if (expected.type == kDLCUDA &&
        g.device().type == kDLCPU &&
        is_cpu_empty_cuda_sentinel_for_slot(*root_raw, i, g)) {
      // Sentinel: logically present but treated as zero; do not store in grads_in.
      root_ib.grads_in[i] = OptionalTensor{};
      continue;
    }

    if (g.device() != expected) {
      throw_explicit_copy_required(
          "seed_mismatch", /*producer=*/nullptr, *root_raw, i, expected, g.device());
    }

    root_ib.grads_in[i] = initial_grads[i];
  }

  root_ib.received = expected_root;
  root_ib.enqueued = true;
  gt.ready.push_back(root);

#if !defined(NDEBUG)
  // Root buffer invariants after seeding.
  assert(root_ib.expected == expected_root);
  assert(root_ib.grads_in.size() == expected_root);
  assert(root_ib.present.size() == expected_root);
  assert(root_ib.received == expected_root);
  assert(root_ib.enqueued);
#endif
}

static void derive_autograd_device_and_seed_root(
    GraphTask& gt,
    const intrusive_ptr<Node>& root,
    const std::vector<OptionalTensor>& initial_grads,
    bool mt_enabled_snapshot) {
  if (gt.device_mode_snapshot == AutogradDeviceMode::SingleDevice) {
    vbt::core::Device dev = derive_autograd_device(root, initial_grads);
    gt.autograd_device = dev;
    gt.has_autograd_device = true;

    const bool device_is_cuda = (dev.type == kDLCUDA);
    seed_root_input_buffer(gt, root, initial_grads, device_is_cuda);
    return;
  }

  // Allow CPU:0 plus multiple CUDA device indices.
  std::set<int> cuda_indices;

  Node* root_raw = root.get();
  auto* v = dynamic_cast<const ValidatableNode*>(root_raw);
  const auto* metas = v ? &v->input_metas() : nullptr;

  // 1) Collect CUDA indices from defined initial grads.
  for (std::size_t i = 0; i < initial_grads.size(); ++i) {
    if (!initial_grads[i].has_value()) continue;
    const TensorImpl& g = initial_grads[i].value();

    if (metas && is_cpu_empty_cuda_sentinel_for_slot(*root_raw, i, g)) {
      const auto& m = (*metas)[i];
      if (m.device.type == kDLCUDA) {
        cuda_indices.insert(m.device.index);
      }
      continue;
    }

    const auto d = g.device();
    if (d.type == kDLCUDA) {
      cuda_indices.insert(d.index);
    }
  }

  // 2) Collect CUDA indices from graph nodes.
  for (Node* n : gt.nodes_in_graph) {
    if (!n) continue;
    if (n->stream_kind() == StreamKind::CudaAllowlisted) {
      const NodeStreamInfo& si = n->stream_info();
      cuda_indices.insert(static_cast<int>(si.device.index));
    }
  }

  gt.cuda_devices_snapshot.assign(cuda_indices.begin(), cuda_indices.end());

  std::set<int> cuda_lane_indices;
  for (Node* n : gt.nodes_in_graph) {
    if (!n) continue;
    // Lane routing in MultiDeviceExperimental excludes AccumulateGrad.
    if (n->stream_kind() == StreamKind::CudaAllowlisted &&
        dynamic_cast<const AccumulateGrad*>(n) == nullptr) {
      const NodeStreamInfo& si = n->stream_info();
      cuda_lane_indices.insert(static_cast<int>(si.device.index));
    }
  }
  gt.cuda_lane_devices_snapshot.assign(cuda_lane_indices.begin(), cuda_lane_indices.end());

  const bool cuda_participates = !cuda_indices.empty();

  // MultiDeviceExperimental with CUDA participation must use the deps-based
  // multithreaded scheduler.
  if (cuda_participates && !mt_enabled_snapshot) {
    throw std::runtime_error(
        "VibeTensor autograd: MultiDeviceExperimental requires multithreading to be enabled");
  }

  if (cuda_participates) {
#if VBT_WITH_CUDA
    const int ndev = vbt::cuda::device_count();
    for (int idx : cuda_indices) {
      if (idx < 0 || idx >= ndev) {
        throw std::invalid_argument(
            "VibeTensor autograd: MultiDeviceExperimental invalid CUDA device index " +
            std::to_string(idx));
      }
    }

    // Primary CUDA device: used only for legacy branching/lane routing.
    const int primary = *cuda_indices.begin();
    gt.autograd_device = vbt::core::Device::cuda(primary);
    gt.has_autograd_device = true;
#else
    throw std::runtime_error(
        "VibeTensor CUDA autograd: graph requires CUDA but build has no CUDA support");
#endif
  } else {
    gt.autograd_device = vbt::core::Device::cpu(0);
    gt.has_autograd_device = true;
  }

  seed_root_input_buffer_multidevice(gt, root, initial_grads);
}

#if VBT_WITH_CUDA
static void enforce_copy_and_maybe_fence_multidevice(
    GraphTask& gt,
    const Node& producer,
    const Node& consumer,
    std::size_t pos,
    OptionalTensor* grad,
    std::vector<int>* fenced_dst_cuda_indices) {
  if (gt.device_mode_snapshot != AutogradDeviceMode::MultiDeviceExperimental) {
    return;
  }

  const vbt::core::Device expected = expected_device_for_slot_multidevice(consumer, pos);

  // Sentinel rewrite (consumer-side): treat CPU empty tensors as nullopt for
  // CUDA-expected slots.
  if (grad && grad->has_value()) {
    TensorImpl& g = grad->value();
    const vbt::core::Device got = g.device();

    if (expected.type == kDLCUDA &&
        got.type == kDLCPU &&
        is_cpu_empty_cuda_sentinel_for_slot(consumer, pos, g)) {
      grad->reset();
      return;
    }
  }

  if (!grad || !grad->has_value()) {
    return;
  }

  const TensorImpl& g = grad->value();
  const vbt::core::Device got = g.device();

  // Consumer mismatch: do not fence; add_gradient will throw a copy-required user error.
  if (got != expected) {
    return;
  }

  const vbt::core::Device producer_exec =
      (producer.stream_kind() == StreamKind::CudaAllowlisted)
          ? producer.stream_info().device
          : vbt::core::Device::cpu(0);

  // Producer cross-device CUDA output check + conservative fencing.
  if (got.type == kDLCUDA && producer_exec != got) {
    const bool is_copy_like = (dynamic_cast<const CopyLikeNode*>(&producer) != nullptr);
    if (!is_copy_like) {
      throw_explicit_copy_required(
          "producer_cross_device_cuda", &producer, consumer, pos, expected, got);
    }

    // Fence on the destination device, once per (producer completion × dst device).
    if (fenced_dst_cuda_indices) {
      for (int d : *fenced_dst_cuda_indices) {
        if (d == got.index) {
          return;
        }
      }
      fenced_dst_cuda_indices->push_back(got.index);
    }

    // NOTE: Correctness-first fence. We assume that any CopyLikeNode producing
    // a CUDA gradient on a different device enqueues the work onto the
    // destination device such that synchronizing the destination device is
    // sufficient before routing.
    const auto dst_idx = static_cast<vbt::cuda::DeviceIndex>(got.index);
    vbt::cuda::DeviceGuard dg(dst_idx);
    (void)cudaDeviceSynchronize();
    ++gt.cuda_device_synchronizes;
  }
}
#endif

static void add_gradient_cpu(AddContext& ctx, OptionalTensor&& grad, bool schedule_by_slots) {
  GraphTask&              gt  = ctx.gt;
  GraphTask::InputBuffer& ib  = ctx.buf;
  const std::size_t       pos = ctx.pos;
  OptionalTensor&         slot = ib.grads_in[pos];

  const bool is_defined = grad.has_value();

  if (!is_defined) {
    // nullopt arrival: counters were already updated; nothing more to do.
    if (schedule_by_slots && !ib.enqueued && ib.received == ib.expected) {
      ib.enqueued = true;
      gt.ready.push_back(ctx.consumer_keep);
    }
    return;
  }

  if (slot.has_value()) {
    // Duplicate defined arrival: accumulate.
    autograd_add_inplace_dense(slot.value(), grad.value(), slot.value().device());
    ++gt.duplicates_coalesced;
  } else {
    // First defined grad or defined after a previous nullopt.
    slot = std::move(grad);
  }

  if (schedule_by_slots && !ib.enqueued && ib.received == ib.expected) {
    ib.enqueued = true;
    gt.ready.push_back(ctx.consumer_keep);
  }
}

#if VBT_WITH_CUDA

static vbt::cuda::Stream make_stream_from_id(
    std::uint64_t id, vbt::cuda::DeviceIndex dev_idx) {
  return vbt::cuda::Stream(vbt::cuda::Stream::UNCHECKED, id, dev_idx);
}

// CUDA branch of add_gradient implementing the streaming state machine.
//
// GraphTask::autograd_device; the wrapper below preserves SingleDevice behavior.
//
// Precondition: when grad is defined, its device must match slot_device.
// In SingleDevice mode this is enforced by add_gradient(...) before dispatch.
static void add_gradient_cuda_device(AddContext& ctx,
                                    OptionalTensor&& grad,
                                    const vbt::core::Device& slot_device,
                                    bool schedule_by_slots) {
  GraphTask&              gt  = ctx.gt;
  GraphTask::InputBuffer& ib  = ctx.buf;
  const std::size_t       pos = ctx.pos;
  Node&                   consumer = ctx.consumer;

  // Ensure CUDA metadata is sized.
  if (ib.is_accel.size() != ib.expected) {
    ib.ensure_cuda_capacity(ib.expected);
  }

  const bool is_defined = grad.has_value();

  // 1. nullopt arrivals: CPU counters only.
  if (!is_defined) {
    if (schedule_by_slots && !ib.enqueued && ib.received == ib.expected) {
      ib.enqueued = true;
      gt.ready.push_back(ctx.consumer_keep);
    }
    return;
  }

  // 2. Defined CUDA arrivals.
  if (consumer.stream_kind() == StreamKind::CpuOnly) {
    throw std::runtime_error(
        "VibeTensor CUDA autograd internal error: CpuOnly node received CUDA gradient");
  }

  const NodeStreamInfo& si = consumer.stream_info();
  if (!si.has_canonical_stream || si.device.type != kDLCUDA ||
      si.device.index != slot_device.index ||
      slot_device.type != kDLCUDA) {
    throw std::runtime_error(
        "VibeTensor CUDA autograd internal error: missing or mismatched canonical stream on CUDA node");
  }

  OptionalTensor& slot = ib.grads_in[pos];

  if (ib.has_accum_stream[pos] && ib.accum_device[pos] != slot_device) {
    throw std::runtime_error(
        "VibeTensor CUDA autograd internal error: attempted cross-device CUDA accumulation");
  }

  // Case A: first defined arrival for this slot.
  if (!slot.has_value()) {
    slot = std::move(grad);  // first defined grad wins

    if (!ib.is_accel[pos]) {
      ib.is_accel[pos] = 1;
    }

    if (!ib.has_accum_stream[pos]) {
      const auto dev_idx = static_cast<vbt::cuda::DeviceIndex>(slot_device.index);
      vbt::cuda::Stream S_acc = make_stream_from_id(si.stream_id, dev_idx);
      ib.has_accum_stream[pos] = 1;
      ib.accum_device[pos]     = slot_device;
      ib.accum_stream_id[pos]  = S_acc.id();

      vbt::cuda::Stream S_prod = vbt::cuda::getCurrentStream(dev_idx);
      if (S_prod.id() != S_acc.id()) {
        vbt::cuda::Event e(false);
        e.record(S_prod);
        ++gt.cuda_events_recorded;
        e.wait(S_acc);
        ++gt.cuda_events_waited;
        ++gt.cuda_cross_stream_routes;
      }

      // Record readiness for this slot.
      ib.ready_events[pos].record(S_acc);
      ib.has_ready_event[pos] = 1;
      ib.ready_device[pos]    = slot_device;
      ib.ready_stream_id[pos] = S_acc.id();
      ++gt.cuda_events_recorded;
    }

    if (schedule_by_slots && !ib.enqueued && ib.received == ib.expected) {
      ib.enqueued = true;
      gt.ready.push_back(ctx.consumer_keep);
    }
    return;
  }

  // Case B: duplicate defined arrival (Nth arrival).
  if (!ib.has_accum_stream[pos]) {
    // Defensive: initialize accum stream from canonical info.
    const auto dev_idx = static_cast<vbt::cuda::DeviceIndex>(slot_device.index);
    vbt::cuda::Stream S_acc = make_stream_from_id(si.stream_id, dev_idx);
    ib.has_accum_stream[pos] = 1;
    ib.accum_device[pos]     = slot_device;
    ib.accum_stream_id[pos]  = S_acc.id();
  }

  const auto dev_idx = static_cast<vbt::cuda::DeviceIndex>(slot_device.index);
  vbt::cuda::Stream S_acc  = make_stream_from_id(ib.accum_stream_id[pos], dev_idx);
  vbt::cuda::Stream S_prod = vbt::cuda::getCurrentStream(dev_idx);

  if (S_prod.id() != S_acc.id()) {
    vbt::cuda::Event e(false);
    e.record(S_prod);
    ++gt.cuda_events_recorded;
    e.wait(S_acc);
    ++gt.cuda_events_waited;
    ++gt.cuda_cross_stream_routes;
  }

  if (ib.has_ready_event[pos] && ib.ready_stream_id[pos] != S_acc.id()) {
    if (ib.ready_device[pos] != slot_device) {
      throw std::runtime_error(
          "VibeTensor CUDA autograd internal error: attempted cross-device CUDA event wait");
    }
    ib.ready_events[pos].wait(S_acc);
    ++gt.cuda_events_waited;
    ++gt.cuda_cross_stream_routes;
  }

  {
    vbt::cuda::DeviceGuard dg(dev_idx);
    vbt::cuda::CUDAStreamGuard guard(S_acc);
    autograd_add_inplace_dense(slot.value(), grad.value(), slot_device);
    ++gt.duplicates_coalesced;
  }

  ib.ready_events[pos].record(S_acc);
  ib.has_ready_event[pos] = 1;
  ib.ready_device[pos]    = slot_device;
  ib.ready_stream_id[pos] = S_acc.id();
  ++gt.cuda_events_recorded;

  if (schedule_by_slots && !ib.enqueued && ib.received == ib.expected) {
    ib.enqueued = true;
    gt.ready.push_back(ctx.consumer_keep);
  }
}

static void add_gradient_cuda(AddContext& ctx, OptionalTensor&& grad, bool schedule_by_slots) {
  return add_gradient_cuda_device(ctx, std::move(grad), ctx.gt.autograd_device, schedule_by_slots);
}

static void record_stream_for_grad(const TensorImpl& t,
                                   const vbt::cuda::Stream& s) {
  if (t.device().type != kDLCUDA) return;
  vbt::cuda::record_stream(t.storage(), s);
}

struct MtCudaWaitDeltas {
  std::uint64_t waited{0};
  std::uint64_t cross{0};
};

// Pointer-based counter sink shared between ST and MT consumer prep helpers.
//
// Thread-ownership contract:
// - ST: pointers target GraphTask counters and are mutated only on the owner thread.
// - MT: pointers target thread-local deltas; worker threads must not mutate GraphTask.
struct CudaWaitCounters {
  std::uint64_t* waited{nullptr};
  std::uint64_t* cross{nullptr};
  inline void on_cross_stream_wait() noexcept {
    if (waited) ++*waited;
    if (cross)  ++*cross;
  }
};

static std::optional<vbt::cuda::CUDAStreamGuard>
prepare_consumer_stream_and_wait_device(const vbt::core::Device& consumer_device,
                                        Node& consumer,
                                        GraphTask::InputBuffer* ib,
                                        std::vector<OptionalTensor>& grads_in,
                                        CudaWaitCounters counters) {
  if (consumer_device.type != kDLCUDA) {
    return std::nullopt;
  }

  if (!ib) {
    // No InputBuffer for this node; still need to tag grads with canonical stream.
    if (consumer.stream_kind() != StreamKind::CudaAllowlisted) {
      return std::nullopt;
    }
  }

  if (consumer.stream_kind() == StreamKind::CpuOnly) {
    if (ib) {
      for (std::size_t i = 0; i < ib->expected; ++i) {
        if (ib->is_accel.size() == ib->expected && ib->is_accel[i]) {
          throw std::runtime_error(
              "VibeTensor CUDA autograd internal error: CpuOnly consumer has accelerated slots");
        }
      }
    }
    return std::nullopt;
  }

  const NodeStreamInfo& si = consumer.stream_info();
  if (!si.has_canonical_stream || si.device.type != kDLCUDA ||
      si.device.index != consumer_device.index) {
    throw std::runtime_error(
        "VibeTensor CUDA autograd internal error: missing or mismatched canonical stream on CUDA node");
  }

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;

  const auto dev_idx = static_cast<DeviceIndex>(consumer_device.index);
  Stream S_cons = Stream(Stream::UNCHECKED, si.stream_id, dev_idx);

  // Wait on any ready events from InputBuffer.
  if (ib && ib->has_ready_event.size() == ib->expected) {
    for (std::size_t pos = 0; pos < ib->expected; ++pos) {
      if (!ib->has_ready_event[pos]) continue;
      if (ib->ready_stream_id[pos] != S_cons.id()) {
        if (ib->ready_device[pos] != consumer_device) {
          throw std::runtime_error(
              "VibeTensor CUDA autograd internal error: attempted cross-device CUDA event wait");
        }
        ib->ready_events[pos].wait(S_cons);
        counters.on_cross_stream_wait();
      }
    }
  }

  // Tag gradients with the consumer stream.
  for (auto& og : grads_in) {
    if (!og.has_value()) continue;
    record_stream_for_grad(og.value(), S_cons);
  }

  return std::optional<vbt::cuda::CUDAStreamGuard>(std::in_place, S_cons);
}

static std::optional<vbt::cuda::CUDAStreamGuard>
prepare_consumer_stream_and_wait_mt(const vbt::core::Device& consumer_device,
                                    Node& consumer,
                                    GraphTask::InputBuffer* ib,
                                    std::vector<OptionalTensor>& grads_in,
                                    MtCudaWaitDeltas* deltas) {
  CudaWaitCounters counters;
  if (deltas) {
    counters.waited = &deltas->waited;
    counters.cross  = &deltas->cross;
  }
  return prepare_consumer_stream_and_wait_device(consumer_device, consumer, ib, grads_in, counters);
}

static std::optional<vbt::cuda::CUDAStreamGuard>
prepare_consumer_stream_and_wait(GraphTask& gt,
                                 Node& consumer,
                                 GraphTask::InputBuffer* ib,
                                 std::vector<OptionalTensor>& grads_in) {
  CudaWaitCounters counters;
  counters.waited = &gt.cuda_events_waited;
  counters.cross  = &gt.cuda_cross_stream_routes;
  return prepare_consumer_stream_and_wait_device(gt.autograd_device, consumer, ib, grads_in, counters);
}

#endif // VBT_WITH_CUDA

static void validate_outputs_if_applicable(const Node& n, const std::vector<OptionalTensor>& out) {
  auto* v = dynamic_cast<const ValidatableNode*>(&n);
  if (!v) return;
  const auto& metas = v->input_metas();
  if (out.size() != metas.size()) {
    throw std::runtime_error("wrong number of gradients (expected " + std::to_string(metas.size()) + ", got " + std::to_string(out.size()) + ")");
  }
  for (std::size_t k = 0; k < metas.size(); ++k) {
    if (!out[k].has_value()) continue;
    const auto& g = out[k].value();
    const auto& m = metas[k];
    if (g.dtype() != m.dtype) {
      throw std::runtime_error("dtype mismatch at input " + std::to_string(k));
    }
    if (g.device() != m.device) {
      throw std::runtime_error(std::string("device mismatch at input ") + std::to_string(k) + " (expected " + m.device.to_string() + ", got " + g.device().to_string() + ")");
    }
    if (m.is_strided_dense && !g.is_non_overlapping_and_dense()) {
      throw std::runtime_error("layout mismatch at input " + std::to_string(k));
    }
    if (g.sizes() != m.sizes) {
      throw std::runtime_error("sizes mismatch at input " + std::to_string(k));
    }
  }
}

// Checks E/S/R invariants and membership relationships between
// inputs, dependencies, and nodes_in_graph.
static void validate_graph_task_structure(GraphTask& gt, Node* root) {
  if (!root) {
    throw std::logic_error("engine debug: root must not be null");
  }

  // Root must be represented in nodes_in_graph and dependencies with E(root) == 0.
  if (!gt.nodes_in_graph.empty() && !gt.nodes_in_graph.count(root)) {
    throw std::logic_error("engine debug: root missing from nodes_in_graph");
  }

  if (!gt.dependencies.empty()) {
    auto it_root = gt.dependencies.find(root);
    if (it_root == gt.dependencies.end()) {
      throw std::logic_error("engine debug: root missing from dependencies");
    }
    if (it_root->second != 0) {
      throw std::logic_error("engine debug: root must have zero structural dependencies");
    }
  }

  // Every node that appears in dependencies must be in nodes_in_graph.
  for (const auto& kv : gt.dependencies) {
    Node* node = kv.first;
    if (!gt.nodes_in_graph.count(node)) {
      throw std::logic_error("engine debug: node in dependencies missing from nodes_in_graph");
    }
  }

  // Per-buffer invariants for all nodes that have InputBuffers.
  for (const auto& kv : gt.inputs) {
    Node* node = kv.first;
    const GraphTask::InputBuffer& ib = kv.second;

    if (!gt.nodes_in_graph.count(node)) {
      throw std::logic_error("engine debug: node in inputs missing from nodes_in_graph");
    }
    if (!gt.dependencies.count(node)) {
      throw std::logic_error("engine debug: node in inputs missing from dependencies");
    }

    const std::size_t expected = ib.expected;
    if (ib.grads_in.size() != expected || ib.present.size() != expected) {
      throw std::logic_error("engine debug: InputBuffer size invariants violated");
    }
    if (ib.received > expected) {
      throw std::logic_error("engine debug: InputBuffer.received exceeds expected");
    }

    for (std::size_t i = 0; i < expected; ++i) {
      const bool has_grad = ib.grads_in[i].has_value();
      const bool is_present = (ib.present[i] != 0);
      // When present[pos] == 0, grads_in[pos] must be disengaged.
      if (!is_present && has_grad) {
        throw std::logic_error("engine debug: slot has gradient but present flag is zero");
      }
    }
  }

  // Global E/S/R and duplicates_coalesced inequality from §4.4 of the design.
  long long sum_e_minus_r = 0;

  for (const auto& kv : gt.inputs) {
    Node* node = kv.first;
    const GraphTask::InputBuffer& ib = kv.second;

    if (node == root) {
      continue;  // Root is excluded from E/S/R accounting.
    }
    if (!gt.nodes_in_graph.count(node)) {
      continue;  // Should not happen after earlier checks, but be defensive.
    }
    auto it_dep = gt.dependencies.find(node);
    if (it_dep == gt.dependencies.end()) {
      continue;  // Node structurally unreachable for E/S/R purposes.
    }

    const int E = it_dep->second;
    if (E < 0) {
      throw std::logic_error("engine debug: negative dependency count");
    }

    const long long S = static_cast<long long>(ib.expected);
    const long long R = static_cast<long long>(ib.received);

    if (R < 0 || R > S) {
      throw std::logic_error("engine debug: R(n) out of bounds relative to S(n)");
    }

    const long long diff = static_cast<long long>(E) - R;
    if (diff < 0) {
      throw std::logic_error("engine debug: E(n) - R(n) must be non-negative");
    }

    sum_e_minus_r += diff;
  }

  if (static_cast<long long>(gt.duplicates_coalesced) > sum_e_minus_r) {
    throw std::logic_error("engine debug: duplicates_coalesced exceeds Σ(E(n) - R(n)) bound");
  }
}

} // anonymous

#if VBT_AUTOGRAD_TESTING && VBT_WITH_CUDA
// Global test-only hook for observing routing-time current streams.
static std::atomic<TestRouteHook> g_test_route_hook{nullptr};

static inline TestRouteHook get_test_route_hook() noexcept {
  return g_test_route_hook.load(std::memory_order_relaxed);
}

void _test_set_route_hook(TestRouteHook hook) noexcept {
  g_test_route_hook.store(hook, std::memory_order_relaxed);
}
#endif

#if VBT_AUTOGRAD_TESTING
// Thread-local snapshot of the most recent backward run.
//
// remains a no-op for CPU-only graphs.
static thread_local bool g_last_backward_valid = false;
static thread_local vbt::core::Device g_last_backward_device{};
static thread_local bool g_last_backward_streaming_enabled_snapshot = false;
static thread_local AutogradDeviceMode g_last_backward_device_mode_snapshot = AutogradDeviceMode::SingleDevice;
static thread_local std::uint64_t g_last_backward_cuda_events_recorded = 0;
static thread_local std::uint64_t g_last_backward_cuda_events_waited = 0;
static thread_local std::uint64_t g_last_backward_cuda_cross_stream_routes = 0;
static thread_local std::uint64_t g_last_backward_cuda_device_synchronizes = 0;
static thread_local std::vector<int> g_last_backward_cuda_devices_snapshot;
static thread_local std::vector<int> g_last_backward_cuda_lane_devices_snapshot;
static thread_local std::uint64_t g_last_backward_routing_device_switches = 0;

static inline void _test_update_last_backward_snapshot(const GraphTask& gt) noexcept {
  g_last_backward_valid = true;
  g_last_backward_device = gt.autograd_device;
  g_last_backward_streaming_enabled_snapshot = gt.streaming_enabled_snapshot;
  g_last_backward_device_mode_snapshot = gt.device_mode_snapshot;
  g_last_backward_cuda_events_recorded = gt.cuda_events_recorded;
  g_last_backward_cuda_events_waited = gt.cuda_events_waited;
  g_last_backward_cuda_cross_stream_routes = gt.cuda_cross_stream_routes;
  g_last_backward_cuda_device_synchronizes = gt.cuda_device_synchronizes;
  g_last_backward_routing_device_switches = gt.routing_device_switches;

  try {
    g_last_backward_cuda_devices_snapshot = gt.cuda_devices_snapshot;
    g_last_backward_cuda_lane_devices_snapshot = gt.cuda_lane_devices_snapshot;
  } catch (...) {
    g_last_backward_cuda_devices_snapshot.clear();
    g_last_backward_cuda_lane_devices_snapshot.clear();
  }
}

void _test_clear_last_backward_snapshot() noexcept {
  g_last_backward_valid = false;
  g_last_backward_device = vbt::core::Device{};
  g_last_backward_streaming_enabled_snapshot = false;
  g_last_backward_device_mode_snapshot = AutogradDeviceMode::SingleDevice;
  g_last_backward_cuda_events_recorded = 0;
  g_last_backward_cuda_events_waited = 0;
  g_last_backward_cuda_cross_stream_routes = 0;
  g_last_backward_cuda_device_synchronizes = 0;
  g_last_backward_cuda_devices_snapshot.clear();
  g_last_backward_cuda_lane_devices_snapshot.clear();
  g_last_backward_routing_device_switches = 0;
}

bool _test_last_backward_has_value() noexcept {
  return g_last_backward_valid;
}

vbt::core::Device _test_last_backward_autograd_device() noexcept {
  return g_last_backward_device;
}

bool _test_last_backward_streaming_enabled_snapshot() noexcept {
  return g_last_backward_streaming_enabled_snapshot;
}

AutogradDeviceMode _test_last_backward_device_mode_snapshot() noexcept {
  return g_last_backward_device_mode_snapshot;
}

void _test_last_backward_cuda_counters(std::uint64_t* recorded,
                                      std::uint64_t* waited,
                                      std::uint64_t* cross) noexcept {
  if (recorded) *recorded = g_last_backward_cuda_events_recorded;
  if (waited) *waited = g_last_backward_cuda_events_waited;
  if (cross) *cross = g_last_backward_cuda_cross_stream_routes;
}

std::uint64_t _test_last_backward_cuda_device_synchronizes() noexcept {
  return g_last_backward_cuda_device_synchronizes;
}

std::vector<int> _test_last_backward_cuda_devices_snapshot() {
  return g_last_backward_cuda_devices_snapshot;
}

std::vector<int> _test_last_backward_cuda_lane_devices_snapshot() {
  return g_last_backward_cuda_lane_devices_snapshot;
}

std::uint64_t _test_last_backward_routing_device_switches() noexcept {
  return g_last_backward_routing_device_switches;
}
#endif

#if VBT_AUTOGRAD_TESTING
void _test_route_edge_with_coalesce(GraphTask& gt,
                                    Node* consumer,
                                    std::size_t pos,
                                    OptionalTensor&& grad,
                                    intrusive_ptr<Node> consumer_keep) {
  coalesce_incoming(gt, consumer, pos, std::move(grad), consumer_keep);
}

void _test_add_gradient(GraphTask& gt,
                        Node* consumer,
                        std::size_t pos,
                        OptionalTensor&& grad,
                        intrusive_ptr<Node> consumer_keep) {
  if (!gt.has_autograd_device) {
    gt.autograd_device = vbt::core::Device::cpu(0);
    gt.has_autograd_device = true;
  }
  GraphTask::InputBuffer& ib = ensure_buffer(gt, consumer, consumer_keep);
  AddContext ctx{gt, *consumer, ib, pos, std::move(consumer_keep)};
  add_gradient(ctx, std::move(grad), /*schedule_by_slots=*/true);
}

#if VBT_WITH_CUDA
void _test_prepare_consumer_stream_and_wait(GraphTask& gt,
                                           Node& consumer,
                                           GraphTask::InputBuffer* ib,
                                           std::vector<OptionalTensor>& grads_in) {
  (void)prepare_consumer_stream_and_wait(gt, consumer, ib, grads_in);
}

bool _test_prepare_consumer_stream_and_wait_device(
    const vbt::core::Device& consumer_device,
    Node& consumer,
    GraphTask::InputBuffer* ib,
    std::vector<OptionalTensor>& grads_in,
    std::uint64_t* waited,
    std::uint64_t* cross) {
  CudaWaitCounters counters;
  counters.waited = waited;
  counters.cross  = cross;
  auto guard = prepare_consumer_stream_and_wait_device(
      consumer_device, consumer, ib, grads_in, counters);
  return guard.has_value();
}

bool _test_prepare_consumer_stream_and_wait_mt_no_deltas(
    const vbt::core::Device& consumer_device,
    Node& consumer,
    GraphTask::InputBuffer* ib,
    std::vector<OptionalTensor>& grads_in) {
  auto guard = prepare_consumer_stream_and_wait_mt(
      consumer_device, consumer, ib, grads_in, /*deltas=*/nullptr);
  return guard.has_value();
}

void _test_add_gradient_cuda_device(GraphTask& gt,
                                   Node* consumer,
                                   std::size_t pos,
                                   OptionalTensor&& grad,
                                   intrusive_ptr<Node> consumer_keep,
                                   const vbt::core::Device& slot_device) {
  GraphTask::InputBuffer& ib = ensure_buffer(gt, consumer, consumer_keep);
  AddContext ctx{gt, *consumer, ib, pos, std::move(consumer_keep)};
  add_gradient_cuda_device(ctx, std::move(grad), slot_device, /*schedule_by_slots=*/false);
}
#endif

void _test_compute_dependencies(GraphTask& gt,
                                intrusive_ptr<Node> root) {
  (void)build_graph_topology(gt, root);
}

void _test_seed_root_buffer(GraphTask& gt,
                            intrusive_ptr<Node> root,
                            const std::vector<OptionalTensor>& initial_grads) {
  derive_autograd_device_and_seed_root(
      gt, root, initial_grads, /*mt_enabled_snapshot=*/is_multithreading_enabled());
}

void _test_validate_graph_task_structure(GraphTask& gt,
                                         intrusive_ptr<Node> root) {
  validate_graph_task_structure(gt, root.get());
}
#endif

void Engine::run_backward_impl(intrusive_ptr<Node> root,
                              const std::vector<OptionalTensor>& initial_grads,
                              const std::vector<std::function<void()>>& callbacks,
                              GraphTask::ValidateHook hook) {
  if (is_in_backward()) {
    throw std::runtime_error(
        "VibeTensor autograd: nested backward is not supported");
  }

  const bool mt_enabled_snapshot = is_multithreading_enabled();

  std::unique_lock<std::mutex> backward_gate;
  if (mt_enabled_snapshot) {
    backward_gate = std::unique_lock<std::mutex>(g_backward_gate_mu, std::try_to_lock);
    if (!backward_gate.owns_lock()) {
      throw std::runtime_error(
          "VibeTensor autograd: another multithreaded backward is already in progress");
    }
  }

  BackwardGuard guard;
  GraphTask gt;
  gt.validate_hook = hook;
  gt.device_mode_snapshot = get_device_mode();
  // Build structural graph topology (dependencies/nodes_in_graph).
  build_graph_topology(gt, root);

  // Snapshot streaming toggle once per backward.
  gt.streaming_enabled_snapshot = is_streaming_backwards_enabled();

  // Resolve logical device & seed root buffer.
  derive_autograd_device_and_seed_root(gt, root, initial_grads, mt_enabled_snapshot);

#if VBT_WITH_CUDA
  if (gt.autograd_device.type == kDLCUDA) {
    if (!gt.streaming_enabled_snapshot) {
      throw std::runtime_error(
          "VibeTensor CUDA autograd is disabled; call set_cuda_autograd_enabled(True) to enable it.");
    }
    vbt::cuda::assert_not_capturing_backward_stream(gt.autograd_device);
  }
#endif

  std::exception_ptr err;

  if (mt_enabled_snapshot && gt.autograd_device.type == kDLCPU) {
    struct MtTask {
      intrusive_ptr<Node> fn;
      std::vector<OptionalTensor> grads_in;
    };

    struct MtCompletion {
      intrusive_ptr<Node> fn;
      std::vector<OptionalTensor> grads_out;
      std::exception_ptr error;
    };

    ConcurrentTaskQueue<MtTask> q;
    OwnerMailbox<MtCompletion> mailbox;
    std::atomic<bool> cancelled{false};

    std::vector<std::thread> workers;
    const unsigned nthreads = default_cpu_worker_threads();
    workers.reserve(nthreads);

    // Join-on-scope-exit guard to avoid leaking threads on exceptions.
    struct WorkerJoinGuard {
      ConcurrentTaskQueue<MtTask>& q;
      std::vector<std::thread>& workers;
      ~WorkerJoinGuard() {
        q.close();
        for (auto& t : workers) {
          if (t.joinable()) t.join();
        }
      }
    } join_guard{q, workers};

    for (unsigned i = 0; i < nthreads; ++i) {
      workers.emplace_back([&]() {
        MtTask task;
        while (q.pop_blocking(&task)) {
          BackwardGuard worker_guard;

          if (cancelled.load(std::memory_order_relaxed)) {
            mailbox.push(MtCompletion{task.fn, {}, nullptr});
            continue;
          }

          try {
            NoGradGuard ng;
            std::vector<OptionalTensor> grads_out = task.fn->apply(std::move(task.grads_in));
            mailbox.push(MtCompletion{task.fn, std::move(grads_out), nullptr});
          } catch (...) {
            mailbox.push(MtCompletion{task.fn, {}, std::current_exception()});
          }
        }
      });
    }

    std::unordered_map<Node*, int> deps_remaining = gt.dependencies;

    std::size_t outstanding = 0;

    auto schedule_node = [&](const intrusive_ptr<Node>& fn) {
      MtTask task;
      task.fn = fn;

      auto it = gt.inputs.find(fn.get());
      if (it != gt.inputs.end()) {
        task.grads_in = std::move(it->second.grads_in);
        // Preserve InputBuffer size invariants for debug tooling.
        it->second.grads_in.assign(it->second.expected, OptionalTensor{});
      } else {
        const std::size_t nslots = static_cast<std::size_t>(fn->num_incoming_grad_slots());
        task.grads_in.assign(nslots, OptionalTensor{});
      }

      if (!q.push(std::move(task))) {
        throw std::logic_error("engine(mt): attempted to schedule onto a closed queue");
      }
      ++outstanding;
    };

    // Root is seeded and enqueued by derive_autograd_device_and_seed_root().
    if (gt.ready.empty()) {
      throw std::logic_error("engine(mt): root not enqueued");
    }
    intrusive_ptr<Node> root_scheduled = gt.ready.pop_front();

    auto it_dep = deps_remaining.find(root_scheduled.get());
    if (it_dep == deps_remaining.end()) {
      throw std::logic_error("engine(mt): missing deps entry for scheduled node");
    }
    if (it_dep->second != 0) {
      throw std::logic_error("engine(mt): node scheduled before deps were satisfied");
    }

    schedule_node(root_scheduled);

    while (outstanding > 0) {
      MtCompletion msg = mailbox.pop_blocking();
      --outstanding;
      gt.nodes_processed++;

      if (msg.error && !err) {
        err = msg.error;
        cancelled.store(true, std::memory_order_relaxed);
        q.close();
      }

      if (cancelled.load(std::memory_order_relaxed)) {
        continue;  // drain-only
      }

      intrusive_ptr<Node> n = std::move(msg.fn);
      std::vector<OptionalTensor> grads_out = std::move(msg.grads_out);

      try {
        // Ensure no grad recording during validate/routing.
        NoGradGuard ng;

        // Detach produced grads before validate/routing: they must not require grad.
        for (auto& og : grads_out) {
          if (og.has_value()) { set_requires_grad(og.value(), false); }
        }

        // Built-in validation for Function-like nodes; optionally log on failure.
        try {
          validate_outputs_if_applicable(*n, grads_out);
          if (gt.validate_hook) gt.validate_hook(*n, grads_out);
        } catch (const std::exception& e) {
          if (kLogValidate) { VBT_LOG(INFO) << "[autograd] validate error: " << e.what(); }
          throw;
        }

#if !defined(NDEBUG)
        if (dynamic_cast<const ValidatableNode*>(n.get())) {
          if (grads_out.size() != n->next_edges.size()) {
            throw std::logic_error(
                "engine: next_edges not sized to produced grads for ValidatableNode");
          }
        }
#endif
        if (grads_out.size() != n->next_edges.size()) {
          throw std::logic_error("engine: grads_out size must equal next_edges size");
        }

        // Route each produced grad to consumer edge.
        std::vector<int> fenced_dst_cuda_indices;
        for (std::size_t k = 0; k < grads_out.size(); ++k) {
          const Edge& e = n->next_edges[k];
          if (!e.fn) continue;

          Node* consumer_raw = e.fn.get();
          const intrusive_ptr<Node>& consumer_keep = e.fn;
          GraphTask::InputBuffer& ib = ensure_buffer(gt, consumer_raw, consumer_keep);

          AddContext ctx{gt,
                         *consumer_raw,
                         ib,
                         static_cast<std::size_t>(e.input_nr),
                         consumer_keep};

#if VBT_WITH_CUDA
          if (gt.device_mode_snapshot == AutogradDeviceMode::MultiDeviceExperimental) {
            enforce_copy_and_maybe_fence_multidevice(
                gt,
                *n,
                *consumer_raw,
                ctx.pos,
                &grads_out[k],
                &fenced_dst_cuda_indices);
          }
#endif

          // MT scheduling: do not use slot-based enqueueing.
          add_gradient(ctx, std::move(grads_out[k]), /*schedule_by_slots=*/false);
          gt.edges_processed++;

          auto dep_it = deps_remaining.find(consumer_raw);
          if (dep_it == deps_remaining.end()) {
            throw std::logic_error("engine(mt): missing deps entry for consumer");
          }
          if (dep_it->second <= 0) {
            throw std::logic_error("engine(mt): deps underflow");
          }
          dep_it->second -= 1;
          if (dep_it->second == 0) {
            schedule_node(consumer_keep);
          }
        }
      } catch (...) {
        if (!err) {
          err = std::current_exception();
          cancelled.store(true, std::memory_order_relaxed);
          q.close();
        }
      }
    }
  }
#if VBT_WITH_CUDA
  else if (mt_enabled_snapshot && gt.autograd_device.type == kDLCUDA) {
    //
    // Lane routing:
    // - Lane::CUDA (StreamKind::CudaAllowlisted, except AccumulateGrad) executes
    //   on the CUDA worker thread pinned to the node's canonical device.
    // - Lane::CPU executes on the CPU worker pool.
    //
    // Note: AccumulateGrad for CUDA leaves is forced onto Lane::CPU (hooks / leaf
    // grad mutex), but still runs under a CUDAStreamGuard via
    // prepare_consumer_stream_and_wait_mt on the CPU worker.
    //
    // All gradient routing is performed on the owner thread, under the
    // producer's canonical stream.
    //
    // Guard discipline (DC3): CUDAStreamGuard only restores TLS stream state for
    // the construction-time current device, so ensure the owner thread is pinned
    // to the autograd CUDA device for the duration of this scheduling block.
    const auto dev_idx = static_cast<vbt::cuda::DeviceIndex>(gt.autograd_device.index);
    vbt::cuda::DeviceGuard owner_dg(dev_idx);

    struct MtTask {
      intrusive_ptr<Node> fn;
      GraphTask::InputBuffer input_buf;
    };

    struct MtCompletion {
      intrusive_ptr<Node> fn;
      std::vector<OptionalTensor> grads_out;
      std::optional<int> producer_device_index;
      std::optional<std::uint64_t> producer_stream_id;
      MtCudaWaitDeltas deltas;
      std::exception_ptr error;
    };

    struct CudaWorker {
      int device_index;
      ConcurrentTaskQueue<MtTask> q;
      std::thread t;
    };

    ConcurrentTaskQueue<MtTask> cpu_q;
    OwnerMailbox<MtCompletion> mailbox;
    std::atomic<bool> cancelled{false};

    std::vector<std::unique_ptr<CudaWorker>> cuda_workers;
    std::unordered_map<int, CudaWorker*> cuda_worker_by_device;

    // Build per-device CUDA workers.
    const auto add_cuda_worker = [&](int device_index) {
      auto w = std::make_unique<CudaWorker>();
      w->device_index = device_index;
      CudaWorker* raw = w.get();
      cuda_workers.emplace_back(std::move(w));
      cuda_worker_by_device.emplace(device_index, raw);
    };

    if (gt.device_mode_snapshot == AutogradDeviceMode::MultiDeviceExperimental) {
      for (int d : gt.cuda_lane_devices_snapshot) {
        add_cuda_worker(d);
      }
    } else {
      add_cuda_worker(static_cast<int>(gt.autograd_device.index));
    }

    std::vector<std::thread> cpu_workers;
    const unsigned nthreads = default_cpu_worker_threads();
    cpu_workers.reserve(nthreads);

    // Join-on-scope-exit guard to avoid leaking threads on exceptions.
    struct WorkerJoinGuard {
      ConcurrentTaskQueue<MtTask>& cpu_q;
      std::vector<std::unique_ptr<CudaWorker>>& cuda_workers;
      std::vector<std::thread>& cpu_workers;
      ~WorkerJoinGuard() {
        cpu_q.close();
        for (auto& w : cuda_workers) {
          w->q.close();
        }
        for (auto& t : cpu_workers) {
          if (t.joinable()) t.join();
        }
        for (auto& w : cuda_workers) {
          if (w->t.joinable()) w->t.join();
        }
      }
    } join_guard{cpu_q, cuda_workers, cpu_workers};

    for (unsigned i = 0; i < nthreads; ++i) {
      cpu_workers.emplace_back([&, dev_idx]() {
        // Preserve legacy behavior: CPU workers default to the backward's primary CUDA
        // device, while allowlisted CPU-lane tasks (AccumulateGrad CUDA leaves) may
        // temporarily switch to their canonical device.
        vbt::cuda::DeviceGuard dg(dev_idx);
        MtTask task;
        while (cpu_q.pop_blocking(&task)) {
          BackwardGuard worker_guard;

          if (cancelled.load(std::memory_order_relaxed)) {
            mailbox.push(MtCompletion{task.fn, {}, std::nullopt, std::nullopt, MtCudaWaitDeltas{}, nullptr});
            continue;
          }

          MtCudaWaitDeltas deltas;
          std::optional<int> producer_device_index;
          std::optional<std::uint64_t> producer_stream_id;

          try {
            std::vector<OptionalTensor> grads_in = std::move(task.input_buf.grads_in);

            // CPU lane tasks may still be CUDA-allowlisted (AccumulateGrad CUDA leaf).
            //
            // For CpuOnly tasks we intentionally keep passing gt.autograd_device into
            // prepare_consumer_stream_and_wait_mt so it can run its internal accelerated-slot
            // sanity checks; it returns nullopt and does not construct a CUDAStreamGuard.
            //
            // For allowlisted tasks, pin the per-task execution device so the CUDAStreamGuard
            // DC3 precondition holds and canonical stream validation uses the correct device.
            const bool is_allowlisted = (task.fn->stream_kind() == StreamKind::CudaAllowlisted);
            const vbt::core::Device exec_dev =
                is_allowlisted ? task.fn->stream_info().device : gt.autograd_device;

            std::optional<vbt::cuda::DeviceGuard> exec_dg;
            if (is_allowlisted) {
              if (exec_dev.type != kDLCUDA) {
                throw std::logic_error(
                    "engine(mt,cuda): CPU lane allowlisted task has non-CUDA execution device");
              }
              const auto exec_idx = static_cast<vbt::cuda::DeviceIndex>(exec_dev.index);
              exec_dg.emplace(exec_idx);
            }

            auto guard = prepare_consumer_stream_and_wait_mt(
                exec_dev, *task.fn, &task.input_buf, grads_in, &deltas);
            if (guard.has_value()) {
              producer_device_index = static_cast<int>(guard->current_stream().device_index());
              producer_stream_id = guard->current_stream().id();
            }

            NoGradGuard ng;
            std::vector<OptionalTensor> grads_out = task.fn->apply(std::move(grads_in));
            mailbox.push(MtCompletion{task.fn,
                                      std::move(grads_out),
                                      producer_device_index,
                                      producer_stream_id,
                                      deltas,
                                      nullptr});
          } catch (...) {
            mailbox.push(MtCompletion{task.fn,
                                      {},
                                      producer_device_index,
                                      producer_stream_id,
                                      deltas,
                                      std::current_exception()});
          }
        }
      });
    }

    for (auto& w_up : cuda_workers) {
      CudaWorker* w = w_up.get();
      const int d = w->device_index;
      w->t = std::thread([&, w, d]() {
        const auto w_dev_idx = static_cast<vbt::cuda::DeviceIndex>(d);
        vbt::cuda::DeviceGuard dg(w_dev_idx);
        MtTask task;
        while (w->q.pop_blocking(&task)) {
          BackwardGuard worker_guard;

          if (cancelled.load(std::memory_order_relaxed)) {
            mailbox.push(MtCompletion{task.fn, {}, std::nullopt, std::nullopt, MtCudaWaitDeltas{}, nullptr});
            continue;
          }

          MtCudaWaitDeltas deltas;
          std::optional<int> producer_device_index;
          std::optional<std::uint64_t> producer_stream_id;

          try {
            std::vector<OptionalTensor> grads_in = std::move(task.input_buf.grads_in);

            const vbt::core::Device exec_dev = task.fn->stream_info().device;
            if (exec_dev.type != kDLCUDA) {
              throw std::logic_error(
                  "engine(mt,cuda): CUDA lane task has non-CUDA execution device");
            }
            if (exec_dev.index != d) {
              throw std::logic_error(
                  "engine(mt,cuda): misrouted CUDA lane task to wrong CUDA worker");
            }

            auto guard = prepare_consumer_stream_and_wait_mt(
                exec_dev, *task.fn, &task.input_buf, grads_in, &deltas);
            if (guard.has_value()) {
              producer_device_index = d;
              producer_stream_id = guard->current_stream().id();
            }

            NoGradGuard ng;
            std::vector<OptionalTensor> grads_out = task.fn->apply(std::move(grads_in));
            mailbox.push(MtCompletion{task.fn,
                                      std::move(grads_out),
                                      producer_device_index,
                                      producer_stream_id,
                                      deltas,
                                      nullptr});
          } catch (...) {
            mailbox.push(MtCompletion{task.fn,
                                      {},
                                      producer_device_index,
                                      producer_stream_id,
                                      deltas,
                                      std::current_exception()});
          }
        }
      });
    }

    std::unordered_map<Node*, int> deps_remaining = gt.dependencies;

    std::size_t outstanding = 0;

    auto schedule_node = [&](const intrusive_ptr<Node>& fn) {
      MtTask task;
      task.fn = fn;

      auto it = gt.inputs.find(fn.get());
      if (it != gt.inputs.end()) {
        task.input_buf = std::move(it->second);

        // Restore size invariants on the moved-from buffer for debug tooling.
        const std::size_t expected = it->second.expected;
        it->second.grads_in.assign(expected, OptionalTensor{});
        it->second.present.assign(expected, 0);
      } else {
        const std::size_t expected = static_cast<std::size_t>(fn->num_incoming_grad_slots());
        task.input_buf.ensure_cpu_capacity(expected);
        task.input_buf.ensure_cuda_capacity(expected);
      }

      const Lane lane = lane_for_node(gt.autograd_device, *fn);
      if (lane == Lane::CUDA) {
        const vbt::core::Device dev = fn->stream_info().device;
        if (dev.type != kDLCUDA) {
          throw std::logic_error(
              "engine(mt,cuda): Lane::CUDA task has non-CUDA stream_info device");
        }
        const int d = static_cast<int>(dev.index);
        auto it_w = cuda_worker_by_device.find(d);
        if (it_w == cuda_worker_by_device.end()) {
          throw std::logic_error(
              "engine(mt,cuda): missing CUDA worker for device " + std::to_string(d));
        }
        if (!it_w->second->q.push(std::move(task))) {
          throw std::logic_error(
              "engine(mt,cuda): attempted to schedule onto a closed CUDA queue for device " +
              std::to_string(d));
        }
      } else {
        if (!cpu_q.push(std::move(task))) {
          throw std::logic_error("engine(mt,cuda): attempted to schedule onto a closed CPU queue");
        }
      }
      ++outstanding;
    };

    // Root is seeded and enqueued by derive_autograd_device_and_seed_root().
    if (gt.ready.empty()) {
      throw std::logic_error("engine(mt,cuda): root not enqueued");
    }
    intrusive_ptr<Node> root_scheduled = gt.ready.pop_front();

    auto it_dep = deps_remaining.find(root_scheduled.get());
    if (it_dep == deps_remaining.end()) {
      throw std::logic_error("engine(mt,cuda): missing deps entry for scheduled node");
    }
    if (it_dep->second != 0) {
      throw std::logic_error("engine(mt,cuda): node scheduled before deps were satisfied");
    }

    schedule_node(root_scheduled);

    while (outstanding > 0) {
      MtCompletion msg = mailbox.pop_blocking();
      --outstanding;
      gt.nodes_processed++;

      // Merge worker deltas.
      gt.cuda_events_waited += msg.deltas.waited;
      gt.cuda_cross_stream_routes += msg.deltas.cross;

      if (msg.error && !err) {
        err = msg.error;
        cancelled.store(true, std::memory_order_relaxed);
        cpu_q.close();
        for (auto& w : cuda_workers) {
          w->q.close();
        }
      }

      if (cancelled.load(std::memory_order_relaxed)) {
        continue;  // drain-only
      }

      intrusive_ptr<Node> n = std::move(msg.fn);
      std::vector<OptionalTensor> grads_out = std::move(msg.grads_out);

      try {
        NoGradGuard ng;

        for (auto& og : grads_out) {
          if (og.has_value()) { set_requires_grad(og.value(), false); }
        }

        try {
          validate_outputs_if_applicable(*n, grads_out);
          if (gt.validate_hook) gt.validate_hook(*n, grads_out);
        } catch (const std::exception& e) {
          if (kLogValidate) { VBT_LOG(INFO) << "[autograd] validate error: " << e.what(); }
          throw;
        }

#if !defined(NDEBUG)
        if (dynamic_cast<const ValidatableNode*>(n.get())) {
          if (grads_out.size() != n->next_edges.size()) {
            throw std::logic_error(
                "engine: next_edges not sized to produced grads for ValidatableNode");
          }
        }
#endif
        if (grads_out.size() != n->next_edges.size()) {
          throw std::logic_error("engine: grads_out size must equal next_edges size");
        }

        if (gt.device_mode_snapshot == AutogradDeviceMode::MultiDeviceExperimental) {
          //   1) plan + fence (no routing CUDAStreamGuard alive)
          //   2) bucket keyed edges by (device, stream_id) and route under matched guards
          struct RoutingKey {
            int device_index;
            std::uint64_t stream_id;
          };

          struct PlannedEdge {
            std::size_t edge_seq;
            Node* consumer_raw;
            intrusive_ptr<Node> consumer_keep;
            std::size_t pos;
            vbt::core::Device expected;
            OptionalTensor grad;
            std::optional<RoutingKey> key;
          };

          if (n->stream_kind() == StreamKind::CudaAllowlisted) {
            if (!msg.producer_device_index.has_value()) {
              throw std::logic_error(
                  "engine(mt,cuda): missing producer device index for CUDA allowlisted node");
            }
            if (!msg.producer_stream_id.has_value()) {
              throw std::logic_error(
                  "engine(mt,cuda): missing producer stream id for CUDA allowlisted node");
            }
          }

          std::vector<int> fenced_dst_cuda_indices;
          std::vector<PlannedEdge> planned;
          planned.reserve(grads_out.size());

          for (std::size_t k = 0; k < grads_out.size(); ++k) {
            const Edge& e = n->next_edges[k];
            if (!e.fn) continue;

            PlannedEdge pe;
            pe.edge_seq = k;
            pe.consumer_raw = e.fn.get();
            pe.consumer_keep = e.fn;
            pe.pos = static_cast<std::size_t>(e.input_nr);
            pe.grad = std::move(grads_out[k]);
            pe.expected = expected_device_for_slot_multidevice(*pe.consumer_raw, pe.pos);

            enforce_copy_and_maybe_fence_multidevice(
                gt,
                *n,
                *pe.consumer_raw,
                pe.pos,
                &pe.grad,
                &fenced_dst_cuda_indices);

            if (pe.grad.has_value()) {
              const vbt::core::Device got = pe.grad.value().device();
              if (got != pe.expected) {
                throw_explicit_copy_required(
                    "consumer_mismatch", /*producer=*/nullptr, *pe.consumer_raw, pe.pos, pe.expected, got);
              }
            }

            // RoutingKey is required only for defined CUDA edges.
            if (pe.grad.has_value() && pe.expected.type == kDLCUDA) {
              const int dst = pe.expected.index;

              // Preferred: same-device output routed under producer stream.
              if (msg.producer_device_index.has_value() &&
                  msg.producer_stream_id.has_value() &&
                  msg.producer_device_index.value() == dst) {
                pe.key = RoutingKey{dst, msg.producer_stream_id.value()};
              } else {
                // Cross-device output: require destination-device fence and
                // route under the consumer canonical stream.
                bool fenced = false;
                for (int d : fenced_dst_cuda_indices) {
                  if (d == dst) {
                    fenced = true;
                    break;
                  }
                }
                if (!fenced) {
                  throw std::logic_error(
                      "engine(mt,cuda): cross-device CUDA route without destination fence");
                }

                const NodeStreamInfo& si = pe.consumer_raw->stream_info();
                if (!si.has_canonical_stream || si.device.type != kDLCUDA || si.device.index != dst) {
                  throw std::logic_error(
                      "engine(mt,cuda): consumer canonical stream/device mismatch in cross-device routing fallback");
                }
                pe.key = RoutingKey{dst, si.stream_id};
              }
            }

            planned.push_back(std::move(pe));
          }

          auto route_one_edge = [&](PlannedEdge& pe) {
            GraphTask::InputBuffer& ib = ensure_buffer(gt, pe.consumer_raw, pe.consumer_keep);
            AddContext ctx{gt,
                           *pe.consumer_raw,
                           ib,
                           pe.pos,
                           pe.consumer_keep};

#if VBT_AUTOGRAD_TESTING
            if (TestRouteHook hook = get_test_route_hook()) {
              const auto hook_dev_idx = static_cast<vbt::cuda::DeviceIndex>(
                  (pe.expected.type == kDLCUDA) ? pe.expected.index : gt.autograd_device.index);
              vbt::cuda::Stream cur = vbt::cuda::getCurrentStream(hook_dev_idx);
              hook(*n, *pe.consumer_raw, cur.id());
            }
#endif

            // MT scheduling: do not use slot-based enqueueing.
            add_gradient(ctx, std::move(pe.grad), /*schedule_by_slots=*/false);
            gt.edges_processed++;

            auto dep_it = deps_remaining.find(pe.consumer_raw);
            if (dep_it == deps_remaining.end()) {
              throw std::logic_error("engine(mt,cuda): missing deps entry for consumer");
            }
            if (dep_it->second <= 0) {
              throw std::logic_error("engine(mt,cuda): deps underflow");
            }
            dep_it->second -= 1;
            if (dep_it->second == 0) {
              schedule_node(pe.consumer_keep);
            }
          };

          // Route unkeyed edges without installing a routing CUDAStreamGuard.
          for (auto& pe : planned) {
            if (pe.key.has_value()) continue;
            route_one_edge(pe);
          }

          // Bucket keyed edges by (device, stream_id) for device-aware routing.
          std::vector<PlannedEdge*> keyed;
          keyed.reserve(planned.size());
          for (auto& pe : planned) {
            if (pe.key.has_value()) {
              keyed.push_back(&pe);
            }
          }

          std::sort(keyed.begin(), keyed.end(), [](const PlannedEdge* a, const PlannedEdge* b) {
            const RoutingKey& ka = a->key.value();
            const RoutingKey& kb = b->key.value();
            if (ka.device_index != kb.device_index) return ka.device_index < kb.device_index;
            if (ka.stream_id != kb.stream_id) return ka.stream_id < kb.stream_id;
            return a->edge_seq < b->edge_seq;
          });

          std::optional<int> prev_bucket_device;

          std::size_t i = 0;
          while (i < keyed.size()) {
            const RoutingKey key = keyed[i]->key.value();
            const int bucket_device = key.device_index;
            const std::uint64_t bucket_stream_id = key.stream_id;

            if (prev_bucket_device.has_value() && prev_bucket_device.value() != bucket_device) {
              ++gt.routing_device_switches;
            }
            prev_bucket_device = bucket_device;

            const auto dev = static_cast<vbt::cuda::DeviceIndex>(bucket_device);
            vbt::cuda::DeviceGuard dg(dev);
            vbt::cuda::Stream S_bucket(vbt::cuda::Stream::UNCHECKED, bucket_stream_id, dev);
            vbt::cuda::CUDAStreamGuard sg(S_bucket);

            while (i < keyed.size()) {
              const RoutingKey k2 = keyed[i]->key.value();
              if (k2.device_index != bucket_device || k2.stream_id != bucket_stream_id) {
                break;
              }
              route_one_edge(*keyed[i]);
              ++i;
            }
          }
        } else {
          // Legacy single-device routing: route under producer canonical stream when available.
          std::optional<vbt::cuda::DeviceGuard> route_dg;
          std::optional<vbt::cuda::CUDAStreamGuard> route_guard;
          if (n->stream_kind() == StreamKind::CudaAllowlisted) {
            if (!msg.producer_device_index.has_value()) {
              throw std::logic_error(
                  "engine(mt,cuda): missing producer device index for CUDA allowlisted node");
            }
            if (!msg.producer_stream_id.has_value()) {
              throw std::logic_error(
                  "engine(mt,cuda): missing producer stream id for CUDA allowlisted node");
            }
            const auto prod_dev = static_cast<vbt::cuda::DeviceIndex>(
                msg.producer_device_index.value());
            const std::uint64_t sid = msg.producer_stream_id.value();
            route_dg.emplace(prod_dev);
            vbt::cuda::Stream S_prod(vbt::cuda::Stream::UNCHECKED, sid, prod_dev);
            route_guard.emplace(S_prod);
          }

          for (std::size_t k = 0; k < grads_out.size(); ++k) {
            const Edge& e = n->next_edges[k];
            if (!e.fn) continue;

            Node* consumer_raw = e.fn.get();
            const intrusive_ptr<Node>& consumer_keep = e.fn;
            GraphTask::InputBuffer& ib = ensure_buffer(gt, consumer_raw, consumer_keep);

            AddContext ctx{gt,
                           *consumer_raw,
                           ib,
                           static_cast<std::size_t>(e.input_nr),
                           consumer_keep};

#if VBT_AUTOGRAD_TESTING
            if (TestRouteHook hook = get_test_route_hook()) {
              vbt::cuda::Stream cur = vbt::cuda::getCurrentStream(dev_idx);
              hook(*n, *consumer_raw, cur.id());
            }
#endif

            add_gradient(ctx, std::move(grads_out[k]), /*schedule_by_slots=*/false);
            gt.edges_processed++;

            auto dep_it = deps_remaining.find(consumer_raw);
            if (dep_it == deps_remaining.end()) {
              throw std::logic_error("engine(mt,cuda): missing deps entry for consumer");
            }
            if (dep_it->second <= 0) {
              throw std::logic_error("engine(mt,cuda): deps underflow");
            }
            dep_it->second -= 1;
            if (dep_it->second == 0) {
              schedule_node(consumer_keep);
            }
          }
        }
      } catch (...) {
        if (!err) {
          err = std::current_exception();
          cancelled.store(true, std::memory_order_relaxed);
          cpu_q.close();
          for (auto& w : cuda_workers) {
            w->q.close();
          }
        }
      }
    }
  }
#endif
  else if (mt_enabled_snapshot) {
    //
    // In this mode, nodes become runnable only after ALL structural incoming
    // "slot-based" early-enqueue behavior when multiple edges map to the same
    // input_nr.
    std::unordered_map<Node*, int> deps_remaining = gt.dependencies;

    // Drain loop.
    while (!gt.ready.empty()) {
      intrusive_ptr<Node> n = gt.ready.pop_front();
      gt.nodes_processed++;

      try {
        auto it_dep = deps_remaining.find(n.get());
        if (it_dep == deps_remaining.end()) {
          throw std::logic_error("engine(mt): missing deps entry for scheduled node");
        }
        if (it_dep->second != 0) {
          throw std::logic_error("engine(mt): node scheduled before deps were satisfied");
        }

        // Move incoming grads into a vector for apply.
        auto it = gt.inputs.find(n.get());
        std::vector<OptionalTensor> grads_in;
        GraphTask::InputBuffer* ib_ptr = nullptr;
        if (it != gt.inputs.end()) {
          ib_ptr = &it->second;
          grads_in = std::move(ib_ptr->grads_in);
        } else {
          // No incoming grads; create empty sized vector.
          grads_in.resize(n->num_inputs());
        }
#if VBT_WITH_CUDA
        auto stream_guard = prepare_consumer_stream_and_wait(gt, *n, ib_ptr, grads_in);
#else
        (void)ib_ptr;
#endif
        // Ensure no grad recording during apply/validate/routing.
        NoGradGuard ng;
        // Apply node to produce grads for its inputs.
        std::vector<OptionalTensor> grads_out = n->apply(std::move(grads_in));
        // Detach produced grads before validate/routing: they must not require grad.
        for (auto& og : grads_out) {
          if (og.has_value()) { set_requires_grad(og.value(), false); }
        }
        // Built-in validation for Function-like nodes; optionally log on failure.
        try {
          validate_outputs_if_applicable(*n, grads_out);
          // External hook (if any)
          if (gt.validate_hook) gt.validate_hook(*n, grads_out);
        } catch (const std::exception& e) {
          if (kLogValidate) { VBT_LOG(INFO) << "[autograd] validate error: " << e.what(); }
          throw; // rethrow to outer catch
        }
#if !defined(NDEBUG)
        // For validatable nodes, assert edges sized.
        if (dynamic_cast<const ValidatableNode*>(n.get())) {
          if (grads_out.size() != n->next_edges.size()) {
            throw std::logic_error("engine: next_edges not sized to produced grads for ValidatableNode");
          }
        }
#endif
        if (grads_out.size() != n->next_edges.size()) {
          throw std::logic_error("engine: grads_out size must equal next_edges size");
        }

        // Route each produced grad to consumer edge.
        std::vector<int> fenced_dst_cuda_indices;
        for (std::size_t k = 0; k < grads_out.size(); ++k) {
          const Edge& e = n->next_edges[k];
          if (!e.fn) continue; // skip null edge

          Node* consumer_raw = e.fn.get();
          const intrusive_ptr<Node>& consumer_keep = e.fn;
          GraphTask::InputBuffer& ib = ensure_buffer(gt, consumer_raw, consumer_keep);

          AddContext ctx{gt,
                         *consumer_raw,
                         ib,
                         static_cast<std::size_t>(e.input_nr),
                         consumer_keep};

#if VBT_AUTOGRAD_TESTING && VBT_WITH_CUDA
          if (gt.autograd_device.type == kDLCUDA) {
            if (TestRouteHook hook = get_test_route_hook()) {
              const auto dev_idx =
                  static_cast<vbt::cuda::DeviceIndex>(gt.autograd_device.index);
              vbt::cuda::Stream cur = vbt::cuda::getCurrentStream(dev_idx);
              hook(*n, *consumer_raw, cur.id());
            }
          }
#endif

#if VBT_WITH_CUDA
          if (gt.device_mode_snapshot == AutogradDeviceMode::MultiDeviceExperimental) {
            enforce_copy_and_maybe_fence_multidevice(
                gt,
                *n,
                *consumer_raw,
                ctx.pos,
                &grads_out[k],
                &fenced_dst_cuda_indices);
          }
#endif

          // MT single-lane mode: do not use slot-based enqueueing.
          add_gradient(ctx, std::move(grads_out[k]), /*schedule_by_slots=*/false);
          gt.edges_processed++;

          auto dep_it = deps_remaining.find(consumer_raw);
          if (dep_it == deps_remaining.end()) {
            throw std::logic_error("engine(mt): missing deps entry for consumer");
          }
          if (dep_it->second <= 0) {
            throw std::logic_error("engine(mt): deps underflow");
          }
          dep_it->second -= 1;
          if (dep_it->second == 0) {
            gt.ready.push_back(consumer_keep);
          }
        }
      } catch (...) {
        err = std::current_exception();
        break;
      }
    }
  } else {
    auto process_ready = [&gt, &err]() {
      // Drain loop
      while (!gt.ready.empty()) {
        intrusive_ptr<Node> n = gt.ready.pop_front();
        gt.nodes_processed++;
        // Move incoming grads into a vector for apply
        auto it = gt.inputs.find(n.get());
        std::vector<OptionalTensor> grads_in;
        GraphTask::InputBuffer* ib_ptr = nullptr;
        if (it != gt.inputs.end()) {
          ib_ptr = &it->second;
          grads_in = std::move(ib_ptr->grads_in);
        } else {
          // No incoming grads; create empty sized vector as per contract
          grads_in.resize(n->num_inputs());
        }

        try {
#if VBT_WITH_CUDA
          auto stream_guard = prepare_consumer_stream_and_wait(gt, *n, ib_ptr, grads_in);
#else
          (void)ib_ptr;
#endif
          // Ensure no grad recording during apply/validate/routing
          NoGradGuard ng;
          // Apply node to produce grads for its inputs
          std::vector<OptionalTensor> grads_out = n->apply(std::move(grads_in));
          // Detach produced grads before validate/routing: they must not require grad
          for (auto& og : grads_out) {
            if (og.has_value()) { set_requires_grad(og.value(), false); }
          }
          // Built-in validation for Function-like nodes; optionally log on failure
          try {
            validate_outputs_if_applicable(*n, grads_out);
            // External hook (if any)
            if (gt.validate_hook) gt.validate_hook(*n, grads_out);
          } catch (const std::exception& e) {
            if (kLogValidate) { VBT_LOG(INFO) << "[autograd] validate error: " << e.what(); }
            throw; // rethrow to outer catch
          }
#if !defined(NDEBUG)
          // For validatable nodes, assert edges sized
          if (dynamic_cast<const ValidatableNode*>(n.get())) {
            if (grads_out.size() != n->next_edges.size()) {
              throw std::logic_error("engine: next_edges not sized to produced grads for ValidatableNode");
            }
          }
#endif
          if (grads_out.size() != n->next_edges.size()) {
            throw std::logic_error("engine: grads_out size must equal next_edges size");
          }
          // Route each produced grad to consumer edge
          for (std::size_t k = 0; k < grads_out.size(); ++k) {
            const Edge& e = n->next_edges[k];
            if (!e.fn) continue; // skip null edge

            Node* consumer_raw = e.fn.get();
            const intrusive_ptr<Node>& consumer_keep = e.fn;
            GraphTask::InputBuffer& ib = ensure_buffer(gt, consumer_raw, consumer_keep);

            AddContext ctx{gt,
                           *consumer_raw,
                           ib,
                           static_cast<std::size_t>(e.input_nr),
                           consumer_keep};

#if VBT_AUTOGRAD_TESTING && VBT_WITH_CUDA
            if (gt.autograd_device.type == kDLCUDA) {
              if (TestRouteHook hook = get_test_route_hook()) {
                const auto dev_idx =
                    static_cast<vbt::cuda::DeviceIndex>(gt.autograd_device.index);
                vbt::cuda::Stream cur = vbt::cuda::getCurrentStream(dev_idx);
                hook(*n, *consumer_raw, cur.id());
              }
            }
#endif

            add_gradient(ctx, std::move(grads_out[k]), /*schedule_by_slots=*/true);
            gt.edges_processed++;
          }
        } catch (...) {
          err = std::current_exception();
          break;
        }
      }
    };

    auto run_single_lane = [&]() {
      // Primary drain loop.
      process_ready();

      // Safety net: if some nodes observed gradients but were never
      // enqueued (e.g., due to conservative expected slot counts), run
      // them once treating missing slots as having zero gradients.
      if (!err) {
        bool progress = true;
        while (progress) {
          progress = false;
          for (auto& kv : gt.inputs) {
            Node* node_ptr = kv.first;
            GraphTask::InputBuffer& ib = kv.second;
            if (!ib.enqueued && ib.received > 0) {
              ib.enqueued = true;
              gt.ready.push_back(gt.keepalive[node_ptr]);
              progress = true;
            }
          }
          if (progress) {
            process_ready();
            if (err) break;
          }
        }
      }
    };

#if VBT_WITH_CUDA
    if (gt.autograd_device.type == kDLCUDA) {
      // Guard discipline (DC3): ensure current device matches the CUDA streams
      // used by prepare_consumer_stream_and_wait.
      const auto dev_idx = static_cast<vbt::cuda::DeviceIndex>(gt.autograd_device.index);
      vbt::cuda::DeviceGuard owner_dg(dev_idx);
      run_single_lane();
    } else {
      run_single_lane();
    }
#else
    run_single_lane();
#endif
  }

#if VBT_AUTOGRAD_ENGINE_DEBUG
  if (!err) {
    try {
      validate_graph_task_structure(gt, root.get());
    } catch (...) {
      if (!err) {
        err = std::current_exception();
      }
    }
  }
#endif

  // Final callbacks (count only those that complete; stop on first exception)
  std::exception_ptr cb_err;
  std::exception_ptr mg_err;  // multi-grad backward-complete callback error
  std::size_t cb_ran = 0;
  for (const auto& cb : callbacks) {
    try {
      cb();
      ++cb_ran;
    } catch (...) {
      cb_err = std::current_exception();
      break;
    }
  }
  gt.callbacks_run = cb_ran;
  if (kLogEngine) {
    VBT_LOG(INFO) << "[autograd] run: nodes=" << gt.nodes_processed
                  << " edges=" << gt.edges_processed
                  << " dups=" << gt.duplicates_coalesced
                  << " callbacks=" << cb_ran;
  }
  _stats_bump_engine(gt.nodes_processed, gt.edges_processed, gt.duplicates_coalesced, cb_ran);

  // This callback is conceptually per-backward-run and is invoked
  // regardless of whether the engine saw an error; the Python side is
  // responsible for deciding how to treat partial gradients in error
  // cases and for resetting its own per-run state.
  {
    BackwardCompleteCallback cb;
    if (g_backward_complete_cb) {
      cb = *g_backward_complete_cb;
    }
    if (cb) {
      try {
        cb();
      } catch (...) {
        mg_err = std::current_exception();
      }
    }
  }

#if VBT_AUTOGRAD_TESTING
  // Only publish the snapshot for successful backward runs.
  if (!err && !cb_err && !mg_err) {
    _test_update_last_backward_snapshot(gt);
  }
#endif

  if (err) std::rethrow_exception(err);
  if (cb_err) std::rethrow_exception(cb_err);
  if (mg_err) std::rethrow_exception(mg_err);
}

void set_backward_complete_callback(BackwardCompleteCallback cb) {
  get_backward_complete_cb_storage() = std::move(cb);
}

Engine& Engine::get_default_engine() noexcept {
  static Engine instance;  // C++11 thread-safe initialization
  return instance;
}

void Engine::run_backward(intrusive_ptr<Node> root,
                          const std::vector<OptionalTensor>& initial_grads,
                          const std::vector<std::function<void()>>& callbacks) {
  run_backward_impl(std::move(root), initial_grads, callbacks, /*hook=*/nullptr);
}

void Engine::run_backward_with_hook(
    intrusive_ptr<Node> root,
    const std::vector<OptionalTensor>& initial_grads,
    const std::vector<std::function<void()>>& callbacks,
    GraphTask::ValidateHook hook) {
  run_backward_impl(std::move(root), initial_grads, callbacks, hook);
}

void Engine::start_device_threads_once() noexcept {}
void Engine::stop() noexcept {}
void Engine::release_workers() noexcept {}

void run_backward(intrusive_ptr<Node> root,
                  const std::vector<OptionalTensor>& initial_grads,
                  const std::vector<std::function<void()>>& callbacks) {
  Engine::get_default_engine().run_backward(std::move(root), initial_grads, callbacks);
}

void run_backward_with_hook(intrusive_ptr<Node> root,
                            const std::vector<OptionalTensor>& initial_grads,
                            const std::vector<std::function<void()>>& callbacks,
                            GraphTask::ValidateHook hook) {
  Engine::get_default_engine().run_backward_with_hook(
      std::move(root), initial_grads, callbacks, hook);
}

}} // namespace vbt::autograd
