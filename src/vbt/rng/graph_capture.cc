// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/rng/graph_capture.h"
#include "vbt/logging/logging.h"

#include <atomic>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1,
              "VBT_WITH_CUDA must be 0 or 1");

namespace vbt {
namespace rng {
namespace graph_capture {

#if VBT_WITH_CUDA

namespace {

struct ActiveCapture {
  bool            capturing{false};
  std::thread::id owner_thread{};     // thread that owns this capture
  PhiloxState     base_state{};       // gen.get_state() at begin
  std::uint64_t   total_blocks{0};    // sum of total_blocks for this capture
  bool            is_default{false};  // is this the default_cuda(device) gen?
#ifdef VBT_INTERNAL_TESTS
  std::vector<SliceRecord> slices;    // per-op slices (capture order)
#endif
};

std::mutex g_mu;  // guards g_active and recompute_any_active_locked()
// Generators are non-copyable/non-movable (see generator.h), so their
// addresses are stable for the lifetime of each instance; using raw
// pointers as map keys is therefore safe.
std::unordered_map<CudaGenerator*, ActiveCapture> g_active;
std::atomic<bool> g_any_active{false};  // fast-path hint: any capture alive?

#ifdef VBT_INTERNAL_TESTS
std::mutex g_dbg_mu;
std::unordered_map<int, CaptureSummary> g_last_summary_by_device;
#endif

// Helper: recompute g_any_active under g_mu.
void recompute_any_active_locked() {
  bool any = false;
  for (const auto& kv : g_active) {
    if (kv.second.capturing) {
      any = true;
      break;
    }
  }
  g_any_active.store(any, std::memory_order_release);
}

bool is_default_generator(const CudaGenerator& gen) {
  auto dev = gen.device();
  if (dev.type != kDLCUDA) {
    return false;
  }
  int idx = dev.index;
  try {
    CudaGenerator& def = default_cuda(idx);
    return &def == &gen;
  } catch (const std::exception& e) {
    VBT_LOG(WARNING) << "[rng][graph_capture] default_cuda(" << idx
                     << ") lookup failed: " << e.what();
  } catch (...) {
    VBT_LOG(WARNING) << "[rng][graph_capture] default_cuda(" << idx
                     << ") lookup failed with non-std exception";
  }
  // Conservative fallback: treat as non-default generator.
  return false;
}

} // anonymous namespace

void on_cuda_graph_capture_begin(CudaGenerator& gen) {
  std::lock_guard<std::mutex> lock(g_mu);
  ActiveCapture& cap = g_active[&gen];
  if (cap.capturing) {
    throw std::runtime_error(kErrCudaRngOverlappingCapture);
  }

  cap.capturing    = true;
  cap.owner_thread = std::this_thread::get_id();
  cap.base_state   = gen.get_state();
  cap.total_blocks = 0;
  cap.is_default   = is_default_generator(gen);
#ifdef VBT_INTERNAL_TESTS
  cap.slices.clear();
#endif

  g_any_active.store(true, std::memory_order_release);
}

PhiloxState reserve_blocks_for_graph_aware_cuda_op(
    CudaGenerator& gen,
    std::uint64_t total_blocks,
    std::uint32_t outputs_per_block,
    RngOpTag      op_tag,
    bool          stream_is_capturing) {
  (void)outputs_per_block;  // used only for tests/debug slices under VBT_INTERNAL_TESTS
  (void)op_tag;

  if (total_blocks == 0) {
    // Degenerate zero-element op: preserve existing RNG behavior and do
    // not interact with capture state at all.
    return gen.get_state();
  }

  // Fast path: no captures anywhere.
  if (!g_any_active.load(std::memory_order_acquire)) {
    return gen.reserve_blocks(total_blocks);
  }

  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_active.find(&gen);
  if (it == g_active.end() || !it->second.capturing) {
    // Another generator is capturing, or this one has already ended.
    return gen.reserve_blocks(total_blocks);
  }

  ActiveCapture& cap = it->second;

  if (!cap.is_default) {
    throw std::runtime_error(kErrCudaRngNonDefaultGeneratorInGraph);
  }

  if (cap.owner_thread != std::this_thread::get_id()) {
    throw std::runtime_error(kErrCudaRngConcurrentUseDuringCapture);
  }

  if (!stream_is_capturing) {
    throw std::runtime_error(kErrCudaRngUseOnNonCaptureStream);
  }

  // Guard the running total so that base_state.offset + total_blocks never
  // overflows std::uint64_t.
  const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();

  if (total_blocks > max - cap.total_blocks) {
    throw std::runtime_error(kErrCudaRngCaptureBlocksOverflow);
  }

  std::uint64_t new_total = cap.total_blocks + total_blocks;
  if (new_total > max - cap.base_state.offset) {
    throw std::runtime_error(kErrCudaRngCaptureBlocksOverflow);
  }

  const std::uint64_t base_offset = cap.base_state.offset + cap.total_blocks;
  PhiloxState slice_state{cap.base_state.seed, base_offset};

#ifdef VBT_INTERNAL_TESTS
  SliceRecord rec{};
  rec.state             = slice_state;
  rec.total_blocks      = total_blocks;
  rec.outputs_per_block = outputs_per_block;
  rec.op_tag            = op_tag;
  cap.slices.push_back(rec);
#endif

  cap.total_blocks = new_total;
  return slice_state;
}

CaptureSummary on_cuda_graph_capture_end_success(CudaGenerator& gen) {
  CaptureSummary summary{};

  {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_active.find(&gen);
    if (it == g_active.end() || !it->second.capturing) {
      throw std::logic_error(kErrCudaRngCaptureEndWithoutActiveCapture);
    }

    ActiveCapture& cap = it->second;
    if (cap.owner_thread != std::this_thread::get_id()) {
      throw std::logic_error(kErrCudaRngCaptureEndWrongThread);
    }

    PhiloxState cur = gen.get_state();
    if (cur.seed != cap.base_state.seed || cur.offset != cap.base_state.offset) {
      // Generator was mutated during capture (manual_seed/set_state or
      // concurrent non-graph reserve_blocks). Tear down capture and report.
      cap.capturing = false;
      g_active.erase(it);
      recompute_any_active_locked();
      throw std::logic_error(kErrCudaRngGeneratorStateMutatedDuringCapture);
    }

    summary.base_state   = cap.base_state;
    summary.total_blocks = cap.total_blocks;
#ifdef VBT_INTERNAL_TESTS
    summary.slices       = cap.slices;
#endif

    cap.capturing = false;
    g_active.erase(it);
    recompute_any_active_locked();
  }

  if (summary.total_blocks > 0) {
    // In well-formed usage (no concurrent RNG on gen), this is exactly
    // equivalent to calling reserve_blocks(summary.total_blocks) from
    // base_state.offset.
    // In misuse scenarios (e.g., someone races reserve_blocks after the
    // equality check above), this call may throw the generator's own
    // overflow error ("rng: offset overflow: ..." in generator.cc), but
    // it prevents wraparound of the offset.
    (void)gen.reserve_blocks(summary.total_blocks);
  }

#ifdef VBT_INTERNAL_TESTS
  auto dev = gen.device();
  if (dev.type == kDLCUDA) {
    std::lock_guard<std::mutex> dbg_lock(g_dbg_mu);
    g_last_summary_by_device[dev.index] = summary;
  }
#endif

  return summary;
}

void on_cuda_graph_capture_abort(CudaGenerator& gen) noexcept {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_active.find(&gen);
  if (it != g_active.end()) {
    it->second.capturing    = false;
    it->second.total_blocks = 0;
#ifdef VBT_INTERNAL_TESTS
    it->second.slices.clear();
#endif
    g_active.erase(it);
  }
  recompute_any_active_locked();
}

bool is_generator_capture_active(const CudaGenerator& gen) noexcept {
  if (!g_any_active.load(std::memory_order_acquire)) {
    return false;
  }
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_active.find(const_cast<CudaGenerator*>(&gen));
  return it != g_active.end() && it->second.capturing;
}

#ifdef VBT_INTERNAL_TESTS
std::optional<CaptureSummary>
  debug_last_capture_summary_for_cuda_device(int device_index) {
  std::lock_guard<std::mutex> lock(g_dbg_mu);
  auto it = g_last_summary_by_device.find(device_index);
  if (it == g_last_summary_by_device.end()) {
    return std::nullopt;
  }
  return it->second;
}
#endif

#else  // !VBT_WITH_CUDA

PhiloxState reserve_blocks_for_graph_aware_cuda_op(
    CudaGenerator& gen,
    std::uint64_t total_blocks,
    std::uint32_t /*outputs_per_block*/,
    RngOpTag      /*op_tag*/,
    bool          /*stream_is_capturing*/) {
  return gen.reserve_blocks(total_blocks);
}

void on_cuda_graph_capture_begin(CudaGenerator& /*gen*/) {}

CaptureSummary on_cuda_graph_capture_end_success(CudaGenerator& gen) {
  CaptureSummary summary{};
  summary.base_state   = gen.get_state();
  summary.total_blocks = 0;
  return summary;
}

void on_cuda_graph_capture_abort(CudaGenerator& /*gen*/) noexcept {}

bool is_generator_capture_active(const CudaGenerator& /*gen*/) noexcept {
  return false;
}

#ifdef VBT_INTERNAL_TESTS
std::optional<CaptureSummary>
  debug_last_capture_summary_for_cuda_device(int /*device_index*/) {
  return std::nullopt;
}
#endif

#endif // VBT_WITH_CUDA

} // namespace graph_capture
} // namespace rng
} // namespace vbt
