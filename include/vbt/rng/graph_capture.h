// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#ifdef VBT_INTERNAL_TESTS
#include <optional>
#include <vector>
#endif

#include "vbt/rng/generator.h"  // PhiloxState, CudaGenerator

namespace vbt {
namespace rng {
namespace graph_capture {

// Tag for the RNG operation type using this capture slice.
// Used only for tests/debug in Step 1.
enum class RngOpTag : std::uint8_t {
  Uniform   = 0,
  Normal    = 1,
  Bernoulli = 2,
  Randint   = 3,
};

struct SliceRecord {
  PhiloxState   state{};             // {seed, base_offset} for this op
  std::uint64_t total_blocks{0};     // Philox blocks consumed by this op
  std::uint32_t outputs_per_block{0};
  RngOpTag      op_tag{RngOpTag::Uniform};
};

struct CaptureSummary {
  PhiloxState              base_state{};    // generator state at capture_begin
  std::uint64_t            total_blocks{0}; // sum(total_blocks over ops)
#ifdef VBT_INTERNAL_TESTS
  std::vector<SliceRecord> slices;          // in capture order (tests/debug)
#endif
};

// Pinned error message constants (single source of truth; tests assert on these).
inline constexpr const char* kErrCudaRngOverlappingCapture =
  "rng: CUDA generator for this device is already being captured by another CUDA graph";
inline constexpr const char* kErrCudaRngConcurrentUseDuringCapture =
  "rng: CUDA RNG operations on this generator are only allowed from the capturing thread while CUDA Graph capture is active";
inline constexpr const char* kErrCudaRngUseOnNonCaptureStream =
  "rng: CUDA RNG operations on this generator are only allowed on the captured stream while CUDA Graph capture is active";
inline constexpr const char* kErrCudaRngCaptureEndWithoutActiveCapture =
  "rng: capture_end called without active RNG capture";
inline constexpr const char* kErrCudaRngCaptureEndWrongThread =
  "rng: CUDA Graph capture_end called on a thread that does not own RNG capture for this generator";
inline constexpr const char* kErrCudaRngGeneratorStateMutatedDuringCapture =
  "rng: generator state mutated during CUDA Graph capture";
inline constexpr const char* kErrCudaRngMutationDuringCapture =
  "rng: generator state mutation is forbidden while CUDA Graph capture is active";  // For bindings in later steps
inline constexpr const char* kErrCudaRngNonDefaultGeneratorInGraph =
  "rng: non-default CUDA generators are not supported inside CUDA Graph captures (v1)";
inline constexpr const char* kErrCudaRngCaptureBlocksOverflow =
  "rng: capture would overflow Philox offset for this generator";

// Exception-type policy:
// - Misuse and overflow errors use std::runtime_error.
// - Lifecycle/state-machine errors in on_cuda_graph_capture_end_success use std::logic_error.

// Reserve Philox blocks for a CUDA RNG op in a graph-aware way.
//
// Parameters:
//   gen               - CUDA generator for a specific device.
//   total_blocks      - ceil_div(numel, P) for this op (P = outputs_per_block).
//   outputs_per_block - P (4 for uniform/normal/bernoulli, 2 for randint).
//   op_tag            - identifies the op for tests/debug; ignored by logic.
//   stream_is_capturing - true iff the current CUDA stream on this device is
//                         under CUDA Graph capture.
//
// Semantics (VBT_WITH_CUDA == 1):
//   * If total_blocks == 0: returns gen.get_state() (current {seed,offset})
//     without touching capture state or generator offset (zero-element op).
//   * If no RNG capture is active for any generator: behaves exactly like
//       gen.reserve_blocks(total_blocks).
//   * If a capture is active for this generator:
//       - If the active capture was started for a non-default generator,
//         throws std::runtime_error(kErrCudaRngNonDefaultGeneratorInGraph).
//       - If called from a non-owner thread, throws
//         std::runtime_error(kErrCudaRngConcurrentUseDuringCapture).
//       - If stream_is_capturing == false, throws
//         std::runtime_error(kErrCudaRngUseOnNonCaptureStream).
//       - Otherwise, returns a PhiloxState {seed, base_offset} from the
//         capture plan without mutating gen.offset. The sequence of slices
//         matches what repeated reserve_blocks(total_blocks) calls would
//         have produced starting from the capture base state, up to overflow
//         checks; overflow in the plan throws
//         std::runtime_error(kErrCudaRngCaptureBlocksOverflow) and leaves
//         the capture active.
//
// Semantics (VBT_WITH_CUDA == 0):
//   * Always forwards to gen.reserve_blocks(total_blocks) (including the
//     blocks==0 fast-path), ignoring stream_is_capturing and op_tag.
//   * No global capture state is created; behavior is identical to today.
PhiloxState reserve_blocks_for_graph_aware_cuda_op(
    CudaGenerator& gen,
    std::uint64_t total_blocks,
    std::uint32_t outputs_per_block,
    RngOpTag      op_tag,
    bool          stream_is_capturing);

// Begin recording an RNG capture for this generator.
//
// Pre-conditions (caller responsibility; violations are misuse):
//   * No other thread is concurrently calling begin/end/abort for this gen.
//   * No other thread is concurrently using gen for RNG operations.
//
// Effects (VBT_WITH_CUDA == 1):
//   * If a capture is already active for this generator, throws
//     std::runtime_error(kErrCudaRngOverlappingCapture).
//   * Otherwise, records:
//       - base_state = gen.get_state();
//       - owner_thread = std::this_thread::get_id();
//       - total_blocks = 0; capturing = true.
//   * Marks that at least one capture is active globally.
//
// Effects (VBT_WITH_CUDA == 0): no-op.
void on_cuda_graph_capture_begin(CudaGenerator& gen);

// Finalize a successful RNG capture for this generator and advance the
// generator offset by the total number of captured blocks.
//
// Semantics (VBT_WITH_CUDA == 1):
//   * Looks up active capture for &gen.
//   * If no active capture exists, throws std::logic_error(
//       kErrCudaRngCaptureEndWithoutActiveCapture).
//   * If called from a non-owner thread, throws std::logic_error(
//       kErrCudaRngCaptureEndWrongThread).
//   * If gen.get_state() != base_state recorded at begin (seed or offset
//     differs), tears down capture bookkeeping and throws std::logic_error(
//       kErrCudaRngGeneratorStateMutatedDuringCapture).
//   * Otherwise:
//       - Produces a CaptureSummary {base_state, total_blocks, slices (tests)}.
//       - Clears capture state for this generator and recomputes the global
//         "any active" flag.
//       - If total_blocks > 0, calls gen.reserve_blocks(total_blocks) to
//         actually advance the generator offset. This may still throw the
//         generator's own overflow error ("rng: offset overflow: ...") in
//         gross misuse scenarios (e.g., racing non-graph reserves), but
//         prevents wraparound.
//
// Semantics (VBT_WITH_CUDA == 0):
//   * Returns a CaptureSummary with base_state = gen.get_state(),
//     total_blocks = 0, and empty slices; does not change generator state.
CaptureSummary on_cuda_graph_capture_end_success(CudaGenerator& gen);

// Abort an RNG capture (e.g., capture_end failure or user forgetting to
// call capture_end and CUDAGraph's RAII fallback firing).
//
// Semantics (VBT_WITH_CUDA == 1):
//   * Best-effort, noexcept cleanup: marks the capture for this generator as
//     not capturing, clears any recorded slices and totals, and updates the
//     global "any active" flag. Does not touch generator state.
//
// Semantics (VBT_WITH_CUDA == 0): no-op.
void on_cuda_graph_capture_abort(CudaGenerator& gen) noexcept;

// Query whether this CUDA generator is currently participating in an RNG
// capture. Used by higher layers (e.g., Python bindings) to guard generator
// mutations during capture.
//
// Semantics (VBT_WITH_CUDA == 1):
//   * If no capture is active for any generator, returns false quickly.
//   * Otherwise, checks whether there is an active capture record for &gen.
//   * Contract:
//       - For a generator with a well-formed capture sequence (begin(),
//         zero or more reserve_blocks_for_graph_aware_cuda_op(), then a
//         single end_success() or abort()),
//           - After begin() returns and until end_success()/abort() has
//             fully completed, returns true.
//           - After end_success()/abort() completes, returns false.
//       - Calls that race with begin/end/abort (i.e., while those
//         functions are executing) are misuse and may see stale values.
//
// Semantics (VBT_WITH_CUDA == 0): always returns false.
bool is_generator_capture_active(const CudaGenerator& gen) noexcept;

#ifdef VBT_INTERNAL_TESTS
// Optional debug hook to inspect the last successful CaptureSummary per
// CUDA device. Returns std::nullopt if this device has not seen a
// successful RNG capture in this process.
std::optional<CaptureSummary>
  debug_last_capture_summary_for_cuda_device(int device_index);
#endif

} // namespace graph_capture
} // namespace rng
} // namespace vbt
