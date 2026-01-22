// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

namespace vbt { namespace autograd {

struct AutogradStatsSnapshot {
  std::uint64_t engine_runs{0};
  std::uint64_t engine_nodes_processed{0};
  std::uint64_t engine_edges_processed{0};
  std::uint64_t engine_duplicates_coalesced{0};
  std::uint64_t engine_callbacks_run{0};
  std::uint64_t wrapper_invocations{0};
  std::uint64_t wrapper_guard_skips{0};
  // Graph inspection counters
  std::uint64_t graph_nodes_exposed{0};
  std::uint64_t graph_edges_exposed{0};
  // Saved-tensor hook counters
  std::uint64_t saved_tensors_packed{0};
  std::uint64_t saved_tensors_unpacked{0};
  std::uint64_t saved_tensors_hook_violations{0};
  // Multi-grad hook counters
  std::uint64_t multi_grad_hooks_registered{0};
  std::uint64_t multi_grad_hooks_fired_all{0};
  std::uint64_t multi_grad_hooks_fired_any{0};
  // Python custom Function node counters
  std::uint64_t py_function_nodes_created{0};
  std::uint64_t py_function_nodes_applied{0};
  std::uint64_t py_function_backward_failures{0};
};

// Return a best‑effort snapshot of autograd counters (not atomic across fields).
AutogradStatsSnapshot stats() noexcept;

// Reset all counters to zero. Other threads may concurrently bump.
void reset_stats() noexcept;

}} // namespace vbt::autograd
