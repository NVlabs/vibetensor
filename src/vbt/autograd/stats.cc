// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/autograd/stats.h"
#include <atomic>

namespace vbt { namespace autograd {

static std::atomic<std::uint64_t> g_engine_runs{0};
static std::atomic<std::uint64_t> g_engine_nodes{0};
static std::atomic<std::uint64_t> g_engine_edges{0};
static std::atomic<std::uint64_t> g_engine_dupes{0};
static std::atomic<std::uint64_t> g_engine_cbs{0};
static std::atomic<std::uint64_t> g_wrap_inv{0};
static std::atomic<std::uint64_t> g_wrap_guard_skips{0};
static std::atomic<std::uint64_t> g_graph_nodes{0};
static std::atomic<std::uint64_t> g_graph_edges{0};
static std::atomic<std::uint64_t> g_saved_packed{0};
static std::atomic<std::uint64_t> g_saved_unpacked{0};
static std::atomic<std::uint64_t> g_saved_violations{0};
static std::atomic<std::uint64_t> g_multi_reg{0};
static std::atomic<std::uint64_t> g_multi_fired_all{0};
static std::atomic<std::uint64_t> g_multi_fired_any{0};
static std::atomic<std::uint64_t> g_py_fn_created{0};
static std::atomic<std::uint64_t> g_py_fn_applied{0};
static std::atomic<std::uint64_t> g_py_fn_failed{0};

AutogradStatsSnapshot stats() noexcept {
  AutogradStatsSnapshot s;
  s.engine_runs = g_engine_runs.load(std::memory_order_relaxed);
  s.engine_nodes_processed = g_engine_nodes.load(std::memory_order_relaxed);
  s.engine_edges_processed = g_engine_edges.load(std::memory_order_relaxed);
  s.engine_duplicates_coalesced = g_engine_dupes.load(std::memory_order_relaxed);
  s.engine_callbacks_run = g_engine_cbs.load(std::memory_order_relaxed);
  s.wrapper_invocations = g_wrap_inv.load(std::memory_order_relaxed);
  s.wrapper_guard_skips = g_wrap_guard_skips.load(std::memory_order_relaxed);
  s.graph_nodes_exposed = g_graph_nodes.load(std::memory_order_relaxed);
  s.graph_edges_exposed = g_graph_edges.load(std::memory_order_relaxed);
  s.saved_tensors_packed = g_saved_packed.load(std::memory_order_relaxed);
  s.saved_tensors_unpacked = g_saved_unpacked.load(std::memory_order_relaxed);
  s.saved_tensors_hook_violations = g_saved_violations.load(std::memory_order_relaxed);
  s.multi_grad_hooks_registered = g_multi_reg.load(std::memory_order_relaxed);
  s.multi_grad_hooks_fired_all = g_multi_fired_all.load(std::memory_order_relaxed);
  s.multi_grad_hooks_fired_any = g_multi_fired_any.load(std::memory_order_relaxed);
  s.py_function_nodes_created = g_py_fn_created.load(std::memory_order_relaxed);
  s.py_function_nodes_applied = g_py_fn_applied.load(std::memory_order_relaxed);
  s.py_function_backward_failures = g_py_fn_failed.load(std::memory_order_relaxed);
  return s;
}
void reset_stats() noexcept {
  g_engine_runs.store(0, std::memory_order_relaxed);
  g_engine_nodes.store(0, std::memory_order_relaxed);
  g_engine_edges.store(0, std::memory_order_relaxed);
  g_engine_dupes.store(0, std::memory_order_relaxed);
  g_engine_cbs.store(0, std::memory_order_relaxed);
  g_wrap_inv.store(0, std::memory_order_relaxed);
  g_wrap_guard_skips.store(0, std::memory_order_relaxed);
  g_graph_nodes.store(0, std::memory_order_relaxed);
  g_graph_edges.store(0, std::memory_order_relaxed);
  g_saved_packed.store(0, std::memory_order_relaxed);
  g_saved_unpacked.store(0, std::memory_order_relaxed);
  g_saved_violations.store(0, std::memory_order_relaxed);
  g_multi_reg.store(0, std::memory_order_relaxed);
  g_multi_fired_all.store(0, std::memory_order_relaxed);
  g_multi_fired_any.store(0, std::memory_order_relaxed);
  g_py_fn_created.store(0, std::memory_order_relaxed);
  g_py_fn_applied.store(0, std::memory_order_relaxed);
  g_py_fn_failed.store(0, std::memory_order_relaxed);
}

void _stats_bump_engine(std::uint64_t nodes, std::uint64_t edges,
                        std::uint64_t dupes, std::uint64_t callbacks) noexcept {
  g_engine_runs.fetch_add(1, std::memory_order_relaxed);
  g_engine_nodes.fetch_add(nodes, std::memory_order_relaxed);
  g_engine_edges.fetch_add(edges, std::memory_order_relaxed);
  g_engine_dupes.fetch_add(dupes, std::memory_order_relaxed);
  g_engine_cbs.fetch_add(callbacks, std::memory_order_relaxed);
}

void _stats_wrapper_invoked() noexcept {
  g_wrap_inv.fetch_add(1, std::memory_order_relaxed);
}

void _stats_wrapper_guard_skipped() noexcept {
  g_wrap_guard_skips.fetch_add(1, std::memory_order_relaxed);
}

void _stats_graph_nodes_exposed(std::uint64_t n) noexcept {
  g_graph_nodes.fetch_add(n, std::memory_order_relaxed);
}

void _stats_graph_edges_exposed(std::uint64_t n) noexcept {
  g_graph_edges.fetch_add(n, std::memory_order_relaxed);
}

void _stats_saved_tensors_packed(std::uint64_t n) noexcept {
  g_saved_packed.fetch_add(n, std::memory_order_relaxed);
}

void _stats_saved_tensors_unpacked(std::uint64_t n) noexcept {
  g_saved_unpacked.fetch_add(n, std::memory_order_relaxed);
}

void _stats_saved_tensors_hook_violations(std::uint64_t n) noexcept {
  g_saved_violations.fetch_add(n, std::memory_order_relaxed);
}

void _stats_multi_grad_registered(std::uint64_t n) noexcept {
  g_multi_reg.fetch_add(n, std::memory_order_relaxed);
}

void _stats_multi_grad_fired_all(std::uint64_t n) noexcept {
  g_multi_fired_all.fetch_add(n, std::memory_order_relaxed);
}

void _stats_multi_grad_fired_any(std::uint64_t n) noexcept {
  g_multi_fired_any.fetch_add(n, std::memory_order_relaxed);
}

void _stats_py_function_node_created(std::uint64_t n) noexcept {
  g_py_fn_created.fetch_add(n, std::memory_order_relaxed);
}

void _stats_py_function_node_applied(std::uint64_t n) noexcept {
  g_py_fn_applied.fetch_add(n, std::memory_order_relaxed);
}

void _stats_py_function_backward_failed(std::uint64_t n) noexcept {
  g_py_fn_failed.fetch_add(n, std::memory_order_relaxed);
}

}} // namespace vbt::autograd
