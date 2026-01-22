// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

namespace vbt { namespace autograd {

// Internal bumpers used by Engine/Wrapper/Dispatcher. Not part of public API.
void _stats_bump_engine(std::uint64_t nodes, std::uint64_t edges,
                        std::uint64_t dupes, std::uint64_t callbacks) noexcept;
void _stats_wrapper_invoked() noexcept;
void _stats_wrapper_guard_skipped() noexcept;

void _stats_graph_nodes_exposed(std::uint64_t n) noexcept;
void _stats_graph_edges_exposed(std::uint64_t n) noexcept;
void _stats_saved_tensors_packed(std::uint64_t n) noexcept;
void _stats_saved_tensors_unpacked(std::uint64_t n) noexcept;
void _stats_saved_tensors_hook_violations(std::uint64_t n) noexcept;
void _stats_multi_grad_registered(std::uint64_t n) noexcept;
void _stats_multi_grad_fired_all(std::uint64_t n) noexcept;
void _stats_multi_grad_fired_any(std::uint64_t n) noexcept;
void _stats_py_function_node_created(std::uint64_t n) noexcept;
void _stats_py_function_node_applied(std::uint64_t n) noexcept;
void _stats_py_function_backward_failed(std::uint64_t n) noexcept;

}} // namespace vbt::autograd
