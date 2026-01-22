# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import vibetensor.torch.cuda as cuda

if not hasattr(cuda, "cuda_graphs_stats"):
    pytest.skip("CUDA Graphs Python overlay not available for this build", allow_module_level=True)


def test_cuda_graphs_stats_shape_and_types() -> None:
    stats = cuda.cuda_graphs_stats()

    assert isinstance(stats, dict)
    assert "graphs" in stats
    assert "pools" in stats

    graphs = stats["graphs"]
    pools = stats["pools"]

    assert isinstance(graphs, dict)
    assert isinstance(pools, dict)

    expected_graph_keys = {
        "captures_started",
        "captures_ended",
        "denied_default_stream",
        "nested_capture_denied",
        "end_in_dtor",
        "end_in_dtor_errors",
        "graphs_instantiated",
        "graphs_replayed",
        "replay_nesting_errors",
        "unsupported_capture_mode",
        "capture_begin_invalid_state",
        "capture_end_invalid_state",
        "instantiate_invalid_state",
        "instantiate_errors",
        "replay_invalid_state",
        "replay_device_mismatch",
        "replay_errors",
        "graphs_reset",
        "reset_invalid_state",
        "reset_inflight_denied",
        "reset_errors",
        "allocator_capture_denied",
    }

    for key in expected_graph_keys:
        assert key in graphs
        assert isinstance(graphs[key], int)

    expected_pool_keys = {
        "device",
        "graphs_pools_created",
        "graphs_pools_active",
        "graphs_pools_released",
    }

    for key in expected_pool_keys:
        assert key in pools
        assert isinstance(pools[key], int)
