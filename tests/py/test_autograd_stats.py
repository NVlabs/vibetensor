# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

import vibetensor._C as C

@pytest.mark.parametrize("call_once", [False, True])
def test_autograd_stats_reset_and_keys(call_once):
    # Ensure submodule exists
    ag = C.autograd
    ag.reset_stats()
    s = ag.stats()
    keys = {
        "engine_runs",
        "engine_nodes_processed",
        "engine_edges_processed",
        "engine_duplicates_coalesced",
        "engine_callbacks_run",
        "wrapper_invocations",
        "wrapper_guard_skips",
        "graph_nodes_exposed",
        "graph_edges_exposed",
        "saved_tensors_packed",
        "saved_tensors_unpacked",
        "saved_tensors_hook_violations",
        "multi_grad_hooks_registered",
        "multi_grad_hooks_fired_all",
        "multi_grad_hooks_fired_any",
        "py_function_nodes_created",
        "py_function_nodes_applied",
        "py_function_backward_failures",
    }
    assert set(s.keys()) == keys
    for k in keys:
        assert isinstance(s[k], int)
        assert s[k] == 0

    # is_grad_enabled exists and defaults to True/False (thread-local)
    assert ag.is_grad_enabled() in (True, False)  # bool

    if call_once:
        # Call a vt op to bump wrapper_invocations
        from vibetensor.torch.factory import tensor
        a = tensor([[1.0, 2.0]], device="cpu")
        b = tensor([[3.0, 4.0]], device="cpu")
        C._call_op("vt::add", a, b)
        s2 = ag.stats()
        assert s2["wrapper_invocations"] >= 1
