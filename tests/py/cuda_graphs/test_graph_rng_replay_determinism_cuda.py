# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
import vibetensor.torch.cuda as vcuda

if not hasattr(vcuda, "graphs"):
    pytest.skip("CUDA Graphs Python overlay not available for this build", allow_module_level=True)

from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _rng_test_utils import _require_cuda_or_skip


DEV_STR = "cuda:0"


def test_graph_rand_replay_determinism_factory() -> None:
    """Replays of a captured rand workload are deterministic and frozen."""

    _require_cuda_or_skip()

    vt.manual_seed(2024)
    st0 = vt.get_rng_state(DEV_STR)

    vt.set_rng_state(st0, device=DEV_STR)
    g = vcuda.graphs.CUDAGraph()
    with vcuda.graphs.graph(g):
        y = vt.rand((32,), device=DEV_STR)

    # First replay establishes baseline output and RNG state.
    g.replay()
    vcuda.graphs.graph.default_capture_stream.synchronize()
    baseline = vt.cuda.from_device(y).copy()
    state_after_first = vt.get_rng_state(DEV_STR)

    for _ in range(3):
        g.replay()
        vcuda.graphs.graph.default_capture_stream.synchronize()
        replay_host = vt.cuda.from_device(y)
        np.testing.assert_array_equal(replay_host, baseline)
        assert vt.get_rng_state(DEV_STR) == state_after_first


def test_graph_multi_rng_ops_parity_and_replay() -> None:
    """Graphs with multiple RNG ops behave like eager and replay deterministically."""

    _require_cuda_or_skip()

    vt.manual_seed(17)
    st0 = vt.get_rng_state(DEV_STR)

    # Eager: rand then randn.
    vt.set_rng_state(st0, device=DEV_STR)
    eager_u = vt.rand((8,), device=DEV_STR)
    eager_n = vt.randn((8,), device=DEV_STR)
    eager_u_h = vt.cuda.from_device(eager_u)
    eager_n_h = vt.cuda.from_device(eager_n)
    state_eager = vt.get_rng_state(DEV_STR)

    # Captured: same sequence.
    vt.set_rng_state(st0, device=DEV_STR)
    g = vcuda.graphs.CUDAGraph()
    with vcuda.graphs.graph(g):
        out_u = vt.rand((8,), device=DEV_STR)
        out_n = vt.randn((8,), device=DEV_STR)

    g.replay()
    vcuda.graphs.graph.default_capture_stream.synchronize()
    out_u_h = vt.cuda.from_device(out_u)
    out_n_h = vt.cuda.from_device(out_n)
    state_after_capture = vt.get_rng_state(DEV_STR)

    np.testing.assert_array_equal(eager_u_h, out_u_h)
    np.testing.assert_array_equal(eager_n_h, out_n_h)
    assert state_eager == state_after_capture

    # Additional replays must reproduce the same outputs and keep state fixed.
    for _ in range(2):
        g.replay()
        vcuda.graphs.graph.default_capture_stream.synchronize()
        out_u_r = vt.cuda.from_device(out_u)
        out_n_r = vt.cuda.from_device(out_n)
        np.testing.assert_array_equal(out_u_r, out_u_h)
        np.testing.assert_array_equal(out_n_r, out_n_h)
        assert vt.get_rng_state(DEV_STR) == state_after_capture
