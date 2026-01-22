# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import struct

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


@pytest.mark.parametrize(
    "op_name, shape",
    [
        ("rand", (16,)),
        ("rand", (4, 5)),
        ("randn", (16,)),
        ("randn", (2, 3)),
    ],
)
def test_factory_parity_eager_vs_graph_float_ops(op_name, shape) -> None:
    """rand / randn under CUDA graphs match non-graph execution and RNG state."""

    _require_cuda_or_skip()

    op = getattr(vt, op_name)

    vt.manual_seed(12345)
    st0 = vt.get_rng_state(DEV_STR)

    # Eager run on CUDA
    vt.set_rng_state(st0, device=DEV_STR)
    eager = op(shape, device=DEV_STR)
    eager_host = vt.cuda.from_device(eager)
    st_eager = vt.get_rng_state(DEV_STR)

    # Captured run with the same initial state
    vt.set_rng_state(st0, device=DEV_STR)
    g = vcuda.graphs.CUDAGraph()
    with vcuda.graphs.graph(g):
        captured = op(shape, device=DEV_STR)

    g.replay()
    vcuda.graphs.graph.default_capture_stream.synchronize()
    captured_host = vt.cuda.from_device(captured)
    st_graph = vt.get_rng_state(DEV_STR)

    np.testing.assert_array_equal(eager_host, captured_host)
    assert st_eager == st_graph


@pytest.mark.parametrize("shape", [(0,), (0, 3)])
def test_factory_zero_length_does_not_advance_rng_state(shape) -> None:
    """Zero-length RNG workloads should not advance CUDA RNG state."""

    _require_cuda_or_skip()

    vt.manual_seed(2025)
    st0 = vt.get_rng_state(DEV_STR)

    # Eager
    vt.set_rng_state(st0, device=DEV_STR)
    out_eager = vt.rand(shape, device=DEV_STR)
    eager_state = vt.get_rng_state(DEV_STR)

    # No elements, so RNG state must stay the same.
    assert eager_state == st0
    assert list(out_eager.sizes) == list(shape)

    # Captured
    vt.set_rng_state(st0, device=DEV_STR)
    g = vcuda.graphs.CUDAGraph()
    with vcuda.graphs.graph(g):
        out_graph = vt.rand(shape, device=DEV_STR)

    g.replay()
    vcuda.graphs.graph.default_capture_stream.synchronize()
    graph_state = vt.get_rng_state(DEV_STR)

    assert graph_state == st0
    assert list(out_graph.sizes) == list(shape)


def test_factory_parity_eager_vs_graph_randint() -> None:
    """randint under CUDA graphs matches non-graph execution and RNG state."""

    _require_cuda_or_skip()

    shape = (17,)
    low, high = 3, 23

    vt.manual_seed(7)
    st0 = vt.get_rng_state(DEV_STR)

    vt.set_rng_state(st0, device=DEV_STR)
    eager = vt.randint(low, high, shape, device=DEV_STR)
    eager_host = vt.cuda.from_device(eager)
    st_eager = vt.get_rng_state(DEV_STR)

    vt.set_rng_state(st0, device=DEV_STR)
    g = vcuda.graphs.CUDAGraph()
    with vcuda.graphs.graph(g):
        captured = vt.randint(low, high, shape, device=DEV_STR)

    g.replay()
    vcuda.graphs.graph.default_capture_stream.synchronize()
    captured_host = vt.cuda.from_device(captured)
    st_graph = vt.get_rng_state(DEV_STR)

    np.testing.assert_array_equal(eager_host, captured_host)
    assert st_eager == st_graph


def test_factory_parity_eager_vs_graph_rand_like() -> None:
    """rand_like / randn_like / randint_like behave the same under graphs."""

    _require_cuda_or_skip()

    base = vt.rand((8, 4), device=DEV_STR)
    base_int = vt.randint(0, 10, base.sizes, device=DEV_STR)

    vt.manual_seed(101)
    st0 = vt.get_rng_state(DEV_STR)

    # Eager: mix of *_like factories
    vt.set_rng_state(st0, device=DEV_STR)
    eager_u = vt.rand_like(base, device=DEV_STR)
    eager_n = vt.randn_like(base, device=DEV_STR)
    eager_i = vt.randint_like(base_int, 0, 10, device=DEV_STR)
    eager_u_h = vt.cuda.from_device(eager_u)
    eager_n_h = vt.cuda.from_device(eager_n)
    eager_i_h = vt.cuda.from_device(eager_i)
    st_eager = vt.get_rng_state(DEV_STR)

    # Captured
    vt.set_rng_state(st0, device=DEV_STR)
    g = vcuda.graphs.CUDAGraph()
    with vcuda.graphs.graph(g):
        out_u = vt.rand_like(base, device=DEV_STR)
        out_n = vt.randn_like(base, device=DEV_STR)
        out_i = vt.randint_like(base_int, 0, 10, device=DEV_STR)

    g.replay()
    vcuda.graphs.graph.default_capture_stream.synchronize()

    out_u_h = vt.cuda.from_device(out_u)
    out_n_h = vt.cuda.from_device(out_n)
    out_i_h = vt.cuda.from_device(out_i)
    st_graph = vt.get_rng_state(DEV_STR)

    np.testing.assert_array_equal(eager_u_h, out_u_h)
    np.testing.assert_array_equal(eager_n_h, out_n_h)
    np.testing.assert_array_equal(eager_i_h, out_i_h)
    assert st_eager == st_graph


def _unpack_state(bts: bytes) -> tuple[int, int]:
    return struct.unpack("<QQ", bts)


def test_factory_parity_preserves_full_rng_state() -> None:
    """Parity tests also preserve full [seed, offset] for the CUDA generator."""

    _require_cuda_or_skip()

    vt.manual_seed(4242)
    st0 = vt.get_rng_state(DEV_STR)

    vt.set_rng_state(st0, device=DEV_STR)
    vt.rand((32,), device=DEV_STR)
    eager_state = vt.get_rng_state(DEV_STR)

    vt.set_rng_state(st0, device=DEV_STR)
    g = vcuda.graphs.CUDAGraph()
    with vcuda.graphs.graph(g):
        vt.rand((32,), device=DEV_STR)

    g.replay()
    vcuda.graphs.graph.default_capture_stream.synchronize()
    graph_state = vt.get_rng_state(DEV_STR)

    assert eager_state == graph_state
    assert _unpack_state(eager_state)[0] == _unpack_state(st0)[0]
