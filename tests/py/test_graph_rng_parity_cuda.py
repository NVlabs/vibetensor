# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import struct

import numpy as np
import pytest

import vibetensor.torch as vt
import vibetensor.torch.cuda as vcuda

if not hasattr(vcuda, "graphs"):
    pytest.skip("CUDA Graphs Python overlay not available for this build", allow_module_level=True)

from vibetensor import _C


def _has_cuda() -> bool:
    return bool(
        getattr(_C, "_has_cuda", False)
        and int(getattr(_C, "_cuda_device_count", lambda: 0)()) > 0  # type: ignore[attr-defined]
    )


def _unpack_state(bts: bytes) -> tuple[int, int]:
    return struct.unpack("<QQ", bts)


@pytest.mark.parametrize("shape", [(16,), (4, 5)])
def test_graph_rand_parity_cuda(shape) -> None:
    """rand under CUDA graphs matches non-graph execution and RNG state.

    This exercises the end-to-end RNG × CUDA Graphs integration at the Python
    level using the default CUDA generator on device 0.
    """

    if not _has_cuda():
        pytest.skip("CUDA required for graph RNG parity tests")

    dev_str = "cuda:0"

    # Establish a common starting RNG state for device 0.
    vt.manual_seed(12345)
    st0 = vt.get_rng_state(dev_str)

    # Non-graph run: fill a preallocated CUDA tensor with uniform_.
    vt.set_rng_state(st0, device=dev_str)
    sizes = list(shape)
    eager = _C._cuda_empty(sizes, "float32", 0)  # type: ignore[attr-defined]
    _C._uniform_(eager, 0.0, 1.0)
    eager_host = vt.cuda.from_device(eager)
    st_eager = vt.get_rng_state(dev_str)

    # Graph capture run using the same initial RNG state.
    vt.set_rng_state(st0, device=dev_str)
    captured = _C._cuda_empty(sizes, "float32", 0)  # type: ignore[attr-defined]
    g = vcuda.graphs.CUDAGraph()
    with vcuda.graphs.graph(g):
        _C._uniform_(captured, 0.0, 1.0)

    # First replay materializes the captured RNG workload into ``captured``.
    g.replay()
    vcuda.graphs.graph.default_capture_stream.synchronize()

    captured_host = vt.cuda.from_device(captured)
    st_graph = vt.get_rng_state(dev_str)

    np.testing.assert_array_equal(eager_host, captured_host)
    assert st_eager == st_graph


def test_graph_rand_replay_determinism_cuda() -> None:
    """Replays produce identical outputs and do not advance RNG state."""

    if not _has_cuda():
        pytest.skip("CUDA required for graph RNG replay tests")

    dev_str = "cuda:0"

    vt.manual_seed(2025)
    st0 = vt.get_rng_state(dev_str)

    # Capture a simple rand workload via in-place uniform_ on a preallocated tensor.
    vt.set_rng_state(st0, device=dev_str)
    y = _C._cuda_empty([32], "float32", 0)  # type: ignore[attr-defined]
    g = vcuda.graphs.CUDAGraph()
    with vcuda.graphs.graph(g):
        _C._uniform_(y, 0.0, 1.0)

    # Materialize the captured graph once to obtain a baseline output.
    g.replay()
    vcuda.graphs.graph.default_capture_stream.synchronize()
    baseline = vt.cuda.from_device(y).copy()
    st_after_capture = vt.get_rng_state(dev_str)

    # Multiple replays must reuse the captured Philox slices and leave the
    # default CUDA generator state unchanged.
    for _ in range(3):
        g.replay()
        vcuda.graphs.graph.default_capture_stream.synchronize()
        replay_host = vt.cuda.from_device(y)
        np.testing.assert_array_equal(replay_host, baseline)
        assert vt.get_rng_state(dev_str) == st_after_capture
