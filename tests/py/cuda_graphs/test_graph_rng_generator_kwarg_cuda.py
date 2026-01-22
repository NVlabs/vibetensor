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


def test_generator_kwarg_parity_eager_vs_graph() -> None:
    """Using a CUDA Generator handle under graphs matches eager execution."""

    _require_cuda_or_skip()

    gen = vt.Generator(DEV_STR)

    vt.manual_seed(31415)
    st0 = vt.get_rng_state(DEV_STR)

    # Eager
    vt.set_rng_state(st0, device=DEV_STR)
    eager = vt.rand((16,), device=DEV_STR, generator=gen)
    eager_host = vt.cuda.from_device(eager)
    st_eager = vt.get_rng_state(DEV_STR)

    # Captured
    vt.set_rng_state(st0, device=DEV_STR)
    g = vcuda.graphs.CUDAGraph()
    with vcuda.graphs.graph(g):
        out = vt.rand((16,), device=DEV_STR, generator=gen)

    g.replay()
    vcuda.graphs.graph.default_capture_stream.synchronize()
    out_host = vt.cuda.from_device(out)
    st_graph = vt.get_rng_state(DEV_STR)

    np.testing.assert_array_equal(eager_host, out_host)
    assert st_eager == st_graph


def test_generator_kwarg_device_mismatch_inside_graph() -> None:
    """generator device mismatch errors surface under graphs as well."""

    _require_cuda_or_skip()

    gen_cpu = vt.Generator("cpu")

    g = vcuda.graphs.CUDAGraph()
    with vcuda.graphs.graph(g):
        with pytest.raises(ValueError, match="generator device mismatch"):
            vt.rand((4,), device=DEV_STR, generator=gen_cpu)
