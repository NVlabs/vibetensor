# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import struct

import pytest
import vibetensor.torch.cuda as vcuda

if not hasattr(vcuda, "graphs"):
    pytest.skip("CUDA Graphs Python overlay not available for this build", allow_module_level=True)

from vibetensor import _C
import vibetensor.torch as vt
from vibetensor.torch.cuda import graphs as vgraphs


ERR = getattr(
    _C,
    "_ERR_CUDA_RNG_MUTATION_DURING_CAPTURE",
    "rng: generator state mutation is forbidden while CUDA Graph capture is active",
)


def _has_cuda() -> bool:
    try:
        return bool(getattr(_C, "_has_cuda", False) and int(_C._cuda_device_count()) > 0)  # type: ignore[attr-defined]
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required for graph RNG guard tests")
def test_cuda_rng_mutations_forbidden_during_capture() -> None:
    dev_str = "cuda:0"

    # Establish a known baseline RNG state for the default CUDA generator.
    vt.manual_seed(123)
    baseline = vt.get_rng_state(dev_str)

    g = vgraphs.CUDAGraph()
    with vgraphs.graph(g):
        # Module-level CUDA helpers must be guarded.
        with pytest.raises(RuntimeError, match=re.escape(ERR)):
            vt.cuda.manual_seed(9)
        with pytest.raises(RuntimeError, match=re.escape(ERR)):
            vt.cuda.manual_seed_all(9)

        with pytest.raises(RuntimeError, match=re.escape(ERR)):
            vt.cuda.set_rng_state(baseline, device=0)

        # Generator API for CUDA must also be guarded.
        gen = vt.Generator(dev_str)
        with pytest.raises(RuntimeError, match=re.escape(ERR)):
            gen.manual_seed(5)
        with pytest.raises(RuntimeError, match=re.escape(ERR)):
            gen.set_state(baseline)

        # Top-level set_rng_state / manual_seed must surface guard errors
        # when they reach the capturing device.
        with pytest.raises(RuntimeError, match=re.escape(ERR)):
            vt.set_rng_state(baseline, device=dev_str)
        with pytest.raises(RuntimeError, match=re.escape(ERR)):
            vt.manual_seed(321)

    # After capture, the CUDA RNG state for the capturing device should
    # remain exactly as it was before entering the graph.
    assert vt.get_rng_state(dev_str) == baseline


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required for graph RNG guard tests")
def test_cuda_rng_mutations_allowed_outside_capture() -> None:
    dev_str = "cuda:0"

    # All of these calls should succeed when no CUDA Graph capture is active.
    vt.manual_seed(11)
    vt.cuda.manual_seed(7)
    vt.cuda.manual_seed_all(13)

    state = vt.cuda.get_rng_state(0)
    vt.cuda.set_rng_state(state, device=0)

    gen = vt.Generator(dev_str)
    gen.manual_seed(42)
    gen_state = gen.get_state()
    gen.set_state(gen_state)

    vt.set_rng_state(gen_state, device=dev_str)

    # Sanity check: RNG state for the device is a 16-byte blob.
    final = vt.cuda.get_rng_state(0)
    assert isinstance(final, (bytes, bytearray)) and len(final) == 16


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required for graph RNG guard tests")
def test_read_only_rng_helpers_allowed_during_capture() -> None:
    dev_idx = 0
    dev_str = "cuda:0"

    # Establish a known baseline RNG state for the default CUDA generator.
    vt.cuda.manual_seed_all(314159)
    baseline = vt.cuda.get_rng_state(dev_idx)
    assert isinstance(baseline, (bytes, bytearray)) and len(baseline) == 16

    seed0, offset0 = struct.unpack("<QQ", baseline)

    g = vgraphs.CUDAGraph()
    with vgraphs.graph(g):
        # Read-only helpers must succeed and report the same state while
        # capture is active.
        s_init = vt.cuda.initial_seed(dev_idx)
        st_dev = vt.cuda.get_rng_state(dev_idx)
        st_mod = vt.get_rng_state(dev_str)

        assert isinstance(s_init, int)
        assert isinstance(st_dev, (bytes, bytearray)) and len(st_dev) == 16
        assert isinstance(st_mod, (bytes, bytearray)) and len(st_mod) == 16
        assert struct.unpack("<QQ", st_dev) == (seed0, offset0)
        assert struct.unpack("<QQ", st_mod) == (seed0, offset0)

    # After capture, the RNG state for the device is unchanged.
    after = vt.cuda.get_rng_state(dev_idx)
    assert after == baseline
