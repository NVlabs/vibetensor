# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import pytest

from vibetensor import _C as _C
import vibetensor.torch as vt
import vibetensor.torch.cuda as vcuda

if not hasattr(vcuda, "graphs"):
    pytest.skip("CUDA Graphs Python overlay not available for this build", allow_module_level=True)

from vibetensor.torch.cuda import graphs as vgraphs

from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _rng_test_utils import _require_cuda_or_skip, _canonical_vt_device_str


ERR_NON_CAPTURE_STREAM = (
    "rng: CUDA RNG operations on this generator are only allowed "
    "on the captured stream while CUDA Graph capture is active"
)


def test_rng_sampling_on_non_capture_stream_raises() -> None:
    """RNG ops on a non-capture stream fail with the pinned error string."""

    _require_cuda_or_skip()

    dev_str = "cuda:0"

    vt.manual_seed(2024)
    baseline = vt.get_rng_state(dev_str)

    g = vgraphs.CUDAGraph()
    s_capture = vcuda.Stream(device=0)

    with s_capture:
        with vgraphs.graph(g, stream=s_capture):
            assert _C._cuda_rng_is_capture_active_for_device(0)  # type: ignore[attr-defined]

            # Create a different stream on the same device and attempt RNG
            # sampling there while capture is active.
            s_other = vcuda.Stream(device=0)
            with pytest.raises(RuntimeError, match=re.escape(ERR_NON_CAPTURE_STREAM)):
                with s_other:
                    vt.rand((4,), device=dev_str)

    # RNG state for the capturing device must be unchanged; the failed op
    # should not have reserved any Philox blocks.
    assert vt.get_rng_state(dev_str) == baseline


def test_rng_sampling_on_non_default_stream_without_capture_is_ok() -> None:
    """Without capture, RNG ops on arbitrary streams are allowed."""

    _require_cuda_or_skip()

    dev_str = "cuda:0"

    vt.manual_seed(17)
    s = vcuda.Stream(device=0)

    with s:
        x = vt.rand((8,), device=dev_str)

    # Just a smoke check that execution succeeded and produced a tensor on
    # the right device.
    assert _canonical_vt_device_str(x) == dev_str
