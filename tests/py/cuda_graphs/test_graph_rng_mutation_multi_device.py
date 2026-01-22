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

from _rng_test_utils import _require_multi_gpu_or_skip


ERR = getattr(
    _C,
    "_ERR_CUDA_RNG_MUTATION_DURING_CAPTURE",
    "rng: generator state mutation is forbidden while CUDA Graph capture is active",
)


def test_manual_seed_all_with_one_capturing_device() -> None:
    """manual_seed_all must raise and leave the capturing device's RNG state unchanged."""

    _require_multi_gpu_or_skip()

    dev_cap = 0
    dev_other = 1

    vt.cuda.manual_seed_all(123)
    baseline_cap = vt.cuda.get_rng_state(dev_cap)
    baseline_other = vt.cuda.get_rng_state(dev_other)

    g = vgraphs.CUDAGraph()
    with vgraphs.graph(g):
        assert _C._cuda_rng_is_capture_active_for_device(dev_cap)  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match=re.escape(ERR)):
            vt.cuda.manual_seed_all(999)

    # Capturing device state must be exactly as before the failing call.
    assert vt.cuda.get_rng_state(dev_cap) == baseline_cap

    # Non-capturing device state should remain a valid 16-byte blob; current
    # semantics (whether device 1 was partially reseeded before the guard
    # error) are documented but not pinned here to keep the test robust.
    after_other = vt.cuda.get_rng_state(dev_other)
    assert isinstance(after_other, (bytes, bytearray)) and len(after_other) == 16
    assert isinstance(baseline_other, (bytes, bytearray)) and len(baseline_other) == 16
