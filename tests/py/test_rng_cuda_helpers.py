# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from vibetensor import _C
import vibetensor.torch as vt


def test_cuda_rng_helpers_availability_and_errors():
    if not getattr(_C, "_has_cuda", False) or int(_C._cuda_device_count()) == 0:  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match=r"CUDA is not available"):
            vt.cuda.manual_seed(1)
        with pytest.raises(ValueError, match=r"CUDA is not available"):
            vt.cuda.manual_seed_all(1)
        with pytest.raises(ValueError, match=r"CUDA is not available"):
            vt.cuda.initial_seed()
        with pytest.raises(ValueError, match=r"CUDA is not available"):
            vt.cuda.get_rng_state()
        with pytest.raises(ValueError, match=r"CUDA is not available"):
            vt.cuda.set_rng_state(b"\x00" * 16)
        return
    # When available, basic calls succeed
    vt.cuda.manual_seed(9)
    assert isinstance(vt.cuda.initial_seed(), int)
    st = vt.cuda.get_rng_state()
    assert isinstance(st, (bytes, bytearray)) and len(st) == 16
    # Wrong length state error
    with pytest.raises(ValueError, match=r"state must be 16 bytes: \{seed:u64, offset:u64\}"):
        vt.set_rng_state(b"\x00" * 8)
    if getattr(_C, "_has_cuda", False) and int(_C._cuda_device_count()) > 0:  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match=r"state must be 16 bytes: \{seed:u64, offset:u64\}"):
            vt.cuda.set_rng_state(b"\x00" * 8)
