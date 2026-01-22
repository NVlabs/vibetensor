# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import struct
import numpy as np
import pytest
import vibetensor.torch as vt
from vibetensor import _C


def _unpack_state(bts: bytes):
    return struct.unpack("<QQ", bts)


def test_manual_seed_sets_cpu_and_cuda():
    s = vt.manual_seed(314159)
    assert s == 314159
    # CPU seed and offset
    assert vt.initial_seed() == 314159
    seed_cpu, off_cpu = _unpack_state(vt.get_rng_state())
    assert seed_cpu == 314159
    assert off_cpu == 0
    # CUDA devices (if available)
    if getattr(_C, "_has_cuda", False) and int(_C._cuda_device_count()) > 0:  # type: ignore[attr-defined]
        n = int(_C._cuda_device_count())  # type: ignore[attr-defined]
        for k in range(n):
            assert vt.cuda.initial_seed(k) == 314159
            seed_k, off_k = _unpack_state(vt.cuda.get_rng_state(k))
            assert seed_k == 314159
            assert off_k == 0


def test_seed_cpu_only_does_not_affect_cuda():
    # Seed CUDA to a known value if available
    if getattr(_C, "_has_cuda", False) and int(_C._cuda_device_count()) > 0:  # type: ignore[attr-defined]
        vt.cuda.manual_seed_all(777)
        before = [vt.cuda.initial_seed(k) for k in range(int(_C._cuda_device_count()))]  # type: ignore[attr-defined]
    # CPU reseed
    ss = vt.seed()
    assert isinstance(ss, int)
    # CPU offset reset
    _, off = _unpack_state(vt.get_rng_state())
    assert off == 0
    if getattr(_C, "_has_cuda", False) and int(_C._cuda_device_count()) > 0:  # type: ignore[attr-defined]
        after = [vt.cuda.initial_seed(k) for k in range(int(_C._cuda_device_count()))]  # type: ignore[attr-defined]
        assert before == after
