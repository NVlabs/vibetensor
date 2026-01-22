# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import struct
import numpy as np
import vibetensor.torch as vt
from vibetensor import _C


def _pack_state(seed, off):
    return struct.pack("<QQ", seed & 0xFFFFFFFFFFFFFFFF, off & 0xFFFFFFFFFFFFFFFF)


def _unpack_state(bts):
    return struct.unpack("<QQ", bts)


def _reset_state(seed=2025, off=0):
    _C._rng_set_state(_pack_state(seed, off))


def test_offset_accounting_normal_and_bernoulli():
    for N in [0, 1, 2, 3, 4, 31, 32, 33, 513]:
        for op in ("_normal_", "_bernoulli_"):
            _reset_state(7, 0)
            t = vt.zeros([N], dtype=np.float32)
            getattr(_C, op)(t, 0.0, 1.0) if op == "_normal_" else getattr(_C, op)(t, 0.5)
            seed, off = _unpack_state(_C._rng_get_state())
            assert off == ((N + 3) // 4)


def test_offset_accounting_randint():
    for N in [0, 1, 2, 3, 4, 31, 32, 33, 513]:
        _reset_state(11, 0)
        t = vt.zeros([N], dtype=np.int64)
        _C._randint_(t, 0, 10000)
        seed, off = _unpack_state(_C._rng_get_state())
        assert off == ((N + 1) // 2)
