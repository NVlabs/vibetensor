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


def _reset_state(seed=1234, off=0):
    _C._rng_set_state(_pack_state(seed, off))


def test_rand_factory_matches_uniform_inplace():
    for N in [0, 1, 2, 3, 4, 31, 32, 33, 1001]:
        _reset_state(999, 0)
        a = vt.rand([N], dtype=np.float32)
        sa = np.from_dlpack(a).astype(np.float32)
        _reset_state(999, 0)
        b = vt.zeros([N], dtype=np.float32)
        _C._uniform_(b, 0.0, 1.0)
        sb = np.from_dlpack(b).astype(np.float32)
        assert np.array_equal(sa, sb)


def test_randn_factory_matches_normal_inplace():
    for N in [0, 1, 2, 3, 4, 31, 32, 33, 513]:
        _reset_state(2024, 0)
        a = vt.randn([N], dtype=np.float32)
        sa = np.from_dlpack(a).astype(np.float32)
        _reset_state(2024, 0)
        b = vt.zeros([N], dtype=np.float32)
        _C._normal_(b, 0.0, 1.0)
        sb = np.from_dlpack(b).astype(np.float32)
        assert np.array_equal(sa, sb)


def test_randint_factory_matches_inplace():
    for N in [0, 1, 2, 3, 4, 31, 32, 33, 1001]:
        for low, high in [(0, 1), (0, 3), (10, 100000)]:
            _reset_state(42, 0)
            a = vt.randint(low, high, [N], dtype=np.int64)
            sa = np.from_dlpack(a).astype(np.int64)
            _reset_state(42, 0)
            b = vt.zeros([N], dtype=np.int64)
            _C._randint_(b, low, high)
            sb = np.from_dlpack(b).astype(np.int64)
            assert np.array_equal(sa, sb)
