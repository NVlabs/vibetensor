# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import struct
import itertools as it

import numpy as np
import pytest
import vibetensor.torch as vt
from vibetensor import _C

# ---- Minimal Philox4x32-10 in Python for KATs ----
K_A = 0x9E3779B9
K_B = 0xBB67AE85
S_A = 0xD2511F53
S_B = 0xCD9E8D57


def _mulhilo32(a, b):
    prod = (a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)
    lo = prod & 0xFFFFFFFF
    hi = (prod >> 32) & 0xFFFFFFFF
    return lo, hi


def _single_round(ctr, key):
    lo0, hi0 = _mulhilo32(S_A, ctr[0])
    lo1, hi1 = _mulhilo32(S_B, ctr[2])
    return [
        (hi1 ^ ctr[1] ^ key[0]) & 0xFFFFFFFF,
        lo1 & 0xFFFFFFFF,
        (hi0 ^ ctr[3] ^ key[1]) & 0xFFFFFFFF,
        lo0 & 0xFFFFFFFF,
    ]


def _philox10(ctr, key):
    c = list(ctr)
    k = list(key)
    for _ in range(9):
        c = _single_round(c, k)
        k[0] = (k[0] + K_A) & 0xFFFFFFFF
        k[1] = (k[1] + K_B) & 0xFFFFFFFF
    return _single_round(c, k)


def _seed_to_key(seed):
    return [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF]


def _block_to_ctr(b):
    return [b & 0xFFFFFFFF, (b >> 32) & 0xFFFFFFFF, 0, 0]


def _u32_to_u01(u):
    # mask 31 bits then scale by 2^-31
    return float(u & 0x7FFFFFFF) * (2.0 ** -31)


def _kat_uniform(seed, base_offset, N, low, high):
    if N <= 0:
        return []
    out = [0.0] * N
    key = _seed_to_key(seed)
    total_blocks = (N + 3) // 4
    for br in range(total_blocks):
        ctr = _block_to_ctr(base_offset + br)
        lanes = _philox10(ctr, key)
        for ln in range(4):
            e = br * 4 + ln
            if e >= N:
                break
            # Match kernel's float32 arithmetic
            U = np.float32(np.float32(lanes[ln] & 0x7FFFFFFF) * np.float32(2.0 ** -31))
            val = np.float32(np.float32(low) + np.float32(high - low) * U)
            out[e] = float(val)
    return out


def _pack_state(seed, off):
    return struct.pack("<QQ", seed & 0xFFFFFFFFFFFFFFFF, off & 0xFFFFFFFFFFFFFFFF)


def _unpack_state(bts):
    return struct.unpack("<QQ", bts)


def test_rng_state_round_trip():
    cases = [
        (0, 0), (1, 1), (123, 2), (9999, 3), (0, 31), (1, 32), (123, 33)
    ]
    for seed, off in cases:
        _C._rng_set_state(_pack_state(seed, off))
        st = _C._rng_get_state()
        s2, o2 = _unpack_state(st)
        assert s2 == seed and o2 == off


def test_uniform_float32_kat_contiguous():
    seeds = [0, 1, 123, 9999]
    sizes = [0, 1, 2, 3, 4, 31, 32, 33]
    for seed in seeds:
        for N in sizes:
            _C._rng_set_state(_pack_state(seed, 0))
            t = vt.zeros([N], dtype=np.float32)
            _C._uniform_(t, 0.0, 1.0)
            got = np.from_dlpack(t).astype(np.float32)
            exp = np.array(_kat_uniform(seed, 0, N, 0.0, 1.0), dtype=np.float32)
            assert np.array_equal(got, exp)
            # offset advanced by ceil_div(N,4)
            s2, o2 = _unpack_state(_C._rng_get_state())
            assert s2 == seed
            assert o2 == ((N + 3) // 4)


def test_uniform_float32_kat_strided():
    # Use a 2D tensor and its transpose to create nontrivial strides
    seed = 123
    _C._rng_set_state(_pack_state(seed, 0))
    base = vt.zeros([3, 5], dtype=np.float32)
    t = base.transpose(0, 1)  # shape [5,3]
    N = 15
    _C._uniform_(t, -1.0, 2.0)
    got = np.from_dlpack(t).reshape(-1).astype(np.float32)  # row-major logical order of t's shape [5,3]
    exp = np.array(_kat_uniform(seed, 0, N, -1.0, 2.0), dtype=np.float32)
    assert np.array_equal(got, exp)
    # offset advanced by ceil_div(N,4)
    s2, o2 = _unpack_state(_C._rng_get_state())
    assert s2 == seed
    assert o2 == ((N + 3) // 4)


def test_rng_seed_and_initial_seed():
    s = vt.rng.manual_seed(42)
    assert s == 42
    assert vt.rng.initial_seed() == 42
    # reseed nondeterministically (value varies) but type/round-trip should work
    s2 = vt.rng.seed()
    assert isinstance(s2, int)
    vt.rng.set_rng_state(vt.rng.get_rng_state())


def test_uniform_dtype_guard():
    t = vt.zeros([2], dtype=np.int32)
    with pytest.raises(TypeError, match="uniform_: expected dtype=float32"):
        _C._uniform_(t, 0.0, 1.0)


def test_rng_set_state_type_and_length_errors():
    with pytest.raises(TypeError, match="state must be a bytes object"):
        vt.rng.set_rng_state(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="state must be 16 bytes: {seed:u64, offset:u64}"):
        vt.rng.set_rng_state(b"short")


def test_rng_concurrency_reservations_smoke():
    import threading
    Ns = [0, 1, 2, 3, 4]
    _C._rng_set_state(_pack_state(777, 0))
    def worker(n):
        t = vt.zeros([n], dtype=np.float32)
        _C._uniform_(t, 0.0, 1.0)
    threads = [threading.Thread(target=worker, args=(n,)) for n in Ns]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    total_blocks = sum((n + 3) // 4 for n in Ns)
    s2, o2 = _unpack_state(_C._rng_get_state())
    assert s2 == 777
    assert o2 == total_blocks

