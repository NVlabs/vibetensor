# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
import vibetensor.torch as vt
from vibetensor import _C
import struct


def _pack_state(seed, off):
    return struct.pack("<QQ", seed & 0xFFFFFFFFFFFFFFFF, off & 0xFFFFFFFFFFFFFFFF)


def _reset_state(seed=777, off=0):
    _C._rng_set_state(_pack_state(seed, off))


def _freqs(samples, low, high):
    n = high - low
    cnt = np.zeros(n, dtype=np.int64)
    for v in samples:
        cnt[int(v - low)] += 1
    return cnt


def test_randint_uniformity_sanity_small_ranges():
    # Basic sanity: empirical frequencies close to uniform within a tolerance.
    # For small ranges we bound the maximum per-bin deviation; for large ranges
    # we use an aggregate chi-square-style check to account for Poisson noise.
    N = 200_000
    for low, high in [(0, 2), (0, 3), (5, 9), (0, 65536)]:
        _reset_state(13579, 0)
        t = vt.randint(low, high, [N], dtype=np.int64)
        arr = np.from_dlpack(t).astype(np.int64)
        assert arr.min() >= low and arr.max() < high
        cnt = _freqs(arr, low, high)
        n = high - low
        exp = N / n
        # Chi-square-like normalized error on individual bins
        err = float(np.max(np.abs(cnt - exp)) / exp)
        if n <= 256:
            # Tight bound for small ranges where per-bin counts are large.
            assert err < 0.02
        else:
            # For very large ranges (n >> N), per-bin counts are small and the
            # max deviation is noisy. Use an aggregate chi-square statistic:
            #   chi2 = sum((cnt - exp)^2 / exp) ~ chi2_{df=n-1}
            chi2 = float(np.sum((cnt - exp) ** 2 / exp))
            df = n - 1
            z = (chi2 - df) / math.sqrt(2.0 * df)
            # Very loose one-sided bound; corresponds roughly to p ~ 3e-7.
            assert z < 5.0
