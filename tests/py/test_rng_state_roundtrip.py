# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import struct
import vibetensor.torch as vt
from vibetensor import _C


def _unpack_state(bts: bytes):
    return struct.unpack("<QQ", bts)


def test_state_roundtrip_cpu_randn():
    vt.manual_seed(123)
    st = vt.get_rng_state()
    a1 = vt.randn([16], dtype=np.float32, device="cpu")
    vt.set_rng_state(st)
    a2 = vt.randn([16], dtype=np.float32, device="cpu")
    np.testing.assert_array_equal(np.from_dlpack(a1), np.from_dlpack(a2))


def test_state_roundtrip_cuda_randn():
    if not getattr(_C, "_has_cuda", False) or int(_C._cuda_device_count()) == 0:  # type: ignore[attr-defined]
        return
    k = 0
    vt.cuda.manual_seed_all(2025)
    st = vt.get_rng_state("cuda:%d" % k)
    a1 = vt.randn([33], dtype=np.float32, device=f"cuda:{k}")
    vt.set_rng_state(st, device=f"cuda:{k}")
    a2 = vt.randn([33], dtype=np.float32, device=f"cuda:{k}")
    # Use host copy helper for CUDA tensors
    np.testing.assert_array_equal(vt.cuda.from_device(a1), vt.cuda.from_device(a2))
