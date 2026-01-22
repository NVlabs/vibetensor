# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import vibetensor.torch as vt


def test_generator_kwarg_acceptance_cpu():
    g = vt.rng.Generator()
    a = vt.rand([8], generator=g)
    b = vt.randn([8], generator=g)
    c = vt.randint(0, 10, [8], generator=g)
    assert np.from_dlpack(a).shape[0] == 8
    assert np.from_dlpack(b).shape[0] == 8
    assert np.from_dlpack(c).shape[0] == 8


def test_generator_kwarg_device_mismatch():
    class FakeGen:
        def __init__(self):
            self.device = "cuda:0"
    g = FakeGen()
    with pytest.raises(ValueError, match=r"generator device mismatch: expected cpu, got cuda:0"):
        vt.rand([2], generator=g)
    with pytest.raises(ValueError, match=r"generator device mismatch: expected cpu, got cuda:0"):
        vt.randn([2], generator=g)
    with pytest.raises(ValueError, match=r"generator device mismatch: expected cpu, got cuda:0"):
        vt.randint(0, 3, [2], generator=g)
