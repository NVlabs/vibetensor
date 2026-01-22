# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import vibetensor._C as C


def _make_cpu_tensor_1d(n: int, dtype: str = "float32"):
    # Use factory helpers to allocate a contiguous CPU tensor of length n
    t = C._cpu_zeros([n], dtype)
    return t


def test_unary_inplace_accepts_negative_stride_and_bumps_version():
    # Build a base and a reversed view with negative stride on CPU
    t = C._cpu_zeros([6], "float32")
    # as_strided can express reverse via stride=-1 and offset=n-1
    rev = t.as_strided([6], [-1], 5)
    v0 = rev.version()
    rev.fill_(0.0)
    assert rev.version() == v0 + 1
    v1 = rev.version()
    rev.relu_()
    assert rev.version() == v1 + 1


def test_unary_zero_size_noop():
    tz = C._cpu_zeros([0], "float32")
    v0 = tz.version()
    tz.fill_(0.0)
    tz.relu_()
    assert tz.version() == v0  # no elements -> no bump
