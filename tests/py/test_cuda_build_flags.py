# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

def test_cuda_build_flag_and_count():
    from vibetensor import _C as C
    assert isinstance(C._has_cuda, bool)
    n = C._cuda_device_count()
    assert isinstance(n, int)
    if not C._has_cuda:
        assert n == 0
