# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import vibetensor.torch as vt


def test_from_numpy_zero_copy_sharing_basic():
    a = np.arange(4, dtype=np.int32)
    t = vt.from_numpy(a)
    # Mutate source; repr of tensor should reflect change if sharing
    a[0] = 42
    s = repr(t)
    assert "42" in s


def test_as_tensor_zero_copy_when_matching_dtype():
    a = np.arange(3, dtype=np.float32)
    t = vt.as_tensor(a, dtype=np.float32)
    a[1] = 7.0
    s = repr(t)
    assert "7." in s
