# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def test_default_convert_vs_default_collate_shapes():
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 4

        def __getitem__(self, idx: int):
            return np.asarray([idx, idx + 1], dtype=np.float32)

    ds = _DS()

    # Auto-batching enabled by default (batch_size=1) -> default_collate -> adds batch dim.
    b1 = next(iter(vtd.DataLoader(ds, batch_size=1)))
    assert b1.sizes == (1, 2)
    np.testing.assert_allclose(b1.numpy(), np.asarray([[0, 1]], dtype=np.float32))

    # Auto-batching disabled (batch_size=None) -> default_convert -> no batch dim.
    s0 = next(iter(vtd.DataLoader(ds, batch_size=None)))
    assert s0.sizes == (2,)
    np.testing.assert_allclose(s0.numpy(), np.asarray([0, 1], dtype=np.float32))


def test_default_convert_vs_default_collate_scalar():
    import vibetensor.torch.utils.data as vtd

    class _ScalarDS:
        def __len__(self):
            return 3

        def __getitem__(self, idx: int):
            return float(idx) + 0.5

    ds = _ScalarDS()

    b1 = next(iter(vtd.DataLoader(ds, batch_size=1)))
    assert b1.sizes == (1,)
    np.testing.assert_allclose(b1.numpy(), np.asarray([0.5], dtype=np.float32))

    s0 = next(iter(vtd.DataLoader(ds, batch_size=None)))
    assert s0.sizes == ()
    np.testing.assert_allclose(s0.numpy(), np.asarray(0.5, dtype=np.float32))
