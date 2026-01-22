# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def test_map_dataset_getitems_fastpath_is_used():
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __init__(self) -> None:
            self.getitem_calls = 0
            self.getitems_calls = 0

        def __len__(self):
            return 10

        def __getitem__(self, idx: int):
            self.getitem_calls += 1
            return int(idx)

        def __getitems__(self, indices: list[int]):
            self.getitems_calls += 1
            # Return items without calling __getitem__ so we can assert the fast-path.
            return [int(i) for i in indices]

    ds = _DS()

    out = []
    for b in vtd.DataLoader(ds, batch_size=4, drop_last=False):
        out.append(b.numpy())

    assert ds.getitems_calls == 3
    assert ds.getitem_calls == 0

    assert len(out) == 3
    np.testing.assert_array_equal(out[0], np.asarray([0, 1, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(out[1], np.asarray([4, 5, 6, 7], dtype=np.int64))
    np.testing.assert_array_equal(out[2], np.asarray([8, 9], dtype=np.int64))
